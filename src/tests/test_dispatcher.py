from __future__ import annotations

import unittest
from unittest import mock


try:
    import torch
except ImportError:  # pragma: no cover - dependency-free environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DispatcherPolicyTests(unittest.TestCase):
    VALIDATED_CONTRACT = {
        "device_type": "cuda",
        "dtype": None,
        "device_name": "NVIDIA GeForce RTX 5080",
        "device_capability": (12, 0),
        "torch_version": "2.13.0+cu130",
        "matmul_precision": "high",
        "allow_tf32": True,
    }

    @staticmethod
    def _config(case_id: int):
        from src.infra import load_official_cases
        from torch_transformer_benchmark import TransformerConfig

        case = load_official_cases()[case_id]
        return TransformerConfig(
            batch_size=case.batch_size,
            seq_len=case.seq_len,
            d_model=case.qkv_dim,
            num_heads=case.heads,
            ffn_dim=case.ffn_dim,
            num_layers=case.layers,
            causal=case.causal,
        )

    def test_routes_all_twelve_feasible_cases(self):
        from src.dispatcher import COMPILED_SDPA_BACKEND, select_route

        expected_modes = {
            1: "reduce-overhead",
            2: "reduce-overhead",
            3: "reduce-overhead",
            4: "reduce-overhead",
            5: "reduce-overhead",
            7: "reduce-overhead",
            8: "reduce-overhead",
            9: "reduce-overhead",
            10: "reduce-overhead",
            11: "reduce-overhead",
            12: "reduce-overhead",
            13: "default",
        }
        for case_id, mode in expected_modes.items():
            with self.subTest(case=case_id):
                route = select_route(
                    self._config(case_id),
                    **{**self.VALIDATED_CONTRACT, "dtype": torch.float32},
                )
                self.assertEqual(route.case_id, case_id)
                self.assertEqual(route.backend, COMPILED_SDPA_BACKEND)
                self.assertEqual(route.compile_mode, mode)

    def test_routes_extreme_cases_to_memory_safe_backend(self):
        from src.dispatcher import EXTREME_MEMORY_BACKEND, select_route

        contracts = {
            6: {**self.VALIDATED_CONTRACT, "dtype": torch.float32},
            14: {**self.VALIDATED_CONTRACT, "dtype": torch.float16},
        }
        for case_id, contract in contracts.items():
            with self.subTest(case=case_id):
                route = select_route(self._config(case_id), **contract)
                self.assertEqual(route.backend, EXTREME_MEMORY_BACKEND)
                self.assertIsNone(route.compile_mode)

    def test_rejects_unsafe_extreme_contracts(self):
        from src.dispatcher import UNSUPPORTED_BACKEND, select_route

        contracts = (
            (6, {**self.VALIDATED_CONTRACT, "dtype": torch.float16}),
            (14, {**self.VALIDATED_CONTRACT, "dtype": torch.float32}),
            (14, {**self.VALIDATED_CONTRACT, "dtype": torch.bfloat16}),
            (
                14,
                {
                    **self.VALIDATED_CONTRACT,
                    "dtype": torch.float16,
                    "device_type": "cpu",
                    "device_capability": None,
                },
            ),
        )
        for case_id, contract in contracts:
            with self.subTest(case=case_id, contract=contract):
                route = select_route(self._config(case_id), **contract)
                self.assertEqual(route.backend, UNSUPPORTED_BACKEND)

    def test_accepts_ampere_and_newer_gpu_models(self):
        from src.dispatcher import COMPILED_SDPA_BACKEND, select_route

        config = self._config(1)
        validated = {**self.VALIDATED_CONTRACT, "dtype": torch.float32}
        contracts = (
            {
                **validated,
                "device_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                "device_capability": (8, 9),
            },
            {
                **validated,
                "device_name": "NVIDIA L4",
                "device_capability": (8, 9),
            },
            {
                **validated,
                "device_name": "unrecognized future CUDA GPU",
                "device_capability": (13, 0),
            },
        )
        for contract in contracts:
            with self.subTest(contract=contract):
                route = select_route(config, **contract)
                self.assertEqual(route.backend, COMPILED_SDPA_BACKEND)

    def test_rejects_unsupported_runtime_contracts(self):
        from src.dispatcher import REFERENCE_BACKEND, select_route

        config = self._config(1)
        validated = {**self.VALIDATED_CONTRACT, "dtype": torch.float32}
        contracts = (
            {**validated, "device_type": "cpu"},
            {**validated, "dtype": torch.float16},
            {**validated, "dtype": torch.bfloat16},
            {**validated, "device_capability": (7, 5)},
            {**validated, "device_capability": None},
            {**validated, "torch_version": "2.12.0+cu128"},
            {**validated, "matmul_precision": "highest"},
            {**validated, "allow_tf32": False},
        )
        for contract in contracts:
            with self.subTest(contract=contract):
                route = select_route(config, **contract)
                self.assertEqual(route.backend, REFERENCE_BACKEND)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DispatcherExecutionTests(unittest.TestCase):
    def setUp(self):
        from torch_transformer_benchmark import TransformerConfig

        self.small_config = TransformerConfig(
            batch_size=2,
            seq_len=8,
            d_model=16,
            num_heads=4,
            ffn_dim=16,
            num_layers=2,
            causal=True,
        )

    @staticmethod
    def _official_config(case_id: int):
        return DispatcherPolicyTests._config(case_id)

    @staticmethod
    def _models(config):
        from src.dispatcher import DispatchingTransformer
        from torch_transformer_benchmark import (
            BaselineTransformer,
            copy_model_weights,
        )

        baseline = BaselineTransformer(config).eval()
        candidate = DispatchingTransformer(config).eval()
        copy_model_weights(baseline, candidate)
        return baseline, candidate

    @staticmethod
    def _cuda_runtime_key(input_shape, *, mask_present=True):
        from src.dispatcher import RuntimeKey

        return RuntimeKey(
            input_shape=tuple(input_shape),
            device_type="cuda",
            device_index=0,
            dtype=torch.float32,
            device_name="NVIDIA GeForce RTX 5080",
            device_capability=(12, 0),
            mask_present=mask_present,
            inference_mode=True,
            grad_enabled=False,
            matmul_precision="high",
            allow_tf32=True,
        )

    def test_preserves_reference_state_dict_contract(self):
        baseline, candidate = self._models(self.small_config)
        self.assertEqual(list(baseline.state_dict()), list(candidate.state_dict()))

    def test_only_case2_uses_packed_qkv_attention(self):
        from src.implementations.sdpa import (
            PackedQKVSDPASelfAttention,
            StridedSDPASelfAttention,
        )

        _, case2 = self._models(self._official_config(2))
        _, case3 = self._models(self._official_config(3))

        self.assertTrue(
            all(
                isinstance(layer.attention, PackedQKVSDPASelfAttention)
                for layer in case2.layers
            )
        )
        self.assertTrue(
            all(
                type(layer.attention) is StridedSDPASelfAttention
                for layer in case3.layers
            )
        )

    def test_case14_uses_flash_only_attention(self):
        from src.implementations.extreme import FlashOnlySDPASelfAttention

        _, candidate = self._models(self._official_config(14))

        self.assertTrue(
            all(
                type(layer.attention) is FlashOnlySDPASelfAttention
                for layer in candidate.layers
            )
        )

    def test_extreme_shape_mismatch_never_uses_reference_fallback(self):
        from src.dispatcher import UNSUPPORTED_BACKEND

        _, candidate = self._models(self._official_config(14))
        key = self._cuda_runtime_key((1, 1024, 1024))
        route = candidate._resolve_route(key)

        self.assertEqual(route.backend, UNSUPPORTED_BACKEND)

    def test_case14_allows_staged_device_and_dtype_conversions(self):
        _, candidate = self._models(self._official_config(14))

        candidate.half()

        self.assertEqual(next(candidate.parameters()).dtype, torch.float16)

    def test_cpu_fallback_is_bitwise_reference_with_padding(self):
        baseline, candidate = self._models(self.small_config)
        torch.manual_seed(5678)
        x = torch.randn(2, 8, 16)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, False, False, False, False, False],
            ]
        )
        x = x.masked_fill(~mask[..., None], 0)
        with torch.inference_mode():
            expected = baseline(x, mask)
            actual = candidate(x, mask)
        self.assertTrue(torch.equal(expected, actual))

    def test_sdpa_path_matches_reference_with_and_without_padding(self):
        from torch_transformer_benchmark import compare_outputs

        baseline, candidate = self._models(self.small_config)
        torch.manual_seed(1234)
        x = torch.randn(2, 8, 16)
        masks = (
            torch.ones(2, 8, dtype=torch.bool),
            torch.tensor(
                [
                    [True, True, True, True, True, True, True, True],
                    [True, True, True, False, False, False, False, False],
                ]
            ),
        )
        for mask in masks:
            with self.subTest(padded=not bool(mask.all())):
                masked_x = x.masked_fill(~mask[..., None], 0)
                with torch.inference_mode():
                    expected = baseline(masked_x, mask)
                    actual = candidate._forward_sdpa(masked_x, mask)
                result = compare_outputs(expected, actual, rtol=0.02, atol=0.002)
                self.assertTrue(result.passed, result)

    def test_compiles_lazily_once_per_runtime_key(self):
        config = self._official_config(2)
        _, candidate = self._models(config)
        x = torch.randn(config.batch_size, config.seq_len, config.d_model)
        mask = torch.ones(config.batch_size, config.seq_len, dtype=torch.bool)
        key = self._cuda_runtime_key(x.shape)
        compile_modes = []
        compiled_calls = []

        def fake_compile(function, *, mode):
            compile_modes.append(mode)

            def wrapped(*args):
                compiled_calls.append(True)
                return function(*args)

            return wrapped

        with (
            mock.patch.object(candidate, "_runtime_key", return_value=key),
            mock.patch("src.dispatcher.torch.compile", side_effect=fake_compile),
            torch.inference_mode(),
        ):
            first = candidate(x, mask)
            second = candidate(x, mask)

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(compile_modes, ["reduce-overhead"])
        self.assertEqual(len(compiled_calls), 3)
        self.assertEqual(candidate.last_route.case_id, 2)
        self.assertEqual(len(candidate._compiled_forwards), 1)

    def test_each_case_has_a_distinct_compiler_entrypoint(self):
        code_objects = set()
        for case_id in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13):
            _, candidate = self._models(self._official_config(case_id))
            entrypoint = candidate._compile_entrypoint(case_id)
            code_objects.add(entrypoint.__func__.__code__)
        self.assertEqual(len(code_objects), 12)

    def test_compile_failure_demotes_key_to_exact_reference(self):
        config = self._official_config(2)
        baseline, candidate = self._models(config)
        x = torch.randn(config.batch_size, config.seq_len, config.d_model)
        mask = torch.ones(config.batch_size, config.seq_len, dtype=torch.bool)
        key = self._cuda_runtime_key(x.shape)

        with torch.inference_mode():
            expected = baseline(x, mask)
        with (
            mock.patch.object(candidate, "_runtime_key", return_value=key),
            mock.patch(
                "src.dispatcher.torch.compile",
                side_effect=RuntimeError("synthetic compiler failure"),
            ),
            torch.inference_mode(),
        ):
            actual = candidate(x, mask)

        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(candidate.last_route.backend, "reference")
        self.assertEqual(len(candidate.compile_failures), 1)

    def test_deferred_compiled_failure_demotes_cached_call(self):
        config = self._official_config(2)
        baseline, candidate = self._models(config)
        x = torch.randn(config.batch_size, config.seq_len, config.d_model)
        mask = torch.ones(config.batch_size, config.seq_len, dtype=torch.bool)
        key = self._cuda_runtime_key(x.shape)
        calls = 0

        def fake_compile(function, *, mode):
            self.assertEqual(mode, "reduce-overhead")

            def fails_after_warmup(*args):
                nonlocal calls
                calls += 1
                if calls > 2:
                    raise RuntimeError("synthetic deferred CUDA failure")
                return function(*args)

            return fails_after_warmup

        with torch.inference_mode():
            expected = baseline(x, mask)
        with (
            mock.patch.object(candidate, "_runtime_key", return_value=key),
            mock.patch("src.dispatcher.torch.compile", side_effect=fake_compile),
            torch.inference_mode(),
        ):
            candidate(x, mask)
            actual = candidate(x, mask)

        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(candidate.last_route.backend, "reference")
        self.assertIn("deferred CUDA failure", next(iter(candidate.compile_failures.values())))

    def test_parameter_conversion_clears_runtime_cache(self):
        from src.dispatcher import CachedForward

        _, candidate = self._models(self.small_config)
        key = self._cuda_runtime_key((2, 8, 16))
        candidate._compiled_forwards[key] = CachedForward(
            candidate._forward_reference,
            compiled=False,
            case_id=None,
        )
        candidate._compile_failures[key] = "synthetic"
        candidate.to(dtype=torch.float64)
        self.assertFalse(candidate._compiled_forwards)
        self.assertFalse(candidate.compile_failures)


if __name__ == "__main__":
    unittest.main()
