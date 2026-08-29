from __future__ import annotations

import unittest
from unittest import mock


try:
    import torch
except ImportError:  # pragma: no cover - dependency-free environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ProjectionsControlTests(unittest.TestCase):
    @staticmethod
    def _config():
        from torch_transformer_benchmark import TransformerConfig

        return TransformerConfig(
            batch_size=2,
            seq_len=8,
            d_model=16,
            num_heads=4,
            ffn_dim=32,
            num_layers=2,
            causal=True,
        )

    @classmethod
    def _models(cls):
        from src.implementations.projections import ProjectionsControl
        from torch_transformer_benchmark import (
            BaselineTransformer,
            copy_model_weights,
        )

        config = cls._config()
        baseline = BaselineTransformer(config).eval()
        candidate = ProjectionsControl(config).eval()
        copy_model_weights(baseline, candidate, strict=True)
        return baseline, candidate

    def test_state_dict_contract_matches_reference(self):
        baseline, candidate = self._models()

        self.assertEqual(list(baseline.state_dict()), list(candidate.state_dict()))
        for name, expected in baseline.state_dict().items():
            actual = candidate.state_dict()[name]
            self.assertEqual(expected.shape, actual.shape)
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertTrue(torch.equal(expected, actual))

    def test_authoritative_weight_copy_is_strict(self):
        from src.implementations.projections import ProjectionsControl
        from torch_transformer_benchmark import (
            BaselineTransformer,
            copy_model_weights,
        )

        config = self._config()
        baseline = BaselineTransformer(config).eval()
        candidate = ProjectionsControl(config).eval()
        with mock.patch.object(
            candidate,
            "load_state_dict",
            wraps=candidate.load_state_dict,
        ) as load_spy:
            copy_model_weights(baseline, candidate, strict=True)

        self.assertTrue(load_spy.call_args.kwargs["strict"])
        for name, expected in baseline.state_dict().items():
            self.assertTrue(torch.equal(expected, candidate.state_dict()[name]))

    def test_functional_ffn_requests_exact_gelu(self):
        from src.implementations import projections

        block = projections.FunctionalFFNControlBlock(16, 4, 32).eval()
        x = torch.randn(2, 8, 16)
        original_gelu = projections.F.gelu
        with (
            mock.patch.object(
                block.attention,
                "forward",
                side_effect=lambda value, *_: torch.zeros_like(value),
            ),
            mock.patch(
                "src.implementations.projections.F.gelu",
                wraps=original_gelu,
            ) as gelu_spy,
            torch.inference_mode(),
        ):
            block(x, None, True)

        self.assertEqual(gelu_spy.call_count, 1)
        self.assertEqual(gelu_spy.call_args.kwargs["approximate"], "none")

    def test_matches_reference_with_and_without_padding(self):
        baseline, candidate = self._models()
        torch.manual_seed(1234)
        x = torch.randn(2, 8, 16)
        masks = (
            None,
            torch.tensor(
                [
                    [True, True, True, True, True, True, True, True],
                    [True, True, True, False, False, False, False, False],
                ]
            ),
        )

        for mask in masks:
            with self.subTest(mask_present=mask is not None), torch.inference_mode():
                expected = baseline(x, mask)
                actual = candidate(x, mask)

            self.assertTrue(torch.equal(expected, actual))
            if mask is not None:
                self.assertTrue(torch.equal(actual[~mask], torch.zeros_like(actual[~mask])))

    def test_preserves_supported_cpu_dtypes(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                baseline, candidate = self._models()
                baseline = baseline.to(dtype=dtype)
                candidate = candidate.to(dtype=dtype)
                x = torch.randn(2, 8, 16, dtype=dtype)
                with torch.inference_mode():
                    expected = baseline(x, None)
                    actual = candidate(x, None)

                self.assertEqual(actual.dtype, dtype)
                self.assertTrue(torch.equal(expected, actual))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_preserves_supported_cuda_low_precision_dtypes(self):
        dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                baseline, candidate = self._models()
                baseline = baseline.to(device="cuda", dtype=dtype)
                candidate = candidate.to(device="cuda", dtype=dtype)
                x = torch.randn(2, 8, 16, device="cuda", dtype=dtype)
                with torch.inference_mode():
                    expected = baseline(x, None)
                    actual = candidate(x, None)

                self.assertEqual(actual.dtype, dtype)
                self.assertTrue(torch.equal(expected, actual))

    def test_candidate_metadata_identifies_a_control(self):
        from src.infra import load_candidate

        candidate = load_candidate("projections")
        description = candidate.description.lower()
        self.assertEqual(candidate.name, "projections-control")
        self.assertTrue(candidate.strict_weight_copy)
        self.assertIsNone(candidate.weight_loader)
        self.assertIn("control", description)
        self.assertNotIn("fused", description)
        self.assertNotIn("fusion", description)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PackedQKVCandidateTests(unittest.TestCase):
    @staticmethod
    def _config():
        return ProjectionsControlTests._config()

    @classmethod
    def _models(cls):
        from src.dispatcher import DispatchingTransformer
        from src.implementations.projections import Case2PackedQKVTransformer
        from torch_transformer_benchmark import (
            BaselineTransformer,
            copy_model_weights,
        )

        config = cls._config()
        baseline = BaselineTransformer(config).eval()
        control = DispatchingTransformer(config).eval()
        candidate = Case2PackedQKVTransformer(config).eval()
        copy_model_weights(baseline, control, strict=True)
        copy_model_weights(baseline, candidate, strict=True)
        return baseline, control, candidate

    def test_preserves_state_dict_and_packs_exact_parameter_rows(self):
        baseline, _, candidate = self._models()

        self.assertEqual(list(baseline.state_dict()), list(candidate.state_dict()))
        self.assertFalse(
            any("_packed_qkv" in name for name in candidate.state_dict())
        )
        for layer in candidate.layers:
            attention = layer.attention
            expected_weight = torch.cat(
                (
                    attention.q_proj.weight,
                    attention.k_proj.weight,
                    attention.v_proj.weight,
                ),
                dim=0,
            )
            expected_bias = torch.cat(
                (
                    attention.q_proj.bias,
                    attention.k_proj.bias,
                    attention.v_proj.bias,
                ),
                dim=0,
            )
            self.assertTrue(
                torch.equal(attention._packed_qkv_weight, expected_weight)
            )
            self.assertTrue(torch.equal(attention._packed_qkv_bias, expected_bias))

    def test_rebuilds_packed_buffers_after_dtype_conversion(self):
        _, _, candidate = self._models()
        candidate = candidate.to(dtype=torch.float64)

        for layer in candidate.layers:
            attention = layer.attention
            self.assertEqual(attention._packed_qkv_weight.dtype, torch.float64)
            self.assertTrue(
                torch.equal(
                    attention._packed_qkv_weight[: attention.d_model],
                    attention.q_proj.weight,
                )
            )

    def test_refresh_updates_existing_buffer_storage_after_mutation(self):
        _, _, candidate = self._models()
        attention = candidate.layers[0].attention
        original_pointer = attention._packed_qkv_weight.data_ptr()
        with torch.no_grad():
            attention.q_proj.weight.add_(1)

        attention.refresh_packed_qkv()

        self.assertEqual(attention._packed_qkv_weight.data_ptr(), original_pointer)
        self.assertTrue(
            torch.equal(
                attention._packed_qkv_weight[: attention.d_model],
                attention.q_proj.weight,
            )
        )

    def test_strict_reload_preserves_packed_buffer_storage(self):
        baseline, _, candidate = self._models()
        attention = candidate.layers[0].attention
        original_pointer = attention._packed_qkv_weight.data_ptr()

        candidate.load_state_dict(baseline.state_dict(), strict=True)

        self.assertEqual(attention._packed_qkv_weight.data_ptr(), original_pointer)

    def test_packed_views_alias_storage_with_expected_strides(self):
        _, _, candidate = self._models()
        attention = candidate.layers[0].attention
        x = torch.randn(2, 8, 16)
        with torch.inference_mode():
            q, k, v = attention.project_qkv(x)

        self.assertEqual(q.shape, (2, 4, 8, 4))
        self.assertEqual(q.stride(), (384, 4, 48, 1))
        self.assertEqual(k.stride(), q.stride())
        self.assertEqual(v.stride(), q.stride())
        self.assertEqual(
            (q.storage_offset(), k.storage_offset(), v.storage_offset()),
            (0, 16, 32),
        )
        self.assertEqual(
            q.untyped_storage().data_ptr(),
            k.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            q.untyped_storage().data_ptr(),
            v.untyped_storage().data_ptr(),
        )

    def test_matches_current_sdpa_control_with_padding(self):
        _, control, candidate = self._models()
        torch.manual_seed(1234)
        x = torch.randn(2, 8, 16)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, False, False, False, False, False],
            ]
        )
        with torch.inference_mode():
            expected = control._forward_sdpa(x, mask)
            actual = candidate(x, mask)

        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(actual[~mask], torch.zeros_like(actual[~mask])))

    def test_gradient_enabled_path_ignores_inference_cache(self):
        _, _, candidate = self._models()
        for layer in candidate.layers:
            layer.attention._packed_qkv_weight.fill_(float("nan"))
        x = torch.randn(2, 8, 16, requires_grad=True)

        actual = candidate(x, None)
        actual.sum().backward()

        self.assertTrue(torch.isfinite(actual).all())
        self.assertIsNotNone(x.grad)

    def test_candidate_contract_is_case2_only(self):
        from src.implementations.projections import PACKED_CASE2

        PACKED_CASE2.validate()
        self.assertEqual(PACKED_CASE2.name, "case2-packed-qkv")
        self.assertFalse(PACKED_CASE2.self_compiling)
        self.assertNotIn(2, PACKED_CASE2.unsupported_official_cases)
        self.assertEqual(
            set(PACKED_CASE2.unsupported_official_cases),
            set(range(1, 15)) - {2},
        )

    def test_cross_case_validation_candidate_excludes_only_extremes(self):
        from src.implementations.projections import PACKED_ALL

        PACKED_ALL.validate()
        self.assertEqual(PACKED_ALL.name, "packed-qkv-cross-case-validation")
        self.assertFalse(PACKED_ALL.self_compiling)
        self.assertEqual(PACKED_ALL.unsupported_official_cases, (6, 14))


if __name__ == "__main__":
    unittest.main()
