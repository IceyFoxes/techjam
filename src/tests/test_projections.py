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


if __name__ == "__main__":
    unittest.main()
