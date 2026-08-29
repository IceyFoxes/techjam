import unittest


try:
    import torch
except ImportError:  # pragma: no cover - dependency-free environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class IntegratedProbeTests(unittest.TestCase):
    def setUp(self):
        from torch_transformer_benchmark import TransformerConfig

        self.config = TransformerConfig(
            batch_size=2,
            seq_len=8,
            d_model=16,
            num_heads=4,
            ffn_dim=16,
            num_layers=2,
            causal=True,
        )

    def _models(self):
        from src.implementations.integrated_probe import IntegratedProbeTransformer
        from torch_transformer_benchmark import (
            BaselineTransformer,
            copy_model_weights,
        )

        baseline = BaselineTransformer(self.config).eval()
        candidate = IntegratedProbeTransformer(self.config).eval()
        copy_model_weights(baseline, candidate)
        return baseline, candidate

    def test_preserves_reference_state_dict_contract(self):
        baseline, candidate = self._models()
        self.assertEqual(list(baseline.state_dict()), list(candidate.state_dict()))

    def test_matches_reference_for_all_valid_causal_input(self):
        from torch_transformer_benchmark import compare_outputs

        torch.manual_seed(1234)
        baseline, candidate = self._models()
        x = torch.randn(2, 8, 16)
        mask = torch.ones(2, 8, dtype=torch.bool)
        with torch.inference_mode():
            expected = baseline(x, mask)
            actual = candidate(x, mask)
        result = compare_outputs(expected, actual, rtol=0.02, atol=0.002)
        self.assertTrue(result.passed, result)

    def test_matches_reference_with_padding(self):
        from torch_transformer_benchmark import compare_outputs

        torch.manual_seed(5678)
        baseline, candidate = self._models()
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
        result = compare_outputs(expected, actual, rtol=0.02, atol=0.002)
        self.assertTrue(result.passed, result)


if __name__ == "__main__":
    unittest.main()
