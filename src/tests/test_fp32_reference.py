from __future__ import annotations

import unittest

import torch

from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    copy_model_weights,
)

from src.implementations.fp32_reference import (
    LinearMemoryFP32Reference,
    LinearMemoryFP32SelfAttention,
    compare_outputs_streamed,
)


class FP32ReferenceTests(unittest.TestCase):
    def _config(self, *, seq_len: int = 32) -> TransformerConfig:
        return TransformerConfig(2, seq_len, 32, 4, 32, 2, True)

    def test_matches_immutable_dense_reference(self):
        torch.manual_seed(1234)
        config = self._config()
        dense = BaselineTransformer(config).eval()
        oracle = LinearMemoryFP32Reference(config).eval()
        copy_model_weights(dense, oracle)
        x = torch.randn(config.batch_size, config.seq_len, config.d_model)
        mask = torch.ones(config.batch_size, config.seq_len, dtype=torch.bool)

        with torch.inference_mode():
            expected = dense(x, mask)
            actual = oracle(x, mask)
        result = compare_outputs_streamed(expected, actual, token_chunk=7)

        self.assertTrue(result.passed)
        self.assertEqual(result.failed_elements, 0)
        self.assertLess(result.max_abs_error, 1e-4)

    def test_replacement_preserves_seeded_reference_weights(self):
        config = self._config()
        torch.manual_seed(9876)
        dense = BaselineTransformer(config).eval()
        torch.manual_seed(9876)
        oracle = LinearMemoryFP32Reference(config).eval()

        for name, expected in dense.state_dict().items():
            self.assertTrue(torch.equal(expected, oracle.state_dict()[name]), name)

    def test_matches_right_padded_dense_reference(self):
        torch.manual_seed(4321)
        config = self._config(seq_len=24)
        dense = BaselineTransformer(config).eval()
        oracle = LinearMemoryFP32Reference(config).eval()
        copy_model_weights(dense, oracle)
        x = torch.randn(config.batch_size, config.seq_len, config.d_model)
        positions = torch.arange(config.seq_len)
        mask = positions[None, :] < torch.tensor([17, 9])[:, None]
        x.masked_fill_(~mask[..., None], 0)

        with torch.inference_mode():
            expected = dense(x, mask)
            actual = oracle(x, mask)
        result = compare_outputs_streamed(expected, actual, token_chunk=5)

        self.assertTrue(result.passed)
        self.assertEqual(result.failed_elements, 0)

    def test_rejects_non_prefix_mask(self):
        attention = LinearMemoryFP32SelfAttention(32, 4).eval()
        x = torch.randn(1, 8, 32)
        mask = torch.tensor([[True, True, False, True, False, False, False, False]])
        with self.assertRaisesRegex(ValueError, "right-padded"):
            attention(x, mask, causal=True)

    def test_rejects_non_fp32_input(self):
        attention = LinearMemoryFP32SelfAttention(32, 4).half().eval()
        with self.assertRaisesRegex(RuntimeError, "requires float32"):
            attention(torch.randn(1, 8, 32).half(), causal=True)

    def test_streamed_checker_uses_benchmark_or_rule(self):
        reference = torch.tensor([[[0.0, 1.0, 100.0]]])
        candidate = torch.tensor([[[0.0019, 1.019, 102.001]]])
        result = compare_outputs_streamed(
            reference, candidate, atol=0.002, rtol=0.02, token_chunk=1
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failed_elements, 1)


if __name__ == "__main__":
    unittest.main()
