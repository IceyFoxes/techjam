from __future__ import annotations

import unittest
from unittest import mock


try:
    import torch
except ImportError:  # pragma: no cover - dependency-free environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ExtremeMemoryTests(unittest.TestCase):
    def test_right_padded_lengths_accepts_prefix_masks(self):
        from src.implementations.extreme import right_padded_lengths

        mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
                [False, False, False, False],
            ]
        )

        self.assertEqual(right_padded_lengths(mask), [4, 2, 0])

    def test_right_padded_lengths_rejects_interior_holes(self):
        from src.implementations.extreme import right_padded_lengths

        mask = torch.tensor([[True, False, True]])
        with self.assertRaisesRegex(ValueError, "right-padded"):
            right_padded_lengths(mask)

    def test_batch_chunks_preserve_order_and_masks(self):
        from src.implementations.extreme import forward_batch_chunks

        x = torch.arange(30, dtype=torch.float32).view(5, 3, 2)
        mask = torch.tensor(
            [
                [True, True, True],
                [True, True, False],
                [True, False, False],
                [True, True, True],
                [True, True, False],
            ]
        )

        def forward(chunk, chunk_mask):
            return (chunk + 1).masked_fill(~chunk_mask[..., None], 0)

        actual = forward_batch_chunks(forward, x, mask, chunk_size=2)
        expected = (x + 1).masked_fill(~mask[..., None], 0)
        self.assertTrue(torch.equal(actual, expected))

    def test_batch_chunks_halves_after_cuda_oom(self):
        from src.implementations.extreme import forward_batch_chunks

        x = torch.arange(24, dtype=torch.float32).view(4, 3, 2)
        attempted = []

        def forward(chunk, chunk_mask):
            del chunk_mask
            attempted.append(chunk.shape[0])
            if chunk.shape[0] > 1:
                raise torch.OutOfMemoryError("synthetic")
            return chunk + 1

        with mock.patch("src.implementations.extreme.torch.cuda.empty_cache"):
            actual = forward_batch_chunks(forward, x, None, chunk_size=4)

        self.assertTrue(torch.equal(actual, x + 1))
        self.assertEqual(attempted, [4, 2, 1, 1, 1, 1])

    def test_prefix_chunks_trim_padding_and_restore_shape(self):
        from src.implementations.extreme import forward_prefix_chunks

        x = torch.arange(32, dtype=torch.float32).view(4, 4, 2)
        mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, False, False],
                [True, False, False, False],
            ]
        )
        calls = []

        def forward(chunk, chunk_mask):
            calls.append((tuple(chunk.shape), chunk_mask))
            return chunk + 1

        actual = forward_prefix_chunks(forward, x, mask, chunk_size=2)
        expected = (x + 1).masked_fill(~mask[..., None], 0)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            calls,
            [((2, 4, 2), None), ((1, 2, 2), None), ((1, 1, 2), None)],
        )

    def test_chunk_chooser_uses_largest_safe_power_of_two(self):
        from src.implementations.extreme import estimate_batch_chunk_size

        chunk = estimate_batch_chunk_size(
            batch_size=10000,
            seq_len=128,
            d_model=128,
            num_heads=4,
            element_size=4,
            free_bytes=24 * 2**30,
            total_bytes=24 * 2**30,
            score_copies=12,
            activation_copies=12,
        )

        self.assertEqual(chunk, 4096)

    def test_case14_estimate_reserves_full_output(self):
        from src.implementations.extreme import estimate_batch_chunk_size

        chunk = estimate_batch_chunk_size(
            batch_size=32,
            seq_len=100000,
            d_model=1024,
            num_heads=16,
            element_size=2,
            free_bytes=23_034 * 2**20,
            total_bytes=23_034 * 2**20,
            score_copies=0,
            activation_copies=12,
        )

        self.assertEqual(chunk, 4)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_flash_attention_matches_reference_at_supported_scale(self):
        from torch_transformer_benchmark import (
            BaselineTransformer,
            TransformerConfig,
            compare_outputs,
            copy_model_weights,
        )
        from src.implementations.extreme import (
            FlashOnlySDPASelfAttention,
            forward_prefix_chunks,
        )

        torch.manual_seed(1234)
        config = TransformerConfig(2, 128, 1024, 16, 1024, 2, True)
        baseline = BaselineTransformer(config).cuda().half().eval()
        candidate = BaselineTransformer(config).cuda().half().eval()
        for layer in candidate.layers:
            layer.attention = FlashOnlySDPASelfAttention(1024, 16).cuda().half()
        copy_model_weights(baseline, candidate)
        x = torch.randn(2, 128, 1024, device="cuda", dtype=torch.float16)
        mask = torch.ones(2, 128, device="cuda", dtype=torch.bool)

        with torch.inference_mode():
            expected = baseline(x, mask)
            actual = forward_prefix_chunks(
                lambda chunk, _: candidate(chunk, None),
                x,
                mask,
                chunk_size=1,
            )
        result = compare_outputs(expected, actual, rtol=0.02, atol=0.002)

        self.assertTrue(result.passed, result)


if __name__ == "__main__":
    unittest.main()
