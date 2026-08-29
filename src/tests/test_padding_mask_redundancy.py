"""The padding key mask is dead code under causal attention.

The spec removes the padding ``masked_fill`` on the strength of an argument
about the reference's own semantics: causal masking already writes -inf
everywhere a right-padded key mask would, and rows that genuinely differ are
zeroed by the reference before they can propagate.

This test pins that property to the reference implementation itself, with the
padding mask as the only variable. If the harness ever emits a mask that is not
right-padded, or stops zeroing invalid rows, this fails -- which is the signal
that the spec's section 3 no longer holds.
"""

from __future__ import annotations

import unittest


try:
    import torch
except ImportError:  # pragma: no cover - dependency-free environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PaddingMaskRedundancyTests(unittest.TestCase):
    def _no_pad_mask_model(self, config):
        from torch_transformer_benchmark import BaselineSelfAttention, BaselineTransformer

        class NoPadMaskAttention(BaselineSelfAttention):
            """Reference arithmetic with the padding masked_fill removed."""

            def forward(self, x, valid_token_mask=None, causal=False):
                batch, seq_len, _ = x.shape
                q = self._split_heads(self.q_proj(x))
                k = self._split_heads(self.k_proj(x))
                v = self._split_heads(self.v_proj(x))
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                if causal:
                    blocked = torch.ones(
                        (seq_len, seq_len), device=x.device, dtype=torch.bool
                    ).triu(1)
                    scores = scores.masked_fill(blocked, float("-inf"))
                probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
                context = torch.matmul(probs, v)
                context = (
                    context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
                )
                output = self.out_proj(context)
                if valid_token_mask is not None:
                    output = output.masked_fill(~valid_token_mask[..., None], 0)
                return output

        model = BaselineTransformer(config)
        for layer in model.layers:
            layer.attention = NoPadMaskAttention(config.d_model, config.num_heads)
        return model.eval()

    def test_removing_the_padding_mask_is_bitwise_exact(self) -> None:
        from torch_transformer_benchmark import (
            BaselineTransformer, TransformerConfig, copy_model_weights,
            generate_random_case,
        )

        config = TransformerConfig(
            batch_size=4, seq_len=16, d_model=16, num_heads=4,
            ffn_dim=16, num_layers=2, causal=True,
        )
        torch.manual_seed(19)
        reference = BaselineTransformer(config).eval()
        variant = self._no_pad_mask_model(config)
        copy_model_weights(reference, variant, strict=True)

        for ratio in (0.0, 0.3, 0.5, 0.9):
            for seed in (101, 202):
                with self.subTest(ratio=ratio, seed=seed):
                    x, mask = generate_random_case(
                        config, torch.device("cpu"), torch.float32, seed, ratio, 1.0
                    )
                    with torch.inference_mode():
                        delta = (reference(x, mask) - variant(x, mask)).abs().max()
                    self.assertEqual(delta.item(), 0.0)

    def test_harness_masks_are_right_padded(self) -> None:
        """The precondition for the proof above."""
        from torch_transformer_benchmark import TransformerConfig, generate_random_case

        config = TransformerConfig(
            batch_size=16, seq_len=32, d_model=8, num_heads=2,
            ffn_dim=8, num_layers=1, causal=True,
        )
        for ratio in (0.1, 0.3, 0.5, 0.9):
            for seed in (1, 2, 3):
                _, mask = generate_random_case(
                    config, torch.device("cpu"), torch.float32, seed, ratio, 1.0
                )
                rises = (mask[:, :-1] < mask[:, 1:]).any()
                self.assertFalse(
                    bool(rises), msg=f"ratio={ratio} seed={seed} is not right-padded"
                )


if __name__ == "__main__":
    unittest.main()
