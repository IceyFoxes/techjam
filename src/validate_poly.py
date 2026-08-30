#!/usr/bin/env python3
"""End-to-end criterion for the polynomial attention path at case-14 shapes.

Two oracles, both required (spec section 7.2):

* ``dense``  -- the immutable ``BaselineTransformer``. Authoritative, but its
  N x N score tensor limits it to about N=8192 in 8 GiB.
* ``flash``  -- the same model with exact Flash SDPA attention. Algebraically
  exact, so it isolates approximation error from float16 reduction-order noise,
  and it runs to N=100000.

Neither alone is sufficient: the dense oracle cannot reach the scale that
matters, and the flash oracle cannot see fp16 reduction-order differences.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch_transformer_benchmark import (
    BaselineSelfAttention,
    BaselineTransformer,
    TransformerConfig,
    copy_model_weights,
)
from src.implementations.poly_attention import poly_attention_forward
from src.implementations.poly_guard import estimate_sigma

ATOL, RTOL = 0.002, 0.02
D_MODEL, HEADS, FFN, LAYERS = 1024, 16, 1024, 2


class _FlashAttention(BaselineSelfAttention):
    """Algebraically exact attention, runnable at N=100000."""

    def forward(self, x, valid_token_mask=None, causal=False):
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            context = F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, scale=self.scale
            )
        return self.out_proj(
            context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        )


class _PolyAttention(BaselineSelfAttention):
    """The candidate under test."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__(d_model, num_heads)
        self.measured_sigma = None
        self.disable = frozenset()

    def forward(self, x, valid_token_mask=None, causal=False):
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if self.measured_sigma is None:
            self.measured_sigma = estimate_sigma(q, k, self.scale)
        context = poly_attention_forward(
            q, k, v, self.scale, sigma=self.measured_sigma, disable=self.disable
        )
        return self.out_proj(
            context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        )


def _build(attn_cls, config, device, seed=0):
    torch.manual_seed(seed)
    model = BaselineTransformer(config)
    if attn_cls is not None:
        for layer in model.layers:
            layer.attention = attn_cls(config.d_model, config.num_heads)
    return model.to(device).half().eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--oracle", choices=("dense", "flash"), default="flash")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--disable", default="")
    parser.add_argument(
        "--scale-qk",
        type=float,
        default=1.0,
        help=(
            "multiply the input. NOTE: this does NOT move sigma -- the first "
            "op in every layer is LayerNorm, which is scale-invariant. Use "
            "--scale-qk-weights to sweep sigma."
        ),
    )
    parser.add_argument(
        "--scale-qk-weights",
        type=float,
        default=1.0,
        help=(
            "scale the q_proj and k_proj weights of BOTH models, which scales "
            "scores by the square of this factor. This is the sigma sweep."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    config = TransformerConfig(1, args.n, D_MODEL, HEADS, FFN, LAYERS, True)

    oracle = _build(
        None if args.oracle == "dense" else _FlashAttention, config, device
    )
    candidate = _build(_PolyAttention, config, device)
    disabled = frozenset(filter(None, args.disable.split(",")))
    for layer in candidate.layers:
        layer.attention.disable = disabled
    copy_model_weights(oracle, candidate)

    if args.scale_qk_weights != 1.0:
        # Scale Q/K projections in BOTH models so they still compute the same
        # function; only the score magnitude, and therefore sigma, changes.
        # This is how a trained model's larger scores are simulated.
        with torch.no_grad():
            for model in (oracle, candidate):
                for layer in model.layers:
                    for projection in (
                        layer.attention.q_proj,
                        layer.attention.k_proj,
                    ):
                        projection.weight.mul_(args.scale_qk_weights)
                        projection.bias.mul_(args.scale_qk_weights)

    torch.manual_seed(args.seed)
    x = torch.randn(1, args.n, D_MODEL, device=device, dtype=torch.float16)
    x = x * args.scale_qk

    with torch.inference_mode():
        ref = oracle(x).float()
        del oracle
        torch.cuda.empty_cache()
        got = candidate(x).float()

    err = (got - ref).abs()
    tol = torch.clamp(ref.abs() * RTOL, min=ATOL)
    failures = int((err > tol).sum())
    sigma = candidate.layers[0].attention.measured_sigma
    print(
        f"N={args.n} oracle={args.oracle} wscale={args.scale_qk_weights} "
        f"sigma={sigma:.4f} failures={failures}/{err.numel()} "
        f"max={err.max().item():.4e} rms={err.pow(2).mean().sqrt().item():.4e} "
        f"{'PASS' if failures == 0 else 'FAIL'}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
