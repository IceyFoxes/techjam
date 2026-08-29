"""Shared strided-SDPA attention implementations used by integration candidates."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineSelfAttention


class StridedSDPASelfAttention(BaselineSelfAttention):
    """Reference-compatible projections around SDPA without Q/K/V copies."""

    def _split_heads_view(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward_reference(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Execute the immutable benchmark's explicit attention arithmetic."""

        return BaselineSelfAttention.forward(self, x, valid_token_mask, causal)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads_view(self.q_proj(x))
        k = self._split_heads_view(self.k_proj(x))
        v = self._split_heads_view(self.v_proj(x))

        # True means that a key participates in attention. Do not inspect mask
        # values on the host: that would synchronize and break graph replay.
        attn_mask = (
            None
            if valid_token_mask is None
            else valid_token_mask[:, None, None, :]
        )
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=causal,
            scale=self.scale,
        )
        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class PackedQKVSDPASelfAttention(StridedSDPASelfAttention):
    """Inference-only packed QKV projection feeding copy-free SDPA views."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__(d_model, num_heads)
        self.register_buffer(
            "_packed_qkv_weight",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer("_packed_qkv_bias", torch.empty(0), persistent=False)
        self.refresh_packed_qkv()
        self.register_load_state_dict_post_hook(self._repack_after_load)

    @torch.no_grad()
    def refresh_packed_qkv(self) -> None:
        """Refresh inference caches after out-of-band parameter mutation."""

        weight = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight),
            dim=0,
        ).detach()
        bias = torch.cat(
            (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias),
            dim=0,
        ).detach()
        if (
            self._packed_qkv_weight.shape == weight.shape
            and self._packed_qkv_weight.device == weight.device
            and self._packed_qkv_weight.dtype == weight.dtype
        ):
            self._packed_qkv_weight.copy_(weight)
            self._packed_qkv_bias.copy_(bias)
        else:
            self._packed_qkv_weight = weight
            self._packed_qkv_bias = bias

    def _repack_after_load(self, module, incompatible_keys) -> None:
        del module, incompatible_keys
        self.refresh_packed_qkv()

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self.refresh_packed_qkv()
        return result

    def project_qkv(self, x: torch.Tensor):
        batch, seq_len, _ = x.shape
        packed = F.linear(x, self._packed_qkv_weight, self._packed_qkv_bias)
        qkv = packed.view(
            batch,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        # The packed buffers are inference caches, not trainable parameters.
        if torch.is_grad_enabled():
            return super().forward(x, valid_token_mask, causal)

        batch, seq_len, _ = x.shape
        q, k, v = self.project_qkv(x)
        attn_mask = (
            None
            if valid_token_mask is None
            else valid_token_mask[:, None, None, :]
        )
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=causal,
            scale=self.scale,
        )
        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output
