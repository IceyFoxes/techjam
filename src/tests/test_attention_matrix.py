"""GPU correctness matrix for the attention candidate (spec section 6).

Skipped without CUDA. Small-N cases only, so the suite stays minutes rather
than hours; the full official sweep is the benchmark task's job.
"""

from __future__ import annotations

import unittest


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

CASES = (1, 7, 9, 10, 11, 12)
RATIOS = (0.0, 0.3)
SCALES = (0.1, 1.0, 10.0)
SEEDS = (1234, 1235, 1236, 1237, 1238)


@unittest.skipIf(torch is None, "PyTorch is not installed")
@unittest.skipIf(
    torch is not None and not torch.cuda.is_available(), "CUDA is required"
)
class AttentionCorrectnessMatrixTests(unittest.TestCase):
    def test_matrix(self) -> None:
        from torch_transformer_benchmark import (
            BaselineTransformer, TransformerConfig, compare_outputs,
            copy_model_weights, generate_random_case,
        )
        from src.infra.cases import load_official_cases
        from src.implementations.attention import AttentionCandidate

        official = load_official_cases()
        device = torch.device("cuda")
        for case_id in CASES:
            case = official[case_id]
            config = TransformerConfig(
                batch_size=case.batch_size, seq_len=case.seq_len,
                d_model=case.qkv_dim, num_heads=case.heads,
                ffn_dim=case.ffn_dim, num_layers=case.layers,
                causal=case.causal,
            )
            baseline = BaselineTransformer(config).to(device, torch.float32).eval()
            candidate = AttentionCandidate(config).to(device, torch.float32).eval()
            copy_model_weights(baseline, candidate, strict=True)
            for ratio in RATIOS:
                for scale in SCALES:
                    for seed in SEEDS:
                        with self.subTest(
                            case=case_id, ratio=ratio, scale=scale, seed=seed
                        ):
                            x, mask = generate_random_case(
                                config, device, torch.float32, seed, ratio, scale
                            )
                            with torch.inference_mode():
                                result = compare_outputs(
                                    baseline(x, mask), candidate(x, mask), 0.02, 0.002
                                )
                            self.assertTrue(
                                result.passed,
                                msg=(
                                    f"case {case_id} ratio={ratio} scale={scale} "
                                    f"seed={seed}: {result.failed_elements} failed, "
                                    f"max_abs={result.max_abs_error}"
                                ),
                            )
            del baseline, candidate
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
