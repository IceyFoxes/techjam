from __future__ import annotations

import sys
import types
import unittest

from src.infra import CandidateSpec, load_candidate, load_official_cases


class CandidateContractTests(unittest.TestCase):
    def test_loads_explicit_candidate_attribute(self) -> None:
        module_name = "test_candidates.example"
        module = types.ModuleType(module_name)
        expected = CandidateSpec(
            name="example",
            model_factory=lambda config: config,
            owner="test",
            description="test candidate",
        )
        module.SPEC = expected
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)

        self.assertIs(load_candidate(f"{module_name}:SPEC"), expected)

    def test_rejects_invalid_candidate_name(self) -> None:
        candidate = CandidateSpec(
            name="Not Valid",
            model_factory=lambda config: config,
            owner="test",
            description="test candidate",
        )
        with self.assertRaisesRegex(ValueError, "candidate name"):
            candidate.validate()

    def test_rejects_non_spec_export(self) -> None:
        module_name = "test_candidates.invalid"
        module = types.ModuleType(module_name)
        module.CANDIDATE = object()
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)

        with self.assertRaisesRegex(TypeError, "must be a CandidateSpec"):
            load_candidate(module_name)

    def test_rejects_invalid_unsupported_case_metadata(self) -> None:
        for unsupported in ((0,), (15,), (6, 6), [6]):
            with self.subTest(unsupported=unsupported):
                candidate = CandidateSpec(
                    name="example",
                    model_factory=lambda config: config,
                    owner="test",
                    description="test candidate",
                    unsupported_official_cases=unsupported,
                )
                with self.assertRaises((TypeError, ValueError)):
                    candidate.validate()

    def test_dispatcher_declares_self_compilation_and_extreme_cases(self) -> None:
        candidate = load_candidate("src.dispatcher")
        self.assertTrue(candidate.self_compiling)
        self.assertEqual(candidate.unsupported_official_cases, (6, 14))

    def test_benchmark_rejects_nested_compilation(self) -> None:
        from src.infra import validate_candidate_execution

        candidate = load_candidate("src.dispatcher")
        with self.assertRaisesRegex(ValueError, "compiles itself"):
            validate_candidate_execution(
                candidate,
                official_case_id=1,
                compile_user=True,
            )

    def test_benchmark_rejects_extreme_case_before_allocation(self) -> None:
        from src.infra import validate_candidate_execution

        candidate = load_candidate("src.dispatcher")
        with self.assertRaisesRegex(ValueError, "before model/input allocation"):
            validate_candidate_execution(
                candidate,
                official_case_id=14,
                compile_user=False,
            )


class OfficialCasesTests(unittest.TestCase):
    def test_disclosed_table_contains_exactly_fourteen_valid_cases(self) -> None:
        cases = load_official_cases()

        self.assertEqual(list(cases), list(range(1, 15)))
        self.assertTrue(all(case.causal for case in cases.values()))

    def test_extreme_cases_match_the_appendix(self) -> None:
        cases = load_official_cases()

        self.assertEqual(cases[6].batch_size, 10000)
        self.assertEqual(cases[13].seq_len, 1024)
        self.assertEqual(
            cases[14].benchmark_config(),
            {
                "batch_size": 32,
                "d_model": 1024,
                "heads": 16,
                "seq_len": 100000,
                "layers": 2,
                "causal": True,
                "ffn_dim": 1024,
            },
        )


class BenchmarkMemoryTests(unittest.TestCase):
    def test_cpu_memory_probe_is_explicitly_unavailable(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        from src.benchmark import _cuda_memory_probe

        self.assertIsNone(
            _cuda_memory_probe(
                baseline=object(),
                candidate=object(),
                x=torch.empty(0),
                valid_mask=torch.empty(0, dtype=torch.bool),
                device=torch.device("cpu"),
            )
        )


if __name__ == "__main__":
    unittest.main()
