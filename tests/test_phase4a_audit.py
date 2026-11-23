"""Regression tests for Issue 1 (Phase 4A label invariants & manipulation check)."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.make_figures import collect_verified_tables, format_verified_markdown


class TestPhase4AAudit(unittest.TestCase):
    def test_issue1_label_invariants_and_llama_antiprior_check(self) -> None:
        tables = collect_verified_tables(Path("results"))
        mc_q = tables["phase4a_manipulation_check"]["qwen_L24"]
        mc_l = tables["phase4a_manipulation_check"]["llama_L19"]

        self.assertEqual(mc_q["label_audit"]["label_consistency_mismatches"], 0)
        self.assertEqual(mc_l["label_audit"]["label_consistency_mismatches"], 0)
        self.assertEqual(mc_q["label_audit"]["n_rows"], 1536)
        self.assertEqual(mc_l["label_audit"]["n_rows"], 1536)

        self.assertGreater(mc_q["fewshot_streak__honest"]["paired_delta_lo"]["mean"], 10.0)
        self.assertAlmostEqual(mc_l["all_mechanisms__honest"]["paired_delta_lo"]["mean"], -2.8557, places=3)
        self.assertEqual(mc_l["all_mechanisms__honest"]["n_lo_pos_prior_true"], 112)
        self.assertEqual(mc_l["all_mechanisms__honest"]["n_lo_pos_prior_false"], 26)

        md = format_verified_markdown(tables)
        self.assertIn("0/1536", md)
        self.assertIn("anti-prior repulsion", md)


if __name__ == "__main__":
    unittest.main()
