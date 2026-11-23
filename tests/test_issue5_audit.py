"""Regression tests for Issue 5 (LOFO cross-model asymmetry & sign flips, w_perp scope, Phase 2 steering ledger)."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.make_figures import collect_verified_tables, format_verified_markdown


class TestIssue5Audit(unittest.TestCase):
    def test_issue5_lofo_asymmetry_wperp_scope_and_phase2_ledger(self) -> None:
        tables = collect_verified_tables(Path("results"))

        q_hon = tables["phase4a_mechanism_lofo"]["qwen_L24"]["L24_dec_std"]["fewshot_streak"]["honest"]
        l_hon = tables["phase4a_mechanism_lofo"]["llama_L19"]["L19_dec_std"]["fewshot_streak"]["honest"]
        l_dis_da = tables["phase4a_mechanism_lofo"]["llama_L19"]["L19_dec_raw"]["draft_autofill"]["distort"]
        l_dis_fs = tables["phase4a_mechanism_lofo"]["llama_L19"]["L19_dec_raw"]["fewshot_streak"]["distort"]

        self.assertAlmostEqual(q_hon["delta"], 0.0972, places=4)
        self.assertAlmostEqual(q_hon["fixed_sign_probe_auroc"], 0.7800, places=4)
        self.assertAlmostEqual(l_hon["delta"], 0.0127, places=4)
        self.assertGreater(l_hon["sign_p"], 0.05)
        self.assertLess(l_dis_da["fixed_sign_probe_auroc"], 0.35)
        self.assertLess(l_dis_da["mean_diff_false_minus_true"], 0.0)
        self.assertLess(l_dis_fs["fixed_sign_probe_auroc"], 0.45)
        self.assertLess(l_dis_fs["mean_diff_false_minus_true"], 0.0)

        p3 = tables["phase3_mirror"]
        self.assertAlmostEqual(p3["qwen_L24_raw"]["cos_dec_surp"], 0.0829, places=3)
        self.assertAlmostEqual(p3["llama_L19_raw"]["cos_dec_surp"], 0.0118, places=3)

        q24_p2 = tables["phase2_steering_pm2"]["qwen_L24"]
        l19_p2 = tables["phase2_steering_pm2"]["llama_L19"]
        self.assertGreater(
            abs(q24_p2["benign_rule"]["dec"]["odd_even"]["pm_2.00"]["odd_directional"]["mean"]),
            abs(q24_p2["reversible_honest"]["dec"]["odd_even"]["pm_2.00"]["odd_directional"]["mean"]),
        )
        self.assertGreater(
            abs(l19_p2["benign_rule"]["dec"]["odd_even"]["pm_2.00"]["odd_directional"]["mean"]),
            abs(l19_p2["reversible_distort"]["dec"]["odd_even"]["pm_2.00"]["odd_directional"]["mean"]),
        )

        md = format_verified_markdown(tables)
        self.assertIn("LOFO sign-flip audit", md)
        self.assertIn("rules out 1D formatting-surprisal debiasing", md)
        self.assertIn("Qwen L24 `dec`:", md)
        self.assertIn("Qwen L22 `dec`:", md)

        ledger = Path("docs/CLAIMS_LEDGER.md").read_text(encoding="utf-8")
        self.assertIn("no fixed-sign incremental gain over probability replicates across both models", ledger)
        self.assertIn("naive 1D linear formatting-surprisal debiasing", ledger)
        self.assertIn("Qwen L24 `-0.816`", ledger)


if __name__ == "__main__":
    unittest.main()
