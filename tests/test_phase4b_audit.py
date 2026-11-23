"""Regression tests for Issue 4 (Phase 4B decomposition: prompt_len, Llama header shift, stratification, 5% FPR primary)."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.make_figures import collect_verified_tables, format_verified_markdown


class TestPhase4BAudit(unittest.TestCase):
    def test_issue4_phase4b_decomposition_and_stratification(self) -> None:
        tables = collect_verified_tables(Path("results"))
        q = tables["phase4b_decomposition"]["qwen_L24"]
        l = tables["phase4b_decomposition"]["llama_L19"]

        self.assertAlmostEqual(q["cross_arm_aurocs"]["prompt_len"], 1.0, places=4)
        self.assertAlmostEqual(l["cross_arm_aurocs"]["prompt_len"], 1.0, places=4)
        self.assertAlmostEqual(q["within_pressured_incentive_aurocs"]["prompt_len"], 0.6353, places=4)
        self.assertGreater(q["within_pressured_incentive_aurocs"]["prompt_len"], q["within_pressured_incentive_aurocs"]["gen_mean_std"])
        self.assertAlmostEqual(l["within_pressured_incentive_aurocs"]["prompt_len"], 0.5508, places=4)

        self.assertLess(abs(q["benign_to_honest_header_shift"]["z_se"]), 1.0)
        self.assertLess(l["benign_to_honest_header_shift"]["z_se"], -14.0)

        q_st = q["pressured_incentive_stratified_regex"]
        l_st = l["pressured_incentive_stratified_regex"]
        self.assertEqual(sum(v["n"] for v in q_st.values()), 40)
        self.assertEqual(sum(v["n"] for v in l_st.values()), 40)
        self.assertEqual(sum(v["n_refusal"] for v in q_st.values()), 0)
        self.assertEqual(sum(v["n_refusal"] for v in l_st.values()), 0)
        self.assertLess(q_st["unqualified_false"]["header_std_mean"], q_st["both_mentioned"]["header_std_mean"])
        self.assertLess(l_st["unqualified_false"]["header_std_mean"], l_st["both_mentioned"]["header_std_mean"])

        self.assertEqual(q["operating_points"]["L24_dec_std_gen_mean"]["primary_benign_fpr_05"]["all40_honest_flagged"], 2)
        self.assertEqual(q["operating_points"]["L24_dec_std_header"]["primary_benign_fpr_05"]["all40_honest_flagged"], 0)
        self.assertEqual(l["operating_points"]["L19_dec_std_gen_mean"]["primary_benign_fpr_05"]["all40_honest_flagged"], 0)

        md = format_verified_markdown(tables)
        self.assertIn("5% Benign FPR Primary", md)
        self.assertIn("prompt_len=0.6353", md)
        self.assertIn("not invariant", md)
        self.assertIn("0 refusals", md)

        ledger = Path("docs/CLAIMS_LEDGER.md").read_text(encoding="utf-8")
        self.assertIn("`0.6353`", ledger)
        self.assertIn("z = -14.95 SE", ledger)
        self.assertIn("0 refusals", ledger)
        self.assertIn("5%` benign FPR = 10 tail samples", ledger)


if __name__ == "__main__":
    unittest.main()
