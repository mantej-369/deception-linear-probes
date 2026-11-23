"""Regression tests for Issue 3 (Phase 3 frozen primary layer reporting: Llama L19 vs secondary sweep)."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.make_figures import collect_verified_tables, format_verified_markdown


class TestPhase3PrimaryLayerAudit(unittest.TestCase):
    def test_issue3_llama_l19_is_primary_and_sweep_peaks_are_accurate(self) -> None:
        tables = collect_verified_tables(Path("results"))
        p3 = tables["phase3_mirror"]

        l19_hon = p3["llama_L19_raw"]["cover_honest_all"]["w_dec"]
        l19_lie = p3["llama_L19_raw"]["cover_lie_verified_flips"]["w_dec"]
        l19_prp = p3["llama_L19_raw"]["cover_lie_verified_flips"]["w_perp"]
        self.assertAlmostEqual(l19_hon["mean"], 0.1321, places=3)
        self.assertEqual(l19_hon["pos_worlds"], 35)
        self.assertAlmostEqual(l19_lie["mean"], -0.6176, places=3)
        self.assertEqual(l19_lie["pos_worlds"], 0)
        self.assertAlmostEqual(l19_prp["mean"], -0.5806, places=3)
        self.assertAlmostEqual(p3["llama_L19_pref_auc"], 0.7606, places=4)

        # Verify true peak-magnitude ordering: |L27| > |L26| > |L23| > |L19|
        m19 = abs(p3["llama_L19_raw"]["cover_lie_verified_flips"]["w_dec"]["mean"])
        m23 = abs(p3["llama_L23_raw"]["cover_lie_verified_flips"]["w_dec"]["mean"])
        m26 = abs(p3["llama_L26_raw"]["cover_lie_verified_flips"]["w_dec"]["mean"])
        m27 = abs(p3["llama_L27_raw"]["cover_lie_verified_flips"]["w_dec"]["mean"])
        self.assertLess(m19, m23)
        self.assertLess(m23, m26)
        self.assertLess(m26, m27)

        md = format_verified_markdown(tables)
        self.assertIn("**Llama L19 RAW (frozen primary):**", md)
        self.assertIn("**Qwen L24 STD (frozen primary):**", md)
        self.assertLess(md.index("Llama L19 RAW (frozen primary)"), md.index("Llama L23 RAW"))
        self.assertLess(md.index("Qwen L24 STD (frozen primary)"), md.index("Qwen L25 STD"))

        ledger = Path("docs/CLAIMS_LEDGER.md").read_text(encoding="utf-8")
        self.assertIn("Llama L19 RAW `+0.132`", ledger)
        self.assertIn("Llama L19 RAW `-0.618`", ledger)
        self.assertIn("PrefAUC `0.7606`", ledger)
        self.assertIn("late-layer PrefAUC local peak `L23 RAW`", ledger)


if __name__ == "__main__":
    unittest.main()
