"""Unit tests for Step 2 assistant-span slicing and JSONL flattening."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.cache_activations import compute_assistant_span, flatten_jsonl_records


class TestCacheUtils(unittest.TestCase):
    def test_valid_span_with_mask(self) -> None:
        start, end = compute_assistant_span(prompt_len=40, total_len=60, exclude_last_n=5)
        self.assertEqual((start, end), (40, 55))

    def test_starved_span_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_assistant_span(prompt_len=40, total_len=44, exclude_last_n=5)

    def test_flatten_contrastive_pairs_doubles_rows(self) -> None:
        rows = flatten_jsonl_records(Path("data/processed/pilot_v2_clean.jsonl"))
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(r["label"] == 0 for r in rows), 10)
        self.assertEqual(sum(r["label"] == 1 for r in rows), 10)
        self.assertTrue(all(r["exclude_last_n"] == 5 for r in rows))

    def test_flatten_hard_negatives_preserves_labels(self) -> None:
        rows = flatten_jsonl_records(Path("data/processed/hard_negatives.jsonl"))
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(r["label"] == 0 for r in rows))
        self.assertTrue(all(r["exclude_last_n"] == 0 for r in rows))


if __name__ == "__main__":
    unittest.main()