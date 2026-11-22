"""Unit tests for HF model revision resolution, top-k vocabulary extraction, and multi-suite no-op checks."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import numpy as np

from src.provenance import extract_top_k_tokens, resolve_model_commit, select_noop_check_indices
from src.surprisal_confound import analyze_layer_orthogonalisation, build_surprisal_confound_rows


class _DummyTokenizer:
    def decode(self, ids: list[int]) -> str:
        return f"tok_{ids[0]}"


class TestProvenance(unittest.TestCase):
    def test_resolve_model_commit_prefers_config_hash(self) -> None:
        m = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash="abcdef1234567890"))
        self.assertEqual(resolve_model_commit("org/model", model=m), "abcdef1234567890")

    def test_resolve_model_commit_falls_back_to_hub_api(self) -> None:
        m = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash=None))
        fake_hub = types.SimpleNamespace(model_info=lambda _mid: types.SimpleNamespace(sha="1234567890abcdef"))
        with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
            self.assertEqual(resolve_model_commit("org/model", model=m), "1234567890abcdef")

    def test_resolve_model_commit_falls_back_to_local_snapshot_path(self) -> None:
        m = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash="unknown"))

        def _fail(_mid: str):
            raise RuntimeError("offline")

        fake_hub = types.SimpleNamespace(
            model_info=_fail,
            try_to_load_from_cache=lambda _m, _f: "/cache/hub/models--x/snapshots/99887766554433221100/config.json",
        )
        with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
            self.assertEqual(resolve_model_commit("org/model", model=m), "99887766554433221100")

    def test_extract_top_k_tokens_is_sorted_and_normalized(self) -> None:
        logits = np.array([1.0, 5.0, -2.0, 3.0, 4.0, 0.5], dtype=np.float32)
        toks = extract_top_k_tokens(logits, _DummyTokenizer(), k=3)
        self.assertEqual([t["token_id"] for t in toks], [1, 4, 3])
        self.assertEqual([t["token"] for t in toks], ["tok_1", "tok_4", "tok_3"])
        self.assertGreater(toks[0]["prob"], toks[1]["prob"])
        self.assertAlmostEqual(np.log(toks[0]["prob"]), toks[0]["log_prob"], places=6)

    def test_select_noop_check_indices_covers_every_suite(self) -> None:
        items = [
            {"id": "h1", "suite": "reversible_honest"},
            {"id": "h2", "suite": "reversible_honest"},
            {"id": "h3", "suite": "reversible_honest"},
            {"id": "d1", "suite": "reversible_distort"},
            {"id": "d2", "suite": "reversible_distort"},
            {"id": "b1", "suite": "benign_rule"},
        ]
        idxs = select_noop_check_indices(items, max_per_suite=2)
        self.assertEqual(idxs, [0, 1, 3, 4, 5])

    def test_surprisal_return_details_contains_all_40_worlds_and_items(self) -> None:
        rows = build_surprisal_confound_rows()
        meta_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ("cover_honest", "cover_lie", "benign_surprisal")}
        rng = np.random.default_rng(1)
        H_arm = {arm: rng.normal(size=(len(m), 6)) for arm, m in meta_arm.items()}
        pair_lo = {r["pair_key"]: (5.0 if r["arm"] == "cover_lie" else -5.0) for r in rows}
        res = analyze_layer_orthogonalisation(H_arm, meta_arm, pair_lo, np.ones(6), return_details=True)
        det = res["_details"]
        self.assertEqual(len(det["world_means"]["lie_dec"]), 40)
        self.assertEqual(len(det["item_scores"]["cover_lie"]["w_dec"]), 160)


if __name__ == "__main__":
    unittest.main()
