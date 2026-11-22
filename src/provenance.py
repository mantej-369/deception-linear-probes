"""Shared provenance, top-k vocabulary inspection, and multi-suite no-op verification helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def resolve_model_commit(
    model_id: str,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> str:
    """Resolve the exact HF model revision SHA via loaded config, Hub API, or local cache snapshot."""
    for obj in (getattr(model, "config", None), tokenizer):
        sha = getattr(obj, "_commit_hash", None)
        if isinstance(sha, str) and len(sha.strip()) >= 7 and sha.strip().lower() != "unknown":
            return sha.strip()

    try:
        from huggingface_hub import model_info

        sha = getattr(model_info(model_id), "sha", None)
        if isinstance(sha, str) and len(sha.strip()) >= 7:
            return sha.strip()
    except Exception:
        pass

    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_id, "config.json")
        if isinstance(cached, str):
            parts = Path(cached).parts
            if "snapshots" in parts:
                idx = parts.index("snapshots")
                if idx + 1 < len(parts) and len(parts[idx + 1]) >= 7:
                    return parts[idx + 1]
    except Exception:
        pass

    return "unresolved"


def extract_top_k_tokens(logits_1d: Any, tokenizer: Any | None = None, k: int = 5) -> list[dict[str, Any]]:
    """Return full-vocabulary top-k next tokens with token ID, decoded piece, probability, and log-prob."""
    arr = (
        logits_1d.detach().float().cpu().numpy()
        if hasattr(logits_1d, "detach")
        else np.asarray(logits_1d, dtype=np.float64)
    )
    arr = np.asarray(arr, dtype=np.float64)
    max_v = float(np.max(arr))
    shifted = arr - max_v
    lse = max_v + float(np.log(np.sum(np.exp(shifted))))
    log_probs = arr - lse

    k_eff = min(int(k), int(arr.shape[0]))
    top_idx = np.argpartition(arr, -k_eff)[-k_eff:]
    top_idx = top_idx[np.argsort(arr[top_idx])[::-1]]

    out: list[dict[str, Any]] = []
    for tid in top_idx:
        tid_i = int(tid)
        lp = float(log_probs[tid_i])
        tok_str = tokenizer.decode([tid_i]) if tokenizer is not None else str(tid_i)
        out.append(
            {
                "token_id": tid_i,
                "token": tok_str,
                "prob": float(np.exp(lp)),
                "log_prob": lp,
            }
        )
    return out


def select_noop_check_indices(
    items: list[dict[str, Any]],
    suite_key: str = "suite",
    max_per_suite: int = 2,
) -> list[int]:
    """Select deterministic indices across every suite for multi-item no-op hook verification."""
    counts: dict[str, int] = {}
    selected: list[int] = []
    for idx, item in enumerate(items):
        suite = str(item.get(suite_key, "default"))
        c = counts.get(suite, 0)
        if c < max_per_suite:
            selected.append(idx)
            counts[suite] = c + 1
    return selected
