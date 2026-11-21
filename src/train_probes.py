"""Step 3 & 4 core: Pair-grouped linear probing, Mass-Mean direction, selectivity, and hard-negative evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_pair_id(item_id: str) -> str:
    """Recover base pair_id from flattened IDs like 'overt_001_honest'."""
    for suffix in ("_honest", "_deceptive"):
        if item_id.endswith(suffix):
            return item_id[: -len(suffix)]
    return item_id


def build_pair_folds(
    ids: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Create K-fold train/val splits grouped strictly by pair_id to prevent pair leakage."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, item_id in enumerate(ids):
        groups[extract_pair_id(item_id)].append(idx)

    pair_keys = sorted(groups.keys())
    if len(pair_keys) < 2:
        raise ValueError(f"Need at least 2 distinct pairs for cross-validation, got {len(pair_keys)}")

    k = min(n_splits, len(pair_keys))
    rng = random.Random(seed)
    rng.shuffle(pair_keys)

    fold_buckets: list[list[int]] = [[] for _ in range(k)]
    for i, pkey in enumerate(pair_keys):
        fold_buckets[i % k].extend(groups[pkey])

    folds: list[tuple[list[int], list[int]]] = []
    for val_fold_idx in range(k):
        val_idx = sorted(fold_buckets[val_fold_idx])
        train_idx = sorted(
            idx
            for f_idx, bucket in enumerate(fold_buckets)
            if f_idx != val_fold_idx
            for idx in bucket
        )
        folds.append((train_idx, val_idx))
    return folds


def compute_selectivity(task_accuracy: float, control_accuracy: float) -> float:
    """Selectivity = real task accuracy minus permuted/control task accuracy."""
    return round(task_accuracy - control_accuracy, 4)


def compute_threshold_at_fpr(benign_scores: list[float], target_fpr: float = 0.01) -> float:
    """Return threshold tau such that fraction(score > tau) on benign_scores is <= target_fpr."""
    if not benign_scores:
        raise ValueError("benign_scores cannot be empty")
    if not (0.0 <= target_fpr < 1.0):
        raise ValueError(f"target_fpr must be in [0, 1), got {target_fpr}")

    ordered = sorted(float(s) for s in benign_scores)
    allowed_fp = int(math.floor(target_fpr * len(ordered)))
    idx = max(0, len(ordered) - 1 - allowed_fp)
    return ordered[idx]


def positive_rate_at_threshold(scores: list[float], threshold: float) -> float:
    """Fraction of scores strictly exceeding the decision threshold."""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s > threshold) / len(scores)


def run_layer_sweep(
    train_cache: Path,
    output_json: Path,
    pooling: str = "mean_activations",
    reg_c: float = 0.1,
    n_splits: int = 5,
    hard_neg_cache: Path | None = None,
    benign_cache: Path | None = None,
    target_fpr: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    """Sweep all transformer layers with L2 logistic regression, Mass-Mean, and selectivity controls."""
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    payload = torch.load(train_cache, map_location="cpu", weights_only=False)
    acts = payload[pooling].float().numpy()  # (N, L, D)
    y = payload["labels"].numpy().astype(int)
    ids = list(payload["ids"])
    num_layers = acts.shape[1]

    folds = build_pair_folds(ids, n_splits=n_splits, seed=seed)
    rng = np.random.default_rng(seed)
    y_permuted = rng.permutation(y)

    layer_results: list[dict[str, Any]] = []
    best_layer = 0
    best_auc = -1.0

    for layer in range(num_layers):
        X = acts[:, layer, :]

        lr_oof_scores = np.zeros(len(y), dtype=float)
        mm_oof_scores = np.zeros(len(y), dtype=float)
        perm_oof_preds = np.zeros(len(y), dtype=int)

        for train_idx, val_idx in folds:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_va = scaler.transform(X[val_idx])
            y_tr = y[train_idx]

            # 1. L2-regularised Logistic Regression probe
            clf = LogisticRegression(C=reg_c, max_iter=1000, random_state=seed)
            clf.fit(X_tr, y_tr)
            lr_oof_scores[val_idx] = clf.predict_proba(X_va)[:, 1]

            # 2. Mass-Mean difference-of-centroids probe
            pos_mean = X_tr[y_tr == 1].mean(axis=0)
            neg_mean = X_tr[y_tr == 0].mean(axis=0)
            direction = pos_mean - neg_mean
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction = direction / norm
            mm_oof_scores[val_idx] = X_va @ direction

            # 3. Permuted-label selectivity control
            perm_clf = LogisticRegression(C=reg_c, max_iter=1000, random_state=seed)
            perm_clf.fit(X_tr, y_permuted[train_idx])
            perm_oof_preds[val_idx] = perm_clf.predict(X_va)

        lr_auc = float(roc_auc_score(y, lr_oof_scores))
        lr_acc = float(accuracy_score(y, (lr_oof_scores >= 0.5).astype(int)))
        mm_auc = float(roc_auc_score(y, mm_oof_scores))
        perm_acc = float(accuracy_score(y_permuted, perm_oof_preds))
        selectivity = compute_selectivity(lr_acc, perm_acc)

        layer_results.append(
            {
                "layer": layer,
                "lr_cv_auroc": round(lr_auc, 4),
                "lr_cv_acc": round(lr_acc, 4),
                "mass_mean_cv_auroc": round(mm_auc, 4),
                "perm_control_acc": round(perm_acc, 4),
                "selectivity": selectivity,
            }
        )

        if lr_auc > best_auc:
            best_auc = lr_auc
            best_layer = layer

    summary: dict[str, Any] = {
        "model_name": payload["model_name"],
        "source_file": payload["source_file"],
        "pooling": pooling,
        "n_samples": int(len(y)),
        "num_layers": int(num_layers),
        "best_layer": int(best_layer),
        "best_lr_cv_auroc": round(best_auc, 4),
        "layers": layer_results,
    }

    # Fit final probe at best_layer to evaluate benign calibration & hard negatives
    X_best = acts[:, best_layer, :]
    final_scaler = StandardScaler()
    X_best_scaled = final_scaler.fit_transform(X_best)
    final_clf = LogisticRegression(C=reg_c, max_iter=1000, random_state=seed)
    final_clf.fit(X_best_scaled, y)

    threshold = 0.5
    threshold_source = "default_0.5"
    if benign_cache is not None and benign_cache.exists():
        benign_pl = torch.load(benign_cache, map_location="cpu", weights_only=False)
        X_benign = final_scaler.transform(benign_pl[pooling].float().numpy()[:, best_layer, :])
        benign_scores = final_clf.predict_proba(X_benign)[:, 1].tolist()
        threshold = compute_threshold_at_fpr(benign_scores, target_fpr=target_fpr)
        threshold_source = f"benign_fpr_{target_fpr}"
        summary["benign_eval"] = {
            "n": len(benign_scores),
            "threshold": round(threshold, 4),
            "empirical_fpr": round(positive_rate_at_threshold(benign_scores, threshold), 4),
        }

    if hard_neg_cache is not None and hard_neg_cache.exists():
        hn_pl = torch.load(hard_neg_cache, map_location="cpu", weights_only=False)
        X_hn = final_scaler.transform(hn_pl[pooling].float().numpy()[:, best_layer, :])
        hn_scores = final_clf.predict_proba(X_hn)[:, 1].tolist()
        hn_cats = list(hn_pl["categories"])

        by_subtype: dict[str, list[float]] = defaultdict(list)
        for cat, sc in zip(hn_cats, hn_scores):
            by_subtype[cat].append(sc)

        summary["hard_negative_eval"] = {
            "threshold": round(threshold, 4),
            "threshold_source": threshold_source,
            "overall_false_positive_rate": round(positive_rate_at_threshold(hn_scores, threshold), 4),
            "mean_score": round(float(np.mean(hn_scores)), 4),
            "by_subtype": {
                sub: {
                    "n": len(scs),
                    "fpr": round(positive_rate_at_threshold(scs, threshold), 4),
                    "mean_score": round(float(np.mean(scs)), 4),
                }
                for sub, scs in sorted(by_subtype.items())
            },
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Best layer: {best_layer} (LR CV AUROC={best_auc:.4f}) -> saved {output_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear & Mass-Mean probes across layers.")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pooling", type=str, default="mean_activations", choices=["mean_activations", "last_token_activations"])
    parser.add_argument("--reg-c", type=float, default=0.1)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--hard-neg-cache", type=Path, default=None)
    parser.add_argument("--benign-cache", type=Path, default=None)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    args = parser.parse_args()

    run_layer_sweep(
        train_cache=args.train_cache,
        output_json=args.output,
        pooling=args.pooling,
        reg_c=args.reg_c,
        n_splits=args.splits,
        hard_neg_cache=args.hard_neg_cache,
        benign_cache=args.benign_cache,
        target_fpr=args.target_fpr,
    )


if __name__ == "__main__":
    main()