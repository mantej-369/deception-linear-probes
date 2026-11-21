"""Step 3 & 4 core: Pair-grouped linear probing, Mass-Mean direction, selectivity, and checkpoint export."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evaluate import (
    compute_threshold_at_fpr,
    evaluate_probe_scores,
    positive_rate_at_threshold,
)


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


def run_layer_sweep(
    train_cache: Path,
    output_json: Path,
    pooling: str = "mean_activations",
    reg_c: float = 0.1,
    n_splits: int = 5,
    hard_neg_cache: Path | None = None,
    benign_cache: Path | None = None,
    ood_deceptive_cache: Path | None = None,
    save_probe: Path | None = None,
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

            clf = LogisticRegression(C=reg_c, max_iter=1000, random_state=seed)
            clf.fit(X_tr, y_tr)
            lr_oof_scores[val_idx] = clf.predict_proba(X_va)[:, 1]

            direction = X_tr[y_tr == 1].mean(axis=0) - X_tr[y_tr == 0].mean(axis=0)
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction = direction / norm
            mm_oof_scores[val_idx] = X_va @ direction

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

    # Fit final probe and steering directions at best_layer
    X_best = acts[:, best_layer, :]
    final_scaler = StandardScaler()
    X_best_scaled = final_scaler.fit_transform(X_best)
    final_clf = LogisticRegression(C=reg_c, max_iter=1000, random_state=seed)
    final_clf.fit(X_best_scaled, y)

    mm_std = X_best_scaled[y == 1].mean(axis=0) - X_best_scaled[y == 0].mean(axis=0)
    mm_std_norm = np.linalg.norm(mm_std)
    if mm_std_norm > 1e-12:
        mm_std = mm_std / mm_std_norm

    mm_raw = X_best[y == 1].mean(axis=0) - X_best[y == 0].mean(axis=0)
    mm_raw_norm = np.linalg.norm(mm_raw)
    if mm_raw_norm > 1e-12:
        mm_raw = mm_raw / mm_raw_norm

    safe_scale = np.where(final_scaler.scale_ == 0.0, 1.0, final_scaler.scale_)
    lr_raw = final_clf.coef_[0] / safe_scale
    lr_raw_norm = np.linalg.norm(lr_raw)
    if lr_raw_norm > 1e-12:
        lr_raw = lr_raw / lr_raw_norm

    def _score_split(cache_path: Path | None) -> tuple[list[float] | None, list[str] | None]:
        if cache_path is None or not cache_path.exists():
            return None, None
        pl = torch.load(cache_path, map_location="cpu", weights_only=False)
        X_sp = final_scaler.transform(pl[pooling].float().numpy()[:, best_layer, :])
        scores = final_clf.predict_proba(X_sp)[:, 1].tolist()
        return scores, list(pl["categories"])

    benign_scores, _ = _score_split(benign_cache)
    hn_scores, hn_cats = _score_split(hard_neg_cache)
    ood_scores, ood_cats = _score_split(ood_deceptive_cache)

    eval_results = evaluate_probe_scores(
        benign_scores=benign_scores,
        hard_neg_scores=hn_scores,
        hard_neg_categories=hn_cats,
        ood_deceptive_scores=ood_scores,
        ood_deceptive_categories=ood_cats,
        target_fpr=target_fpr,
    )
    summary.update(eval_results)

    if save_probe is not None:
        save_probe.parent.mkdir(parents=True, exist_ok=True)
        calibrated_tau = eval_results.get("benign_eval", {}).get("threshold", 0.5)
        checkpoint = {
            "model_name": payload["model_name"],
            "source_file": payload["source_file"],
            "pooling": pooling,
            "layer": int(best_layer),
            "reg_c": float(reg_c),
            "calibrated_threshold": float(calibrated_tau),
            "scaler_mean": torch.tensor(final_scaler.mean_, dtype=torch.float32),
            "scaler_scale": torch.tensor(final_scaler.scale_, dtype=torch.float32),
            "lr_coef": torch.tensor(final_clf.coef_[0], dtype=torch.float32),
            "lr_intercept": float(final_clf.intercept_[0]),
            "lr_raw_unit_direction": torch.tensor(lr_raw, dtype=torch.float32),
            "mass_mean_std_direction": torch.tensor(mm_std, dtype=torch.float32),
            "mass_mean_raw_unit_direction": torch.tensor(mm_raw, dtype=torch.float32),
        }
        torch.save(checkpoint, save_probe)
        summary["saved_probe_path"] = str(save_probe)
        print(f"Saved best-layer probe & steering vectors to {save_probe}")

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
    parser.add_argument("--ood-deceptive-cache", type=Path, default=None)
    parser.add_argument("--save-probe", type=Path, default=None)
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
        ood_deceptive_cache=args.ood_deceptive_cache,
        save_probe=args.save_probe,
        target_fpr=args.target_fpr,
    )


if __name__ == "__main__":
    main()
