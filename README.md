# Deception Linear Probes: Confound Auditing & OOD Generalisation

An empirical mechanistic interpretability project investigating how synthetic dataset confounds affect linear probes trained to detect deceptive intent in open-weight LLMs.

## Research Objective
Linear probes trained on contrastive honest-vs-deceptive pairs can achieve high in-distribution AUROC while silently relying on surface shortcuts (prompt length asymmetry, stakeholder imbalance, or topic-intent confounding). This repository implements an end-to-end, confound-audited probing pipeline:

1. **Symmetric Contrastive Dataset & Audit (`prompts/`, `data/`, `src/validate_dataset.py`)** — Controlled workplace deception pairs (`overt_lie`, `concealment`, `pressure_dishonesty`) with automated confound checks, benign calibration traffic, and 6-subtype hard negatives.
2. **Residual Stream Caching (`src/cache_activations.py`)** — Frozen-weight hidden-state extraction with trailing-token masking.
3. **Probe Training & Selectivity (`src/train_probes.py`)** — L2-regularised logistic regression, Mass-Mean probes, and shuffled-label selectivity controls across layers.
4. **Calibrated Evaluation (`src/evaluate.py`)** — Out-of-distribution recall evaluated at a fixed 1% False Positive Rate (FPR) plus hard-negative stress testing.
5. **Causal Intervention (`src/steer.py`)** — Residual-stream activation steering to separate causal circuits from correlational features.

## Project Status
- [x] Phase 1a: Pilot contrastive data generation (`v1` vs `v2`) and manual confound audit
- [ ] Phase 1b: Automated dataset validator (`validate_dataset.py`), scaled symmetric dataset, and Tier 1/2 controls
- [ ] Phase 2: Residual stream caching on open-weight models (Google Colab / Kaggle GPU)
- [ ] Phase 3-5: `v1` vs `v2` ablation study, 1% FPR evaluation, and causal steering

## Repository Structure
- `prompts/` — Versioned generation prompts (`v1` baseline vs `v2` symmetric)
- `data/raw/` — Unedited synthetic outputs preserved for audit reproducibility
- `data/processed/` — Validated JSONL splits (`contrastive_train`, `control_calibration`, `hard_negatives`)
- `src/` — Core Python modules for auditing, caching, training, evaluation, and steering
- `notebooks/` — Reproducible GPU notebooks
- `results/` — Metrics summaries and layer-sweep visualisations