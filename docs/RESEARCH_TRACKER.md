# Research Tracker & Scientific Context

This file anchors the theoretical motivation, experimental pipeline, open failure modes, and governance relevance of `deception-linear-probes` so engineering work never drifts from the core alignment problem.

---

## 1. Macro Context: The Four Unsolved Alignment Frontiers

1. **Mapping tangled weights into causal circuits:** Neural networks pack overlapping concepts into shared polysemantic directions. Sparse autoencoders (SAEs) and circuit tracing decompose small circuits, but full-model coverage at frontier scale remains incomplete.
2. **Detecting deception and alignment faking:** Behavioural evaluations only measure *what* a model outputs under testing, not whether internal objectives shift when oversight disappears (e.g., Anthropic/Redwood alignment-faking results reaching 78% after conflicting retraining).
3. **Unfaithful reasoning and reward hacking:** Chain-of-thought text can be post-hoc rationalisation, and RL reward hacking can generalise into broader misalignment.
4. **Scalable oversight on fuzzy tasks:** RLHF hits Goodhart's Law past human verification limits, and debate protocols struggle on fuzzy heuristic judgements without a single binary error.

### Field Problem Map

| Sub-problem | Best current approach | Unsolved gap |
|---|---|---|
| **Internal transparency** | Sparse autoencoders (SAEs), linear probes, circuit tracing | Full-model coverage at frontier scale; stateful long-horizon agents |
| **Honesty & hidden goals** | Activation probing (deception/sandbagging) and steering vectors | Proving deceptive circuits are not dormant during audits |
| **Supervising stronger models** | Weak-to-strong generalisation, amplified oversight, debate | Verifying fuzzy, heuristic-heavy judgements |
| **Behavioural control** | Causal ablation, activation clamping, polysemanticity penalties | Surgical edits without degrading general capability |

---

## 2. Core Method: Linear Probes

Linear probes test the **linear representation hypothesis**—whether high-level internal states (truthfulness, refusal, deceptive intent) sit along linear directions in the frozen residual stream $h_l$ at layer $l$:

$$p(y = 1 \mid h_l) = \sigma(w^\top h_l + b)$$

Three governing experimental rules:
- **Frozen base weights (blocked gradients):** Never backpropagate into the base LLM; read existing internal structure rather than training new features.
- **Depth & separability:** Early layers encode syntax/tokens; middle-to-late layers disentangle abstract intent.
- **Selectivity over raw accuracy:** Compare real-task probe accuracy against randomised/permuted label controls to separate genuine internal representation from memorisation or leakage.

---

## 3. Five-Step Experimental Pipeline & Live Status

- [x] **Step 1 — Contrastive pairs & neutral control set**
  - Commit `02e7581`: Repo scaffold, ML `.gitignore`, `pyproject.toml`, `README.md`.
  - Commit `b8f1bbf`: Raw `v1` (baseline) vs `v2` (symmetric) pilot datasets, `src/validate_dataset.py`, and `data/audit_report.md`.
  - Commit `4fc5109`: Sanitised `pilot_v1_clean.jsonl` & `pilot_v2_clean.jsonl`, 6-subtype `hard_negatives.jsonl` (18 records), and integrity unit tests.
  - *Remaining in Phase 1:* Scale symmetric contrastive dataset to 300-400 pairs (rotating 4-5 directive templates) + add 500-sample Tier 1 benign calibration set (Alpaca).
- [x] **Step 2 — Cache frozen residual stream activations across layers**
  - Commit `47107bc`: `src/cache_activations.py` (supports dual mean + last-token pooling, trailing 5-token masking) and `tests/test_cache_utils.py`.
- [x] **Step 3 — Fit regularised probes & check selectivity (`src/train_probes.py`)** (Commit `87a872d`)
  - Train L2 logistic regression and Mass-Mean probes across all layers using pair-grouped cross-validation (zero pair leakage) + permutation selectivity control.
- [ ] **Step 4 — Evaluate OOD recall under a strict 1% FPR budget (`src/evaluate.py`)**
  - Calibrate threshold on Tier 1 benign traffic (<= 1% FPR), stress-test Tier 2 hard negatives across all 6 subtypes, and compare `v1` (confounded) vs `v2` (symmetric) probes.
- [ ] **Step 5 — Causal steering & adversarial checks (`src/steer.py`)**
  - Add/subtract learned direction $w$ into the residual stream during generation to verify causality vs correlation, then test prompt/suffix evasion.

---

## 4. The Four Failure Modes This Project Targets

| Failure mode | What goes wrong in practice | How this repo tests it |
|---|---|---|
| **Topic–intent confounding** | Probes fire on honest confessions or high-stakes truths | 6-subtype `hard_negatives.jsonl` stress test + symmetric stakeholder matching (`v2`) |
| **Layer fragility** | OOD recall collapses within a 2-layer shift despite clean ID accuracy | Full-layer sweep metrics & plots across all 36 layers of `Qwen2.5-3B-Instruct` |
| **Obfuscated activations** | Adversarial prompts/optimisation hide deceptive intent in benign directions | Planned Phase 5 evasion & multi-layer pooling checks |
| **Distributional & policy drift** | Probes overfit to one system-prompt skeleton (`"Your mandate is to"`) | Template rotation when scaling + cross-category holdout evaluation |

---

## 5. Governance & Strategy Bridge (Ethics of AI + AGI Strategy)

Technical white-box monitoring sits directly at the intersection of AI ethics and AGI governance:
- **Transparency & accountability (Ethics of AI):** Audits whether safety claims rest on genuine internal intent detection or fragile surface shortcuts (prompt length, authority keywords).
- **Verification infrastructure (AGI Strategy):** Deployment governance and frontier safety frameworks depend on reliable low-FPR monitors; quantifying probe failure modes (confounding, layer fragility, false alarms on honest confessions) defines what oversight guarantees policy can—and cannot—safely assume today.