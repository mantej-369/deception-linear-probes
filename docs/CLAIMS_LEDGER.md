# Claims Ledger & Provenance Audit Status

## 1. Retired Claims (Frozen as Exploration / Failure Cases)

1. **L22 matched-window OOD separation (`0.9381` AUROC, tag `dev-v1-v4`):** Retired as evidence of deception detection. Inside the 45–54 assistant-token window (30 OOD vs. 35 HN), zero-parameter system-prompt token length achieves `1.0000` AUROC (HN 18–34 vs. OOD 37–45), and absolute readout position achieves `0.9843` AUROC.
2. **Replicated deception-specific causal steering / double dissociation (Phase 2):** Retired.
   - On `Llama-3.2-3B-Instruct` at L19 (`results/cross_family_llama32_3b/steer_v1/steer_L19_decision_only_summary.json`), the unnormalized sum/difference construction yields substantial axis overlap: `cos(dec_raw, cmp_raw) = -0.490`.
   - At `alpha = ±2`, localized decision-site `dec` steering shifts lie log-odds negatively (`-0.070` under `distort`, `-0.041` under `honest`)—opposite in sign to `Qwen2.5-3B-Instruct` L24 (`+0.316` / `+0.459`)—while inducing a larger shift on benign rule compliance (`-0.171`).
   - Phase 2 is reported strictly as small, non-specific token-choice perturbations with no replicated deception-specific causal dissociation.

## 2. Active Verified Findings (Committed Full-Precision Artefacts)

1. **Zero-nuisance reversible-world evaluation (Phase 1):** On 40 unseen fixed-string worlds (1,192 rows), all 8 scalar word/BPE/position baselines and all 3 TF-IDF baselines equal `0.5000` on both `Qwen2.5-3B-Instruct` (`4ede7c9`) and `Llama-3.2-3B-Instruct` (`ab6d85b`). Retrospective `answer_tok` readouts retain predictive signal above raw probability on ordinary reversible reports (improving AUROC across all 8 held-out families on Llama).
2. **Cross-family expectation-inverted reversal (Phase 3):** Across `80/80` balanced verified on-policy flips (`40 confirmed_first`, `40 alternative_first`), `w_dec` flips from positive on off-policy forced lies (`cover_honest`: Qwen L24 RAW `+3.713`, `40/40`; Llama L23 RAW `+0.645`, `40/40`) to strictly negative on on-policy lies (`cover_lie`: Qwen L24 RAW `-2.759`, `0/40`; Llama L10–L27 all `0/40`, $p = 1.82 \times 10^{-12}$). Removing a single formatting-surprisal axis (`w_perp`) zeroes formatting surprisal but does not invert the sign back to positive.
3. **Provenance audit notes on prior Phase 2/3 commits (`4ede7c9`, `4c9ae06`, `ab6d85b`):**
   - `model_commit` logged `"unknown"` under `transformers 5.22.1`; fixed via `src/provenance.py::resolve_model_commit()`.
   - Phase 2 no-op verification checked `prepared[0]` while printing `len(prepared)`; fixed to verify across all suites and log `checked_n` vs. `evaluated_n` explicitly.
   - Phase 3 now persists full-precision `per_world_scores.jsonl`, `per_item_scores.jsonl`, candidate token IDs, and full-vocabulary top-5 next tokens.

## 3. Preregistered Phase 4 Design (Frozen Before Inference)

- **Frozen primary readouts:** `Qwen2.5-3B-Instruct` **L24** and `Llama-3.2-3B-Instruct` **L19** (`w_dec_raw`, `w_dec_std`, `w_perp`), fitted solely on `v3_symmetric_120` + `v4_anti_60`.
- **Part A (Fixed-Directive Expectation Disentanglement):** Cross verified fact $\times$ directive (`honest`/`distort`) $\times$ non-authoritative prior/draft (`prior_A`/`prior_B`) $\times$ within-world mention order on brand-new unseen domain families and templates, verifying logit shifts and reporting all items (including failed manipulations).
- **Part B (On-Policy Free-Generation Monitoring):** Independent on-policy benign calibration (`n=200`, discrete order-statistic threshold) + private-fact free-generation suite (`n=80`) graded separately for factual accuracy, refusal, explicit disclosure, and ambiguity.
