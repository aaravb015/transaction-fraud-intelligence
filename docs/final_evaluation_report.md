# Final evaluation report

## Status

The V2 fraud-ranking experiment completed its one-time reserved-period evaluation. The result is accepted as the final outcome of the frozen modelling specification and evaluation protocol; the reserved period is no longer unseen and must not be reused for tuning or model selection.

The independently reviewed final archive is:

- File: `fraud_final_20260915T101601_058909Z.zip`
- SHA-256: `d82f31734a3a0255d2b59704217cb79829bc861a2bde7ca9cba20a59d4344b53`
- Evaluated at: `2026-09-15T10:22:59.867480+00:00`
- Final evaluator version recorded by the artifact: `0.3.0`
- Reserved period: `2025-04-13 00:00:00+00:00` through `2025-05-01 00:00:00+00:00` exclusive
- Frozen implementation commit: `a4ab1de96e2d0bed7fa1f19836f9585f3f10344f`
- Frozen engine SHA-256: `6debf4f7501ee0f937bd53c9992f8ecc642c946039dd653aa284229078d2cb1e`
- Preselected primary model family/specification: `catboost_history`
- Headline operating point: `50` reviews per UTC day
- Seeds: `42`, `123`, `2025`, `31415`, `27182`

No model family, feature, seed, simulator, review-budget, or threshold selection was performed after final-period outcomes were exposed.

## Integrity review

The official ZIP itself contains the complete `5 seeds × 4 models × 3 budgets = 60` result grid and all five per-seed prediction sets. The artifact's own output-contract file reports `contract_passed = true`.

For every seed, both frozen-development safeguards passed:

- exact row-for-row development-prefix match: `true`
- exact development-feature match: `true`

The artifact manifest records:

- `prefix_integrity_passed_for_all_seeds = true`
- `locked_test_materialized = true`
- `locked_test_evaluated = true`
- `official_evaluation_count_for_this_run = 1`
- `selection_after_final_outcomes = false`
- `longitudinal_challenger_promoted = false`
- `synthetic_data_only = true`
- `attempted_value_is_not_prevented_loss = true`

To make those claims auditable from the repository, byte-for-byte copies of `final_manifest.json`, `prefix_integrity_checks.csv`, and `final_output_contract_check.json` from the official ZIP are committed under [`artifacts/final_evaluation/`](../artifacts/final_evaluation/), together with the official ZIP checksum.

The headline metrics were independently recomputed from the exported prediction CSVs and matched `final_comparison.csv` to floating-point precision. [`scripts/verify_final_artifact.py`](../scripts/verify_final_artifact.py) now performs that verification directly against the official ZIP without simulating, fitting, or scoring anything again.

## Final reserved-period results

Mean results across the five frozen seeds at the 50-review/day operating point:

| Model | Average precision | ROC-AUC | Precision | Fraud recall | Attempted-value capture | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| rules | 0.045 | 0.568 | 0.077 | 0.228 | 0.299 | 0.076 |
| catboost_current | 0.086 | 0.628 | 0.077 | 0.231 | 0.051 | 0.076 |
| logistic_history | 0.388 | 0.856 | 0.196 | 0.588 | 0.438 | 0.066 |
| **catboost_history** | **0.582** | **0.880** | **0.224** | **0.671** | **0.552** | **0.063** |

For the preselected primary model, final fraud recall at 50/day ranged from `0.624` to `0.715` across the five seeds.

### Fraud prevalence and lift

Average transaction-level fraud prevalence across the five reserved-period seeds was **2.673%**. Pooled across the five exported primary-model prediction files, there were **1,509 fraudulent transactions among 56,538 total transactions (2.669%)**.

At the 50/day operating point, mean `catboost_history` precision was **22.44%**, about **8.4×** the underlying transaction-level fraud rate. This prevalence/lift context is essential when interpreting both precision and average precision. The per-seed calculations are committed in [`artifacts/final_evaluation/prevalence_audit.csv`](../artifacts/final_evaluation/prevalence_audit.csv).

## Development versus final evidence

The frozen development means for `catboost_history` at 50/day were approximately:

- average precision: `0.617`
- precision: `0.346`
- fraud recall: `0.651`
- attempted-value capture: `0.598`

The final reserved-period means were:

- average precision: `0.582`
- precision: `0.224`
- fraud recall: `0.671`
- attempted-value capture: `0.552`

The primary conclusion survives the reserved-period test: behavioural-history models materially outperform the current-transaction model and rules at equal review capacity, and the preselected `catboost_history` specification remains the strongest of the four frozen comparators on the headline metrics.

No formal numeric pass threshold was predeclared. The final result is therefore reported as evidence supporting the frozen research conclusion, not as a threshold-based certification or production-performance claim.

## Scenario behaviour

Mean final-period `catboost_history` fraud recall by scenario:

| Scenario | Fraud recall | Attempted-value capture |
|---|---:|---:|
| card testing | 0.906 | 0.851 |
| account takeover | 0.648 | 0.681 |
| low-and-slow | 0.353 | 0.281 |

The main development weakness therefore persists: low-and-slow fraud is substantially harder than card testing and account takeover.

## Legitimate-context behaviour

Mean final-period `catboost_history` alert rates for legitimate transactions were approximately:

| Context | Alert rate |
|---|---:|
| routine | 0.063 |
| new phone | 0.085 |
| travel | 0.086 |
| large purchase | 0.050 |
| legitimate burst | 0.059 |

New-phone and travel contexts remain more alert-prone than routine activity on average, consistent with the earlier development diagnosis.

## Rules-baseline tie limitation

The rules comparator is deliberately coarse and produces many identical scores. Selection ties are resolved deterministically by ascending `transaction_id`, which is chronological in the synthetic data.

In the official reserved-period exports, roughly **95%** of rules scores are exactly zero and approximately **37%–40%** of the rules model's selected 50/day queue consists of zero-score transactions. The exact rules precision is therefore sensitive to the deterministic tie-break and should be interpreted as a coarse operational baseline, not a finely ranked comparator. This does not materially alter the much larger separation between the behavioural-history models and the current-transaction/rules baselines. The per-seed audit is committed in [`artifacts/final_evaluation/rules_tie_audit.csv`](../artifacts/final_evaluation/rules_tie_audit.csv).

A future experiment should predeclare a seeded-random or otherwise business-meaningful tie-break for equal rule scores.

## Development-fingerprint archive provenance

The original freeze document names `fraud_v2_20260914T212244_410358Z.zip` as the reviewed full-development results archive. The final artifact manifest instead records `fraud_v2_20260915T093200_432180Z.zip` with SHA-256 `94ce8bd6b6f0a2490ee1b31c90bbaf1abc90a5433730a0f55a68f20dc3a47b05` as the `development_fingerprint_archive` used by the final evaluator.

This later archive is treated as the reproducibility/fingerprint input to the final-evaluation path, not as a new modelling round. The frozen implementation commit and engine hash remained unchanged, no final-period outcomes were used for model development, and the final run recorded exact development-row and feature equality for every seed before accepting the reserved-period result.

The repository does not currently contain the later development fingerprint ZIP itself, so the relationship between those two development archives cannot be independently reconstructed from repository files alone. That is an audit-trail limitation and is now stated explicitly rather than inferred from prose.

## Evaluator-source provenance limitation

The official artifact identifies its evaluator as **version `0.3.0`**. The repository history currently contains the earlier pre-exposure `src/final_eval_runner.py` (`0.1.0`) and does **not** contain the exact `0.3.0` evaluator source that produced the official ZIP.

Accordingly, the checked-in `0.1.0` runner must not be presented as the exact producer of the official artifact. This is a reproducibility limitation, not a reason to rerun the consumed final period. The official ZIP, its checksum, its internal manifest/prefix evidence, and independent recomputation from its exported predictions remain the record of the completed evaluation.

Any future V3 must commit and tag the exact final evaluator source before its untouched evaluation is executed.

## Reserved-period continuation limitation

The addendum intentionally preserves the frozen development RNG sequence by completing all original development draws before generating new reserved-period draws. As a consequence, fraud episodes that began in development but were truncated at day 102 are **not resumed** into the reserved period.

That creates a boundary discontinuity relative to a naturally observed continuous stream. The final period is therefore best described as a **synthetic reserved-period continuation experiment**, not a stationary real-world holdout. Low-and-slow fraud still occurs in the final exports, but the boundary construction itself remains a methodological limitation.

## Limitations

This remains a synthetic research and portfolio experiment. It does not establish live fraud-detection performance, prevented loss, savings, or production readiness.

Scores are uncalibrated. Daily top-K review is an offline ranking diagnostic rather than live decisioning. Attempted-value capture includes attempted fraudulent transaction value and must not be described as settled loss avoided or money saved. Repeated seeds measure variation among synthetic worlds rather than a real-population confidence interval.

The project also has a documented evaluator-source provenance gap for the official `0.3.0` run and a synthetic continuation boundary discontinuity. Those limitations do not justify reopening or rerunning the consumed final period.

## Final decision

The V2 experiment is closed. The reserved period must not be treated as unseen again. Any future V3 work must be framed as a new experiment with a newly defined development and untouched evaluation protocol rather than tuning against days 102–119.
