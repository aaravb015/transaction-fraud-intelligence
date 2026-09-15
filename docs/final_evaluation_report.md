# Final evaluation report

## Status

The frozen V2 fraud-ranking experiment has completed its one-time reserved-period evaluation. The result is accepted as the final outcome of the frozen protocol; the reserved period is no longer unseen and must not be reused for tuning or model selection.

The independently reviewed final archive is:

- File: `fraud_final_20260915T101601_058909Z.zip`
- SHA-256: `d82f31734a3a0255d2b59704217cb79829bc861a2bde7ca9cba20a59d4344b53`
- Evaluated at: `2026-09-15T10:22:59.867480+00:00`
- Reserved period: `2025-04-13 00:00:00+00:00` through `2025-05-01 00:00:00+00:00` exclusive
- Frozen implementation commit: `a4ab1de96e2d0bed7fa1f19836f9585f3f10344f`
- Frozen engine SHA-256: `6debf4f7501ee0f937bd53c9992f8ecc642c946039dd653aa284229078d2cb1e`
- Preselected primary model: `catboost_history`
- Headline operating point: `50` reviews per UTC day
- Seeds: `42`, `123`, `2025`, `31415`, `27182`

No model, feature, seed, simulator, review-budget, or threshold selection was performed after final-period outcomes were exposed.

## Integrity review

The final output contract passed. The archive contains the complete `5 seeds × 4 models × 3 budgets = 60` result grid and all five per-seed prediction sets.

For every seed, both frozen-development safeguards passed:

- exact row-for-row development-prefix match: `true`
- exact development-feature match: `true`

The manifest records:

- `prefix_integrity_passed_for_all_seeds = true`
- `locked_test_materialized = true`
- `locked_test_evaluated = true`
- `official_evaluation_count_for_this_run = 1`
- `selection_after_final_outcomes = false`
- `longitudinal_challenger_promoted = false`
- `synthetic_data_only = true`
- `attempted_value_is_not_prevented_loss = true`

The headline metrics were independently recomputed from the exported prediction CSVs and matched `final_comparison.csv` to floating-point precision.

## Final reserved-period results

Mean results across the five frozen seeds at the 50-review/day operating point:

| Model | Average precision | ROC-AUC | Precision | Fraud recall | Attempted-value capture | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| rules | 0.045 | 0.568 | 0.077 | 0.228 | 0.299 | 0.076 |
| catboost_current | 0.086 | 0.628 | 0.077 | 0.231 | 0.051 | 0.076 |
| logistic_history | 0.388 | 0.856 | 0.196 | 0.588 | 0.438 | 0.066 |
| **catboost_history** | **0.582** | **0.880** | **0.224** | **0.671** | **0.552** | **0.063** |

For the preselected primary model, final fraud recall at 50/day ranged from `0.624` to `0.715` across the five seeds.

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

The primary conclusion survives the reserved-period test: behavioural-history models materially outperform the current-transaction model and rules at equal review capacity, and the preselected `catboost_history` model remains the strongest of the frozen four-model comparison on the headline metrics.

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

## Development-fingerprint archive provenance

The original freeze document names `fraud_v2_20260914T212244_410358Z.zip` as the reviewed full-development results archive. The final manifest instead records `fraud_v2_20260915T093200_432180Z.zip` with SHA-256 `94ce8bd6b6f0a2490ee1b31c90bbaf1abc90a5433730a0f55a68f20dc3a47b05` as the `development_fingerprint_archive` used by the final evaluator.

This later archive must be interpreted as the reproducibility/fingerprint input to the final-evaluation path, not as a new modelling round. The frozen implementation commit and engine hash remained unchanged, no final-period outcomes were used for model development, and the final run proved exact development-row and feature equality for every seed before accepting the reserved-period result.

## Limitations

This remains a synthetic research and portfolio experiment. It does not establish live fraud-detection performance, prevented loss, savings, or production readiness.

Scores are uncalibrated. Daily top-K review is an offline ranking diagnostic rather than live decisioning. Attempted-value capture includes attempted fraudulent transaction value and must not be described as settled loss avoided or money saved. Repeated seeds measure variation among synthetic worlds rather than a real-population confidence interval.

## Final decision

The V2 experiment is closed. The reserved period must not be treated as unseen again. Any future V3 work must be framed as a new experiment with a newly defined development and untouched evaluation protocol rather than tuning against days 102–119.
