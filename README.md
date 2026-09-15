# Transaction Fraud Intelligence

**Does customer history improve fraud ranking within a fixed daily review budget?**

A reproducible synthetic fraud experiment comparing fixed rules, logistic regression with behavioural history, CatBoost on the current transaction, and CatBoost with behavioural history.

[![Open V2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aaravb015/transaction-fraud-intelligence/blob/main/notebooks/fraud_v2.ipynb)

**Status: experiment complete.** V2 development was frozen before the reserved period was materialized. The one-time reserved-period evaluation has now been completed and independently reviewed. Days 102–119 are no longer unseen and must not be used for further tuning.

## Final result

The preselected primary specification was `catboost_history`. At the predeclared headline operating point of **50 reviews per UTC day**, its mean reserved-period performance across the five frozen seeds was:

| Metric | Development | Final reserved period |
|---|---:|---:|
| Average precision | 0.617 | **0.582** |
| Precision | 0.346 | **0.224** |
| Fraud recall | 0.651 | **0.671** |
| Attempted-value capture | 0.598 | **0.552** |

The final four-model comparison at 50 reviews/day was:

| Model | Average precision | Precision | Fraud recall | Attempted-value capture |
|---|---:|---:|---:|---:|
| rules | 0.045 | 0.077 | 0.228 | 0.299 |
| catboost_current | 0.086 | 0.077 | 0.231 | 0.051 |
| logistic_history | 0.388 | 0.196 | 0.588 | 0.438 |
| **catboost_history** | **0.582** | **0.224** | **0.671** | **0.552** |

The reserved-period transaction-level fraud prevalence averaged **2.673%** across the five seeds; pooled across the five primary-model prediction files it was **1,509 / 56,538 = 2.669%**. Mean `catboost_history` precision of **22.44%** therefore represents roughly **8.4× enrichment over the underlying fraud rate**.

The primary conclusion survived the reserved-period test: **customer behavioural history materially improves fraud ranking at equal review capacity, and the gain is not explained by model complexity alone.** A history-aware logistic model also substantially outperformed current-transaction CatBoost.

No formal numeric pass threshold was declared in advance, so the final outcome is reported as supporting evidence for the frozen research question rather than as a production certification.

See [`docs/final_evaluation_report.md`](docs/final_evaluation_report.md) for the full final review, integrity evidence, prevalence/lift context, scenario results, artifact hashes, and limitations.

## Experimental design

The nominal synthetic horizon is 120 days. Development generated and used only days 0–101. Before final evaluation, days 102–119 were not materialized.

The frozen protocol used:

- 1,200 synthetic customers and 240 merchants
- five fixed seeds: `42`, `123`, `2025`, `31415`, `27182`
- a seven-day label delay
- four fixed comparators: `rules`, `logistic_history`, `catboost_current`, `catboost_history`
- daily review budgets of `20`, `50`, and `100`
- `50/day` as the predeclared headline operating point
- the same 21 behavioural-history features for the two history models
- no raw entity IDs, hidden simulator parameters, scenario tags, context tags, or labels as model features

Training, early stopping, development validation, primary-model selection, feature definitions, simulator behaviour, review budgets, and evaluation metrics were frozen before the reserved-period result was exposed.

## Final-evaluation integrity

The official final archive is:

`fraud_final_20260915T101601_058909Z.zip`

SHA-256:

`d82f31734a3a0255d2b59704217cb79829bc861a2bde7ca9cba20a59d4344b53`

The final output contract passed with the complete **5 seeds × 4 models × 3 budgets = 60** result grid and all five per-seed prediction sets.

For every seed, the official artifact records that:

- every development transaction exactly matched the frozen development prefix;
- every development feature exactly matched the frozen feature set;
- the preselected primary specification remained `catboost_history`;
- there was no model, feature, seed, simulator, threshold, or review-budget selection after final outcomes were exposed.

The repository now includes the official artifact manifest, prefix-integrity table, output-contract result, and ZIP checksum under [`artifacts/final_evaluation/`](artifacts/final_evaluation/). The headline metrics can be independently recomputed from the official prediction exports with [`scripts/verify_final_artifact.py`](scripts/verify_final_artifact.py), which does **not** rerun simulation, fitting, or scoring.

### Evaluator-source provenance

The official ZIP identifies its final evaluator as version **`0.3.0`**. The repository history contains an earlier pre-exposure `src/final_eval_runner.py` (`0.1.0`) but does **not** contain the exact `0.3.0` evaluator source that produced the official artifact.

That is a documented reproducibility limitation. The checked-in `0.1.0` runner must not be represented as the exact producer of the official ZIP, and the consumed final period must not be rerun merely to repair that provenance gap.

## What the model learned

The final scenario breakdown preserved the main development finding:

| Fraud scenario | Final `catboost_history` recall |
|---|---:|
| card testing | 0.906 |
| account takeover | 0.648 |
| low-and-slow | 0.353 |

Low-and-slow fraud remains the principal weakness.

For legitimate activity, travel and new-phone contexts remained somewhat more alert-prone than routine transactions, reinforcing that behavioural novelty can produce false positives as well as useful fraud signal.

## Why review capacity matters

Fraud teams have finite investigation capacity. A detector that appears better only because it sends more transactions for review does not demonstrate a useful operational improvement.

This project therefore compares models at identical daily top-K review budgets. The core operational metrics are precision, fraud recall, legitimate reviews, false-positive rate, and attempted fraudulent value captured. Average precision and ROC-AUC describe ranking quality.

Daily top-K is an **offline end-of-day ranking diagnostic**. `REVIEW` and `NO_REVIEW` are not live approval or decline decisions.

### Rules-baseline tie caveat

The rules comparator is coarse: roughly **95%** of its final scores are exactly zero. Equal scores are resolved by ascending `transaction_id`, which is chronological in the synthetic data. Across the five seeds, about **37%–40%** of the rules model's selected 50/day queue consisted of zero-score transactions.

Its exact precision should therefore be treated as a rough baseline rather than a finely ranked comparator. This limitation does not explain the much larger gap between the behavioural-history models and the current-transaction/rules baselines. See [`artifacts/final_evaluation/rules_tie_audit.csv`](artifacts/final_evaluation/rules_tie_audit.csv).

## Repository guide

| Path | Purpose |
|---|---|
| `notebooks/fraud_v2.ipynb` | Reproducible V2 development experiment |
| `notebooks/fraud_final_evaluation.ipynb` | Historical one-time evaluation notebook; the final period is consumed and must not be treated as unseen |
| `src/fraud_v2.py` | V2 simulation, point-in-time features, modelling, analysis and export engine |
| `src/final_eval_continuation.py` | Predeclared reserved-period continuation logic |
| `src/final_eval_runner.py` | Earlier pre-exposure runner (`0.1.0`); **not** the exact producer of the official `0.3.0` artifact |
| `scripts/verify_final_artifact.py` | Verifies the official ZIP and recomputes exported metrics without rerunning models |
| `artifacts/final_evaluation/` | Official manifest/integrity evidence plus small post-hoc prevalence and tie audits |
| `docs/v1_review_and_v2_plan.md` | V1 review and V2 source specification |
| `docs/v2_experiment_design.md` | Predeclared V2 experiment design |
| `docs/v2_freeze.md` | Development freeze and primary-model choice before final exposure |
| `docs/final_evaluation_protocol.md` | Predeclared final-evaluation protocol |
| `docs/final_evaluation_protocol_addendum.md` | Pre-exposure continuation-rule tightening |
| `docs/final_evaluation_fix_01.md` | Pre-exposure Python 3.13 loader compatibility fix |
| `docs/final_evaluation_report.md` | Final reserved-period results and independent review |

## Reproducibility and safeguards

Same-timestamp events are scored together before history updates. Failed outcomes enter history only once their availability time arrives. A transaction cannot enter its own historical baseline. Labels must mature before they are eligible for fitting or early-stopping use.

The final-evaluation design preserved the exact development-generation sequence before appending reserved-period draws, and the official artifact records exact development-row and development-feature equality for every seed.

That continuation has a known boundary limitation: fraud episodes that began in development but were truncated at day 102 are not resumed into the reserved period. The final period is therefore a **synthetic continuation experiment**, not a naturally observed stationary holdout.

The original development freeze records the frozen implementation commit and engine SHA. The final evidence directory records the official final archive checksum and manifest. The later development fingerprint archive named by the final manifest is not itself committed, so its relationship to the original reviewed development ZIP cannot be independently reconstructed from repository files alone; this is explicitly documented in the final report.

## Code checks

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
python scripts/build_notebook.py --check
python scripts/check_notebook.py --execute
python scripts/verify_final_artifact.py /path/to/fraud_final_20260915T101601_058909Z.zip
```

CI validates the development notebook, temporal safeguards, and static final-evaluation/evidence checks. The final-evaluation notebook is intentionally not executed in CI because doing so would repeatedly materialize and score the already-consumed reserved period.

## Limitations

This is a **synthetic research/portfolio experiment**, not evidence of production fraud-detection accuracy.

Scores are uncalibrated. Synthetic-world seed variation is not a real-population confidence interval. Attempted-value capture is **not** prevented loss, settled loss avoided, or financial savings. The project does not include a production API, live decisioning, monitoring, a public-data benchmark, or a deployment claim.

Additional documented limitations are the exact `0.3.0` evaluator-source provenance gap, deterministic tie-breaking in the coarse rules baseline, the synthetic continuation boundary discontinuity, and the absence of the later development fingerprint ZIP from the repository.

The final reserved period has been consumed. Any future V3 must be defined as a new experiment with a new development/evaluation protocol rather than tuning V2 against days 102–119.

MIT licence; see [LICENSE](LICENSE).
