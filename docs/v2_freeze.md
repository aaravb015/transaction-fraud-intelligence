# V2 development freeze

This document freezes the V2 development experiment **before any reserved final-test period is materialized, inspected, scored, or used for decisions**.

## Freeze point

- Repository: `aaravb015/transaction-fraud-intelligence`
- Development branch: `codex-fraud-v1`
- Frozen implementation commit: `a4ab1de96e2d0bed7fa1f19836f9585f3f10344f`
- V2 implementation version: `0.2.0`
- Primary notebook: `notebooks/fraud_v2.ipynb`
- Experiment engine SHA-256: `6debf4f7501ee0f937bd53c9992f8ecc642c946039dd653aa284229078d2cb1e`
- Reviewed full-development results ZIP: `fraud_v2_20260914T212244_410358Z.zip`
- Reviewed results ZIP SHA-256: `650d3fbbce83ae82f0d821e73f30174998a2956e347dab7bc74f5bb2c41ae58a`

The commit above is the experiment code freeze. This documentation commit does not change the V2 modelling implementation.

## Frozen research question

**Does customer history improve fraud ranking within a fixed daily review budget?**

The final test is not a new modelling round. It is a one-time evaluation of the procedure selected from development evidence.

## Frozen configuration

- Synthetic customers: `1200`
- Nominal timeline: `120` days
- Merchants: `240`
- Label delay: `7` days
- Fraud-customer fraction: `0.35`
- Fixed seeds: `[42, 123, 2025, 31415, 27182]`
- Review budgets: `[20, 50, 100]` per UTC day
- Primary operating point: `50` reviews per UTC day
- Maximum boosting iterations: `600`
- Model threads: `2`
- Scores remain uncalibrated
- Attempted-value capture is not interpreted as prevented loss, savings, or settled loss

## Frozen temporal protocol

- Fit cutoff: `2025-03-14 00:00:00+00:00`
- Development-validation start: `2025-03-28 00:00:00+00:00`
- Reserved final-test boundary: `2025-04-13 00:00:00+00:00`

At freeze time:

- `locked_test_evaluated = false`
- `locked_test_materialized = false`

All recorded development timing safeguards passed:

- future invariance
- same-timestamp batching
- delayed outcomes
- current amount excluded from its own baseline
- label maturity
- development boundary

The reserved period must remain untouched until the explicit final-evaluation implementation is created and reviewed.

## Frozen primary comparison

The four-model comparison is fixed as:

1. `rules`
2. `logistic_history`
3. `catboost_current`
4. `catboost_history`

The current-transaction CatBoost model remains restricted to:

- `log_amount`
- `hour_sin`
- `hour_cos`
- `category`
- `country`

The history models remain fixed to the 21-feature V2 feature list recorded in `manifest.json`, including transaction amount, timing, velocity, historical baselines, novelty, known failures, shared-device context, category, and country.

No raw entity IDs, simulator-only hidden parameters, outcome labels, scenario tags, or context tags may enter model features.

## Frozen primary model choice

The selected primary model for later final evaluation is:

**`catboost_history`**

This choice was made only from development evidence and before any final-period evaluation.

Across the five declared development seeds at the primary 50-review/day operating point, the development means were approximately:

| Model | Average precision | Precision | Fraud recall | Attempted-value capture |
|---|---:|---:|---:|---:|
| rules | 0.054 | 0.100 | 0.188 | 0.201 |
| catboost_current | 0.106 | 0.104 | 0.196 | 0.035 |
| logistic_history | 0.395 | 0.273 | 0.514 | 0.411 |
| **catboost_history** | **0.617** | **0.346** | **0.651** | **0.598** |

For `catboost_history`, development fraud recall at 50/day across the five seeds ranged from about `0.620` to `0.694` with mean `0.651` and standard deviation about `0.037`.

These are development results only. They are not final-test results and they are not real-world performance claims.

## Development interpretation frozen before final test

The development evidence supports the following interpretation:

- Customer behavioural history materially improves ranking at equal review capacity.
- A simpler history-aware logistic model substantially outperforms a transaction-only CatBoost model, indicating that behavioural context matters more than model complexity alone.
- Velocity/recency was the most important ablated feature family on the declared ablation seed.
- Circadian behaviour, historical baselines, and device-network context also contributed useful signal.
- Raw transaction amount was not a dominant source of the history model's advantage.
- Low-and-slow fraud remains the main scenario weakness.
- Legitimate new-phone and travel contexts remain important false-positive contexts.

These statements are frozen as development conclusions; they must not be rewritten after seeing the final test merely to make the project appear stronger.

## Longitudinal challenger decision

The optional longitudinal challenger remains **separate from the primary four-model comparison** and is **not promoted to the primary model**.

Across the five development seeds it produced only small overall recall gains, a modest low-and-slow improvement, and no consistent average-precision improvement. Therefore the additional complexity is not accepted into the frozen primary specification.

Its development role remains diagnostic/exploratory only.

## Simulator and robustness decisions

The hardened V2 simulator, its fixed rule weights, feature-generation logic, legitimate lookalikes, fraud scenarios, and seed list are frozen.

No simulator parameter may be altered after this point to improve the final-test result.

The following are specifically prohibited between freeze and final evaluation:

- changing fraud-generation probabilities or scenario mechanics
- changing legitimate lookalike behaviour
- adding/removing/redefining primary features
- changing model classes
- changing the primary seed list
- selecting a more favourable subset of seeds
- changing review budgets or the 50/day headline operating point
- changing rule weights
- hyperparameter searches based on final-period outcomes
- promoting the longitudinal challenger after seeing final-period results
- changing the selected primary model after seeing final-period results
- inspecting reserved-period labels, summaries, examples, distributions, scores, or metrics before the one-time evaluation

Engineering-only changes needed to make the frozen protocol executable may be made only if they do not change the data-generating process, feature definitions, model specification, fitting protocol, thresholds/capacity rules, seed list, or evaluation metrics. Any such change must be documented before execution.

## Final-evaluation protocol

The next experimental step is a **single reserved-period evaluation** using the frozen protocol.

When a final-evaluation path is implemented:

1. Materialize only the previously reserved period required for evaluation while preserving the frozen simulator mechanics and point-in-time history rules.
2. Use the same five declared seeds.
3. Preserve the same four-model comparison.
4. Preserve the 20/50/100 review budgets and 50/day primary operating point.
5. Preserve the frozen `catboost_history` primary-model choice.
6. Do not perform model, feature, simulator, threshold, or seed selection using final-period outcomes.
7. Export final-period predictions and metrics separately from development outputs.
8. Record that the final period has been evaluated exactly once under the frozen protocol.
9. Accept and report the result even if performance deteriorates or the development ranking does not hold.

If an implementation problem is found before final-period results are exposed, fix the problem transparently, document the exact reason and code change, and re-freeze before evaluation. If final-period outcomes have already been exposed, they must not be treated as unseen again.

## Merge policy

Do not merge the experimental branch into `main` merely because development results are strong.

Merge should happen only after:

- the frozen final evaluation is completed,
- the final outputs are independently reviewed,
- development and final-test claims are clearly separated,
- README language is updated to reflect the actual final evidence and limitations.

## Scope reminder

This remains a synthetic research/portfolio experiment. It does not establish production fraud-detection performance. Calibration, live decisioning, API deployment, public-data benchmarking, SHAP, graph modelling, and production monitoring remain outside this freeze unless started later as separate work after the final evaluation.
