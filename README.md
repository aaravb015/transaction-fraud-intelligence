# Transaction Fraud Intelligence

**Does customer history improve fraud ranking within a fixed daily review budget?**

A reproducible synthetic fraud experiment comparing fixed rules, logistic regression with history, CatBoost on the current transaction, and CatBoost with history.

[![Open V2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aaravb015/transaction-fraud-intelligence/blob/codex-fraud-v1/notebooks/fraud_v2.ipynb)

**Status:** V2 development implementation. The full V2 experiment is awaiting the user's run and independent review. **The final test remains locked. Nothing is merged into `main`.**

## Run V2

1. Open the notebook using the Colab button.
2. Use a fresh CPU runtime and select **Runtime → Run all**.
3. Let all five declared seeds and the six ablations finish.
4. Download the timestamped `fraud_v2_…zip` offered by the final cell.
5. Return that ZIP for independent review before choosing the next experiment.

No dataset, account credentials, repository clone, package edits or kernel restart are required for the supported Python 3.11–3.13 runtime range. The notebook includes its own engine, installs wheel-based dependencies into a separate directory, and runs the scientific code in fresh subprocesses. Colab's preloaded NumPy/Pandas modules remain unchanged. Python and package versions are printed before the experiment starts and recorded in the manifest.

The button points to the development branch so it works before a merge. The original [V1 notebook](notebooks/fraud_v1.ipynb) is retained for source comparison; its historical installation cell is not the V2 launch path.

## What changed in V2

| Area | Implemented behaviour |
|---|---|
| Environment | Isolated, self-contained execution; CI matrix for Python 3.11, 3.12 and 3.13 |
| Locked test | Reserved-period events are never materialized; feature, scoring and export boundaries reject them |
| Simulator | More ordinary-looking fraud, legitimate lookalikes retained, and a paired untrained V1-like separability reference |
| Primary comparison | The original four comparators, 21 history features, current-transaction fields, and 20/50/100 daily budgets are preserved |
| Low-and-slow | Caught/missed/routine diagnosis, 7/14/30-day longitudinal measures and a conditional separate challenger |
| Robustness | Every fixed seed: **42, 123, 2025, 31415, 27182**; per-seed results and mean/std/min/max |
| Ablation | Six feature families removed one at a time on the predeclared primary seed |
| False positives | Context review rates, median/Q75 scores, recurring evidence and representative examples |
| Daily operations | Day-level variability, exhausted budgets, underfilled days and unused capacity |
| Outputs | Complete V2 ZIP with metrics, predictions, cases, charts, models, configuration and timing checks |

See the [source specification](docs/v1_review_and_v2_plan.md), [predeclared experiment design](docs/v2_experiment_design.md), and [implementation details and acceptance mapping](docs/v2_implementation.md).

## Why review capacity matters

Investigators have limited time. A model that surfaces more fraud by sending every transaction for review does not establish a useful operational improvement.

At the same daily review budget, the experiment measures precision, fraud recall, legitimate reviews, false-positive rate, and **attempted fraudulent value** captured. Average precision and ROC-AUC describe ranking quality. Scores remain **uncalibrated**.

Daily top-K is an **offline end-of-day ranking diagnostic**. REVIEW and NO_REVIEW are diagnostic labels, not live approval or blocking. Amounts include failed attempts; value captured is not settled loss, prevented loss or savings.

## Preserve the V1 story, test it more carefully

The reviewed full V1 run reported average precision of approximately 0.8956 for CatBoost with history, 0.8487 for logistic regression with history, and 0.1405 for current-transaction CatBoost. Those are **historical synthetic V1 results**, documented in the [V1 review](docs/v1_review_and_v2_plan.md), not V2 results or real-world performance.

The V2 question is whether the value of behavioural context remains credible under a harder simulation and across all declared seeds. No requirement says V2 scores must exceed V1 scores. The simulator changed, so an across-version score difference is not a pure model improvement.

The optional longitudinal challenger stays separate from the four-model comparison. Its features are nominated from the primary seed's development diagnosis by a fixed gate. Seed-42 gains are exploratory; the same frozen candidate list is then evaluated on the other four synthetic worlds. If no candidate qualifies, the ZIP records that outcome without inventing an improvement.

## Timeline and safeguards

The nominal horizon remains 120 days. V2 only generates development days 0–101:

- Training labels must be available at the day-72 fitting cutoff.
- Early-stopping labels must be available by day 86.
- Development validation covers days 86–101.
- Day 102 onward remains reserved, ungenerated and unexamined.

Same-timestamp events are scored together before history updates. Failed outcomes only enter history once their availability time arrives. Current attempts cannot enter their own baselines. New longitudinal windows compare recent activity with a disjoint earlier baseline, with history-coverage checks.

Labels, scenario/context tags, hidden customer-generation parameters and raw entity IDs are excluded from model features. Every seed records timing checks and a development-data fingerprint. Both `locked_test_evaluated` and `locked_test_materialized` remain `false`.

## Results ZIP

The timestamped V2 ZIP includes:

- `comparison.csv`, `daily_metrics.csv`, `daily_variability_summary.csv`
- `scenario_breakdown.csv`, `legitimate_context_breakdown.csv`, `false_positive_examples.csv`
- `feature_importance.csv`, `feature_separation_audit.csv`, `simulator_overlap_audit.csv`
- `low_and_slow_diagnostics.csv`, `low_and_slow_overview.csv`, `longitudinal_feature_decisions.csv`
- `longitudinal_tradeoffs.csv`, `longitudinal_context_tradeoffs.csv`
- `robustness_summary.csv`, `ablation_summary.csv`
- Each primary detector's validation predictions and investigation cases
- Separate per-seed prediction/case exports under `seeds/`
- Three concise validation charts
- Saved primary models and the conditional challenger when applicable
- Development transactions, `split_audit.csv`, `manifest.json`, `resolved_packages.txt`, `output_contract_check.json`

The export is checked for required files, the complete five-seed/model/budget grid and development-only prediction timestamps before the ZIP is offered.

## Code and checks

| Path | Purpose |
|---|---|
| `notebooks/fraud_v2.ipynb` | Self-contained top-to-bottom Colab experiment |
| `src/fraud_v2.py` | Readable simulation, history, modelling, analysis and export engine |
| `scripts/notebook_bootstrap.py` | Isolated dependency setup and notebook presentation |
| `scripts/build_notebook.py` | Generates the notebook from the reviewed source; `--check` detects divergence |
| `scripts/check_notebook.py` | Schema/syntax checks, actual notebook smoke execution, host-module isolation and ZIP checks |
| `tests/test_v2_safeguards.py` | Temporal, maturity, capacity and V1-comparison regression checks |
| `requirements.txt` / `requirements-dev.txt` | Experiment and verification dependencies |

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
python scripts/build_notebook.py --check
python scripts/check_notebook.py --execute
```

CI runs the real Jupyter notebook on Python 3.11, 3.12 and 3.13. All five seeds are included in smoke mode. A local environment that cannot start a Jupyter socket can explicitly use `--execute --plain` to execute the same cells; that does not replace the CI kernel check.

After changing the engine, frontend or dependencies, run `python scripts/build_notebook.py` and commit the resulting notebook. Generated data, checkpoints and packages remain outside source control.

## Limits and next decision

This is a simplified synthetic benchmark. Its fraud/legitimate overlap can be measured, but it cannot establish real-world accuracy. The paired V1-like reference is not an exact reconstruction of the archived V1 run. Evidence strings are descriptive, context tags are exclusive, and labels mature after an assumed seven days.

Repeated seeds measure variation among synthetic worlds, not a real-population confidence interval. Ablations use one declared seed. The conditional challenger includes validation-based feature selection and must be interpreted accordingly.

No production API, deployed dashboard, public-data benchmark, SHAP, anomaly ensemble, financial savings estimate or final-test evaluation is included. The next decision follows independent review of the user's full V2 ZIP.

MIT licence; see [LICENSE](LICENSE).
