# Transaction Fraud Intelligence

A reproducible first experiment in transaction fraud detection: synthetic payment histories, point-in-time behavioural features, rules, logistic regression, and CatBoost.

[![Open development notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aaravb015/transaction-fraud-intelligence/blob/codex-fraud-v1/notebooks/fraud_v1.ipynb)

**Status:** initial modelling implementation. Performance must come from an executed run. The final chronological test period stays reserved during development. The dashboard is a later milestone.

## Start here

1. Open the development notebook using the button above.
2. Choose a fresh CPU runtime in Colab.
3. Select **Runtime → Run all**. The notebook installs its packages and creates the data itself.
4. Download the ZIP offered by the last cell. Save the notebook itself from Colab's File menu if you want its executed output.
5. Start with `comparison.csv`, `scenario_breakdown.csv`, and the selected model's `*_error_cases.csv`.

The development button targets the implementation branch so it works before a merge. Keep that branch while using this link, or change it to `main` after merging.

## The question

**Does customer history improve fraud ranking within a fixed daily review budget?**

The experiment compares:

| Detector | Inputs / role |
|---|---|
| Rules | Fixed, transparent conditions; 0–100 points |
| Logistic regression | Full historical feature set; simple ML baseline |
| CatBoost, current transaction | Amount, hour, country and category |
| CatBoost, with history | Adds customer, timing, novelty, failed-outcome and shared-device features |

All detectors use the same validation transactions and budgets of 20, 50 and 100 reviews per UTC day. The 50-review operating point supplies a provisional leader for case investigation.

No claimed improvement is written into the project in advance.

## Data and timing

The self-contained simulator generates fictional customers, merchants, devices and payment attempts. It includes travel, permanent phone changes, large legitimate purchases and legitimate bursts, plus account takeover, card testing and low-and-slow fraud.

Default settings: 1,200 customers, 120 days, 240 merchants, seed 42, synthetic INR amounts. The automated smoke run uses 360 customers and shorter model training.

- Historical features use events strictly before the decision time.
- Same-timestamp events are scored together before history updates.
- Failed outcomes become available after a simulated 2–120 seconds.
- Fraud confirmations and legitimate labels mature after an assumed seven days.
- Training and early-stopping labels must be available at their respective cutoffs.
- Simulator labels, hidden customer profiles and raw entity IDs do not enter the models.
- The final 15% of the timeline is not evaluated. Its labels exist inside the simulator; the lock is procedural.

See [the readable implementation blueprint](docs/blueprint.md) for field meanings and assumptions.

## What the results mean

The notebook reports average precision, ROC-AUC, precision, recall, false-positive rate, reviewed legitimate transactions, and labelled fraudulent **attempted value** captured.

Daily top-K is an offline end-of-day ranking diagnostic. REVIEW and NO_REVIEW describe this diagnostic, not live payment approval or blocking. Attempted amounts include failed transactions; captured value is not prevented loss or savings.

Scores are uncalibrated. Observed evidence and triggered rules are reported separately; neither is a causal explanation. Context tags are simulator annotations, not confirmed real-world attack diagnoses.

## Run outputs

Each execution creates a timestamped folder and ZIP under `outputs/`:

- `comparison.csv` and `daily_metrics.csv`
- `scenario_breakdown.csv` and `legitimate_context_breakdown.csv`
- Per-detector validation predictions and investigation cases
- `validation_comparison.png`
- Saved CatBoost models and the logistic preprocessing/model pipeline
- `development_transactions.csv.gz` (excludes the reserved test period)
- `feature_importance.csv`, `split_audit.csv`, and `manifest.json`

The manifest records configuration, dependency versions, dataset fingerprint, feature order, rules and checks. Saved models need the same history builder at inference. Do not treat a single transaction row as sufficient input to the history model.

## Repository contents

| Path | Purpose |
|---|---|
| `notebooks/fraud_v1.ipynb` | Complete executable experiment |
| `docs/blueprint.md` | Human-readable implementation specification |
| `requirements.txt` | Pinned experiment dependencies |
| `scripts/check_notebook.py` | Notebook validation and optional small-data execution |
| `.github/workflows/notebook-smoke.yml` | Automated check of the submitted notebook |

To run the automated check locally with Python 3.11:

```bash
python -m pip install -r requirements.txt nbformat==5.10.4 nbclient==0.10.2 ipykernel
python scripts/check_notebook.py --execute
```

This validates the notebook schema and Python syntax, then executes it on the smoke configuration. Execution also checks future invariance, identical timestamps, delayed outcomes and exclusion of the current amount from its baseline. GitHub Actions retains the executed notebook and generated outputs as a downloadable artifact.

## Development path

1. Execute the first full notebook and inspect mistakes.
2. Make one evidence-led feature or modelling change and compare it on validation.
3. Repeat the chosen experiment across seeds and temporal windows.
4. Freeze the approach and evaluate the reserved final period.
5. Extract reusable modules and build the investigation dashboard.
6. Add a separate public-data benchmark and document what it can establish.

Calibration, operational queue replay, approval/block policy, anomaly detection, SHAP and graph analysis are not implemented in this first notebook.

## Limits

This project demonstrates engineering and results under a simplified simulator. It does not establish real-world fraud performance, causality, regulatory suitability, or production readiness. In particular, the model can learn biases and mechanisms we programmed into the simulation.

## References

- [Fraud Detection Handbook: validation strategies](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html)
- [CatBoost: fitting a classifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit)
- [scikit-learn: OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [nbclient: executing notebooks](https://nbclient.readthedocs.io/en/latest/client.html)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

MIT licence; see [LICENSE](LICENSE).
