# V1 Review and V2 Implementation Plan

## Purpose

This document is the implementation brief for the next development pass of `transaction-fraud-intelligence`.

The goal is **not** to chase a higher validation score for its own sake. The goal is to make the project more credible, more robust, more technically defensible, and more useful as a public portfolio project.

Work on the existing `codex-fraud-v1` development branch. **Do not merge to `main` yet. Do not evaluate the locked final test period yet.**

---

## 1. V1 result summary

The first full development run completed successfully on the 1,200-customer / 120-day configuration with seed 42.

Validation period:

- 10,307 transactions
- 370 fraud transactions
- fraud rate: ~3.59%
- main review budget: 50 transactions per UTC day

At the 50-review/day operating point:

| Model | Average precision | ROC-AUC | Precision | Fraud recall | Fraud attempted-value capture |
|---|---:|---:|---:|---:|---:|
| Rules | 0.1765 | 0.6787 | 0.1863 | 0.4027 | 0.4636 |
| Logistic + history | 0.8487 | 0.9682 | 0.3888 | 0.8405 | 0.8829 |
| CatBoost, current transaction only | 0.1405 | 0.7019 | 0.1150 | 0.2486 | 0.0619 |
| CatBoost + history | **0.8956** | **0.9769** | **0.4063** | **0.8784** | **0.8983** |

The strongest V1 finding is therefore not simply that CatBoost is the best model. It is that **behavioural history is dramatically more important than using a more sophisticated algorithm on only the current transaction**.

This is the main story to preserve.

### Scenario recall for CatBoost + history

- Account takeover: **93.3%**
- Card testing: **95.9%**
- Low-and-slow: **67.7%**

Low-and-slow fraud is the main V1 weakness and should be investigated rather than hidden.

### Largest CatBoost-history feature importances in V1

1. `hour_deviation` — 18.36
2. `hours_since_previous` — 16.68
3. `new_merchant` — 10.33
4. `history_days` — 6.59
5. `prior_accounts_on_device` — 6.41
6. `log_prior_median_30d` — 5.35
7. `hour_sin` — 5.25
8. `prior_count_30d` — 4.72
9. `attempts_1h` — 4.71

This feature profile is plausible, but it also means V2 must test whether the simulator is making timing/novelty patterns too clean.

### V1 false-positive behaviour worth preserving in analysis

At the 50-review/day operating point, CatBoost + history still reviewed legitimate activity across several contexts, including travel, new phones, large purchases, routine activity and legitimate bursts. This is useful because it gives the project a realistic trade-off story instead of a near-perfect classifier.

The project should explicitly distinguish:

- a model being useful,
- a model being operationally perfect,
- and a simulator making the task artificially easy.

---

## 2. Non-negotiable safeguards

Implement these first and preserve them throughout V2.

### 2.1 Do not touch the locked final test

The current manifest shows:

`locked_test_evaluated = false`

Keep it that way.

Do not calculate test-set metrics, inspect test labels, tune from test performance, create test plots, or use the final chronological period in model or simulator decisions.

The final test is to be opened only after V2 is frozen and reviewed.

### 2.2 Preserve point-in-time correctness

Continue to enforce all current timing checks:

- future invariance
- same-timestamp batching
- delayed outcome availability
- exclusion of the current transaction from its own historical baseline
- label maturity at training / early-stopping cutoffs

Any new history feature must obey the same event-time discipline.

### 2.3 Do not leak simulator-only information

Never place these into model features:

- `is_fraud`
- `fraud_scenario`
- `legitimate_context`
- hidden customer-generation parameters
- raw customer IDs
- raw device IDs
- any variable that directly identifies whether an event was injected by the simulator

Scenario/context labels remain audit-only.

### 2.4 Do not optimise by seed shopping

Do not search simulation seeds until a flattering result appears.

Use a fixed, declared seed set before running robustness tests and report every result.

---

## 3. Priority 0 — fix environment and Colab reproducibility

The V1 automated smoke run used Python 3.11, while the manual Colab run encountered a dependency/runtime incompatibility on a newer Python environment.

V2 must run cleanly in a fresh current Google Colab CPU runtime without the user manually editing package versions.

### Required changes

1. Review all pinned dependencies in `requirements.txt` and in the notebook install cell.
2. Choose versions that are compatible with both:
   - CI Python 3.11
   - current Colab Python 3.13 or the current supported Colab runtime at implementation time
3. Do not merely silence import errors.
4. Update CI so notebook smoke execution is tested against at least Python 3.11 and Python 3.13 if GitHub Actions support is available.
5. Keep the notebook self-contained and Colab-friendly.
6. Print Python version and core package versions near the start of every run.
7. If a restart is genuinely required after dependency installation, detect/document this clearly; prefer dependency choices that avoid requiring a manual restart.

### Acceptance gate

A fresh Colab user should be able to:

`Open notebook -> Runtime -> Run all -> receive results ZIP`

without editing code.

---

## 4. Priority 1 — harden the simulator so the task is less self-fulfilling

V1 is promising, but the result is still generated inside a world we designed. V2 should intentionally make that world harder and document the changes.

### 4.1 Increase overlap between fraud and legitimate behaviour

Do not make fraud uniformly anomalous.

Increase the share of fraud that can occur with combinations such as:

- an already-seen device
- the customer's home country
- a previously seen merchant or merchant category
- a normal-looking amount ratio
- a normal transaction hour
- no immediate failed-attempt burst

At the same time, keep legitimate lookalikes such as:

- travel
- new phones
- legitimate transaction bursts
- large purchases
- shared household devices

The purpose is to make `new_device`, `new_country`, `hour_deviation`, and velocity useful but not near-deterministic shortcuts.

### 4.2 Audit simulator separability

Add a clear development diagnostic that compares fraud vs legitimate distributions for the strongest features, especially:

- `hour_deviation`
- `hours_since_previous`
- `new_merchant`
- `prior_accounts_on_device`
- `attempts_1h`
- `new_device`
- `new_country`
- `amount_ratio_30d`

Export a compact table such as `feature_separation_audit.csv` containing distribution summaries and/or simple univariate discrimination measures.

The aim is to identify features that are suspiciously easy, not to remove every useful feature.

### 4.3 Preserve the core experiment

Do **not** change the central question:

> Does customer history improve fraud ranking within a fixed daily review budget?

Keep all four comparators:

1. rules
2. logistic regression with history
3. CatBoost using current-transaction information only
4. CatBoost with history

The V2 simulator can be harder even if headline metrics fall. A more believable 0.80 result is preferable to an artificial 0.99 result.

---

## 5. Priority 1 — investigate low-and-slow fraud properly

V1 CatBoost-history recall for low-and-slow fraud is ~67.7%, compared with >93% for the other two scenarios.

This is the clearest model weakness and should become a deliberate analysis section.

### 5.1 Diagnose the misses before adding features

Create a low-and-slow miss analysis that compares:

- caught low-and-slow cases
- missed low-and-slow cases
- legitimate routine transactions

For each group, inspect at least:

- amount ratios / z-scores
- inter-transaction time
- 7-day / 14-day / 30-day transaction count change
- rolling attempted value
- merchant novelty
- category novelty or category concentration
- device/account sharing
- country novelty
- time-of-day deviation

Export a file such as `low_and_slow_diagnostics.csv` or an equivalent concise artefact.

### 5.2 Add only defensible longitudinal features

If diagnostics justify them, consider features such as:

- recent 7-day count vs previous 30-day baseline
- recent 7-day attempted value vs prior baseline
- rolling change in median/mean ticket size
- count of new merchants over 7/14 days
- merchant/category diversity shift
- repeated near-normal transactions accumulating unusual total value
- customer-specific trend or acceleration measures

Every feature must be calculable using only information available before the current transaction.

Do not add a feature solely because it encodes how the simulator injects low-and-slow fraud.

### 5.3 Report whether improvement is real

If low-and-slow recall improves, show what trade-off occurred:

- change in low-and-slow recall
- change in total recall
- change in precision
- change in false positives
- which legitimate contexts absorbed the extra alerts

Do not claim improvement if it simply floods the review queue with legitimate transactions.

---

## 6. Priority 1 — multi-seed robustness

The V1 result is from seed 42 only. V2 must show that the central conclusion is not dependent on one synthetic world.

### Required design

Use a fixed declared seed set of at least five seeds. Example:

`[42, 123, 2025, 31415, 27182]`

You may choose a different fixed list, but commit it in code/documentation before evaluating results.

For each seed, keep the same:

- number of customers
- timeline length
- review budgets
- chronological split logic
- model definitions
- feature definitions
- evaluation methodology

### Required outputs

Create an aggregated file such as:

`robustness_summary.csv`

with per-seed and aggregate statistics for each model / review budget, including at least:

- average precision
- ROC-AUC
- precision
- fraud recall
- attempted-value capture
- false-positive rate

Also provide mean, standard deviation and min/max (or confidence intervals if implemented correctly).

Create one concise visual showing whether the ranking of the four detector approaches is stable across seeds.

### Important

The question is not whether every seed produces the same exact metric. The question is whether the core conclusion — **history materially improves ranking** — is stable.

Do not open the locked final period for any seed.

---

## 7. Priority 1 — add feature-family ablation

V1 shows that history helps, but the portfolio story becomes much stronger if we can explain **which types of history matter**.

Add a small, controlled ablation analysis for the historical CatBoost model.

Suggested feature families:

- transaction amount / ticket features
- velocity / recency features
- novelty features (`new_device`, `new_country`, `new_merchant`)
- historical baseline/deviation features
- device-network features (`prior_accounts_on_device`)
- timing/circadian features

For each ablation, retrain without one family and compare against the full history model on development validation.

Do not turn this into dozens of hyperparameter experiments. Keep it interpretable and bounded.

Export:

`ablation_summary.csv`

with the main 50-review/day business metrics plus average precision.

This should answer questions such as:

- Is history useful only because of device novelty?
- Do velocity signals add independent value?
- How much do customer-specific baselines contribute?

---

## 8. Priority 1 — improve false-positive analysis

V1 already contains useful legitimate lookalikes. V2 should turn this into a stronger investigation story.

For the selected provisional validation leader, report legitimate review rates and examples for:

- routine activity
- travel
- new phone
- large purchase
- legitimate burst

For each context, show:

- total legitimate transactions
- reviewed legitimate transactions
- review rate
- median/upper-quartile model score
- top recurring evidence patterns

Include a few representative case records but avoid dumping hundreds of rows into the notebook.

The narrative should explicitly discuss the cost of false positives and why a fixed review budget is used.

---

## 9. Priority 2 — strengthen evaluation presentation

Keep current metrics, but improve the way they are surfaced.

### Required main metrics

At review budgets 20, 50 and 100 per day:

- review count
- fraud reviewed
- precision
- fraud recall
- false-positive rate
- legitimate reviews
- fraud attempted-value capture

Also retain:

- average precision
- ROC-AUC

### Additions

1. Add day-level variability summaries so the result is not just one aggregate number.
2. Show whether some validation days exhaust the review budget while others have fewer candidate transactions.
3. Add precision-recall curve(s) where useful, but do not clutter the notebook.
4. Clearly label attempted-value capture as **attempted fraudulent value**, not prevented loss or savings.
5. Keep scores explicitly labelled uncalibrated unless calibration is actually implemented.

Do not introduce an arbitrary accuracy metric as a headline number for this imbalanced task.

---

## 10. Priority 2 — notebook and repository quality

The notebook should remain executable from top to bottom, but V2 should be easier for a technical reviewer to follow.

### Notebook structure

Use clear sections:

1. objective and assumptions
2. configuration and reproducibility
3. simulation
4. point-in-time history builder
5. temporal split / label maturity
6. rules baseline
7. logistic baseline
8. current-transaction CatBoost
9. history CatBoost
10. main validation comparison
11. scenario analysis
12. low-and-slow analysis
13. false-positive analysis
14. multi-seed robustness
15. ablation analysis
16. limitations and next decision
17. export artefacts

Avoid repeating large blocks of code if helper functions can make the notebook clearer without making it opaque.

### README

Update the README only with results that have actually been executed.

Do not write marketing claims such as "production-ready" or "real-world fraud accuracy".

The README should emphasize:

- the business question
- why fixed review capacity matters
- the current-vs-history comparison
- robustness across seeds
- known limitations
- synthetic-data caveat
- that the final test remains locked

---

## 11. Output contract for V2

Keep the current timestamped output folder and ZIP workflow.

The V2 ZIP should include at least:

- `comparison.csv`
- `daily_metrics.csv`
- `scenario_breakdown.csv`
- `legitimate_context_breakdown.csv`
- `feature_importance.csv`
- `feature_separation_audit.csv`
- `low_and_slow_diagnostics.csv` (or equivalent)
- `robustness_summary.csv`
- `ablation_summary.csv`
- validation prediction files
- investigation/error-case files
- `split_audit.csv`
- `manifest.json`
- a concise set of validation charts
- saved model artefacts where appropriate

The manifest must record:

- implementation version
- Python version
- package versions
- seed list
- configuration
- feature list
- model columns
- split cutoffs
- timing-check status
- whether final test was evaluated
- selected provisional validation leader
- known limitations

`locked_test_evaluated` must remain `false`.

---

## 12. V2 acceptance criteria

V2 is ready for human review when all of the following are true:

1. Fresh Colab `Run all` works without manual code edits.
2. CI smoke execution passes on supported target Python versions.
3. All point-in-time leakage checks pass.
4. The locked final test remains untouched.
5. The simulator is demonstrably less trivial / more overlapping than V1.
6. Multi-seed results are exported and the core history-vs-current conclusion can be evaluated across seeds.
7. Low-and-slow misses are explicitly analysed.
8. Any new longitudinal features are justified by diagnosis rather than simulator knowledge.
9. Feature-family ablation is available.
10. False-positive behaviour is broken down by legitimate context.
11. The notebook and README make no unsupported real-world claims.
12. A new timestamped V2 results ZIP is produced for independent review.

Do **not** require V2 metrics to exceed V1 metrics. If simulator hardening lowers performance but improves credibility, report that honestly.

---

## 13. What not to do in this pass

Do not expand scope unnecessarily.

Do not yet implement:

- a production API
- Streamlit/dashboard deployment
- cloud infrastructure
- graph neural networks
- SHAP solely for decoration
- anomaly-detection ensembles without a clear reason
- extensive hyperparameter sweeps
- the public real-data benchmark
- evaluation of the locked final test

Those can be later milestones. V2 should first make the core experiment robust and defensible.

---

## 14. Final instruction to the implementing agent

Implement the changes above rather than merely writing recommendations.

Keep work on `codex-fraud-v1` (or create a clearly named V2 development branch only if branch hygiene requires it). Preserve all useful V1 behaviour and outputs unless a change is necessary for correctness.

When finished:

1. ensure the notebook passes automated smoke execution,
2. push all code/documentation changes,
3. summarize exactly what changed,
4. state any compromises or items not implemented,
5. provide the Colab link for the updated notebook,
6. instruct the user to run the full V2 experiment and return the generated ZIP,
7. **do not merge to `main`,**
8. **do not evaluate the locked final test period.**

The next decision will be made only after the V2 output ZIP is independently reviewed against the V1 baseline.