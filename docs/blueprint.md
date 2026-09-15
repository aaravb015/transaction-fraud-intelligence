# First implementation blueprint — v0.1

This is the readable specification for the first executable notebook. It records the implementation scope agreed in the project conversation; it is not a verbatim copy of the original uploaded master blueprint.

## What we are building

A model-development experiment for fictional payment attempts. We generate history, score transactions, compare detectors under a daily review budget, and inspect errors. Actual model results are produced by executing the notebook.

The target is whether an attempt is fraudulent. A failed fraudulent attempt remains fraud-labelled. Monetary metrics therefore refer to attempted value, not settled loss.

## Data source

The primary source is our Python simulator. No customer data, API key, paid dataset or external download is required beyond Python packages.

The public-data benchmark is a later, separate experiment. It will not retroactively prove that this simulator matches real payment populations.

## Entities and availability

| Field / entity | Meaning | Availability and use |
|---|---|---|
| Customer ID | Stable fictional account | Groups historical observations; excluded from predictors |
| Merchant ID | Stable fictional merchant | Novelty calculations; excluded from predictors |
| Device ID | Observed payment device | Account novelty and previous cross-account use; excluded as a raw predictor |
| Transaction ID | Unique event identifier | Joins and deterministic ranking ties; excluded from predictors |
| Timestamp | Attempt arrival time in UTC | Known at scoring |
| Amount | Positive attempted amount, standardized to fictional INR | Known at scoring |
| Country and category | Current payment attributes | Known at scoring |
| Status | Succeeded or failed payment attempt | Becomes usable only at outcome_available_at |
| outcome_available_at | Simulated result arrival, 2–120 seconds later | Controls the known-failure queue |
| is_fraud | Simulator ground truth | Available for supervised training only once label_available_at is reached |
| label_available_at | Timestamp plus seven days | Assumed confirmation delay for both classes |
| fraud_scenario | Injected mechanism or legitimate | Evaluation only |
| legitimate_context | Routine, travel, new phone, large purchase, legitimate burst, or not applicable | Evaluation only |
| Hidden customer preferences | Spend distribution, activity rate, preferred hour and home country | Generation and auditing only; excluded from features |

Merchant categories are assigned once and remain consistent. Customers can share household devices legitimately. A new phone becomes the customer's ongoing device after its change date. Context tags are exclusive and can simplify overlapping events.

## Simulator scope

Normal customers vary in amount distribution, activity rate, preferred hour, merchant preferences and home country. They can travel, replace phones, make large purchases and produce legitimate bursts.

Three fraud scenarios are injected:
1. Account takeover: several purchases, sometimes using an existing device.
2. Card testing: rapid smaller attempts with variable failures.
3. Low-and-slow: several less unusual purchases spread across days.

A configured 35% of customers receive one fraud episode. This is an artificial episode-injection rate, not the transaction fraud rate or a population estimate. Fraud episode starts are sampled across time; episodes are truncated at the simulation horizon. Transaction-level prevalence is measured by the run.

## Feature design

The notebook builds 19 numerical and 2 categorical features.

| Family | Features |
|---|---|
| Current attempt | log amount; sine/cosine of hour; category; country |
| History coverage | total prior count; count in prior 30 days; days since first observation |
| Recent activity | counts in 1 hour and 24 hours; log attempted amount in 24 hours; hours since previous attempt |
| Personal amount context | log prior 30-day median; current amount divided by that median; deviation from prior mean scaled by prior standard deviation plus 50 |
| Novelty | unseen account device, country and merchant |
| Outcome history | failed outcomes that became known in the preceding hour |
| Relationships | distinct accounts observed using the current device before this time |
| Time preference | circular deviation from the account's historical transaction-hour direction |

The amount comparisons are missing until at least three prior attempts exist. The coverage fields show this limitation to the model. Historical amounts include all attempts, not just successful payments.

For a timestamp t, history contains events strictly before t. All events at t are scored before updating any history. Thirty-day and shorter windows include their left boundary. Failed outcomes are queued and released when their availability time is reached.

The feature builder receives an explicit observable-column list and never receives the fraud or context label columns. The full chronological history is transformed, but no future observations influence earlier features.

## Training and validation

For the default 120-day simulation:
- Fit time is day 72. Training uses earlier events whose labels are known by then.
- Validation begins on day 86. The early-stopping period starts on day 72, using only labels available by day 86.
- Development validation covers days 86 through 101.
- The reserved final test begins on day 102.

The resulting gaps model the seven-day label delay. History may update during gaps from observed unlabelled activity. No resampling is applied.

The same customers may appear across periods because returning-customer history is central to the task. Separate new-customer validation is a future extension; it is not claimed by this run.

## Detectors

- Rules: five illustrative fixed conditions, scored with weights totaling 100.
- Logistic regression: fitted preprocessing, numerical missing-value indicators, standardized numerical features, one-hot categoricals, regularized classifier.
- CatBoost using current fields: a transaction-only baseline.
- CatBoost using all 21 features: tests the value of history and relationship context.

CatBoost uses the early-stopping period. Neither classifier tuning nor model selection uses the reserved test. These initial models use no class weighting, probability calibration, anomaly ensemble or hyperparameter search.

The history comparison uses the same tree settings but permits independent early stopping. It measures feature-set performance under those settings, not the result of equally optimized exhaustive searches.

## Evaluation and investigation

We report average precision and ROC-AUC for ranking, then precision, recall, false-positive rate, number of legitimate reviews, and fraudulent attempted-value capture at 20, 50 and 100 daily reviews.

Ranking is offline within each UTC day. Ties use transaction ID, which follows event order in the simulator. The 50-review budget selects a provisional development leader by precision, then attempted-value capture, then average precision, then name.

Scenario breakdowns show recall and attempted-value capture. Legitimate-context breakdowns show alert rates. Daily breakdowns show variation without claiming a confidence interval.

Each detector exports high-score false positives, high-value missed fraud and selected true positives. Explanations separate triggered rules from observed feature evidence. Investigation history masks outcomes not known at the selected decision time.

There is no live blocking decision, calibrated probability, financial cost model or production queue. REVIEW and NO_REVIEW describe the offline diagnostic only.

## Outputs and reproducibility

A run folder and ZIP contain metrics, case files, predictions, a chart, development data, models, feature importances and a manifest. The reserved test period is excluded from transaction exports and all model evaluation.

The manifest records package versions, feature order, configuration, rule weights, timeline, timing-check outcomes and a fingerprint of the generated dataset.

Models require the same feature-building process at inference. The first version stores this in the self-contained notebook; reusable modules are planned after the experiment is measured.

## Required checks

- Unique IDs, positive amounts, valid entity references and ordered timestamps.
- Training and validation subsets contain both classes.
- Current transaction does not enter its own historical amount baseline.
- Same-time events cannot observe one another.
- Failed outcomes are unavailable before their release time.
- Appending future transactions leaves earlier features unchanged.
- Logistic regression converges.
- Notebook passes schema validation, Python syntax checks and an automated small-data execution.

The automated run tests functionality. Its metrics are not the intended final portfolio result.

## After the first run

Investigate failures on validation, record a hypothesis, change one aspect and compare the same metrics. Repeat predefined experiments across seeds and temporal periods. Then freeze the approach and evaluate the reserved test.

The larger roadmap includes reusable modules, an investigation dashboard, a separate public-data benchmark, calibration and an operational decision policy. Graphs, streaming and extra infrastructure are later additions.
