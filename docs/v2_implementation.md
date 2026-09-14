# V2 implementation and review guide

Source of truth: `docs/v1_review_and_v2_plan.md`. The original V1 notebook, blueprint and review document are preserved. V2 lives in `notebooks/fraud_v2.ipynb` on `codex-fraud-v1`.

## Acceptance mapping

| Specification | Implementation / evidence |
|---|---|
| Current Colab / 3.11 and 3.13 compatibility | CatBoost updated to 1.2.10; all experiment packages installed as wheels into an isolated target; real imports happen in a fresh process; CI covers 3.11, 3.12 and 3.13 |
| No manual restart | The notebook kernel is never upgraded or asked to reload a binary module; the same source runs in subprocesses using isolated dependencies |
| Version disclosure | Notebook prints host/worker Python and core package versions; manifest and resolved package file preserve them |
| Never inspect final period | Generation stops at the development boundary before outcomes/labels are constructed; guard functions reject reserved timestamps before scoring or label access |
| Point-in-time features | Batching, outcome-release queue, disjoint historical windows, coverage rules and regression checks |
| Preserve primary V1 experiment | Exact four comparator names, current-input list and original 21 history features; original fixed rule weights and budgets |
| Harder simulator | 80% episode-level mimic mixture; 85% familiar-device choice; 92% home-country choice; 80% familiar-merchant choice; variable ordinary amounts/hours/failures/spacing; lookalikes preserved |
| Demonstrate overlap | Paired `simulator_overlap_audit.csv` and distribution/univariate `feature_separation_audit.csv`, plus one concise chart |
| Low-and-slow analysis | Caught, missed and routine groups across all requested feature themes; raw distributions and a compact median overview |
| Defensible new measures | 7/14/30-day activity compared with disjoint earlier 30-day windows; ticket, novelty and diversity/concentration changes; no simulator parameters |
| Diagnose before adding features | Predeclared diagnostic gate nominates a frozen list; separate challenger only after diagnosis, with a documented no-candidate outcome |
| Honest longitudinal trade-offs | Baseline/challenger low-and-slow and total recall, precision, legitimate reviews, FPR, value capture and context deltas; seed 42 explicitly exploratory |
| Five-seed robustness | Exactly 42, 123, 2025, 31415, 27182 in both smoke and full; all per-seed metrics plus mean, sample std, min/max; rank heatmap |
| Feature-family ablations | Six disjoint numerical families, retrained on primary seed 42; same folds/max iterations and separate early stopping; full reference row |
| False-positive analysis | Five legitimate contexts; total/reviewed counts, rates, all-score and reviewed-score median/Q75, three recurring patterns and up to three examples/context |
| Day-level variability/capacity | Calendar includes zero-candidate days; exhausted and underfilled budgets, unused slots, daily metric distributions and defined-day counts |
| Output contract | Required files, all five seeds, complete primary model/budget grid and exported timestamp boundaries validated before ZIP creation |
| Review before next step | Full V2 remains for the user to run and independently review; no merge or final-test switch exists |

## Runtime design

The notebook embeds the exact engine source and the installation/frontend helper. The generator verifies this matches the repository. This preserves one-file portability without maintaining two independent implementations.

Pinned scientific packages are installed into `.fraud_v2_runtime/py<version>_<requirements hash>/packages`. Python subprocesses start with `-S` and that package directory on `PYTHONPATH`, so they do not inherit already-loaded Colab binary modules. A real import probe runs before simulation. Import/install errors stop the notebook with the original output; they are not suppressed.

The stage process checkpoints only development state between notebook cells. A new Run all creates a new timestamped work folder and output ZIP. Dependencies can be reused within the same runtime; package fingerprints and recorded versions make the environment explicit.

Google's published [runtime-version documentation](https://research.google.com/colaboratory/runtime-version-faq.html) lists Python 3.12 for recent Colab runtimes. CI includes 3.12 as well as the specification's requested 3.11 and 3.13. [CatBoost 1.2.10](https://pypi.org/project/catboost/) supplies binary wheels for those versions. A direct browser-based full Colab run remains the user's acceptance step; CI exercises the same notebook installation and execution route on the supported interpreters.

## Why the longitudinal challenger is separate

The main comparison answers the same question as V1 with the same feature definitions. Diagnostic measures are calculated point in time but do not automatically become predictors.

The nomination rule uses missed low-and-slow versus routine legitimate development observations in seed 42: at least five finite observations per group, 50% coverage in each, and an absolute standardized mean difference of 0.25. Features passing the rule are all retained; no ranking-score search selects a flattering subset. Zero passing features is a valid, explicitly recorded result.

When present, the challenger is trained on the same training rows with the same maximum tree budget. Its feature list is frozen once and reused on all five worlds. It has separate prediction and trade-off exports and does not participate in selecting the primary provisional leader. Feature selection on seed-42 validation makes its seed-42 difference exploratory; the other four seeds provide synthetic replication. No gains are advertised as final-test improvement.

## Simulator reference and interpretation

The paired reference has the same development-only normal-event random streams and V1-like fraud shortcut settings. It deliberately shares the V2 development boundary and RNG-stream architecture, so it is **not** a byte-for-byte recreation of the archived V1 simulation. Every reference artifact is labelled accordingly.

Observed overlap/discrimination tables establish what changed in the generated task. They do not establish realism against actual payment data. V2 need not exceed V1 scores. The review should consider signal dependence, low-and-slow weaknesses, false-positive costs and consistency across seeds.

## Point-in-time details

- Original attempt features keep their V1 definitions and include attempted amounts from both succeeded and failed attempts.
- Recent windows include their left boundary and exclude the current timestamp.
- The earlier comparison window is disjoint: for a seven-day recent window it is `[t-37d,t-7d)`, and analogously for 14 and 30 days.
- Baseline ratios require three earlier observations and a full observed-history span. Sparse recent ticket/diversity comparisons also require three recent observations.
- New-merchant counts are distinct merchants first observed within the window, using novelty as it was known at the prior event.
- Category concentration is the sum of squared historical category shares.
- Same-time records never see one another's histories or device-account updates.
- Failure counts reflect outcomes released in the preceding hour, not the current attempt's eventual status.
- Training and early-stopping membership require label maturity at the stated cutoff.
- Final-period tests use fabricated boundary sentinels, not actual final-period observations.

## Output interpretation

`robustness_summary.csv` is a single wide metric table with `row_type` equal to `seed`, `mean`, `std`, `min` or `max`. The primary four models retain their names. `ablation_summary.csv` has a reference and six family-removal rows at 50 reviews/day. Deltas are relative to the full history model; early-stopping iterations are recorded.

Context score summaries include all legitimate rows in the context and separate summaries for reviewed rows. Evidence patterns are finite descriptive tags, not a claim that each detector used each feature. Daily means are unweighted across defined days; aggregate business metrics remain transaction-weighted via total counts.

The ZIP includes primary-model files, the optional challenger, primary and other-seed predictions/cases, charts, development transactions, audit tables, full configuration and an export validation record. Model files need this feature builder and column order at inference.

## Deliberate limits

No unsupported V2 result is entered into the README. Smoke metrics are engineering checks, not the final portfolio experiment. No real-data benchmark, API, deployed dashboard, calibration, SHAP, anomaly ensemble or production loss estimate is added. No final-period data is generated, inspected or evaluated, and nothing is merged into `main`.
