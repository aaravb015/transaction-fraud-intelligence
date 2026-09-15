# V2 experiment design fixed before execution

This implements `v1_review_and_v2_plan.md`. The locked final period is never generated, read, scored, plotted or exported by the V2 pipeline. Work stays on `codex-fraud-v1`; `main` is not merged.

## Fixed choices

- Seeds, in order: **42, 123, 2025, 31415, 27182**. All are included in smoke and full runs.
- Full configuration: 1,200 customers, a nominal 120-day timeline, 240 merchants, 600 maximum boosting iterations, two CPU threads. Smoke: 360 customers, 100 iterations. Other settings are identical.
- Cutoffs: fit on day 72, validation starts on day 86, reserved final starts on day 102. Both classes' labels mature after seven days.
- All four primary comparators, their original 21-feature/current-feature definitions, and review budgets 20/50/100 are preserved.
- Six disjoint numerical feature-family ablations are run on seed 42 only. Country and merchant category stay in each ablation. Every ablation uses the same training, early stopping and validation periods and maximum iteration count as the primary model.
- Auxiliary longitudinal measures are diagnostic-only initially. After the seed-42 low-and-slow diagnosis, a fixed gate may nominate features: at least five missed low-and-slow and five routine legitimate validation rows, at least 50% finite coverage in each group, and an absolute standardized mean difference of at least 0.25. All passing predeclared candidates are retained, with no metric-based search or cap.
- If the gate passes, a separately named longitudinal challenger uses the same frozen feature list on every seed. It never replaces the four primary comparators. Seed-42 gains are exploratory because its validation informed selection; the other seeds provide a separate synthetic replication, not a real-world claim. If the gate does not pass, the decision and lack of a challenger are explicitly exported.
- A paired, untrained V1-like simulator reference uses the same normal-event streams and old fraud shortcut settings. It is a controlled separability reference, **not an exact reconstruction of the archived V1 run**. Only development events are generated in either world.
- No score target, seed selection, hyperparameter search, final-test evaluation, production API, deployed dashboard, SHAP or external-data benchmark is added.

## Colab runtime strategy

The notebook installs a wheel-only dependency set in a project-specific directory, then runs the experiment in fresh Python subprocesses using that directory. Colab's notebook kernel and its preloaded NumPy/Pandas modules are not modified. This avoids a manual kernel restart while keeping one-click Run all and an executable, self-contained notebook.

CI targets Python 3.11, 3.12 and 3.13. Python 3.12 is included because Google's published runtime documentation lists that version; the specification's requested 3.13 is tested as well. Versions and resolved packages are recorded in the ZIP.

## Interpretation

Daily top-K is offline ranking over all payment attempts, including failed ones. Attempted fraudulent value is not prevented loss or savings. Scores are uncalibrated. No numerical V2 improvement is asserted before the user independently reviews the full results.
