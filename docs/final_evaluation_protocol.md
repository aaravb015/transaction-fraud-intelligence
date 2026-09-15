# Frozen final-evaluation protocol

This document is written **before** the reserved final period is materialized or scored. It implements the next step authorized by `docs/v2_freeze.md`; it does not reopen model development.

## Frozen inputs

- Frozen V2 implementation commit: `a4ab1de96e2d0bed7fa1f19836f9585f3f10344f`
- Freeze-document commit: `455b37921e90193199543a1207026e68770faeac`
- Frozen V2 engine SHA-256: `6debf4f7501ee0f937bd53c9992f8ecc642c946039dd653aa284229078d2cb1e`
- Frozen development-results ZIP SHA-256: `650d3fbbce83ae82f0d821e73f30174998a2956e347dab7bc74f5bb2c41ae58a`
- Per-seed development fingerprint source: `fraud_v2_20260915T093200_432180Z.zip`
- Fingerprint-source ZIP SHA-256: `94ce8bd6b6f0a2490ee1b31c90bbaf1abc90a5433730a0f55a68f20dc3a47b05`
- Seeds: `[42, 123, 2025, 31415, 27182]`
- Models: `rules`, `logistic_history`, `catboost_current`, `catboost_history`
- Preselected primary model: `catboost_history`
- Review budgets: `20`, `50`, `100` per UTC day; headline operating point `50/day`
- Training cutoff, early-stop boundary, feature definitions, rule weights, model classes and hyperparameters remain frozen.
- The longitudinal challenger remains diagnostic only and is not promoted.

## Why a materialization rule is required

V2 deliberately did **not** generate days 102–119 at all. Therefore there is no hidden file of final rows waiting to be opened. The final evaluator must materialize the reserved continuation without changing the already reviewed development world.

The rule below is predeclared before any final-period result exists. The evaluator must prove that every development transaction and every point-in-time development feature is exactly identical to frozen V2 before it is allowed to score the final period.

## Reserved-period continuation rule

The evaluator loads the exact frozen V2 engine and verifies its SHA-256. It then creates an in-memory final-evaluation variant with the smallest possible continuation patch:

1. The original development simulation and both RNG streams are consumed in exactly the frozen order first.
2. The horizon is extended from day 102 to day 120 only so already-started fraud episodes may naturally spill across the day-102 boundary.
3. After all frozen development draws are consumed, additional normal attempts are generated for days 102–119 using each customer's already-drawn rate, amount distribution, usual hour, favourite merchants, home country and device history.
4. One-off phone-change, travel, legitimate-burst and fraud-episode mechanisms are extended only for customers who did not already receive that event in development. Their conditional probabilities are duration-scaled from the frozen 102-day probabilities using `1 - (1-p)^(18/102)`.
5. New reserved-period fraud episodes reuse the frozen account-takeover, card-testing and low-and-slow mechanics, mimic probability, amount logic, device/country/merchant familiarity logic, failure probabilities and scenario cycle. No new fraud type or easier signature is introduced.
6. Final-period feature engineering uses the same frozen point-in-time history builder over the full chronological stream. Final attempts can see only earlier observable attempts/outcomes, never future rows or their own values in history.
7. The evaluator requires an exact row-for-row match between the frozen V2 development dataframe and the `< day 102` prefix of the extended simulation, and an exact feature-for-feature match for every development transaction. Any mismatch invalidates the run and stops before a result ZIP is accepted.

This continuation is synthetic and should be described as such in the final write-up. It is not a real-world holdout dataset.

## Model fitting and scoring

For each of the five frozen seeds:

- fit rows remain the original frozen training rows;
- early stopping remains the original frozen early-stop window;
- development validation rows are **not** added to training after the freeze;
- no hyperparameter, feature, rule, seed or threshold selection is performed;
- all four frozen comparators score only reserved-period rows;
- top-K review selection is applied independently by UTC day at the same 20/50/100 capacities;
- average precision, ROC-AUC, precision, fraud recall, false-positive rate, legitimate reviews and attempted fraudulent value capture use the same definitions as V2;
- the 50/day predictions are used for scenario and legitimate-context breakdowns.

The valid output must contain the complete `5 seeds × 4 models × 3 budgets = 60` final-result grid plus per-seed final predictions, prefix-integrity checks, scenario/context breakdowns and a final manifest.

## One-time rule

`notebooks/fraud_final_evaluation.ipynb` is intentionally **not executed in CI**. CI/static checks may validate its structure, source hashes and syntax, but they must never materialize days 102–119.

The official final run is the first intentional full `Runtime → Run all` execution after this protocol and its notebook are committed and reviewed. Once valid final-period outcomes have been exposed, the period is no longer unseen and must never be described as locked again.

Whatever valid result appears will be reported. A poor result is not grounds to change the frozen simulator, features, model family, seeds, budgets or primary-model choice and rerun as though the final period were still unseen.

## After the run

Download the timestamped `fraud_final_....zip` and independently review it before merging anything to `main`. The final README should keep development evidence and reserved-period evidence clearly separated and retain the synthetic-data limitations.

## Implemented runner contract

The dedicated implementation is generated as `notebooks/fraud_final_evaluation.ipynb` from readable sources. It:

- verifies the frozen `src/fraud_v2.py` SHA-256 before doing any work;
- derives a narrowly scoped continuation from the frozen simulator source;
- consumes every original development RNG draw before adding boundary spillovers or new reserved-period draws;
- checks each regenerated development dataframe against the reviewed full-run fingerprint;
- requires exact row and feature equality between frozen development and the extended prefix;
- fits the frozen three learned models only on the original training/early-stop partitions;
- evaluates the four fixed comparators on days 102–119 at 20/50/100 reviews per day;
- exports the complete 60-row grid, per-seed predictions, daily/scenario/context evidence, prefix checks and a final manifest;
- requires an explicit official-run command-line acknowledgement; and
- is checked statically in CI but is never executed there.

The continuation draw order is also frozen: after all development draws, normal-stream additions proceed in customer order with conditional phone, conditional travel, routine attempts and conditional legitimate burst; fraud-stream additions then proceed in customer order for accounts without an earlier fraud episode. Duration-scaled probabilities use `1 - (1-p)^(18/102)`.
