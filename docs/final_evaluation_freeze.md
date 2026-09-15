# Final-evaluation implementation freeze

This record freezes the dedicated final-evaluation implementation **before days 102–119 are materialized, inspected or scored**.

## Freeze point

- Repository: `aaravb015/transaction-fraud-intelligence`
- Development branch: `codex-fraud-v1`
- Final-runner implementation commit: `2a36e532a075953b458f0346d3de42309c7cf858`
- Frozen V2 implementation commit: `a4ab1de96e2d0bed7fa1f19836f9585f3f10344f`
- Final evaluator: `src/fraud_final.py`
- Final evaluator SHA-256: `c21c0a84706399ab6758e9970426e418efc94ecb6c3e4d64bc9a6726d68424a4`
- Runnable notebook: `notebooks/fraud_final_evaluation.ipynb`
- Runnable notebook SHA-256: `ae2997a2ad54c52348bf7a75a8be047c4450a309ed9b200596f996346ed945e6`
- Frozen V2 engine SHA-256: `6debf4f7501ee0f937bd53c9992f8ecc642c946039dd653aa284229078d2cb1e`
- Continuation-source SHA-256 from static preflight: `0436a0505b8c6ce15864dedeb0851a27b1792930afa5f3a79ff0e131dfdf6243`
- Final evaluator version: `0.3.0`

The implementation commit above is the executable freeze. This documentation-only commit does not change the evaluator or notebook.

## Frozen evaluation choices

- Seeds: `[42, 123, 2025, 31415, 27182]`
- Comparators: `rules`, `logistic_history`, `catboost_current`, `catboost_history`
- Preselected primary model: `catboost_history`
- Review budgets: `20`, `50`, `100` per UTC day
- Headline operating point: `50` reviews per UTC day
- Reserved period: `[2025-04-13 00:00:00+00:00, 2025-05-01 00:00:00+00:00)`
- Frozen feature sets, rule weights, fitting cutoffs, early stopping, hyperparameters and metric definitions remain unchanged.
- The longitudinal challenger remains diagnostic and is not included in the final primary comparison.

## Pre-run verification completed

- All 13 unit/regression safeguards passed.
- The final notebook matched its generated sources and passed schema and Python syntax validation.
- Static continuation compilation and source-hash verification passed.
- CI contains no command that can acknowledge or execute the official final path.
- The existing V2 notebook completed its full small-data smoke workflow, including all five seeds, model fitting, ablations and ZIP validation.
- Full-size development-only regeneration reproduced the reviewed dataframe fingerprint exactly for every seed.
- No check invoked the continuation simulator.

At this freeze:

- `locked_test_materialized = false`
- `locked_test_evaluated = false`

## One-time execution rule

The next permitted action is to open `notebooks/fraud_final_evaluation.ipynb` at this frozen branch state and select **Runtime → Run all once**. A valid run must pass all five exact prefix checks and produce the complete 60-row result grid before its ZIP is accepted.

If execution stops before final outcomes are displayed, preserve the complete error and review it before deciding whether an engineering-only correction and explicit re-freeze are permissible. Once any valid reserved-period results are exposed, do not tune, replace, subset or rerun the frozen procedure as though the period remained unseen.

Do not merge to `main` until the resulting final ZIP has been independently reviewed and development versus final evidence is reported separately.
