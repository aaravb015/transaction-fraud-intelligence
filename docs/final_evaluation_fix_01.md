# Final-evaluation engineering fix 01

## Status before fix

The first attempted execution of `notebooks/fraud_final_evaluation.ipynb` failed before any simulator call, reserved-period materialization, model fitting, scoring, or metric computation.

The traceback occurred while dynamically loading the frozen V2 source. Python 3.13's `dataclasses` implementation resolves `sys.modules[cls.__module__]` while decorating the frozen `Config` dataclass. The transient module created by `types.ModuleType` had not yet been inserted into `sys.modules`, causing:

`AttributeError: 'NoneType' object has no attribute '__dict__'`

Therefore the reserved period remained unseen after this failed attempt.

## Engineering-only change

Commit `c240fe3c07f62c719d134093d2e438c5551a56e9` changes only `src/final_eval_continuation.py::load_module`:

- import `sys`;
- register the transient module in `sys.modules` before `exec`;
- remove it again if execution raises.

This is module-loader compatibility plumbing only. It does **not** change:

- the frozen V2 engine or its SHA-256;
- simulator probabilities/mechanics;
- RNG ordering;
- features;
- labels;
- train/early-stop/final boundaries;
- model classes or hyperparameters;
- seeds;
- review budgets;
- selected primary model;
- evaluation metrics.

The Colab notebook is repinned to this fixed final-evaluation code commit before the next intentional run. The next successful run remains the first exposure of reserved-period outcomes.