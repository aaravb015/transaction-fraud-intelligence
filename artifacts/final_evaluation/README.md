# Final-evaluation evidence snapshot

This directory makes the official one-time final-evaluation evidence auditable from the repository without rerunning the consumed reserved period.

## Official artifact

- Archive: `fraud_final_20260915T101601_058909Z.zip`
- SHA-256: `d82f31734a3a0255d2b59704217cb79829bc861a2bde7ca9cba20a59d4344b53`
- Evaluated at: `2026-09-15T10:22:59.867480+00:00`
- Final evaluator recorded by the artifact: `0.3.0`

The following files are copied from the official ZIP:

- `final_manifest.json`
- `prefix_integrity_checks.csv`
- `final_output_contract_check.json`

`official_artifact.sha256` records the checksum of the complete ZIP.

The following two small files are post-hoc audits derived only from the exported prediction CSVs in that same official ZIP; they do not change or rerun the experiment:

- `prevalence_audit.csv` — per-seed transaction counts, fraud counts, transaction-level prevalence, primary-model precision, and precision lift over prevalence at 50 reviews/day.
- `rules_tie_audit.csv` — quantifies the deterministic tie-breaking limitation of the coarse rules baseline.

## Evaluator-source provenance limitation

The official artifact identifies its evaluator as version `0.3.0` and records hashes for the frozen engine, generated final engine, and continuation source. The repository history currently contains the earlier pre-exposure `src/final_eval_runner.py` (`0.1.0`) and does **not** contain the exact `0.3.0` evaluator source that produced the official ZIP.

That is a reproducibility/audit-trail limitation. It does **not** make the final artifact's metrics disappear, and the official ZIP contains exact development-prefix/feature checks and the full exported predictions from which the headline metrics can be independently recomputed. However, the repository must not imply that the checked-in `0.1.0` runner is the exact program that produced the official result.

Accordingly:

- the reserved period must not be rerun to repair this provenance gap;
- the official artifact and its checksum remain the record of the completed evaluation;
- `scripts/verify_final_artifact.py` verifies the official ZIP and recomputes evidence from its exports without fitting or scoring models;
- any future V3 should commit and tag the exact evaluator source before its untouched evaluation is run.

## Interpretation cautions

The final synthetic transaction-level fraud prevalence averaged about **2.673%** across the five seeds (pooled: **1,509 / 56,538 = 2.669%**). The primary model's mean precision at 50 reviews/day was **22.44%**, roughly **8.4×** the base fraud rate.

The rules baseline has many tied zero scores. Across the five seeds, roughly **37–40%** of its selected 50/day queue consisted of zero-score transactions whose ordering was resolved deterministically by `transaction_id`. Its exact precision should therefore be treated as a coarse baseline rather than a finely ranked comparator.

The reserved-period continuation also has a documented boundary discontinuity: development fraud episodes truncated at day 102 are not resumed. The final period is therefore a synthetic continuation experiment, not a naturally observed stationary holdout.
