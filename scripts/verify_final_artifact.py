"""Verify the official final-evaluation ZIP without rerunning the experiment.

This is a post-hoc artifact verifier. It never simulates data, fits models, or
scores the reserved period again. It checks the official archive checksum and
recomputes the exported metrics directly from the prediction CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

OFFICIAL_ARCHIVE = "fraud_final_20260915T101601_058909Z.zip"
OFFICIAL_SHA256 = "d82f31734a3a0255d2b59704217cb79829bc861a2bde7ca9cba20a59d4344b53"
SEEDS = (42, 123, 2025, 31415, 27182)
MODELS = ("rules", "logistic_history", "catboost_current", "catboost_history")
BUDGETS = (20, 50, 100)
PRIMARY_MODEL = "catboost_history"
METRICS = (
    "average_precision",
    "roc_auc",
    "review_count",
    "fraud_reviewed",
    "precision",
    "fraud_recall",
    "fraud_attempt_value_capture",
    "legitimate_reviewed",
    "false_positive_rate",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(zf.read(name)))


def ratio(a: float, b: float) -> float:
    return float(a / b) if b else np.nan


def selected_mask(frame: pd.DataFrame, k: int) -> pd.Series:
    work = frame[["timestamp", "score", "transaction_id"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work["day"] = work["timestamp"].dt.floor("D")
    ordered = work.sort_values(
        ["day", "score", "transaction_id"], ascending=[True, False, True]
    )
    chosen = set(ordered.groupby("day", sort=False).head(k)["transaction_id"])
    return frame["transaction_id"].isin(chosen)


def recompute_metrics(frame: pd.DataFrame, k: int) -> dict[str, float]:
    selected = selected_mask(frame, k)
    picked = frame.loc[selected]
    hits = picked.loc[picked["is_fraud"].eq(1)]
    fraud_n = int(frame["is_fraud"].sum())
    legit_n = int(frame["is_fraud"].eq(0).sum())
    fraud_value = float(frame.loc[frame["is_fraud"].eq(1), "amount"].sum())
    false_n = int(picked["is_fraud"].eq(0).sum())
    return {
        "average_precision": float(average_precision_score(frame["is_fraud"], frame["score"])),
        "roc_auc": float(roc_auc_score(frame["is_fraud"], frame["score"])),
        "review_count": float(len(picked)),
        "fraud_reviewed": float(len(hits)),
        "precision": ratio(len(hits), len(picked)),
        "fraud_recall": ratio(len(hits), fraud_n),
        "fraud_attempt_value_capture": ratio(float(hits["amount"].sum()), fraud_value),
        "legitimate_reviewed": float(false_n),
        "false_positive_rate": ratio(false_n, legit_n),
    }


def assert_close(actual: float, expected: float, label: str) -> None:
    if pd.isna(actual) and pd.isna(expected):
        return
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise AssertionError(f"{label}: actual={actual!r}, expected={expected!r}")


def verify(path: Path) -> dict:
    if path.name != OFFICIAL_ARCHIVE:
        raise AssertionError(f"Expected official archive name {OFFICIAL_ARCHIVE!r}")
    actual_sha = sha256_file(path)
    if actual_sha != OFFICIAL_SHA256:
        raise AssertionError(f"Official archive SHA-256 mismatch: {actual_sha}")

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        required = {
            "final_manifest.json",
            "final_output_contract_check.json",
            "prefix_integrity_checks.csv",
            "final_comparison.csv",
        }
        missing = sorted(required - names)
        if missing:
            raise AssertionError(f"Missing required files: {missing}")

        manifest = json.loads(zf.read("final_manifest.json"))
        expected_manifest = {
            "final_evaluator_version": "0.3.0",
            "frozen_implementation_commit": "a4ab1de96e2d0bed7fa1f19836f9585f3f10344f",
            "frozen_engine_sha256": "6debf4f7501ee0f937bd53c9992f8ecc642c946039dd653aa284229078d2cb1e",
            "preselected_primary_model": PRIMARY_MODEL,
            "prefix_integrity_passed_for_all_seeds": True,
            "locked_test_materialized": True,
            "locked_test_evaluated": True,
            "official_evaluation_count_for_this_run": 1,
            "selection_after_final_outcomes": False,
            "longitudinal_challenger_promoted": False,
            "synthetic_data_only": True,
            "attempted_value_is_not_prevented_loss": True,
        }
        for key, expected in expected_manifest.items():
            if manifest.get(key) != expected:
                raise AssertionError(f"Manifest {key!r}: {manifest.get(key)!r} != {expected!r}")
        if tuple(manifest["seed_list"]) != SEEDS:
            raise AssertionError("Seed list mismatch")
        if tuple(manifest["models"]) != MODELS:
            raise AssertionError("Model list mismatch")
        if tuple(manifest["review_budgets"]) != BUDGETS:
            raise AssertionError("Review-budget list mismatch")

        contract = json.loads(zf.read("final_output_contract_check.json"))
        if contract != {
            "required_files": 8,
            "result_grid_rows": 60,
            "seed_prediction_sets": 5,
            "contract_passed": True,
        }:
            raise AssertionError(f"Unexpected output contract: {contract}")

        prefix = read_csv(zf, "prefix_integrity_checks.csv")
        if set(prefix["seed"].astype(int)) != set(SEEDS):
            raise AssertionError("Prefix check seed set mismatch")
        if not prefix[["development_rows_exact_match", "development_features_exact_match"]].all().all():
            raise AssertionError("At least one prefix-integrity check failed")
        expected_hashes = {int(k): v for k, v in manifest["expected_development_hashes"].items()}
        for row in prefix.itertuples(index=False):
            if row.development_dataset_sha256 != expected_hashes[int(row.seed)]:
                raise AssertionError(f"Seed {row.seed}: development dataset hash mismatch")
            if row.development_dataset_sha256 != row.expected_development_dataset_sha256:
                raise AssertionError(f"Seed {row.seed}: exported expected hash mismatch")

        comparison = read_csv(zf, "final_comparison.csv")
        if len(comparison) != len(SEEDS) * len(MODELS) * len(BUDGETS):
            raise AssertionError(f"Expected 60 comparison rows, got {len(comparison)}")

        prevalence_rows = []
        rules_tie_rows = []
        verified_rows = 0
        for seed in SEEDS:
            for model in MODELS:
                pred_name = f"seeds/{seed}/{model}_final_predictions.csv"
                if pred_name not in names:
                    raise AssertionError(f"Missing prediction file: {pred_name}")
                frame = read_csv(zf, pred_name)
                required_cols = {"transaction_id", "timestamp", "amount", "is_fraud", "score"}
                if not required_cols <= set(frame.columns):
                    raise AssertionError(f"{pred_name}: missing prediction columns")
                for k in BUDGETS:
                    actual = recompute_metrics(frame, k)
                    match = comparison.loc[
                        comparison["seed"].eq(seed)
                        & comparison["model"].eq(model)
                        & comparison["daily_review_cap"].eq(k)
                    ]
                    if len(match) != 1:
                        raise AssertionError(f"Comparison row missing/duplicated: seed={seed}, model={model}, k={k}")
                    expected = match.iloc[0]
                    for metric in METRICS:
                        assert_close(actual[metric], float(expected[metric]), f"{seed}/{model}/{k}/{metric}")
                    verified_rows += 1

                if model == PRIMARY_MODEL:
                    selected = selected_mask(frame, 50)
                    prevalence = float(frame["is_fraud"].mean())
                    precision = float(frame.loc[selected, "is_fraud"].mean())
                    prevalence_rows.append({
                        "seed": seed,
                        "transactions": int(len(frame)),
                        "frauds": int(frame["is_fraud"].sum()),
                        "fraud_prevalence": prevalence,
                        "reviews_at_50_per_day": int(selected.sum()),
                        "catboost_history_precision": precision,
                        "precision_lift_over_prevalence": precision / prevalence,
                    })

                if model == "rules":
                    selected = selected_mask(frame, 50)
                    zero = frame["score"].eq(0)
                    rules_tie_rows.append({
                        "seed": seed,
                        "transactions": int(len(frame)),
                        "zero_score_transactions": int(zero.sum()),
                        "zero_score_share": float(zero.mean()),
                        "reviews_at_50_per_day": int(selected.sum()),
                        "selected_zero_score_transactions": int((selected & zero).sum()),
                        "selected_zero_score_share": float((selected & zero).sum() / selected.sum()),
                        "rules_precision": float(frame.loc[selected, "is_fraud"].mean()),
                    })

    prevalence = pd.DataFrame(prevalence_rows)
    rules_ties = pd.DataFrame(rules_tie_rows)
    return {
        "archive": path.name,
        "sha256": actual_sha,
        "verified_comparison_rows": verified_rows,
        "prefix_integrity_passed": True,
        "mean_final_fraud_prevalence": float(prevalence["fraud_prevalence"].mean()),
        "pooled_final_transactions": int(prevalence["transactions"].sum()),
        "pooled_final_frauds": int(prevalence["frauds"].sum()),
        "pooled_final_fraud_prevalence": float(prevalence["frauds"].sum() / prevalence["transactions"].sum()),
        "mean_primary_precision": float(prevalence["catboost_history_precision"].mean()),
        "mean_primary_precision_lift": float(prevalence["precision_lift_over_prevalence"].mean()),
        "rules_selected_zero_score_share_min": float(rules_ties["selected_zero_score_share"].min()),
        "rules_selected_zero_score_share_max": float(rules_ties["selected_zero_score_share"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="Path to the official final ZIP")
    args = parser.parse_args()
    summary = verify(args.archive)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
