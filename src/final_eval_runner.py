"""One-time reserved-period evaluator for the frozen V2 fraud experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from final_eval_continuation import load_module, make_final_engine_source

RUNNER_VERSION = "0.1.0"
FROZEN_IMPLEMENTATION_COMMIT = "a4ab1de96e2d0bed7fa1f19836f9585f3f10344f"
FREEZE_DOCUMENT_COMMIT = "455b37921e90193199543a1207026e68770faeac"
FROZEN_ENGINE_SHA256 = "6debf4f7501ee0f937bd53c9992f8ecc642c946039dd653aa284229078d2cb1e"
FROZEN_RESULTS_SHA256 = "650d3fbbce83ae82f0d821e73f30174998a2956e347dab7bc74f5bb2c41ae58a"
ACK = "EVALUATE_RESERVED_FINAL_TEST_ONCE"
PRIMARY_MODEL = "catboost_history"
MODELS = ("rules", "logistic_history", "catboost_current", "catboost_history")
SEEDS = (42, 123, 2025, 31415, 27182)
METRICS = (
    "average_precision", "roc_auc", "review_count", "fraud_reviewed", "precision",
    "fraud_recall", "fraud_attempt_value_capture", "legitimate_reviewed", "false_positive_rate",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def final_guard(engine):
    def guard(frame, cfg):
        if "timestamp" not in frame:
            raise ValueError("Timestamp metadata is required.")
        t = frame["timestamp"]
        end = engine.START + pd.Timedelta(days=cfg.days)
        if t.isna().any() or (t < engine.START).any() or (t >= end).any():
            raise ValueError("Timestamp outside the frozen 120-day horizon.")
    return guard


def ratio(a, b):
    return float(a / b) if b else np.nan


def select_daily(frame, k):
    ordered = frame.reset_index().sort_values(
        ["day", "score", "transaction_id"], ascending=[True, False, True]
    )
    chosen = ordered.groupby("day", sort=False).head(k).transaction_id
    return pd.Series(frame.index.isin(chosen), index=frame.index)


def evaluate(engine, tx, scores, name, cfg):
    frame = tx[["timestamp", "customer_id", "amount", "is_fraud", "fraud_scenario",
                "legitimate_context", "category", "country"]].copy()
    frame["score"] = np.asarray(scores, dtype=float)
    frame["model"] = name
    if not np.isfinite(frame.score).all() or frame.is_fraud.nunique() != 2:
        raise ValueError("Invalid final scoring frame.")
    frame["day"] = frame.timestamp.dt.floor("D")
    fraud_n = int(frame.is_fraud.sum())
    legit_n = int(frame.is_fraud.eq(0).sum())
    fraud_value = float(frame.loc[frame.is_fraud.eq(1), "amount"].sum())
    ap = float(average_precision_score(frame.is_fraud, frame.score))
    auc = float(roc_auc_score(frame.is_fraud, frame.score))
    rows, daily = [], []
    calendar = pd.date_range(cfg.test_at, engine.START + pd.Timedelta(days=cfg.days), inclusive="left", freq="D")
    for k in cfg.review_budgets:
        selected = select_daily(frame, k)
        picked = frame.loc[selected]
        hits = picked.loc[picked.is_fraud.eq(1)]
        false_n = int(picked.is_fraud.eq(0).sum())
        rows.append({
            "model": name, "daily_review_cap": k, "average_precision": ap, "roc_auc": auc,
            "review_count": len(picked), "fraud_reviewed": len(hits),
            "precision": ratio(len(hits), len(picked)), "fraud_recall": ratio(len(hits), fraud_n),
            "fraud_attempt_value_capture": ratio(float(hits.amount.sum()), fraud_value),
            "legitimate_reviewed": false_n, "false_positive_rate": ratio(false_n, legit_n),
        })
        for day in calendar:
            group = frame.loc[frame.day.eq(day)]
            reviewed = group.loc[selected.loc[group.index]]
            hit = reviewed.loc[reviewed.is_fraud.eq(1)]
            daily.append({
                "model": name, "daily_review_cap": k, "day": str(day),
                "transactions": len(group), "frauds": int(group.is_fraud.sum()),
                "reviews": len(reviewed), "fraud_reviewed": len(hit),
                "legitimate_reviewed": int(reviewed.is_fraud.eq(0).sum()),
                "precision": ratio(len(hit), len(reviewed)),
                "recall": ratio(len(hit), int(group.is_fraud.sum())),
                "false_positive_rate": ratio(int(reviewed.is_fraud.eq(0).sum()), int(group.is_fraud.eq(0).sum())),
                "fraud_attempt_value_capture": ratio(float(hit.amount.sum()), float(group.loc[group.is_fraud.eq(1), "amount"].sum())),
            })
        if k == cfg.main_review_budget:
            frame["selected_for_review"] = selected
            frame["action"] = np.where(selected, "REVIEW", "NO_REVIEW")
    return pd.DataFrame(rows), pd.DataFrame(daily), frame


def scenario_table(predictions):
    rows = []
    for name, frame in predictions.items():
        for scenario, group in frame.loc[frame.is_fraud.eq(1)].groupby("fraud_scenario"):
            picked = group.loc[group.selected_for_review]
            rows.append({
                "model": name, "scenario": scenario, "fraud_count": len(group),
                "fraud_reviewed": len(picked), "recall": ratio(len(picked), len(group)),
                "attempt_value_capture": ratio(float(picked.amount.sum()), float(group.amount.sum())),
            })
    return pd.DataFrame(rows)


def context_table(predictions):
    rows = []
    for name, frame in predictions.items():
        for context in ("routine", "travel", "new_phone", "large_purchase", "legitimate_burst"):
            group = frame.loc[frame.is_fraud.eq(0) & frame.legitimate_context.eq(context)]
            picked = group.loc[group.selected_for_review]
            rows.append({
                "model": name, "context": context, "legitimate_count": len(group),
                "legitimate_reviewed": len(picked), "alert_rate": ratio(len(picked), len(group)),
                "score_median": float(group.score.median()) if len(group) else np.nan,
                "score_q75": float(group.score.quantile(.75)) if len(group) else np.nan,
            })
    return pd.DataFrame(rows)


def aggregate(seed_frames):
    raw = pd.concat(seed_frames, ignore_index=True)
    raw.insert(0, "row_type", "seed")
    summary = []
    for (model, k), group in raw.groupby(["model", "daily_review_cap"], sort=False):
        for stat in ("mean", "std", "min", "max"):
            row = {"row_type": stat, "seed": np.nan, "model": model, "daily_review_cap": k}
            row.update({metric: float(getattr(group[metric], stat)()) for metric in METRICS})
            summary.append(row)
    return pd.concat([raw, pd.DataFrame(summary)], ignore_index=True)


def chart(summary, out):
    table = summary.loc[(summary.row_type == "mean") & (summary.daily_review_cap == 50)].set_index("model")
    fig, ax = plt.subplots(figsize=(8, 4))
    table.reindex(MODELS)["fraud_recall"].plot.bar(ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean final-period fraud recall")
    ax.set_title("Frozen models — reserved final period — 50 reviews/day")
    fig.tight_layout()
    fig.savefig(out / "final_recall_50_per_day.png", dpi=150)
    plt.close(fig)


def run(args):
    if args.ack != ACK:
        raise SystemExit("Explicit acknowledgement missing; final period not evaluated.")
    frozen_bytes = Path(args.frozen_engine).read_bytes()
    actual_hash = sha256(frozen_bytes)
    if actual_hash != FROZEN_ENGINE_SHA256:
        raise SystemExit("Frozen engine hash mismatch; final period not evaluated.")

    frozen_source = frozen_bytes.decode("utf-8")
    frozen = load_module("fraud_v2_frozen", frozen_source)
    if tuple(frozen.SEEDS) != SEEDS or tuple(frozen.MODELS) != MODELS:
        raise SystemExit("Frozen V2 contract mismatch.")

    final_engine = load_module("fraud_v2_final", make_final_engine_source(frozen_source))
    final_engine.guard_development = final_guard(final_engine)
    cfg = final_engine.Config(customers=args.customers, boosting_iterations=args.iterations,
                              threads=args.threads, mode="final_evaluation")
    frozen_cfg = frozen.Config(customers=args.customers, boosting_iterations=args.iterations,
                               threads=args.threads, mode="development")
    out = Path(args.work_dir) / ("fraud_final_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(parents=True, exist_ok=False)

    seed_metrics, seed_daily, scenarios, contexts, checks = [], [], [], [], []
    for seed in SEEDS:
        print("Final seed", seed, "— verifying frozen prefix before scoring", flush=True)
        _, _, frozen_dev = frozen.simulate(frozen_cfg, seed)
        _, _, full = final_engine.simulate(cfg, seed)
        dev_prefix = full.loc[full.timestamp < cfg.test_at].reset_index(drop=True)
        pd.testing.assert_frame_equal(frozen_dev.reset_index(drop=True), dev_prefix, check_exact=True)

        frozen_features = frozen.build_features(frozen_dev, frozen_cfg)
        full_features = final_engine.build_features(full, cfg)
        pd.testing.assert_frame_equal(frozen_features, full_features.loc[frozen_dev.transaction_id], check_exact=True)

        tx = full.set_index("transaction_id")
        train = (tx.timestamp < cfg.fit_at) & (tx.label_available_at <= cfg.fit_at)
        early = ((tx.timestamp >= cfg.fit_at) & (tx.timestamp < cfg.validation_at)
                 & (tx.label_available_at <= cfg.validation_at))
        final_mask = (tx.timestamp >= cfg.test_at) & (tx.timestamp < final_engine.START + pd.Timedelta(days=cfg.days))
        if any(tx.loc[m].is_fraud.nunique() != 2 for m in (train, early, final_mask)):
            raise RuntimeError("A frozen split lacks both classes; stop rather than skip.")

        state = {"seed": seed, "tx": tx, "features": full_features,
                 "masks": {"train": train, "early_stop": early}, "models": {}, "model_columns": {}}
        for name in MODELS:
            if name != "rules":
                state["models"][name], state["model_columns"][name] = final_engine.fit_detector(state, cfg, name)

        final_tx = tx.loc[final_mask]
        final_f = full_features.loc[final_mask]
        predictions, parts, daily_parts = {}, [], []
        for name in MODELS:
            scores = (final_engine.rule_scores(final_f) if name == "rules" else
                      state["models"][name].predict_proba(final_f[state["model_columns"][name]])[:, 1])
            m, d, p = evaluate(final_engine, final_tx, scores, name, cfg)
            m["seed"], d["seed"] = seed, seed
            parts.append(m)
            daily_parts.append(d)
            predictions[name] = p

        metrics = pd.concat(parts, ignore_index=True)
        daily = pd.concat(daily_parts, ignore_index=True)
        seed_metrics.append(metrics)
        seed_daily.append(daily)
        s = scenario_table(predictions).assign(seed=seed)
        c = context_table(predictions).assign(seed=seed)
        scenarios.append(s)
        contexts.append(c)

        seed_out = out / "seeds" / str(seed)
        seed_out.mkdir(parents=True)
        metrics.to_csv(seed_out / "final_comparison.csv", index=False)
        daily.to_csv(seed_out / "final_daily_metrics.csv", index=False)
        s.to_csv(seed_out / "final_scenario_breakdown.csv", index=False)
        c.to_csv(seed_out / "final_context_breakdown.csv", index=False)
        for name, frame in predictions.items():
            frame.to_csv(seed_out / f"{name}_final_predictions.csv", index_label="transaction_id")
        state["models"][PRIMARY_MODEL].save_model(str(seed_out / "catboost_history_frozen_final.cbm"))
        checks.append({
            "seed": seed, "development_rows": len(frozen_dev), "final_rows": int(final_mask.sum()),
            "final_frauds": int(tx.loc[final_mask, "is_fraud"].sum()),
            "development_prefix_exact": True, "development_features_exact": True,
            "first_final_event": str(tx.loc[final_mask, "timestamp"].min()),
            "last_final_event": str(tx.loc[final_mask, "timestamp"].max()),
        })
        print("Seed", seed, "complete — frozen development prefix matched exactly", flush=True)

    robustness = aggregate(seed_metrics)
    robustness.to_csv(out / "final_robustness_summary.csv", index=False)
    pd.concat(seed_daily, ignore_index=True).to_csv(out / "final_daily_metrics_all_seeds.csv", index=False)
    pd.concat(scenarios, ignore_index=True).to_csv(out / "final_scenario_breakdown_all_seeds.csv", index=False)
    pd.concat(contexts, ignore_index=True).to_csv(out / "final_context_breakdown_all_seeds.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "final_prefix_checks.csv", index=False)
    chart(robustness, out)

    primary = robustness.loc[(robustness.row_type == "mean") &
                             (robustness.model == PRIMARY_MODEL) &
                             (robustness.daily_review_cap == 50)]
    primary.to_csv(out / "primary_model_final_summary.csv", index=False)

    manifest = {
        "project": "transaction-fraud-intelligence",
        "runner_version": RUNNER_VERSION,
        "frozen_implementation_commit": FROZEN_IMPLEMENTATION_COMMIT,
        "freeze_document_commit": FREEZE_DOCUMENT_COMMIT,
        "frozen_engine_sha256": actual_hash,
        "frozen_development_results_sha256": FROZEN_RESULTS_SHA256,
        "configuration": asdict(cfg),
        "seed_list": list(SEEDS), "model_list": list(MODELS),
        "primary_model_selected_before_final_test": PRIMARY_MODEL,
        "review_budgets": list(cfg.review_budgets), "main_review_budget": cfg.main_review_budget,
        "fit_at": str(cfg.fit_at), "early_stop_boundary": str(cfg.validation_at),
        "reserved_final_start": str(cfg.test_at),
        "reserved_final_end": str(final_engine.START + pd.Timedelta(days=cfg.days)),
        "continuation_rule": "FINAL_EVALUATION_CONTINUATION_V1",
        "development_model_tuning_after_freeze": False,
        "longitudinal_challenger_promoted": False,
        "seed_checks": checks,
        "locked_test_materialized": True,
        "locked_test_evaluated": True,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation_note": "Synthetic reserved-period evidence; not real-world performance.",
    }
    (out / "final_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    required = {
        "final_robustness_summary.csv", "final_daily_metrics_all_seeds.csv",
        "final_scenario_breakdown_all_seeds.csv", "final_context_breakdown_all_seeds.csv",
        "final_prefix_checks.csv", "primary_model_final_summary.csv",
        "final_recall_50_per_day.png", "final_manifest.json",
    }
    if [x for x in required if not (out / x).is_file()]:
        raise RuntimeError("Final output contract incomplete.")
    rows = robustness.loc[robustness.row_type.eq("seed")]
    if len(rows) != len(SEEDS) * len(MODELS) * len(cfg.review_budgets):
        raise RuntimeError("Final 5×4×3 result grid incomplete.")
    if not pd.DataFrame(checks)[["development_prefix_exact", "development_features_exact"]].all().all():
        raise RuntimeError("Frozen development prefix changed; result invalid.")

    archive = Path(shutil.make_archive(str(out), "zip", root_dir=out))
    with zipfile.ZipFile(archive) as zf:
        if not required <= set(zf.namelist()):
            raise RuntimeError("Final ZIP verification failed.")
    result = {"output_directory": str(out), "archive": str(archive)}
    (Path(args.work_dir) / "final_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_EVALUATION_COMPLETE", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-engine", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--customers", type=int, default=1200)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ack", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
