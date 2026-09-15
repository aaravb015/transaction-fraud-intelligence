"""One-time reserved-period evaluator for the frozen V2 fraud experiment.

This module is intentionally separate from ``fraud_v2.py``.  It verifies the
frozen engine byte-for-byte, derives the smallest declared continuation patch,
and refuses to materialize days 102--119 unless the explicit command-line
acknowledgement is supplied.  Static checks import and compile this module but
never call :func:`run_official_final_evaluation`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import shutil
import textwrap
import zipfile

import numpy as np
import pandas as pd

import fraud_v2 as frozen


FINAL_VERSION = "0.3.0"
FROZEN_ENGINE_SHA256 = "6debf4f7501ee0f937bd53c9992f8ecc642c946039dd653aa284229078d2cb1e"
FROZEN_IMPLEMENTATION_COMMIT = "a4ab1de96e2d0bed7fa1f19836f9585f3f10344f"
FREEZE_DOCUMENT_COMMIT = "455b37921e90193199543a1207026e68770faeac"
PROTOCOL_COMMIT = "40e90a6af7f4428ed87e56c227f7647a67626de9"
DEVELOPMENT_FINGERPRINT_ARCHIVE = "fraud_v2_20260915T093200_432180Z.zip"
DEVELOPMENT_FINGERPRINT_ARCHIVE_SHA256 = "94ce8bd6b6f0a2490ee1b31c90bbaf1abc90a5433730a0f55a68f20dc3a47b05"
PRIMARY_MODEL = "catboost_history"
FINAL_START = frozen.START + pd.Timedelta(days=102)
FINAL_END = frozen.START + pd.Timedelta(days=120)
EXTENSION_DAYS = 18

# Recorded by the completed full V2 run.  These hashes describe the frozen
# development dataframes, not the ZIP container (whose metadata is timestamped).
EXPECTED_DEVELOPMENT_HASHES = {
    42: "7bfa4146dac935d1778c96ba37588263651b48032c979b941323661968b565a8",
    123: "70a641992dd7c72a73420ee207a1da97472368852ebaef4ed19dac2abe90d626",
    2025: "a8a7e942fad7f3d536e38781d040e8a7bb78686db440e07bedd02f954e35ec6f",
    31415: "9582c0956d0738c841b2c519208a9ff2fe9ca76a7e3d1718dda3f84d0347973b",
    27182: "1ef5dbf2d16a26c7707c7b372e4c8b8156d53ed6794b06f016d2bb8c946c4dda",
}


@dataclass(frozen=True)
class FinalWindow:
    """Only the fields used by the frozen evaluation routine."""

    review_budgets: tuple = (20, 50, 100)
    main_review_budget: int = 50
    validation_at: pd.Timestamp = FINAL_START
    test_at: pd.Timestamp = FINAL_END


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataframe_sha256(frame):
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()


def conditional_extension_probability(probability):
    """Duration-scale a 102-day one-off probability to the reserved 18 days."""
    return 1.0 - (1.0 - float(probability)) ** (EXTENSION_DAYS / 102.0)


def verify_frozen_engine():
    actual = file_sha256(frozen.__file__)
    if actual != FROZEN_ENGINE_SHA256:
        raise RuntimeError(
            "Frozen V2 engine mismatch. Final evaluation is blocked before materialization: "
            + actual
        )
    if frozen.VERSION != "0.2.0":
        raise RuntimeError("Unexpected frozen V2 version.")
    if tuple(frozen.SEEDS) != (42, 123, 2025, 31415, 27182):
        raise RuntimeError("Frozen seed list changed.")
    if tuple(frozen.MODELS) != (
        "rules", "logistic_history", "catboost_current", "catboost_history"
    ):
        raise RuntimeError("Frozen comparator set changed.")
    cfg = frozen.Config()
    if cfg.test_at != FINAL_START or tuple(cfg.review_budgets) != (20, 50, 100):
        raise RuntimeError("Frozen temporal boundary or review budgets changed.")
    return actual


def _continuation_source():
    """Return the reviewed source-derived continuation without executing it.

    The original simulator body remains the source of truth.  Exact, asserted
    textual edits retain rejected boundary spillovers and append new events only
    after every frozen development RNG draw has been consumed.
    """
    source = textwrap.dedent(inspect.getsource(frozen.simulate))
    replacements = []

    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError("Frozen simulator patch anchor changed: " + old.splitlines()[0])
        source = source.replace(old, new, 1)
        replacements.append(old.splitlines()[0])

    replace_once("def simulate(cfg, seed, hardened=True):", "def _simulate_extended(cfg, seed, hardened=True):")
    replace_once(
        "    horizon = dev_days * 86400\n",
        "    horizon = dev_days * 86400\n"
        "    final_horizon = cfg.days * 86400\n"
        "    boundary_spillovers = []\n",
    )
    replace_once(
        "        if sec < 0 or sec >= horizon:\n            return\n",
        "        if sec < 0 or sec >= horizon:\n"
        "            if horizon <= sec < final_horizon:\n"
        "                boundary_spillovers.append((p, sec, amount, device, country, merchant, fraud, scenario, context, fail_p, generator))\n"
        "            return\n",
    )
    replace_once(
        "        travel_country = str(rng.choice([c for c in COUNTRIES if c != p[\"home\"]]))\n",
        "        travel_country = str(rng.choice([c for c in COUNTRIES if c != p[\"home\"]]))\n"
        "        p[\"travel_day\"], p[\"travel_country\"] = travel_day, travel_country\n",
    )
    replace_once(
        "        if rng.random() < 0.35:\n            day = int(rng.integers(10, dev_days))\n",
        "        p[\"had_legitimate_burst\"] = rng.random() < 0.35\n"
        "        if p[\"had_legitimate_burst\"]:\n"
        "            day = int(rng.integers(10, dev_days))\n",
    )
    continuation = r'''
    # Only now, after every frozen development draw, materialize boundary
    # spillovers and draw the declared 18-day continuation.
    def add_final(p, sec, amount, device, country, merchant, fraud, scenario, context, fail_p, generator):
        sec = int(sec)
        if sec < horizon or sec >= final_horizon:
            return
        row = {"customer_id": p["customer_id"], "second": sec,
               "amount": round(float(np.clip(amount, 1, 200000)), 2),
               "device_id": str(device), "country": str(country),
               "merchant_id": str(merchant), "category": merchant_category[str(merchant)],
               "status": "failed" if generator.random() < fail_p else "succeeded",
               "delay": int(generator.integers(2, 121)), "is_fraud": int(fraud),
               "fraud_scenario": scenario, "legitimate_context": context}
        events.append(row)
        if not fraud:
            normal_histories[p["customer_id"]].append(row)

    for args in boundary_spillovers:
        add_final(*args)

    extension_days = cfg.days - dev_days
    phone_probability = 1 - (1 - 0.30) ** (extension_days / dev_days)
    travel_probability = 1 - (1 - 0.25) ** (extension_days / dev_days)
    burst_probability = 1 - (1 - 0.35) ** (extension_days / dev_days)
    fraud_probability = 1 - (1 - cfg.fraud_customer_fraction) ** (extension_days / dev_days)

    # Continuation draw order is frozen: customer order; phone, travel, routine
    # attempts, then burst.  All draws use the already-advanced normal stream.
    for i, p in enumerate(profiles):
        had_phone = p["phone_day"] < dev_days
        if not had_phone:
            p["phone_day"] = (int(rng.integers(dev_days, cfg.days))
                              if rng.random() < phone_probability else cfg.days + 1)
        had_travel = p["travel_day"] < dev_days
        if not had_travel:
            if rng.random() < travel_probability:
                p["travel_day"] = int(rng.integers(dev_days, cfg.days))
                p["travel_country"] = str(rng.choice([c for c in COUNTRIES if c != p["home"]]))
            else:
                p["travel_day"] = cfg.days + 1
        n = int(rng.poisson(p["rate"] * extension_days))
        days = rng.integers(dev_days, cfg.days, n)
        hours = np.mod(rng.normal(p["hour"], 3, n), 24)
        for day, hour in zip(days, hours):
            context, country = "routine", p["home"]
            device = p["device1"] if day >= p["phone_day"] else p["device0"]
            if p["phone_day"] <= day < p["phone_day"] + 5:
                context = "new_phone"
            if p["travel_day"] <= day < p["travel_day"] + 6:
                context, country = "travel", p["travel_country"]
            if rng.random() < 0.025:
                device = "household_%04d" % (i // 4)
            amount = p["base"] * rng.lognormal(0, p["spread"])
            if rng.random() < 0.018:
                amount *= rng.uniform(2, 7)
                context = "large_purchase"
            merchant = rng.choice(p["favourites"] if rng.random() < 0.85 else merchant_ids)
            add_final(p, day * 86400 + int(hour * 3600), amount, device, country,
                      merchant, 0, "legitimate", context, 0.05, rng)
        if not p["had_legitimate_burst"] and rng.random() < burst_probability:
            day = int(rng.integers(dev_days, cfg.days))
            for j in range(int(rng.integers(4, 11))):
                add_final(p, day * 86400 + p["hour"] * 3600 + j * 70,
                          p["base"] * rng.uniform(0.08, 2.2),
                          p["device1"] if day >= p["phone_day"] else p["device0"],
                          p["home"], rng.choice(p["favourites"]), 0, "legitimate",
                          "legitimate_burst", 0.25, rng)

    # New fraud episodes are drawn only for customers not selected in V2.
    # Their scenario cycle continues after the frozen selected cohort.
    selected_set = {int(x) for x in selected}
    new_attack_index = 0
    for index, p in enumerate(profiles):
        if index in selected_set or attack_rng.random() >= fraud_probability:
            continue
        scenario = ("account_takeover", "card_testing", "low_and_slow")[(len(selected_set) + new_attack_index) % 3]
        new_attack_index += 1
        day = int(attack_rng.integers(dev_days, cfg.days))
        mimic = hardened and attack_rng.random() < 0.80
        hour = float(np.mod(attack_rng.normal(p["hour"], 3), 24)) if mimic else float(attack_rng.uniform(0, 24))
        base_sec = day * 86400 + int(hour * 3600)
        previous = [r for r in normal_histories[p["customer_id"]] if r["second"] < base_sec]
        latest = max(previous, key=lambda r: r["second"]) if previous else None
        familiar_device = latest["device_id"] if latest else p["device0"]
        device = familiar_device if attack_rng.random() < (0.85 if hardened else 0.45) else "shared_actor_%03d" % int(attack_rng.integers(0, 30))
        country = p["home"] if attack_rng.random() < (0.92 if hardened else 0.75) else str(attack_rng.choice(COUNTRIES))
        known_merchants = [r["merchant_id"] for r in previous] or list(p["favourites"])
        if scenario == "account_takeover":
            count, step = int(attack_rng.integers(2, 7)), int(attack_rng.integers(120, 1800))
            if mimic:
                step = int(attack_rng.integers(1800, 36000))
            fail_p = 0.05 if mimic else 0.18
        elif scenario == "card_testing":
            count, step = int(attack_rng.integers(5, 13)), int(attack_rng.integers(15, 150))
            if hardened and attack_rng.random() < 0.50:
                step = int(attack_rng.integers(600, 5400))
            fail_p = float(attack_rng.uniform(0.02, 0.12)) if mimic else 0.45
        else:
            count, step, fail_p = int(attack_rng.integers(3, 9)), int(attack_rng.integers(1, 4)) * 86400, 0.06
        for j in range(count):
            event_sec = base_sec + j * step
            if hardened and scenario == "low_and_slow":
                event_sec = max(base_sec, event_sec + int(attack_rng.normal(0, 4 * 3600)))
            if mimic and scenario != "card_testing":
                amount = p["base"] * attack_rng.lognormal(0, p["spread"])
            elif scenario == "account_takeover":
                amount = p["base"] * attack_rng.uniform(1.2, 6)
            elif scenario == "card_testing":
                amount = p["base"] * (attack_rng.lognormal(-0.2, p["spread"]) if mimic and attack_rng.random() < 0.55 else attack_rng.uniform(0.02, 0.35))
            else:
                amount = p["base"] * attack_rng.lognormal(0, p["spread"] * 0.7)
            merchant = attack_rng.choice(known_merchants if attack_rng.random() < (0.80 if hardened else 0.35) else merchant_ids)
            add_final(p, event_sec, amount, device, country, merchant, 1, scenario,
                      "not_applicable", fail_p, attack_rng)
'''
    anchor = "    tx = pd.DataFrame(events)\n"
    replace_once(anchor, continuation + "\n" + anchor)
    # The frozen development guard must not inspect the newly authorized final
    # rows.  Range and integrity guards below replace it for this evaluator.
    replace_once("    guard_development(tx, cfg)\n", "    _final_range_guard(tx, cfg)\n")
    return source, replacements


def _final_range_guard(frame, cfg):
    if "timestamp" not in frame:
        raise ValueError("Timestamp metadata is required.")
    times = frame["timestamp"]
    if times.isna().any() or (times < frozen.START).any() or (times >= FINAL_END).any():
        raise ValueError("Final evaluator produced timestamps outside days 0--119.")


def build_extended_simulator():
    source, replacements = _continuation_source()
    namespace = dict(frozen.__dict__)
    namespace["_final_range_guard"] = _final_range_guard
    exec(compile(source, "<frozen-v2-continuation>", "exec"), namespace)
    return namespace["_simulate_extended"], source, replacements


def verify_static_contract():
    engine_hash = verify_frozen_engine()
    source, replacements = _continuation_source()
    compile(source, "<frozen-v2-continuation>", "exec")
    expected_probabilities = {
        "phone_change": conditional_extension_probability(0.30),
        "travel": conditional_extension_probability(0.25),
        "legitimate_burst": conditional_extension_probability(0.35),
        "fraud_episode": conditional_extension_probability(0.35),
    }
    if len(replacements) != 7:
        raise RuntimeError("Unexpected continuation patch count.")
    return {
        "frozen_engine_sha256": engine_hash,
        "continuation_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "patch_anchor_count": len(replacements),
        "extension_probabilities": expected_probabilities,
        "reserved_period_materialized": False,
        "reserved_period_evaluated": False,
    }


def build_features_authorized(full_tx, cfg):
    """Run the exact frozen feature builder with a strict 120-day range guard."""
    original_guard = frozen.guard_development
    frozen.guard_development = _final_range_guard
    try:
        return frozen.build_features(full_tx, cfg)
    finally:
        frozen.guard_development = original_guard


def prefix_integrity(seed, cfg, full_tx, full_features):
    """Prove exact development row and feature identity before scoring final."""
    _, _, expected_tx = frozen.simulate(cfg, seed)
    prefix = full_tx.loc[full_tx.timestamp < cfg.test_at].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_tx, prefix, check_exact=True)
    expected_hash = EXPECTED_DEVELOPMENT_HASHES[int(seed)]
    actual_hash = dataframe_sha256(expected_tx.set_index("transaction_id"))
    if actual_hash != expected_hash:
        raise RuntimeError("Development dataset does not match the reviewed V2 run for seed %s." % seed)
    expected_features = frozen.build_features(expected_tx, cfg)
    prefix_features = full_features.loc[expected_features.index]
    pd.testing.assert_frame_equal(expected_features, prefix_features, check_exact=True)
    return expected_tx, expected_features, {
        "seed": int(seed),
        "development_rows": len(expected_tx),
        "development_dataset_sha256": actual_hash,
        "expected_development_dataset_sha256": expected_hash,
        "development_rows_exact_match": True,
        "development_features_exact_match": True,
        "first_final_timestamp": str(full_tx.loc[full_tx.timestamp >= cfg.test_at, "timestamp"].min()),
        "last_final_timestamp": str(full_tx.loc[full_tx.timestamp >= cfg.test_at, "timestamp"].max()),
    }


def train_frozen_models(seed, cfg, development_tx, development_features):
    tx = development_tx.set_index("transaction_id")
    masks, split_audit = frozen.temporal_split(tx, cfg)
    state = {
        "seed": int(seed), "tx": tx, "features": development_features,
        "masks": masks, "models": {}, "model_columns": {}, "predictions": {},
    }
    for name in frozen.MODELS:
        if name != "rules":
            state["models"][name], state["model_columns"][name] = frozen.fit_detector(state, cfg, name)
    return state, split_audit


def score_final_period(state, full_tx, full_features):
    final_mask = full_tx.timestamp >= FINAL_START
    tx = full_tx.loc[final_mask].set_index("transaction_id")
    features = full_features.loc[tx.index]
    window = FinalWindow()
    metrics, daily, predictions = [], [], {}
    for name in frozen.MODELS:
        if name == "rules":
            scores = frozen.rule_scores(features)
        else:
            columns = state["model_columns"][name]
            scores = state["models"][name].predict_proba(features[columns])[:, 1]
        m, d, p = frozen.evaluate_scores(tx, scores, name, window)
        metrics.append(m)
        daily.append(d)
        predictions[name] = p
    return pd.concat(metrics, ignore_index=True), pd.concat(daily, ignore_index=True), predictions


def final_context_table(predictions):
    rows = []
    for name, frame in predictions.items():
        for context in ("routine", "travel", "new_phone", "large_purchase", "legitimate_burst"):
            group = frame.loc[frame.is_fraud.eq(0) & frame.legitimate_context.eq(context)]
            picked = group.loc[group.selected_for_review]
            rows.append({
                "model": name, "context": context, "legitimate_count": len(group),
                "legitimate_reviewed": len(picked),
                "alert_rate": frozen.ratio(len(picked), len(group)),
                "score_median": float(group.score.median()),
                "score_q75": float(group.score.quantile(.75)),
                "preselected_primary_model": name == PRIMARY_MODEL,
            })
    return pd.DataFrame(rows)


FINAL_REQUIRED = (
    "final_comparison.csv", "final_daily_metrics.csv", "final_daily_variability_summary.csv",
    "final_scenario_breakdown.csv", "final_legitimate_context_breakdown.csv",
    "prefix_integrity_checks.csv", "training_split_audit.csv", "final_manifest.json",
)


def validate_final_output(out):
    missing = [name for name in FINAL_REQUIRED if not (out / name).is_file()]
    if missing:
        raise AssertionError("Incomplete final output contract: " + ", ".join(missing))
    comparison = pd.read_csv(out / "final_comparison.csv")
    expected = len(frozen.SEEDS) * len(frozen.MODELS) * 3
    if len(comparison) != expected:
        raise AssertionError("Final comparison must contain exactly 60 rows.")
    if set(comparison.seed.astype(int)) != set(frozen.SEEDS):
        raise AssertionError("Final comparison seed grid is incomplete.")
    if set(comparison.model) != set(frozen.MODELS):
        raise AssertionError("Final comparator grid is incomplete.")
    if set(comparison.daily_review_cap.astype(int)) != {20, 50, 100}:
        raise AssertionError("Final review-budget grid is incomplete.")
    if not np.isfinite(comparison[frozen.METRICS].to_numpy()).all():
        raise AssertionError("Final comparison contains non-finite metrics.")
    integrity = pd.read_csv(out / "prefix_integrity_checks.csv")
    if len(integrity) != len(frozen.SEEDS):
        raise AssertionError("Missing prefix-integrity record.")
    if not integrity[["development_rows_exact_match", "development_features_exact_match"]].all().all():
        raise AssertionError("A development prefix check failed.")
    for seed in frozen.SEEDS:
        seed_dir = out / "seeds" / str(seed)
        for name in frozen.MODELS:
            path = seed_dir / (name + "_final_predictions.csv")
            if not path.is_file():
                raise AssertionError("Missing final predictions: " + str(path))
            timestamps = pd.to_datetime(pd.read_csv(path, usecols=["timestamp"]).timestamp, utc=True)
            if (timestamps < FINAL_START).any() or (timestamps >= FINAL_END).any():
                raise AssertionError("Prediction outside the reserved period.")
    manifest = json.loads((out / "final_manifest.json").read_text(encoding="utf-8"))
    if manifest["preselected_primary_model"] != PRIMARY_MODEL:
        raise AssertionError("Primary model changed.")
    if not manifest["locked_test_materialized"] or not manifest["locked_test_evaluated"]:
        raise AssertionError("Final manifest state is invalid.")
    return {"required_files": len(FINAL_REQUIRED), "result_grid_rows": expected,
            "seed_prediction_sets": len(frozen.SEEDS), "contract_passed": True}


def run_official_final_evaluation(work_dir, cfg):
    static = verify_static_contract()
    simulator, continuation_source, _ = build_extended_simulator()
    work = Path(work_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = work / ("fraud_final_" + run_id)
    out.mkdir(parents=True, exist_ok=False)
    all_metrics, all_daily, all_scenarios, all_contexts = [], [], [], []
    integrity_rows, split_rows = [], []
    for seed in frozen.SEEDS:
        print("Starting frozen final-evaluation seed", seed, flush=True)
        _, _, full_tx = simulator(cfg, seed, hardened=True)
        full_features = build_features_authorized(full_tx, cfg)
        development_tx, development_features, integrity = prefix_integrity(
            seed, cfg, full_tx, full_features
        )
        integrity_rows.append(integrity)
        state, split_audit = train_frozen_models(
            seed, cfg, development_tx, development_features
        )
        split_rows.append(split_audit.assign(seed=seed))
        metrics, daily, predictions = score_final_period(state, full_tx, full_features)
        all_metrics.append(metrics.assign(seed=seed))
        all_daily.append(daily.assign(seed=seed))
        all_scenarios.append(frozen.scenario_table(predictions).assign(seed=seed))
        all_contexts.append(final_context_table(predictions).assign(seed=seed))
        seed_out = out / "seeds" / str(seed)
        seed_out.mkdir(parents=True)
        for name, frame in predictions.items():
            frame.to_csv(seed_out / (name + "_final_predictions.csv"), index_label="transaction_id")
        print("Completed frozen final-evaluation seed", seed, flush=True)

    comparison = pd.concat(all_metrics, ignore_index=True)
    daily = pd.concat(all_daily, ignore_index=True)
    comparison.to_csv(out / "final_comparison.csv", index=False)
    daily.to_csv(out / "final_daily_metrics.csv", index=False)
    pd.concat(
        [frozen.daily_summary(group).assign(seed=seed) for seed, group in daily.groupby("seed")],
        ignore_index=True,
    ).to_csv(out / "final_daily_variability_summary.csv", index=False)
    pd.concat(all_scenarios, ignore_index=True).to_csv(out / "final_scenario_breakdown.csv", index=False)
    pd.concat(all_contexts, ignore_index=True).to_csv(out / "final_legitimate_context_breakdown.csv", index=False)
    pd.DataFrame(integrity_rows).to_csv(out / "prefix_integrity_checks.csv", index=False)
    pd.concat(split_rows, ignore_index=True).to_csv(out / "training_split_audit.csv", index=False)
    manifest = {
        "project": "transaction-fraud-intelligence",
        "final_evaluator_version": FINAL_VERSION,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "frozen_engine_sha256": FROZEN_ENGINE_SHA256,
        "final_engine_sha256": file_sha256(__file__),
        "continuation_source_sha256": hashlib.sha256(continuation_source.encode()).hexdigest(),
        "frozen_implementation_commit": FROZEN_IMPLEMENTATION_COMMIT,
        "freeze_document_commit": FREEZE_DOCUMENT_COMMIT,
        "protocol_commit": PROTOCOL_COMMIT,
        "development_fingerprint_archive": DEVELOPMENT_FINGERPRINT_ARCHIVE,
        "development_fingerprint_archive_sha256": DEVELOPMENT_FINGERPRINT_ARCHIVE_SHA256,
        "seed_list": list(frozen.SEEDS), "models": list(frozen.MODELS),
        "review_budgets": list(cfg.review_budgets), "headline_review_budget": 50,
        "preselected_primary_model": PRIMARY_MODEL,
        "feature_list": list(frozen.FEATURES), "current_feature_list": list(frozen.CURRENT_FEATURES),
        "rule_weights": dict(frozen.RULE_WEIGHTS), "configuration": cfg.__dict__,
        "reserved_period": {"start": str(FINAL_START), "end_exclusive": str(FINAL_END)},
        "expected_development_hashes": EXPECTED_DEVELOPMENT_HASHES,
        "prefix_integrity_passed_for_all_seeds": True,
        "locked_test_materialized": True, "locked_test_evaluated": True,
        "official_evaluation_count_for_this_run": 1,
        "selection_after_final_outcomes": False,
        "longitudinal_challenger_promoted": False,
        "synthetic_data_only": True,
        "attempted_value_is_not_prevented_loss": True,
        "static_preflight": static,
    }
    (out / "final_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    check = validate_final_output(out)
    (out / "final_output_contract_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    archive = Path(shutil.make_archive(str(out), "zip", root_dir=out))
    with zipfile.ZipFile(archive) as z:
        if not set(FINAL_REQUIRED) <= set(z.namelist()):
            raise AssertionError("Final ZIP is incomplete.")
    return out, archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--static-check", action="store_true")
    parser.add_argument("--official-final-evaluation", action="store_true")
    args = parser.parse_args()
    if args.static_check:
        print(json.dumps(verify_static_contract(), indent=2))
        return
    if not args.official_final_evaluation:
        raise SystemExit(
            "Reserved period not materialized. Supply --official-final-evaluation only for the one official run."
        )
    cfg = frozen.Config(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    if cfg.mode != "final_evaluation":
        raise SystemExit("Final evaluation requires mode=final_evaluation.")
    out, archive = run_official_final_evaluation(args.work_dir, cfg)
    record = {"output_directory": str(out), "archive": str(archive)}
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.work_dir) / "final_result.json").write_text(json.dumps(record), encoding="utf-8")
    print("FINAL OUTPUT CONTRACT PASSED:", archive, flush=True)


if __name__ == "__main__":
    main()
