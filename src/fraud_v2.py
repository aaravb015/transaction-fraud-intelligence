"""Self-contained V2 experiment engine, embedded verbatim in the Colab notebook.

Only development-period events are materialized. All evaluation entry points
enforce the reserved-period boundary before accessing labels or model inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import importlib.metadata
from itertools import groupby
import json
import math
from pathlib import Path
import platform
import shutil
import warnings
import zipfile

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

VERSION = "0.2.0"
SEEDS = (42, 123, 2025, 31415, 27182)
START = pd.Timestamp("2025-01-01", tz="UTC")
CATEGORIES = ["groceries", "retail", "dining", "travel", "digital", "utilities"]
COUNTRIES = ["IN", "SG", "AE", "GB", "US"]
MODELS = ("rules", "logistic_history", "catboost_current", "catboost_history")
CORE_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "matplotlib", "joblib")
OBSERVABLE_COLUMNS = [
    "transaction_id", "timestamp", "customer_id", "merchant_id", "device_id",
    "amount", "country", "category", "status", "outcome_available_at",
]
NUMERIC_FEATURES = [
    "log_amount", "hour_sin", "hour_cos", "prior_count", "prior_count_30d",
    "attempts_1h", "attempts_24h", "log_attempt_value_24h", "log_prior_median_30d",
    "amount_ratio_30d", "amount_z_30d", "history_days", "hours_since_previous",
    "new_device", "new_country", "new_merchant", "known_failures_1h",
    "prior_accounts_on_device", "hour_deviation",
]
CATEGORICAL_FEATURES = ["category", "country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CURRENT_FEATURES = ["log_amount", "hour_sin", "hour_cos", "category", "country"]
FEATURE_FAMILIES = {
    "transaction_amount": ["log_amount"],
    "velocity_recency": ["attempts_1h", "attempts_24h", "log_attempt_value_24h", "hours_since_previous", "known_failures_1h"],
    "novelty": ["new_device", "new_country", "new_merchant"],
    "historical_baselines": ["prior_count", "prior_count_30d", "log_prior_median_30d", "amount_ratio_30d", "amount_z_30d", "history_days"],
    "device_network": ["prior_accounts_on_device"],
    "circadian": ["hour_sin", "hour_cos", "hour_deviation"],
}
LONGITUDINAL_CANDIDATES = [
    "count_ratio_7d_prior30", "count_ratio_14d_prior30", "count_ratio_30d_prior30",
    "attempt_value_ratio_7d_prior30", "attempt_value_ratio_14d_prior30", "attempt_value_ratio_30d_prior30",
    "mean_ticket_ratio_7d_prior30", "median_ticket_ratio_7d_prior30",
    "new_merchants_7d", "new_merchants_14d", "merchant_diversity_shift",
    "category_concentration_shift", "new_category",
]
DIAGNOSTIC_FEATURES = [
    "amount_ratio_30d", "amount_z_30d", "hours_since_previous",
    "attempts_7d", "attempts_14d", "prior_count_30d",
    "attempt_value_7d", "attempt_value_14d", "attempt_value_30d",
    "new_merchant", "prior_accounts_on_device", "new_country", "hour_deviation",
    "category_concentration_7d",
] + LONGITUDINAL_CANDIDATES
SEPARATION_FEATURES = ["hour_deviation", "hours_since_previous", "new_merchant",
                       "prior_accounts_on_device", "attempts_1h", "new_device",
                       "new_country", "amount_ratio_30d"]
RULE_WEIGHTS = {"unusual_amount": 25, "new_device_high_amount": 25, "attempt_burst": 20,
                "known_failure_sequence": 15, "shared_device_new_country": 15}
METRICS = ["average_precision", "roc_auc", "review_count", "fraud_reviewed", "precision",
           "fraud_recall", "fraud_attempt_value_capture", "legitimate_reviewed", "false_positive_rate"]


@dataclass(frozen=True)
class Config:
    customers: int = 1200
    days: int = 120
    merchants: int = 240
    label_delay_days: int = 7
    fraud_customer_fraction: float = 0.35
    review_budgets: tuple = (20, 50, 100)
    main_review_budget: int = 50
    boosting_iterations: int = 600
    threads: int = 2
    mode: str = "development"

    def __post_init__(self):
        if self.days != 120 or tuple(self.review_budgets) != (20, 50, 100):
            raise ValueError("V2 fixes the timeline and review budgets; change the protocol before experimentation.")
        if self.customers < 100 or self.merchants < 6:
            raise ValueError("The declared experiment requires at least 100 customers and six merchants.")

    @property
    def fit_at(self):
        return START + pd.Timedelta(days=int(self.days * 0.60))

    @property
    def validation_at(self):
        return START + pd.Timedelta(days=int(self.days * 0.72))

    @property
    def test_at(self):
        return START + pd.Timedelta(days=int(self.days * 0.85))


def guard_development(frame, cfg):
    """Check timestamp metadata before any access to labels or predictors."""
    if "timestamp" not in frame:
        raise ValueError("Timestamp metadata is required to enforce the final-test lock.")
    times = frame["timestamp"]
    if times.isna().any() or (times < START).any() or (times >= cfg.test_at).any():
        raise ValueError("Reserved final-period or invalid timestamps are forbidden in V2.")


def guard_validation(frame, cfg):
    guard_development(frame, cfg)
    if (frame.timestamp < cfg.validation_at).any():
        raise ValueError("Evaluation accepts development-validation rows only.")


def simulate(cfg, seed, hardened=True):
    """Generate only days [0,102); the final period has no events or labels here.

    Normal and fraud RNG streams are separate so the V1-like reference preserves
    the normal world while varying the explicitly documented fraud mechanisms.
    """
    normal_seq, fraud_seq = np.random.SeedSequence(int(seed)).spawn(2)
    rng, attack_rng = np.random.default_rng(normal_seq), np.random.default_rng(fraud_seq)
    dev_days = int((cfg.test_at - START).days)
    horizon = dev_days * 86400
    merchant_ids = np.array(["M%04d" % i for i in range(cfg.merchants)])
    merchants = pd.DataFrame({"merchant_id": merchant_ids,
        "category": [CATEGORIES[i % len(CATEGORIES)] for i in range(cfg.merchants)]})
    merchant_category = dict(zip(merchants.merchant_id, merchants.category))
    profiles, customer_rows, events = [], [], []
    normal_histories = defaultdict(list)

    def add(p, sec, amount, device, country, merchant, fraud, scenario, context, fail_p, generator):
        # Filter before status/label creation, not after reading a test dataframe.
        sec = int(sec)
        if sec < 0 or sec >= horizon:
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

    for i in range(cfg.customers):
        cid = "C%05d" % i
        p = {"customer_id": cid,
             "base": float(np.clip(rng.lognormal(np.log(850), 0.8), 120, 7000)),
             "spread": float(rng.uniform(0.3, 0.85)),
             "rate": float(np.clip(rng.gamma(2, 0.25), 0.12, 1.3)),
             "hour": int(rng.integers(7, 24)),
             "home": str(rng.choice(COUNTRIES, p=[0.8, 0.06, 0.06, 0.04, 0.04])),
             "favourites": rng.choice(merchant_ids, 6, replace=False),
             "device0": "phone_" + cid, "device1": "replacement_" + cid}
        p["phone_day"] = int(rng.integers(25, dev_days - 10)) if rng.random() < 0.30 else dev_days + 1
        travel_day = int(rng.integers(15, dev_days - 8)) if rng.random() < 0.25 else dev_days + 1
        travel_country = str(rng.choice([c for c in COUNTRIES if c != p["home"]]))
        profiles.append(p)
        customer_rows.append({"customer_id": cid, "simulator_base_amount": p["base"],
                              "simulator_daily_rate": p["rate"], "simulator_hour": p["hour"]})
        n = int(rng.poisson(p["rate"] * dev_days))
        days = rng.integers(0, dev_days, n)
        hours = np.mod(rng.normal(p["hour"], 3, n), 24)
        for day, hour in zip(days, hours):
            context, country = "routine", p["home"]
            device = p["device1"] if day >= p["phone_day"] else p["device0"]
            if p["phone_day"] <= day < p["phone_day"] + 5:
                context = "new_phone"
            if travel_day <= day < travel_day + 6:
                context, country = "travel", travel_country
            if rng.random() < 0.025:
                device = "household_%04d" % (i // 4)
            amount = p["base"] * rng.lognormal(0, p["spread"])
            if rng.random() < 0.018:
                amount *= rng.uniform(2, 7)
                context = "large_purchase"
            merchant = rng.choice(p["favourites"] if rng.random() < 0.85 else merchant_ids)
            add(p, day * 86400 + int(hour * 3600), amount, device, country,
                merchant, 0, "legitimate", context, 0.05, rng)
        if rng.random() < 0.35:
            day = int(rng.integers(10, dev_days))
            for j in range(int(rng.integers(4, 11))):
                add(p, day * 86400 + p["hour"] * 3600 + j * 70,
                    p["base"] * rng.uniform(0.08, 2.2),
                    p["device1"] if day >= p["phone_day"] else p["device0"],
                    p["home"], rng.choice(p["favourites"]), 0, "legitimate", "legitimate_burst", 0.25, rng)

    selected = attack_rng.choice(len(profiles), max(3, int(cfg.customers * cfg.fraud_customer_fraction)), replace=False)
    for k, index in enumerate(selected):
        p = profiles[int(index)]
        scenario = ("account_takeover", "card_testing", "low_and_slow")[k % 3]
        day = int(attack_rng.integers(12, dev_days))
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
                # Per-event jitter prevents exact day/hour spacing being a tag.
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
            add(p, event_sec, amount, device, country, merchant, 1, scenario, "not_applicable", fail_p, attack_rng)
    tx = pd.DataFrame(events)
    tx["timestamp"] = START + pd.to_timedelta(tx.pop("second"), unit="s")
    tx["outcome_available_at"] = tx.timestamp + pd.to_timedelta(tx.pop("delay"), unit="s")
    tx["label_available_at"] = tx.timestamp + pd.Timedelta(days=cfg.label_delay_days)
    tx = tx.sort_values("timestamp", kind="stable").reset_index(drop=True)
    tx.insert(0, "transaction_id", ["T%08d" % i for i in range(len(tx))])
    guard_development(tx, cfg)
    assert tx.transaction_id.is_unique and tx.timestamp.is_monotonic_increasing
    assert (tx.amount > 0).all()
    assert (tx.outcome_available_at > tx.timestamp).all()
    return pd.DataFrame(customer_rows), merchants, tx


def concentration(rows):
    counts = Counter(r[3] for r in rows)
    return sum((n / len(rows)) ** 2 for n in counts.values()) if rows else np.nan


def build_features(tx, cfg):
    guard_development(tx, cfg)
    events = tx[OBSERVABLE_COLUMNS].sort_values(["timestamp", "transaction_id"])
    state, device_accounts, pending = {}, defaultdict(set), []
    known_failures, records = defaultdict(deque), []
    hour_ns, day_ns = 3600 * 10**9, 86400 * 10**9
    for timestamp, batch_iter in groupby(events.itertuples(index=False), key=lambda r: r.timestamp):
        batch, t = list(batch_iter), int(timestamp.value)
        while pending and pending[0][0] <= t:
            available, event_id, cid = heapq.heappop(pending)
            known_failures[cid].append(available)
        flags = {}
        for r in batch:
            cid = r.customer_id
            if cid not in state:
                state[cid] = {"history": deque(), "count": 0, "devices": set(),
                              "countries": set(), "merchants": set(), "categories": set(),
                              "first": t, "last": None, "sin_sum": 0., "cos_sum": 0.}
            s = state[cid]
            while s["history"] and s["history"][0][0] < t - 60 * day_ns:
                s["history"].popleft()
            h = list(s["history"])
            h30 = [x for x in h if x[0] >= t - 30 * day_ns]
            amounts = np.array([x[1] for x in h30], dtype=float)
            median = float(np.median(amounts)) if len(amounts) else np.nan
            h24 = [x for x in h30 if x[0] >= t - day_ns]
            f = known_failures[cid]
            while f and f[0] < t - hour_ns:
                f.popleft()
            angle = 2 * np.pi * (timestamp.hour + timestamp.minute / 60) / 24
            sine, cosine = float(np.sin(angle)), float(np.cos(angle))
            norm, hour_dev = math.hypot(s["sin_sum"], s["cos_sum"]), np.nan
            if s["count"] >= 3 and norm > 1e-8:
                hour_dev = float(np.arccos(np.clip((sine*s["sin_sum"]+cosine*s["cos_sum"])/norm, -1, 1))*12/np.pi)
            history_days = (t - s["first"]) / day_ns
            flags[r.transaction_id] = int(r.merchant_id not in s["merchants"])
            row = {
                "transaction_id": r.transaction_id, "log_amount": float(np.log1p(r.amount)),
                "hour_sin": sine, "hour_cos": cosine, "prior_count": s["count"],
                "prior_count_30d": len(h30), "attempts_1h": sum(x[0] >= t-hour_ns for x in h30),
                "attempts_24h": len(h24), "log_attempt_value_24h": float(np.log1p(sum(x[1] for x in h24))),
                "log_prior_median_30d": float(np.log1p(median)) if len(h30) else np.nan,
                "amount_ratio_30d": float(r.amount/median) if len(h30) >= 3 else np.nan,
                "amount_z_30d": float((r.amount-amounts.mean())/(amounts.std()+50)) if len(h30) >= 3 else np.nan,
                "history_days": history_days,
                "hours_since_previous": (t-s["last"])/hour_ns if s["last"] is not None else np.nan,
                "new_device": int(r.device_id not in s["devices"]),
                "new_country": int(r.country not in s["countries"]),
                "new_merchant": flags[r.transaction_id], "known_failures_1h": len(f),
                "prior_accounts_on_device": len(device_accounts[r.device_id]), "hour_deviation": hour_dev,
                "category": str(r.category), "country": str(r.country),
                "new_category": int(r.category not in s["categories"]),
                "attempt_value_30d": float(sum(x[1] for x in h30)),
            }
            for days in (7, 14, 30):
                recent = [x for x in h if x[0] >= t-days*day_ns]
                # Disjoint baseline [t-(days+30), t-days); no overlapping denominator.
                baseline = [x for x in h if t-(days+30)*day_ns <= x[0] < t-days*day_ns]
                rv, bv = [x[1] for x in recent], [x[1] for x in baseline]
                reliable = len(baseline) >= 3 and history_days >= days + 30
                row["attempts_%dd" % days] = len(recent)
                row["attempt_value_%dd" % days] = float(sum(rv))
                row["count_ratio_%dd_prior30" % days] = len(recent)/(len(baseline)*days/30) if reliable else np.nan
                row["attempt_value_ratio_%dd_prior30" % days] = sum(rv)/(sum(bv)*days/30) if reliable else np.nan
                row["new_merchants_%dd" % days] = len({x[2] for x in recent if x[4]})
                if days == 7:
                    enough = reliable and len(recent) >= 3
                    row["mean_ticket_ratio_7d_prior30"] = float(np.mean(rv)/np.mean(bv)) if enough else np.nan
                    row["median_ticket_ratio_7d_prior30"] = float(np.median(rv)/np.median(bv)) if enough else np.nan
                    row["merchant_diversity_shift"] = (len({x[2] for x in recent})/len(recent)-len({x[2] for x in baseline})/len(baseline)) if enough else np.nan
                    row["category_concentration_7d"] = concentration(recent)
                    row["category_concentration_shift"] = concentration(recent)-concentration(baseline) if enough else np.nan
            records.append(row)
        # No state updates until every transaction at this timestamp has been scored.
        for r in batch:
            s = state[r.customer_id]
            angle = 2*np.pi*(timestamp.hour+timestamp.minute/60)/24
            s["history"].append((t, float(r.amount), r.merchant_id, r.category, flags[r.transaction_id]))
            s["count"] += 1
            s["last"] = t
            for key, value in [("devices", r.device_id), ("countries", r.country),
                               ("merchants", r.merchant_id), ("categories", r.category)]:
                s[key].add(value)
            s["sin_sum"] += float(np.sin(angle))
            s["cos_sum"] += float(np.cos(angle))
            device_accounts[r.device_id].add(r.customer_id)
            if r.status == "failed":
                heapq.heappush(pending, (int(r.outcome_available_at.value), r.transaction_id, r.customer_id))
    result = pd.DataFrame(records).set_index("transaction_id")
    numeric = result.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Non-finite historical feature.")
    return result


def temporal_split(tx, cfg):
    guard_development(tx, cfg)
    masks = {
        "train": (tx.timestamp < cfg.fit_at) & (tx.label_available_at <= cfg.fit_at),
        "early_stop": (tx.timestamp >= cfg.fit_at) & (tx.timestamp < cfg.validation_at)
                      & (tx.label_available_at <= cfg.validation_at),
        "validation": tx.timestamp >= cfg.validation_at,
    }
    audit = []
    for name, mask in masks.items():
        part = tx.loc[mask]
        if part.is_fraud.nunique() != 2:
            raise ValueError(name + " needs both classes; the declared run failed rather than skipping this seed.")
        audit.append({"period": name, "rows": len(part), "frauds": int(part.is_fraud.sum()),
                      "fraud_rate": float(part.is_fraud.mean()),
                      "first_event": str(part.timestamp.min()), "last_event": str(part.timestamp.max()),
                      "label_cutoff": str(cfg.fit_at if name == "train" else cfg.validation_at) if name != "validation" else "retrospective validation labels"})
    assert (sum(mask.astype(int) for mask in masks.values()) <= 1).all()
    assert (tx.loc[masks["train"], "label_available_at"] <= cfg.fit_at).all()
    assert (tx.loc[masks["early_stop"], "label_available_at"] <= cfg.validation_at).all()
    return masks, pd.DataFrame(audit)


def timing_checks(tx, features, cfg):
    times = [START, START+pd.Timedelta(seconds=30), START+pd.Timedelta(seconds=30), START+pd.Timedelta(seconds=120)]
    probe = pd.DataFrame({"transaction_id": ["p0", "p1", "p2", "p3"], "timestamp": times,
        "customer_id": ["c"]*4, "merchant_id": ["m"]*4, "device_id": ["d"]*4,
        "amount": [100., 200., 400., 800.], "country": ["IN"]*4, "category": ["retail"]*4,
        "status": ["failed", "succeeded", "succeeded", "succeeded"],
        "outcome_available_at": [t+pd.Timedelta(seconds=60) for t in times]})
    f = build_features(probe, cfg)
    assert f.loc["p1", "prior_count"] == f.loc["p2", "prior_count"] == 1
    assert f.loc["p1", "known_failures_1h"] == 0 and f.loc["p3", "known_failures_1h"] == 1
    assert np.isclose(np.expm1(f.loc["p1", "log_prior_median_30d"]), 100)
    assert np.isclose(f.loc["p3", "amount_ratio_30d"], 4.)
    before = build_features(probe.iloc[:3], cfg)
    pd.testing.assert_frame_equal(before, f.loc[before.index])
    # Test the real development prefix, including every new longitudinal measure.
    boundary = tx.timestamp.iloc[min(1200, len(tx)-1)]
    prefix = tx.loc[tx.timestamp <= boundary]
    rebuilt = build_features(prefix, cfg)
    pd.testing.assert_frame_equal(rebuilt, features.loc[rebuilt.index])
    return {name: "passed" for name in ["future_invariance", "same_timestamp_batching",
            "delayed_outcomes", "current_amount_excluded", "label_maturity", "development_boundary"]}


def prepare_seed(cfg, seed):
    _, _, tx = simulate(cfg, seed)
    f = build_features(tx, cfg)
    tx = tx.set_index("transaction_id")
    assert f.index.equals(tx.index)
    masks, audit = temporal_split(tx, cfg)
    checks = timing_checks(tx.reset_index(), f, cfg)
    return {"seed": int(seed), "tx": tx, "features": f, "masks": masks, "split_audit": audit,
            "checks": checks, "models": {}, "predictions": {}, "model_columns": {}}


def rule_triggers(frame):
    established = frame.prior_count >= 5
    return pd.DataFrame({
        "unusual_amount": established & (frame.amount_ratio_30d >= 4),
        "new_device_high_amount": established & frame.new_device.eq(1) & (frame.amount_ratio_30d >= 2),
        "attempt_burst": frame.attempts_1h >= 4,
        "known_failure_sequence": frame.known_failures_1h >= 2,
        "shared_device_new_country": established & (frame.prior_accounts_on_device >= 3) & frame.new_country.eq(1),
    }, index=frame.index).astype(int)


def rule_scores(frame):
    return rule_triggers(frame).mul(pd.Series(RULE_WEIGHTS)).sum(axis=1).to_numpy(dtype=float)


def fit_detector(state, cfg, name, columns=None):
    guard_development(state["tx"], cfg)
    f, y, masks = state["features"], state["tx"].is_fraud.astype(int), state["masks"]
    columns = list(columns or (CURRENT_FEATURES if name == "catboost_current" else FEATURES))
    forbidden = {"is_fraud", "fraud_scenario", "legitimate_context", "customer_id", "device_id", "status"}
    if set(columns) & forbidden or not set(columns) <= set(FEATURES + LONGITUDINAL_CANDIDATES):
        raise ValueError("Undeclared model feature; simulator-only columns are forbidden.")
    if name == "logistic_history":
        pre = ColumnTransformer([
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                                  ("scale", StandardScaler())]), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL_FEATURES)])
        model = Pipeline([("preprocess", pre), ("classifier", LogisticRegression(
            C=1., max_iter=1500, solver="lbfgs", random_state=state["seed"]))])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(f.loc[masks["train"], columns], y.loc[masks["train"]])
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError("Logistic regression did not converge.")
    else:
        model = CatBoostClassifier(iterations=cfg.boosting_iterations, depth=6, learning_rate=0.06,
            l2_leaf_reg=5, loss_function="Logloss", eval_metric="Logloss", random_seed=state["seed"],
            thread_count=cfg.threads, allow_writing_files=False)
        model.fit(f.loc[masks["train"], columns], y.loc[masks["train"]],
            cat_features=[c for c in CATEGORICAL_FEATURES if c in columns],
            eval_set=(f.loc[masks["early_stop"], columns], y.loc[masks["early_stop"]]),
            early_stopping_rounds=50, verbose=False)
    return model, columns


def ratio(a, b):
    return float(a / b) if b else np.nan


def select_daily(frame, k):
    ordered = frame.reset_index().sort_values(["day", "score", "transaction_id"], ascending=[True, False, True])
    chosen = ordered.groupby("day", sort=False).head(k).transaction_id
    return pd.Series(frame.index.isin(chosen), index=frame.index)


def evaluate_scores(tx, scores, name, cfg):
    guard_validation(tx, cfg)
    frame = tx[["timestamp", "customer_id", "amount", "is_fraud", "fraud_scenario",
                "legitimate_context", "category", "country"]].copy()
    frame["score"], frame["model"] = np.asarray(scores), name
    if not np.isfinite(frame.score).all():
        raise ValueError("Non-finite validation scores.")
    frame["day"] = frame.timestamp.dt.floor("D")
    fraud_n, legit_n = int(frame.is_fraud.sum()), int(frame.is_fraud.eq(0).sum())
    fraud_value = float(frame.loc[frame.is_fraud.eq(1), "amount"].sum())
    ap = float(average_precision_score(frame.is_fraud, frame.score))
    auc = float(roc_auc_score(frame.is_fraud, frame.score))
    rows, daily = [], []
    calendar = pd.date_range(cfg.validation_at, cfg.test_at, inclusive="left", freq="D")
    for k in cfg.review_budgets:
        selected = select_daily(frame, k)
        picked, hits = frame.loc[selected], frame.loc[selected & frame.is_fraud.eq(1)]
        false_n = int(picked.is_fraud.eq(0).sum())
        rows.append({"model": name, "daily_review_cap": k, "average_precision": ap,
            "roc_auc": auc, "review_count": len(picked), "fraud_reviewed": len(hits),
            "precision": ratio(len(hits), len(picked)), "fraud_recall": ratio(len(hits), fraud_n),
            "fraud_attempt_value_capture": ratio(float(hits.amount.sum()), fraud_value),
            "legitimate_reviewed": false_n, "false_positive_rate": ratio(false_n, legit_n)})
        for day in calendar:
            group = frame.loc[frame.day.eq(day)]
            reviewed = group.loc[selected.loc[group.index]]
            hit = reviewed.loc[reviewed.is_fraud.eq(1)]
            daily.append({"model": name, "daily_review_cap": k, "day": str(day),
                "transactions": len(group), "frauds": int(group.is_fraud.sum()),
                "reviews": len(reviewed), "fraud_reviewed": len(hit),
                "legitimate_reviewed": int(reviewed.is_fraud.eq(0).sum()),
                "precision": ratio(len(hit), len(reviewed)), "recall": ratio(len(hit), int(group.is_fraud.sum())),
                "false_positive_rate": ratio(int(reviewed.is_fraud.eq(0).sum()), int(group.is_fraud.eq(0).sum())),
                "fraud_attempt_value_capture": ratio(float(hit.amount.sum()), float(group.loc[group.is_fraud.eq(1), "amount"].sum())),
                "budget_exhausted": len(reviewed) == k, "fewer_candidates_than_budget": len(group) < k,
                "unused_capacity": max(0, k-len(reviewed))})
        if k == cfg.main_review_budget:
            frame["selected_for_review"] = selected
            frame["action"] = np.where(selected, "REVIEW", "NO_REVIEW")
    return pd.DataFrame(rows), pd.DataFrame(daily), frame


def score_detector(state, cfg, name, model=None, columns=None):
    mask = state["masks"]["validation"]
    tx = state["tx"].loc[mask]
    guard_validation(tx, cfg)
    f = state["features"].loc[mask]
    scores = rule_scores(f) if name == "rules" else model.predict_proba(f[columns])[:, 1]
    return evaluate_scores(tx, scores, name, cfg)


def complete_primary(state, cfg):
    metrics, daily = [], []
    for name in MODELS:
        if name != "rules" and name not in state["models"]:
            state["models"][name], state["model_columns"][name] = fit_detector(state, cfg, name)
        result = score_detector(state, cfg, name, state["models"].get(name), state["model_columns"].get(name))
        m, d, p = result
        metrics.append(m)
        daily.append(d)
        state["predictions"][name] = p
    state["comparison"], state["daily_metrics"] = pd.concat(metrics, ignore_index=True), pd.concat(daily, ignore_index=True)
    op = state["comparison"].loc[lambda x: x.daily_review_cap.eq(cfg.main_review_budget)]
    state["leader"] = op.sort_values(["precision", "fraud_attempt_value_capture", "average_precision", "model"],
        ascending=[False, False, False, True]).iloc[0]["model"]
    return state


def scenario_table(predictions):
    rows = []
    for name, frame in predictions.items():
        for scenario, group in frame.loc[frame.is_fraud.eq(1)].groupby("fraud_scenario"):
            picked = group.loc[group.selected_for_review]
            rows.append({"model": name, "scenario": scenario, "fraud_count": len(group),
                         "fraud_reviewed": len(picked), "recall": ratio(len(picked), len(group)),
                         "attempt_value_capture": ratio(float(picked.amount.sum()), float(group.amount.sum()))})
    return pd.DataFrame(rows)


def evidence_tags(row):
    tags = []
    for condition, label in [
        (row.prior_count < 5, "limited_history"), (row.amount_ratio_30d >= 2, "amount_at_least_2x_history"),
        (row.new_device == 1, "unseen_device"), (row.new_country == 1, "unseen_country"),
        (row.new_merchant == 1, "unseen_merchant"), (row.attempts_1h >= 4, "recent_attempt_burst"),
        (row.known_failures_1h >= 2, "known_failure_sequence"),
        (row.prior_accounts_on_device >= 3, "device_previously_used_by_3plus_accounts"),
        (row.hour_deviation >= 6, "hour_deviation_at_least_6h")]:
        if condition:
            tags.append(label)
    return tags


def evidence_text(row):
    parts = ["%d earlier attempts in 1 hour" % row.attempts_1h,
             "%d failed outcomes known in 1 hour" % row.known_failures_1h,
             "%d previously observed accounts on device" % row.prior_accounts_on_device]
    if pd.notna(row.amount_ratio_30d):
        parts.insert(0, "amount %.2fx prior 30-day median attempted amount" % row.amount_ratio_30d)
    return "; ".join(parts + evidence_tags(row))


def enrich_predictions(state):
    f = state["features"].loc[state["masks"]["validation"]]
    rules = rule_triggers(f).apply(lambda row: "; ".join(n for n in RULE_WEIGHTS if row[n]), axis=1)
    text = f.apply(evidence_text, axis=1)
    patterns = f.apply(lambda row: "; ".join(evidence_tags(row)), axis=1)
    for p in state["predictions"].values():
        p["triggered_rules"], p["observed_evidence"], p["evidence_patterns"] = rules, text, patterns


def context_table(state):
    enrich_predictions(state)
    rows, examples = [], []
    contexts = ("routine", "travel", "new_phone", "large_purchase", "legitimate_burst")
    for name, frame in state["predictions"].items():
        for context in contexts:
            group = frame.loc[frame.is_fraud.eq(0) & frame.legitimate_context.eq(context)]
            picked = group.loc[group.selected_for_review]
            patterns = Counter(tag for value in picked.evidence_patterns for tag in value.split("; ") if tag)
            rows.append({"model": name, "context": context, "legitimate_count": len(group),
                "legitimate_reviewed": len(picked), "alert_rate": ratio(len(picked), len(group)),
                "score_median": float(group.score.median()), "score_q75": float(group.score.quantile(.75)),
                "reviewed_score_median": float(picked.score.median()), "reviewed_score_q75": float(picked.score.quantile(.75)),
                "top_evidence_patterns": json.dumps(patterns.most_common(3)),
                "provisional_leader": name == state["leader"]})
            if name == state["leader"] and len(picked):
                # At most three distinct representative false positives per context.
                ids = [picked.score.idxmax(), (picked.score-picked.score.median()).abs().idxmin(), picked.amount.idxmax()]
                example = picked.loc[list(dict.fromkeys(ids))].reset_index()
                examples.extend(example.to_dict("records"))
    return pd.DataFrame(rows), pd.DataFrame(examples, columns=["transaction_id"] + list(next(iter(state["predictions"].values())).columns))


def separation_audit(tx, features, cfg, world):
    guard_validation(tx, cfg)
    rows = []
    for col in SEPARATION_FEATURES:
        values = features.loc[tx.index, col]
        finite = values.notna()
        y = tx.is_fraud
        auc = float(roc_auc_score(y.loc[finite], values.loc[finite])) if y.loc[finite].nunique() == 2 else np.nan
        missing_auc = float(roc_auc_score(y, (~finite).astype(int))) if y.nunique() == 2 else np.nan
        for label, name in [(0, "legitimate"), (1, "fraud")]:
            group = values.loc[y.eq(label)]
            rows.append({"world": world, "feature": col, "class": name, "rows": len(group),
                "finite_count": int(group.notna().sum()), "missing_rate": float(group.isna().mean()),
                "mean": float(group.mean()), "std": float(group.std()), "q25": float(group.quantile(.25)),
                "median": float(group.median()), "q75": float(group.quantile(.75)), "q95": float(group.quantile(.95)),
                "roc_auc_raw_direction": auc, "roc_auc_best_direction": max(auc, 1-auc) if np.isfinite(auc) else np.nan,
                "missingness_roc_auc": missing_auc})
    return pd.DataFrame(rows)


def overlap_audit(tx, features, cfg, world):
    guard_validation(tx, cfg)
    f = features.loc[tx.index]
    conditions = {"familiar_device": f.new_device.eq(0), "familiar_country": f.new_country.eq(0),
        "familiar_merchant": f.new_merchant.eq(0), "normal_amount_0.5_to_2x": f.amount_ratio_30d.between(.5, 2),
        "hour_deviation_under_3h": f.hour_deviation.le(3), "no_known_failed_outcome_1h": f.known_failures_1h.eq(0),
        "fewer_than_4_prior_attempts_1h": f.attempts_1h.lt(4)}
    conditions["all_familiar_normal_conditions"] = pd.concat(conditions, axis=1).all(axis=1)
    rows = []
    for label, name in [(0, "legitimate"), (1, "fraud")]:
        mask = tx.is_fraud.eq(label)
        for condition, passed in conditions.items():
            rows.append({"world": world, "class": name, "condition": condition,
                "rows": int(mask.sum()), "count": int(passed.loc[mask].sum()),
                "share": float(passed.loc[mask].mean())})
    return pd.DataFrame(rows)


def low_slow_diagnostics(state):
    frame = state["predictions"]["catboost_history"]
    groups = {
        "caught_low_and_slow": frame.fraud_scenario.eq("low_and_slow") & frame.selected_for_review,
        "missed_low_and_slow": frame.fraud_scenario.eq("low_and_slow") & ~frame.selected_for_review,
        "legitimate_routine": frame.is_fraud.eq(0) & frame.legitimate_context.eq("routine"),
    }
    rows = []
    f = state["features"].loc[frame.index]
    for group, mask in groups.items():
        for col in DIAGNOSTIC_FEATURES:
            v = f.loc[mask, col]
            rows.append({"group": group, "feature": col, "rows": len(v), "finite_count": int(v.notna().sum()),
                         "missing_rate": float(v.isna().mean()), "mean": float(v.mean()),
                         "std": float(v.std()), "q25": float(v.quantile(.25)),
                         "median": float(v.median()), "q75": float(v.quantile(.75))})
    decisions = []
    for col in LONGITUDINAL_CANDIDATES:
        a, b = f.loc[groups["missed_low_and_slow"], col], f.loc[groups["legitimate_routine"], col]
        aa, bb = a.dropna(), b.dropna()
        denom = np.sqrt((aa.var()+bb.var())/2) if min(len(aa), len(bb)) >= 2 else np.nan
        effect = float((aa.mean()-bb.mean())/denom) if np.isfinite(denom) and denom > 0 else np.nan
        enough = min(len(aa), len(bb)) >= 5 and min(ratio(len(aa), len(a)), ratio(len(bb), len(b))) >= .5
        selected = bool(enough and np.isfinite(effect) and abs(effect) >= .25)
        decisions.append({"feature": col, "diagnostic_seed": state["seed"],
            "missed_finite_count": len(aa), "routine_finite_count": len(bb),
            "standardized_mean_difference": effect, "selected": selected,
            "reason": "predeclared_development_diagnostic_gate_passed" if selected else "insufficient_coverage_or_development_separation",
            "claim_status": "exploratory_validation_selection; no final-test evidence"})
    return pd.DataFrame(rows), pd.DataFrame(decisions)


def longitudinal_challenger(state, cfg, selected):
    base = state["comparison"].loc[lambda x: x.model.eq("catboost_history") & x.daily_review_cap.eq(50)].iloc[0]
    baseline_scenario = scenario_table({"catboost_history": state["predictions"]["catboost_history"]})
    base_low = baseline_scenario.loc[baseline_scenario.scenario.eq("low_and_slow"), "recall"]
    row = {"seed": state["seed"], "status": "not_run_no_features_passed_diagnostic_gate",
           "selected_features": json.dumps(selected), "daily_review_cap": 50,
           "selection_period": "seed_42_development_validation",
           "evidence_role": "exploratory_selection_seed" if state["seed"] == SEEDS[0] else "separate_synthetic_seed_replication"}
    for metric in METRICS:
        row["baseline_" + metric] = base[metric]
        row["challenger_" + metric] = np.nan
        row["delta_" + metric] = np.nan
    row["baseline_low_and_slow_recall"] = float(base_low.iloc[0]) if len(base_low) else np.nan
    row["challenger_low_and_slow_recall"] = row["delta_low_and_slow_recall"] = np.nan
    contexts = []
    if selected:
        model, columns = fit_detector(state, cfg, "longitudinal_challenger", FEATURES + selected)
        metrics, _, frame = score_detector(state, cfg, "longitudinal_challenger", model, columns)
        op = metrics.loc[metrics.daily_review_cap.eq(50)].iloc[0]
        row["status"] = "evaluated_on_development_validation"
        for metric in METRICS:
            row["challenger_" + metric], row["delta_" + metric] = op[metric], op[metric] - base[metric]
        slow = frame.loc[frame.fraud_scenario.eq("low_and_slow")]
        row["challenger_low_and_slow_recall"] = ratio(int(slow.selected_for_review.sum()), len(slow))
        row["delta_low_and_slow_recall"] = row["challenger_low_and_slow_recall"] - row["baseline_low_and_slow_recall"]
        old = state["predictions"]["catboost_history"]
        for context in ["routine", "travel", "new_phone", "large_purchase", "legitimate_burst"]:
            mask = frame.is_fraud.eq(0) & frame.legitimate_context.eq(context)
            before, after = int(old.loc[mask].selected_for_review.sum()), int(frame.loc[mask].selected_for_review.sum())
            contexts.append({"seed": state["seed"], "context": context, "legitimate_count": int(mask.sum()),
                "baseline_legitimate_reviews": before, "challenger_legitimate_reviews": after,
                "delta_legitimate_reviews": after-before, "delta_alert_rate": ratio(after-before, int(mask.sum()))})
        state["longitudinal_model"], state["longitudinal_columns"], state["longitudinal_predictions"] = model, columns, frame
    return row, contexts


def daily_summary(daily):
    rows = []
    for (model, k), g in daily.groupby(["model", "daily_review_cap"], sort=False):
        row = {"model": model, "daily_review_cap": k, "validation_days": len(g),
            "days_budget_exhausted": int(g.budget_exhausted.sum()),
            "days_fewer_candidates_than_budget": int(g.fewer_candidates_than_budget.sum()),
            "days_without_candidates": int(g.transactions.eq(0).sum()),
            "unused_review_slots": int(g.unused_capacity.sum())}
        for metric in ["precision", "recall", "false_positive_rate", "fraud_attempt_value_capture", "reviews"]:
            for stat in ["mean", "std", "min", "max"]:
                row[metric + "_daily_" + stat] = float(getattr(g[metric], stat)())
            row[metric + "_defined_days"] = int(g[metric].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def save_primary(state, cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    guard_development(state["tx"], cfg)
    state["comparison"].to_csv(out / "comparison.csv", index=False)
    state["daily_metrics"].to_csv(out / "daily_metrics.csv", index=False)
    daily_summary(state["daily_metrics"]).to_csv(out / "daily_variability_summary.csv", index=False)
    scenario_table(state["predictions"]).to_csv(out / "scenario_breakdown.csv", index=False)
    contexts, examples = context_table(state)
    contexts.to_csv(out / "legitimate_context_breakdown.csv", index=False)
    examples.to_csv(out / "false_positive_examples.csv", index=False)
    state["split_audit"].to_csv(out / "split_audit.csv", index=False)
    for name, frame in state["predictions"].items():
        guard_validation(frame, cfg)
        frame.to_csv(out / (name + "_validation_predictions.csv"), index_label="transaction_id")
        cases = pd.concat([
            frame.loc[frame.selected_for_review & frame.is_fraud.eq(0)].nlargest(50, "score").assign(case_type="false_positive"),
            frame.loc[~frame.selected_for_review & frame.is_fraud.eq(1)].nlargest(50, "amount").assign(case_type="missed_fraud"),
            frame.loc[frame.selected_for_review & frame.is_fraud.eq(1)].nlargest(20, "score").assign(case_type="true_positive"),
        ])
        cases.to_csv(out / (name + "_error_cases.csv"), index_label="transaction_id")


def validation_chart(state, cfg, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, p in state["predictions"].items():
        guard_validation(p, cfg)
        precision, recall, _ = precision_recall_curve(p.is_fraud, p.score)
        axes[0].plot(recall, precision, label=name)
        g = state["comparison"].loc[lambda x: x.model.eq(name)]
        axes[1].plot(g.daily_review_cap, g.fraud_recall, marker="o", label=name)
    axes[0].set(xlabel="Fraud recall", ylabel="Precision", title="Development validation: precision–recall")
    axes[1].set(xlabel="Reviews per UTC day", ylabel="Fraud recall", title="Same capacity for every detector")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "validation_comparison.png", dpi=150)
    plt.close(fig)


def robustness_summary(per_seed):
    raw = pd.concat(per_seed, ignore_index=True)
    raw.insert(0, "row_type", "seed")
    aggregates = []
    for (model, k), group in raw.groupby(["model", "daily_review_cap"], sort=False):
        for stat in ["mean", "std", "min", "max"]:
            row = {"row_type": stat, "seed": np.nan, "model": model, "daily_review_cap": k}
            row.update({metric: float(getattr(group[metric], stat)()) for metric in METRICS})
            aggregates.append(row)
    return pd.concat([raw, pd.DataFrame(aggregates)], ignore_index=True)


def robustness_chart(summary, out):
    table = summary.loc[summary.row_type.eq("seed") & summary.daily_review_cap.eq(50)].pivot(index="model", columns="seed", values="average_precision")
    table = table.reindex(index=MODELS, columns=SEEDS)
    ranks = table.rank(axis=0, ascending=False, method="min")
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    im = ax.imshow(table.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for i in range(len(table)):
        for j in range(len(SEEDS)):
            value = table.iloc[i, j]
            ax.text(j, i, "%.3f\nrank %d" % (value, ranks.iloc[i, j]), ha="center", va="center",
                    color="white" if value > .6 else "black", fontsize=9)
    ax.set_xticks(range(len(SEEDS)), labels=[str(x) for x in SEEDS])
    ax.set_yticks(range(len(MODELS)), labels=MODELS)
    ax.set(title="Average precision and rank across all five declared seeds", xlabel="Simulation seed")
    fig.colorbar(im, ax=ax, label="Average precision")
    fig.tight_layout()
    fig.savefig(out / "robustness_ranking.png", dpi=150)
    plt.close(fig)


def run_ablation(state, cfg):
    base = state["comparison"].loc[lambda x: x.model.eq("catboost_history") & x.daily_review_cap.eq(50)].iloc[0].to_dict()
    rows = [dict(base, seed=state["seed"], ablation="full_history_reference", removed_features="[]", best_iteration=state["models"]["catboost_history"].get_best_iteration(), **{"delta_"+m: 0. for m in METRICS})]
    for family, removed in FEATURE_FAMILIES.items():
        columns = [f for f in FEATURES if f not in removed]
        model, columns = fit_detector(state, cfg, "ablation_"+family, columns)
        metrics, _, _ = score_detector(state, cfg, "ablation_"+family, model, columns)
        row = metrics.loc[metrics.daily_review_cap.eq(50)].iloc[0].to_dict()
        row.update({"seed": state["seed"], "ablation": "without_"+family,
                    "removed_features": json.dumps(removed), "best_iteration": model.get_best_iteration()})
        row.update({"delta_"+m: row[m]-base[m] for m in METRICS})
        rows.append(row)
        print("Ablation completed:", family, flush=True)
    return pd.DataFrame(rows)

STAGES = ("simulation", "history", "split", "rules", "logistic", "current", "history_model",
          "comparison", "scenario", "low_slow", "false_positives", "robustness", "ablation", "export")
REQUIRED_OUTPUTS = (
    "comparison.csv", "daily_metrics.csv", "daily_variability_summary.csv", "scenario_breakdown.csv",
    "legitimate_context_breakdown.csv", "false_positive_examples.csv", "feature_importance.csv",
    "feature_separation_audit.csv", "simulator_overlap_audit.csv", "low_and_slow_diagnostics.csv",
    "longitudinal_feature_decisions.csv", "longitudinal_tradeoffs.csv", "longitudinal_context_tradeoffs.csv",
    "robustness_summary.csv", "ablation_summary.csv", "split_audit.csv", "manifest.json",
    "validation_comparison.png", "robustness_ranking.png", "feature_separation.png",
    "logistic_history.joblib", "catboost_current.cbm", "catboost_history.cbm",
) + tuple(name+suffix for name in MODELS for suffix in ("_validation_predictions.csv", "_error_cases.csv"))


def seed_manifest(state, cfg):
    guard_development(state["tx"], cfg)
    return {"seed": state["seed"], "configuration": cfg.__dict__, "timing_checks": state["checks"],
            "model_columns": state["model_columns"], "provisional_validation_leader": state["leader"],
            "fit_at": str(cfg.fit_at), "validation_at": str(cfg.validation_at), "locked_test_at": str(cfg.test_at),
            "last_development_event": str(state["tx"].timestamp.max()),
            "development_dataset_sha256": hashlib.sha256(pd.util.hash_pandas_object(state["tx"], index=True).values.tobytes()).hexdigest(),
            "locked_test_evaluated": False, "locked_test_materialized": False}


def separation_stage(state, cfg, out):
    tx = state["tx"].loc[state["masks"]["validation"]]
    actual = separation_audit(tx, state["features"], cfg, "v2_hardened")
    actual_overlap = overlap_audit(tx, state["features"], cfg, "v2_hardened")
    _, _, ref_tx = simulate(cfg, state["seed"], hardened=False)
    ref_f = build_features(ref_tx, cfg)
    ref_tx = ref_tx.set_index("transaction_id")
    ref_val = ref_tx.loc[ref_tx.timestamp >= cfg.validation_at]
    ref = separation_audit(ref_val, ref_f, cfg, "v1_like_reference_not_archived_v1")
    ref_overlap = overlap_audit(ref_val, ref_f, cfg, "v1_like_reference_not_archived_v1")
    audit = pd.concat([ref, actual], ignore_index=True)
    audit.to_csv(out / "feature_separation_audit.csv", index=False)
    pd.concat([ref_overlap, actual_overlap], ignore_index=True).to_csv(out / "simulator_overlap_audit.csv", index=False)
    table = audit.loc[audit["class"].eq("fraud")].pivot(index="feature", columns="world", values="roc_auc_best_direction")
    ax = table.reindex(SEPARATION_FEATURES).plot.barh(figsize=(9, 5), xlim=(.45, 1), fontsize=8)
    ax.set(xlabel="Univariate ROC-AUC, better of both directions", ylabel="", title="Development-only simulator separability diagnostic")
    ax.legend(fontsize=7)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out / "feature_separation.png", dpi=150)
    plt.close(fig)


def robustness_stage(state, cfg, out):
    per_seed = [state["comparison"].assign(seed=state["seed"])]
    long_rows, context_rows = [state["longitudinal_row"]], list(state["longitudinal_context_rows"])
    manifests = [seed_manifest(state, cfg)]
    for seed in SEEDS[1:]:
        print("Starting declared robustness seed", seed, flush=True)
        other = complete_primary(prepare_seed(cfg, seed), cfg)
        per_seed.append(other["comparison"].assign(seed=seed))
        row, contexts = longitudinal_challenger(other, cfg, state["selected_longitudinal"])
        long_rows.append(row)
        context_rows.extend(contexts)
        seed_out = out / "seeds" / str(seed)
        save_primary(other, cfg, seed_out)
        if "longitudinal_predictions" in other:
            other["longitudinal_predictions"].to_csv(seed_out / "longitudinal_validation_predictions.csv", index_label="transaction_id")
        details = seed_manifest(other, cfg)
        manifests.append(details)
        (seed_out / "manifest.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        print("Completed declared seed", seed, "— reserved final period absent", flush=True)
    summary = robustness_summary(per_seed)
    summary.to_csv(out / "robustness_summary.csv", index=False)
    robustness_chart(summary, out)
    pd.DataFrame(long_rows).to_csv(out / "longitudinal_tradeoffs.csv", index=False)
    pd.DataFrame(context_rows, columns=["seed", "context", "legitimate_count", "baseline_legitimate_reviews",
        "challenger_legitimate_reviews", "delta_legitimate_reviews", "delta_alert_rate"]).to_csv(out / "longitudinal_context_tradeoffs.csv", index=False)
    state["seed_manifests"] = manifests


def validate_export(out, cfg):
    missing = [name for name in REQUIRED_OUTPUTS if not (out/name).is_file()]
    if missing:
        raise AssertionError("Incomplete V2 output contract: " + ", ".join(missing))
    manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["locked_test_evaluated"] is False
    assert manifest["locked_test_materialized"] is False
    assert manifest["seed_list"] == list(SEEDS)
    for item in manifest["seed_manifests"]:
        assert item["locked_test_evaluated"] is False and item["locked_test_materialized"] is False
        assert pd.Timestamp(item["last_development_event"]) < cfg.test_at
    comparison = pd.read_csv(out/"comparison.csv")
    assert set(comparison.model) == set(MODELS)
    assert set(comparison.daily_review_cap) == set(cfg.review_budgets)
    r = pd.read_csv(out/"robustness_summary.csv")
    seeds = r.loc[r.row_type.eq("seed")]
    assert set(seeds.seed.astype(int)) == set(SEEDS) and len(seeds) == len(SEEDS)*len(MODELS)*len(cfg.review_budgets)
    assert set(r.row_type) == {"seed", "mean", "std", "min", "max"}
    assert np.isfinite(seeds[METRICS].to_numpy()).all()
    assert len(pd.read_csv(out/"ablation_summary.csv")) == len(FEATURE_FAMILIES)+1
    for path in out.rglob("*validation_predictions.csv"):
        times = pd.read_csv(path, usecols=["timestamp"])
        times["timestamp"] = pd.to_datetime(times.timestamp, utc=True)
        guard_validation(times, cfg)
    return {"required_files": len(REQUIRED_OUTPUTS), "seed_rows": len(seeds), "final_period_absent": True}


def export_stage(state, cfg, out):
    save_primary(state, cfg, out)
    for name, model in state["models"].items():
        if name.startswith("catboost"):
            model.save_model(str(out/(name+".cbm")))
        else:
            joblib.dump(model, out/(name+".joblib"))
    if "longitudinal_model" in state:
        state["longitudinal_model"].save_model(str(out/"longitudinal_challenger.cbm"))
        state["longitudinal_predictions"].to_csv(out/"longitudinal_validation_predictions.csv", index_label="transaction_id")
    pd.DataFrame({"feature": FEATURES, "importance": state["models"]["catboost_history"].get_feature_importance()}).sort_values(
        "importance", ascending=False).to_csv(out/"feature_importance.csv", index=False)
    state["tx"].to_csv(out/"development_transactions.csv.gz", index_label="transaction_id", compression="gzip")
    manifest = {
        "project": "transaction-fraud-intelligence", "implementation_version": VERSION,
        "python_version": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in CORE_PACKAGES},
        "seed_list": list(SEEDS), "configuration": cfg.__dict__, "feature_list": FEATURES,
        "model_columns": state["model_columns"], "categorical_features": CATEGORICAL_FEATURES,
        "diagnostic_features": DIAGNOSTIC_FEATURES, "selected_longitudinal_features": state["selected_longitudinal"],
        "longitudinal_model_columns": state.get("longitudinal_columns", []),
        "longitudinal_selection": {"seed": 42, "period": "development_validation", "min_finite_per_group": 5,
            "min_coverage_per_group": .5, "min_absolute_standardized_difference": .25,
            "separate_from_primary_comparison": True},
        "feature_families": FEATURE_FAMILIES, "ablation_seed": 42,
        "split_cutoffs": {"fit_at": str(cfg.fit_at), "validation_at": str(cfg.validation_at), "locked_test_at": str(cfg.test_at)},
        "timing_checks": state["checks"], "seed_manifests": state["seed_manifests"],
        "locked_test_evaluated": False, "locked_test_materialized": False,
        "selected_provisional_validation_leader": state["leader"], "rule_weights": RULE_WEIGHTS,
        "source": "synthetic standardized INR payment attempts", "score_status": "uncalibrated",
        "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "known_limitations": [
            "Synthetic development results do not establish real-world performance.",
            "The paired V1-like reference is not an exact reproduction of archived V1 data.",
            "Simulator hardening changes the benchmark; V1/V2 score differences are not a pure model comparison.",
            "Labels mature after an assumed seven days for both classes.",
            "Daily top-K is offline ranking of attempts, not online approval, blocking or prevented loss.",
            "Validation selection is exploratory, particularly the primary-seed longitudinal challenger.",
            "Repeated seeds measure synthetic-world variability, not a real-population confidence interval.",
            "Ablations use one declared seed and independent early stopping under equal maximum budgets.",
            "Context tags are exclusive; multiple legitimate changes may overlap in reality.",
            "No final-test events are materialized or examined; later evaluation needs a frozen protocol.",
            "API, deployment, public-data benchmark, SHAP and calibration remain out of scope.",
        ],
    }
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    check = validate_export(out, cfg)
    (out/"output_contract_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    (out/"resolved_packages.txt").write_text("\n".join(sorted(
        d.metadata["Name"]+"=="+d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")))+"\n", encoding="utf-8")
    archive = Path(shutil.make_archive(str(out), "zip", root_dir=out))
    with zipfile.ZipFile(archive) as z:
        assert set(REQUIRED_OUTPUTS) <= set(z.namelist())
    return archive


def execute_stage(stage, work, cfg):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work/"state.joblib"
    if stage not in STAGES:
        raise ValueError("Unknown experiment stage.")
    if stage == "simulation":
        out = work/"results"/("fraud_v2_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        out.mkdir(parents=True)
        _, _, tx = simulate(cfg, SEEDS[0])
        state = {"seed": SEEDS[0], "tx": tx.set_index("transaction_id"), "models": {},
                 "model_columns": {}, "predictions": {}, "completed": [], "out": str(out),
                 "configuration": cfg.__dict__}
    else:
        state = joblib.load(checkpoint)
        if state["configuration"] != cfg.__dict__:
            raise ValueError("Configuration changed mid-run. Begin a fresh run.")
        if state["completed"] != list(STAGES[:STAGES.index(stage)]):
            raise ValueError("Run notebook stages in order; repeat Run all for a fresh run.")
        out = Path(state["out"])
    guard_development(state["tx"], cfg)
    if stage == "history":
        state["features"] = build_features(state["tx"].reset_index(), cfg)
    elif stage == "split":
        state["masks"], state["split_audit"] = temporal_split(state["tx"], cfg)
        state["checks"] = timing_checks(state["tx"].reset_index(), state["features"], cfg)
        state["split_audit"].to_csv(out/"split_audit.csv", index=False)
    elif stage == "rules":
        _, _, state["predictions"]["rules"] = score_detector(state, cfg, "rules")
    elif stage in ("logistic", "current", "history_model"):
        name = {"logistic": "logistic_history", "current": "catboost_current", "history_model": "catboost_history"}[stage]
        state["models"][name], state["model_columns"][name] = fit_detector(state, cfg, name)
    elif stage == "comparison":
        complete_primary(state, cfg)
        save_primary(state, cfg, out)
        validation_chart(state, cfg, out)
    elif stage == "scenario":
        scenario_table(state["predictions"]).to_csv(out/"scenario_breakdown.csv", index=False)
        separation_stage(state, cfg, out)
    elif stage == "low_slow":
        diagnosis, decisions = low_slow_diagnostics(state)
        diagnosis.to_csv(out/"low_and_slow_diagnostics.csv", index=False)
        diagnosis.pivot(index="feature", columns="group", values="median").reindex([
            "amount_ratio_30d", "hours_since_previous", "count_ratio_7d_prior30", "count_ratio_14d_prior30",
            "count_ratio_30d_prior30", "attempt_value_7d", "new_merchant", "category_concentration_7d",
            "prior_accounts_on_device", "hour_deviation"]).reset_index().to_csv(out/"low_and_slow_overview.csv", index=False)
        decisions.to_csv(out/"longitudinal_feature_decisions.csv", index=False)
        state["selected_longitudinal"] = decisions.loc[decisions.selected, "feature"].tolist()
        state["longitudinal_row"], state["longitudinal_context_rows"] = longitudinal_challenger(state, cfg, state["selected_longitudinal"])
        pd.DataFrame([state["longitudinal_row"]]).to_csv(out/"longitudinal_tradeoffs.csv", index=False)
    elif stage == "false_positives":
        contexts, examples = context_table(state)
        contexts.to_csv(out/"legitimate_context_breakdown.csv", index=False)
        examples.to_csv(out/"false_positive_examples.csv", index=False)
    elif stage == "robustness":
        robustness_stage(state, cfg, out)
    elif stage == "ablation":
        run_ablation(state, cfg).to_csv(out/"ablation_summary.csv", index=False)
    elif stage == "export":
        archive = export_stage(state, cfg, out)
        (work/"result.json").write_text(json.dumps({"output_directory": str(out), "archive": str(archive)}), encoding="utf-8")
    state["completed"].append(stage)
    joblib.dump(state, checkpoint)
    (work/"progress.json").write_text(json.dumps({"output_directory": str(out), "completed": state["completed"]}), encoding="utf-8")
    print("Completed", stage, "| final test remains absent and unevaluated", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = Config(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.all or args.stage == "simulation":
        print("Python", platform.python_version(), "|", json.dumps({p: importlib.metadata.version(p) for p in CORE_PACKAGES}), flush=True)
    for stage in STAGES if args.all else [args.stage]:
        execute_stage(stage, args.work_dir, cfg)


if __name__ == "__main__":
    main()
