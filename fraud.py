from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
from itertools import groupby
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

START = pd.Timestamp("2025-01-01", tz="UTC")
CATEGORIES = ["groceries", "retail", "dining", "travel", "digital", "utilities"]
COUNTRIES = ["IN", "SG", "AE", "GB", "US"]
MODELS = ("rules", "logistic_history", "catboost_current", "catboost_history")
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
RULE_WEIGHTS = {"unusual_amount": 25, "new_device_high_amount": 25, "attempt_burst": 20,
                "known_failure_sequence": 15, "shared_device_new_country": 15}


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
    if "timestamp" not in frame:
        raise ValueError("Timestamp metadata is required to enforce the final-test lock.")
    times = frame["timestamp"]
    if times.isna().any() or (times < START).any() or (times >= cfg.test_at).any():
        raise ValueError("Reserved final-period or invalid timestamps are forbidden in V2.")


def guard_validation(frame, cfg):
    guard_development(frame, cfg)
    if (frame.timestamp < cfg.validation_at).any():
        raise ValueError("Evaluation accepts development-validation rows only.")


def simulate(cfg, seed):
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
        mimic = attack_rng.random() < 0.80
        hour = float(np.mod(attack_rng.normal(p["hour"], 3), 24)) if mimic else float(attack_rng.uniform(0, 24))
        base_sec = day * 86400 + int(hour * 3600)
        previous = [r for r in normal_histories[p["customer_id"]] if r["second"] < base_sec]
        latest = max(previous, key=lambda r: r["second"]) if previous else None
        familiar_device = latest["device_id"] if latest else p["device0"]
        device = familiar_device if attack_rng.random() < 0.85 else "shared_actor_%03d" % int(attack_rng.integers(0, 30))
        country = p["home"] if attack_rng.random() < 0.92 else str(attack_rng.choice(COUNTRIES))
        known_merchants = [r["merchant_id"] for r in previous] or list(p["favourites"])
        if scenario == "account_takeover":
            count, step = int(attack_rng.integers(2, 7)), int(attack_rng.integers(120, 1800))
            if mimic:
                step = int(attack_rng.integers(1800, 36000))
            fail_p = 0.05 if mimic else 0.18
        elif scenario == "card_testing":
            count, step = int(attack_rng.integers(5, 13)), int(attack_rng.integers(15, 150))
            if attack_rng.random() < 0.50:
                step = int(attack_rng.integers(600, 5400))
            fail_p = float(attack_rng.uniform(0.02, 0.12)) if mimic else 0.45
        else:
            count, step, fail_p = int(attack_rng.integers(3, 9)), int(attack_rng.integers(1, 4)) * 86400, 0.06
        for j in range(count):
            event_sec = base_sec + j * step
            if scenario == "low_and_slow":
                event_sec = max(base_sec, event_sec + int(attack_rng.normal(0, 4 * 3600)))
            if mimic and scenario != "card_testing":
                amount = p["base"] * attack_rng.lognormal(0, p["spread"])
            elif scenario == "account_takeover":
                amount = p["base"] * attack_rng.uniform(1.2, 6)
            elif scenario == "card_testing":
                amount = p["base"] * (attack_rng.lognormal(-0.2, p["spread"]) if mimic and attack_rng.random() < 0.55 else attack_rng.uniform(0.02, 0.35))
            else:
                amount = p["base"] * attack_rng.lognormal(0, p["spread"] * 0.7)
            merchant = attack_rng.choice(known_merchants if attack_rng.random() < 0.80 else merchant_ids)
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
        for r in batch:
            cid = r.customer_id
            if cid not in state:
                state[cid] = {"history": deque(), "count": 0, "devices": set(),
                              "countries": set(), "merchants": set(),
                              "first": t, "last": None, "sin_sum": 0., "cos_sum": 0.}
            s = state[cid]
            while s["history"] and s["history"][0][0] < t - 30 * day_ns:
                s["history"].popleft()
            h30 = list(s["history"])
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
                "new_merchant": int(r.merchant_id not in s["merchants"]), "known_failures_1h": len(f),
                "prior_accounts_on_device": len(device_accounts[r.device_id]), "hour_deviation": hour_dev,
                "category": str(r.category), "country": str(r.country),
            }
            records.append(row)
        for r in batch:
            s = state[r.customer_id]
            angle = 2*np.pi*(timestamp.hour+timestamp.minute/60)/24
            s["history"].append((t, float(r.amount)))
            s["count"] += 1
            s["last"] = t
            for key, value in [("devices", r.device_id), ("countries", r.country),
                               ("merchants", r.merchant_id)]:
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


def prepare_seed(cfg, seed):
    _, _, tx = simulate(cfg, seed)
    f = build_features(tx, cfg)
    tx = tx.set_index("transaction_id")
    assert f.index.equals(tx.index)
    masks, _ = temporal_split(tx, cfg)
    return {"seed": int(seed), "tx": tx, "features": f, "masks": masks}


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
    if set(columns) & forbidden or not set(columns) <= set(FEATURES):
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


def run(seed=42, output="outputs"):
    cfg = Config()
    state = prepare_seed(cfg, seed)
    results = []
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    for name in MODELS:
        model, columns = (None, None) if name == "rules" else fit_detector(state, cfg, name)
        metrics, daily, predictions = score_detector(state, cfg, name, model, columns)
        results.append(metrics)
        if name == "catboost_history":
            predictions.loc[predictions.selected_for_review].to_csv(out / "review_queue.csv")

    comparison = pd.concat(results, ignore_index=True)
    comparison.to_csv(out / "comparison.csv", index=False)
    print(comparison.loc[comparison.daily_review_cap.eq(50)].round(3).to_string(index=False))
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    run(args.seed, args.output)
