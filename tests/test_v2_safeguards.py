"""Regression tests for temporal boundaries, history math and comparison design."""
import ast
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import fraud_v2 as f


def attempts(days, amounts=None):
    timestamps = [f.START+pd.Timedelta(days=d) for d in days]
    return pd.DataFrame({"transaction_id": ["p%02d" % i for i in range(len(days))],
        "timestamp": timestamps, "customer_id": ["c"]*len(days), "merchant_id": ["m"]*len(days),
        "device_id": ["d"]*len(days), "amount": amounts or [100.]*len(days),
        "country": ["IN"]*len(days), "category": ["retail"]*len(days),
        "status": ["succeeded"]*len(days),
        "outcome_available_at": [t+pd.Timedelta(seconds=60) for t in timestamps]})


def test_original_four_comparators_and_primary_features_are_preserved():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root/"notebooks/fraud_v1.ipynb").read_text())
    original = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in ("NUMERIC_FEATURES", "CATEGORICAL_FEATURES"):
                        original[name] = ast.literal_eval(node.value)
    assert f.FEATURES == original["NUMERIC_FEATURES"] + original["CATEGORICAL_FEATURES"]
    assert f.MODELS == ("rules", "logistic_history", "catboost_current", "catboost_history")
    members = [feature for family in f.FEATURE_FAMILIES.values() for feature in family]
    assert len(members) == len(set(members))
    assert set(members) == set(f.NUMERIC_FEATURES)


def test_disjoint_longitudinal_windows_and_future_invariance():
    cfg = f.Config()
    # At day 48, recent [41,48) has 3 attempts (100,200,300).
    # Prior baseline [11,41) has 3 attempts, each 100. Boundaries do not overlap.
    probe = attempts([0, 10, 15, 20, 40, 41, 41, 47, 48], [100,100,100,100,100,100,200,300,999])
    result = f.build_features(probe, cfg)
    row = result.loc["p08"]
    assert row.attempts_7d == 3
    assert row.attempt_value_7d == 600
    assert np.isclose(row.count_ratio_7d_prior30, 30/7)
    assert np.isclose(row.attempt_value_ratio_7d_prior30, 60/7)
    assert np.isclose(row.mean_ticket_ratio_7d_prior30, 2)
    assert result.loc["p05", "prior_count"] == result.loc["p06", "prior_count"] == 5
    future = attempts([55], [5000])
    future["transaction_id"] = ["future"]
    extended = f.build_features(pd.concat([probe, future]), cfg)
    pd.testing.assert_frame_equal(result, extended.loc[result.index])


def test_delayed_failure_and_same_time_cross_account_device_use():
    cfg = f.Config()
    probe = attempts([0, 0, 30/86400, 120/86400])
    probe["customer_id"] = ["a", "b", "a", "a"]
    probe.loc[0, "status"] = "failed"
    features = f.build_features(probe, cfg)
    assert features.loc["p00", "prior_accounts_on_device"] == 0
    assert features.loc["p01", "prior_accounts_on_device"] == 0
    assert features.loc["p02", "prior_accounts_on_device"] == 2
    assert features.loc["p02", "known_failures_1h"] == 0
    assert features.loc["p03", "known_failures_1h"] == 1


def test_label_maturity_excludes_recent_unconfirmed_rows():
    cfg = f.Config()
    tx = attempts([20, 21, 71, 72, 73, 85, 87, 88])
    tx["is_fraud"] = [0,1,1,0,1,1,0,1]
    tx["label_available_at"] = tx.timestamp+pd.Timedelta(days=7)
    masks, _ = f.temporal_split(tx, cfg)
    assert list(tx.index[masks["train"]]) == [0,1]
    assert list(tx.index[masks["early_stop"]]) == [3,4]
    assert list(tx.index[masks["validation"]]) == [6,7]


def test_reserved_timestamp_rejected_before_model_or_label_access():
    cfg = f.Config()
    probe = attempts([102]).set_index("transaction_id")
    model = Mock()
    state = {"tx": probe, "masks": {"validation": pd.Series(True, index=probe.index)}}
    with pytest.raises(ValueError, match="Reserved final-period"):
        f.score_detector(state, cfg, "catboost_history", model, f.FEATURES)
    model.predict_proba.assert_not_called()
    # No labels or other schema columns are needed for the boundary to reject it.
    with pytest.raises(ValueError, match="Reserved final-period"):
        f.build_features(probe[["timestamp"]], cfg)
    with pytest.raises(ValueError, match="Reserved final-period"):
        f.evaluate_scores(probe[["timestamp"]], [0.1], "rules", cfg)


def test_primary_feature_frame_does_not_depend_on_audit_labels():
    cfg = f.Config()
    probe = attempts([0,1,2,40,41,42,48])
    a = f.build_features(probe, cfg)
    probe["is_fraud"] = [1,0,1,0,1,0,1]
    probe["fraud_scenario"] = "injected_secret"
    probe["legitimate_context"] = "hidden_context"
    probe["simulator_base_amount"] = 99999
    pd.testing.assert_frame_equal(a, f.build_features(probe, cfg))


def test_daily_capacity_includes_days_without_candidates():
    cfg = f.Config()
    tx = attempts([86,86,86,86,86]).set_index("transaction_id")
    tx["is_fraud"] = [0,1,0,1,0]
    tx["fraud_scenario"] = ["legitimate","low_and_slow","legitimate","card_testing","legitimate"]
    tx["legitimate_context"] = "routine"
    metrics, daily, predictions = f.evaluate_scores(tx, [0.1,.8,.2,.7,.3], "rules", cfg)
    budget = daily.loc[daily.daily_review_cap.eq(20)]
    assert len(budget) == 16 and int(budget.transactions.eq(0).sum()) == 15
    assert budget.unused_capacity.sum() == 20*16-5
    assert not budget.budget_exhausted.any()
    assert metrics.review_count.eq(5).all()
    assert predictions.selected_for_review.all()
