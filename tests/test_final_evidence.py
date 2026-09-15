import json
from pathlib import Path

import pandas as pd

from src.final_eval_continuation import load_module, make_final_engine_source


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "final_evaluation"


def test_committed_final_evidence_is_internally_consistent():
    manifest = json.loads((EVIDENCE / "final_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((EVIDENCE / "final_output_contract_check.json").read_text(encoding="utf-8"))
    prefix = pd.read_csv(EVIDENCE / "prefix_integrity_checks.csv")

    assert manifest["final_evaluator_version"] == "0.3.0"
    assert manifest["preselected_primary_model"] == "catboost_history"
    assert manifest["prefix_integrity_passed_for_all_seeds"] is True
    assert manifest["official_evaluation_count_for_this_run"] == 1
    assert manifest["selection_after_final_outcomes"] is False
    assert manifest["synthetic_data_only"] is True
    assert manifest["attempted_value_is_not_prevented_loss"] is True

    assert contract == {
        "required_files": 8,
        "result_grid_rows": 60,
        "seed_prediction_sets": 5,
        "contract_passed": True,
    }

    assert set(prefix["seed"].astype(int)) == set(manifest["seed_list"])
    assert prefix["development_rows_exact_match"].all()
    assert prefix["development_features_exact_match"].all()
    expected = {int(k): v for k, v in manifest["expected_development_hashes"].items()}
    for row in prefix.itertuples(index=False):
        assert row.development_dataset_sha256 == expected[int(row.seed)]
        assert row.development_dataset_sha256 == row.expected_development_dataset_sha256


def test_final_continuation_builder_is_static_and_compilable_without_simulation():
    frozen_source = (ROOT / "src" / "fraud_v2.py").read_text(encoding="utf-8")
    patched = make_final_engine_source(frozen_source)

    assert patched.count("FINAL_EVALUATION_CONTINUATION_V1") == 1
    assert "final_days = cfg.days - dev_days" in patched
    assert "1 - (1 - cfg.fraud_customer_fraction) ** extension_share" in patched
    compile(patched, "<final-continuation-static-test>", "exec")


def test_dynamic_loader_handles_dataclasses_without_running_simulator():
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class Probe:\n"
        "    value: int = 1\n"
    )
    module = load_module("final_eval_test_probe", source)
    assert module.Probe().value == 1


def test_posthoc_prevalence_and_rule_tie_audits_cover_all_seeds():
    prevalence = pd.read_csv(EVIDENCE / "prevalence_audit.csv")
    rules = pd.read_csv(EVIDENCE / "rules_tie_audit.csv")

    assert set(prevalence.seed.astype(int)) == {42, 123, 2025, 31415, 27182}
    assert set(rules.seed.astype(int)) == {42, 123, 2025, 31415, 27182}
    assert prevalence.fraud_prevalence.between(0, 1).all()
    assert prevalence.precision_lift_over_prevalence.gt(1).all()
    assert rules.zero_score_share.gt(0.90).all()
    assert rules.selected_zero_score_share.between(0.35, 0.41).all()
