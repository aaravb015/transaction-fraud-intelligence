"""Static safeguards for the one-time reserved-period evaluator.

Nothing in this file calls the extended simulator or materializes days 102--119.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fraud_final as final
import fraud_v2 as frozen


def test_final_protocol_constants_match_frozen_v2():
    assert final.verify_frozen_engine() == final.FROZEN_ENGINE_SHA256
    assert tuple(frozen.SEEDS) == (42, 123, 2025, 31415, 27182)
    assert tuple(frozen.MODELS) == (
        "rules", "logistic_history", "catboost_current", "catboost_history"
    )
    assert final.PRIMARY_MODEL == "catboost_history"
    assert final.FinalWindow().review_budgets == (20, 50, 100)
    assert final.FINAL_START == frozen.Config().test_at
    assert final.FINAL_END == frozen.START + pd.Timedelta(days=120)
    assert set(final.EXPECTED_DEVELOPMENT_HASHES) == set(frozen.SEEDS)


def test_static_preflight_compiles_but_does_not_materialize(monkeypatch):
    monkeypatch.setattr(final, "build_extended_simulator", lambda: pytest.fail("must not execute"))
    report = final.verify_static_contract()
    assert report["reserved_period_materialized"] is False
    assert report["reserved_period_evaluated"] is False
    assert report["patch_anchor_count"] == 7
    assert len(report["continuation_source_sha256"]) == 64


def test_continuation_probability_is_predeclared_duration_scaling():
    for probability in (0.25, 0.30, 0.35):
        expected = 1 - (1 - probability) ** (18 / 102)
        assert final.conditional_extension_probability(probability) == expected


def test_source_derived_patch_preserves_required_mechanics():
    source, anchors = final._continuation_source()
    ast.parse(source)
    assert len(anchors) == 7
    assert "def _simulate_extended" in source
    assert "for args in boundary_spillovers" in source
    assert "selected_set = {int(x) for x in selected}" in source
    assert "add_final" in source
    assert "_final_range_guard(tx, cfg)" in source
    # The frozen source is only transformed in memory; the file remains exact.
    assert hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest() == final.FROZEN_ENGINE_SHA256


def test_authorized_range_guard_rejects_outside_declared_horizon():
    cfg = frozen.Config()
    valid = pd.DataFrame({"timestamp": [final.FINAL_START, final.FINAL_END - pd.Timedelta(seconds=1)]})
    final._final_range_guard(valid, cfg)
    with pytest.raises(ValueError, match="outside days 0--119"):
        final._final_range_guard(pd.DataFrame({"timestamp": [final.FINAL_END]}), cfg)
    with pytest.raises(ValueError, match="outside days 0--119"):
        final._final_range_guard(pd.DataFrame({"timestamp": [frozen.START - pd.Timedelta(seconds=1)]}), cfg)


def test_official_entrypoint_is_not_invoked_by_import_or_ci_script():
    root = Path(__file__).resolve().parents[1]
    checker = (root / "scripts" / "check_final_notebook.py").read_text(encoding="utf-8")
    assert "--official-final-evaluation" not in checker
    assert "NotebookClient" not in checker
    source = inspect.getsource(final.main)
    assert "--official-final-evaluation" in source
    assert "if not args.official_final_evaluation" in source

