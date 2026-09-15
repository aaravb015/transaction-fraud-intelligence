"""Generate the reviewed one-time final-evaluation Colab notebook."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ENGINE_SHA256 = "6debf4f7501ee0f937bd53c9992f8ecc642c946039dd653aa284229078d2cb1e"


def build():
    frozen_engine = (ROOT / "src" / "fraud_v2.py").read_text(encoding="utf-8")
    final_engine = (ROOT / "src" / "fraud_final.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "scripts" / "final_notebook_bootstrap.py").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    actual = hashlib.sha256(frozen_engine.encode()).hexdigest()
    if actual != FROZEN_ENGINE_SHA256:
        raise RuntimeError("Refusing to build final notebook from a changed V2 engine: " + actual)
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "id": "final-%02d" % len(cells), "metadata": {},
                "source": source.strip().splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """# Transaction Fraud Intelligence — one-time final evaluation

V2 development is complete and frozen. This notebook performs the **first and only intentional evaluation** of the synthetic reserved period, days **102–119**.

The procedure is already fixed: seeds **[42, 123, 2025, 31415, 27182]**; models `rules`, `logistic_history`, `catboost_current`, `catboost_history`; daily review budgets **20/50/100**; headline operating point **50/day**; and preselected primary model **`catboost_history`**.

Do not edit this notebook, tune anything, choose a subset of seeds, or rerun after seeing outcomes. Open a fresh CPU runtime and select **Runtime → Run all once**. If the run stops with an error, preserve the output and obtain an implementation review before doing anything else.""")
    add("markdown", """## 1. Frozen source and isolated runtime

The cell below embeds the byte-verified V2 engine and the separately reviewed final evaluator. Package versions and the full experimental configuration remain frozen. No repository clone or data upload is required.""")
    add("code", "FROZEN_ENGINE_SOURCE = " + json.dumps(frozen_engine, ensure_ascii=False)
        + "\nFINAL_ENGINE_SOURCE = " + json.dumps(final_engine, ensure_ascii=False)
        + "\nREQUIREMENTS = " + json.dumps(requirements)
        + "\nFROZEN_ENGINE_SHA256 = " + json.dumps(FROZEN_ENGINE_SHA256)
        + "\n\n" + bootstrap)
    add("markdown", """## 2. Execute the frozen final evaluation

This call materializes and scores the reserved period. Before any final score is accepted, every development row and every development feature must match frozen V2 exactly for all five seeds.""")
    add("code", "run_final_once()")
    add("markdown", """## 3. Headline reserved-period evidence

The table shows every model at the predeclared 50-review/day operating point. `catboost_history` remains the primary model regardless of its final rank.""")
    add("code", "show_final_table('final_comparison.csv', filters={'daily_review_cap':'50'}, columns=['seed','model','average_precision','roc_auc','precision','fraud_recall','fraud_attempt_value_capture','legitimate_reviewed','false_positive_rate'], limit=20)")
    add("markdown", """## 4. Integrity evidence and download

The integrity table must show exact development-row and feature matches for all five seeds. The ZIP contains the complete 60-row result grid, daily metrics, scenario/context breakdowns, per-seed predictions, split audits, integrity checks and final manifest.""")
    add("code", "show_final_table('prefix_integrity_checks.csv')\ndownload_final_results()")
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": "fraud_final_evaluation.ipynb", "provenance": []},
            "frozen_engine_sha256": FROZEN_ENGINE_SHA256,
            "final_engine_sha256": hashlib.sha256(final_engine.encode()).hexdigest(),
            "reserved_period_execution_forbidden_in_ci": True,
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "notebooks" / "fraud_final_evaluation.ipynb"
    expected = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != expected:
            raise SystemExit("Final notebook differs from reviewed source. Run python scripts/build_final_notebook.py.")
        print("Final notebook matches its reviewed sources and frozen engine hash.")
    else:
        target.write_text(expected, encoding="utf-8")
        print("Generated", target)


if __name__ == "__main__":
    main()

