"""Generate a self-contained notebook from the reviewed engine and frontend."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    engine = (ROOT/"src/fraud_v2.py").read_text(encoding="utf-8")
    bootstrap = (ROOT/"scripts/notebook_bootstrap.py").read_text(encoding="utf-8")
    requirements = (ROOT/"requirements.txt").read_text(encoding="utf-8")
    cells = []
    def add(kind, source):
        cell = {"cell_type": kind, "id": "v2-%02d" % len(cells), "metadata": {},
                "source": source.strip().splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    def section(title, explanation, code=None):
        add("markdown", "## " + title + "\n\n" + explanation)
        if code:
            add("code", code)

    add("markdown", """# Transaction Fraud Intelligence — V2

## 1. Objective and assumptions

**Does customer history improve fraud ranking within a fixed daily review budget?**

This is a synthetic development experiment. V2 keeps the four V1 comparators, hardens the simulator, diagnoses low-and-slow misses, tests all five declared seeds, and measures six feature-family ablations. A lower score on a harder simulation is not a failure.

**Run:** open a fresh Colab CPU runtime and select **Runtime → Run all**. The notebook installs its own isolated packages, runs every stage in order and offers a timestamped V2 results ZIP. No repository clone, dataset upload, package edits or kernel restart is required for supported Python 3.11–3.13.

The main development configuration has 1,200 customers and a nominal 120-day timeline. **Days 102 onward are reserved and never generated or inspected.** Every output uses development data only. Labels, scenario/context tags, hidden customer preferences and raw entity IDs cannot enter the model feature list.

Amounts are fictional standardized INR attempts, including failed payments. Scores are uncalibrated; value capture is not prevented loss or savings.""")
    section("2. Configuration and reproducibility", """The fixed seed list is **[42, 123, 2025, 31415, 27182]**. All seeds are reported, including in smoke mode. Full runs use 600 maximum boosting iterations and two CPU threads; CI uses 360 customers and 100 iterations with the same timeline, seeds and methodology.

The large source string below is the embedded experiment engine. It keeps this notebook self-contained and is generated from the readable `src/fraud_v2.py`. Scientific packages run in fresh subprocesses so they cannot conflict with modules already loaded in Colab's notebook kernel. Both Python and package versions are printed before simulation.

Do not alter parameters or select seeds after seeing results. The final period is not an available run option.""",
        "ENGINE_SOURCE = " + json.dumps(engine, ensure_ascii=False) + "\nREQUIREMENTS = " + json.dumps(requirements) + "\n\n" + bootstrap)
    section("3. Simulation", """Normal activity includes travel, phone changes, large purchases, legitimate bursts and shared household devices. Fraud sometimes uses a familiar device, familiar merchant, home country, ordinary amount, ordinary hour and no failed-attempt burst.

Account takeover, card testing and low-and-slow remain separate audit scenarios. The simulator uses separate normal/fraud random streams and drops reserved-time candidates before creating outcomes or labels. A paired V1-like reference is used later for diagnostics; it is not a reproduction of the archived V1 dataset.""", "run_stage('simulation')")
    section("4. Point-in-time history builder", """The original 21 primary features are preserved. Equal-timestamp attempts are scored together before history updates. Failed outcomes become available only at their release time.

Additional longitudinal measures are **diagnostics first**: 7/14/30-day count and attempted-value changes, ticket-size changes, merchant novelty/diversity and category concentration. Recent windows are compared against a disjoint earlier 30-day baseline, with explicit coverage requirements. Neither the current attempt nor later events enter those baselines.""", "run_stage('history')")
    section("5. Temporal split and label maturity", """Fit cutoff: day 72. Early-stopping labels must mature by day 86. Development validation: days 86–101. Both classes have a declared seven-day confirmation delay.

Checks cover future invariance for **all** features, same-timestamp batching, delayed outcomes, exclusion of the current amount, label maturity and the development boundary. Unlabelled observed activity may update history during maturation gaps.""", "run_stage('split')\nshow_table('split_audit.csv')")
    section("6. Rules baseline", "The original five illustrative rules and weights remain fixed. Their output is a 0–100 point score, not a fraud probability.", "run_stage('rules')")
    section("7. Logistic baseline", "The original history feature set is used. Imputation, scaling and category encoding are fitted on training rows only. A convergence failure stops the experiment.", "run_stage('logistic')")
    section("8. Current-transaction CatBoost", "Only log amount, hour sine/cosine, category and country enter this comparator. It is the control for the value of history.", "run_stage('current')")
    section("9. History CatBoost", "The original 21 features enter the primary history model. The same maximum tree budget and early-stopping rules apply across seeds. No parameter sweep is performed.", "run_stage('history_model')")
    section("10. Main validation comparison", """All four detectors use identical validation attempts and 20/50/100 daily review budgets. This is **offline end-of-day top-K**, not live blocking. Ties use transaction ID/event order.

False positives consume limited investigator attention. A fixed budget prevents a model from claiming better recall simply by reviewing everything. Daily summaries include zero-candidate days, exhausted budgets, underfilled days and unused slots. Daily means are unweighted summaries; the main table aggregates all selected attempts.""",
        "run_stage('comparison')\nshow_table('comparison.csv', columns=['model','daily_review_cap','review_count','fraud_reviewed','average_precision','roc_auc','precision','fraud_recall','false_positive_rate','legitimate_reviewed','fraud_attempt_value_capture'])\nshow_table('daily_variability_summary.csv', columns=['model','daily_review_cap','validation_days','days_budget_exhausted','days_fewer_candidates_than_budget','unused_review_slots','precision_daily_mean','precision_daily_std','recall_daily_min','recall_daily_max'])\nshow_chart('validation_comparison.png')")
    section("11. Scenario analysis and simulator separability", """Scenario recall shows which mechanisms remain difficult. The separation audit compares fraud and legitimate distributions and univariate discrimination for the eight strongest V1 signals, including missingness.

The untrained V1-like reference preserves normal-event streams while using the earlier fraud shortcut settings. The overlap audit reports familiar/ordinary conditions, including their intersection. This diagnoses how the task changed; a V1/V2 metric difference cannot be attributed solely to a better model.""",
        "run_stage('scenario')\nshow_table('scenario_breakdown.csv')\nshow_table('simulator_overlap_audit.csv', filters={'class':'fraud'}, columns=['world','condition','rows','count','share'], limit=16)\nshow_chart('feature_separation.png')")
    section("12. Low-and-slow diagnosis and conditional challenger", """Caught low-and-slow attempts, missed attempts and routine legitimate transactions are compared before selecting any extra model features. The ZIP includes distributions for ticket size, inter-attempt timing, 7/14/30-day changes, merchant/category behaviour, sharing, location and hours.

A predeclared gate requires at least five finite observations per comparison group, at least 50% coverage and an absolute standardized mean difference of 0.25. If any candidates pass, a **separate longitudinal challenger** uses that frozen list across all five seeds. It never replaces the primary four comparators.

Seed-42 challenger gains are exploratory because that validation informed selection. The other four worlds provide separate synthetic replication. Trade-off files include low-and-slow and total recall, precision, false positives, and which legitimate contexts absorb extra alerts. If no candidate qualifies, that outcome is exported explicitly.""",
        "run_stage('low_slow')\nshow_table('low_and_slow_overview.csv', limit=10)\nshow_table('longitudinal_feature_decisions.csv', columns=['feature','missed_finite_count','routine_finite_count','standardized_mean_difference','selected'], limit=15)\nshow_table('longitudinal_tradeoffs.csv', columns=['seed','status','baseline_low_and_slow_recall','challenger_low_and_slow_recall','delta_fraud_recall','delta_precision','delta_legitimate_reviewed'])")
    section("13. False-positive analysis", """For the provisional validation leader, inspect routine activity, travel, new phones, large purchases and legitimate bursts. The table reports total/flagged legitimate attempts, review rate, median and upper-quartile scores, and recurring evidence patterns.

Only a few representative false positives are shown. All four models' detailed cases remain in the ZIP. Evidence is an observed feature summary; it is not a causal explanation or proof that the model used each listed signal.""",
        "run_stage('false_positives')\nshow_table('legitimate_context_breakdown.csv', filters={'provisional_leader':'True'}, columns=['model','context','legitimate_count','legitimate_reviewed','alert_rate','score_median','score_q75','top_evidence_patterns'], limit=5)\nshow_table('false_positive_examples.csv', columns=['transaction_id','legitimate_context','amount','score','evidence_patterns'], limit=5)")
    section("14. Multi-seed robustness", """Every declared seed runs the same simulator settings, four detectors, feature definitions, time cutoffs and review budgets. A failed seed stops the run rather than disappearing from the report.

`robustness_summary.csv` contains per-seed rows plus mean, sample standard deviation and min/max for each model/budget. The chart reports average precision and rank for every seed. This measures variation between synthetic worlds, not a real-population confidence interval.""",
        "run_stage('robustness')\nshow_table('robustness_summary.csv', filters={'row_type':'mean','daily_review_cap':50}, columns=['model','average_precision','roc_auc','precision','fraud_recall','fraud_attempt_value_capture','false_positive_rate'], limit=4)\nshow_chart('robustness_ranking.png')\nshow_table('longitudinal_tradeoffs.csv', columns=['seed','status','evidence_role','delta_low_and_slow_recall','delta_fraud_recall','delta_precision','delta_legitimate_reviewed'], limit=5)")
    section("15. Feature-family ablation", """On the predeclared primary seed 42, retrain after removing each of six disjoint feature families: amount, velocity/recency, novelty, personal baselines, device network, and circadian timing. Category and country remain as background covariates.

The full model is the reference. Each ablation has the same maximum training budget and time splits, with its own early stopping. Negative metric deltas indicate deterioration for higher-is-better metrics; fewer legitimate reviews is desirable. These are bounded diagnostics, not an extensive search.""",
        "run_stage('ablation')\nshow_table('ablation_summary.csv', columns=['ablation','average_precision','precision','fraud_recall','false_positive_rate','legitimate_reviewed','fraud_attempt_value_capture','delta_average_precision','delta_fraud_recall'], limit=7)")
    section("16. Limitations and next decision", """- All evidence comes from simulated development data. Hardening can lower scores while improving credibility.
- The main four-model comparison is preserved; the conditional challenger is additional exploratory evidence.
- False-positive costs are represented through review capacity, not invented monetary savings.
- Scores are uncalibrated, evidence is descriptive, and top-K is offline.
- Seed robustness is not proof of generalisation to real clients. Ablations use one seed.
- No reserved final-period events or labels have been materialized, inspected or evaluated.
- APIs, deployed dashboards, public-data benchmarks, SHAP and further infrastructure are outside this pass.

**Next:** return the full V2 ZIP for independent review against the V1 baseline. Do not open the final period or choose a new seed based on these results.""")
    section("17. Export and download the V2 artefacts", """The export step validates the required files, the five-seed result grid and development-only prediction timestamps before creating the ZIP. It includes comparisons, daily/scenario/context reports, separability, low-and-slow diagnosis, robustness, ablations, predictions, cases, charts, saved primary models and a detailed manifest.

`locked_test_evaluated` and `locked_test_materialized` must both be **false**. Download the ZIP before closing Colab. The final period remains locked.""", "run_stage('export')\ndownload_results()")
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": "fraud_v2.ipynb", "provenance": []},
        "fraud_engine_sha256": hashlib.sha256(engine.encode()).hexdigest()}, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT/"notebooks/fraud_v2.ipynb"
    expected = json.dumps(build(), indent=2, ensure_ascii=False)+"\n"
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != expected:
            raise SystemExit("Notebook differs from reviewed source. Run python scripts/build_notebook.py.")
        print("Embedded source, frontend and dependency list match the notebook.")
    else:
        target.write_text(expected, encoding="utf-8")
        print("Generated", target)


if __name__ == "__main__":
    main()
