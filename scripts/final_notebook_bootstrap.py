"""Standard-library frontend for the one-time final-evaluation notebook."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

from IPython.display import HTML, FileLink, display


def prepare_final_runtime():
    print("Notebook Python:", platform.python_version(), flush=True)
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError("This frozen release supports Python 3.11--3.13.")
    digest = hashlib.sha256(REQUIREMENTS.encode()).hexdigest()[:12]
    runtime = Path.cwd() / ".fraud_final_runtime" / (
        "py%d%d_" % sys.version_info[:2] + digest
    )
    packages = runtime / "packages"
    marker = runtime / "installed.txt"
    runtime.mkdir(parents=True, exist_ok=True)
    requirements = runtime / "requirements.txt"
    requirements.write_text(REQUIREMENTS, encoding="utf-8")
    if not marker.exists() or marker.read_text(encoding="utf-8") != REQUIREMENTS:
        print("Installing the frozen experiment packages in isolation.", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "--only-binary=:all:", "--ignore-installed", "--upgrade", "--target",
             str(packages), "-r", str(requirements)],
            capture_output=True, text=True,
        )
        if result.returncode:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("Dependency installation failed; final evaluation did not start.")
        marker.write_text(REQUIREMENTS, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(packages), PYTHONNOUSERSITE="1", MPLBACKEND="Agg")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    work = Path.cwd() / "outputs" / ("final_work_" + run_id)
    work.mkdir(parents=True)
    frozen_engine = work / "fraud_v2.py"
    final_engine = work / "fraud_final.py"
    frozen_engine.write_text(FROZEN_ENGINE_SOURCE, encoding="utf-8")
    final_engine.write_text(FINAL_ENGINE_SOURCE, encoding="utf-8")
    if hashlib.sha256(FROZEN_ENGINE_SOURCE.encode()).hexdigest() != FROZEN_ENGINE_SHA256:
        raise RuntimeError("Embedded frozen engine hash mismatch; final evaluation did not start.")
    config = {
        "customers": 1200, "days": 120, "merchants": 240,
        "label_delay_days": 7, "fraud_customer_fraction": 0.35,
        "review_budgets": [20, 50, 100], "main_review_budget": 50,
        "boosting_iterations": 600, "threads": 2, "mode": "final_evaluation",
    }
    config_path = work / "configuration.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Frozen seeds: 42, 123, 2025, 31415, 27182", flush=True)
    print("Preselected primary model: catboost_history", flush=True)
    print("Reserved period: days 102--119; this notebook is the one official evaluation.", flush=True)
    return work, final_engine, config_path, env


WORK, FINAL_ENGINE, CONFIG_PATH, WORKER_ENV = prepare_final_runtime()


def run_final_once():
    command = [
        sys.executable, "-S", "-u", str(FINAL_ENGINE),
        "--work-dir", str(WORK), "--config", str(CONFIG_PATH),
        "--official-final-evaluation",
    ]
    process = subprocess.Popen(
        command, env=WORKER_ENV, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError("Final evaluation stopped. Do not rerun; share the complete error output first.")


def final_result_dir():
    record = json.loads((WORK / "final_result.json").read_text(encoding="utf-8"))
    return Path(record["output_directory"])


def show_final_table(filename, columns=None, filters=None, limit=20):
    with (final_result_dir() / filename).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = columns or reader.fieldnames
        rows = [r for r in reader if not filters or all(r.get(k) == str(v) for k, v in filters.items())]
    def format_cell(value):
        try:
            return "%.4g" % float(value) if value else ""
        except ValueError:
            return value
    header = "".join("<th>" + html.escape(c) + "</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join("<td>" + html.escape(format_cell(r.get(c, ""))) + "</td>" for c in columns) + "</tr>"
        for r in rows[:limit]
    )
    display(HTML("<table><thead><tr>" + header + "</tr></thead><tbody>" + body + "</tbody></table>"))
    if len(rows) > limit:
        print("Showing", limit, "of", len(rows), "rows; the ZIP contains the full table.")


def download_final_results():
    record = json.loads((WORK / "final_result.json").read_text(encoding="utf-8"))
    archive = Path(record["archive"])
    print("Final output contract verified. Download this ZIP:", archive)
    try:
        from google.colab import files
    except ImportError:
        display(FileLink(str(archive.relative_to(Path.cwd()))))
    else:
        files.download(str(archive))

