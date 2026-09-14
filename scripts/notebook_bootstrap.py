"""Standard-library notebook frontend; scientific packages run in isolation.

The notebook generator supplies ENGINE_SOURCE and REQUIREMENTS before this code.
"""
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

from IPython.display import HTML, Image, FileLink, display


def prepare_runtime():
    print("Notebook Python:", platform.python_version(), flush=True)
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError("This release supports Python 3.11–3.13. Choose a supported Colab runtime version.")
    digest = hashlib.sha256(REQUIREMENTS.encode()).hexdigest()[:12]
    runtime = Path.cwd()/".fraud_v2_runtime"/("py%d%d_" % sys.version_info[:2] + digest)
    packages = runtime/"packages"
    marker = runtime/"installed.txt"
    runtime.mkdir(parents=True, exist_ok=True)
    requirements = runtime/"requirements.txt"
    requirements.write_text(REQUIREMENTS, encoding="utf-8")
    if not marker.exists() or marker.read_text() != REQUIREMENTS:
        print("Installing the experiment's isolated packages; the notebook kernel is unchanged.", flush=True)
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--only-binary=:all:", "--ignore-installed", "--upgrade", "--target", str(packages),
            "-r", str(requirements)], capture_output=True, text=True)
        if result.returncode:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("Dependency installation failed. The experiment has not started.")
        marker.write_text(REQUIREMENTS, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(packages), PYTHONNOUSERSITE="1", MPLBACKEND="Agg")
    # Import the binaries in a fresh process; do not hide incompatible imports.
    probe = "import numpy,pandas,scipy,sklearn,catboost,matplotlib,joblib; import importlib.metadata as m,json,platform; print('Experiment Python: '+platform.python_version()); print(json.dumps({p:m.version(p) for p in ['numpy','pandas','scipy','scikit-learn','catboost','matplotlib','joblib']},indent=2))"
    subprocess.run([sys.executable, "-S", "-c", probe], env=env, check=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    work = Path.cwd()/"outputs"/("v2_work_"+run_id)
    work.mkdir(parents=True)
    engine = work/"fraud_v2.py"
    engine.write_text(ENGINE_SOURCE, encoding="utf-8")
    smoke = os.environ.get("FRAUD_SMOKE_TEST") == "1"
    config = {"customers": 360 if smoke else 1200, "boosting_iterations": 100 if smoke else 600,
              "mode": "smoke" if smoke else "development"}
    config_path = work/"configuration.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Five fixed seeds: 42, 123, 2025, 31415, 27182. Mode:", config["mode"], flush=True)
    print("Reserved final period: not generated, inspected or evaluated.", flush=True)
    return work, engine, config_path, env


WORK, ENGINE, CONFIG_PATH, WORKER_ENV = prepare_runtime()


def run_stage(stage):
    command = [sys.executable, "-S", "-u", str(ENGINE), "--stage", stage,
               "--work-dir", str(WORK), "--config", str(CONFIG_PATH)]
    process = subprocess.Popen(command, env=WORKER_ENV, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError("The '"+stage+"' stage failed. Stop here and share the error output.")


def result_dir():
    return Path(json.loads((WORK/"progress.json").read_text())["output_directory"])


def show_table(filename, columns=None, filters=None, limit=12):
    with (result_dir()/filename).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = columns or reader.fieldnames
        rows = [r for r in reader if not filters or all(r.get(k) == str(v) for k, v in filters.items())]
    def format_cell(value):
        try:
            return "%.4g" % float(value) if value else ""
        except ValueError:
            return value
    header = "".join("<th>"+html.escape(c)+"</th>" for c in columns)
    body = "".join("<tr>"+"".join("<td>"+html.escape(format_cell(r.get(c, "")))+"</td>" for c in columns)+"</tr>" for r in rows[:limit])
    display(HTML("<table><thead><tr>"+header+"</tr></thead><tbody>"+body+"</tbody></table>"))
    if len(rows) > limit:
        print("Showing", limit, "of", len(rows), "rows. The ZIP contains the full table.")


def show_chart(filename):
    display(Image(filename=str(result_dir()/filename)))


def download_results():
    result = json.loads((WORK/"result.json").read_text())
    archive = Path(result["archive"])
    print("V2 output contract verified. ZIP:", archive)
    if os.environ.get("FRAUD_SMOKE_TEST") == "1":
        display(FileLink(str(archive.relative_to(Path.cwd()))))
        return
    try:
        from google.colab import files
    except ImportError:
        display(FileLink(str(archive.relative_to(Path.cwd()))))
    else:
        files.download(str(archive))
