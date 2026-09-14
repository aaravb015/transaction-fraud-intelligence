"""Validate notebook syntax and optionally execute the small-data experiment."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import nbformat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plain", action="store_true", help="Execute the exact cells without a Jupyter socket; CI uses the real kernel.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root/"scripts/build_notebook.py"), "--check"], check=True)
    path = root / "notebooks" / "fraud_v2.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for index, cell in enumerate(code_cells, start=1):
        ast.parse(cell.source, filename="notebook_cell_%02d.py" % index)
    print("Notebook schema and %d Python cells validated." % len(code_cells))
    if not args.execute:
        return

    from nbclient import NotebookClient

    os.environ["FRAUD_SMOKE_TEST"] = "1"
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    # Preload the host's scientific stack to exercise isolation from loaded modules.
    notebook.cells.insert(0, nbformat.v4.new_code_cell(
        "import sys\nHOST_NUMPY_BEFORE = None\ntry:\n import numpy\n HOST_NUMPY_BEFORE = numpy.__version__\nexcept ImportError:\n pass\nprint('Host NumPy before:', HOST_NUMPY_BEFORE)"))
    notebook.cells.append(nbformat.v4.new_code_cell(
        "if HOST_NUMPY_BEFORE is not None:\n assert sys.modules['numpy'].__version__ == HOST_NUMPY_BEFORE\nprint('Host modules unchanged; isolated worker completed.')"))
    if args.plain:
        namespace = {"__name__": "__main__"}
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                exec(compile(cell.source, "notebook_cell_%02d.py" % i, "exec"), namespace)
    else:
        # Start this interpreter's kernel, rather than an unrelated global kernel.
        from jupyter_client.kernelspec import KernelSpecManager
        kernel_dir = Path(tempfile.mkdtemp(prefix="fraud-kernel-"))
        (kernel_dir/"kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Fraud check", "language": "python"}))
        manager = KernelSpecManager()
        manager.install_kernel_spec(str(kernel_dir), kernel_name="fraud-check", user=True)
        client = NotebookClient(notebook, timeout=1800, kernel_name="fraud-check",
                                resources={"metadata": {"path": str(root)}})
        try:
            client.execute()
        finally:
            nbformat.write(notebook, output / "executed_v2_smoke.ipynb")
    result_files = sorted(output.glob("v2_work_*/result.json"), key=lambda p: p.stat().st_mtime)
    if not result_files:
        raise AssertionError("Notebook execution produced no V2 result record.")
    result = json.loads(result_files[-1].read_text())
    with zipfile.ZipFile(result["archive"]) as z:
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["implementation_version"] == "0.2.0"
        assert manifest["locked_test_evaluated"] is False
        assert manifest["locked_test_materialized"] is False
        assert manifest["configuration"]["mode"] == "smoke"
        assert manifest["seed_list"] == [42,123,2025,31415,27182]
    print("V2 small-data notebook execution and ZIP contract passed.")
    print("Executed notebook and experiment outputs:", output)


if __name__ == "__main__":
    main()
