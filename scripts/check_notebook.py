"""Validate notebook syntax and optionally execute the small-data experiment."""
import argparse
import ast
import os
from pathlib import Path

import nbformat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "fraud_v1.ipynb"
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
    os.environ["FRAUD_SKIP_INSTALL"] = "1"
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    client = NotebookClient(
        notebook, timeout=600, kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output / "executed_smoke.ipynb")
    print("Small-data notebook execution passed.")
    print("Executed notebook and experiment outputs:", output)


if __name__ == "__main__":
    main()
