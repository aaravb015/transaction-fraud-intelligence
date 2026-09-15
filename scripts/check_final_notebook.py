"""Static-only checks for the final notebook; never materialize the reserve."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import nbformat


def main():
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "build_final_notebook.py"), "--check"], check=True)
    notebook = nbformat.read(root / "notebooks" / "fraud_final_evaluation.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for index, cell in enumerate(code_cells, start=1):
        ast.parse(cell.source, filename="final_notebook_cell_%02d.py" % index)
    if len(code_cells) != 4:
        raise AssertionError("Unexpected final notebook code-cell count.")
    if "run_final_once()" not in code_cells[1].source:
        raise AssertionError("The official final-evaluation call is not isolated in its own cell.")
    if notebook.metadata.get("reserved_period_execution_forbidden_in_ci") is not True:
        raise AssertionError("Missing CI non-execution marker.")
    print("Final notebook schema, source hashes and Python syntax passed static checks.")
    print("Reserved days 102--119 were not materialized or evaluated.")


if __name__ == "__main__":
    main()

