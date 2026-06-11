"""Smoke-test Arabic notebooks by executing code cells."""
import json
import os
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.show = lambda *args, **kwargs: None

BASE = r"c:\Users\A7MED\Desktop\New folder (32)\Regression_Arabic"
os.chdir(BASE)


def run_notebook(name: str) -> None:
    path = os.path.join(BASE, name)
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n=== {name} ({len(nb['cells'])} cells) ===")
    ns = {"__name__": "__main__", "display": print}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "!pip" in src:
            print(f"  cell {i}: skip pip")
            continue
        try:
            exec(compile(src, f"{name}:cell{i}", "exec"), ns)
            print(f"  cell {i}: OK")
        except Exception as e:
            print(f"  cell {i}: FAIL — {e}")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    for nb in ["Simple_linear_regression.ipynb", "SVR_regression.ipynb"]:
        run_notebook(nb)
    print("\nSmoke tests passed.")
