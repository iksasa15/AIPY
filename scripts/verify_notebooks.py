"""Smoke-test Arabic notebooks by executing code cells."""
import json
import os
import sys
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.show = lambda *args, **kwargs: None

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_notebook(nb_dir: str, name: str) -> None:
    os.chdir(nb_dir)
    path = os.path.join(nb_dir, name)
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


def main() -> None:
    targets = sys.argv[1:] if len(sys.argv) > 1 else [
        "Regression_Arabic/Simple_linear_regression.ipynb",
        "Regression_Arabic/SVR_regression.ipynb",
    ]
    for rel in targets:
        parts = rel.replace("\\", "/").split("/")
        nb_dir = os.path.join(BASE, parts[0])
        run_notebook(nb_dir, parts[1])
    print("\nSmoke tests passed.")


if __name__ == "__main__":
    main()
