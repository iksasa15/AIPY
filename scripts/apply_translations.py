"""Apply Arabic cell translations to notebooks in Regression_Arabic/."""
import importlib.util
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AR_DIR = os.path.join(BASE, "Regression_Arabic")
TRANS_DIR = os.path.join(BASE, "scripts", "translations")

NOTEBOOKS = [
    "Simple_linear_regression.ipynb",
    "Multiple_linear_regression.ipynb",
    "Decision_Tree_regression.ipynb",
    "Random_Forest_regression.ipynb",
    "Regularization_regression.ipynb",
    "SVR_regression.ipynb",
]


def load_cells(module_name: str) -> dict:
    path = os.path.join(TRANS_DIR, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CELLS


def source_to_lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return [line for line in text.splitlines(keepends=True)]


def apply_notebook(filename: str) -> None:
    module_name = filename.replace(".ipynb", "")
    cells_map = load_cells(module_name)
    nb_path = os.path.join(AR_DIR, filename)

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    if len(nb["cells"]) != len(cells_map):
        raise ValueError(
            f"{filename}: notebook has {len(nb['cells'])} cells, "
            f"translation has {len(cells_map)}"
        )

    for i, cell in enumerate(nb["cells"]):
        if i not in cells_map:
            raise KeyError(f"{filename}: missing translation for cell {i}")
        cell["source"] = source_to_lines(cells_map[i])
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    nb["metadata"].setdefault("language_info", {})["name"] = "python"

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

    print(f"OK {filename}: {len(cells_map)} cells translated, outputs cleared")


def main() -> None:
    sys.path.insert(0, TRANS_DIR)
    for fn in NOTEBOOKS:
        apply_notebook(fn)
    print("All notebooks updated.")


if __name__ == "__main__":
    main()
