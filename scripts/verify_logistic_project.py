import json
import os

import pandas as pd

base = r"c:\Users\A7MED\Desktop\New folder (32)"
csv_path = os.path.join(base, "Datasets", "student_admission.csv")
nb_path = os.path.join(base, "Project", "Student_Admission_Project.ipynb")

df = pd.read_csv(csv_path)
assert len(df) == 250
assert list(df.columns) == [
    "GPA",
    "SAT",
    "Extracurricular_Hours",
    "Attendance_Pct",
    "Admitted",
]
assert set(df["Admitted"].unique()) <= {0, 1}
assert 115 <= df["Admitted"].sum() <= 135
print("CSV OK:", df.shape, df["Admitted"].value_counts().to_dict())

os.chdir(os.path.join(base, "Project"))
df2 = pd.read_csv("../Datasets/student_admission.csv")
print("Path from Project/ OK:", df2.shape)

with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)
md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
empty_code = sum(
    1
    for c in nb["cells"]
    if c["cell_type"] == "code" and "".join(c["source"]).strip() == "# Write your code here"
)
outputs = sum(len(c.get("outputs", [])) for c in nb["cells"] if c["cell_type"] == "code")
print(f"Notebook OK: {len(nb['cells'])} cells ({md} md, {code} code)")
print(f"  empty code cells: {empty_code}, outputs: {outputs}")
print("All checks passed.")
