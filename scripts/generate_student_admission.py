"""Generate student_admission.csv for Logistic Regression project."""
import os

import numpy as np
import pandas as pd

np.random.seed(42)

N = 250
N_ADMIT = 125

# Admitted students: higher GPA, SAT, attendance on average
gpa = np.concatenate([
    np.clip(np.random.normal(3.5, 0.35, N_ADMIT), 2.0, 4.0),
    np.clip(np.random.normal(2.7, 0.4, N - N_ADMIT), 1.5, 3.8),
])
sat = np.concatenate([
    np.clip(np.random.normal(1220, 90, N_ADMIT).astype(int), 900, 1600),
    np.clip(np.random.normal(1050, 100, N - N_ADMIT).astype(int), 800, 1400),
])
extracurricular = np.concatenate([
    np.clip(np.random.normal(5.0, 2.0, N_ADMIT), 0, 20),
    np.clip(np.random.normal(2.5, 1.5, N - N_ADMIT), 0, 15),
])
attendance = np.concatenate([
    np.clip(np.random.normal(93, 5, N_ADMIT), 75, 100),
    np.clip(np.random.normal(82, 8, N - N_ADMIT), 55, 95),
])
admitted = np.array([1] * N_ADMIT + [0] * (N - N_ADMIT))

# Shuffle rows together
idx = np.random.permutation(N)
gpa = gpa[idx]
sat = sat[idx]
extracurricular = extracurricular[idx]
attendance = attendance[idx]
admitted = admitted[idx]

df = pd.DataFrame(
    {
        "GPA": np.round(gpa, 2),
        "SAT": sat,
        "Extracurricular_Hours": np.round(extracurricular, 1),
        "Attendance_Pct": np.round(attendance, 1),
        "Admitted": admitted,
    }
)

# Introduce missing values (~7%) in feature columns only
for col in ["GPA", "SAT", "Extracurricular_Hours", "Attendance_Pct"]:
    mask = np.random.rand(N) < 0.07
    df.loc[mask, col] = np.nan

out_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Datasets",
    "student_admission.csv",
)
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["Admitted"].value_counts())
print(df.isnull().sum())
