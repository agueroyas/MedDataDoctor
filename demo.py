# Copyright (c) 2026 Abdulrahman Muhammed Abdalnafea
# MedData Doctor — https://github.com/agueroyas/MedDataDoctor
# Author profile: https://github.com/agueroyas
# See LICENSE for permitted use. Third-party components retain their own licenses.

"""Synthetic demo data (NOT real patients) with deliberate flaws, so every check has something to find."""
import numpy as np
import pandas as pd


def make_demo_df(n=800, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 80, n)
    bmi = rng.normal(32, 7, n).clip(18, 60)
    glucose = rng.normal(120, 30, n).clip(60, 250)
    bp = rng.normal(72, 12, n).clip(40, 110)
    skin = rng.normal(29, 10, n).clip(7, 70)
    insulin = rng.lognormal(4.8, 0.7, n).clip(15, 600)
    pregnancies = rng.poisson(3, n)

    logit = -8.8 + 0.045 * glucose + 0.05 * bmi + 0.02 * age + rng.logistic(0, 0.8, n)
    outcome = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    # Flaw 1: a column recorded AFTER diagnosis (leaks the answer, with 1% noise)
    on_insulin_therapy = np.where(rng.random(n) < 0.01, 1 - outcome, outcome)
    # Flaw 2: impossible zeros that really mean "not measured"
    bp[rng.random(n) < 0.05] = 0
    skin[rng.random(n) < 0.29] = 0
    insulin[rng.random(n) < 0.45] = 0
    bmi[rng.random(n) < 0.01] = 0
    # Flaw 3: one group dominates
    sex = np.where(rng.random(n) < 0.88, "M", "F")

    df = pd.DataFrame({
        "Pregnancies": pregnancies, "Glucose": glucose.round(0), "BloodPressure": bp.round(0),
        "SkinThickness": skin.round(0), "Insulin": insulin.round(0), "BMI": bmi.round(1),
        "Age": age, "Sex": sex, "OnInsulinTherapy": on_insulin_therapy, "Outcome": outcome,
    })
    # Flaw 4: duplicate rows (~3%)
    return pd.concat([df, df.sample(int(n * 0.03), random_state=seed)], ignore_index=True)
