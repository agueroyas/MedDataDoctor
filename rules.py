# Copyright (c) 2026 Abdulrahman Muhammed Abdalnafea
# MedData Doctor — https://github.com/agueroyas/MedDataDoctor
# Author profile: https://github.com/agueroyas
# See LICENSE for permitted use. Third-party components retain their own licenses.

"""Plausibility rules for common clinical measurements.

Aliases are written in normalized form (lowercase, letters and digits only).
Short aliases (4 characters or fewer) must match the column name exactly;
longer aliases may also match the start or the end of the column name.
Ranges are deliberately generous: they catch impossible values, not unusual ones.
Have a clinician review this file before you quote results from it.
"""

RULES = [
    {"label": "Systolic blood pressure", "aliases": ["systolic", "sbp", "restingbp", "trestbps", "aphi"],
     "min": 50, "max": 300, "zero_missing": True, "unit": "mmHg"},
    {"label": "Diastolic blood pressure", "aliases": ["diastolic", "dbp", "bloodpressure", "diabp", "aplo"],
     "min": 20, "max": 200, "zero_missing": True, "unit": "mmHg"},
    {"label": "Heart rate", "aliases": ["heartrate", "pulse", "maxhr", "thalach", "thalch"],
     "min": 20, "max": 250, "zero_missing": True, "unit": "bpm"},
    {"label": "BMI", "aliases": ["bmi", "bodymassindex"],
     "min": 10, "max": 80, "zero_missing": True, "unit": "kg/m2"},
    {"label": "Age", "aliases": ["age"],
     "min": 0, "max": 120, "zero_missing": False, "unit": "years"},
    {"label": "Glucose", "aliases": ["glucose", "bloodsugar"],
     "min": 20, "max": 800, "zero_missing": True, "unit": "mg/dL (ignore if your data is in mmol/L)"},
    {"label": "Insulin", "aliases": ["insulin"],
     "min": 1, "max": 1000, "zero_missing": True, "unit": "uU/mL"},
    {"label": "Skin-fold thickness", "aliases": ["skinthickness", "triceps"],
     "min": 2, "max": 100, "zero_missing": True, "unit": "mm"},
    {"label": "Total cholesterol", "aliases": ["cholesterol", "chol"],
     "min": 50, "max": 700, "zero_missing": True, "unit": "mg/dL"},
    {"label": "Oxygen saturation", "aliases": ["spo2", "oxygensaturation", "o2sat"],
     "min": 50, "max": 100, "zero_missing": True, "unit": "%"},
    {"label": "Respiratory rate", "aliases": ["respiratoryrate", "resprate"],
     "min": 4, "max": 70, "zero_missing": True, "unit": "breaths/min"},
    {"label": "Hemoglobin", "aliases": ["hemoglobin", "haemoglobin", "hgb"],
     "min": 3, "max": 25, "zero_missing": True, "unit": "g/dL"},
    {"label": "Creatinine", "aliases": ["creatinine"],
     "min": 0.1, "max": 20, "zero_missing": True, "unit": "mg/dL"},
    {"label": "Number of pregnancies", "aliases": ["pregnancies"],
     "min": 0, "max": 20, "zero_missing": False, "unit": "count"},
]

# Column names containing these parts are often recorded AFTER the outcome is known.
SUSPICIOUS_NAME_PARTS = [
    "treatment", "therapy", "medication", "discharge", "outcome", "death",
    "died", "mortality", "readmi", "complication", "icd", "diagnosis", "postop",
]
