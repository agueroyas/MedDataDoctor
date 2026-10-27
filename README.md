# MedData Doctor

**A pre-training data quality audit for medical machine-learning datasets.**

MedData Doctor helps developers and data teams inspect a dataset before training a model. It highlights data-quality risks, potentially misleading fields, representation gaps, and suspicious predictors so that problems can be reviewed before they affect model evaluation.

> **Important:** MedData Doctor is a research and educational screening tool. It is not a medical device, does not provide diagnoses or treatment recommendations, and does not certify that a dataset is safe or clinically valid. Findings are signals to investigate with the data owner and qualified domain experts.

[![Live demo](https://img.shields.io/badge/Live%20Demo-Open%20App-1f6feb)](https://meddata-hpro.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Proprietary-lightgrey)](LICENSE)

## Why it exists

A model can appear to perform well even when the dataset contains hidden missing values, impossible measurements, duplicates, class imbalance, under-represented groups, or information that would not be available at prediction time. MedData Doctor provides a first-pass report to help teams spot and discuss these risks before model training.

## What it checks

- **Measurement plausibility:** compares supported numeric columns with broad, configurable ranges and flags values outside those ranges.
- **Potential hidden missing values:** identifies zero values in fields where zero may represent “not measured,” according to the configured rules.
- **Missing labels and basic dataset issues:** checks missing outcome values, duplicate rows, and likely high-cardinality identifiers.
- **Class balance and representation:** reports outcome-class balance and checks supported demographic group and age-band representation.
- **Potential label leakage:** measures how strongly each individual feature predicts a binary target and flags suspiciously strong features, including names that suggest information may be recorded after the outcome.
- **Before/after experiment:** compares a Random Forest baseline with a cleaned pipeline using repeated train/test splits. Results include ROC-AUC and balanced accuracy.
- **Downloadable report:** exports findings and experiment results as Markdown.

These are heuristic checks, not universal clinical rules. Data definitions, measurement units, local workflows, and intended prediction time must be considered before acting on a finding.

## Try the live application

Open **[MedData Doctor](https://meddata-hpro.vercel.app/)**.

1. Upload a CSV file with one record per row, or select **Load synthetic demo data**.
2. Select the column containing the outcome you want to predict (the target/label).
3. Click **Run audit**.
4. Review each finding, its rationale, and the suggested next step.
5. Optionally run **Run before/after experiment** to compare the baseline with the cleaned pipeline.
6. Download the Markdown report for review or documentation.

For an initial test, the built-in demo creates synthetic rows with deliberate issues. It contains no real patient records.

## Run locally

### Requirements

- Python 3.10 or later
- pip

### Setup

```bash
git clone https://github.com/agueroyas/MedDataDoctor.git
cd MedDataDoctor
python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

### Browser-based deployment

The repository also includes `index.html`, which loads the Python application in the browser using [Stlite](https://github.com/whitphx/stlite). It fetches `app.py`, `audit.py`, `rules.py`, and `demo.py` as separate text files and runs them in a WebAssembly-backed environment.

To preview this entry point locally, serve the repository over HTTP rather than opening `index.html` directly:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000`. The deployed web app is available at the live-demo link above.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, CSV input, results, experiment controls, and report download. |
| `audit.py` | Audit checks, report data structures, leakage scan, cleaning experiment, and report generation. |
| `rules.py` | Aliases and configurable plausibility rules for supported measurements. |
| `demo.py` | Synthetic demonstration dataset generator with deliberate example issues. |
| `index.html` | Browser entry point using Stlite. |
| `vercel.json` | Response headers and static deployment configuration. |
| `requirements.txt` | Python package dependencies. |

## How to read a finding

A finding is a prompt for investigation, not an automatic instruction to modify a medical dataset. For example, a zero can mean missingness in one dataset and a real value in another; a highly predictive feature can be legitimate if it is available at the intended prediction time. Confirm the data dictionary, collection workflow, units, and clinical context before removing or transforming fields.

The data-health score is a simple heuristic based on findings and severity. It is **not** model accuracy, probability of error, clinical quality certification, or a substitute for expert review.

## Privacy and safety

- Use public, synthetic, or properly de-identified datasets only.
- Do not upload patient-identifiable information, direct identifiers, or sensitive records to the hosted demo.
- Review your organization's privacy, security, consent, and data-governance requirements before processing any dataset.
- Do not use this application to make clinical decisions or determine a patient's diagnosis or treatment.

## Limitations

- The supported measurement rules are name-based heuristics and may not match every dataset or unit system.
- Leakage detection uses single-feature predictive strength and column-name clues; it cannot prove the presence or absence of leakage.
- The before/after results depend on the dataset, label, split, preprocessing assumptions, and model configuration. They should not be treated as a benchmark or performance guarantee.
- Group representation is not the same as fairness; evaluate model performance separately for relevant groups before making fairness claims.

## Technology

Python · Streamlit · pandas · NumPy · scikit-learn · Stlite · Vercel

## Author and ownership

**Developed by Abdulrahman Muhammed Abdalnafea**  
GitHub: [@agueroyas](https://github.com/agueroyas)

Copyright (c) 2026 Abdulrahman Muhammed Abdalnafea. All rights reserved. See [LICENSE](LICENSE) for permitted use.

---

*Built as a practical project to make dataset risks easier to inspect before machine-learning experiments.*
