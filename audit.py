# Copyright (c) 2026 Abdulrahman Muhammed Abdalnafea
# MedData Doctor — https://github.com/agueroyas/MedDataDoctor
# Author profile: https://github.com/agueroyas
# See LICENSE for permitted use. Third-party components retain their own licenses.

"""MedData Doctor - audit engine (no UI, so it is easy to test)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from rules import RULES, SUSPICIOUS_NAME_PARTS

SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Issue:
    severity: str  # high / medium / low
    category: str
    column: str
    title: str
    detail: str
    fix: str


@dataclass
class Report:
    n_rows: int
    n_cols: int
    label_col: str
    issues: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)
    score: int = 100
    drop_cols: list = field(default_factory=list)  # recommended to drop before training


# ---------------------------------------------------------------- helpers
def norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def match_rule(col):
    n = norm(col)
    for rule in RULES:
        for alias in rule["aliases"]:
            if n == alias:
                return rule
            if len(alias) >= 5 and (n.startswith(alias) or n.endswith(alias)):
                return rule
    return None


def to_binary(series: pd.Series):
    """Return a 0/1/NaN series, or None if the label is not binary."""
    uniq = series.dropna().unique()
    if len(uniq) != 2:
        return None
    try:
        ordered = sorted(uniq)
    except TypeError:
        ordered = sorted(uniq, key=str)
    positive = ordered[1]
    return (series == positive).astype(float).where(series.notna())


# ---------------------------------------------------------------- checks
def check_plausibility(df, label_col):
    issues, rows = [], []
    for col in df.columns:
        if col == label_col or not is_num(df[col]):
            continue
        rule = match_rule(col)
        if rule is None:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        zero_mask = (s == 0) if rule["zero_missing"] else pd.Series(False, index=s.index)
        range_mask = ((s < rule["min"]) | (s > rule["max"])) & ~zero_mask
        zeros, bad = int(zero_mask.sum()), int(range_mask.sum())
        rows.append({
            "column": col, "matched rule": rule["label"],
            "plausible range": f"{rule['min']}-{rule['max']}",
            "zeros (hidden missing)": zeros, "out of range": bad,
            "% affected": round((zeros + bad) / len(s) * 100, 1),
        })
        if zeros + bad == 0:
            continue
        pct = (zeros + bad) / len(s) * 100
        sev = "high" if pct >= 20 else "medium" if pct >= 2 else "low"
        parts = []
        if zeros:
            parts.append(f"{zeros} value(s) ({zeros / len(s):.0%}) are exactly 0, which is impossible for "
                         f"{rule['label']} and almost certainly means 'not measured'")
        if bad:
            parts.append(f"{bad} value(s) fall outside the plausible range {rule['min']}-{rule['max']} {rule['unit']}")
        issues.append(Issue(
            sev, "Impossible values", col,
            f"{rule['label']}: {pct:.1f}% implausible values",
            "; ".join(parts) + ".",
            "Convert them to missing values (NaN), then impute or drop them. Otherwise the model "
            "learns from numbers that are not real measurements.",
        ))
    return issues, pd.DataFrame(rows)


def clean_impossible(df):
    """Set implausible values to NaN. Returns (cleaned copy, number of values changed)."""
    out, changed = df.copy(), 0
    for col in out.columns:
        if not is_num(out[col]):
            continue
        rule = match_rule(col)
        if rule is None:
            continue
        s = out[col]
        mask = (s < rule["min"]) | (s > rule["max"])
        if rule["zero_missing"]:
            mask |= s == 0
        changed += int(mask.sum())
        if mask.any():
            out[col] = out[col].astype(float)
            out.loc[mask, col] = np.nan
    return out, changed


def check_missing_and_basics(df, label_col):
    issues, n = [], len(df)
    for col, pct in df.isna().mean().mul(100).items():
        if col != label_col and pct >= 30:
            issues.append(Issue(
                "high" if pct >= 60 else "medium", "Missing data", col,
                f"{pct:.0f}% of '{col}' is missing",
                "A column that is mostly empty gives the model little real information, and the "
                "pattern of what is missing can itself leak information about the outcome.",
                "Check why it is missing. Drop the column or impute carefully, and report how much was imputed.",
            ))
    labels = df[label_col]
    if labels.isna().any():
        issues.append(Issue(
            "medium", "Labels", label_col, f"{int(labels.isna().sum())} rows have no label",
            "Rows without an outcome cannot be used for supervised training.",
            "Remove them or find the missing outcomes.",
        ))
    counts = labels.value_counts(dropna=True)
    if len(counts) >= 2:
        share = counts.min() / counts.sum()
        if share < 0.20:
            issues.append(Issue(
                "high" if share < 0.05 else "medium", "Class balance", label_col,
                f"Rare outcome: smallest class is only {share:.1%} of rows",
                "With a rare outcome, a model can look accurate by always predicting the common class.",
                "Use stratified splits, report AUC / precision / recall (not accuracy alone), "
                "and consider class weights.",
            ))
    dups = int(df.duplicated().sum())
    if dups:
        pct = dups / n * 100
        if pct >= 1:
            issues.append(Issue(
                "high" if pct >= 10 else "medium", "Duplicates", "(rows)",
                f"{dups} duplicate rows ({pct:.1f}%)",
                "Duplicates can land in both the training and test sets, so the model is graded on "
                "rows it has already seen and its score looks better than it really is.",
                "Remove exact duplicates BEFORE splitting into train and test.",
            ))
    drop = []
    for col in df.columns:
        if col == label_col or pd.api.types.is_float_dtype(df[col]):
            continue
        n_ = norm(col)
        if (n_.startswith("id") or n_.endswith("id") or n_ == "mrn") and df[col].nunique() / max(n, 1) > 0.95:
            drop.append(col)
            issues.append(Issue(
                "low", "Identifier", col, f"'{col}' looks like a record identifier",
                "Identifiers carry no clinical signal and can let a model memorize individual patients.",
                "Drop it before training.",
            ))
    return issues, drop


def check_representation(df, label_col):
    issues, tables = [], {}
    sex_col = next((c for c in df.columns if norm(c) in ("sex", "gender") and c != label_col), None)
    age_col = next((c for c in df.columns if norm(c) == "age" and is_num(df[c]) and c != label_col), None)
    has_eth = any(norm(c) in ("race", "ethnicity", "ethnicgroup") for c in df.columns)
    if sex_col:
        shares = df[sex_col].value_counts(normalize=True, dropna=True)
        tables["Sex / gender"] = pd.DataFrame({"group": shares.index.astype(str), "share": shares.values.round(3)})
        if len(shares) >= 2 and shares.min() < 0.20:
            issues.append(Issue(
                "high" if shares.min() < 0.10 else "medium", "Representation", sex_col,
                f"Imbalanced groups in '{sex_col}': smallest group is {shares.min():.0%}",
                "A model trained mostly on one group often performs worse on the others.",
                "Collect more data for the small group, or at least report performance per group.",
            ))
    if age_col:
        bands = pd.cut(df[age_col], bins=[-1, 17, 39, 64, 200], labels=["<18", "18-39", "40-64", "65+"])
        shares = bands.value_counts(normalize=True, dropna=True).reindex(["<18", "18-39", "40-64", "65+"]).fillna(0)
        tables["Age bands"] = pd.DataFrame({"band": shares.index.astype(str), "share": shares.values.round(3)})
        thin = [b for b, s in shares.items() if 0 < s < 0.05]
        if thin:
            issues.append(Issue(
                "low", "Representation", age_col,
                f"Very few patients in age band(s): {', '.join(thin)}",
                "Predictions for these ages rest on very little data.",
                "Check that this matches the population the model is meant for, and report accuracy per age band.",
            ))
    if not sex_col and not has_eth:
        issues.append(Issue(
            "low", "Representation", "(dataset)",
            "No sex/gender or ethnicity column found",
            "Without these columns you cannot check whether the model works equally well for different groups.",
            "Say so explicitly when you describe the dataset's limits, and avoid claims about all patients.",
        ))
    return issues, tables


def check_leakage(df, label_col, seed=0, max_rows=20000):
    """Score every column on its own: how well does it predict the label by itself?"""
    y = to_binary(df[label_col])
    if y is None:
        return [], pd.DataFrame()
    data = df.loc[y.notna()]
    yv = y.loc[data.index].astype(int)
    if len(data) > max_rows:
        idx = data.sample(max_rows, random_state=seed).index
        data, yv = data.loc[idx], yv.loc[idx]
    folds = min(5, int(yv.value_counts().min()))
    if folds < 2:
        return [], pd.DataFrame()
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows, issues = [], []
    for col in data.columns:
        if col == label_col:
            continue
        x = data[col]
        if is_num(x) or pd.api.types.is_bool_dtype(x):
            x = pd.to_numeric(x.astype(float), errors="coerce")
            if x.notna().sum() == 0:
                continue
            x = x.fillna(x.median())
        else:
            x = x.astype("category").cat.codes
        if x.nunique() < 2:
            continue
        try:
            auc = float(cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0),
                                        x.values.reshape(-1, 1), yv.values, cv=cv, scoring="roc_auc").mean())
        except ValueError:
            continue
        suspicious = any(p in norm(col) for p in SUSPICIOUS_NAME_PARTS)
        rows.append({"feature": col, "single-feature AUC": round(auc, 3), "name looks post-outcome": suspicious})
        if auc >= 0.97:
            issues.append(Issue(
                "high", "Label leakage", col, f"'{col}' predicts the outcome almost perfectly (AUC {auc:.2f})",
                "Real clinical signals are rarely this clean. The column is probably recorded after the outcome "
                "is known, or is a stand-in for it, so the model 'cheats' instead of learning medicine.",
                "Ask when this value is recorded. If it is only known after the diagnosis/outcome, remove it. "
                "(It can also be a legitimately strong predictor, so confirm with a clinician.)",
            ))
        elif auc >= 0.90 and suspicious:
            issues.append(Issue(
                "high", "Label leakage", col,
                f"'{col}' is a very strong predictor AND its name suggests it is recorded after the outcome (AUC {auc:.2f})",
                "A column about treatment, discharge or outcome that predicts the label this well is very likely "
                "recorded after the diagnosis, so the model 'cheats' instead of learning medicine.",
                "Ask when this value is recorded. If it is only known after the diagnosis/outcome, remove it. "
                "(Confirm with a clinician before dropping.)",
            ))
        elif auc >= 0.90:
            issues.append(Issue(
                "medium", "Label leakage", col, f"'{col}' is a suspiciously strong predictor (AUC {auc:.2f})",
                "Very strong single columns deserve a second look: they can hide leakage.",
                "Check how and when this value is recorded.",
            ))
        elif suspicious:
            issues.append(Issue(
                "low", "Label leakage", col, f"'{col}' has a name that often means 'recorded after the outcome'",
                "Columns about treatment, discharge or outcome are sometimes only known after the label.",
                "Verify the timing: would this value be available at prediction time?",
            ))
    table = pd.DataFrame(rows).sort_values("single-feature AUC", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    return issues, table


# ---------------------------------------------------------------- orchestration
def audit(df: pd.DataFrame, label_col: str) -> Report:
    rep = Report(n_rows=len(df), n_cols=df.shape[1], label_col=label_col)
    p_issues, p_table = check_plausibility(df, label_col)
    b_issues, id_cols = check_missing_and_basics(df, label_col)
    r_issues, r_tables = check_representation(df, label_col)
    l_issues, l_table = check_leakage(df, label_col)
    rep.issues = sorted(p_issues + b_issues + r_issues + l_issues, key=lambda i: SEV_ORDER[i.severity])
    rep.tables = {"Plausibility": p_table, "Leakage scan": l_table, **r_tables}
    leak_high = [i.column for i in l_issues if i.severity == "high"]
    rep.drop_cols = list(dict.fromkeys(leak_high + id_cols))
    penalty = sum({"high": 20, "medium": 8, "low": 3}[i.severity] for i in rep.issues)
    rep.score = max(0, 100 - penalty)
    return rep


def score_label(score: int) -> str:
    return "Healthy" if score >= 80 else "Needs attention" if score >= 50 else "Risky"


# ---------------------------------------------------------------- before / after experiment
def build_matrix(df, label_col, drop_cols):
    X = df.drop(columns=[label_col] + [c for c in drop_cols if c in df.columns])
    keep = [c for c in X.columns if is_num(X[c]) or pd.api.types.is_bool_dtype(X[c]) or X[c].nunique(dropna=True) <= 50]
    X = pd.get_dummies(X[keep], dummy_na=False)
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns left.")
    return X.astype(float)


def run_experiment(df, label_col, report, repeats=5, max_rows=20000, seed=42):
    """Train the same model on the data as-is and after cleaning; compare honest scores."""
    y_all = to_binary(df[label_col])
    if y_all is None:
        raise ValueError("The experiment needs a binary label column (exactly 2 classes).")
    data = df.loc[y_all.notna()].copy()
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=seed)

    cleaned = data.drop_duplicates().copy()
    n_dups = len(data) - len(cleaned)
    cleaned, n_fixed = clean_impossible(cleaned)
    notes = [f"Removed {n_dups} duplicate rows.",
             f"Set {n_fixed} impossible values to missing (then median-imputed inside the model).",
             f"Dropped columns: {', '.join(report.drop_cols) if report.drop_cols else 'none'}."]

    pipelines = {"As-is (what most people do)": (data, []), "After cleaning": (cleaned, report.drop_cols)}
    rows = []
    for name, (frame, drops) in pipelines.items():
        X = build_matrix(frame, label_col, drops)
        y = to_binary(frame[label_col]).astype(int).values
        aucs, baccs = [], []
        for r in range(repeats):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed + r)
            model = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("rf", RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced",
                                              random_state=seed + r, n_jobs=1)),
            ])
            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(yte, proba))
            baccs.append(balanced_accuracy_score(yte, (proba >= 0.5).astype(int)))
        rows.append({
            "pipeline": name, "rows": len(frame), "features": X.shape[1],
            "ROC-AUC": round(float(np.mean(aucs)), 3), "AUC +/-": round(float(np.std(aucs)), 3),
            "balanced accuracy": round(float(np.mean(baccs)), 3), "bal. acc. +/-": round(float(np.std(baccs)), 3),
        })
    return pd.DataFrame(rows), notes


def interpret(results: pd.DataFrame) -> str:
    raw, clean = results.iloc[0], results.iloc[1]
    delta = raw["ROC-AUC"] - clean["ROC-AUC"]
    if delta >= 0.03:
        return (f"The as-is AUC is {delta:.2f} HIGHER than the cleaned one. That gap is the inflation "
                f"caused by the data problems above: the first number looks impressive but would not hold up on real patients.")
    if delta <= -0.03:
        return (f"Cleaning improved AUC by {-delta:.2f}: the impossible values were adding noise the model had to fight.")
    return ("The two scores are close, so in this dataset the problems found did not noticeably change performance "
            "on this model. They are still worth documenting.")


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def to_markdown(report: Report, results=None, notes=None, interp=None, dataset_name="dataset") -> str:
    lines = [f"# MedData Doctor report: {dataset_name}", "",
             f"- Rows: {report.n_rows}, columns: {report.n_cols}, label: `{report.label_col}`",
             f"- Data health score: **{report.score}/100 ({score_label(report.score)})** (heuristic)", "",
             "## Findings", ""]
    for i in report.issues:
        lines += [f"### [{i.severity.upper()}] {i.title}", f"- Column: `{i.column}` ({i.category})",
                  f"- Why it matters: {i.detail}", f"- Suggested fix: {i.fix}", ""]
    if results is not None:
        lines += ["## Before / after experiment", "", _md_table(results), ""]
        lines += [f"- {n}" for n in (notes or [])] + ["", interp or "", ""]
    lines += ["---", "Screening tool: findings are hypotheses to verify with the data owner and a clinician."]
    return "\n".join(lines)
