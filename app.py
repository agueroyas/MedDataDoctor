import pandas as pd
import streamlit as st

from audit import audit, interpret, norm, run_experiment, score_label, to_markdown
from demo import make_demo_df

st.set_page_config(page_title="MedData Doctor", page_icon="🩻", layout="wide")

ICON = {"high": "🔴", "medium": "🟡", "low": "🔵"}
LABEL_GUESSES = ["outcome", "target", "label", "class", "diagnosis", "deathevent", "heartdisease", "output", "y"]

st.title("🩻 MedData Doctor")
st.caption("A health check-up for medical datasets. Run it BEFORE you train a model.")
st.info(
    "Screening tool: findings are hypotheses to verify with the data owner and a clinician. "
    "Use public or fully de-identified data only."
)

# ------------------------------------------------------------------ data input
c1, c2 = st.columns([3, 1])
with c1:
    upload = st.file_uploader("Upload a CSV file (one row per patient)", type=["csv"])
with c2:
    st.write("")
    st.write("")
    if st.button("Load synthetic demo data"):
        st.session_state.df = make_demo_df()
        st.session_state.name = "synthetic demo (not real patients)"
        st.session_state.pop("report", None)
        st.session_state.pop("experiment", None)

if upload is not None and st.session_state.get("name") != upload.name:
    try:
        st.session_state.df = pd.read_csv(upload)
        st.session_state.name = upload.name
        st.session_state.pop("report", None)
        st.session_state.pop("experiment", None)
    except Exception as e:
        st.error(f"Could not read that file as CSV: {e}")

df = st.session_state.get("df")
if df is None:
    st.write("Upload a dataset (for example the Pima diabetes or Heart Failure CSV) or load the demo to begin.")
    st.stop()

st.subheader(f"Dataset: {st.session_state.name}")
st.dataframe(df.head(8), use_container_width=True)

guess = next((c for c in df.columns if norm(c) in LABEL_GUESSES), df.columns[-1])
label_col = st.selectbox("Which column is the outcome the model should predict?",
                         list(df.columns), index=list(df.columns).index(guess))

if st.button("Run audit", type="primary"):
    with st.spinner("Examining the data..."):
        st.session_state.report = audit(df, label_col)
        st.session_state.pop("experiment", None)

report = st.session_state.get("report")
if report is None or report.label_col != label_col:
    st.stop()

# ------------------------------------------------------------------ results
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Data health score", f"{report.score}/100", score_label(report.score))
m2.metric("Rows", f"{report.n_rows:,}")
m3.metric("Columns", report.n_cols)
m4.metric("Issues found", len(report.issues))
st.caption("The score is a simple heuristic (each issue subtracts points by severity), not a clinical measure.")

st.subheader("Findings")
if not report.issues:
    st.success("No issues found by these checks. That does not prove the data is perfect.")
for issue in report.issues:
    with st.expander(f"{ICON[issue.severity]} {issue.title}", expanded=issue.severity == "high"):
        st.write(f"**Category:** {issue.category}  |  **Column:** `{issue.column}`")
        st.write(f"**Why it matters:** {issue.detail}")
        st.write(f"**Suggested fix:** {issue.fix}")

with st.expander("Detail tables"):
    for name, table in report.tables.items():
        if table is not None and len(table):
            st.write(f"**{name}**")
            st.dataframe(table, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ experiment
st.divider()
st.subheader("What do these problems cost? Before / after experiment")
st.write(
    "The same model (random forest) is trained twice: on the data as-is, and after removing duplicates, "
    "turning impossible values into missing values and dropping leaking columns. "
    "Each result is the average of 5 random train/test splits."
)
if st.button("Run before/after experiment"):
    with st.spinner("Training models..."):
        try:
            st.session_state.experiment = run_experiment(df, label_col, report)
        except Exception as e:
            st.error(f"Experiment could not run: {e}")

exp = st.session_state.get("experiment")
if exp is not None:
    results, notes = exp
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.bar_chart(results.set_index("pipeline")[["ROC-AUC", "balanced accuracy"]])
    for n in notes:
        st.write(f"- {n}")
    verdict = interpret(results)
    st.success(verdict)
    st.caption("Results depend on the dataset and model. Treat them as an illustration, not a guarantee.")
    md = to_markdown(report, results, notes, verdict, st.session_state.name)
else:
    md = to_markdown(report, dataset_name=st.session_state.name)

st.download_button("Download report (Markdown)", md, file_name="meddata-doctor-report.md", mime="text/markdown")
