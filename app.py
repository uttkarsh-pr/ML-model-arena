"""Machine Learning Model Comparison System (Streamlit front end).

Run with:   streamlit run app.py
"""
import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import DataError, run_comparison

APP_TITLE = "Machine Learning Model Comparison System"   # placeholder - rename later
HERE = os.path.dirname(os.path.abspath(__file__))

SAMPLES = {
    "Breast cancer diagnosis (classification)": ("breast_cancer_classification.csv", "diagnosis"),
    "Diabetes progression (regression)": ("diabetes_regression.csv", "target"),
    "Customer churn (mixed data types, missing values)": ("customer_churn_mixed.csv", "churned"),
}

# Fixed colour per model (validated categorical palette, light surface).
MODEL_COLORS = {
    "XGBoost": "#2a78d6", "LightGBM": "#eb6834", "CatBoost": "#1baf7a", "Random Forest": "#eda100",
    "SVM": "#e87ba4", "KNN": "#008300", "Logistic Regression": "#6250d6", "Linear Regression": "#6250d6",
    "Baseline": "#898781",
}
INK, INK2, GRID, AXIS = "#1a1a1a", "#52514e", "#e1e0d9", "#c3c2b7"
SEQ = [[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#0d366b"]]
DIV = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#256abf"]]
ACCENT = "#1f3a5f"

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1250px; }
      h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif !important; font-weight: 600 !important; letter-spacing: 0; }
      h1 { font-size: 1.9rem !important; border-bottom: 2px solid #1f3a5f; padding-bottom: .4rem; }
      h2 { font-size: 1.35rem !important; margin-top: 1.6rem !important; }
      .caption { font-size: .86rem; color: #52514e; margin: -.2rem 0 1.1rem 0; font-style: italic; }
      .abstract { background: #f3f4f6; border-left: 4px solid #1f3a5f; padding: .9rem 1.2rem; margin: .8rem 0 1.2rem 0; line-height: 1.65; }
      .small { font-size: .85rem; color: #52514e; }
      div[data-testid="stMetricValue"] { font-size: 1.5rem; }
      table { font-size: .92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

fmt = lambda v, d=4: "-" if v is None else f"{v:.{d}f}"


def caption(kind: str, n: int, text: str):
    st.markdown(f"<div class='caption'><b>{kind} {n}.</b> {text}</div>", unsafe_allow_html=True)


def base_layout(fig, height=380, **kw):
    cfg = dict(
        template="plotly_white", height=height, margin=dict(l=60, r=20, t=20, b=55),
        font=dict(family="Arial, sans-serif", color=INK2, size=12),
        paper_bgcolor="white", plot_bgcolor="white", legend=dict(orientation="h", y=-0.22),
        hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    cfg.update(kw)
    fig.update_layout(**cfg)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, automargin=True)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "displayModeBar": False})


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_table(name: str, raw: bytes) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


@st.cache_data(show_spinner=False)
def cached_run(df: pd.DataFrame, target: str, task: str, max_rows: int):
    return run_comparison(df, target, task, max_rows)


# --------------------------------------------------------------------------- #
# Sidebar - configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Configuration")
    st.markdown("**1. Data source**")
    sample_name = st.selectbox("Built-in sample dataset", ["None"] + list(SAMPLES), index=0)
    up = st.file_uploader("Or upload a data file (CSV or Excel)", type=["csv", "tsv", "txt", "xlsx", "xls"])

    df, source = None, None
    try:
        if up is not None:
            df, source = load_table(up.name, up.getvalue()), up.name
        elif sample_name != "None":
            fname, _ = SAMPLES[sample_name]
            df, source = pd.read_csv(os.path.join(HERE, "sample_data", fname)), sample_name
    except Exception:
        st.error("The file could not be read. Please provide a valid CSV or Excel file with a header row.")

    run_clicked = False
    if df is not None:
        st.markdown("**2. Prediction target**")
        default_t = SAMPLES[sample_name][1] if (up is None and sample_name != "None") else df.columns[-1]
        cols = [str(c) for c in df.columns]
        target = st.selectbox("Target variable", cols, index=cols.index(default_t) if default_t in cols else len(cols) - 1)
        st.markdown("**3. Problem type**")
        task_label = st.radio("Problem type", ["Automatic detection", "Classification", "Regression"], label_visibility="collapsed")
        task = {"Automatic detection": "auto", "Classification": "classification", "Regression": "regression"}[task_label]
        st.markdown("**4. Maximum number of rows used**")
        max_rows = st.select_slider("Rows", options=[1000, 2000, 5000, 10000, 20000], value=5000, label_visibility="collapsed")
        run_clicked = st.button("Run comparison", type="primary", width="stretch")
    st.markdown("---")
    st.markdown(
        "<div class='small'>Evaluation: k-fold cross-validation (k = 5, or fewer when a class is very small). "
        "All models are evaluated on identical folds with identical pre-processing.</div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title(APP_TITLE)
st.markdown(
    "Upload a dataset to compare machine-learning models. The app shows which model performs best, how accurate it is, and useful charts that help explain the results."
)

if run_clicked:
    try:
        with st.spinner("Training and cross-validating the models. This may take up to a minute."):
            st.session_state["result"] = cached_run(df, target, task, max_rows)
            st.session_state["source"] = source
    except DataError as exc:
        st.session_state.pop("result", None)
        st.error(str(exc))
    except Exception as exc:
        st.session_state.pop("result", None)
        st.error(f"The analysis could not be completed ({type(exc).__name__}). Please check the selected target variable and data.")

R = st.session_state.get("result")

# --------------------------------------------------------------------------- #
# Landing page (no result yet)
# --------------------------------------------------------------------------- #
if R is None:
    if df is not None:
        st.subheader("Data preview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Observations", f"{len(df):,}")
        c2.metric("Variables", f"{df.shape[1]}")
        c3.metric("Missing cells", f"{df.isna().mean().mean() * 100:.1f}%")
        st.dataframe(df.head(15), width="stretch")
        st.info("Select the target variable in the sidebar and choose **Run comparison**.")
    else:
        st.subheader("Procedure")
        st.markdown(
            "1. Select a built-in sample dataset or upload a CSV / Excel file using the sidebar.\n"
            "2. Specify the target variable (the quantity to be predicted).\n"
            "3. Choose the problem type, or allow it to be detected automatically.\n"
            "4. Run the comparison and review the results, figures and tables."
        )
        st.subheader("Algorithms evaluated")
        st.table(pd.DataFrame({
            "Algorithm": ["Logistic Regression", "Linear Regression", "Support Vector Machine (SVM)", "k-Nearest Neighbours (KNN)",
                          "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
            "Applicable to": ["Classification", "Regression", "Both", "Both", "Both", "Both", "Both", "Both"],
            "Family": ["Linear model", "Linear model", "Kernel method", "Instance-based", "Bagged decision trees",
                       "Gradient-boosted trees", "Gradient-boosted trees", "Gradient-boosted trees"],
        }))
        st.markdown(
            "<div class='small'>A naive baseline (majority class or mean prediction) is included as a reference for judging "
            "whether a model learns anything beyond chance.</div>", unsafe_allow_html=True)
    st.stop()

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
S = R["summary"]
clf = S["task"] == "classification"
P = S["primary_metric"]
P_NAME = "weighted F1-score" if clf else "coefficient of determination (R²)"
rows = {r["model"]: r for r in R["results"]}
order = R["ranking"]
cont = [rows[m] for m in order]
best = cont[0]
base = rows.get("Baseline")
metrics = [m for m in S["metrics"] if any(r.get(m) is not None for r in cont)]
LOWER = {"RMSE", "MAE"}
col = lambda m: MODEL_COLORS.get(m, "#898781")

st.caption(f"Data source: {st.session_state.get('source', '-')}")

st.header("1. Quick summary")
score_name = "F1 score" if clf else "R² score"
summary_lines = [
    f"**Best model:** {best['model']}",
    f"**Its {score_name}:** {fmt(best[P], 3)} (higher is better).",
]
tied = R["close"]
if tied:
    summary_lines.append(f"**Close alternatives:** {', '.join(tied)}. Their results are very similar to the top model.")
else:
    summary_lines.append("**Other models:** None had a result close to the top model in this test.")
if base and base.get(P) is not None:
    summary_lines.append(f"**Simple baseline:** {fmt(base[P], 3)}. This is the score from a basic model that does not learn useful patterns.")
st.markdown("\n\n".join(summary_lines))

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Problem type", S["task"].capitalize())
k2.metric("Observations used", f"{S['rows_used']:,}", f"of {S['rows_uploaded']:,}" if S["sampled"] else None, delta_color="off")
k3.metric("Predictor variables", f"{S['n_features']}", f"{S['n_numeric']} numeric, {S['n_categorical']} categorical", delta_color="off")
k4.metric("Classes" if clf else "Missing cells", f"{len(S['classes'])}" if clf else f"{S['missing_cells_pct']:.1f}%")
k5.metric("Run time", f"{S['elapsed']:.1f} s")

# ---------------- Table 1: leaderboard ----------------
st.header("2. Performance of the evaluated models")
table = []
for r in cont:
    row = {"Rank": str(r["rank"]), "Model": r["model"]}
    for m in metrics:
        row[m + (" (lower is better)" if m in LOWER else "")] = f"{fmt(r[m])} ± {fmt(r[m + '_std'])}"
    row["Mean fit time (s)"] = round(r["fit_time"], 3)
    table.append(row)
if base:
    row = {"Rank": "-", "Model": "Baseline (" + ("majority class" if clf else "mean") + ")"}
    for m in metrics:
        row[m + (" (lower is better)" if m in LOWER else "")] = fmt(base.get(m))
    row["Mean fit time (s)"] = round(base["fit_time"], 3)
    table.append(row)
tdf = pd.DataFrame(table)
st.dataframe(
    tdf.style.apply(lambda s: ["background-color:#e8eef7;font-weight:600" if s["Rank"] == "1" else "" for _ in s], axis=1)
    .format({"Mean fit time (s)": "{:.3f}"}),
    width="stretch", hide_index=True,
)
caption("Table", 1, f"Model scores averaged across {S['n_splits']} tests. The top-ranked model is highlighted. Lower RMSE and MAE are better; higher scores are better for the other measures.")

csv_df = pd.DataFrame([{**{"model": r["model"], "rank": r.get("rank")}, **{m: r.get(m) for m in S["metrics"]},
                       **{m + "_std": r.get(m + "_std") for m in S["metrics"]}, "fit_time_s": r["fit_time"]} for r in R["results"]])
st.download_button("Download results table (CSV)", csv_df.to_csv(index=False).encode(), "model_comparison_results.csv", "text/csv")

tab_cmp, tab_best, tab_data, tab_method = st.tabs(
    ["3. Model comparison", "4. Best-model diagnostics", "5. Dataset description", "6. Methodology and limitations"])

# ---------------- Comparison figures ----------------
with tab_cmp:
    names = [r["model"] for r in cont]
    vals = [r[P] for r in cont]

    # Figure 1 - ranking
    fig = go.Figure(go.Bar(
        x=vals[::-1], y=names[::-1], orientation="h", marker_color=[col(n) for n in names[::-1]],
        error_x=dict(type="data", array=[r[P + "_std"] for r in cont][::-1], color=INK2, thickness=1.4, width=4),
        text=[f"{v:.3f}" for v in vals[::-1]], textposition="inside", insidetextanchor="start", textfont=dict(color="white"),
        hovertemplate="%{y}<br>" + P + ": %{x:.4f}<extra></extra>"))
    if base and base.get(P) is not None:
        fig.add_vline(x=base[P], line_dash="dash", line_color="#898781", annotation_text="Baseline", annotation_position="top")
    base_layout(fig, 360, bargap=0.35, margin=dict(l=130, r=20, t=30, b=50))
    fig.update_xaxes(title_text=P_NAME[0].upper() + P_NAME[1:], range=[min(0, (base[P] - .02) if base and base.get(P) is not None else 0), max(1, max(vals) * 1.05)])
    show(fig)
    caption("Figure", 1, f"This chart ranks the models. The dashed line shows the score of the simple baseline.")

    c1, c2 = st.columns(2)
    with c1:
        fx = [np.log10(max(r["fit_time"], 1e-3)) for r in cont]
        ypad = ((max(vals) - min(vals)) or 0.1) * 0.2
        fig = go.Figure()
        for r in cont:
            fig.add_trace(go.Scatter(x=[max(r["fit_time"], 1e-3)], y=[r[P]], mode="markers+text", name=r["model"], text=[r["model"]],
                                     textposition="top center", cliponaxis=False, textfont=dict(color=INK2, size=11),
                                     marker=dict(size=14, color=col(r["model"]), line=dict(color="white", width=2)),
                                     hovertemplate=f"{r['model']}<br>{P}: {r[P]:.4f}<br>Fit time: {r['fit_time']:.3f} s<extra></extra>"))
        base_layout(fig, 380, showlegend=False)
        fig.update_xaxes(type="log", range=[min(fx) - .35, max(fx) + .35], title_text="Training time per fold (s, logarithmic scale)")
        fig.update_yaxes(range=[min(vals) - ypad, max(vals) + ypad], title_text=P)
        show(fig)
        caption("Figure", 2, "This chart compares model quality with training time. Models closer to the top-left give a better score in less time.")
    with c2:
        fig = go.Figure()
        for r in cont:
            fig.add_trace(go.Box(y=r[P + "_folds"], name=r["model"], boxpoints="all", jitter=.4, pointpos=0,
                                 line=dict(color=col(r["model"]), width=1.5), fillcolor=col(r["model"]) + "33",
                                 marker=dict(color=col(r["model"]), size=5)))
        base_layout(fig, 380, showlegend=False, margin=dict(l=60, r=20, t=20, b=80))
        fig.update_yaxes(title_text=P)
        show(fig)
        caption("Figure", 3, "This chart shows how much each model’s score changed between tests. A smaller box means more consistent results.")

    if clf and R["roc"]:
        fig = go.Figure()
        for m, d in R["roc"].items():
            fig.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines", name=f"{m} (AUC = {fmt(d['auc'], 3)})",
                                     line=dict(color=col(m), width=3.5 if m == best["model"] else 2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#898781", dash="dot", width=1), showlegend=False, hoverinfo="skip"))
        base_layout(fig, 430, margin=dict(l=60, r=20, t=20, b=110))
        fig.update_xaxes(title_text="False positive rate", range=[0, 1])
        fig.update_yaxes(title_text="True positive rate", range=[0, 1.02])
        show(fig)
        caption("Figure", 6, "This chart shows how well each model separates the two classes. Curves closer to the top-left are generally better.")

# ---------------- Best model diagnostics ----------------
with tab_best:
    st.subheader(f"Diagnostics for the selected model: {best['model']}")
    D = R["diag"]
    c1, c2 = st.columns(2)
    n = 7
    if clf:
        cm = D["confusion"]
        lbl = cm["labels"]
        mat = np.array(cm["matrix"], dtype=int)
        with c1:
            st.markdown("**Prediction results**")
            st.caption("Rows show the actual class. Columns show what the model predicted. Larger numbers on the diagonal mean more correct predictions.")
            cm_df = pd.DataFrame(mat, index=[f"Actual: {x}" for x in lbl], columns=[f"Predicted: {x}" for x in lbl])
            st.dataframe(cm_df, width="stretch")
    else:
        sc = D["scatter"]; a, p = np.array(sc["actual"]), np.array(sc["pred"]); lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=a, y=p, mode="markers", marker=dict(color=col(best["model"]), size=6, opacity=.65, line=dict(color="white", width=.5)),
                                     hovertemplate="Actual %{x:.2f}<br>Predicted %{y:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#898781", dash="dot"), hoverinfo="skip"))
            base_layout(fig, 420, showlegend=False)
            fig.update_xaxes(title_text="Observed value"); fig.update_yaxes(title_text="Predicted value")
            show(fig)
            caption("Figure", n, "Each dot compares a real value with the model’s prediction. Dots closer to the dotted line are better.")
        with c2:
            fig = go.Figure(go.Histogram(x=D["residuals"], nbinsx=30, marker=dict(color=col(best["model"]), line=dict(color="white", width=1))))
            base_layout(fig, 420, showlegend=False, bargap=.06)
            fig.update_xaxes(title_text="Residual (observed - predicted)"); fig.update_yaxes(title_text="Frequency")
            show(fig)
            caption("Figure", n + 1, "This chart shows prediction errors. Errors centred around zero suggest the model is not regularly over- or under-predicting.")
    if R["importance"]:
        imp = R["importance"][::-1]
        target_col = c2 if clf else st.container()
        with target_col:
            fig = go.Figure(go.Bar(x=[i["importance"] for i in imp], y=[i["feature"] for i in imp], orientation="h", marker_color=col(best["model"]),
                                   error_x=dict(type="data", array=[i["std"] for i in imp], color=INK2, thickness=1.2, width=3),
                                   hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
            base_layout(fig, 420, bargap=.35, margin=dict(l=140, r=20, t=20, b=55))
            fig.update_xaxes(title_text="Decrease in score when the variable is permuted")
            show(fig)
            caption("Figure", n + 1 if clf else n + 2, "Variables with longer bars have a bigger effect on the model’s predictions.")

    if clf:
        rep = pd.DataFrame(D["class_report"]).rename(columns={"class": "Class", "precision": "Precision", "recall": "Recall", "f1": "F1-score", "support": "Support"})
        st.dataframe(rep.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}), width="stretch", hide_index=True)
        caption("Table", 2, "Per-class precision, recall, F1-score and number of observations (out-of-fold predictions).")
    else:
        rs = D["resid_summary"]
        st.dataframe(pd.DataFrame({"Statistic": ["R² (mean of folds)", "RMSE", "MAE", "Mean residual", "Standard deviation of residuals"],
                                   "Value": [fmt(best["R²"]), fmt(best["RMSE"]), fmt(best["MAE"]), fmt(rs["mean"]), fmt(rs["std"])]}),
                     width="stretch", hide_index=True)
        caption("Table", 2, "Summary of the prediction errors of the selected model.")

# ---------------- Dataset description ----------------
with tab_data:
    c1 = st.container()
    td = R["target_dist"]
    with c1:
        if td["kind"] == "classes":
            fig = go.Figure(go.Bar(x=td["labels"], y=td["counts"], marker_color=ACCENT, hovertemplate="%{x}: %{y}<extra></extra>"))
            base_layout(fig, 360, bargap=.35, showlegend=False)
            fig.update_yaxes(title_text="Number of observations"); fig.update_xaxes(title_text="Class")
            note = " The classes are strongly imbalanced; the weighted F1-score is therefore a more reliable criterion than accuracy." if (S["class_balance"] or 0) > 0.8 else ""
        else:
            mid = [(td["edges"][i] + td["edges"][i + 1]) / 2 for i in range(len(td["counts"]))]
            fig = go.Figure(go.Bar(x=mid, y=td["counts"], marker_color=ACCENT, hovertemplate="%{x:.2f}: %{y}<extra></extra>"))
            base_layout(fig, 360, bargap=.06, showlegend=False)
            fig.update_yaxes(title_text="Frequency"); fig.update_xaxes(title_text=S["target"])
            note = ""
        show(fig)
        caption("Figure", 20, f"Distribution of the target variable '{S['target']}'.{note}")

    if R["missing"]["labels"]:
        m = R["missing"]
        fig = go.Figure(go.Bar(x=m["values"][::-1], y=m["labels"][::-1], orientation="h", marker_color=ACCENT, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        base_layout(fig, 320, bargap=.4, showlegend=False, margin=dict(l=140, r=20, t=20, b=50))
        fig.update_xaxes(title_text="Missing values (%)")
        show(fig)
        caption("Figure", 22, "This chart shows how much information is missing from each variable. The app fills missing values before training.")

    cdf = pd.DataFrame(R["columns"]).rename(columns={"column": "Variable", "role": "Role", "type": "Type", "missing": "Missing (%)", "unique": "Distinct values"})
    st.dataframe(cdf.style.format({"Missing (%)": "{:.1f}"}), width="stretch", hide_index=True)
    caption("Table", 3, "Variables of the dataset, their role in the analysis, and basic descriptive information. "
                        "Variables marked as dropped were excluded because they cannot support learning (identifiers, constants, dates or mostly empty columns).")

# ---------------- Methodology ----------------
with tab_method:
    st.markdown(
        f"""
### How the comparison works
- The app prepares the data so the models can use it. Missing numbers are filled with the median; missing categories are filled with the most common value.
- Each model is tested on the same {S['n_splits']} data splits. This makes the comparison fairer.
- The main score used for ranking is **{score_name}**. A higher score is better.
- A simple baseline is included so you can see whether the trained models do better than a basic guess.
- The feature-importance chart shows which input variables matter most to the winning model.

### Things to keep in mind
- Models use standard settings; the app does not search for the best possible settings for every model.
- The app uses up to {S['rows_used']:,} rows{' (a sample of your file)' if S['sampled'] else ''}.
- Small score differences do not always mean one model is truly better. Treat very close results as a tie.
"""
    )
    if R["skipped"]:
        st.markdown("**Algorithms not evaluated in this run**")
        st.dataframe(pd.DataFrame(R["skipped"]).rename(columns={"model": "Algorithm", "reason": "Reason"}), width="stretch", hide_index=True)
