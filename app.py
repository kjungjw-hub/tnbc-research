"""Phase E: interactive demo for the TNBC basal-vs-non-basal fusion model.

Run locally:
    source .venv/bin/activate
    streamlit run app.py

Pick a real (anonymized) patient from the METABRIC TNBC cohort and see the
model's predicted probability of a basal-like subtype, compared against the
known ground-truth call, plus a local SHAP explanation of *why* the model
made that call for that specific patient. A "what if" panel lets you nudge
the handful of genes the model relies on most and watch the prediction
respond live.
"""
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="TNBC Subtype Explorer", page_icon="🧬", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load("models/fusion_model.joblib")
    features = joblib.load("models/fusion_features.joblib")
    medians = joblib.load("models/fusion_feature_medians.joblib")
    explainer = shap.TreeExplainer(model)
    return model, features, medians, explainer


@st.cache_data
def load_cohort(features):
    df = pd.read_csv("data/metabric_tnbc.csv", index_col=0)
    y = (df["CLAUDIN_SUBTYPE"] == "Basal").astype(int)
    X = df[features].apply(pd.to_numeric, errors="coerce")
    return df, X, y


model, features, medians, explainer = load_model()
raw_df, X, y = load_cohort(features)

st.title("🧬 TNBC Subtype Explorer")
st.caption(
    "Predicts basal-like vs. non-basal triple-negative breast cancer from clinical + "
    "gene expression + copy-number features, trained on the public METABRIC cohort. "
    "See the technical write-up in `reports/baseline_results.md` and the project overview "
    "in `flow.html` / `eli5.html`."
)

col_select, col_result = st.columns([1, 1.4], gap="large")

with col_select:
    st.subheader("1. Pick a patient")
    sample_id = st.selectbox("METABRIC sample (anonymized)", X.index.tolist())
    true_label = "Basal-like" if y.loc[sample_id] == 1 else "Non-basal"
    st.metric("Known subtype call", true_label)

    st.subheader("2. Or tweak the top genes")
    st.caption("Nudge the genes the model leans on most and watch the prediction update live.")

    # Global importance-ish ordering: use SHAP on the full cohort once, cached.
    @st.cache_data
    def top_genes(_features_tuple):
        sv = explainer(X)
        mean_abs = np.abs(sv.values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        return [features[i] for i in order if features[i].endswith("_expr")][:6]

    genes_to_tweak = top_genes(tuple(features))

    base_row = X.loc[sample_id].copy()
    edited_row = base_row.copy()
    for g in genes_to_tweak:
        gene_name = g.replace("_expr", "")
        lo, hi = float(X[g].min()), float(X[g].max())
        val = float(base_row[g]) if not pd.isna(base_row[g]) else float(medians[g])
        edited_row[g] = st.slider(
            f"{gene_name} expression (z-score)", min_value=round(lo, 1), max_value=round(hi, 1),
            value=round(val, 2), step=0.1, key=g,
        )

with col_result:
    st.subheader("3. Model prediction")
    row_df = pd.DataFrame([edited_row[features]])
    proba = model.predict_proba(row_df)[0, 1]

    st.metric("P(basal-like)", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))

    predicted = "Basal-like" if proba >= 0.5 else "Non-basal"
    match = "matches" if predicted == true_label else "differs from"
    st.write(f"Model predicts **{predicted}**, which {match} the known call (**{true_label}**).")

    st.subheader("Why the model said that (local SHAP explanation)")
    sv_row = explainer(row_df)
    contrib = pd.Series(sv_row.values[0], index=features).sort_values(key=np.abs, ascending=False)
    top_contrib = contrib.head(10)

    chart_df = pd.DataFrame({
        "feature": [f.replace("_expr", " (expr)").replace("_cna", " (CNA)") for f in top_contrib.index],
        "shap_value": top_contrib.values,
    }).set_index("feature")
    st.bar_chart(chart_df, horizontal=True)
    st.caption(
        "Positive bars push the prediction toward basal-like; negative bars push toward "
        "non-basal. Same genes surfaced in the cohort-wide SHAP analysis "
        "(`reports/shap_summary.png`) show up here for individual patients too."
    )

st.divider()
st.caption(
    f"Model: HistGradientBoostingClassifier fused over {len(features)} clinical + expression + "
    f"CNA features. Cohort: {len(X)} METABRIC TNBC patients (ER-/PR-/HER2- by IHC). "
    "5-fold CV AUC 0.920 vs. 0.610 for a clinical-only baseline — see `reports/baseline_results.md`."
)
