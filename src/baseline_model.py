"""Phase B baseline: predict basal-like vs non-basal TNBC from clinical
features alone, then from clinical + expression + CNA (the multi-omics
fusion), and compare. Adds SHAP explainability on the fusion model so
top features can be checked against known TNBC biology.

Label: CLAUDIN_SUBTYPE == "Basal" within the METABRIC TNBC cohort
(ER-/PR-/HER2- by IHC). Basal vs non-basal is a real, clinically used
split within TNBC (non-basal TNBC skews toward the LAR/MSL-like,
better-prognosis end of Lehmann's subtypes).
"""
import os

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLINICAL_COLS = ["GRADE", "TUMOR_SIZE", "TUMOR_STAGE", "MUTATION_COUNT"]


def load_data():
    df = pd.read_csv("data/metabric_tnbc.csv", index_col=0)
    y = (df["CLAUDIN_SUBTYPE"] == "Basal").astype(int)

    clinical = df[CLINICAL_COLS].apply(pd.to_numeric, errors="coerce")
    expr_cols = [c for c in df.columns if c.endswith("_expr")]
    cna_cols = [c for c in df.columns if c.endswith("_cna")]
    expr = df[expr_cols].apply(pd.to_numeric, errors="coerce")
    cna = df[cna_cols].apply(pd.to_numeric, errors="coerce")

    return y, clinical, expr, cna


def evaluate(X, y, model_name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if model_name == "clinical_baseline":
        # Impute/scale inside the pipeline so each CV fold's median/scale is
        # fit only on that fold's training data (no leakage into the fold's
        # held-out samples).
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=0.1),
        )
    else:
        pipe = HistGradientBoostingClassifier(random_state=42)  # native NaN support
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return scores


if __name__ == "__main__":
    y, clinical, expr, cna = load_data()
    print(f"Cohort: {len(y)} TNBC samples, {y.sum()} basal-like / {len(y) - y.sum()} non-basal.\n")

    print("=== Baseline: clinical features only ===")
    print(f"  Features: {CLINICAL_COLS}")
    baseline_scores = evaluate(clinical, y, "clinical_baseline")
    print(f"  5-fold CV AUC: {baseline_scores.mean():.3f} +/- {baseline_scores.std():.3f}")

    print("\n=== Fusion: clinical + expression + CNA ===")
    fusion_X = pd.concat([clinical, expr, cna], axis=1)
    print(f"  Features: {fusion_X.shape[1]} (clinical + {expr.shape[1]} genes expr + {cna.shape[1]} genes CNA)")
    fusion_scores = evaluate(fusion_X, y, "fusion")
    print(f"  5-fold CV AUC: {fusion_scores.mean():.3f} +/- {fusion_scores.std():.3f}")

    lift = fusion_scores.mean() - baseline_scores.mean()
    print(f"\nMulti-omics fusion lift over clinical-only baseline: {lift:+.3f} AUC")

    # ---------------- Explainability on the full-data fusion model ----------------
    print("\n=== SHAP: which features drive basal-like predictions? ===")
    final_model = HistGradientBoostingClassifier(random_state=42)
    final_model.fit(fusion_X, y)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(fusion_X)

    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0), index=fusion_X.columns
    ).sort_values(ascending=False)
    print("Top 15 features by mean |SHAP value|:")
    print(mean_abs_shap.head(15).to_string())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap.summary_plot(shap_values, fusion_X, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=150)
    print("\nSaved reports/shap_summary.png")

    # Persist the fitted model + the exact feature layout it expects, so the
    # demo app (app.py) can load a ready model instead of retraining.
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, "models/fusion_model.joblib")
    joblib.dump(list(fusion_X.columns), "models/fusion_features.joblib")
    joblib.dump(list(clinical.columns), "models/clinical_features.joblib")
    # Per-feature median, for the demo to pre-fill sliders with something typical.
    joblib.dump(fusion_X.median(), "models/fusion_feature_medians.joblib")
    print("Saved models/fusion_model.joblib (+ feature metadata)")

    with open("reports/baseline_results.md", "w") as f:
        f.write("# TNBC Basal vs Non-Basal: Baseline Results\n\n")
        f.write(f"Cohort: METABRIC TNBC (ER-/PR-/HER2-), n={len(y)} "
                f"({y.sum()} basal-like, {len(y) - y.sum()} non-basal)\n\n")
        f.write("## Model comparison (5-fold stratified CV, ROC-AUC)\n\n")
        f.write("| Model | Features | Mean AUC | Std |\n|---|---|---|---|\n")
        f.write(f"| Clinical-only baseline | {len(CLINICAL_COLS)} | {baseline_scores.mean():.3f} | {baseline_scores.std():.3f} |\n")
        f.write(f"| Multi-omics fusion | {fusion_X.shape[1]} | {fusion_scores.mean():.3f} | {fusion_scores.std():.3f} |\n")
        f.write(f"\n**Lift from multi-omics fusion: {lift:+.3f} AUC**\n\n")
        f.write("## Top 15 features by SHAP importance\n\n")
        f.write(mean_abs_shap.head(15).to_markdown())
        f.write("\n")
    print("Saved reports/baseline_results.md")
