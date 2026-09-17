# TNBC Basal vs Non-Basal: Baseline Results

Cohort: METABRIC TNBC (ER-/PR-/HER2-), n=320 (157 basal-like, 163 non-basal)

## Model comparison (5-fold stratified CV, ROC-AUC)

| Model | Features | Mean AUC | Std |
|---|---|---|---|
| Clinical-only baseline | 4 | 0.610 | 0.047 |
| Multi-omics fusion | 154 | 0.920 | 0.004 |

**Lift from multi-omics fusion: +0.311 AUC**

## Top 15 features by SHAP importance

|            |        0 |
|:-----------|---------:|
| ACSL4_expr | 0.821352 |
| MELK_expr  | 0.660729 |
| CCNE1_expr | 0.587571 |
| CXCL9_expr | 0.484118 |
| VIM_expr   | 0.403589 |
| BLVRA_expr | 0.399547 |
| SFRP1_expr | 0.37728  |
| GZMB_expr  | 0.366386 |
| CT83_expr  | 0.349101 |
| GRB7_expr  | 0.318897 |
| FGFR4_expr | 0.307811 |
| CCNB1_cna  | 0.274233 |
| CENPF_expr | 0.266901 |
| NCOA4_expr | 0.241067 |
| FTL_expr   | 0.235847 |
