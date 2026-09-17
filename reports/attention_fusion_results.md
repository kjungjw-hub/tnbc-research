# TNBC Basal vs Non-Basal: Attention-Fusion Model

Gated-attention fusion network (clinical / expression / CNA branches, each encoded separately then combined with a learned attention gate) vs. the tree-based baseline in `reports/baseline_results.md`.

## 5-fold stratified CV, ROC-AUC

| Fold | AUC |
|---|---|
| 1 | 0.878 |
| 2 | 0.861 |
| 3 | 0.891 |
| 4 | 0.855 |
| 5 | 0.925 |
| **Mean** | **0.882 +/- 0.025** |

## Learned attention weight per modality (mean over held-out folds)

| Modality | Attention weight |
|---|---|
| clinical | 0.286 |
| expression | 0.399 |
| cna | 0.315 |
