# TNBC Multi-Omics Subtyping

Rebuild of the TNBC research project (previously team-based, private clinical
data) on public cohorts, moving from a single-modality classifier toward a
multi-omics, explainable, deployable pipeline. Full rationale and phased plan:
see the design doc this repo was scaffolded from (multi-omics fusion →
foundation-model imaging → explainability → interactive demo → writeup).

**[View the pipeline flow diagram](flow.html)** — visual status of every
phase, kept in sync with this repo (open locally, or via GitHub Pages once
enabled).

## Status: Phase A + B (baseline) complete

**Data (Phase A)** — pulled from the public [cBioPortal](https://www.cbioportal.org)
REST API, no auth required:
- `data/metabric_tnbc.csv` — 320 TNBC samples (ER-/PR-/HER2- by IHC) from
  METABRIC, with clinical features, PAM50+claudin-low subtype call, mRNA
  z-scores and discrete CNA for a curated 75-gene panel
  (`src/gene_panel.py`: PAM50 + basal markers + ferroptosis genes + immune
  genes — chosen so SHAP output stays biologically checkable).
- `data/tcga_tnbc.csv` — 42 TNBC samples from TCGA (`brca_tcga_pub2015`),
  unlabeled (no public PAM50 call), held out for a future qualitative
  external-biology check rather than a second supervised split.

Regenerate with:
```
source .venv/bin/activate
python3 src/fetch_data.py
```

**Baseline model (Phase B, omics-only)** — `src/baseline_model.py` predicts
basal-like vs. non-basal TNBC (a real, clinically used split within TNBC;
non-basal skews toward the better-prognosis LAR/MSL-like end of Lehmann's
subtypes) via 5-fold stratified CV:

| Model | Features | Mean AUC |
|---|---|---|
| Clinical-only baseline (grade, size, stage, mutation count) | 4 | 0.610 |
| Multi-omics fusion (clinical + expression + CNA) | 154 | **0.920** |

SHAP (`reports/shap_summary.png`, `reports/baseline_results.md`) shows the
top drivers are biologically sensible: proliferation/basal genes (MELK,
CENPF, CCNB1, VIM), an immune signal (CXCL9, GZMB) consistent with the
immunomodulatory TNBC subtype, and a ferroptosis gene (ACSL4) — echoing
recent TNBC multi-omics literature.

Run with:
```
python3 src/baseline_model.py
```

**Attention-fusion model (Phase B, stretch)** — `src/attention_fusion.py`
replaces naive feature concatenation with a small PyTorch network: each
modality (clinical / expression / CNA) is encoded separately, then combined
via a learned attention gate (per DEDUCE/PACS-style multi-omics fusion
architectures), instead of one tree ensemble over all features pooled
together.

| Model | Mean AUC |
|---|---|
| Tree-based fusion (`baseline_model.py`) | **0.920 +/- 0.004** |
| Attention-fusion network (`attention_fusion.py`) | 0.882 +/- 0.025 |

Honest result: the tree ensemble still edges out the neural network on raw
AUC, which is expected for a dataset this size (n=320) — trees generally win
on small tabular data. What the attention model adds instead is a *native*
per-modality interpretability signal that a tree ensemble doesn't give you
directly: learned attention weight of 0.399 on expression, 0.315 on CNA,
0.286 on clinical (`reports/attention_fusion_results.md`) — i.e. the model
itself reports how much it leaned on each data type, not just which genes.

Run with:
```
python3 src/attention_fusion.py
```

## Next
- Add the imaging branch: frozen UNI/CONCH pathology foundation-model
  embeddings on matched TCGA H&E slides, fused in as a fourth branch of the
  attention network.
- Wrap the model in a small Streamlit/Gradio demo and deploy to HuggingFace
  Spaces.
- Write up as a short report/preprint once the fused model is in place.

## Setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
