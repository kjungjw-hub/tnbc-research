# TNBC Multi-Omics Subtyping

Rebuild of the TNBC research project (previously team-based, private clinical
data) on public cohorts, moving from a single-modality classifier toward a
multi-omics, explainable, deployable pipeline. Full rationale and phased plan:
see the design doc this repo was scaffolded from (multi-omics fusion →
foundation-model imaging → explainability → interactive demo → writeup).

**[View the pipeline flow diagram](flow.html)** — visual status of every
phase, kept in sync with this repo (open locally, or via GitHub Pages once
enabled). Prefer plain language? **[Explain-like-I'm-5 version](eli5.html)**.

## Status: Phase A-E complete (F, the writeup, still open)

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

**Imaging branch (Phase D)** — `src/fetch_imaging.py` pulls H&E tissue tiles
straight from the GDC tile server (no multi-GB `.svs` download: it fetches
individual DeepZoom-style tiles, scores each for tissue content, and keeps
the top 3 per slide) for the 40 TCGA TNBC slides with a diagnostic image
available, then embeds them with **Phikon** (Owkin) — an open, non-gated
pathology foundation model pretrained on TCGA histology via self-supervised
learning (`mahmoodlab/UNI` and `CONCH` from the original plan are
HuggingFace-gated and need manual license acceptance on an account this
environment doesn't have; Phikon is the public equivalent, and a fitting one
since our slides are TCGA slides too).

`src/imaging_analysis.py` then asks an honest question: with no subtype
label for the imaging cohort, does an unsupervised cluster of the image
embeddings alone line up with basal-marker gene expression from the matched
RNA-seq (computed independently)? Result (`reports/imaging_analysis.png`,
`reports/imaging_analysis.md`): **no**, not at this sample size (n=39,
Mann-Whitney p=0.72, PC1 correlation r=-0.12, p=0.45). Reported as a null
result rather than reframed to look positive — a real limitation of a
39-sample exploratory check, not evidence the idea is wrong.

Run with:
```
python3 src/fetch_imaging.py
python3 src/imaging_analysis.py
```

**Interactive demo (Phase E)** — `app.py` is a Streamlit app: pick a real
(anonymized) METABRIC patient and see the fusion model's predicted
probability of basal-like subtype next to the known ground-truth call, plus
a live local SHAP explanation. A "what if" panel lets you drag the model's
top genes and watch the prediction update in real time. Verified working
end-to-end locally (sample switching, live SHAP, live slider re-prediction).

Run with:
```
python3 src/baseline_model.py   # writes models/fusion_model.joblib first
streamlit run app.py
```

Not yet deployed to HuggingFace Spaces — that needs an HF account/token this
environment doesn't have; the app runs identically once deployed there.

## Next
- Phase F: write up as a short report/preprint.
- Deploy `app.py` to HuggingFace Spaces once there's an HF token to push with.

## Setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
