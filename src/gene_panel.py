"""Curated gene panel for TNBC subtyping and biomarker interpretability.

Combines four biologically-motivated groups so SHAP outputs can be checked
against known TNBC biology instead of being a black box over ~20k genes:

- PAM50: standard breast cancer intrinsic subtype genes (basal-like signal).
- BASAL_MARKERS: classic basal/myoepithelial and EGFR-axis markers used to
  distinguish basal-like TNBC from non-basal (LAR/MSL-like) TNBC.
- FERROPTOSIS: genes flagged in recent TNBC multi-omics literature
  (see plan file) as prognostic in TNBC.
- IMMUNE: immune-checkpoint / cytotoxic T-cell genes relevant to the
  immunomodulatory (IM) TNBC subtype.
"""

PAM50 = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2",
    "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C",
    "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67",
    "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B",
    "TYMS", "UBE2C", "UBE2T",
]

BASAL_MARKERS = ["KRT8", "KRT18", "VIM", "MET", "AR", "SOX10", "GATA3"]

FERROPTOSIS = [
    "GPX4", "ACSL4", "SLC7A11", "TFRC", "PTGS2", "ALOX15", "FTH1", "FTL",
    "NCOA4", "CT83",
]

IMMUNE = ["CD274", "PDCD1", "CTLA4", "CD8A", "GZMB", "STAT1", "IDO1", "CXCL9"]

ALL_GENES = sorted(set(PAM50 + BASAL_MARKERS + FERROPTOSIS + IMMUNE))
