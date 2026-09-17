"""Pull TNBC-relevant multi-omics data from the public cBioPortal REST API.

Discovery cohort: METABRIC (brca_metabric) - has ER/PR/HER2 IHC status,
PAM50+claudin-low subtype call, mRNA z-scores, discrete CNA, and survival.
Used for supervised training/testing (Basal-like vs non-basal TNBC).

External cohort: TCGA breast cancer (brca_tcga_pub2015) - has ER/PR/HER2
IHC status and RNA-seq v2 z-scores, but no PAM50 call in the public
clinical table, so it's used unlabeled for a qualitative biology check
rather than a second supervised split.

No API key required; cBioPortal's public instance serves published studies
for research use.
"""
import json
import sys
import time

import pandas as pd
import requests

from gene_panel import ALL_GENES

API = "https://www.cbioportal.org/api"
TIMEOUT = 60


def get(path, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def post(path, body, **params):
    r = requests.post(f"{API}{path}", json=body, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def resolve_entrez_ids(gene_symbols):
    genes = post("/genes/fetch", gene_symbols, geneIdType="HUGO_GENE_SYMBOL")
    symbol_to_entrez = {g["hugoGeneSymbol"]: g["entrezGeneId"] for g in genes}
    missing = sorted(set(gene_symbols) - set(symbol_to_entrez))
    if missing:
        print(f"  WARNING: {len(missing)} symbols not resolved: {missing}", file=sys.stderr)
    return symbol_to_entrez


def fetch_clinical(study_id, data_type):
    rows = get(
        f"/studies/{study_id}/clinical-data",
        clinicalDataType=data_type,
        projection="SUMMARY",
        pageSize=100000,
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    wide = df.pivot_table(
        index="sampleId" if data_type == "SAMPLE" else "patientId",
        columns="clinicalAttributeId",
        values="value",
        aggfunc="first",
    )
    return wide.reset_index()


def fetch_molecular(profile_id, sample_ids, entrez_ids, value_name):
    body = {"entrezGeneIds": entrez_ids, "sampleIds": sample_ids}
    rows = post(f"/molecular-profiles/{profile_id}/molecular-data/fetch", body, projection="SUMMARY")
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    entrez_to_symbol = {v: k for k, v in resolved.items()}
    df["gene"] = df["entrezGeneId"].map(entrez_to_symbol)
    wide = df.pivot_table(index="sampleId", columns="gene", values="value", aggfunc="first")
    wide = wide.add_suffix(f"_{value_name}")
    return wide


def is_negative(x):
    if x is None:
        return None
    x = str(x).strip().lower()
    if x in {"negative", "neg", "0"}:
        return True
    if x in {"positive", "pos", "1", "equivocal"}:
        return False
    return None


if __name__ == "__main__":
    print("Resolving gene panel to Entrez IDs...")
    resolved = resolve_entrez_ids(ALL_GENES)
    entrez_ids = list(resolved.values())
    print(f"  Resolved {len(resolved)}/{len(ALL_GENES)} genes.")

    # ---------------- METABRIC (discovery/labeled cohort) ----------------
    print("\n[METABRIC] Fetching sample-level clinical data...")
    meta_clin = fetch_clinical("brca_metabric", "SAMPLE")
    print(f"  {len(meta_clin)} samples with clinical data.")

    print("[METABRIC] Fetching patient-level clinical data (subtype call lives here)...")
    meta_clin_patient = fetch_clinical("brca_metabric", "PATIENT")
    print(f"  {len(meta_clin_patient)} patients with clinical data.")
    # METABRIC is one-sample-per-patient with sampleId == patientId.
    meta_clin = meta_clin.merge(
        meta_clin_patient[["patientId", "CLAUDIN_SUBTYPE", "THREEGENE"]],
        left_on="sampleId",
        right_on="patientId",
        how="left",
        suffixes=("", "_pt"),
    )

    meta_clin["er_neg"] = meta_clin.get("ER_STATUS").map(is_negative)
    meta_clin["pr_neg"] = meta_clin.get("PR_STATUS").map(is_negative)
    meta_clin["her2_neg"] = meta_clin.get("HER2_STATUS").map(is_negative)
    tnbc_mask = (
        (meta_clin["er_neg"] == True)
        & (meta_clin["pr_neg"] == True)
        & (meta_clin["her2_neg"] == True)
    )
    meta_tnbc = meta_clin[tnbc_mask].copy()
    print(f"  {len(meta_tnbc)} samples are ER-/PR-/HER2- (TNBC by IHC).")
    print("  CLAUDIN_SUBTYPE distribution within TNBC:")
    print(meta_tnbc["CLAUDIN_SUBTYPE"].value_counts(dropna=False).to_string())

    sample_ids = meta_tnbc["sampleId"].tolist()
    print(f"\n[METABRIC] Fetching mRNA z-scores for {len(sample_ids)} TNBC samples x {len(entrez_ids)} genes...")
    meta_expr = fetch_molecular(
        "brca_metabric_mrna_median_all_sample_Zscores", sample_ids, entrez_ids, "expr"
    )
    print(f"  Got expression matrix: {meta_expr.shape}")

    print("[METABRIC] Fetching discrete CNA for the same genes...")
    meta_cna = fetch_molecular("brca_metabric_cna", sample_ids, entrez_ids, "cna")
    print(f"  Got CNA matrix: {meta_cna.shape}")

    meta_full = (
        meta_tnbc.set_index("sampleId")
        .join(meta_expr, how="inner")
        .join(meta_cna, how="left")
    )
    meta_full.to_csv("data/metabric_tnbc.csv")
    print(f"  Saved data/metabric_tnbc.csv ({meta_full.shape[0]} samples, {meta_full.shape[1]} columns).")

    # ---------------- TCGA (external, unlabeled biology check) ----------------
    print("\n[TCGA pub2015] Fetching sample-level clinical data...")
    tcga_clin = fetch_clinical("brca_tcga_pub2015", "SAMPLE")
    print(f"  {len(tcga_clin)} samples with clinical data.")

    tcga_clin["er_neg"] = tcga_clin.get("ER_STATUS_BY_IHC").map(is_negative)
    tcga_clin["pr_neg"] = tcga_clin.get("PR_STATUS_BY_IHC").map(is_negative)
    her2_col = "IHC_HER2" if "IHC_HER2" in tcga_clin.columns else "HER2_FISH_STATUS"
    tcga_clin["her2_neg"] = tcga_clin.get(her2_col).map(is_negative)
    tnbc_mask_tcga = (
        (tcga_clin["er_neg"] == True)
        & (tcga_clin["pr_neg"] == True)
        & (tcga_clin["her2_neg"] == True)
    )
    tcga_tnbc = tcga_clin[tnbc_mask_tcga].copy()
    print(f"  {len(tcga_tnbc)} samples are ER-/PR-/HER2- (TNBC by IHC).")

    sample_ids_tcga = tcga_tnbc["sampleId"].tolist()
    print(f"\n[TCGA] Fetching RNA-seq z-scores for {len(sample_ids_tcga)} TNBC samples...")
    tcga_expr = fetch_molecular(
        "brca_tcga_pub2015_mrna_median_all_sample_Zscores", sample_ids_tcga, entrez_ids, "expr"
    )
    print(f"  Got expression matrix: {tcga_expr.shape}")

    tcga_full = tcga_tnbc.set_index("sampleId").join(tcga_expr, how="inner")
    tcga_full.to_csv("data/tcga_tnbc.csv")
    print(f"  Saved data/tcga_tnbc.csv ({tcga_full.shape[0]} samples, {tcga_full.shape[1]} columns).")

    print("\nDone.")
