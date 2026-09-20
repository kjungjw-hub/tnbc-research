"""Phase D analysis: does what Phikon "sees" in H&E tiles track known TNBC
biology?

We don't have subtype labels for the TCGA imaging cohort (see README), so
this isn't a second supervised split - it's an honest, exploratory check:
cluster the frozen Phikon image embeddings into two groups, then ask
whether those image-only clusters differ in a basal-marker expression score
computed independently from the matched RNA-seq data for the same samples.
n=39, so this is a directional signal, not a claim.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASAL_MARKERS = ["KRT5", "KRT14", "KRT17", "EGFR", "VIM", "FOXC1"]
LUMINAL_MARKERS = ["ESR1", "FOXA1", "GATA3", "PGR"]


def load_merged():
    emb = pd.read_csv("data/tcga_imaging_embeddings.csv", index_col=0)
    expr = pd.read_csv("data/tcga_tnbc.csv", index_col=0)
    expr = expr.copy()
    expr["case_id"] = [s[:12] for s in expr.index]
    expr = expr.set_index("case_id")
    merged_ids = emb.index.intersection(expr.index)
    return emb.loc[merged_ids], expr.loc[merged_ids]


def basal_score(expr):
    basal_cols = [f"{g}_expr" for g in BASAL_MARKERS if f"{g}_expr" in expr.columns]
    luminal_cols = [f"{g}_expr" for g in LUMINAL_MARKERS if f"{g}_expr" in expr.columns]
    b = expr[basal_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    l = expr[luminal_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return b - l


if __name__ == "__main__":
    emb, expr = load_merged()
    print(f"Matched imaging + expression samples: {len(emb)}")

    scaled = StandardScaler().fit_transform(emb.values)
    # Full (deterministic) SVD: n_samples=39 is small, and the randomized
    # solver's power iterations are numerically unstable on a matrix this
    # short and wide (39 x 768).
    pca = PCA(n_components=10, svd_solver="full", random_state=42)
    reduced = pca.fit_transform(scaled)
    print(f"PCA: top 10 components explain {pca.explained_variance_ratio_.sum():.1%} of embedding variance.")

    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    cluster = km.fit_predict(reduced)

    score = basal_score(expr)
    df = pd.DataFrame({"cluster": cluster, "basal_score": score.values}, index=emb.index)

    g0 = df.loc[df.cluster == 0, "basal_score"]
    g1 = df.loc[df.cluster == 1, "basal_score"]
    u_stat, p_value = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    pc1_corr, pc1_p = stats.pearsonr(reduced[:, 0], score.values)

    print(f"\nCluster 0 (n={len(g0)}): basal_score mean={g0.mean():.3f} +/- {g0.std():.3f}")
    print(f"Cluster 1 (n={len(g1)}): basal_score mean={g1.mean():.3f} +/- {g1.std():.3f}")
    print(f"Mann-Whitney U p-value: {p_value:.3f}")
    print(f"Pearson r(PC1 of image embedding, basal_score): {pc1_corr:.3f} (p={pc1_p:.3f})")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].boxplot([g0.values, g1.values], tick_labels=["Image cluster 0", "Image cluster 1"])
    axes[0].set_ylabel("Basal marker score (expression-based)")
    axes[0].set_title(f"Mann-Whitney p={p_value:.3f}, n={len(df)}")

    axes[1].scatter(reduced[:, 0], score.values, c=cluster, cmap="coolwarm", edgecolor="k")
    axes[1].set_xlabel("Image embedding PC1")
    axes[1].set_ylabel("Basal marker score")
    axes[1].set_title(f"Pearson r={pc1_corr:.2f}, p={pc1_p:.3f}")
    plt.tight_layout()
    plt.savefig("reports/imaging_analysis.png", dpi=150)
    print("\nSaved reports/imaging_analysis.png")

    with open("reports/imaging_analysis.md", "w") as f:
        f.write("# Phase D: imaging embeddings vs. known TNBC biology\n\n")
        f.write(
            "**Scope note:** the TCGA imaging cohort has no subtype ground truth "
            "(see README), so this is not a second supervised split. It's an "
            "exploratory check on whether frozen Phikon image embeddings, "
            "clustered without using any expression data, land on groups that "
            "differ in basal-marker expression computed independently from "
            f"matched RNA-seq. n={len(df)} - directional signal, not a claim.\n\n"
        )
        f.write(f"- Tissue tiles: top-3 by tissue density per slide (of a {4}x{4} coarse grid), "
                f"pulled directly from the GDC tile server (no full .svs download).\n")
        f.write("- Embedding: Phikon (owkin/phikon), CLS token, mean-pooled over each sample's tiles.\n")
        f.write(f"- Basal marker score: mean z-score of {BASAL_MARKERS} minus mean z-score of {LUMINAL_MARKERS}.\n\n")
        f.write("## Result\n\n")
        f.write("| Image cluster | n | Basal score mean +/- std |\n|---|---|---|\n")
        f.write(f"| 0 | {len(g0)} | {g0.mean():.3f} +/- {g0.std():.3f} |\n")
        f.write(f"| 1 | {len(g1)} | {g1.mean():.3f} +/- {g1.std():.3f} |\n\n")
        f.write(f"Mann-Whitney U p-value: **{p_value:.3f}**\n\n")
        f.write(f"Pearson r(image embedding PC1, basal score): **{pc1_corr:.3f}** (p={pc1_p:.3f})\n\n")
        f.write("![imaging analysis](imaging_analysis.png)\n")
    print("Saved reports/imaging_analysis.md")
