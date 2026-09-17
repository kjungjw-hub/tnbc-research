"""Phase B (stretch): attention-based multi-branch fusion network.

Upgrades the tree-based baseline (src/baseline_model.py) to a small neural
architecture that mirrors current multi-omics fusion literature (e.g.
DEDUCE, PACS): each modality (clinical / expression / CNA) is encoded
separately, then combined with a learned attention gate instead of naive
concatenation. The attention weights are themselves an interpretability
signal - which modality the model leaned on - complementary to the
gene-level SHAP values from the baseline.

Same label and cohort as the baseline: basal-like vs non-basal TNBC in the
METABRIC ER-/PR-/HER2- cohort, 5-fold stratified CV, ROC-AUC.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

CLINICAL_COLS = ["GRADE", "TUMOR_SIZE", "TUMOR_STAGE", "MUTATION_COUNT"]
BRANCH_NAMES = ["clinical", "expression", "cna"]


class BranchEncoder(nn.Module):
    def __init__(self, in_dim, hidden=16, embed=16, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class AttentionFusion(nn.Module):
    """Gated-attention fusion over an arbitrary number of modality branches."""

    def __init__(self, in_dims, embed=16, dropout=0.3):
        super().__init__()
        self.branches = nn.ModuleList(
            [BranchEncoder(d, embed=embed, dropout=dropout) for d in in_dims]
        )
        self.gate = nn.Linear(embed, 1)
        self.classifier = nn.Linear(embed, 1)

    def forward(self, xs):
        embeds = [branch(x) for branch, x in zip(self.branches, xs)]
        stacked = torch.stack(embeds, dim=1)  # (batch, n_branches, embed)
        scores = self.gate(stacked).squeeze(-1)  # (batch, n_branches)
        attn = torch.softmax(scores, dim=1)
        fused = (stacked * attn.unsqueeze(-1)).sum(dim=1)
        logit = self.classifier(fused).squeeze(-1)
        return logit, attn


def load_branches():
    df = pd.read_csv("data/metabric_tnbc.csv", index_col=0)
    y = (df["CLAUDIN_SUBTYPE"] == "Basal").astype(int).values

    clinical = df[CLINICAL_COLS].apply(pd.to_numeric, errors="coerce").values
    expr_cols = [c for c in df.columns if c.endswith("_expr")]
    cna_cols = [c for c in df.columns if c.endswith("_cna")]
    expr = df[expr_cols].apply(pd.to_numeric, errors="coerce").values
    cna = df[cna_cols].apply(pd.to_numeric, errors="coerce").values
    return y, [clinical, expr, cna]


def fit_transform_branch(train, test, val=None):
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_t = scaler.fit_transform(imp.fit_transform(train))
    test_t = scaler.transform(imp.transform(test))
    val_t = scaler.transform(imp.transform(val)) if val is not None else None
    return train_t, test_t, val_t


def to_tensors(branches):
    return [torch.tensor(b, dtype=torch.float32) for b in branches]


def train_one_fold(train_branches, y_train, val_branches, y_val, epochs=200, patience=20):
    dims = [b.shape[1] for b in train_branches]
    model = AttentionFusion(dims)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = to_tensors(train_branches)
    yt = torch.tensor(y_train, dtype=torch.float32)
    xv = to_tensors(val_branches)
    yv_np = y_val

    best_auc, best_state, patience_left = -1, None, patience
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logit, _ = model(xt)
        loss = loss_fn(logit, yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logit, _ = model(xv)
            val_pred = torch.sigmoid(val_logit).numpy()
        val_auc = roc_auc_score(yv_np, val_pred)

        if val_auc > best_auc:
            best_auc, best_state, patience_left = val_auc, {k: v.clone() for k, v in model.state_dict().items()}, patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    y, branches = load_branches()
    n_branches = len(branches)
    print(f"Cohort: {len(y)} samples, {y.sum()} basal-like / {len(y) - y.sum()} non-basal.")
    print(f"Branches: {list(zip(BRANCH_NAMES, [b.shape[1] for b in branches]))}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    attn_records = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(branches[0], y)):
        # carve an early-stopping validation split out of the fold's training data
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.15, stratify=y[train_idx], random_state=42
        )

        train_b, val_b, test_b = [], [], []
        for b in branches:
            tr_t, te_t, va_t = fit_transform_branch(b[train_idx], b[test_idx], b[val_idx])
            train_b.append(tr_t)
            val_b.append(va_t)
            test_b.append(te_t)

        model = train_one_fold(train_b, y[train_idx], val_b, y[val_idx])

        model.eval()
        with torch.no_grad():
            test_logit, test_attn = model(to_tensors(test_b))
            test_pred = torch.sigmoid(test_logit).numpy()
        auc = roc_auc_score(y[test_idx], test_pred)
        fold_aucs.append(auc)
        attn_records.append(test_attn.numpy())
        print(f"  Fold {fold + 1}: AUC = {auc:.3f}")

    fold_aucs = np.array(fold_aucs)
    mean_attn = np.concatenate(attn_records, axis=0).mean(axis=0)

    print(f"\n5-fold CV AUC: {fold_aucs.mean():.3f} +/- {fold_aucs.std():.3f}")
    print("\nMean attention weight per modality (how much the fused prediction leans on each branch):")
    for name, w in zip(BRANCH_NAMES, mean_attn):
        print(f"  {name:12s}: {w:.3f}")

    with open("reports/attention_fusion_results.md", "w") as f:
        f.write("# TNBC Basal vs Non-Basal: Attention-Fusion Model\n\n")
        f.write(
            "Gated-attention fusion network (clinical / expression / CNA branches, "
            "each encoded separately then combined with a learned attention gate) "
            "vs. the tree-based baseline in `reports/baseline_results.md`.\n\n"
        )
        f.write("## 5-fold stratified CV, ROC-AUC\n\n")
        f.write("| Fold | AUC |\n|---|---|\n")
        for i, a in enumerate(fold_aucs):
            f.write(f"| {i + 1} | {a:.3f} |\n")
        f.write(f"| **Mean** | **{fold_aucs.mean():.3f} +/- {fold_aucs.std():.3f}** |\n\n")
        f.write("## Learned attention weight per modality (mean over held-out folds)\n\n")
        f.write("| Modality | Attention weight |\n|---|---|\n")
        for name, w in zip(BRANCH_NAMES, mean_attn):
            f.write(f"| {name} | {w:.3f} |\n")
    print("\nSaved reports/attention_fusion_results.md")
