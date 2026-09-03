"""
Unsupervised Feature Selection (UFS) baselines for Experiment 2.

Ported verbatim (same math, same defaults) from
experiment_results/experiment_2_ufs/ufs_extended.ipynb's first code cell --
the notebook that produced
experiment_results/experiment_2_ufs/ufs_results_final.csv, the paper's
DIADA-vs-UFS noise-filtering comparison. See
src/diada/experiments/experiment_2_ufs/ufs_benchmark.py for the live,
reproducible runner and its own docstring for every deliberate difference
from the notebook.

No behavior changes were made porting this module out of the notebook --
only import cleanup and removing the notebook's own I/O (CSV writing,
progress printing), which the runner now owns.
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


# ============================================================
# NOISE HELPERS
# ============================================================

def is_noise(col) -> bool:
    s = str(col)
    return "noise" in s or "shuffle" in s or "spurious" in s


def count_noise(cols) -> int:
    return sum(is_noise(c) for c in cols)


# ============================================================
# PREPROCESSING
# ============================================================

def prepare_dataframe(df):
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    groups = {}
    parts = []
    offset = 0

    if num_cols:
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        arr = imp.fit_transform(df[num_cols])
        arr = scaler.fit_transform(arr)

        parts.append(arr)

        for i, c in enumerate(num_cols):
            groups[c] = [offset + i]

        offset += len(num_cols)

    if cat_cols:
        cat = df[cat_cols].astype(str).fillna("__missing__")

        try:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

        carr = ohe.fit_transform(cat)
        parts.append(carr)

        for j, c in enumerate(cat_cols):
            nlev = len(ohe.categories_[j])
            groups[c] = list(range(offset, offset + nlev))
            offset += nlev

    X = np.hstack(parts)

    return X, groups


# ============================================================
# SCORE AGGREGATION
# ============================================================

def aggregate_scores(scores, groups, larger_better=True):
    out = {}
    scores = np.asarray(scores).ravel()

    for col, idxs in groups.items():
        vals = scores[idxs]
        out[col] = float(np.max(vals) if larger_better else np.min(vals))

    return out


# ============================================================
# AUTO-SELECTION HELPERS
# ============================================================

def _elbow_cut(values: np.ndarray, larger_better: bool) -> int:
    n = len(values)

    if n == 1:
        return 1

    sorted_vals = np.sort(values)[::-1] if larger_better else np.sort(values)

    lo, hi = sorted_vals.min(), sorted_vals.max()

    if hi == lo:
        return n

    normed = (sorted_vals - lo) / (hi - lo)

    if n == 2:
        return 1

    d2 = np.diff(normed, n=2)

    elbow_pos = int(np.argmax(np.abs(d2))) + 1

    k = elbow_pos + 1

    return max(1, min(k, n))


def _select_top_k(scores: np.ndarray, k: int, larger_better: bool):
    if larger_better:
        return np.argsort(scores)[::-1][:k]
    else:
        return np.argsort(scores)[:k]


def _auto_select(scores, k, larger_better):
    if k is None:
        k = _elbow_cut(scores, larger_better)

    return _select_top_k(scores, k, larger_better)


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_similarity_graph(X, k=5, sigma=1.0):
    knn = kneighbors_graph(
        X,
        n_neighbors=k,
        mode="distance",
        include_self=False,
    )

    D = knn.toarray()

    W = np.exp(-(D ** 2) / (2 * sigma ** 2))

    W[D == 0] = 0

    W = np.maximum(W, W.T)

    return W


# ============================================================
# VARIANCE SCORE
# ============================================================

def variance_score(X, k=None):
    scores = np.var(X, axis=0)
    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# LAPLACIAN SCORE
# ============================================================

def laplacian_score(X, W, k=None):
    D_mat = np.diag(W.sum(axis=1))
    L = D_mat - W
    d_diag = np.diag(D_mat)

    scores = []
    for j in range(X.shape[1]):
        f = X[:, j]
        f = f - (f @ d_diag) / d_diag.sum()
        num = f @ L @ f
        den = f @ D_mat @ f + 1e-12
        scores.append(num / den)

    scores = np.array(scores)
    selected = _auto_select(scores, k, larger_better=False)
    return scores, selected


# ============================================================
# SPEC
# ============================================================

def spec_score(X, W, k=None):
    D_mat = np.diag(W.sum(axis=1))
    L = D_mat - W

    eigvals, eigvecs = eigh(L, D_mat)

    scores = []
    for j in range(X.shape[1]):
        f = X[:, j]
        coeff = np.abs(eigvecs.T @ f)
        scores.append(np.sum(coeff[1:] / (eigvals[1:] + 1e-12)))

    scores = np.array(scores)
    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# MCFS
# ============================================================

def mcfs_score(X, W, n_clusters=5, k=None):
    D_mat = np.diag(W.sum(axis=1))
    L = D_mat - W

    eigvals, eigvecs = eigh(L, D_mat)
    U = eigvecs[:, 1:n_clusters + 1]

    scores = []
    for j in range(X.shape[1]):
        f = X[:, j]
        score = np.sum(np.abs(np.corrcoef(f, U.T)[0, 1:]))
        scores.append(score)

    scores = np.array(scores)
    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# NDFS
# ============================================================

def ndfs_score(X, W, n_clusters=5, alpha=1.0, beta=1.0, k=None):
    D = np.diag(W.sum(axis=1))
    L = D - W

    eigvals, eigvecs = eigh(L, D)
    Y = eigvecs[:, 1:n_clusters + 1]

    p = X.shape[1]
    scores = np.zeros(p)

    for j in range(p):
        f = X[:, j]
        corr = np.sum(np.abs(np.corrcoef(f, Y.T)[0, 1:]))
        sparsity = np.linalg.norm(f, ord=1)
        scores[j] = corr / (alpha * sparsity + beta)

    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# RSR
# ============================================================

def rsr_score(X, W, n_clusters=5, alpha=0.01, k=None):
    D = np.diag(W.sum(axis=1))
    L = D - W

    eigvals, eigvecs = eigh(L, D)
    Y = eigvecs[:, 1:n_clusters + 1]

    p = X.shape[1]
    scores = np.zeros(p)

    for i in range(Y.shape[1]):
        target = Y[:, i]
        model = Lasso(alpha=alpha, max_iter=2000)
        model.fit(X, target)
        scores += np.abs(model.coef_)

    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# AGUFS
# ============================================================

def agufs_score(X, W, gamma=1.0, k=None):
    D = np.diag(W.sum(axis=1))
    L = D - W

    p = X.shape[1]
    scores = np.zeros(p)

    for j in range(p):
        f = X[:, j]
        smoothness = (f.T @ L @ f) / (f.T @ D @ f + 1e-12)
        variance = np.var(f)
        scores[j] = variance / (smoothness + gamma)

    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# UDFS-LIKE
# ============================================================

def udfs_like_score(X, alpha=0.01, k=None):
    p = X.shape[1]
    scores = np.zeros(p)

    for j in range(p):
        y = X[:, j]
        X_other = np.delete(X, j, axis=1)
        model = Lasso(alpha=alpha, max_iter=2000)
        model.fit(X_other, y)
        scores[j] = np.sum(np.abs(model.coef_))

    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# CORRELATION UNIQUENESS
# ============================================================

def correlation_uniqueness_score(X, k=None):
    corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr)

    scores = []
    for i in range(corr.shape[0]):
        mean_abs_corr = np.mean(np.abs(np.delete(corr[i], i)))
        scores.append(1.0 / (mean_abs_corr + 1e-8))

    scores = np.array(scores)
    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# mRMR
# ============================================================

def mrmr_score(X, n_clusters=5, k=None):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(X)

    p = X.shape[1]
    relevance = np.array([
        mutual_info_classif(X[:, [j]], labels)[0]
        for j in range(p)
    ])

    corr = np.abs(np.corrcoef(X.T))
    redundancy = np.zeros(p)
    for j in range(p):
        redundancy[j] = np.mean(np.delete(corr[j], j))

    scores = relevance - redundancy
    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# JMI
# ============================================================

def jmi_score(X, n_clusters=5, k=None):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(X)

    p = X.shape[1]
    relevance = np.array([
        mutual_info_classif(X[:, [j]], labels)[0]
        for j in range(p)
    ])

    corr = np.abs(np.corrcoef(X.T))
    scores = np.zeros(p)
    for j in range(p):
        redundancy = np.mean(np.delete(corr[j], j))
        scores[j] = relevance[j] / (redundancy + 1e-12)

    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# CMIM
# ============================================================

def cmim_score(X, n_clusters=5, k=None):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(X)

    p = X.shape[1]
    relevance = np.array([
        mutual_info_classif(X[:, [j]], labels)[0]
        for j in range(p)
    ])

    corr = np.abs(np.corrcoef(X.T))
    scores = np.zeros(p)
    for j in range(p):
        redundancy = np.max(np.delete(corr[j], j))
        scores[j] = relevance[j] - redundancy

    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# INF-FS
# ============================================================

def inf_fs_score(X, alpha=0.5, k=None):
    p = X.shape[1]

    corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(corr, 0)

    stds = np.std(X, axis=0)
    std_matrix = np.outer(stds, stds)

    A = alpha * corr + (1 - alpha) * std_matrix

    I = np.eye(p)
    S = np.linalg.inv(I - 0.9 * A)

    scores = np.sum(S, axis=1)
    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# CLUSTER DISPERSION
# ============================================================

def cluster_dispersion_score(X, n_clusters=5, random_state=0, k=None):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)

    p = X.shape[1]
    scores = np.zeros(p)
    global_mean = X.mean(axis=0)

    for j in range(p):
        between = 0.0
        within = 0.0

        for c in np.unique(labels):
            idx = labels == c
            xc = X[idx, j]
            mu_c = xc.mean()
            n_c = idx.sum()

            between += n_c * (mu_c - global_mean[j]) ** 2
            within += np.sum((xc - mu_c) ** 2)

        scores[j] = between / (within + 1e-12)

    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# PFA
# ============================================================

def principal_feature_analysis_score(X, n_components=None, k=None):
    if n_components is None:
        n_components = min(5, X.shape[1])

    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1][:n_components]
    V = eigvecs[:, idx]

    scores = np.sum(V ** 2, axis=1)
    selected = _auto_select(scores, k, larger_better=True)
    return scores, selected


# ============================================================
# FEATURE CLUSTER REPRESENTATIVE
# ============================================================

def feature_cluster_representative_score(X, n_feature_clusters=5, k=None):
    F = X.T
    n_clusters_eff = min(n_feature_clusters, F.shape[0])

    km = KMeans(n_clusters=n_clusters_eff, n_init=10, random_state=0)
    labels = km.fit_predict(F)
    centers = km.cluster_centers_

    scores = np.zeros(F.shape[0])
    for i in range(F.shape[0]):
        c = labels[i]
        dist = np.linalg.norm(F[i] - centers[c])
        scores[i] = 1.0 / (dist + 1e-12)

    if k is None:
        selected = []
        for c in range(n_clusters_eff):
            members = np.where(labels == c)[0]
            best = members[np.argmax(scores[members])]
            selected.append(int(best))
        selected = np.array(selected)
    else:
        selected = _select_top_k(scores, k, larger_better=True)

    return scores, selected


# ============================================================
# CONCRETE AUTOENCODER (CAE) -- optional, needs torch
# ============================================================

def _build_cae_score():
    """Returns cae_score(X, k_select=10, epochs=50, lr=1e-3) if torch is
    installed, else None. Same auto-detect-and-skip pattern as Experiment
    1's Apriori/FP-Growth (see scalability_benchmark.py)."""
    if not TORCH_AVAILABLE:
        return None

    import torch
    import torch.nn as nn
    import torch.optim as optim

    class ConcreteSelector(nn.Module):
        def __init__(self, input_dim, k):
            super().__init__()
            self.logits = nn.Parameter(torch.randn(k, input_dim))

        def forward(self, x, temperature=0.1):
            weights = torch.softmax(self.logits / temperature, dim=1)
            selected = torch.matmul(x, weights.T)
            return selected, weights

    class CAEModel(nn.Module):
        def __init__(self, input_dim, k, hidden_dim=128):
            super().__init__()
            self.selector = ConcreteSelector(input_dim, k)
            self.decoder = nn.Sequential(
                nn.Linear(k, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            selected, weights = self.selector(x)
            recon = self.decoder(selected)
            return recon, weights

    def cae_score(X, k_select=10, epochs=50, lr=1e-3):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        model = CAEModel(X.shape[1], k_select)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            recon, weights = model(X_tensor)
            loss = loss_fn(recon, X_tensor)
            loss.backward()
            optimizer.step()

        final_weights = model.selector.logits.detach().cpu().numpy()
        scores = np.max(final_weights, axis=0)
        selected = np.argsort(scores)[::-1][:k_select]

        return scores, selected

    return cae_score


cae_score = _build_cae_score()


# ============================================================
# X INDICES -> ORIGINAL COLUMNS
# ============================================================

def x_indices_to_columns(selected_x_indices, groups):
    selected_set = set(int(i) for i in selected_x_indices)
    return [
        col
        for col, idxs in groups.items()
        if any(i in selected_set for i in idxs)
    ]


# ============================================================
# METHOD REGISTRY
# ============================================================

def build_methods(X, W, n_keep=None, mcfs_clusters=5, udfs_alpha=0.01) -> Dict[str, Tuple[callable, bool]]:
    """{method_name: (thunk, larger_better)}, thunk() -> (scores, selected_x_indices).
    Mirrors the notebook's `methods` dict in run_ufs_benchmark, but with cae
    included (re-enabled per explicit instruction -- it was commented out in
    the notebook -- rather than left out; auto-skipped if torch isn't
    installed, same as the notebook's own TORCH_AVAILABLE guard)."""
    methods: Dict[str, Tuple[callable, bool]] = {
        "variance_score": (lambda: variance_score(X, k=n_keep), True),
        "laplacian_score": (lambda: laplacian_score(X, W, k=n_keep), False),
        "spec": (lambda: spec_score(X, W, k=n_keep), True),
        "mcfs": (lambda: mcfs_score(X, W, n_clusters=mcfs_clusters, k=n_keep), True),
        "ndfs": (lambda: ndfs_score(X, W, n_clusters=mcfs_clusters, k=n_keep), True),
        "rsr": (lambda: rsr_score(X, W, n_clusters=mcfs_clusters, k=n_keep), True),
        "agufs": (lambda: agufs_score(X, W, k=n_keep), True),
        "udfs_like": (lambda: udfs_like_score(X, alpha=udfs_alpha, k=n_keep), True),
        "correlation_uniqueness": (lambda: correlation_uniqueness_score(X, k=n_keep), True),
        "mrmr": (lambda: mrmr_score(X, n_clusters=mcfs_clusters, k=n_keep), True),
        "jmi": (lambda: jmi_score(X, n_clusters=mcfs_clusters, k=n_keep), True),
        "cmim": (lambda: cmim_score(X, n_clusters=mcfs_clusters, k=n_keep), True),
        "inf_fs": (lambda: inf_fs_score(X, k=n_keep), True),
        "cluster_dispersion": (lambda: cluster_dispersion_score(X, n_clusters=mcfs_clusters, k=n_keep), True),
        "principal_feature_analysis": (lambda: principal_feature_analysis_score(X, k=n_keep), True),
        "feature_cluster_representative": (lambda: feature_cluster_representative_score(X, n_feature_clusters=mcfs_clusters, k=n_keep), True),
    }

    if cae_score is not None:
        methods["cae"] = (lambda: cae_score(X, k_select=n_keep or 10), True)

    return methods
