"""Method B — KGAB: DIFFERENTIABLE training of the gate α through frozen TabPFN,
with LAYER-SELECTIVE injection (all layers vs the 'causal' middle band only).

The paper's fix (§8) is a ZERO-INITIALISED gate α trained against the in-context NLL
by back-prop through the FROZEN forward (only α updates). Because α is zero-init,
L(α=0) = vanilla TabPFN, so training "can only help or stay neutral".

NEW — WHERE to inject (per-layer gate).  Feature-interaction / causal-probing studies
report that a transformer's MIDDLE blocks carry the feature–feature dependency structure,
while early blocks do low-level encoding and late blocks do read-out. Forcing the KG prior
on ALL 24 blocks therefore risks distorting representations that are not about feature
relations — a plausible reason the fixed-α sweep hurt easy tasks. This file answers
"which layers want the KG?" two complementary ways (`--mode`):

  * mode=scan  — DIAGNOSTIC, no training. Light up ONE block at a time at a fixed α and
    read test acc. No optimiser coupling, so the per-layer acc profile is a clean estimate
    of each block's marginal effect — a robust check of the "middle blocks are causal"
    premise BEFORE trusting anything learned.
  * mode=train — METHOD. A SEPARATE zero-init α per block (a length-24 vector), trained
    against the in-context NLL through the frozen forward. All-α=0 ≡ vanilla TabPFN, so it
    "can only help or stay neutral", and the learned α-profile shows where the KG helped —
    an EMPIRICAL answer instead of presupposing 4-6. Optional --l1 sparsifies the profile.
  * mode=both  — scan, then train, then you compare the two profiles (consistency check).

Caveat: per-layer α's interact (a late block can compensate an early one), so the learned
profile is suggestive, not a clean causal attribution — hence the fixed scan as ground
truth. A "4-6 of 12" finding maps to ~8-15 here (TabPFNV2p6 has 24 blocks). Wiring:
tag_block_layers tags each AlongRowAttention with its block index; _KGAB["alpha"] then
accepts a length-24 tensor indexed by that block index (_resolve_alpha).

Differentiable forward. TabPFN's grad path is the "batched" engine used by fine-tuning:
    chunks = get_preprocessed_dataset_chunks(clf, X, y, split_fn, ...)   # ctx/query splits
    for batch in DataLoader(chunks, collate_fn=meta_dataset_collator):
        clf.fit_from_preprocessed(batch.X_context, batch.y_context, cat, batch.configs)
        logits = clf.forward(batch.X_query, return_raw_logits=True)      # differentiable
        loss = CE(logits, batch.y_query); loss.backward()                # grad -> α only
α lives in the patched AlongRowAttention.forward (kg_b_kgab); as a torch leaf the bias
α·M builds graph, so loss.backward() fills α.grad. TabPFN weights are frozen so only α
moves. Evaluation reuses the frozen sklearn inference path (kg_b_kgab.score_with_bias).

Data (intuitive M). Each community is 2 feature TOKENS = 6 raw features that are NEAR
IDENTICAL noisy copies of one latent group factor (FEATURES_PER_GROUP=3, so a "redundant
triple" is one token; two such tokens form a community). The group-pooled KG M is then a
clean block-diagonal of 2×2 cross-token blocks — exactly the structure KGAB biases. Note
the bias only matters BETWEEN tokens, so each community must span ≥2 tokens (≥6 features);
"every 3 features identical" alone would collapse a community to ONE token and leave M
empty. Lower --feature-noise => more identical triples (less head-room for the KG).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch
from functools import partial
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from kg_a_kgfp import corrupt_graph, kg_quality, normalize_adj  # noqa: E402
import kg_a_kgfp as A  # noqa: E402
import tabpfn.architectures.tabpfn_v2_6 as v26  # noqa: E402
import math
import torch.nn.functional as F

from tabpfn.classifier import TabPFNClassifier  # noqa: E402
from tabpfn.preprocessing import PreprocessorConfig  # noqa: E402
from tabpfn.finetuning.data_util import (  # noqa: E402
    get_preprocessed_dataset_chunks,
    meta_dataset_collator,
)

DGPS = A.DGPS
# B-specific label families (interaction dropped — KG is feature-feature, not joint-cause).
DGPS_B = ("linear", "quadratic", "interaction", "mlp")
DGP_N_REL_B = {"linear": 1, "quadratic": 1, "interaction": 2, "mlp": 3}
FEATURES_PER_GROUP = 3


# ---------------------------------------------------------------------------
# Simulation families (decoupled from kg_a's redundancy-only block SCM)
# ---------------------------------------------------------------------------
#   family="redundant"  legacy block SCM: community = noisy copies of one latent factor;
#                       the KG is a hard 0/1 block grouping. Kept only as a control.
#   family="factor"/"similarity"   REALISTIC SIMILARITY / ASSOCIATION KG (e.g. gene
#                       co-expression, embedding cosine). LOW-RANK FACTOR MODEL:
#                         e_j ~ unit vector in R^r           (random feature embedding)
#                         z_i ~ N(0, I_r)                     (per-sample latent)
#                         x_ij = <z_i, e_j> + noise
#                       Then Cov(x_p, x_q) = <e_p, e_q>, so the TRUE feature affinity is the
#                       Gram matrix A = E Eᵀ — a smooth similarity, NOT a partition. The KGE
#                       is literally E. Rank r is the structure knob (small r => strong
#                       low-rank redundancy => informative KG; r→d => features independent).
#                       `--knn k>0` sparsifies the Gram into a top-k similarity GRAPH (the
#                       form real co-expression / association nets take). NB this KG = feature
#                       COVARIANCE, so it is asymptotically RECOVERABLE from the in-context
#                       data => this is A/D's denoising turf; B (attention pooling) is the weak
#                       in-forward analog. Expect B ≈ base here even at quality=1 — by design.
#   family="interaction"/"epistasis"   EPISTASIS / SYNTHETIC-LETHAL / DRUG-SYNERGY KG: the
#                       graph carries info NOT recoverable from the data (see the branch
#                       below). `--n-edges` disjoint interacting node pairs; label = Σ products
#                       over the pairs; the pairing is ORTHOGONAL to the feature covariance, so
#                       at small n the data cannot reveal which pair interacts — the KG can.
#                       This is B's home turf (bring the partner token in for the MLP to
#                       multiply); A/D column-smoothing only forms a linear term and cannot.

def generate_data_b(
    *, dgp: str, family: str, n_train: int, n_test: int, n_features: int, n_groups: int,
    feature_noise: float, label_noise: float, rank: int, seed: int,
    shuffle_features: bool = False, n_edges: int = 1, knn: int = 0,
    label_combiner: str = "mlp", mlp_hidden: int = 16,
):
    """Self-contained B simulation; returns (A.Data, E) where E is the (d×r) feature
    embedding matrix for family='factor' (else None). Communities/features stay in fixed
    order (shuffle_features=False) so the 3-feature token layout (hence M) is stable."""
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    rel = DGP_N_REL_B[dgp]

    if family in ("factor", "similarity"):
        E = rng.standard_normal((n_features, rank)).astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12       # unit rows -> Gram is cosine
        Z = rng.standard_normal((n_total, rank)).astype(np.float32)
        X = Z @ E.T + feature_noise * rng.standard_normal((n_total, n_features))
        A_true = (E @ E.T).astype(np.float32)
        np.fill_diagonal(A_true, 0.0)
        if knn and knn > 0:                       # sparsify Gram -> realistic kNN similarity graph
            A_true = knn_graph(A_true, knn)
        U_rel = Z @ rng.standard_normal((rank, rel)).astype(np.float32)  # label dirs in z-space
        groups = np.zeros(n_features, dtype=int)
        g = A._score(U_rel, dgp, rng)
    elif family in ("interaction", "epistasis"):
        # INTERACTION-KG family: the KG carries info NOT recoverable from the data.
        #   * each node = FEATURES_PER_GROUP near-identical features of one latent u_a
        #     (within-node correlation -> coherent token; CROSS-node independent).
        #   * the KG is a sparse PAIRING (perfect matching) of NODES; the label is the sum
        #     of PRODUCTS u_a*u_b over the matched pairs. Cross-node Cov=0, so the matching
        #     is ORTHOGONAL to the feature covariance — you cannot read it off correlations,
        #     and at small n you cannot fit which of C(G,2) pairs interact. The KG tells you.
        #   * pure column smoothing (A/D) averages neighbours -> a LINEAR term, it cannot
        #     form u_a*u_b; biasing feature-attention (B) brings the partner token in so the
        #     downstream MLP can multiply -> this is B's home turf.
        fpg = FEATURES_PER_GROUP
        G_node = n_features // fpg
        n_features = G_node * fpg                         # trim to a clean node layout
        U = rng.standard_normal((n_total, G_node)).astype(np.float32)
        groups = np.repeat(np.arange(G_node), fpg)
        X = U[:, groups] + feature_noise * rng.standard_normal((n_total, n_features))
        # Pick `n_edges` DISJOINT interacting node pairs; the remaining nodes are DISTRACTORS
        # (present in X, independent, absent from the label and isolated in the KG). The KG
        # thus says both WHICH nodes matter and HOW they pair -> decisive, non-data info.
        ne = max(1, min(n_edges, G_node // 2))
        perm = rng.permutation(G_node)
        edges = [(int(perm[2 * i]), int(perm[2 * i + 1])) for i in range(ne)]
        A_node = np.zeros((G_node, G_node), np.float32)
        # === 改动:收集每条 KG 边的交互项(只放乘积 -> 纯交互,无主效应) ===
        prods = []
        for a, b in edges:
            A_node[a, b] = A_node[b, a] = 1.0
            prods.append(U[:, a] * U[:, b])          # 每条边贡献 1 个交互项
        P = np.stack(prods, axis=1)                  # (n_total, ne)
        if label_combiner == "sum":                  # 老逻辑:乘积直接相加(留作对照)
            g = P.sum(axis=1).astype(np.float32)
        else:                                         # 新逻辑:乘积过固定随机 MLP(非线性读出)
            h = int(mlp_hidden)
            W1 = rng.standard_normal((ne, h)).astype(np.float32)
            b1 = rng.standard_normal(h).astype(np.float32)
            W2 = rng.standard_normal((h, 1)).astype(np.float32)
            g = (np.tanh(P @ W1 + b1) @ W2)[:, 0].astype(np.float32)
        # === 改动结束 ===
        A_true = A_node[np.ix_(groups, groups)].astype(np.float32)  # lift node graph to feats  # lift node graph to feats
        np.fill_diagonal(A_true, 0.0)
        g = (g - g.mean()) / max(g.std(), 1e-6)
        E = None
    else:  # redundant block SCM
        groups = np.sort(np.arange(n_features) % n_groups)
        if shuffle_features:
            rng.shuffle(groups)
        U = rng.standard_normal((n_total, n_groups))
        X = U[:, groups] + feature_noise * rng.standard_normal((n_total, n_features))
        A_true = (groups[:, None] == groups[None, :]).astype(np.float32)
        np.fill_diagonal(A_true, 0.0)
        U_rel = U[:, np.arange(rel)]
        E = None
        g = A._score(U_rel, dgp, rng)

    g = g + label_noise * rng.standard_normal(n_total)
    thr = np.quantile(g[:n_train], 0.5)
    y = (g > thr).astype(np.int64)

    X = X.astype(np.float32)
    d = A.Data(
        X_train=X[:n_train], y_train=y[:n_train],
        X_test=X[n_train:], y_test=y[n_train:],
        groups=groups, A_true=A_true, dgp=dgp,
    )
    return d, E


def reorder_by_kg(A_obs: np.ndarray) -> np.ndarray:
    """Column ORDER that puts KG-similar features adjacent, so TabPFN's 3-feature token packs
    a coherent (mutually similar) triple instead of a random one. Without this, block-mean
    pooling collapses each token to the centroid of 3 UNRELATED embeddings (≈noise) and the
    token-level M loses the KG structure — the reason B is inert on the factor family.

    Spectral 1-D layout: sort features by their coordinate on the leading eigenvector of the
    symmetrised affinity (the dominant latent direction), a cheap proxy for a KG-community
    ordering. Apply the SAME permutation to X and the KG (permutation-equivariant)."""
    S = 0.5 * (A_obs + A_obs.T)
    _, V = np.linalg.eigh(S)
    return np.argsort(V[:, -1])


def knn_graph(S: np.ndarray, k: int) -> np.ndarray:
    """Sparsify a dense symmetric similarity matrix into a kNN graph: keep each node's top-k
    strongest neighbours, then symmetrise (edge union). Turns the continuous cosine Gram into
    a recognizable SPARSE similarity KG (the form real co-expression / association nets take),
    while preserving edge WEIGHTS so pool_to_groups still sees graded affinity."""
    d = S.shape[0]
    k = int(min(max(k, 1), d - 1))
    A = np.zeros_like(S)
    Sel = S.copy()
    np.fill_diagonal(Sel, -np.inf)                          # never select self as a neighbour
    idx = np.argpartition(-Sel, kth=k - 1, axis=1)[:, :k]   # top-k neighbours per row
    rows = np.repeat(np.arange(d), k)
    A[rows, idx.ravel()] = S[rows, idx.ravel()]
    A = np.maximum(A, A.T)                                   # symmetric union of the kNN edges
    np.fill_diagonal(A, 0.0)
    return A.astype(np.float32)


def build_problem(
    *, dgp, family, frac, seed, n_train, n_test, n_features, n_groups,
    feature_noise, label_noise, rank, reorder=False, n_edges=1, knn=0,
    label_combiner="mlp", mlp_hidden=16,
):
    """Generate data + the OBSERVED KG (optionally corrupted by `frac`) + its quality + the
    pooled bias M. Branches on family because corruption differs for a 0/1 graph vs a
    continuous Gram."""
    d, E = generate_data_b(
        dgp=dgp, family=family, n_train=n_train, n_test=n_test, n_features=n_features,
        n_groups=n_groups, feature_noise=feature_noise, label_noise=label_noise,
        rank=rank, seed=seed, shuffle_features=False, n_edges=n_edges, knn=knn,
        label_combiner=label_combiner, mlp_hidden=mlp_hidden,
    )
    if family in ("factor", "similarity"):
        if frac == 0.0:
            A_obs = d.A_true
        else:  # blend the embeddings toward random, then re-form (and re-sparsify) the Gram
            rng = np.random.default_rng(seed + 999)
            E_obs = (1.0 - frac) * E + frac * rng.standard_normal(E.shape).astype(np.float32)
            E_obs /= np.linalg.norm(E_obs, axis=1, keepdims=True) + 1e-12
            A_obs = (E_obs @ E_obs.T).astype(np.float32)
            np.fill_diagonal(A_obs, 0.0)
            if knn and knn > 0:
                A_obs = knn_graph(A_obs, knn)
        q_val = kg_quality(A_obs, d.A_true)                 # Frobenius cosine of the Grams
    elif family in ("interaction", "epistasis"):
        # Corrupt at the NODE level (rewire the matching), then lift back to features so the
        # block/token alignment is preserved. q_val is the node-graph Frobenius cosine.
        A_node_true = (pool_to_groups(d.A_true) > 0.5).astype(np.float32)
        if frac == 0.0:
            A_node_obs = A_node_true
            A_obs = d.A_true
        else:
            rng = np.random.default_rng(seed + 999)
            A_node_obs = corrupt_graph(A_node_true, frac, rng)
            A_obs = A_node_obs[np.ix_(d.groups, d.groups)].astype(np.float32)
            np.fill_diagonal(A_obs, 0.0)
        q_val = kg_quality(normalize_adj(A_node_obs), normalize_adj(A_node_true))
    else:
        Ahat_true = normalize_adj(d.A_true)
        if frac == 0.0:
            A_obs = d.A_true
        else:
            rng = np.random.default_rng(seed + 999)
            A_obs = corrupt_graph(d.A_true, frac, rng)
        q_val = kg_quality(normalize_adj(A_obs), Ahat_true)

    # Align TabPFN's 3-feature tokens with KG communities (no-op-ish for the already-ordered
    # block SCM; essential for the randomly-ordered factor family). q_val is permutation-
    # invariant (same perm on both Grams), so it is computed above, before reordering.
    if reorder:
        order = reorder_by_kg(A_obs)
        A_obs = A_obs[np.ix_(order, order)]
        d = dataclasses.replace(
            d, X_train=d.X_train[:, order], X_test=d.X_test[:, order], groups=d.groups[order],
        )
    M = group_M(A_obs)
    assert float(M.abs().sum()) > 0, (
        "Pooled KG bias M is all zeros — KG 不会被注入。检查 n_features / n_edges / "
        "FEATURES_PER_GROUP,确保 community 至少跨到 token 层面。"
    )
    return d, M, q_val
# ---------------------------------------------------------------------------
# Monkey-patch: additive KG bias on feature-attention logits (Eq. 6)
# ---------------------------------------------------------------------------

_KGAB: dict = {"M": None, "alpha": 0.0, "last_C": None, "layers": None}
_ORIG_ROW_FWD = v26.AlongRowAttention.forward


def tag_block_layers(clf) -> int:
    """Annotate each feature-attention module with its block (layer) index so the patch
    can restrict the KG bias to a chosen subset of layers. Returns the layer count.

    NB this checkpoint (TabPFNV2p6) has 24 blocks"""
    blocks = clf.models_[0].blocks
    for i, blk in enumerate(blocks):
        blk.per_sample_attention_between_features._kgab_layer = i
    return len(blocks)


def _resolve_alpha(self):
    """The gate for THIS block. `_KGAB["alpha"]` may be:
      * a python float / 0-d tensor  -> one global α shared by every block (legacy);
      * a 1-d tensor of length n_layers -> a SEPARATE α per block, indexed by the block
        index set by tag_block_layers (per-layer learnable gate, §8 extension).
    """
    a = _KGAB["alpha"]
    if torch.is_tensor(a) and a.ndim == 1:
        li = getattr(self, "_kgab_layer", None)
        return a[li] if li is not None else 0.0
    return a


def _patched_row_forward(self, x_BrSE):
    if _KGAB["M"] is None:
        return _ORIG_ROW_FWD(self, x_BrSE)

    # Layer-selective injection: when `layers` is set, only the listed blocks get the KG
    # bias; all others fall back to vanilla feature attention. Requires tag_block_layers.
    layers = _KGAB["layers"]
    if layers is not None and getattr(self, "_kgab_layer", None) not in layers:
        return _ORIG_ROW_FWD(self, x_BrSE)

    alpha = _resolve_alpha(self)
    # Skip only when there is provably no bias to add (keeps α=0 ≡ vanilla TabPFN).
    if isinstance(alpha, (int, float)) and alpha == 0.0:
        return _ORIG_ROW_FWD(self, x_BrSE)

    Br, C, _ = x_BrSE.shape
    _KGAB["last_C"] = C
    q = self.q_projection(x_BrSE).view(Br, C, -1, self.head_dim).permute(0, 2, 1, 3)
    k = self.k_projection(x_BrSE).view(Br, C, -1, self.head_dim).permute(0, 2, 1, 3)
    v = self.v_projection(x_BrSE).view(Br, C, -1, self.head_dim).permute(0, 2, 1, 3)

    # Additive C×C mask: KG block in the top-left, trailing target token gets 0.
    # Built with F.pad so gradients flow to α when it is a trainable tensor.
    M = _KGAB["M"].to(x_BrSE.device)
    g = min(M.shape[0], C)
    core = alpha * M[:g, :g]                      # (g, g), may carry grad via α
    mask = F.pad(core, (0, C - g, 0, C - g)).to(x_BrSE.dtype)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (Br, H, C, D)
    out = torch.nan_to_num(out)  # guard: a large α·M can saturate softmax -> NaN on some seeds
    out = out.permute(0, 2, 1, 3).reshape(Br, C, self.head_dim * self.num_heads)
    return self.out_projection(out)


v26.AlongRowAttention.forward = _patched_row_forward


# ---------------------------------------------------------------------------
# Group-level KG affinity M (pool d×d -> G×G, target padded with zeros)
# ---------------------------------------------------------------------------

def pool_to_groups(Aff: np.ndarray, fpg: int = FEATURES_PER_GROUP) -> np.ndarray:
    """Pool a d×d feature affinity into G×G group affinity (G=⌈d/fpg⌉) by block mean.
    Diagonal zeroed (self-group bias is redundant with the existing self-attention)."""
    d = Aff.shape[0]
    G = math.ceil(d / fpg)
    Ag = np.zeros((G, G), dtype=np.float32)
    for a in range(G):
        ra = slice(a * fpg, min((a + 1) * fpg, d))
        for b in range(G):
            rb = slice(b * fpg, min((b + 1) * fpg, d))
            block = Aff[ra, rb]
            if block.size:
                Ag[a, b] = float(block.mean())
    np.fill_diagonal(Ag, 0.0)
    return Ag


def group_M(A_obs: np.ndarray) -> torch.Tensor:
    """KG affinity used as the bias: raw collapsed adjacency pooled to groups (paper §5
    'raw scores are used for bias'). α controls strength, so M is left unnormalised."""
    return torch.from_numpy(pool_to_groups(A_obs)).float()


# ---------------------------------------------------------------------------
# Eval with the bias applied (clf is fit ONCE, then re-scored under different M, α)
# ---------------------------------------------------------------------------

def score_with_bias(clf, Xte, yte, *, M: torch.Tensor | None, alpha: float,
                    layers=None) -> float:
    _KGAB["M"], _KGAB["alpha"], _KGAB["layers"] = M, alpha, layers
    try:
        return float(clf.score(Xte.astype(np.float32), yte))
    finally:
        _KGAB["M"], _KGAB["alpha"], _KGAB["layers"] = None, 0.0, None


# ---------------------------------------------------------------------------
# TabPFN plumbing (inlined here instead of importing from kg_post_injection)
# ---------------------------------------------------------------------------
# The 'none' preprocessing keeps the raw d features in fixed order so the 3-feature token
# layout (hence M, and the block-index tagging) is well-defined and stable.

def _simple_inference_config() -> dict:
    return {
        "PREPROCESS_TRANSFORMS": [PreprocessorConfig("none")],
        "FINGERPRINT_FEATURE": False,
        "FEATURE_SHIFT_METHOD": None,
        "CLASS_SHIFT_METHOD": None,
        "POLYNOMIAL_FEATURES": "no",
        "OUTLIER_REMOVAL_STD": None,
        "ENABLE_GPU_PREPROCESSING": False,
    }


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_tabpfn(X_train, y_train, *, model_path, device, random_state) -> TabPFNClassifier:
    """Frozen sklearn inference path used for evaluation (fit context once, re-score)."""
    clf = TabPFNClassifier(
        n_estimators=1, model_path=model_path, device=device,
        fit_mode="fit_preprocessors", random_state=random_state,
        ignore_pretraining_limits=True, inference_config=_simple_inference_config(),
    )
    clf.fit(X_train, y_train)
    return clf


def _batched_clf(model_path, device, seed) -> TabPFNClassifier:
    """A TabPFNClassifier set up for the differentiable batched forward, with the same
    'none' preprocessing as the eval path so the 3-feature token layout (hence M) matches."""
    return TabPFNClassifier(
        n_estimators=1, model_path=model_path, device=device, random_state=seed,
        fit_mode="batched", differentiable_input=False,
        ignore_pretraining_limits=True,
        inference_config=_simple_inference_config(),
    )


def _episode_batch(clf, Xtr, ytr, *, seed):
    """Re-preprocess ONE fresh context/query split (episodic). Returns the collated
    batch. A new split every call so the gradient reflects generalisable structure, not
    one tiny fixed query set (paper §8 'average over many resampled (C,Q) splits')."""
    split_fn = partial(train_test_split, test_size=0.3, random_state=seed)
    chunks = get_preprocessed_dataset_chunks(
        clf, Xtr.astype(np.float32), ytr, split_fn, max_data_size=None,
        model_type="classifier", equal_split_size=True,
        data_shuffle_seed=seed, preprocessing_random_state=seed,
    )
    return next(iter(DataLoader(chunks, batch_size=1, collate_fn=meta_dataset_collator)))


def train_alpha(
    Xtr, ytr, M: torch.Tensor, *, model_path, device, seed,
    epochs: int, lr: float, query_frac: float, per_layer: bool = False,
    n_layers: int = 24, l1: float = 0.0,
) -> tuple[object, list[float]]:
    """Train the gate α (zero-init) against in-context NLL via the frozen batched forward,
    RESAMPLING the context/query split every step.

      * per_layer=False -> ONE global scalar α (shared by all blocks).
      * per_layer=True  -> a length-n_layers vector, a SEPARATE α per block; the patch
        indexes it by each block's tag (B._resolve_alpha). `l1`>0 adds λ·Σ|α| so the
        learned profile is sparse / easier to read.

    Zero-init => the bias starts at 0 in every block, so step 0 is exactly vanilla TabPFN.
    Returns (α as float or 1-d cpu tensor, loss history)."""
    clf = _batched_clf(model_path, device, seed)
    if per_layer:
        alpha = torch.nn.Parameter(torch.zeros(n_layers, device=device))
    else:
        alpha = torch.nn.Parameter(torch.zeros((), device=device))
    opt = torch.optim.Adam([alpha], lr=lr)
    M = M.to(device)
    history: list[float] = []

    frozen = False
    for ep in range(epochs):
        batch = _episode_batch(clf, Xtr, ytr, seed=seed + 1000 + ep)  # fresh split/step
        clf.fit_from_preprocessed(batch.X_context, batch.y_context,
                                  batch.cat_indices, batch.configs)
        if not frozen:  # freeze TabPFN weights once + tag block indices for per-layer α
            for prm in clf.models_[0].parameters():
                prm.requires_grad_(False)
            tag_block_layers(clf)
            frozen = True

        _KGAB["M"], _KGAB["alpha"] = M, alpha
        try:
            logits_QBEL = clf.forward(batch.X_query, return_raw_logits=True)
            Q, Bn, E, L = logits_QBEL.shape
            logits = logits_QBEL.permute(1, 2, 3, 0).reshape(Bn * E, L, Q)
            targets = batch.y_query.repeat(Bn * E, 1).to(device)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if per_layer and l1 > 0.0:
                loss = loss + l1 * alpha.abs().sum()
        finally:
            _KGAB["M"], _KGAB["alpha"] = None, 0.0

        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    learned = alpha.detach().cpu() if per_layer else float(alpha.detach())
    return learned, history


def scan_layers_fixed(eval_clf, Xte, yte, M, *, alpha: float, n_layers: int) -> list[float]:
    """DIAGNOSTIC: light up ONE block at a time at a FIXED α, return the per-block test acc
    (length n_layers). No training, no optimiser coupling -> a clean estimate of each
    block's marginal effect. eval_clf must already be tag_block_layers'd."""
    return [score_with_bias(eval_clf, Xte, yte, M=M, alpha=alpha, layers={li})
            for li in range(n_layers)]


# ---------------------------------------------------------------------------
# One condition: dispatch on mode (scan | train | both) -> long-format records
# ---------------------------------------------------------------------------

def run(
    *, dgp, seed, mode, family, rank, reorder, n_train, n_test, n_features, n_groups,
    feature_noise, label_noise, scan_alphas, epochs, lr, query_frac, l1, frac, model_path,
    device, n_edges=1, knn=0, label_combiner="mlp", mlp_hidden=16,
) -> list[dict]:
    set_seed(seed)
    d, M, q_val = build_problem(
        dgp=dgp, family=family, frac=frac, seed=seed, n_train=n_train, n_test=n_test,
        n_features=n_features, n_groups=n_groups, feature_noise=feature_noise,
        label_noise=label_noise, rank=rank, reorder=reorder, n_edges=n_edges, knn=knn,
        label_combiner=label_combiner, mlp_hidden=mlp_hidden,
    )

    eval_clf = fit_tabpfn(d.X_train.astype(np.float32), d.y_train,
                          model_path=model_path, device=device, random_state=seed)
    n_layers = tag_block_layers(eval_clf)
    base = score_with_bias(eval_clf, d.X_test, d.y_test, M=None, alpha=0.0)
    recs: list[dict] = []

    if mode in ("scan", "both"):
        for scan_alpha in scan_alphas:
            all_acc = score_with_bias(eval_clf, d.X_test, d.y_test, M=M, alpha=scan_alpha, layers=None)
            
            accs = scan_layers_fixed(eval_clf, d.X_test, d.y_test, M, alpha=scan_alpha, n_layers=n_layers)
            best = int(np.argmax(accs))
            print(f"  [scan α={scan_alpha}] {dgp:11s} s={seed} base={100*base:.1f} all={100*all_acc:.1f} "
                  f"| best block={best} acc={100*accs[best]:.1f}")
            for li, a in enumerate(accs):
                recs.append(dict(dgp=dgp, seed=seed, kind="scan", layer=li, value=a, n_train=n_train,
                                 base=base, all_acc=all_acc, alpha_val=scan_alpha, frac=frac, quality=q_val))

    if mode in ("train", "both"):
        alpha_vec, hist = train_alpha(
            d.X_train, d.y_train, M, model_path=model_path, device=device, seed=seed,
            epochs=epochs, lr=lr, query_frac=query_frac, per_layer=True,
            n_layers=n_layers, l1=l1,
        )
        acc_tr = score_with_bias(eval_clf, d.X_test, d.y_test, M=M, alpha=alpha_vec, layers=None)
        av = alpha_vec.numpy()
        top = int(np.argmax(av))
        print(f"  [train] {dgp:11s} s={seed} base={100*base:.1f} -> per-layer acc={100*acc_tr:.1f} "
              f"| max α at block {top}={av[top]:+.3f} (loss {hist[0]:.3f}->{hist[-1]:.3f})")
        for li in range(n_layers):
            recs.append(dict(dgp=dgp, seed=seed, kind="train", layer=li, value=float(av[li]), n_train=n_train,
                             base=base, all_acc=acc_tr, alpha_val="train", frac=frac, quality=q_val))
    return recs


# ---------------------------------------------------------------------------
# Plot: per-layer profile (scan acc and/or learned α) with base/all reference
# ---------------------------------------------------------------------------

def plot_profiles(df, save_path, frac, q_val) -> None:
    import matplotlib.pyplot as plt
    scan_alphas = sorted([a for a in set(df.alpha_val) if isinstance(a, (int, float))])
    has_train = "train" in set(df.kind)
    n_rows = len(scan_alphas) + (1 if has_train else 0)
    dgps = [g for g in DGPS if g in set(df.dgp)]
    
    fig, axes = plt.subplots(n_rows, len(dgps), figsize=(4.6 * len(dgps), 3.8 * n_rows),
                             squeeze=False)
                             
    for c, dgp in enumerate(dgps):
        sub_dgp = df[df.dgp == dgp]
        
        # Plot scan alphas
        for r, sa in enumerate(scan_alphas):
            ax = axes[r][c]
            sub = sub_dgp[(sub_dgp.kind == "scan") & (sub_dgp.alpha_val == sa)]
            if len(sub) == 0: continue
            prof = sub.groupby("layer").value.mean().sort_index()
            ax.plot(prof.index, 100 * prof.values, "o-", color="tab:blue", label=f"scan α={sa}")
            
            base = sub.base.mean()
            allv = sub.all_acc.mean()
            ax.axhline(100 * base, ls="--", color="black", alpha=0.6, label=f"base ({100*base:.0f})")
            ax.axhline(100 * allv, ls=":", color="tab:red", alpha=0.7, label=f"all-layers ({100*allv:.0f})")
            
            if has_train:
                sub_tr = sub_dgp[sub_dgp.kind == "train"]
                if len(sub_tr) > 0:
                    train_acc = sub_tr.all_acc.mean()
                    ax.axhline(100 * train_acc, ls="--", color="tab:green", alpha=0.8, label=f"train ({100*train_acc:.0f})")
                    
            ax.legend(fontsize=8)
            ax.set_title(f"Scan α={sa}: {dgp}")
            if r == len(scan_alphas) - 1 and not has_train:
                ax.set_xlabel("block index (0-23)")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("test acc (%)")
            
        # Plot train alpha bar chart at the bottom
        if has_train:
            r = len(scan_alphas)
            ax = axes[r][c]
            sub = sub_dgp[sub_dgp.kind == "train"]
            if len(sub) == 0: continue
            prof = sub.groupby("layer").value.mean().sort_index()
            ax.bar(prof.index, prof.values, color="tab:green", alpha=0.8)
            ax.axhline(0.0, color="black", lw=0.8)
            ax.set_title(f"Learned α: {dgp}")
            ax.set_xlabel("block index (0-23)")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("learned α")

    n_tr = df.n_train.iloc[0] if "n_train" in df.columns else "unknown"
    fig.suptitle(f"KGAB per-layer (n_train={n_tr}, frac={frac}, quality={q_val:.2f})", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {save_path}")


def plot_nsweep(df, save_path) -> None:
    import matplotlib.pyplot as plt
    has_train = "train" in set(df.kind)
    if not has_train:
        print("No train records found for nsweep plot.")
        return
        
    dgps = [g for g in DGPS if g in set(df.dgp)]
    fig, axes = plt.subplots(1, len(dgps), figsize=(4.6 * len(dgps), 4.2), squeeze=False)
    
    for c, dgp in enumerate(dgps):
        ax = axes[0][c]
        sub = df[(df.kind == "train") & (df.dgp == dgp)]
        if len(sub) == 0: continue
        
        m_base = sub.groupby("n_train").base.mean().sort_index()
        ax.plot(m_base.index, 100 * m_base.values, "--", color="black", marker="o", alpha=0.6, label="base")
        
        colors = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:orange"]
        for i, (frac, sub_frac) in enumerate(sub.groupby("frac")):
            c_color = colors[i % len(colors)]
            m_train = sub_frac.groupby("n_train").all_acc.mean().sort_index()
            q_mean = sub_frac.quality.mean()
            ax.plot(m_train.index, 100 * m_train.values, "-", color=c_color, marker="o", alpha=0.8, 
                    label=f"train (frac={frac}, q={q_mean:.2f})")
                    
        ax.set_title(dgp)
        ax.set_xlabel("n_train")
        ax.set_ylabel("test acc (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
    fig.suptitle("KGAB: Learned Alpha Performance vs n_train", fontsize=15, y=1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {save_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=["profiles", "nsweep", "both"], default="profiles")
    p.add_argument("--mode", choices=["scan", "train", "both"], default="scan")
    p.add_argument("--model-path", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dgps", nargs="+", default=list(DGPS_B))
    p.add_argument("--family",
                   choices=["redundant", "factor", "similarity", "interaction", "epistasis"],
                   default="interaction",
                   help="simulation family / KG TYPE: "
                        "'similarity'(=='factor') = realistic similarity/association KG "
                        "(low-rank cosine Gram; A/D turf, B≈base by design); "
                        "'epistasis'(=='interaction') = synthetic-lethal/synergy KG "
                        "(pairwise products, info not in the data — B's home turf); "
                        "'redundant' = legacy 0/1 block SCM (control)")
    p.add_argument("--rank", type=int, default=5,
                   help="latent rank r for family='similarity'/'factor'")
    p.add_argument("--knn", type=int, default=0,
                   help="family='similarity'/'factor': keep only each feature's top-k cosine "
                        "neighbours (0 = dense weighted Gram). >0 => a realistic SPARSE "
                        "similarity graph (e.g. gene co-expression).")
    p.add_argument("--n-edges", type=int, default=1,
                   help="number of interacting node pairs for family='epistasis'/'interaction' "
                        "(remaining nodes are distractors)")
    p.add_argument("--label-combiner", choices=["mlp", "sum"], default="mlp",
                   help="interaction-family label readout: mlp=nonlinear (default), "
                        "sum=legacy linear sum (control)")
    p.add_argument("--mlp-hidden", type=int, default=16,
                   help="hidden width of the label MLP combiner")
    p.add_argument("--reorder", action="store_true",
                   help="spectrally reorder columns so each 3-feature token is a KG-coherent "
                        "triple (essential for family='factor')")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--n-train", type=int, nargs="+", default=[40],
                   help="n_train values for the profiles experiment")
    p.add_argument("--n-trains", type=int, nargs="+", default=[40, 80, 160, 320],
                   help="n grid for the n-sweep")
    p.add_argument("--n-test", type=int, default=300)
    # Intuitive redundant data: 6 communities x 6 near-identical features = 2 tokens each.
    p.add_argument("--n-features", type=int, default=120)
    p.add_argument("--n-groups", type=int, default=8)
    p.add_argument("--feature-noise", type=float, default=2.0,
                   help="lower => triples within a community are more identical")
    p.add_argument("--label-noise", type=float, default=0.3)
    p.add_argument("--scan-alphas", type=float, nargs="+", default=[2],
                   help="fixed α values used for the single-block diagnostic scan")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.0],
                   help="edge-rewire fractions for graph corruption")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--query-frac", type=float, default=0.3)
    p.add_argument("--l1", type=float, default=0.0,
                   help="L1 penalty on the per-layer α vector (sparser profile)")
    p.add_argument("--out", default="kg_b_kgab_train")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd
    
    common = dict(
        mode=args.mode, family=args.family, rank=args.rank, reorder=args.reorder,
        n_edges=args.n_edges, knn=args.knn,
        label_combiner=args.label_combiner, mlp_hidden=args.mlp_hidden,
        n_test=args.n_test, n_features=args.n_features, n_groups=args.n_groups,
        feature_noise=args.feature_noise, label_noise=args.label_noise,
        scan_alphas=args.scan_alphas, epochs=args.epochs, lr=args.lr,
        query_frac=args.query_frac, l1=args.l1, model_path=args.model_path, device=args.device,
    )
    
    if args.experiment in ("profiles", "both"):
        print(f"\n=== Profiles Experiment (n_trains={args.n_train}, scan alphas={args.scan_alphas}, fracs={args.fracs}) ===")
        for n_train in args.n_train:
            for frac in args.fracs:
                print(f"\n--- Running n_train={n_train}, frac={frac} ---")
                recs: list[dict] = []
                for dgp in args.dgps:
                    for seed in args.seeds:
                        recs.extend(run(
                            dgp=dgp, seed=seed, n_train=n_train, frac=frac, **common
                        ))
                df = pd.DataFrame(recs)
                if len(df) > 0:
                    q_val = df.quality.mean()
                    plot_name = f"{args.out}_ntrain{n_train}_frac{frac}_layers.png"
                    plot_profiles(df, plot_name, frac, q_val)
                    print(f"Results for n_train={n_train}, frac={frac} -> {plot_name}")
                
    if args.experiment in ("nsweep", "both"):
        print(f"\n=== N-Sweep Experiment (n_trains={args.n_trains}, fracs={args.fracs}) ===")
        recs_n: list[dict] = []
        # In nsweep, if mode == "scan", it will still scan alphas and run training.
        # This can be slow, but we rely on the user to specify args.mode="train" if they want it fast.
        for dgp in args.dgps:
            for n in args.n_trains:
                for frac in args.fracs:
                    for seed in args.seeds:
                        recs_n.extend(run(
                            dgp=dgp, seed=seed, n_train=n, frac=frac, **common
                        ))
        df_n = pd.DataFrame(recs_n)
        if len(df_n) > 0:
            plot_nsweep(df_n, f"{args.out}_nsweep.png")


if __name__ == "__main__":
    main()
