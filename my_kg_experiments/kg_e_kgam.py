"""KGAM — coverage-gated KG ATTENTION MIXTURE injection into frozen TabPFN.

Faithful implementation of the method in wileyNJD-Doc.tex:

  * The KG prior is mixed into the feature-axis attention as a CONVEX COMBINATION of
    row-stochastic distributions (Eq. mix/mixy). 
    Two channels with different routes/directionality:
      - feature-feature edges A^(r): mixed into the feature-token rows (bidirectional);
      - feature-label relevance b^(r): mixed into the TARGET-token readout row only
        (unidirectional; the target COLUMN is never touched).
  * Pipeline: map columns -> KG entities (coverage o); cluster
    the OBSERVED KG into communities (b treated as edges to a virtual target node);
    order columns community-contiguously; pool d×d / d priors to token resolution by a
    MASKED MEAN over covered pairs; per-token coverage c_a and per-relation support s_a;
    masked row-softmax with a LEARNABLE temperature tau_r; per-block gates lambda^(l)_r,
    lambda^(l)_{y,r} (zero-init, projected onto {λ>=0, Σ_r λ_r <= 1} after each step;
    --gate-mode one TIES lam_y = lam into a single shared gate set — routing asymmetry
    kept, strengths shared);
    in-context NLL through the frozen forward with a FRESH context/query split AND a
    FRESH community-preserving permutation each episode; + L1 (+ optional TV) penalty.
  * Inference: E ensemble members, each under its own community-preserving permutation
    with its own pooled prior; gates shared across members; member probs averaged.
  * Safety: all gates zero => P' = P in every block => bit-exact vanilla TabPFN.

THE KG INTERFACE is coverage + one matrix pair PER RELATION:
  o       (d,)  bool  which columns are MAPPED to a KG entity;
  b_list  R arrays (d,)   b^(r)  feature->label relevance channel of relation r;
  A_list  R arrays (d,d)  A^(r)  feature-feature channel of relation r.
EVERY relation carries BOTH channels (either may be empty; its gates then multiply a
zero base and stay inert) — nothing is hard-coded to a dedicated 'label relation'.
o is NOT part of the KG itself: in real use it comes from matching the observed
data's feature headers against the KG's triples (a separate front-end component,
deferred); the simulations emit o directly. Keeping o separate from the channels is
what partially distinguishes 'mapped but no known relation' (o=1, all channels zero
at j) from 'not in the KG at all' (o=0).
每个关系r同时携带feature-feature矩阵A^(r)和feature-label向量b^(r)；o由表头↔KG实体
配对得到（该前端组件后续再加，模拟中直接给出），用于区分"没关系"和"没观测到"。

TWO SIMULATION SCENARIOS (--dgp).

--dgp label  EXTREME CASE (default): X ~ N(0, I_d) i.i.d., relevant set R at random
  positions (|R| = k_rel), y = sign(sum_{j in R} ±x_j + noise). The KG observes only
  PART of the x-y relations: b^(0) = 1 on a random subset K ⊂ R (|K| = n_known).
  Nothing else — one relation whose feature-feature side is empty.

--dgp mixed  FEATURE-FEATURE + FEATURE-LABEL, three KG relations (R = 3):
  * relation 0 'correlation': n_corr CORRELATED PAIRS (j,j') at random positions:
    x_j = u + ε, x_j' = u + ε' (noisy copies of a latent u); the label's linear part
    uses u — the relation's value is denoising (average the copies).
  * relation 1 'interaction': n_int INTERACTION PAIRS (a,b) of INDEPENDENT columns:
    label += beta_int * x_a*x_b. Cross-column covariance is 0, so the pairing is
    invisible to data correlations — genuinely non-data info.
  * relation 2 'relevance': b^(2) = 1 on relevant columns (label channel as in
    --dgp label).
  The KG observes only a kg_frac fraction of each channel's entries (partial
  knowledge). Feature-feature relations are pooled with a masked MAX by default
  (tex §pooling max-variant: mean would dilute a single strong edge by up to 1/p^2).

Baselines per condition (all member-averaged over --members unless noted):
  base      SAME pipeline ('none' preprocessing, member averaging) under UNIFORM column
            permutations, NO KG — the honest baseline (holds everything but the KG fixed)
  vanilla   off-the-shelf TabPFN (default preprocessing, n_estimators=--members); kept
            only as a reference point — it differs from `base` by preprocessing, not KG
  ours0     community-preserving perms, gates at zero (isolates the perm-sampler effect;
            also the safety arm: patch off == gates zero by construction)
  ours      community-preserving perms + trained gates (the method)
  oracle    TabPFN on the TRUE relevant columns (upper reference)

FPG=1 ARMS (--fpg1, on by default): force ONE column per token by monkey-patching the
encoder's feature grouping (each column becomes the in-distribution [x, 0, 0] padding
pattern the checkpoint saw in training — the same trick as
my_kg_experiments/fpg_token_compare.py). Token == column, so the pooling step
(Eq. pool) becomes the IDENTITY: the d×d / d prior is injected EXACTLY, coverage
c_a = o_j ∈ {0,1}, and NO community clustering / community-preserving permutation is
needed — these arms run under plain uniform permutations end to end.
  base_fpg1     fpg=1, uniform perms, NO KG (isolates the cost of losing in-token
                encoder mixing — known to hurt interaction-style signal)
  vanilla_fpg1  fpg=1 + DEFAULT preprocessing, NO KG. Works because the grouping
                patch sits INSIDE the model forward, i.e. after the whole
                preprocessing pipeline (transforms -> fingerprint -> feature
                shuffle): every post-preprocessing column gets its own token. The
                KG arm still needs 'none' preprocessing: default transforms CHANGE
                the column set (append_original copies, SVD components, fingerprint
                +1), so prior rows can no longer be aligned to tokens without
                tracing the pipeline's column mapping (deferred front-end work).
  ours_fpg1     fpg=1, uniform perms, exact prior + gates trained under fpg=1
Reads on the tex story: (ours - base) vs (ours_fpg1 - base_fpg1) separates
'KG value at token resolution' from 'KG value at column resolution', and
base_fpg1 - base prices the tokenization itself (the co-tokenize ablation).
fpg=1 时 token 即列，池化无损、无需图聚类；代价是丢掉 token 内 encoder 混合。

Edit parse_args defaults and run directly:  python kg_e_kgam.py
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import tabpfn.architectures.tabpfn_v2_5 as v25
import tabpfn.architectures.tabpfn_v2_6 as v26
from tabpfn.classifier import TabPFNClassifier
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.finetuning.data_util import (
    get_preprocessed_dataset_chunks,
    meta_dataset_collator,
)

FPG = 3  # features per token in the released checkpoint (tex: p=3)


# ---------------------------------------------------------------------------
# fpg=1: force one column per token (ported from my_kg_experiments/fpg_token_compare)
# ---------------------------------------------------------------------------

def _one_feature_per_group(x_RiBC: torch.Tensor, num_features_per_group: int):
    """[Ri, B, C] -> [Ri, B*C, fpg]: each column gets its own group, remaining slots
    zero-padded — every group is the [x, 0, 0] padding pattern the checkpoint was
    trained on, so the encoder weights stay in-distribution. Same (tensor, G) return
    convention as the native _pad_and_reshape_feature_groups, with G = C."""
    num_rows, B, Cc = x_RiBC.shape
    x = x_RiBC.reshape(num_rows, B * Cc, 1)
    x = F.pad(x, (0, num_features_per_group - 1), value=0.0)
    return x, Cc


@contextlib.contextmanager
def force_single_feature_tokens():
    """Temporarily patch the v2.5/v2.6 feature-grouping fn; yields a call counter so
    callers can assert the patch actually ran (fit_mode='fit_preprocessors' defers the
    forward to predict time)."""
    counter = {"n_calls": 0}

    def patched(x_RiBC, num_features_per_group):
        counter["n_calls"] += 1
        return _one_feature_per_group(x_RiBC, num_features_per_group)

    orig26 = v26._pad_and_reshape_feature_groups
    orig25 = v25._pad_and_reshape_feature_groups
    v26._pad_and_reshape_feature_groups = patched
    v25._pad_and_reshape_feature_groups = patched
    try:
        yield counter
    finally:
        v26._pad_and_reshape_feature_groups = orig26
        v25._pad_and_reshape_feature_groups = orig25


# ---------------------------------------------------------------------------
# Extreme-case DGP: sparse relevant set at random positions, partial-coverage KG
# ---------------------------------------------------------------------------

def make_problem(*, n_train, n_test, n_features, k_rel, n_known, n_false,
                 label_noise, seed, nl_frac=0.4, hetero_w=True, link="logit"):
    """i.i.d. features, y from k_rel random columns; the KG knows only n_known of them
    (b=1, o=1), optionally plus n_false WRONG columns (KG claims irrelevant ones are
    relevant — robustness knob, default 0 实际上无关，但是KG说有关). Column positions are random, so the data is
    'pre-shuffled'; the method must earn it via the community-contiguous ordering.

    NONLINEAR ADDITIVE single-index (still purely feature->label, NO interactions —
    those live in --dgp mixed). The score is g = Σ_j w_j φ_j(x_j) + noise, where an
    nl_frac fraction of the relevant columns use a NONLINEAR per-feature response φ_j
    drawn from {x²−1 (centered even → non-monotone), tanh(1.5x) (saturating monotone)}
    and the rest stay linear. Weights w_j are HETEROGENEOUS in magnitude (hetero_w):
    columns differ in importance, so WHICH of them the KG happens to know matters — but
    the KG only ever sees the 0/1 relevance edge (b=1), never the strength, matching a
    real KG that carries relation type, not weight.  link='logit' draws soft labels
    y ~ Bernoulli(σ((g−med)/std)) (Bayes error > 0, more realistic); link='hard' keeps
    the deterministic median threshold (auto-balanced classes).
    非线性可加模型:每个相关列的响应形状可以非线性(x²−1 / tanh),权重幅度异质,但
    KG 只知道连线与否(b=1),不知道强度;logit 软标签让 Bayes 误差>0。"""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    rel = rng.choice(n_features, size=k_rel, replace=False)

    # heterogeneous signed weights (magnitude varies; KG never sees this, only the edge)
    mag = (rng.uniform(0.5, 1.5, size=k_rel) if hetero_w
           else np.ones(k_rel)).astype(np.float32)
    w = mag * rng.choice([-1.0, 1.0], size=k_rel).astype(np.float32)

    # per-feature response shape: an nl_frac fraction is nonlinear (quad/tanh alternate)
    n_nl = int(round(nl_frac * k_rel))
    nl_kinds = [("quad" if i % 2 == 0 else "tanh") for i in range(n_nl)]
    shapes = np.array(nl_kinds + ["lin"] * (k_rel - n_nl))
    rng.shuffle(shapes)

    def phi(kind, x):
        if kind == "quad":
            return x * x - 1.0            # centered even → non-monotone boundary
        if kind == "tanh":
            return np.tanh(1.5 * x)       # saturating monotone
        return x

    g = np.zeros(n, dtype=np.float32)
    for t, j in enumerate(rel):
        g += w[t] * phi(shapes[t], X[:, j])
    g += label_noise * rng.standard_normal(n).astype(np.float32)

    if link == "logit":
        mu = np.median(g[:n_train]); sd = np.std(g[:n_train]) + 1e-8
        p = 1.0 / (1.0 + np.exp(-(g - mu) / sd))
        y = (rng.random(n) < p).astype(np.int64)
    else:
        y = (g > np.quantile(g[:n_train], 0.5)).astype(np.int64)

    known = rng.choice(rel, size=n_known, replace=False)
    o = np.zeros(n_features, dtype=bool)
    b = np.zeros(n_features, dtype=np.float32)
    o[known] = True
    b[known] = 1.0
    if n_false > 0:
        pool = np.setdiff1d(np.arange(n_features), rel)
        false = rng.choice(pool, size=n_false, replace=False)
        o[false] = True
        b[false] = 1.0
        known = np.concatenate([known, false])

    # single relation (r=0): label channel b^(0) only, its feature-feature block empty
    return dict(
        X_train=X[:n_train], y_train=y[:n_train],
        X_test=X[n_train:], y_test=y[n_train:],
        rel=np.sort(rel), known=np.sort(known),
        o=o, b_list=[b],
        A_list=[np.zeros((n_features, n_features), dtype=np.float32)],
    )


def make_problem_mixed(*, n_train, n_test, n_features, n_corr, n_int, pair_noise,
                       beta_int, kg_frac, label_noise, seed):
    """Scenario 2: correlated pairs (linear signal via a shared
    latent) + interaction pairs (product signal, invisible to correlations) + label
    relevance, each channel observed only at a kg_frac fraction. Everything else is an
    i.i.d. distractor column; all positions random."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    d = n_features
    pos = rng.choice(d, size=2 * (n_corr + n_int), replace=False)
    corr_pairs = [(int(pos[2 * i]), int(pos[2 * i + 1])) for i in range(n_corr)]
    int_pairs = [(int(pos[2 * n_corr + 2 * j]), int(pos[2 * n_corr + 2 * j + 1]))
                 for j in range(n_int)]

    X = rng.standard_normal((n, d)).astype(np.float32)
    g = np.zeros(n, dtype=np.float32)
    for j, jp in corr_pairs:
        u = rng.standard_normal(n).astype(np.float32)
        X[:, j] = u + pair_noise * rng.standard_normal(n).astype(np.float32)
        X[:, jp] = u + pair_noise * rng.standard_normal(n).astype(np.float32)
        g += float(rng.choice([-1.0, 1.0])) * u
    for a, bc in int_pairs:
        g += beta_int * X[:, a] * X[:, bc]
    g += label_noise * rng.standard_normal(n).astype(np.float32)
    y = (g > np.quantile(g[:n_train], 0.5)).astype(np.int64)

    # OBSERVED KG: a kg_frac subset of each channel, sampled independently.
    def observe(items):
        k = int(round(kg_frac * len(items)))
        if k == 0:
            return []
        idx = rng.choice(len(items), size=k, replace=False)
        return [items[i] for i in idx]

    rel_cols = np.array(sorted({c for p in corr_pairs + int_pairs for c in p}))
    obs_corr = observe(corr_pairs)
    obs_int = observe(int_pairs)
    obs_b = observe(list(rel_cols))

    # observed KG, one (A^(r), b^(r)) pair per relation: 0 = correlation (ff only),
    # 1 = interaction (ff only), 2 = relevance (label only). 真实KG里同一个关系可以
    # 两个通道都非空；这里数据生成没造这种关系，但加权机制按逐关系双通道处理。
    A_corr = np.zeros((d, d), dtype=np.float32)
    A_int = np.zeros((d, d), dtype=np.float32)
    b_rel = np.zeros(d, dtype=np.float32)
    o = np.zeros(d, dtype=bool)
    for j, jp in obs_corr:
        A_corr[j, jp] = A_corr[jp, j] = 1.0
        o[[j, jp]] = True
    for a, bc in obs_int:
        A_int[a, bc] = A_int[bc, a] = 1.0
        o[[a, bc]] = True
    for c in obs_b:
        o[c] = True
        b_rel[c] = 1.0

    zeros_b = np.zeros(d, dtype=np.float32)
    return dict(
        X_train=X[:n_train], y_train=y[:n_train],
        X_test=X[n_train:], y_test=y[n_train:],
        rel=rel_cols, known=np.nonzero(o)[0],
        o=o,
        b_list=[zeros_b, zeros_b.copy(), b_rel],
        A_list=[A_corr, A_int, np.zeros((d, d), dtype=np.float32)],
        # A_list[r]为关系r下feature与feature之间的关系矩阵
        # b_list[r]为关系r下feature与target之间的关系向量（与A_list逐关系对齐）
        # o为feature是否被映射到KG实体（由表头↔KG配对得到，模拟中直接给出；
        #   o=1且各通道全零 = 有节点但没关系，o=0 = 不在KG里）
        # known为观测到的feature在KG中的位置
        # rel为所有有关系的feature在KG中的真实位置
    )


# ---------------------------------------------------------------------------
# KG -> communities -> community-preserving permutations (tex §pooling, §ensemble)
# ---------------------------------------------------------------------------

def kg_communities(o, b_list, A_list):
    """Partition columns into communities from the OBSERVED KG only (no leakage):
    union-find over the covered columns, with feature-feature edges A^(r)>0 as edges and
    b^(r)>0 as edges to a VIRTUAL TARGET node (so all label-relevant columns form one
    community even without feature-feature edges). Uncovered columns are one extra
    block (tex: 'uncovered columns form one extra block').
    o为存在于知识图谱的特征的indicator
    b_list[r]为关系r下feature与target之间的关系强度
    A_list[r]为关系r下观测到的feature与feature之间的关系矩阵"""

    d = len(o)
    # d+1个节点，最后一个节点为虚拟目标节点
    parent = list(range(d + 1))  # node d = virtual target

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    has_ff_edge = np.zeros(d, dtype=bool)
    for A in A_list:
        ii, jj = np.nonzero(A > 0)
        has_ff_edge[ii] = True
        has_ff_edge[jj] = True
        for i, j in zip(ii.tolist(), jj.tolist()):
            union(i, j)
    # Only b-ONLY columns cluster via the virtual target; columns already in a
    # feature-feature community keep their finer community (otherwise every covered
    # column would collapse into one blob through the target node and the
    # community-preserving permutation would scatter e.g. pair partners).
    b_any = np.zeros(d, dtype=bool)
    for b in b_list:
        b_any |= b > 0
    for j in np.nonzero(b_any)[0].tolist():
        if not has_ff_edge[j]:
            union(j, d)

    comms: dict[int, list[int]] = {}
    uncovered = []
    for j in range(d):
        if o[j]:
            comms.setdefault(find(j), []).append(j)
        else:
            uncovered.append(j)
    blocks = [np.array(v) for v in comms.values()]
    if uncovered:
        blocks.append(np.array(uncovered))
    return blocks


def community_perm(blocks, rng):
    """Hierarchical permutation: random order of the community blocks composed with an
    independent random order within each block (tex §ensemble). Returns a length-d
    column permutation."""
    order = rng.permutation(len(blocks))
    return np.concatenate([rng.permutation(blocks[i]) for i in order])


# ---------------------------------------------------------------------------
# Pooling to token resolution (masked mean, Eq. pool) + coverage + supports
# ---------------------------------------------------------------------------

def pool_prior(o, b, A, perm, pool="mean", fpg=FPG):
    """Pool the (permuted) d-column prior to token resolution G = ceil(d/fpg).
    With fpg=1 this is the IDENTITY (token == column): m_tok = b, M_tok = A, c_tok = o.
    `pool` governs BOTH channels the same way (mean = masked average over covered
    columns/pairs; max = 'token is relevant/related if ANY covered column is'): the
    dilution argument is the same for both — a token mixing one b=1 column with two
    covered b=0 columns mean-pools to 1/3 but max-pools to 1. On the label DGP the
    two are identical for b (o=1 exactly where b=1, so every covered entry is 1).
    Returns numpy pieces (constants w.r.t. the learnable temperatures):
      m_tok (G,)  masked mean/max of b over covered columns per token (label channel),
      M_tok (G,G) masked mean/max (tex §pooling max-variant for edge-sparse
                  relations) of A over covered pairs, zero diag (or None),
      c_tok (G,)  per-token coverage c_a = |covered ∩ S_a| / |S_a| (real columns only).
    """
    d = len(perm)
    G = math.ceil(d / fpg)
    pad = G * fpg - d
    op = o[perm].astype(np.float64)
    bp = b[perm].astype(np.float64)
    real = np.ones(d)
    if pad:
        op = np.pad(op, (0, pad))
        bp = np.pad(bp, (0, pad))
        real = np.pad(real, (0, pad))
    op_t = op.reshape(G, fpg)
    cnt = op_t.sum(1)                                   # covered columns per token
    size = real.reshape(G, fpg).sum(1)                  # real columns per token (|S_a|)
    c_tok = cnt / np.maximum(size, 1.0)
    if pool == "max":
        m_tok = (bp * op).reshape(G, fpg).max(1)   # b >= 0, masked entries are 0
    else:
        m_tok = (bp * op).reshape(G, fpg).sum(1) / np.maximum(cnt, 1.0)

    M_tok = None
    if A is not None:
        Ap = np.zeros((G * fpg, G * fpg))
        Ap[:d, :d] = A[np.ix_(perm, perm)]
        O2 = np.outer(op, op)
        blocks_ = (Ap * O2).reshape(G, fpg, G, fpg)
        if pool == "max":  # 'two tokens are related if ANY strong edge joins them'
            M_tok = blocks_.max(axis=(1, 3))  # A >= 0, masked entries are 0
        else:
            num = blocks_.sum(axis=(1, 3))
            den = O2.reshape(G, fpg, G, fpg).sum(axis=(1, 3))
            M_tok = num / np.maximum(den, 1.0)
        np.fill_diagonal(M_tok, 0.0)
    return m_tok, M_tok, c_tok


def _masked_row_softmax(M, tau):
    """Row-softmax over the SUPPORT only (Eq. rowsoftmax); rows with empty support stay
    all-zero (the support switch s_a kills them in the mixture anyway). Differentiable
    w.r.t. tau.

    -1e9 instead of -inf: an empty-support row would be all -inf -> softmax NaN. The
    final mask zeroes those rows and torch.where's select-style backward drops their
    NaN grads, so -inf is not an actual bug — but with -1e9 the NaN is never created
    (empty rows softmax to uniform, then get masked). Bit-identical on non-empty rows:
    support scores are >= 0, so exp(-1e9 - max) underflows to exactly 0."""
    sup = M > 0
    has = sup.any(dim=-1, keepdim=True)
    scores = torch.where(sup, M / tau, torch.full_like(M, -1e9))
    P = torch.softmax(scores, dim=-1)
    return torch.where(has & sup, P, torch.zeros_like(P))


def build_relation_prior(m_tok, M_tok, c_tok, tau, tau_y, C, device, dtype=torch.float32):
    """Assemble one relation's token-level prior for the patched attention:
      Pi     (C,C) row-stochastic-on-support prior matrix: feature rows = normalized
             M' (target column zero-padded), last row = normalized label prior m';
      base_f (C,)  c_a * s_a on feature rows (row-wise mixing weight, before lambda);
      base_t (C,)  c_y * s_y on the target row only.
    Differentiable w.r.t. tau / tau_y (used during gate training)."""
    G = C - 1
    c = torch.as_tensor(c_tok, device=device, dtype=dtype)
    Pi = torch.zeros(C, C, device=device, dtype=dtype)
    base_f = torch.zeros(C, device=device, dtype=dtype)
    base_t = torch.zeros(C, device=device, dtype=dtype)

    if M_tok is not None:
        M = torch.as_tensor(M_tok, device=device, dtype=dtype)
        Mp = _masked_row_softmax(M, tau)
        Pi = Pi.clone()
        Pi[:G, :G] = Mp                                   # target column stays zero
        s_f = (M > 0).any(dim=1).to(dtype)                # support switch s^(r)_a
        base_f = torch.zeros(C, device=device, dtype=dtype)
        base_f[:G] = c * s_f

    m = torch.as_tensor(m_tok, device=device, dtype=dtype)
    supy = m > 0
    if bool(supy.any()):
        # -1e9 not -inf, same as _masked_row_softmax (the supy.any() guard already
        # rules out the all-masked NaN case here; this is just the consistent idiom)
        scores = torch.where(supy, m / tau_y, torch.full_like(m, -1e9))
        mp = torch.softmax(scores, dim=-1)
        mp = torch.where(supy, mp, torch.zeros_like(mp))
        Pi = Pi.clone()
        Pi[C - 1, :G] = mp                                # readout row prior (Eq. mixy)
        base_t[C - 1] = 1.0                               # c_y = 1 (target mapped), s_y = 1
    return Pi, base_f, base_t


def build_priors(kg, perm, C, tau, tau_y, device, pool_ff="max", fpg=FPG):
    """Token-level priors for ALL relations under one permutation. EVERY relation r
    carries both channels of its matrix pair: the feature-feature block A^(r) (feature
    rows of Pi) and the feature->label column b^(r) (target readout row of Pi); both
    channels are pooled with the SAME pool_ff method (mean/max — see pool_prior). An
    empty channel leaves its rows zero, so its gate multiplies a zero base and is
    inert. Differentiable w.r.t. tau / tau_y."""
    priors = []
    for r, (b_r, A_r) in enumerate(zip(kg["b_list"], kg["A_list"])):
        A = A_r if A_r.any() else None  # skip pooling an all-zero ff channel
        priors.append(build_relation_prior(
            *pool_prior(kg["o"], b_r, A, perm, pool=pool_ff, fpg=fpg),
            tau[r], tau_y[r], C, device))
    return priors


# ---------------------------------------------------------------------------
# Monkey-patch: gated attention MIXTURE on the feature axis (Eq. mix / mixy)
# ---------------------------------------------------------------------------

_KGAM: dict = {"on": False, "priors": None, "lam": None, "lam_y": None, "C": None}
_ORIG_ROW_FWD = v26.AlongRowAttention.forward


def tag_block_layers(clf) -> int:
    blocks = clf.models_[0].blocks
    for i, blk in enumerate(blocks):
        blk.per_sample_attention_between_features._kgam_layer = i
    return len(blocks)


def _patched_row_forward(self, x_BrSE):
    if not _KGAM["on"]:
        return _ORIG_ROW_FWD(self, x_BrSE)
    li = getattr(self, "_kgam_layer", None)
    Br, C, _ = x_BrSE.shape
    if li is None or C != _KGAM["C"]:
        return _ORIG_ROW_FWD(self, x_BrSE)

    H, D = self.num_heads, self.head_dim
    q = self.q_projection(x_BrSE).view(Br, C, H, D).permute(0, 2, 1, 3)
    k = self.k_projection(x_BrSE).view(Br, C, H, D).permute(0, 2, 1, 3)
    v = self.v_projection(x_BrSE).view(Br, C, H, D).permute(0, 2, 1, 3)
    P = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(D), dim=-1)  # (Br,H,C,C)

    # Convex mixture, gates shared across heads (tex §inject). w_a = Σ_r λ_r c_a s_a on
    # feature rows / λ_{y,r} c_y s_y on the target row; feasibility of λ (projection)
    # keeps every row of P' a probability distribution.
    w = None
    prior = None
    for r, (Pi, base_f, base_t) in enumerate(_KGAM["priors"]):
        wr = _KGAM["lam"][li, r] * base_f + _KGAM["lam_y"][li, r] * base_t  # (C,)
        w = wr if w is None else w + wr
        pr = wr.unsqueeze(-1) * Pi
        prior = pr if prior is None else prior + pr

    dt = P.dtype
    Pp = (1.0 - w).view(1, 1, C, 1).to(dt) * P + prior.view(1, 1, C, C).to(dt)
    out = (Pp @ v).permute(0, 2, 1, 3).reshape(Br, C, H * D)
    return self.out_projection(out)


v26.AlongRowAttention.forward = _patched_row_forward


class _kgam_ctx:
    """Context manager: activate the mixture with the given priors/gates, always reset."""

    def __init__(self, priors, lam, lam_y, C):
        self.state = {"on": True, "priors": priors, "lam": lam, "lam_y": lam_y, "C": C}

    def __enter__(self):
        _KGAM.update(self.state)

    def __exit__(self, *exc):
        _KGAM.update({"on": False, "priors": None, "lam": None, "lam_y": None, "C": None})


# ---------------------------------------------------------------------------
# TabPFN plumbing ('none' preprocessing => the column order we set is the token order)
# ---------------------------------------------------------------------------

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


def fit_member(Xtr, ytr, *, model_path, device, seed) -> TabPFNClassifier:
    clf = TabPFNClassifier(
        n_estimators=1, model_path=model_path, device=device,
        fit_mode="fit_preprocessors", random_state=seed,
        ignore_pretraining_limits=True, inference_config=_simple_inference_config(),
    )
    clf.fit(Xtr.astype(np.float32), ytr)
    return clf


# 不在内部做集成，所以n_estimator=1，_simple_inference_config()关掉了TabPFN默认的各种数据预处理
def _batched_clf(model_path, device, seed) -> TabPFNClassifier:
    return TabPFNClassifier(
        n_estimators=1, model_path=model_path, device=device, random_state=seed,
        fit_mode="batched", differentiable_input=False,
        ignore_pretraining_limits=True, inference_config=_simple_inference_config(),
    )


def _episode_batch(clf, Xtr, ytr, *, query_frac, seed):
    split_fn = partial(train_test_split, test_size=query_frac, random_state=seed)
    chunks = get_preprocessed_dataset_chunks(
        clf, Xtr.astype(np.float32), ytr, split_fn, max_data_size=None,
        model_type="classifier", equal_split_size=True,
        data_shuffle_seed=seed, preprocessing_random_state=seed,
    )
    return next(iter(DataLoader(chunks, batch_size=1, collate_fn=meta_dataset_collator)))


# ---------------------------------------------------------------------------
# Gate training (tex §gates, §train): zero-init, NLL + L1 (+TV), simplex projection,
# fresh context/query split AND fresh community-preserving permutation per episode.
# ---------------------------------------------------------------------------

RHO0 = math.log(math.e - 1.0)  # softplus(RHO0) = 1.0  =>  tau init 1.0


def _project_gates(lam: torch.Tensor) -> None:
    """Project each block's gate vector onto {λ >= 0, Σ_r λ_r <= 1} (tex §gates)."""
    with torch.no_grad():
        lam.clamp_(min=0.0)
        s = lam.sum(dim=1, keepdim=True)
        lam.div_(torch.clamp(s, min=1.0))


def train_gates(Xtr, ytr, kg, blocks, *, model_path, device, seed, epochs, lr,
                query_frac, l1, tv, C, pool_ff, patience, min_delta,
                smooth_win=10, gate_mode="two", fpg=FPG, perm_mode="community"):
    """Returns (lam (L,R), lam_y (L,R), tau (R,), tau_y (R,), loss history, traj).

    C为token个数

    fpg / perm_mode: the fpg=1 arms train under fpg=1 tokenization (caller must hold
    the force_single_feature_tokens patch open around this call) with plain UNIFORM
    permutations — token == column makes the prior exact under any ordering, so the
    community sampler is unnecessary there by construction.

    gate_mode='two' (default): separate feature-row gates λ and target-readout gates
    λ_y per (block, relation). gate_mode='one': a SINGLE shared gate set — lam_y IS
    lam (same Parameter), so one λ^(l)_r drives both routes; the routing asymmetry
    itself (feature rows mix A^(r), readout row mixes b^(r), target column untouched)
    lives in the prior structure and is unchanged. 共享一组λ：两条路由强度绑定，
    但注入路径的不对称性保留在先验结构里。

    traj = {"lam": (E,R), "lam_y": (E,R)}: per-episode Σ_l λ^(l)_r trajectories — the
    DIRECT convergence readout (the per-episode loss is a high-variance estimate and
    oscillates even at fixed gates; a plateaued gate trajectory = converged).
    R = len(b_list) = len(A_list): each relation contributes a feature-row gate column
    lam[:, r] AND a target-readout gate column lam_y[:, r]; a gate whose channel is
    empty multiplies a zero base and stays inert (its L1 keeps it at 0).

    EARLY STOPPING: the per-episode loss is NOISY by design (fresh permutation + fresh
    context/query split every episode), so a raw 'this step did not improve' test would
    stop almost immediately. Instead the plateau test runs on a moving average over the
    last smooth_win episodes: stop once `patience` consecutive smoothed values fail to
    beat the best smoothed loss by more than min_delta (patience=0 disables; `epochs`
    stays the hard ceiling). Earliest possible stop = smooth_win + patience episodes,
    which also gives the zero-init gates time to travel into the helpful region."""
    R = len(kg["A_list"])
    tied = gate_mode == "one"
    clf = _batched_clf(model_path, device, seed)

    lam = lam_y = opt = None
    rho = torch.nn.Parameter(torch.full((R,), RHO0, device=device))
    rho_y = torch.nn.Parameter(torch.full((R,), RHO0, device=device))
    history: list[float] = []
    traj_lam: list[np.ndarray] = []
    traj_lam_y: list[np.ndarray] = []
    n_layers = None
    best_smooth, wait = float("inf"), 0

    for ep in range(epochs):
        rng = np.random.default_rng(seed * 10_000 + ep)
        perm = (community_perm(blocks, rng) if perm_mode == "community"
                else rng.permutation(Xtr.shape[1]))
        Xp = Xtr[:, perm]
        batch = _episode_batch(clf, Xp, ytr, query_frac=query_frac, seed=seed + 1000 + ep)
        clf.fit_from_preprocessed(batch.X_context, batch.y_context,
                                  batch.cat_indices, batch.configs)
        if lam is None:  # freeze weights once, tag blocks, create gates lazily (L known now)
            for prm in clf.models_[0].parameters():
                prm.requires_grad_(False)
            n_layers = tag_block_layers(clf)
            lam = torch.nn.Parameter(torch.zeros(n_layers, R, device=device))
            lam_y = (lam if tied
                     else torch.nn.Parameter(torch.zeros(n_layers, R, device=device)))
            gate_params = [lam] if tied else [lam, lam_y]
            opt = torch.optim.Adam(gate_params + [rho, rho_y], lr=lr)

        # Per-episode prior under THIS permutation, differentiable w.r.t. temperatures.
        tau = F.softplus(rho) + 1e-3
        tau_y = F.softplus(rho_y) + 1e-3
        priors = build_priors(kg, perm, C, tau, tau_y, device, pool_ff=pool_ff, fpg=fpg)

        with _kgam_ctx(priors, lam, lam_y, C):
            logits_QBEL = clf.forward(batch.X_query, return_raw_logits=True)
            Q, Bn, E, L = logits_QBEL.shape
            logits = logits_QBEL.permute(1, 2, 3, 0).reshape(Bn * E, L, Q)
            targets = batch.y_query.repeat(Bn * E, 1).to(device)
            loss = F.cross_entropy(logits, targets)
            if l1 > 0:  # penalize each DISTINCT parameter once (tied: lam_y is lam)
                pen = lam.sum() if tied else lam.sum() + lam_y.sum()
                loss = loss + l1 * pen                         # λ >= 0 after projection
            if tv > 0:
                pen = (lam[1:] - lam[:-1]).abs().sum()
                if not tied:
                    pen = pen + (lam_y[1:] - lam_y[:-1]).abs().sum()
                loss = loss + tv * pen

        opt.zero_grad()
        loss.backward()
        opt.step()
        _project_gates(lam)
        if not tied:
            _project_gates(lam_y)
        history.append(float(loss.detach()))
        traj_lam.append(lam.detach().sum(0).cpu().numpy())
        traj_lam_y.append(lam_y.detach().sum(0).cpu().numpy())

        if patience > 0 and len(history) >= smooth_win:
            smooth = float(np.mean(history[-smooth_win:]))
            if smooth < best_smooth - min_delta:
                best_smooth, wait = smooth, 0
            else:
                wait += 1
                if wait >= patience:
                    break

    tau = (F.softplus(rho) + 1e-3).detach()
    tau_y = (F.softplus(rho_y) + 1e-3).detach()
    traj = {"lam": np.array(traj_lam), "lam_y": np.array(traj_lam_y)}
    return lam.detach(), lam_y.detach(), tau, tau_y, history, traj


# ---------------------------------------------------------------------------
# Inference: KG-aware ensembling (community-preserving perms, shared gates)
# ---------------------------------------------------------------------------

def eval_kgam(prob, blocks, lam, lam_y, tau, tau_y, *, members, model_path, device,
              seed, gates_on=True, perm_mode="community", pool_ff="max", fpg=FPG):
    """E members, each with its own permutation and re-pooled prior; averaged probs.
    perm_mode='community' = KG-aware sampler (tex §ensemble); 'uniform' = TabPFN's
    native uniform shuffle (control / no-KG baseline). gates_on=False = patch off
    (bit-exact vanilla members under the same permutations). fpg=1 requires the caller
    to hold the force_single_feature_tokens patch open around this call."""
    kg = prob
    d = prob["X_train"].shape[1]
    C = math.ceil(d / fpg) + 1
    probs = None
    for e in range(members):
        rng = np.random.default_rng(seed * 777 + e)
        perm = (community_perm(blocks, rng) if perm_mode == "community"
                else rng.permutation(d))
        clf = fit_member(prob["X_train"][:, perm], prob["y_train"],
                         model_path=model_path, device=device, seed=seed + e)
        tag_block_layers(clf)
        Xte = prob["X_test"][:, perm].astype(np.float32)
        if gates_on:
            priors = build_priors(kg, perm, C, tau, tau_y, device,
                                  pool_ff=pool_ff, fpg=fpg)
            with _kgam_ctx(priors, lam, lam_y, C):
                p = clf.predict_proba(Xte)
        else:
            p = clf.predict_proba(Xte)
        probs = p if probs is None else probs + p
    return float((probs.argmax(1) == prob["y_test"]).mean())


def eval_vanilla(prob, *, members, model_path, device, seed, cols=None):
    """Real vanilla TabPFN (default preprocessing, its own uniform column shuffling),
    optionally restricted to a column subset (select / oracle baselines)."""
    Xtr, Xte = prob["X_train"], prob["X_test"]
    if cols is not None:
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    clf = TabPFNClassifier(n_estimators=members, model_path=model_path, device=device,
                           random_state=seed, ignore_pretraining_limits=True)
    clf.fit(Xtr.astype(np.float32), prob["y_train"])
    return float(clf.score(Xte.astype(np.float32), prob["y_test"]))


# ---------------------------------------------------------------------------
# One condition -> record;  main sweep + table + plot
# ---------------------------------------------------------------------------

class _Timer:
    """Times a block in CPU process time AND wall time.

    time.process_time() counts only CPU actually consumed by THIS process, so it is
    immune to contention from other jobs on a shared node.
    Caveats: it sums over threads (can exceed wall on multi-threaded CPU torch), and
    GPU compute is not CPU time, so on cuda we synchronize (otherwise async kernels
    escape both clocks) and wall is the honest end-to-end number there.
    """

    def __init__(self, device=None):
        self._sync = str(device).startswith("cuda") and torch.cuda.is_available()

    def __enter__(self):
        if self._sync:
            torch.cuda.synchronize()
        self._cpu0 = time.process_time()
        self._wall0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._sync:
            torch.cuda.synchronize()
        self.cpu = time.process_time() - self._cpu0
        self.wall = time.perf_counter() - self._wall0
        return False


def run_condition(*, seed, n_train, args) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.dgp == "mixed":
        prob = make_problem_mixed(
            n_train=n_train, n_test=args.n_test, n_features=args.n_features,
            n_corr=args.n_corr, n_int=args.n_int, pair_noise=args.pair_noise,
            beta_int=args.beta_int, kg_frac=args.kg_frac,
            label_noise=args.label_noise, seed=seed,
        )
        tag = f"mixed kg_frac={args.kg_frac}"
    else:
        prob = make_problem(
            n_train=n_train, n_test=args.n_test, n_features=args.n_features,
            k_rel=args.k_rel, n_known=args.n_known, n_false=args.n_false,
            label_noise=args.label_noise, seed=seed,
            nl_frac=args.nl_frac, hetero_w=args.hetero_w, link=args.link,
        )
        tag = f"known={args.n_known}/{args.k_rel}"
    blocks = kg_communities(prob["o"], prob["b_list"], prob["A_list"])
    d = args.n_features
    C = math.ceil(d / FPG) + 1

    with _Timer(args.device) as t:
        lam, lam_y, tau, tau_y, hist, traj = train_gates(
            prob["X_train"], prob["y_train"], prob, blocks,
            model_path=args.model_path, device=args.device, seed=seed,
            epochs=args.epochs, lr=args.lr, query_frac=args.query_frac,
            l1=args.l1, tv=args.tv, C=C, pool_ff=args.pool_ff,
            patience=args.patience, min_delta=args.min_delta,
            gate_mode=args.gate_mode,
        )

    common = dict(members=args.members, model_path=args.model_path,
                  device=args.device, seed=seed)
    rec = dict(seed=seed, n_train=n_train, dgp=args.dgp,
               n_features=args.n_features, k_rel=args.k_rel,
               n_known=args.n_known, n_test=args.n_test,
               kg_frac=args.kg_frac, gate_mode=args.gate_mode,
               pool_ff=args.pool_ff)
    rec["time_train"], rec["time_train_wall"] = t.cpu, t.wall

    with _Timer(args.device) as t:
        rec["base"] = eval_kgam(prob, blocks, lam, lam_y, tau, tau_y,
                                gates_on=False, perm_mode="uniform", **common)
    rec["time_base"], rec["time_base_wall"] = t.cpu, t.wall

    with _Timer(args.device) as t:
        rec["vanilla"] = eval_vanilla(prob, **common)
    rec["time_vanilla"], rec["time_vanilla_wall"] = t.cpu, t.wall

    with _Timer(args.device) as t:
        rec["ours0"] = eval_kgam(prob, blocks, lam, lam_y, tau, tau_y,
                                 gates_on=False, **common)
    rec["time_ours0"], rec["time_ours0_wall"] = t.cpu, t.wall

    with _Timer(args.device) as t:
        rec["ours"] = eval_kgam(prob, blocks, lam, lam_y, tau, tau_y,
                                gates_on=True, pool_ff=args.pool_ff, **common)
    rec["time_ours"], rec["time_ours_wall"] = t.cpu, t.wall

    with _Timer(args.device) as t:
        rec["oracle"] = eval_vanilla(prob, cols=prob["rel"], **common)
    rec["time_oracle"], rec["time_oracle_wall"] = t.cpu, t.wall

    # --- fpg=1 arms: token == column. Everything runs inside the grouping patch
    # (gate training AND both evals — the patch must cover every forward). Uniform
    # perms end to end: with exact per-column priors the community machinery is
    # unnecessary by construction, which is the point of these arms.
    if getattr(args, "fpg1", False):
        C1 = d + 1
        with force_single_feature_tokens() as cnt:
            with _Timer(args.device) as t:
                lam1, lam_y1, tau1, tau_y1, hist1, _ = train_gates(
                    prob["X_train"], prob["y_train"], prob, blocks,
                    model_path=args.model_path, device=args.device, seed=seed,
                    epochs=args.epochs, lr=args.lr, query_frac=args.query_frac,
                    l1=args.l1, tv=args.tv, C=C1, pool_ff=args.pool_ff,
                    patience=args.patience, min_delta=args.min_delta,
                    gate_mode=args.gate_mode, fpg=1, perm_mode="uniform",
                )
            rec["time_train_fpg1"], rec["time_train_fpg1_wall"] = t.cpu, t.wall

            with _Timer(args.device) as t:
                rec["base_fpg1"] = eval_kgam(prob, blocks, lam1, lam_y1, tau1, tau_y1,
                                             gates_on=False, perm_mode="uniform",
                                             fpg=1, **common)
            rec["time_base_fpg1"], rec["time_base_fpg1_wall"] = t.cpu, t.wall

            # default preprocessing + fpg=1, no KG: the grouping patch runs inside
            # the forward, after the whole preprocessing pipeline, so transforms /
            # fingerprint / feature shuffle all just become extra single-column tokens
            with _Timer(args.device) as t:
                rec["vanilla_fpg1"] = eval_vanilla(prob, **common)
            rec["time_vanilla_fpg1"], rec["time_vanilla_fpg1_wall"] = t.cpu, t.wall

            with _Timer(args.device) as t:
                rec["ours_fpg1"] = eval_kgam(prob, blocks, lam1, lam_y1, tau1, tau_y1,
                                             gates_on=True, perm_mode="uniform",
                                             pool_ff=args.pool_ff, fpg=1, **common)
            rec["time_ours_fpg1"], rec["time_ours_fpg1_wall"] = t.cpu, t.wall
        assert cnt["n_calls"] > 0, "fpg=1 patch 未生效：forward 没走被替换的分组函数"
        rec["lam_sums_fpg1"] = np.round(lam1.sum(0).cpu().numpy(), 2).tolist()
        rec["lamy_sums_fpg1"] = np.round(lam_y1.sum(0).cpu().numpy(), 2).tolist()
        rec["loss_hist_fpg1"] = hist1

    # Per-relation gate mass over blocks: every relation r has a feature-row gate
    # column lam[:, r] and a target-readout gate column lam_y[:, r] (mixed: r=0 corr,
    # r=1 int, r=2 relevance; label: r=0). Gates on an empty channel stay ~0 (inert).
    lam_sums = np.round(lam.sum(0).cpu().numpy(), 2).tolist()
    lamy_sums = np.round(lam_y.sum(0).cpu().numpy(), 2).tolist()
    fpg1_str = (f"base_fpg1={100*rec['base_fpg1']:.1f} "
                f"van_fpg1={100*rec['vanilla_fpg1']:.1f} "
                f"ours_fpg1={100*rec['ours_fpg1']:.1f} "
                if "base_fpg1" in rec else "")
    print(f"[n={n_train} {tag} s={seed} gates={args.gate_mode}] "
          f"base={100*rec['base']:.1f} van={100*rec['vanilla']:.1f} "
          f"ours0={100*rec['ours0']:.1f} ours={100*rec['ours']:.1f} "
          f"{fpg1_str}")
    print("  Times cpu/wall (s): " + ", ".join(
        f"{k}={rec['time_' + k]:.1f}/{rec['time_' + k + '_wall']:.1f}"
        for k in ("train", "base", "vanilla", "ours0", "ours", "oracle")))
    rec["lam_profile"] = np.round(lam.cpu().numpy(), 3).tolist()      # (L,R) per block
    rec["lam_y_profile"] = np.round(lam_y.cpu().numpy(), 3).tolist()  # (L,R) per block
    rec["lam_sums"] = lam_sums
    rec["lamy_sums"] = lamy_sums
    rec["loss_hist"] = hist
    rec["gate_traj_lam"] = np.round(traj["lam"], 3).tolist()      # (E,R) per episode
    rec["gate_traj_lam_y"] = np.round(traj["lam_y"], 3).tolist()  # (E,R) per episode
    return rec


METHODS = ["base", "vanilla", "ours0", "ours",
           "base_fpg1", "vanilla_fpg1", "ours_fpg1", "oracle"]
STYLE = {"base": ("black", "--"), "vanilla": ("tab:brown", "--"),
         "ours0": ("tab:gray", ":"),
         "ours": ("tab:red", "-"),
         "base_fpg1": ("tab:blue", "--"),
         "vanilla_fpg1": ("tab:purple", "--"),
         "ours_fpg1": ("tab:blue", "-"),
         "oracle": ("tab:green", "-.")}


def plot_results(df, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for m in METHODS:
        if m not in df.columns:  # fpg=1 arms are optional (--no-fpg1)
            continue
        g = df.groupby("n_train")[m]
        mean, se = g.mean(), g.std() / np.sqrt(g.count())
        color, ls = STYLE[m]
        ax.errorbar(mean.index, 100 * mean.values, yerr=100 * se.values,
                    color=color, ls=ls, marker="o", capsize=3, label=m)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_train")
    ax.set_ylabel("test acc (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    gm = df.gate_mode.iloc[0] if "gate_mode" in df.columns else "two"
    if df.dgp.iloc[0] == "mixed":
        pf = df.pool_ff.iloc[0] if "pool_ff" in df.columns else "?"
        ax.set_title(f"KGAM mixed DGP (corr+int+label KG, kg_frac={df.kg_frac.iloc[0]})"
                     f" | pool_ff: {pf} | λ-sets: {gm}")
    else:
        ax.set_title(f"KGAM label DGP: KG knows {df.n_known.iloc[0]} relevant cols"
                     f" | λ-sets: {gm}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved -> {save_path}")


def plot_lambdas(df, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # 获取所有的 n_train 取值
    n_trains = df["n_train"].unique()
    
    # 动态创建画布，2 行（lam 和 lam_y），n_trains 这么多个列
    fig, axes = plt.subplots(2, len(n_trains), figsize=(6 * len(n_trains), 8), squeeze=False)
    
    r_count = len(df.iloc[0]["lam_sums"])
    l_count = len(df.iloc[0]["lam_profile"])
    layers = np.arange(l_count)
    
    for idx, n_train in enumerate(n_trains):
        group = df[df["n_train"] == n_train]
        
        # 将相同 n_train 的多个 seed 的矩阵平均化
        # 结果形状为 (L, R)，即 (24层, 关系数)
        avg_lam = np.mean(np.array(group["lam_profile"].tolist()), axis=0) 
        avg_lamy = np.mean(np.array(group["lam_y_profile"].tolist()), axis=0) 
        
        # 遍历每一种关系并画出它的层级曲线
        for r in range(r_count):
            axes[0, idx].plot(layers, avg_lam[:, r], marker='.', label=f"Relation {r}")
            axes[1, idx].plot(layers, avg_lamy[:, r], marker='.', label=f"Relation {r}")
            
        axes[0, idx].set_title(f"lam (Feature-Feature) | n_train={n_train}")
        axes[0, idx].set_xlabel("Layer (0 to 23)")
        axes[0, idx].set_ylabel("Lambda Value")
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)
        
        axes[1, idx].set_title(f"lam_y (Target-Readout) | n_train={n_train}")
        axes[1, idx].set_xlabel("Layer (0 to 23)")
        axes[1, idx].set_ylabel("Lambda Value")
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)
        
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Layer-wise Lambda Plot saved -> {save_path}")

def plot_times(df, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    
    time_cols = ["time_train", "time_base", "time_vanilla", "time_ours0", "time_ours",
                 "time_train_fpg1", "time_base_fpg1", "time_vanilla_fpg1",
                 "time_ours_fpg1", "time_oracle"]
    labels = ["Train (Gates)", "Base", "Vanilla", "Ours0", "Ours",
              "Train fpg1", "Base fpg1", "Vanilla fpg1", "Ours fpg1", "Oracle"]
    colors = ["tab:purple", "black", "tab:brown", "tab:gray", "tab:red",
              "tab:cyan", "tab:blue", "tab:pink", "tab:orange", "tab:green"]
    
    for col, label, color in zip(time_cols, labels, colors):
        if col in df.columns:
            g = df.groupby("n_train")[col]
            mean, se = g.mean(), g.std() / np.sqrt(g.count())
            ax.errorbar(mean.index, mean.values, yerr=se.values,
                        color=color, marker="o", capsize=3, label=label)
                        
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_train")
    ax.set_ylabel("CPU process time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if df.dgp.iloc[0] == "mixed":
        pf = df.pool_ff.iloc[0] if "pool_ff" in df.columns else "?"
        ax.set_title(f"Execution Times (mixed DGP, kg_frac={df.kg_frac.iloc[0]}, "
                     f"pool_ff={pf})")
    else:
        ax.set_title(f"Execution Times (label DGP, known={df.n_known.iloc[0]})")
        
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Time Plot saved -> {save_path}")

def _pad_last(seqs):
    """Stack per-seed trajectories of unequal length (early stopping) by repeating
    each one's last value to the longest length. Works for 1-D (loss) and 2-D (E,R)
    gate trajectories."""
    arrs = [np.asarray(s, dtype=float) for s in seqs]
    L = max(a.shape[0] for a in arrs)
    return np.stack([
        np.concatenate([a, np.repeat(a[-1:], L - a.shape[0], axis=0)])
        if a.shape[0] < L else a
        for a in arrs
    ])


def plot_loss(df, save_path, smooth_win=10):
    """Training diagnostics, one column per n_train.
    Row 1  per-episode NLL: each seed as a faint raw curve + the seed-averaged
           moving average (bold) — the quantity the early stop watches. The raw
           curve oscillates BY DESIGN (fresh permutation + fresh context/query split
           every episode gives a high-variance loss estimate even at fixed gates),
           so convergence is judged on the smoothed curve only.
    Row 2  Σ_l λ^(l)_r gate-mass trajectories per relation (dashed = feature-row
           gates λ, solid = target-readout gates λ_y), seed-averaged — the DIRECT
           'did the gates converge' readout: a plateau here = converged, regardless
           of how noisy row 1 looks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_trains = sorted(df["n_train"].unique())
    fig, axes = plt.subplots(2, len(n_trains), figsize=(6 * len(n_trains), 8),
                             squeeze=False)
    rel_colors = plt.get_cmap("tab10").colors  # fixed order: rel 0, 1, 2, ...

    for idx, n_train in enumerate(n_trains):
        group = df[df["n_train"] == n_train]

        # --- row 1: raw loss (faint, per seed) + seed-mean moving average (bold)
        ax = axes[0, idx]
        H = _pad_last(group["loss_hist"].tolist())          # (seeds, E)
        E = H.shape[1]
        for h in H:
            ax.plot(range(1, E + 1), h, color="tab:blue", alpha=0.25, lw=0.8)
        k = min(smooth_win, E)
        kern = np.ones(k) / k
        S = np.array([np.convolve(h, kern, mode="valid") for h in H])
        ax.plot(range(k, E + 1), S.mean(0), color="tab:blue", lw=2.2,
                label=f"{k}-ep moving avg (early-stop signal)")
        ax.set_title(f"gate-training NLL | n_train={n_train}")
        ax.set_xlabel("episode")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # --- row 2: Σλ trajectories per relation (convergence readout)
        ax = axes[1, idx]
        TL = _pad_last(group["gate_traj_lam"].tolist()).mean(0)    # (E, R)
        TY = _pad_last(group["gate_traj_lam_y"].tolist()).mean(0)  # (E, R)
        eps = range(1, TL.shape[0] + 1)
        for r in range(TL.shape[1]):
            c = rel_colors[r % len(rel_colors)]
            if TL[:, r].max() > 0:
                ax.plot(eps, TL[:, r], color=c, ls="--", label=f"Σλ rel {r} (feat rows)")
            if TY[:, r].max() > 0:
                ax.plot(eps, TY[:, r], color=c, ls="-", label=f"Σλ_y rel {r} (readout)")
        ax.set_title(f"gate mass Σ_l λ | n_train={n_train}")
        ax.set_xlabel("episode")
        ax.set_ylabel("Σλ over blocks")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Loss/gate-trajectory plot saved -> {save_path}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(15)))
    # log2-spaced; 240 is the right endpoint where the KG gain should have shrunk
    # towards 0 (enough data to find the relevant columns without the KG)
    p.add_argument("--n-train", type=int, nargs="+", default=[30, 60, 120, 240])
    # 500 test rows: per-seed SE ~2.2pp, ~0.7pp after 10 seeds — needed to resolve
    # 1-3pp gains; test rows only cost eval forwards, linear
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--dgp", choices=["label", "mixed"], default="label",
                   help="'label' = extreme label-channel-only case; 'mixed' = "
                        "correlation + interaction + label KG (3 relations)")
    # shared DGP knobs. The d sweep (one PBS job per value: 50/100/200) keeps k_rel
    # FIXED — same signal, more distractor columns, so 'KG value vs d' reads cleanly.
    p.add_argument("--n-features", type=int, default=100)
    p.add_argument("--label-noise", type=float, default=0.2)
    # --dgp label: k_rel relevant i.i.d. columns at random positions, KG knows n_known.
    # SPARSE signal by default (12/100): with the old dense setting (50/100) even the
    # oracle on the true columns was near chance at n<=120 (headroom ~0), so no KG
    # method could show anything. KG's value proposition is sparse signal + small n.
    p.add_argument("--k-rel", type=int, default=12)
    # partial coverage (6/12) is the MAIN arm — the honest scenario and where the
    # ours-vs-hard-select crossover lives; full coverage (12/12) is the ceiling arm
    # (selecting the known columns there ~= oracle), submitted separately via PBS
    p.add_argument("--n-known", type=int, default=6,
                   help="how many of the k_rel relevant columns the KG observes (b=1)")
    p.add_argument("--n-false", type=int, default=0,
                   help="WRONG KG entries: irrelevant columns the KG claims relevant")
    p.add_argument("--nl-frac", type=float, default=0.4,
                   help="fraction of relevant columns with a NONLINEAR per-feature "
                        "response (x²−1 / tanh); the rest stay linear")
    p.add_argument("--hetero-w", action="store_true", default=True,
                   help="heterogeneous weight magnitudes (KG still only sees the 0/1 "
                        "edge, never the strength)")
    p.add_argument("--no-hetero-w", dest="hetero_w", action="store_false")
    # default 'hard': the logit link caps Bayes acc at ~67% and crushes the
    # oracle-base headroom to ~5pp (measured: hard link ~11pp at n=60); keep logit
    # as a robustness arm, not the main setting.
    p.add_argument("--link", choices=["logit", "hard"], default="hard",
                   help="'logit' = soft labels y~Bernoulli(σ(g)) (Bayes error>0); "
                        "'hard' = deterministic median threshold")
    # --dgp mixed: correlated pairs + interaction pairs + label relevance
    # sparser + stronger than before (3+3 pairs instead of 10+10, tighter pair copies,
    # bigger interaction weight): with the old dense setting oracle-base headroom was
    # ~0.7pp at n=60, i.e. nothing for any KG method to win; this setting measures
    # base~54 oracle~62 (headroom ~8pp, 4 seeds, members=2).
    p.add_argument("--n-corr", type=int, default=3,
                   help="correlated pairs (linear signal via a shared latent)")
    p.add_argument("--n-int", type=int, default=3,
                   help="interaction pairs (label += beta_int * x_a * x_b)")
    p.add_argument("--pair-noise", type=float, default=0.3,
                   help="noise of the two copies around the pair latent")
    p.add_argument("--beta-int", type=float, default=2.5)
    p.add_argument("--kg-frac", type=float, nargs="+", default=[1.0, 0.7],
                   help="fraction of each KG channel's entries that is observed")
    # 决定lambda的数量。默认 two：one 是 two 的严格子集（label DGP 下两者数学等价，
    # 结果逐位一致），参数量 24R vs 48R 个标量都远不到过拟合量级，two 的逐路由
    # gate 还能分开读出（可解释性）；one 保留作消融。
    p.add_argument("--gate-mode", choices=["two", "one"], default="two",
                   help="'two' = separate gate sets: feature-row λ and target-readout "
                        "λ_y per relation; 'one' = a single shared "
                        "gate set λ drives both routes (KG nodes treated uniformly; "
                        "the routing asymmetry — readout row mixed via b, target "
                        "column untouched — is kept, only gate strengths are tied)")
    # 池化方法
    p.add_argument("--pool-ff", choices=["mean", "max"], default="mean",
                   help="pooling for BOTH KG channels: the feature-feature matrices "
                        "AND the feature-label vectors (tex: max for edge-sparse "
                        "graphs; mean dilutes single edges by up to 1/p^2; on the "
                        "label DGP the two are identical for b, so pool is inert "
                        "there either way)")
    # gate training. NB the helpful gate region can be LARGE (λ~0.4-0.7 per block in the
    # fixed sweep) — zero-init needs enough epochs × lr to travel there; keep L1 gentle.
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.1)
    # 用训练集训练gate时需要随机将其分成训练组和测试组（query组），query组的比例
    p.add_argument("--query-frac", type=float, default=0.3)
    # l1惩罚项的系数
    p.add_argument("--l1", type=float, default=3e-4)
    # 总变分惩罚项的系数
    p.add_argument("--tv", type=float, default=0.0)
    # 早停的轮数，如果连续15次的华东平均值没有进步超过min-delta，那么就会早停。如果设置为0，则必须跑满所有epoch
    p.add_argument("--patience", type=int, default=15,
                   help="early stop after this many episodes without the smoothed "
                        "(10-episode moving average) loss improving by > min_delta; "
                        "0 disables (--epochs is always the hard ceiling)")
    p.add_argument("--min-delta", type=float, default=2e-3,
                   help="minimum smoothed-loss improvement that resets patience")
    # fpg=1 arms (token == column; exact prior, uniform perms, no clustering)
    p.add_argument("--fpg1", action="store_true", default=True,
                   help="also run base_fpg1/ours_fpg1: one column per token via the "
                        "encoder grouping patch; prior injected exactly, uniform "
                        "perms, no community clustering (roughly doubles runtime, "
                        "and each fpg=1 forward is ~3x slower: C goes d/3 -> d)")
    p.add_argument("--no-fpg1", dest="fpg1", action="store_false")
    # inference ensembling
    p.add_argument("--members", type=int, default=4,
                   help="ensemble members (community-preserving perms; also vanilla's "
                        "n_estimators for a fair comparison)")
    p.add_argument("--out", default="kg_e_kgam")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd
    import copy

    recs = []
    kg_fracs = args.kg_frac if isinstance(args.kg_frac, list) else [args.kg_frac]
    if args.dgp == "label":
        # the label DGP never reads kg_frac — one pass only (sweeping it would just
        # repeat identical runs and emit duplicate plot sets)
        kg_fracs = [kg_fracs[0]]

    for kg_f in kg_fracs:
        for n_train in args.n_train:
            for seed in args.seeds:
                curr_args = copy.copy(args)
                curr_args.kg_frac = kg_f
                recs.append(run_condition(seed=seed, n_train=n_train, args=curr_args))
    df = pd.DataFrame(recs)

    print("\n=== mean test acc (%) ===")
    mets = [m for m in METHODS if m in df.columns]  # fpg=1 arms optional (--no-fpg1)
    summary = df.groupby(["kg_frac", "n_train"])[mets].mean().mul(100).round(1)
    summary["headroom"] = (summary["oracle"] - summary["base"]).round(1)
    summary["gain"] = (summary["ours"] - summary["base"]).round(1)
    # captured = KG 增益占 headroom 的比例；headroom 太小时该比值无意义，置 NaN
    summary["captured%"] = np.where(
        summary["headroom"] >= 1.0,
        (100 * summary["gain"] / summary["headroom"]).round(0), np.nan)
    if "ours_fpg1" in df.columns:
        # tok_cost prices losing in-token encoder mixing; gain_fpg1 is the KG gain at
        # column resolution (exact prior, no clustering) on its own fpg=1 baseline
        summary["tok_cost"] = (summary["base_fpg1"] - summary["base"]).round(1)
        summary["gain_fpg1"] = (summary["ours_fpg1"] - summary["base_fpg1"]).round(1)
    print(summary.to_string())
    print("\n(read: gain = ours - base = total KG gain; "
          "headroom = oracle - base = what a perfect KG could buy; "
          "captured% = gain/headroom; ours0 - base = perm-sampler share; "
          "vanilla differs from base only by preprocessing, not KG; "
          "tok_cost = base_fpg1 - base = price of 1-column tokens; "
          "gain_fpg1 = ours_fpg1 - base_fpg1 = exact-prior KG gain, no clustering)")
          
    # Raw records: everything needed to replot locally without rerunning.
    run_stem = f"{args.out}_{args.dgp}_lam{args.gate_mode}"
    df.to_json(f"{run_stem}_records.json", orient="records")
    print(f"Records saved -> {run_stem}_records.json")

    # Filenames: <out>_<dgp>_lam<gate_mode>_<DGP-specific knob>_<plot>.png
    #   label: the KG knob is coverage of the relevance channel -> known{n_known}of{k_rel}
    #   mixed: the KG knob is the observed fraction per channel -> frac{kg_frac}
    # so one/two-gate runs and the two DGPs can never overwrite each other.
    for kg_f, group_df in df.groupby("kg_frac"):
        knob = (f"known{args.n_known}of{args.k_rel}" if args.dgp == "label"
                else f"frac{kg_f}_p{args.pool_ff}")
        stem = f"{run_stem}_{knob}"
        plot_results(group_df.copy(), f"{stem}_acc.png")
        plot_lambdas(group_df.copy(), f"{stem}_lambdas.png")
        plot_times(group_df.copy(), f"{stem}_times.png")
        plot_loss(group_df.copy(), f"{stem}_loss.png")


if __name__ == "__main__":
    main()
