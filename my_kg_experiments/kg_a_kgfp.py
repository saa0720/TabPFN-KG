"""Method A — KGFP: KG feature propagation in DATA space (TabPFN___RAG.pdf §4).

This is a DIFFERENT injection from kg_pre_injection.py. There the KG knew about the
label (it gave feature embeddings E *and* a target/concept node E_y, and we projected
X @ (E @ E_y^T)). Here the KG is PURELY feature-feature (column-column): y is NOT a
node (y ∉ V, by the paper's design), so there is no possible label leak. The KG only
tells us which COLUMNS are related.

KGFP idea (paper §4):
  * treat each sample's row x_i ∈ R^d as a graph signal over the d feature NODES;
  * smooth it along the FEATURE axis with the KG propagation operator Â:
        one step:  X' = X Â            (x'_iq = Σ_p x_ip Â_pq, aggregates KG neighbors)
        APPNP/PPR: X^(0)=X, X^(k+1) = α X^(k) Â + (1-α) X^(0),  X' = X^(K)   (Eq. 4)
  * AUGMENT the table (Eq. 5):  X̃ = [X ∥ X'] ∈ R^{n×2d}, feed FROZEN TabPFN.
  * training-free: α, K are hyperparameters. No weight change, no hook.

Why this can help (and stays honest):
  X' is a KG-GUIDED DENOISER. In the SCM below, graph-connected features share a latent
  group factor measured with noise σ. Averaging a feature's KG neighbors estimates that
  group factor with variance ≈ σ²/group_size — i.e. the KG hands TabPFN the
  feature-grouping it cannot estimate itself from n ≪ d samples. So the prediction the
  paper makes is: KGFP helps when (a) n is SMALL and (b) σ is HIGH, and it must
  COLLAPSE BACK TO BASE when n is large, σ is small, or the KG is wrong/random/permuted
  (the augment safety net). Those "correctly useless" curves are the honesty evidence.

  This survives TabPFN's per-column z-scoring because X' changes the column CONTENT
  (a cross-feature combination), not its scale — consistent with the verified no-op of a
  scalar λ on a column. See kg_pre_injection.py header.

Data generative model (community / block SCM — the column-level KG is the block structure)
-----------------------------------------------------------------------------------------
    d features partitioned into G groups; group(j) ∈ {1..G}, sizes ≈ d/G
    u_i  ~ N(0, I_G)                          per-sample latent GROUP factors
    X_ij = u_{i, group(j)} + σ * eps_ij       feature value (noisy view of its group)
    R    ⊂ {1..G}                             the |R| label-relevant groups
    y    = 1[ g(u_{i,R}) + ln*eps > median ]  g: linear | quadratic | interaction | mlp

True KG  A_true[p,q] = 1 iff group(p)==group(q), p≠q   (block-diagonal feature graph).
KG quality is degraded by REWIRING a fraction of edges (true→random); measured on the
operator as Frobenius cosine cos_F(Â_obs, Â_true) ∈ [0,1] (true→1, random→~0).
Controls: `random` (Erdős–Rényi, matched density) and `permuted` (shuffle node→group).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the verified TabPFN plumbing (same env as the other kg_* files).

DGPS = ("linear", "quadratic", "interaction", "mlp")
DGP_N_REL = {"linear": 1, "quadratic": 1, "interaction": 2, "mlp": 3}


# ---------------------------------------------------------------------------
# Data: community / block SCM. The KG IS the block structure of the columns.
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Data:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    groups: np.ndarray      # (d,) int group id of each feature node
    A_true: np.ndarray      # (d, d) block-diagonal 0/1 feature graph (no self loops)
    dgp: str

import torch
from tabpfn.classifier import TabPFNClassifier
from tabpfn.preprocessing import PreprocessorConfig

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
    clf = TabPFNClassifier(
        n_estimators=1, model_path=model_path, device=device,
        fit_mode="fit_preprocessors", random_state=random_state,
        ignore_pretraining_limits=True, inference_config=_simple_inference_config(),
    )
    clf.fit(X_train, y_train)
    return clf

def _score(U_rel: np.ndarray, dgp: str, rng: np.random.Generator) -> np.ndarray:
    """Map relevant group factors U_rel (n, |R|) -> standardised 1-D pre-threshold g."""
    if dgp == "linear":
        g = U_rel[:, 0]
    elif dgp == "quadratic":
        g = U_rel[:, 0] ** 2
    elif dgp == "interaction":
        g = U_rel[:, 0] * U_rel[:, 1]
    elif dgp == "mlp":
        k, h = U_rel.shape[1], 8
        A = rng.standard_normal((k, h))
        b = rng.standard_normal(h)
        g = np.tanh(U_rel @ A) @ b
    else:
        raise ValueError(f"unknown dgp {dgp}")
    return (g - g.mean()) / max(g.std(), 1e-6)


def generate_data(
    *, dgp: str, n_train: int, n_test: int, n_features: int, n_groups: int,
    feature_noise: float, label_noise: float, seed: int,
    shuffle_features: bool = True,
) -> Data:
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    n_rel = DGP_N_REL[dgp]

    # Balanced group assignment over the d feature nodes. KGFP is invariant to column
    # order (it acts on the d×d KG directly), so shuffling is fine there. KGAB injects
    # at TabPFN's feature-TOKEN level (consecutive blocks of 3), so it needs columns
    # ordered by community (shuffle_features=False) for tokens to be within-community.
    groups = np.sort(np.arange(n_features) % n_groups)
    if shuffle_features:
        rng.shuffle(groups)

    U = rng.standard_normal((n_total, n_groups))               # group latent factors
    X = U[:, groups] + feature_noise * rng.standard_normal((n_total, n_features))

    rel = np.arange(n_rel)                                       # relevant groups = 0..n_rel-1
    g = _score(U[:, rel], dgp, rng) + label_noise * rng.standard_normal(n_total)
    thr = np.quantile(g[:n_train], 0.5)
    y = (g > thr).astype(np.int64)

    # True column-level KG: connect features sharing a group (block diagonal).
    A_true = (groups[:, None] == groups[None, :]).astype(np.float32)
    np.fill_diagonal(A_true, 0.0)

    X = X.astype(np.float32)
    return Data(
        X_train=X[:n_train], y_train=y[:n_train],
        X_test=X[n_train:], y_test=y[n_train:],
        groups=groups, A_true=A_true, dgp=dgp,
    )


# ---------------------------------------------------------------------------
# KG operator + KGFP propagation (paper §3 Eq.2, §4 Eq.4–5)
# ---------------------------------------------------------------------------

def normalize_adj(A: np.ndarray) -> np.ndarray:
    """Â = D̃^{-1/2}(A + I) D̃^{-1/2}, symmetric-normalised propagation operator (Eq. 2)."""
    A = A + np.eye(A.shape[0], dtype=A.dtype)
    deg = A.sum(1)
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    return dinv[:, None] * A * dinv[None, :]


def appnp(X: np.ndarray, Ahat: np.ndarray, *, alpha: float, K: int) -> np.ndarray:
    """X^(k+1) = α X^(k) Â + (1-α) X^(0); return X^(K) (Eq. 4). K=1,α=1 -> one-step X Â."""
    X0 = X
    Xk = X
    for _ in range(K):
        Xk = alpha * (Xk @ Ahat) + (1.0 - alpha) * X0
    return Xk


def corrupt_graph(A_true: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Rewire a `frac` of the true edges to random non-edges (density preserved).
    frac=0 -> A_true; frac=1 -> (essentially) a random graph of the same density."""
    d = A_true.shape[0]
    iu, ju = np.triu_indices(d, k=1)
    is_edge = A_true[iu, ju] > 0
    edge_pos = np.flatnonzero(is_edge)
    non_pos = np.flatnonzero(~is_edge)

    A = A_true.copy()
    n_rw = int(round(frac * edge_pos.size))
    if n_rw == 0:
        return A
    drop = rng.choice(edge_pos, size=n_rw, replace=False)
    add = rng.choice(non_pos, size=min(n_rw, non_pos.size), replace=False)
    for idx in drop:
        A[iu[idx], ju[idx]] = A[ju[idx], iu[idx]] = 0.0
    for idx in add:
        A[iu[idx], ju[idx]] = A[ju[idx], iu[idx]] = 1.0
    return A


def random_graph(A_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Erdős–Rényi graph with the same number of edges as A_true (KG carries no signal)."""
    d = A_true.shape[0]
    iu, ju = np.triu_indices(d, k=1)
    n_edges = int(A_true[iu, ju].sum())
    A = np.zeros_like(A_true)
    pick = rng.choice(iu.size, size=n_edges, replace=False)
    A[iu[pick], ju[pick]] = A[ju[pick], iu[pick]] = 1.0
    return A


def permuted_graph(A_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle node labels: real block structure, but mapped to the WRONG features."""
    perm = rng.permutation(A_true.shape[0])
    return A_true[np.ix_(perm, perm)]


def kg_quality(Ahat_obs: np.ndarray, Ahat_true: np.ndarray) -> float:
    """Measured KG quality = Frobenius cosine of the two propagation operators ∈ [0,1].
    This is the actual object KGFP consumes. true->1, random/permuted->~0."""
    a, b = Ahat_obs.ravel(), Ahat_true.ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Injection + TabPFN eval
# ---------------------------------------------------------------------------

def _tabpfn_acc(Xtr, ytr, Xte, yte, *, model_path, device, seed) -> float:
    clf = fit_tabpfn(Xtr.astype(np.float32), ytr,
                     model_path=model_path, device=device, random_state=seed)
    return float(clf.score(Xte.astype(np.float32), yte))


def kgfp_acc(d: Data, A_obs: np.ndarray, *, alpha, K, model_path, device, seed) -> float:
    """Build Â from A_obs, smooth train+test rows, augment [X ∥ X'], score frozen TabPFN."""
    Ahat = normalize_adj(A_obs)
    Xtr_s = appnp(d.X_train, Ahat, alpha=alpha, K=K)
    Xte_s = appnp(d.X_test, Ahat, alpha=alpha, K=K)
    atr = np.concatenate([d.X_train, Xtr_s], axis=1)
    ate = np.concatenate([d.X_test, Xte_s], axis=1)
    return _tabpfn_acc(atr, d.y_train, ate, d.y_test,
                       model_path=model_path, device=device, seed=seed)


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Point:
    dgp: str
    n_train: int
    seed: int
    mode: str        # base | true | rewire | random | permuted
    frac: float      # frac of edges rewired
    quality: float   # measured cos_F(Âobs, Âtrue); nan for base
    acc: float


def run_condition(
    *, dgp, seed, n_train, n_test, n_features, n_groups, feature_noise, label_noise,
    fracs, alpha, K, model_path, device,
) -> list[Point]:
    set_seed(seed)
    d = generate_data(
        dgp=dgp, n_train=n_train, n_test=n_test, n_features=n_features,
        n_groups=n_groups, feature_noise=feature_noise, label_noise=label_noise, seed=seed,
    )
    rng = np.random.default_rng(seed + 4242)
    Ahat_true = normalize_adj(d.A_true)
    tab = dict(alpha=alpha, K=K, model_path=model_path, device=device, seed=seed)

    pts = [Point(dgp, n_train, seed, "base", float("nan"), float("nan"),
                 _tabpfn_acc(d.X_train, d.y_train, d.X_test, d.y_test,
                             model_path=model_path, device=device, seed=seed))]

    # Graded KG quality via edge rewiring (frac=0 is the clean/true KG).
    for f in fracs:
        A = d.A_true if f == 0.0 else corrupt_graph(d.A_true, f, rng)
        q = kg_quality(normalize_adj(A), Ahat_true)
        mode = "true" if f == 0.0 else "rewire"
        pts.append(Point(dgp, n_train, seed, mode, f, q, kgfp_acc(d, A, **tab)))

    # Controls: random graph and permuted node labels (should fall back to ≈base).
    A_rnd = random_graph(d.A_true, rng)
    pts.append(Point(dgp, n_train, seed, "random", float("nan"), kg_quality(normalize_adj(A_rnd), Ahat_true),
                     kgfp_acc(d, A_rnd, **tab)))
    A_prm = permuted_graph(d.A_true, rng)
    pts.append(Point(dgp, n_train, seed, "permuted", float("nan"), kg_quality(normalize_adj(A_prm), Ahat_true),
                     kgfp_acc(d, A_prm, **tab)))

    for p in pts:
        print(f"  {dgp:11s} n={n_train:<4d} s={seed} {p.mode:8s} "
              f"q={p.quality:.3f} acc={100*p.acc:.1f}")
    return pts


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_quality(points: list[Point], save_path: str) -> None:
    """Headline 1: accuracy vs KG quality (one panel per dgp), fixed n_train."""
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame([dataclasses.asdict(p) for p in points])
    dgps = [g for g in DGPS if g in set(df.dgp)]
    fig, axes = plt.subplots(1, len(dgps), figsize=(5 * len(dgps), 4.4),
                             sharey=True, squeeze=False)
    for ax, dgp in zip(axes[0], dgps):
        sub = df[df.dgp == dgp]
        base = sub[sub["mode"] == "base"].acc.mean()
        grad_sub = sub[sub["mode"].isin(["true", "rewire"])]
        grad_acc = grad_sub.groupby("frac").acc.mean().sort_index()
        grad_q = grad_sub.groupby("frac").quality.mean().sort_index()
        ax.plot(grad_acc.index, 100 * grad_acc.values, "o-", color="tab:blue",
                label="KGFP (augment)")
        
        xticks = grad_acc.index
        xticklabels = [f"{x:.2f}\n(q={grad_q[x]:.2f})" for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=8)

        for mode, mk, c in [("random", "s", "gray"), ("permuted", "x", "tab:orange")]:
            m = sub[sub["mode"] == mode]
            if len(m):
                ax.axhline(100 * m.acc.mean(), ls=":", color=c, alpha=0.8,
                           label=f"{mode}")
        ax.axhline(100 * base, ls="--", color="black", alpha=0.6,
                   label="base")
        ax.set_title(dgp)
        ax.set_xlabel("Edge rewiring fraction (frac)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("test accuracy (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("KGFP: accuracy vs KG quality", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {save_path}")


def plot_nsweep(points: list[Point], save_path: str) -> None:
    """Headline 2: accuracy vs n_train, base vs KGFP(true) vs controls (one panel per dgp)."""
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame([dataclasses.asdict(p) for p in points])
    dgps = [g for g in DGPS if g in set(df.dgp)]
    fig, axes = plt.subplots(1, len(dgps), figsize=(5 * len(dgps), 4.4),
                             sharey=True, squeeze=False)
    for ax, dgp in zip(axes[0], dgps):
        sub = df[df.dgp == dgp]
        
        m_base = sub[sub["mode"] == "base"].groupby("n_train").acc.mean().sort_index()
        if len(m_base):
            ax.plot(m_base.index, 100 * m_base.values, "--", color="black", marker="o", alpha=0.6, label="base")
            
        m_true = sub[sub["mode"] == "true"]
        if len(m_true):
            q_mean = m_true.quality.mean()
            m_true_acc = m_true.groupby("n_train").acc.mean().sort_index()
            ax.plot(m_true_acc.index, 100 * m_true_acc.values, "-", color="tab:blue", marker="o", alpha=0.7, label=f"KGFP (true KG, q={q_mean:.2f})")
            
        m_rewire = sub[sub["mode"] == "rewire"]
        if len(m_rewire):
            colors = ["tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
            for i, (frac, sub_frac) in enumerate(m_rewire.groupby("frac")):
                q_mean = sub_frac.quality.mean()
                m_rewire_acc = sub_frac.groupby("n_train").acc.mean().sort_index()
                c = colors[i % len(colors)]
                ax.plot(m_rewire_acc.index, 100 * m_rewire_acc.values, "-", color=c, marker="o", alpha=0.7, label=f"KGFP (miss {frac}, q={q_mean:.2f})")
                
        for mode, lbl, c, ls in [("random", "random KG", "gray", ":"), ("permuted", "permuted KG", "tab:orange", ":")]:
            m_ctrl = sub[sub["mode"] == mode]
            if len(m_ctrl):
                m_ctrl_acc = m_ctrl.groupby("n_train").acc.mean().sort_index()
                ax.plot(m_ctrl_acc.index, 100 * m_ctrl_acc.values, ls, color=c, marker="o", alpha=0.6, label=lbl)

        ax.set_title(dgp)
        ax.set_xlabel("n_train")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("test accuracy (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("KGFP: accuracy vs n  (KG helps at small n, should vanish at large n)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=["quality", "nsweep", "both"], default="both")
    p.add_argument("--model-path", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dgps", nargs="+", default=list(DGPS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--n-train", type=int, default=40, help="fixed n for the quality sweep")
    p.add_argument("--n-trains", type=int, nargs="+", default=[40, 80, 160, 320],
                   help="n grid for the n-sweep")
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--n-features", type=int, default=120)
    p.add_argument("--n-groups", type=int, default=12)
    p.add_argument("--feature-noise", type=float, default=2.0)
    p.add_argument("--label-noise", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=1.0, help="APPNP propagated-term weight")
    p.add_argument("--K", type=int, default=1, help="propagation steps (K=1,α=1 -> X Â)")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="edge-rewire fractions for the quality sweep")
    p.add_argument("--out", default="kg_a_kgfp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    common = dict(
        n_test=args.n_test, n_features=args.n_features, n_groups=args.n_groups,
        feature_noise=args.feature_noise, label_noise=args.label_noise,
        alpha=args.alpha, K=args.K, model_path=args.model_path, device=args.device,
    )

    if args.experiment in ("quality", "both"):
        pts: list[Point] = []
        print(f"\n=== quality sweep (n_train={args.n_train}) ===")
        for dgp in args.dgps:
            for seed in args.seeds:
                pts.extend(run_condition(
                    dgp=dgp, seed=seed, n_train=args.n_train, fracs=args.fracs, **common))
        plot_quality(pts, f"{args.out}_quality.png")

    if args.experiment in ("nsweep", "both"):
        pts = []
        print("\n=== n sweep (true KG + rewire + controls) ===")
        for dgp in args.dgps:
            for n in args.n_trains:
                for seed in args.seeds:
                    pts.extend(run_condition(
                        dgp=dgp, seed=seed, n_train=n, fracs=[0.0, 0.25, 0.5], **common))
        plot_nsweep(pts, f"{args.out}_nsweep.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
