"""Method C — KGCE: KG node embedding ADDED TO THE COLUMN TOKEN of a frozen TabPFN.

TabPFN___RAG.pdf §6 (Graphormer *centrality* encoding — a NODE-level addition, as
opposed to KGAB's edge-level attention bias §5, and KGFP's data-space smoothing §4).

Mechanism (Eq. 8). Each cell (i, j) is a token h_{i,j} ∈ R^E inside TabPFN. KGCE adds a
projected per-feature KG embedding to every column token, shared across all samples i:

        h_{i,j}  ←  h_{i,j}  +  g_ϕ(z_j)          (∀ sample i)

  * z_j ∈ R^{dKG}: the KG embedding of feature node f_j (per feature, NOT per sample).
  * g_ϕ: R^{dKG} → R^E: a small trainable adapter (MLP). Its LAST layer is zero-init, so
    g_ϕ(z_j)=0 at the start → KGCE is EXACTLY vanilla TabPFN at init. Training (§8) can
    therefore only help or stay neutral — the honesty safety net (cf. KGAB's α=0).
  * g_ϕ is the LEARNED ALIGNMENT between the KG embedding space and TabPFN's token space;
    no a-priori shared space is assumed (the paper's "align ... and add" made precise).

Unlike KGAB, KGCE has NO meaningful fixed-hyperparameter version: g_ϕ is a whole MLP that
is 0 at init, so it does nothing until trained. KGCE is therefore train-only (paper §8):
zero-init adapter, in-context NLL through the FROZEN forward, episodic (resampled C/Q).

How it is wired into the FROZEN model (no weight change)
-------------------------------------------------------
The token state inside a block is x_BRCE = (batch, rows, C feature-tokens, E). With
`_simple_inference_config` (PreprocessorConfig("none")) the d raw features are grouped
into tokens of FEATURES_PER_GROUP=3 plus ONE trailing target token, so C = ⌈d/3⌉ + 1. We
monkey-patch `TabPFNBlock.forward` to ADD a (C, E) vector to x_BRCE at the block entry,
broadcast over (batch, rows). The first G=⌈d/3⌉ rows hold g_ϕ(z) for the feature tokens;
the trailing target row is left 0 (y ∉ V → no KG vector on the label token). Whether the
vector is injected only at the INPUT layer (block 0, paper default) or at EVERY block is a
flag (`--site`). With add=None the patch is a no-op → exactly vanilla TabPFN.

KG node embeddings z (the KGE stand-in for the synthetic KG)
-----------------------------------------------------------
The paper trains a KGE model (TransE/RotatE) on the typed edges. For the synthetic
community KG we use the deterministic spectral factorisation of the normalised operator:
Â ≈ Z Zᵀ with Z = U_k √Λ_k (top-k eigenpairs). Then z_pᵀz_q ≈ Â_pq — exactly the
embedding affinity M^emb = zᵀz of Eq. 3 — so same-community features get aligned z. The
SAME corrupted graph A_obs that feeds KGFP/KGAB feeds KGCE here, and KG quality is the
same cos_F(Â_obs, Â_true) ∈ [0,1], keeping the stage-of-injection axis (§9.4) comparable.

Honesty axes (identical to KGFP / KGAB):
  * true KG (q≈1)  : z encodes the real communities -> g_ϕ can tag related columns -> helps.
  * rewire / random / permuted: z carries no / wrong structure -> trained g_ϕ stays ≈0
    (zero-init + nothing to gain) -> falls back to ≈ base. The control curves are evidence.
  * helps at SMALL n, must vanish at LARGE n (TabPFN then estimates structure itself).

Data generation + KG-quality machinery are reused from kg_a_kgfp (community/block SCM).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[0]
for p in (ROOT / "src", ROOT, ROOT / "my_kg_experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import kg_a_kgfp as A  # noqa: E402  (data SCM + graph corruption + kg_quality)

import tabpfn.architectures.tabpfn_v2_6 as v26  # noqa: E402
from tabpfn.classifier import TabPFNClassifier  # noqa: E402
from tabpfn.preprocessing import PreprocessorConfig  # noqa: E402
from tabpfn.finetuning.data_util import (  # noqa: E402
    get_preprocessed_dataset_chunks,
    meta_dataset_collator,
)

FEATURES_PER_GROUP = 3
DGPS = A.DGPS


# ---------------------------------------------------------------------------
# Monkey-patch: add a per-column KG vector to the token state (Eq. 8)
# ---------------------------------------------------------------------------
# `_KGCE["add"]` is the (C-or-G, E) vector added at the block entry, broadcast over the
# (batch, rows) dims. It MAY carry grad (it is g_ϕ(z), a leaf-dependent tensor) so that a
# loss.backward() through the frozen forward fills g_ϕ.grad. With add=None we call the
# original block forward unchanged -> exactly vanilla TabPFN (the add=None ≡ base check).

_KGCE: dict = {"add": None, "layers": None}
_ORIG_BLOCK_FWD = v26.TabPFNBlock.forward


def tag_block_layers(clf) -> int:
    """Tag each block with its index so the patch can restrict injection to chosen layers
    (e.g. {0} for input-layer-only). Returns the block count (24 for this checkpoint)."""
    blocks = clf.models_[0].blocks
    for i, blk in enumerate(blocks):
        blk._kgce_layer = i
    return len(blocks)


def _patched_block_forward(self, x_BRCE, single_eval_pos, save_peak_memory_factor):
    add = _KGCE["add"]
    if add is not None:
        li = getattr(self, "_kgce_layer", None)
        layers = _KGCE["layers"]
        if layers is None or li in layers:
            # add may be one (G,E) tensor (shared) or a list of per-layer (G,E) tensors.
            vec = add[li] if isinstance(add, (list, tuple)) else add
            C = x_BRCE.shape[2]
            G = vec.shape[0]
            # Pad to C with zeros so the trailing target token gets no KG vector.
            pad = F.pad(vec.to(x_BRCE.dtype).to(x_BRCE.device), (0, 0, 0, C - G))
            x_BRCE = x_BRCE + pad[None, None]  # broadcast over (batch, rows)
    return _ORIG_BLOCK_FWD(self, x_BRCE, single_eval_pos, save_peak_memory_factor)


v26.TabPFNBlock.forward = _patched_block_forward


# ---------------------------------------------------------------------------
# KG node embeddings z (spectral KGE) + pool to per-token
# ---------------------------------------------------------------------------

def kg_node_embeddings(A_obs: np.ndarray, dim: int) -> np.ndarray:
    """z_j = spectral factorisation of the propagation operator Â (Eq. 2): Â ≈ Z Zᵀ via the
    top-`dim` eigenpairs, Z = U_k √max(Λ_k,0). Then z_pᵀz_q ≈ Â_pq (the embedding affinity
    M^emb=zᵀz of Eq. 3), so same-community features get aligned embeddings. Deterministic
    stand-in for a trained KGE model on the synthetic KG. Returns (d, dim)."""
    Ahat = A.normalize_adj(A_obs)
    w, V = np.linalg.eigh(Ahat)                  # ascending eigenvalues
    idx = np.argsort(w)[::-1][:dim]              # top-`dim` by magnitude of eigenvalue
    Z = V[:, idx] * np.sqrt(np.clip(w[idx], 0.0, None))[None, :]
    return Z.astype(np.float32)


def pool_to_tokens(Z: np.ndarray, fpg: int = FEATURES_PER_GROUP) -> np.ndarray:
    """Pool per-feature embeddings (d, k) -> per-token embeddings (G, k), G=⌈d/fpg⌉, by
    mean over each consecutive fpg-feature block (TabPFN's token = 3 raw features)."""
    d, k = Z.shape
    G = math.ceil(d / fpg)
    out = np.zeros((G, k), dtype=np.float32)
    for g in range(G):
        out[g] = Z[g * fpg: min((g + 1) * fpg, d)].mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# The adapter g_ϕ : R^{dKG} -> R^E  (zero-init last layer => add=0 => vanilla)
# ---------------------------------------------------------------------------

class Adapter(nn.Module):
    """Small MLP aligning the KG embedding space to TabPFN's token space. Last layer is
    zero-initialised so g_ϕ(z)=0 at start (KGCE ≡ vanilla TabPFN at init)."""

    def __init__(self, d_kg: int, d_model: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_kg, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:  # (G, d_kg) -> (G, d_model)
        return self.net(Z)


# ---------------------------------------------------------------------------
# TabPFN plumbing (same 'none' preprocessing as KGAB so the 3-feature token layout holds)
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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_tabpfn(X_train, y_train, *, model_path, device, random_state) -> TabPFNClassifier:
    """Frozen sklearn inference path used for evaluation."""
    clf = TabPFNClassifier(
        n_estimators=1, model_path=model_path, device=device,
        fit_mode="fit_preprocessors", random_state=random_state,
        ignore_pretraining_limits=True, inference_config=_simple_inference_config(),
    )
    clf.fit(X_train, y_train)
    return clf


def _batched_clf(model_path, device, seed) -> TabPFNClassifier:
    """A classifier set up for the differentiable batched forward (gradient path)."""
    return TabPFNClassifier(
        n_estimators=1, model_path=model_path, device=device, random_state=seed,
        fit_mode="batched", differentiable_input=False,
        ignore_pretraining_limits=True, inference_config=_simple_inference_config(),
    )


def _episode_batch(clf, Xtr, ytr, *, seed):
    """One fresh context/query split (episodic, §8) collated for the batched forward."""
    split_fn = partial(train_test_split, test_size=0.3, random_state=seed)
    chunks = get_preprocessed_dataset_chunks(
        clf, Xtr.astype(np.float32), ytr, split_fn, max_data_size=None,
        model_type="classifier", equal_split_size=True,
        data_shuffle_seed=seed, preprocessing_random_state=seed,
    )
    return next(iter(DataLoader(chunks, batch_size=1, collate_fn=meta_dataset_collator)))


# ---------------------------------------------------------------------------
# Train the adapter g_ϕ against the in-context NLL through the frozen forward (§8)
# ---------------------------------------------------------------------------

def train_adapter(
    Xtr, ytr, Z_tok: np.ndarray, *, model_path, device, seed,
    epochs: int, lr: float, hidden: int, site: str,
) -> tuple[Adapter, list[float]]:
    """Zero-init adapter, train ONLY g_ϕ (TabPFN frozen) against in-context NLL, resampling
    the C/Q split every step. Returns (trained adapter, loss history).

      * site='input' -> inject only at block 0 (paper default, one adapter).
      * site='all'   -> inject the SAME g_ϕ(z) at every block (shared adapter).

    Because the adapter's last layer is zero-init, step 0 is exactly vanilla TabPFN, so
    training "can only help or stay neutral"."""
    clf = _batched_clf(model_path, device, seed)
    Z = torch.from_numpy(Z_tok).to(device)                    # (G, dKG) fixed
    adapter: Adapter | None = None
    opt: torch.optim.Optimizer | None = None
    n_layers = 0
    history: list[float] = []

    frozen = False
    for ep in range(epochs):
        batch = _episode_batch(clf, Xtr, ytr, seed=seed + 1000 + ep)
        clf.fit_from_preprocessed(batch.X_context, batch.y_context,
                                  batch.cat_indices, batch.configs)
        if not frozen:  # freeze TabPFN, tag blocks, build adapter now that E is known
            for prm in clf.models_[0].parameters():
                prm.requires_grad_(False)
            n_layers = tag_block_layers(clf)
            E = clf.models_[0].ninp
            adapter = Adapter(Z.shape[1], E, hidden=hidden).to(device)
            opt = torch.optim.Adam(adapter.parameters(), lr=lr)
            frozen = True

        layers = {0} if site == "input" else None
        _KGCE["add"], _KGCE["layers"] = adapter(Z), layers
        try:
            logits_QBEL = clf.forward(batch.X_query, return_raw_logits=True)
            Q, Bn, Ecls, L = logits_QBEL.shape
            logits = logits_QBEL.permute(1, 2, 3, 0).reshape(Bn * Ecls, L, Q)
            targets = batch.y_query.repeat(Bn * Ecls, 1).to(device)
            loss = F.cross_entropy(logits, targets)
        finally:
            _KGCE["add"], _KGCE["layers"] = None, None

        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return adapter, history


# ---------------------------------------------------------------------------
# Eval with the KG vector applied (frozen sklearn path)
# ---------------------------------------------------------------------------

def score_with_kgce(clf, Xte, yte, adapter: Adapter | None, Z_tok: np.ndarray | None,
                    *, site: str, device) -> float:
    """Score the frozen clf with the (detached) per-column KG vector injected."""
    if adapter is None or Z_tok is None:
        add, layers = None, None
    else:
        with torch.no_grad():
            Z = torch.from_numpy(Z_tok).to(device)
            add = adapter(Z).detach()
        layers = {0} if site == "input" else None
    _KGCE["add"], _KGCE["layers"] = add, layers
    try:
        return float(clf.score(Xte.astype(np.float32), yte))
    finally:
        _KGCE["add"], _KGCE["layers"] = None, None


# ---------------------------------------------------------------------------
# One condition: base + KGCE(true) + rewire grid + controls
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Point:
    dgp: str
    n_train: int
    seed: int
    mode: str        # base | true | rewire | random | permuted
    frac: float
    quality: float
    acc: float


def _kgce_acc(d, A_obs, *, kg_dim, model_path, device, seed, epochs, lr, hidden, site):
    """Build z from A_obs, train the adapter, score the frozen model with it injected."""
    Z_tok = pool_to_tokens(kg_node_embeddings(A_obs, kg_dim))
    adapter, _ = train_adapter(
        d.X_train, d.y_train, Z_tok, model_path=model_path, device=device, seed=seed,
        epochs=epochs, lr=lr, hidden=hidden, site=site,
    )
    eval_clf = fit_tabpfn(d.X_train.astype(np.float32), d.y_train,
                          model_path=model_path, device=device, random_state=seed)
    tag_block_layers(eval_clf)
    return score_with_kgce(eval_clf, d.X_test, d.y_test, adapter, Z_tok,
                           site=site, device=device)


def run_condition(
    *, dgp, seed, n_train, n_test, n_features, n_groups, feature_noise, label_noise,
    fracs, kg_dim, epochs, lr, hidden, site, model_path, device,
) -> list[Point]:
    set_seed(seed)
    # shuffle_features=False so consecutive 3-feature tokens stay within a community ->
    # the pooled per-token KG embedding aligns with the feature tokens.
    d = A.generate_data(
        dgp=dgp, n_train=n_train, n_test=n_test, n_features=n_features,
        n_groups=n_groups, feature_noise=feature_noise, label_noise=label_noise,
        seed=seed, shuffle_features=False,
    )
    rng = np.random.default_rng(seed + 4242)
    Ahat_true = A.normalize_adj(d.A_true)
    tab = dict(kg_dim=kg_dim, model_path=model_path, device=device, seed=seed,
               epochs=epochs, lr=lr, hidden=hidden, site=site)

    # base = adapter off (add=None) -> must equal vanilla TabPFN (the no-op check).
    eval_clf = fit_tabpfn(d.X_train.astype(np.float32), d.y_train,
                          model_path=model_path, device=device, random_state=seed)
    tag_block_layers(eval_clf)
    base = score_with_kgce(eval_clf, d.X_test, d.y_test, None, None, site=site, device=device)
    pts = [Point(dgp, n_train, seed, "base", float("nan"), float("nan"), base)]

    # Graded KG quality via edge rewiring (frac=0 is the clean/true KG).
    for f in fracs:
        A_obs = d.A_true if f == 0.0 else A.corrupt_graph(d.A_true, f, rng)
        q = A.kg_quality(A.normalize_adj(A_obs), Ahat_true)
        mode = "true" if f == 0.0 else "rewire"
        pts.append(Point(dgp, n_train, seed, mode, f, q, _kgce_acc(d, A_obs, **tab)))

    # Controls: random graph + permuted node labels -> should fall back to ≈ base.
    A_rnd = A.random_graph(d.A_true, rng)
    pts.append(Point(dgp, n_train, seed, "random", float("nan"),
                     A.kg_quality(A.normalize_adj(A_rnd), Ahat_true),
                     _kgce_acc(d, A_rnd, **tab)))
    A_prm = A.permuted_graph(d.A_true, rng)
    pts.append(Point(dgp, n_train, seed, "permuted", float("nan"),
                     A.kg_quality(A.normalize_adj(A_prm), Ahat_true),
                     _kgce_acc(d, A_prm, **tab)))

    for p in pts:
        print(f"  {dgp:11s} n={n_train:<4d} s={seed} {p.mode:8s} "
              f"q={p.quality:.3f} acc={100*p.acc:.1f}")
    return pts


# ---------------------------------------------------------------------------
# Plots (mirror kg_a_kgfp)
# ---------------------------------------------------------------------------

def plot_quality(points: list[Point], save_path: str) -> None:
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
                label="KGCE (token add)")
        ax.set_xticks(grad_acc.index)
        ax.set_xticklabels([f"{x:.2f}\n(q={grad_q[x]:.2f})" for x in grad_acc.index],
                           fontsize=8)
        for mode, c in [("random", "gray"), ("permuted", "tab:orange")]:
            m = sub[sub["mode"] == mode]
            if len(m):
                ax.axhline(100 * m.acc.mean(), ls=":", color=c, alpha=0.8, label=mode)
        ax.axhline(100 * base, ls="--", color="black", alpha=0.6, label="base")
        ax.set_title(dgp)
        ax.set_xlabel("Edge rewiring fraction (frac)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("test accuracy (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("KGCE: accuracy vs KG quality", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {save_path}")


def plot_nsweep(points: list[Point], save_path: str) -> None:
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
            ax.plot(m_base.index, 100 * m_base.values, "--", color="black",
                    marker="o", alpha=0.6, label="base")
        m_true = sub[sub["mode"] == "true"]
        if len(m_true):
            q_mean = m_true.quality.mean()
            acc = m_true.groupby("n_train").acc.mean().sort_index()
            ax.plot(acc.index, 100 * acc.values, "-", color="tab:blue", marker="o",
                    alpha=0.8, label=f"KGCE (true KG, q={q_mean:.2f})")
        m_rewire = sub[sub["mode"] == "rewire"]
        if len(m_rewire):
            colors = ["tab:green", "tab:red", "tab:purple", "tab:brown"]
            for i, (frac, sf) in enumerate(m_rewire.groupby("frac")):
                acc = sf.groupby("n_train").acc.mean().sort_index()
                ax.plot(acc.index, 100 * acc.values, "-", color=colors[i % len(colors)],
                        marker="o", alpha=0.7,
                        label=f"KGCE (miss {frac}, q={sf.quality.mean():.2f})")
        for mode, c in [("random", "gray"), ("permuted", "tab:orange")]:
            m = sub[sub["mode"] == mode]
            if len(m):
                acc = m.groupby("n_train").acc.mean().sort_index()
                ax.plot(acc.index, 100 * acc.values, ":", color=c, marker="o",
                        alpha=0.6, label=f"{mode} KG")
        ax.set_title(dgp)
        ax.set_xlabel("n_train")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("test accuracy (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("KGCE: accuracy vs n  (KG helps at small n, should vanish at large n)",
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
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-train", type=int, default=40, help="fixed n for the quality sweep")
    p.add_argument("--n-trains", type=int, nargs="+", default=[40, 80, 160, 320],
                   help="n grid for the n-sweep")
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--n-features", type=int, default=120)
    p.add_argument("--n-groups", type=int, default=12)
    p.add_argument("--feature-noise", type=float, default=2.0)
    p.add_argument("--label-noise", type=float, default=0.3)
    p.add_argument("--kg-dim", type=int, default=12, help="dim of the KG node embedding z")
    p.add_argument("--hidden", type=int, default=64, help="adapter g_ϕ hidden width")
    p.add_argument("--site", choices=["input", "all"], default="input",
                   help="inject at the input block only, or at every block (shared g_ϕ)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--fracs", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="edge-rewire fractions for the quality sweep")
    p.add_argument("--out", default="kg_c_kgce")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd
    common = dict(
        n_test=args.n_test, n_features=args.n_features, n_groups=args.n_groups,
        feature_noise=args.feature_noise, label_noise=args.label_noise,
        kg_dim=args.kg_dim, epochs=args.epochs, lr=args.lr, hidden=args.hidden,
        site=args.site, model_path=args.model_path, device=args.device,
    )

    if args.experiment in ("quality", "both"):
        pts: list[Point] = []
        print(f"\n=== KGCE quality sweep (n_train={args.n_train}, site={args.site}) ===")
        for dgp in args.dgps:
            for seed in args.seeds:
                pts.extend(run_condition(
                    dgp=dgp, seed=seed, n_train=args.n_train, fracs=args.fracs, **common))
        pd.DataFrame([dataclasses.asdict(p) for p in pts]).to_csv(
            f"{args.out}_quality.csv", index=False)
        plot_quality(pts, f"{args.out}_quality.png")

    if args.experiment in ("nsweep", "both"):
        pts = []
        print("\n=== KGCE n sweep (true KG + rewire + controls) ===")
        for dgp in args.dgps:
            for n in args.n_trains:
                for seed in args.seeds:
                    pts.extend(run_condition(
                        dgp=dgp, seed=seed, n_train=n, fracs=[0.0, 0.25, 0.5], **common))
        pd.DataFrame([dataclasses.asdict(p) for p in pts]).to_csv(
            f"{args.out}_nsweep.csv", index=False)
        plot_nsweep(pts, f"{args.out}_nsweep.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
