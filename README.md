# KG-TabPFN: Injecting Knowledge-Graph Priors into a Frozen Tabular Foundation Model

This repository is a research project exploring how **external knowledge graphs (KGs)** can be
injected into the in-context inference of a **frozen** [TabPFN](https://github.com/PriorLabs/TabPFN)
— a tabular foundation model that performs supervised learning in a single forward pass.

The motivating setting is small-sample tabular prediction (e.g. biomedical data: few patients,
many features), where a domain KG carries structure that the model cannot estimate from the
data alone: which features are related, which pairs interact, and which features are known to
be relevant to the target. The question this project asks is:

> **Where and how should a column-level KG prior enter a frozen tabular transformer, so that a
> good KG helps, and a wrong or empty KG provably does nothing?**

The main method is **KGAM** (Knowledge-Graph Attention Mixture, `my_kg_experiments/kg_e_kgam.py`),
supported by three earlier injection variants (Methods A/B/C) that map out the design space.

All methods share two design principles:

1. **Safety / no-op guarantee** — every method has a gate (a scalar, a vector, or a zero-init
   adapter) such that at gate = 0 the forward pass is **bit-exact vanilla TabPFN**. Training the
   gates can therefore only help or stay neutral.
2. **Lightness** — the transformer is never updated. Only a handful of scalar gates (KGAM: ~100
   parameters) are trained through the frozen forward pass against the in-context NLL.

---

## Main method: KGAM — a coverage-gated attention mixture

### Background: feature-token attention in TabPFN

TabPFN encodes every table row as a sequence of $C = G+1$ tokens: the $d$ feature columns are
packed, in their current column order, into $G = \lceil d/p \rceil$ feature tokens of $p$
consecutive columns each ($p=3$ in the released checkpoint), plus one **target token** that
carries the label for context rows and a placeholder for query rows. Each transformer block
attends along two axes; KGAM modifies only the **feature axis** (attention among the $C$ tokens
of one row), which is where feature–feature dependence is computed.

Two facts drive the design. First, raw attention logits carry no canonical scale — only the
softmaxed distribution $P$ is normalized — so the prior is expressed **as a distribution** and
combined with $P$, rather than added to the logits. Second, the prediction is read out from the
target token, which gives feature–label prior knowledge a natural injection route.

### KG interface: two channels per relation

The prior is a multi-relation graph over the feature columns and the target. A coverage
indicator $o_j \in \{0,1\}$ records whether column $j$ is mapped to a KG entity at all. Each
relation $r = 1,\dots,R$ supplies:

- **Feature–feature edges** $A^{(r)} \in \mathbb{R}^{d\times d}_{\ge 0}$ — e.g. similarity /
  association relations (co-expression, shared pathway) or interaction relations (epistasis,
  synthetic lethality, drug synergy). Mixed **bidirectionally** into token–token attention.
- **Feature–label relevance** $b^{(r)} \in \mathbb{R}^{d}_{\ge 0}$ — e.g. known gene–phenotype
  associations. Mixed **unidirectionally** into the target-token readout row only; the reverse
  direction (feature tokens attending to the target) is deliberately never touched, because on
  query rows the target token is an uninformative placeholder and on context rows it would open
  a label-leak shortcut that inflates context fit without generalizing.

### From KG to token-level priors: alignment, pooling, coverage

Because TabPFN packs $p$ consecutive columns into one token, the pooled prior is sharp only if
KG-related columns are adjacent. KGAM therefore clusters the **observed KG** into communities
(union-find over feature–feature edges, with label-relevance edges attached to a virtual target
node), and orders columns community-contiguously before tokenization — using only the KG, so no
label leakage.

Each relation's $d\times d$ prior is pooled to token resolution $G\times G$ by a **masked mean
over covered pairs** (or a masked **max** for edge-sparse interaction relations, where a single
strong edge would otherwise be diluted by up to $1/p^2$):

$$M^{(r)}_{ab} = \frac{\sum_{j\in S_a}\sum_{k\in S_b} o_j\, o_k\, A^{(r)}_{jk}}{\max\left(1,\ \sum_{j\in S_a}\sum_{k\in S_b} o_j\, o_k\right)}, \qquad M^{(r)}_{aa} = 0,$$

where $S_a$ is the set of columns packed into token $a$. "How much we know" is tracked
separately as per-token **coverage** $c_a = \frac{1}{|S_a|}\sum_{j\in S_a} o_j \in [0,1]$, which
gates the injection strength row-wise, so that missing KG information automatically shrinks the
prior's share toward zero.

### Prior normalization

The pooled prior must be row-stochastic like $P$. Rows are normalized by a **masked row-softmax
over the support only**, with one learnable temperature $\tau_r$ per relation; rows with empty
support are switched off entirely via a support switch $s^{(r)}_a$. This guarantees
*"no KG information ⇒ no intervention"* — a naive softmax would map an all-zero row to the
uniform distribution, injecting maximal-entropy noise exactly where the KG is silent.

### Injection: a gated convex mixture

In every block $\ell$ and every feature-axis head (gates shared across heads), the attention
distribution is replaced by a convex combination. For feature-token rows $a$:

$$P'_{a\cdot} = \Bigl(1 - \sum_r \lambda^{(\ell)}_r\, c_a\, s^{(r)}_a\Bigr) P_{a\cdot} + \sum_{r=1}^{R} \lambda^{(\ell)}_r\, c_a\, s^{(r)}_a\; M'^{(r)}_{a\cdot},$$

and analogously for the target readout row with its own gates $\lambda^{(\ell)}_{y,r}$ and the
normalized label prior $m'^{(r)}$. The head output is $P'V$ as usual.

Gates are zero-initialized and projected after every optimizer step onto
$\{\lambda \ge 0,\ \sum_r \lambda_r \le 1\}$, so every row of $P'$ remains a probability
distribution, and **all gates zero ⇒ $P' = P$ ⇒ exactly vanilla TabPFN**. With $L = 24$ blocks
and $R$ relations the trainable set is $24\cdot 2R + 2R$ scalars (e.g. $R=2$: 100 parameters).

### Training and KG-aware ensembling

Only gates and temperatures are trained, against the **in-context NLL through the frozen
forward**: each episode resamples a fresh context/query split *and* a fresh
community-preserving column permutation, so the gradient reflects generalizable structure. An
$\ell_1$ penalty biases the solution back toward the vanilla model (sharpening the safety
property); an optional total-variation penalty encourages a contiguous band of active blocks in
depth.

At inference, TabPFN's ensemble over random column permutations would scatter KG-related
columns across tokens and wash the pooled prior out toward uniform. KGAM replaces the
permutation sampler with **community-preserving permutations** (random order of community
blocks × random order within each block): every member still sees a different column layout,
but tokens stay KG-coherent and every member's pooled prior keeps its block structure. Gates
are shared across members, licensed by the construction's permutation-equivariance.

### Variant: one feature per token (fpg = 1, exact injection)

The community clustering, pooling, and permutation machinery above all exist to cope with
TabPFN packing $p=3$ columns into one token. We also implemented the alternative that removes
the problem at its root: **force one column per token** by patching the encoder's feature
grouping so that each column is presented as the $[x, 0, 0]$ padding pattern — exactly the
layout the checkpoint already saw during pretraining for tables whose width is not a multiple
of $p$, so the encoder weights stay in-distribution and no retraining is needed.

With token ≡ column, the pooling step becomes the **identity**: the $d\times d$ / $d$-dim prior
is injected *exactly*, coverage reduces to $c_a = o_j \in \{0,1\}$, and no community clustering
or community-preserving permutation is required — these arms run under plain uniform
permutations end to end. Comparing (ours − base) at $p=3$ against (ours − base) at fpg = 1
separates "KG value at token resolution" from "KG value at column resolution", and
(base at fpg = 1 − base at $p=3$) prices the tokenization itself.

The trade-off we found: fpg = 1 is essentially free on linear / additive-nonlinear / real data,
but *hurts* on interaction-style signal — the in-token encoder mixing of the $p=3$ layout does
real computational work for feature interactions, which single-column tokens give up. So exact
injection is not a free lunch; whether it pays depends on whether the task's signal lives
between columns or within a token.

### Overhead

The mixture acts on attention probabilities, so patched heads compute the $C\times C$ feature
attention explicitly: extra memory $O(BHC^2)$ per block with $C = \lceil d/p \rceil + 1$
(e.g. $d=120 \Rightarrow C = 41$) — negligible next to the $O(n^2)$ sample-axis attention. All
KG processing (clustering, pooling, normalization) happens once at preprocessing time.

---

## Repository layout

All project code lives in `my_kg_experiments/`; the rest of the repo is the upstream TabPFN
codebase it runs against.

| File | Method | Injection surface | Trainable |
|---|---|---|---|
| `kg_e_kgam.py` | **KGAM (main)** — coverage-gated attention mixture | feature-axis attention **distributions**, per block | per-block per-relation gates $\lambda, \lambda_y$ + temperatures $\tau$ (~100 scalars) |
| `kg_a_kgfp.py` | KGFP — KG feature propagation | **data space**: smooth rows along the feature graph ($X' = X\hat{A}$, APPNP), augment $[X \,\|\, X']$ | none (training-free) |
| `kg_b_kgab_train_version2.py` | KGAB — additive attention bias | feature-attention **logits**: $u + \alpha M$, with per-layer gates and a diagnostic single-layer scan | per-layer $\alpha$ (zero-init), trained through the frozen forward |
| `kg_c_kgce.py` | KGCE — column-token embedding addition | **token space**: add an aligned KG node embedding $g_\phi(z_j)$ to each column token | small zero-init adapter MLP $g_\phi$ |

Methods A/B/C are staged probes of the injection surface (data space → logits → token space);
KGAM subsumes their lessons: distributions instead of unscaled logits, coverage/support gating
instead of a single global strength, and a tokenization-aware ensemble instead of hoping the
prior survives column shuffling.

### Simulation & evaluation protocol (shared across methods)

Each script is self-contained: it generates synthetic tables from a structural causal model in
which the KG is the ground-truth column structure, then evaluates the frozen TabPFN with and
without the injected prior. Scenarios include:

- **Label-relevance DGP** — sparse relevant columns at random positions, nonlinear additive
  responses, heterogeneous weights; the KG knows only a subset of the relevant columns
  (partial coverage), optionally with false edges (robustness knob).
- **Mixed DGP** — three relation types at once: correlated pairs (denoising value), interaction
  pairs whose pairing is *invisible to data correlations* (genuinely non-data information), and
  label relevance; each channel observed only at a fraction `kg_frac`.
- **Graded KG quality** — edges rewired at fractions 0→1, quality measured as the Frobenius
  cosine between observed and true propagation operators.

Every run reports honesty controls alongside the method:

- `base` — the same pipeline with the KG switched off (holds everything but the KG fixed);
- `ours0` — community-preserving permutations with gates at zero (isolates the permutation
  sampler; also verifies the safety property empirically);
- **random / permuted KG** — an Erdős–Rényi graph of matched density, and the true graph with
  shuffled node labels: a correct method must fall back to ≈ base on both;
- `oracle` — TabPFN on the true relevant columns (upper reference).

Each condition additionally runs the fpg = 1 arms (`base_fpg1` / `vanilla_fpg1` / `ours_fpg1`,
see the single-column-token variant above), so every table reads out both the token-resolution
and the exact column-resolution value of the KG.

### Key empirical observations

- When the KG carries information the data cannot reveal at small $n$ (partial label relevance,
  interaction pairings orthogonal to the feature covariance), the trained mixture recovers a
  large fraction of the base→oracle gap, and the gain decays as $n$ grows — consistent with the
  prior-conditioning interpretation (TabPFN as an amortized posterior, the KG as extra prior
  evidence).
- Under random or permuted KGs, and with gates at zero, all methods fall back to base — the
  no-op guarantee holds both by construction and empirically.
- The injection surface matters: data-space smoothing (A) wins when the KG encodes redundancy /
  similarity (recoverable from covariance in principle); attention-level injection (B/E) is
  required when the KG encodes interactions invisible to correlations.

## Running

Requires the TabPFN package in this repo (see `pyproject.toml`; a v2.6-family checkpoint is
downloaded on first use). Each experiment script runs standalone, e.g.:

```bash
# Main method: gated attention mixture, label-relevance DGP with partial KG coverage
python my_kg_experiments/kg_e_kgam.py

# Training-free feature propagation with KG-quality and n-sweeps
python my_kg_experiments/kg_a_kgfp.py --experiment both

# Attention-bias method: per-layer scan + trained per-layer gates
python my_kg_experiments/kg_b_kgab_train_version2.py --mode both --family interaction

# Column-token embedding addition
python my_kg_experiments/kg_c_kgce.py --experiment quality
```

Each script prints per-condition accuracies and saves comparison plots (`*.png`) covering the
method, baselines, controls, and the oracle.

**Note.** The KG-injection code patches the model *in place* (monkey-patching
`AlongRowAttention.forward` / `TabPFNBlock.forward` in `tabpfn.architectures.tabpfn_v2_6`)
rather than forking the architecture, so it targets the v2.6 checkpoint family and is pinned to
this repo's TabPFN version.

## Status

Research prototype (synthetic-data validation stage; not under active development). The method
document behind KGAM is in `wileyNJD-Doc.tex`.

## Acknowledgements

Built on [TabPFN](https://github.com/PriorLabs/TabPFN) by Prior Labs — this repo is a fork; all
credit for the base model belongs to the original authors:

```bibtex
@article{hollmann2025tabpfn,
 title={Accurate predictions on small data with a tabular foundation model},
 author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and
         Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and
         Schirrmeister, Robin Tibor and Hutter, Frank},
 journal={Nature},
 year={2025},
 doi={10.1038/s41586-024-08328-6},
}
```

The TabPFN code is licensed under the Prior Labs License (see `LICENSE`); the TabPFN-2.5/2.6
model weights are under a separate
[non-commercial license](https://huggingface.co/Prior-Labs/tabpfn_2_6/blob/main/LICENSE).
