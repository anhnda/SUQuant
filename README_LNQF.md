<h1 align="center">LNQ-F</h1>

<p align="center"><b>LNQ with the first-order end-loss term retained.</b><br>
An extension of <a href="https://arxiv.org/abs/2505.07004">GuidedQuant</a>'s LNQ solver.</p>

---

## What this is

GuidedQuant / LNQ quantize each layer by minimizing the **second-order** term of
the Taylor expansion of the change in end loss,

$$\Delta\ell \;\approx\; \tfrac{1}{2}\,(\hat w - w)^\top H\,(\hat w - w),
\qquad H = X^\top \mathrm{Diag}(s)\,X, \tag{B2}$$

where $s = (\partial\ell/\partial z)^2$ is the squared, group-averaged end-loss
gradient w.r.t. the layer output. The **first-order** term
$\bar g^\top(\hat w - w)$ is dropped under the assumption that the pretrained
model has converged, so the *mean* gradient $\bar g = \tfrac1n\sum_i \nabla\ell_i
\approx 0$.

That assumption is only approximately true on a small, distribution-shifted
calibration set. The residual first-order term is exactly the part that
end-to-end fine-tuning is later observed to recover — GuidedQuant's own Table 15
shows the LNQ-vs-baseline gap **narrowing at higher bit-width**, the regime where
$\|\delta\|$ is small and the linear term is no longer dominated by the quadratic.

**LNQ-F keeps the first-order term:**

$$\phi(\hat w) \;=\; \bar g^\top(\hat w - w) \;+\; \tfrac{1}{2}\,(\hat w - w)^\top H\,(\hat w - w). \tag{B1+B2}$$

## The one idea: it is a target shift, not a new solver

$\phi$ is a quadratic in $\hat w$ with the **same curvature $H$**. Completing the
square,

$$\phi(\hat w) \;=\; \tfrac{1}{2}\,(\hat w - \tilde w)^\top H\,(\hat w - \tilde w) + \text{const},
\qquad \boxed{\;\tilde w = w - H^{-1}\bar g\;}$$

So retaining the first-order term is **exactly** the published B2 objective with
the reconstruction target moved from $w$ to $\tilde w$ — one Newton step against
$H$. Every piece of the LNQ machine is reused **verbatim** on $\tilde w$:

- the Cholesky factorization of $H$,
- the closed-form codebook solve (Eq. 9),
- the cyclic-CD assignment update (Eq. 11),
- the Prop. 4.1 monotone-descent / convergence guarantee.

Nothing inside the solver changes. Only the target it fits.

> [!NOTE]
> The $H^{-1}$ is **not** introduced by the first-order term itself — the raw B1
> term $\bar g^\top u$ has no inverse in it. $H^{-1}$ appears only because
> *completing the square* (finding the new parabola's vertex) requires solving
> $H u_0 = -\bar g$. It is the vertex of the tilted parabola, i.e. a Newton step.

## Damping: why $H^{-1}\bar g$ is not used raw

$H = X^\top\mathrm{Diag}(s)X$ is typically ill-conditioned ($\mathrm{Diag}(s)$
suppresses directions where the loss is flat), and $\bar g$ is a noisy average
over a small calibration set. A raw $H^{-1}\bar g$ amplifies that noise along the
small-eigenvalue directions and can push $\tilde w$ far from $w$ — precisely
where the second-order Taylor model stops being accurate. LNQ-F therefore uses a
**damped (Levenberg-style) Newton step**

$$\tilde w = w - \big(H + \mu\,\overline{\mathrm{diag}}(H)\,I\big)^{-1}\bar g,$$

controlled by `MU`:

| `MU` | shift | behaviour |
|------|-------|-----------|
| large (e.g. `1e6`) | $\tilde w \to w$ | **degenerates to plain LNQ (B2 only)** |
| moderate (e.g. `1.0`) | partial Newton step | first-order correction, damped |
| small (e.g. `1e-1`) | near-full Newton step | strongest B1 correction |

`MU` is thus a **continuous knob** between the published LNQ and first+second
order. `MAX_SHIFT` additionally caps the per-row $L_2$ norm of the shift
$\lVert\tilde w - w\rVert$ (a trust region), so a bad row cannot be dragged out
of the region where B2 is valid. `MAX_SHIFT=0` disables the cap.

## Why the signed gradient has to be recomputed

The cached GuidedQuant saliency is $s = (\partial\ell/\partial z)^2$ — the **sign
is squared away**, so it cannot supply $\bar g$. Per output channel $j$,

$$\bar g_j = \frac1n\sum_i \frac{\partial\ell_i}{\partial z_{ij}}\,X_{i,:} \in \mathbb{R}^{d_{in}}$$

(the chain-rule form of Remark 3.1, $\partial\ell_i/\partial w_j =
(\partial\ell_i/\partial z_{ij})X_{i,:}$). LNQ-F computes this with **one extra
forward/backward pass** (`firstorder_cache.py`), capturing the *signed* output
gradient and contracting it with the inputs online. When no first-order cache is
found, LNQ-F **falls back to plain LNQ and says so in the log** — it never
silently fabricates a shift.

### Scale bookkeeping (read before trusting a number)

The saliency pipeline scales gradients by `1e3` and stores the square, so
$H \sim 10^6\,X^\top\mathrm{Diag}(\text{grad}^2)X$. For $H^{-1}\bar g$ to land at
the raw-weight scale, $\bar g$ must carry the **same $10^6$ factor**
(`GBAR_SCALE = 1e6` in `gradients_firstorder.py`), *not* $10^3$. The `MU=1e6`
sanity check below is what verifies this end to end.

---

## Quick start

```bash
MODEL=meta-llama/Llama-2-7b-hf
BITS=3
G=1                      # GuidedQuant output-channel group count

# 1. SqueezeLLM init -- REQUIRED (same as every other solver).
bash scripts/run_sqllm.sh $MODEL $BITS $G

# 2. GuidedQuant Hessians -- solver-independent, cache once and share.
bash scripts/run_lnq.sh $MODEL $BITS $G -m hessians

# 3. LNQ-F. Builds the signed first-order cache automatically, then quantizes.
bash scripts/run_lnqf.sh $MODEL $BITS $G
```

Positional args and env overrides are identical to `run_lnq.sh`
(`DATASET`, `SEQ_LEN`, `NUM_EXAMPLES`, `NUM_ITER`, `CD_CYCLES`), plus:

| env var | default | meaning |
|---------|---------|---------|
| `MU` | `1.0` | damping on the Newton target shift (large → plain LNQ) |
| `MAX_SHIFT` | `0.0` | per-row $L_2$ trust-region cap on the shift (0 = off) |
| `SKIP_FIRSTORDER` | `0` | reuse an existing first-order cache without rebuilding |

Sweep example:

```bash
for MU in 1e6 10 1 0.1; do
  MU=$MU bash scripts/run_lnqf.sh $MODEL $BITS $G
done
```

## Validating a run

The ablation ladder is **not optional**. Run the top rung first.

| rung | command | expectation |
|------|---------|-------------|
| **sanity** | `MU=1e6 bash scripts/run_lnqf.sh …` | must reproduce `run_lnq.sh` perplexity to within noise. If it does not, the first-order path (most likely the `1e6` scale factor) is wired wrong — stop and fix it. |
| **effect** | `MU=1 …`, `MU=0.1 …` | the shift $\lVert\tilde w - w\rVert$ (logged per module) grows as `MU` shrinks; watch whether perplexity improves or degrades. |
| **baseline** | `bash scripts/run_lnq.sh …` | the number LNQ-F must beat. |

Per-module log lines to look for:

```
[lnqf] first-order target shift active | mu=1 max_shift=0 | ||w_tilde - w|| = 3.2e-02 (0.41% of ||w||)
```

or, when the cache is absent:

```
[lnqf] no first-order term supplied (g_bar=None) -> falling back to plain LNQ (second-order only).
```

**Where LNQ-F is expected to help, and where it provably will not.** Its entire
gain rides on $\bar g$ being non-negligible. If the model is a base checkpoint on
in-distribution calibration data, $\bar g \approx 0$, $\tilde w \approx w$, and
LNQ-F correctly reduces to LNQ. The regime to test is **high bit-width**
(3–4-bit, where the quadratic no longer dominates the linear term) and
**fine-tuned / distribution-shifted** checkpoints (where "converged ⇒ $\bar g
\approx 0$" is false). This mirrors exactly the Table 15 regime where LNQ's edge
narrows.

---

## Where the code lives

| file | role |
|------|------|
| `scripts/run_lnqf.sh` | driver: builds the first-order cache, then runs `--solver lnqf` |
| `firstorder_cache.py` | one-shot signed first-order pass → `cache/firstorder/…/l{L}.pt` |
| `any_precision/quantization/gradients_firstorder.py` | `get_firstorder` — signed $\bar g$ accumulation |
| `any_precision/quantization/layerwise_lnqf.py` | `train_least_squares_firstorder` — damped target shift, then LNQ |
| `any_precision/quantization/layerwise_quantize.py` | `lnqf` dispatch branch + `_load_firstorder_term` |
| `layerwise_nuq.py` | `--solver lnqf` and the `--lnqf_*` knobs |

## Cache paths

```
cache/firstorder/<model_name>-<dataset>_s<N>_blk<seq>/l{L}.pt   <- signed g_bar (LNQ-F only)
cache/hessians/<model_name>-...-g<G>/l{L}.pt                    <- H (shared with LNQ)
cache/layerwise_quantized/<model_name>-w<bits>-...-g<G>_iter<I>_cd<C>_lnqf  <- LNQ-F output
cache/layerwise_packed/layerwise-<model_name>-...-_lnqf                     <- packed LNQ-F
```

The `_lnqf` suffix keeps LNQ-F outputs from colliding with plain LNQ, so both can
sit in the same cache. Evaluate exactly as for LNQ (`eval_ppl.py` /
`run_eval.py`), pointing at the `…_lnqf` packed directory.
