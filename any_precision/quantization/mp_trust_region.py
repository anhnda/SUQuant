"""
MP / BBP Trust-Region denoising for the saliency-weighted Hessian H = X^T diag(s) X.

Motivation (see MP_TrustRegion_Strategy):
  LNQ solves per-output-channel weighted least squares whose normal equations are
  governed by H. The bottom eigenspace of H is estimated very poorly at small
  calibration (n/d ~ 64) and gets amplified by inversion; "H is PD" does not save
  you because PD only means lambda_min > 0, not well-conditioned (cond ~ 1e6-1e7).
  The default fix is Tikhonov damping (H += lambda * mean(diag)), a blunt instrument
  that shrinks the bottom eigenspace but neither removes it nor is scale-free.

  This module replaces damping with a random-matrix-theory (Marchenko-Pastur / BBP)
  filter: whiten H, locate the bulk edge, keep only eigen-directions that stand out
  as BBP spikes (signal), shrink their biased eigenvalues, and rebuild a clean PD
  surrogate whose bottom eigenspace is floored at the bulk level instead of an
  arbitrary damping constant.

Scope note (IMPORTANT):
  The strategy doc proposes a diagonal+low-rank surrogate inverted via Woodbury,
  wired into update_C / update_P. That is a full solver rewrite. Here we take the
  conservative, isolated route requested: MP acts as a *drop-in replacement for the
  damping loop*. It returns a reconstructed, well-conditioned PD matrix H_mp of the
  SAME shape as H, so the existing Cholesky / update_C / update_P path runs
  unchanged on the denoised H. Woodbury is a later speed optimization; for H-quality
  the reconstruction is sufficient. End-loss guidance is preserved because s is
  already baked into H before this filter ever sees it.

All routines are per-group (H is (num_groups, d, d)); we filter each group's dxd
block independently.
"""

import logging
import torch


def effective_sample_size(s: torch.Tensor) -> float:
    """
    Kish effective sample size for saliency weights s (1-D over tokens):
        n_eff = (sum s)^2 / sum(s^2)
    Saliency concentrates mass on few high-gradient tokens, so n_eff << n and the
    MP aspect ratio must use n_eff, not n. Returns a float.
    """
    s = s.double().flatten()
    num = (s.sum()) ** 2
    den = (s * s).sum().clamp_min(1e-30)
    return float((num / den).item())


def _whiten(H: torch.Tensor):
    """
    D^{-1/2} H D^{-1/2} with D = diag(H). MP assumes a white bulk; LLM activations
    are not white, so we whiten first, run all spectral analysis on H_tilde, and map
    back through D^{1/2} when rebuilding. Returns (H_tilde, d_sqrt) where
    d_sqrt = sqrt(diag(H)).
    """
    d = torch.diagonal(H).clamp_min(1e-12)
    d_inv_sqrt = d.rsqrt()
    H_tilde = H * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)
    # symmetrize against fp drift
    H_tilde = 0.5 * (H_tilde + H_tilde.transpose(-2, -1))
    return H_tilde, d.sqrt()


def _mp_median(gamma: float) -> float:
    """
    Median of the standard Marchenko-Pastur distribution with ratio gamma (sigma=1),
    computed numerically. Used to de-bias the sample-median estimate of sigma^2:
    for a white bulk,  median(sample_eigs) ~= sigma^2 * mp_median(gamma),  so
    sigma^2 ~= median(sample_eigs) / mp_median(gamma). This is the standard robust
    RMT noise-level estimator (avoids the raw-median bias that makes lambda_+ too low
    and lets bulk eigenvalues leak through as false spikes).
    """
    import math
    root = gamma ** 0.5
    lam_minus = (1.0 - root) ** 2
    lam_plus = (1.0 + root) ** 2
    # MP density p(x) = sqrt((lam_plus - x)(x - lam_minus)) / (2 pi gamma x) on
    # [lam_minus, lam_plus]. Integrate CDF numerically to find the median.
    N = 20000
    xs = torch.linspace(lam_minus + 1e-9, lam_plus - 1e-9, N, dtype=torch.float64)
    dens = torch.sqrt((lam_plus - xs) * (xs - lam_minus)) / (2 * math.pi * gamma * xs)
    cdf = torch.cumsum(dens, dim=0)
    cdf = cdf / cdf[-1]
    idx = int(torch.searchsorted(cdf, torch.tensor(0.5, dtype=torch.float64)).item())
    idx = max(0, min(idx, N - 1))
    return float(xs[idx].item())


def _bulk_sigma2(eigs: torch.Tensor, gamma_eff: float = None) -> float:
    """
    Estimate the bulk noise level sigma^2 of the whitened spectrum.

    Whitening by diag(H) forces trace(H_tilde)/d = 1 exactly, hence mean(eig) = 1.
    Empirically (see diagnostics) this anchors the bulk so that the MP edge
    lambda_+ = sigma^2 (1+sqrt(gamma))^2 with sigma^2 = mean(eig) cleanly separates
    the true spikes. We therefore use the trace estimator directly. A single
    spike-trace subtraction is applied ONLY if it does not push sigma^2 down (guards
    against the self-reinforcing collapse where an over-count of spikes shrinks
    sigma^2 and thus over-counts further).
    """
    ev = eigs.clamp_min(0.0).double()
    d = ev.numel()
    sigma2 = float(ev.mean().item())        # trace/d = 1 after whitening
    if gamma_eff is None:
        return max(sigma2, 1e-8)
    g = min(max(gamma_eff, 1e-6), 0.999999)
    # One conservative correction: subtract only the clearly-separated top spikes
    # (eigenvalues beyond 1.5x the nominal edge), never letting sigma^2 decrease.
    lam_plus0 = sigma2 * (1.0 + g ** 0.5) ** 2
    strong = ev[ev > 1.5 * lam_plus0]
    k = strong.numel()
    if 0 < k < d:
        bulk_trace = float(ev.sum().item()) - float(strong.sum().item())
        cand = bulk_trace / (d - k)
        # keep sigma^2 anchored near 1: only accept if it stays within [0.5, 1.5]x
        if 0.5 * sigma2 <= cand <= 1.5 * sigma2:
            sigma2 = cand
    return max(sigma2, 1e-8)


def mp_edges(gamma_eff: float, sigma2: float):
    """Marchenko-Pastur bulk edges lambda_+/-  = sigma^2 (1 +/- sqrt(gamma_eff))^2."""
    root = gamma_eff ** 0.5
    lam_plus = sigma2 * (1.0 + root) ** 2
    lam_minus = sigma2 * (1.0 - root) ** 2
    return lam_minus, lam_plus


def nonlinear_shrinkage_eigs(evals: torch.Tensor, gamma_eff: float, sigma2: float):
    """
    Ledoit-Wolf-style nonlinear shrinkage of sample eigenvalues (whitened units).

    For a continuous (non-spiked) spectrum there is no threshold to cut at; instead
    every sample eigenvalue lambda is a biased estimate of the true one and must be
    shrunk toward the bulk by an amount that depends on where it sits. The optimal
    (minimum-Frobenius-loss) shrinkage replaces lambda_i by

        d_i = lambda_i / |1 - gamma - gamma * lambda_i * m(lambda_i)|^2

    where m(x) is the Stieltjes transform of the limiting sample spectral
    distribution evaluated just above the real axis. Large, well-estimated
    eigenvalues are barely touched; small, noise-dominated ones are pulled up
    strongly toward the bulk -- which is exactly the "cut the noise without a hard
    threshold" behaviour we want, and it collapses the 1/lambda_min blow-up that
    makes H^{-1} noisy.

    We estimate m(x) empirically from the sample eigenvalues themselves (the sample
    spectral distribution is the empirical measure of {evals}), regularized off the
    real axis by eta ~ sample spacing. This is the standard QuEST/analytical-NLShrink
    approximation and needs no spike/threshold assumption.

    Args:
        evals:     sample eigenvalues (whitened), shape [d], >= 0.
        gamma_eff: aspect ratio d / n_eff.
        sigma2:    bulk level (whitened ~1); used only as a scale for the imaginary
                   regularizer.
    Returns:
        shrunk eigenvalues, shape [d], all > 0.
    """
    lam = evals.double().clamp_min(0.0)
    d = lam.numel()
    gamma = min(max(gamma_eff, 1e-6), 0.999999)

    # Imaginary regularizer: tie to the average eigenvalue gap so m(x) is smooth.
    # eta larger -> smoother/more conservative shrinkage.
    span = (lam.max() - lam.min()).clamp_min(1e-8)
    eta = (span / max(d, 1)) * (d ** 0.5)          # ~ sqrt(d) * mean spacing
    eta = eta.clamp_min(1e-6 * sigma2)

    # Empirical Stieltjes transform of the sample spectrum at z = lam_i + i*eta:
    #   m(z) = (1/d) sum_j 1 / (lam_j - z)
    z_re = lam.unsqueeze(1)                          # [d,1]
    diff_re = lam.unsqueeze(0) - z_re               # [d,d] real part (lam_j - lam_i)
    denom = diff_re * diff_re + eta * eta           # |lam_j - z|^2
    m_re = (diff_re / denom).mean(dim=1)            # Re m(z_i)
    m_im = (eta / denom).mean(dim=1)                # Im m(z_i)  (> 0)

    # d_i = lam_i / |1 - gamma - gamma*lam_i*m(z_i)|^2
    a = 1.0 - gamma - gamma * lam * m_re            # real part of (1 - g - g*lam*m)
    b = -gamma * lam * m_im                          # imag part
    denom2 = (a * a + b * b).clamp_min(1e-12)
    shrunk = lam / denom2

    # Never let shrinkage push an eigenvalue below the bulk floor (keeps PD and
    # avoids re-introducing tiny eigenvalues) nor above its original value.
    shrunk = torch.clamp(shrunk, min=0.0)
    shrunk = torch.minimum(shrunk, lam.clamp_min(0.0) + sigma2)  # mild upper guard
    floor = sigma2 * (1.0 - gamma ** 0.5) ** 2 if gamma < 1.0 else sigma2 * 1e-2
    shrunk = shrunk.clamp_min(max(floor, 1e-8))
    return shrunk.to(evals.dtype)


def _shrink_spike_eigs(lam_hat: torch.Tensor, gamma_eff: float, sigma2: float):
    """
    BBP eigenvalue de-biasing. A sampled spike eigenvalue lam_hat (whitened, in units
    of sigma^2) is inflated relative to the true spike theta. Invert the BBP map
        lam_hat/sigma^2 = (1+theta_r)(1 + gamma/theta_r),   theta_r = theta/sigma^2
    for theta_r (positive root of the quadratic), and return the de-biased eigenvalue
    theta = sigma^2 * theta_r. Directions that are not confidently above the spike
    threshold are left at the bulk level (no negative/again-inflated values).
    """
    l = (lam_hat / sigma2).double()
    # Solve theta_r^2 + (1 + gamma - l) theta_r + gamma = 0  -> positive root.
    b = (1.0 + gamma_eff - l)
    disc = (b * b - 4.0 * gamma_eff).clamp_min(0.0)
    theta_r = (-b + disc.sqrt()) / 2.0
    theta_r = theta_r.clamp_min(0.0)
    return (sigma2 * theta_r).to(lam_hat.dtype)


def denoise_hessian_group(
    H_g: torch.Tensor,
    gamma_eff: float,
    k_max: int = 256,
    mode: str = "shrinkage",
    verbose_tag: str = "",
):
    """
    RMT denoise of a single dxd PSD block. Two modes:

      mode="shrinkage" (DEFAULT, for continuous/non-spiked spectra):
        Whiten, eig, then apply nonlinear (Ledoit-Wolf) shrinkage to EVERY
        eigenvalue -- large ones barely move, small noise-dominated ones are pulled
        up toward the bulk. No threshold, no rank cut. This is the right tool when
        the spectrum has no clean bulk/spike gap (the measured LLM case): it removes
        noise everywhere it lives (worst at the tail) and collapses the 1/lambda_min
        blow-up, without discarding any direction.

      mode="threshold" (BBP spike model, for spectra that DO separate):
        Keep only eigenvalues above the MP edge lambda_+ as de-biased spikes, floor
        the rest at the bulk sigma^2. Use only if measure_mp shows a clear gap.

    Rebuild:  H_tilde_new = V diag(d_i) V^T   (shrinkage)  or  the spike surrogate,
    then map back  H_new = D^{1/2} H_tilde_new D^{1/2}.

    Returns (H_new, info). H_new is PD by construction, same shape/dtype/device.
    """
    d = H_g.shape[0]
    dtype_in, device = H_g.dtype, H_g.device
    Hf = H_g.double()

    H_tilde, d_sqrt = _whiten(Hf)
    evals, evecs = torch.linalg.eigh(H_tilde)          # ascending
    evals = evals.clamp_min(0.0)
    cond_before = float((evals.max() / evals.clamp_min(1e-12).min()).item())

    sigma2 = _bulk_sigma2(evals, gamma_eff)
    lam_minus, lam_plus = mp_edges(gamma_eff, sigma2)
    n_spike = int((evals > lam_plus).sum().item())

    if mode == "threshold":
        # --- BBP spike model: hard keep/discard (requires a spectral gap). ---
        spike_mask = evals > lam_plus
        k = int(spike_mask.sum().item())
        if k > k_max:
            idx_sorted = torch.argsort(evals, descending=True)[:k_max]
            spike_mask = torch.zeros_like(spike_mask)
            spike_mask[idx_sorted] = True
            k = k_max
        theta = _shrink_spike_eigs(evals[spike_mask], gamma_eff, sigma2)
        V = evecs[:, spike_mask]
        delta = (theta - sigma2).clamp_min(0.0)
        H_tilde_new = torch.eye(d, dtype=torch.float64, device=device) * sigma2
        if k > 0:
            H_tilde_new = H_tilde_new + (V * delta.unsqueeze(0)) @ V.transpose(-2, -1)
        new_eigs = torch.full((d,), sigma2, dtype=torch.float64, device=device)
        new_eigs[-k:] = theta if k > 0 else new_eigs[-k:]
    else:
        # --- Nonlinear shrinkage: continuous shrink of the whole spectrum. ---
        new_eigs = nonlinear_shrinkage_eigs(evals, gamma_eff, sigma2)
        # Rebuild from the SAME eigenvectors with shrunk eigenvalues.
        H_tilde_new = (evecs * new_eigs.unsqueeze(0)) @ evecs.transpose(-2, -1)

    # Map back through D^{1/2}.
    H_new = H_tilde_new * d_sqrt.unsqueeze(0) * d_sqrt.unsqueeze(1)
    H_new = 0.5 * (H_new + H_new.transpose(-2, -1))

    cond_after = float((new_eigs.max() / new_eigs.clamp_min(1e-12).min()).item())
    info = {
        "d": d, "gamma_eff": gamma_eff, "sigma2": sigma2, "lambda_plus": lam_plus,
        "n_spike": n_spike, "mode": mode,
        "cond_before": cond_before, "cond_after": cond_after,
    }
    if verbose_tag:
        logging.info(
            f"[MP-TR {verbose_tag}] mode={mode} d={d} gamma_eff={gamma_eff:.4f} "
            f"sigma^2={sigma2:.3e} spikes>{lam_plus:.2f}={n_spike} "
            f"cond {cond_before:.2e} -> {cond_after:.2e}"
        )
    return H_new.to(dtype_in), info


def denoise_hessian(
    H: torch.Tensor,
    n_eff,
    k_max: int = 256,
    mode: str = "shrinkage",
    verbose: bool = True,
):
    """
    Apply RMT denoise to every group block of H (num_groups, d, d).

    Args:
      H:      (num_groups, d, d) saliency-weighted Hessian on GPU.
      n_eff:  effective sample size. Either a scalar (shared) or a per-group list/
              tensor of length num_groups. gamma_eff = d / n_eff per group.
      k_max:  cap on kept eigen-directions per group (threshold mode only).
      mode:   "shrinkage" (default, continuous) or "threshold" (BBP spike).
    Returns:
      H_new:  denoised, well-conditioned PD Hessian, same shape as H.
    """
    num_groups, d, _ = H.shape
    if not torch.is_tensor(n_eff):
        if isinstance(n_eff, (list, tuple)):
            n_eff_list = [float(x) for x in n_eff]
        else:
            n_eff_list = [float(n_eff)] * num_groups
    else:
        n_eff_list = [float(x) for x in n_eff.flatten().tolist()]

    # Reconcile length with num_groups. Saliency n_eff was computed at the ORIGINAL
    # grouping; if H was re-grouped to fewer groups, average the n_eff entries that
    # fall into each target group (harmonic-style average is closer to how sample
    # sizes combine, but arithmetic mean is fine here as a proxy).
    if len(n_eff_list) != num_groups:
        if len(n_eff_list) == 1:
            n_eff_list = n_eff_list * num_groups
        elif len(n_eff_list) % num_groups == 0:
            per = len(n_eff_list) // num_groups
            n_eff_list = [
                sum(n_eff_list[i * per:(i + 1) * per]) / per
                for i in range(num_groups)
            ]
        else:
            # Fallback: use the mean for all groups.
            m = sum(n_eff_list) / len(n_eff_list)
            n_eff_list = [m] * num_groups

    out = torch.empty_like(H)
    for i in range(num_groups):
        ne = max(n_eff_list[i], 1.0)
        gamma_eff = d / ne
        out[i], _ = denoise_hessian_group(
            H[i], gamma_eff, k_max=k_max, mode=mode,
            verbose_tag=(f"g{i}" if verbose else ""),
        )
    return out