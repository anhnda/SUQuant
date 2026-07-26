"""
Signed first-order end-loss term for LNQ-F.

The GuidedQuant saliency cache stores  s = (d l / d z)^2  -- the SIGN of the
end-loss gradient w.r.t. each layer output is squared away, so it cannot supply
the linear term of the Taylor expansion. This module produces the SIGNED term
LNQ-F needs, per output channel j of every linear module:

        g_bar_j = (1/n) sum_i  (d l_i / d z_ij) * X_{i,:}      in R^{d_in}

which is exactly the per-row first-order term of GuidedQuant's Remark 3.1
(g_bar_j = mean_i (d l_i / d w_j), via the chain rule d l_i/d w_j =
(d l_i/d z_ij) X_{i,:}). Stacked over channels this is a [out, in] tensor per
module, saved one file per layer:  {gbar_path}/l{L}.pt -> {module_name: g_bar}.

It is computed with a single forward/backward pass, capturing (a) the SIGNED
output-activation gradient d l/d z via a tensor hook on each module's output and
(b) the module input X via a forward hook, then accumulating X^T (d l/d z) online
so the [N, seq, hidden] activation stream never has to be stored.

The gradient is scaled by GRAD_SCALE (1e3) to match the saliency pipeline's
convention (see gradients.py) and to avoid fp underflow; the same 1e3 is baked
into the cached saliency s = (1e3 * grad)^2 = 1e6 * grad^2, i.e. H carries a
1e6 factor. To keep the Newton shift H^{-1} g_bar dimensionally consistent, we
therefore ALSO scale g_bar by 1e6 (not 1e3): H ~ 1e6 * X^T Diag(grad^2) X while
g_bar ~ 1e6 * X^T grad, so H^{-1} g_bar has the correct 1e0 scale of the raw
weights. See the assertion in the LNQ-F README section "Scale bookkeeping".
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

import torch
from tqdm import tqdm

from any_precision.analyzer import dispatch_model

GRAD_SCALE = 1e3          # matches gradients.py (saliency uses (1e3*grad)^2)
GBAR_SCALE = GRAD_SCALE ** 2   # = 1e6, so H^{-1} g_bar lands at raw-weight scale


def get_firstorder(
    analyzer,
    input_tokens,
    gbar_path: str,
    sub_layer: Optional[Tuple[int, int]] = None,
    overwrite: bool = False,
):
    """
    Compute and cache the signed first-order term g_bar for every linear module.

    Args:
        analyzer:     Analyzer with .model, .get_layers(), .get_modules(layer).
        input_tokens: iterable of token tensors, each shape [seq_len].
        gbar_path:    output directory; writes l{L}.pt -> {module_name: [out,in]}.
        sub_layer:    optional (start, end) layer range to restrict to.
        overwrite:    recompute even if files already exist.
    """
    os.makedirs(gbar_path, exist_ok=True)

    model = analyzer.model
    if torch.cuda.device_count() > 1:
        model = dispatch_model(model)
    model = model.bfloat16()
    model.eval()
    if model.device.type != "cuda" and torch.cuda.device_count() == 1:
        model.cuda()

    layers = analyzer.get_layers()
    start_layer, end_layer = (sub_layer if sub_layer is not None else (None, None))

    def in_range(i):
        return (start_layer is None) or (start_layer <= i < end_layer)

    # If every requested file already exists and not overwriting, skip.
    if not overwrite:
        need = [i for i in range(len(layers)) if in_range(i)
                and not os.path.exists(os.path.join(gbar_path, f"l{i}.pt"))]
        if not need:
            logging.info(f"[lnqf] first-order cache already complete at {gbar_path}.")
            return

    # Online accumulators:  acc[layer][module] = sum_i grad_z_i outer X_i  -> [out, in]
    acc = [
        {name: None for name in analyzer.get_modules(layer).keys()}
        for layer in layers
    ]
    counts = [0]  # number of (token) rows accumulated, shared across modules

    fwd_hooks, grad_hooks_state = [], {}

    def make_forward_hook(li, name):
        def forward_hook(module, inp, out):
            # stash the input X for this module's current forward
            x = inp[0]
            grad_hooks_state[(li, name)] = x.detach()
            out.retain_grad()

            def grad_hook(grad_out):
                # grad_out: d l / d z, shape [bsz, seq, out_features], SIGNED
                x_local = grad_hooks_state.pop((li, name))
                b, s, out_f = grad_out.shape
                in_f = x_local.shape[-1]
                # raw signed gradient here; the full GBAR_SCALE (=1e6) factor is
                # applied ONCE at normalization below. Scaling by GRAD_SCALE here
                # too would double-count and inflate g_bar by 1e3 (blow-up bug).
                g2 = grad_out.reshape(-1, out_f).float()               # [Nrows, out]
                x2 = x_local.reshape(-1, in_f).float()                   # [Nrows, in]
                # sum_i grad_i outer x_i  ->  [out, in]
                contrib = g2.transpose(0, 1) @ x2
                a = acc[li][name]
                acc[li][name] = contrib if a is None else a + contrib
            out.register_hook(grad_hook)
        return forward_hook

    for li, layer in enumerate(layers):
        if not in_range(li):
            continue
        for name, module in analyzer.get_modules(layer).items():
            fwd_hooks.append(module.register_forward_hook(make_forward_hook(li, name)))

    row_total = 0
    for tokens in tqdm(input_tokens, desc="[lnqf] first-order pass"):
        tokens = tokens.to(model.device).unsqueeze(0)
        out = model(input_ids=tokens, labels=tokens)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        row_total += tokens.numel()

    for h in fwd_hooks:
        h.remove()
    model.cpu()

    # normalise by number of rows (the (1/n) in g_bar), keep GBAR_SCALE factor
    inv_n = GBAR_SCALE / max(row_total, 1)
    for li, layer in enumerate(layers):
        if not in_range(li):
            continue
        layer_dict = {}
        for name in analyzer.get_modules(layer).keys():
            a = acc[li][name]
            layer_dict[name] = None if a is None else (a * inv_n).cpu()
        out_file = os.path.join(gbar_path, f"l{li}.pt")
        torch.save(layer_dict, out_file)
        logging.info(f"[lnqf] saved first-order term -> {out_file}")

    logging.info("[lnqf] first-order term computation complete.")