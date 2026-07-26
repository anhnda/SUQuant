import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from typing import Optional, Tuple
from .config import *
from any_precision.analyzer import dispatch_model


def _mc_fisher_loss(logits, mc_samples: int, generator: Optional[torch.Generator] = None):
    """
    Monte-Carlo GGN/Fisher pseudo-loss for one sequence.

    Instead of the true-label cross-entropy (which yields the *empirical* Fisher,
    anchored to the calibration labels), we draw pseudo-labels from the model's own
    predictive distribution  y~ ~ softmax(logits)  and return the mean-over-tokens
    negative log-likelihood of those samples. The backward of this loss produces
    output-gradients delta = p - onehot(y~) whose outer product is an unbiased
    estimator of the per-logit Hessian  C_t = diag(p_t) - p_t p_t^T  (the GGN),
    matching the predictive distribution rather than the observed label.

    Scale note: HuggingFace's outputs.loss is a MEAN over valid tokens. To keep the
    resulting saliency H on the SAME scale as the true-label path (so the downstream
    1e3/1e6 conventions and LNQ damping are unchanged), this loss is ALSO a mean over
    tokens. Averaging over mc_samples draws reduces the single-sample variance without
    changing the scale (it stays a mean, not a sum).

    Args:
        logits:     [1, seq, vocab] model logits (float upcast done internally).
        mc_samples: number of pseudo-label draws K to average over (K>=1).
        generator:  optional torch.Generator for reproducible sampling.
    Returns:
        scalar loss tensor with grad flowing back into `logits`.
    """
    # logits: [1, seq, vocab] -> flatten tokens
    flat = logits.reshape(-1, logits.shape[-1])                 # [T, V]
    logp = F.log_softmax(flat.float(), dim=-1)                  # [T, V]
    probs = logp.exp()                                          # [T, V], = softmax
    T = flat.shape[0]

    # Draw K pseudo-labels per token from p_t and average their NLL. Sampling is
    # done under no_grad (it must not be differentiated); grad flows only through
    # the gathered log-probs of the sampled classes.
    with torch.no_grad():
        # multinomial over [T, V]; num_samples=K gives [T, K]
        y_tilde = torch.multinomial(probs, num_samples=mc_samples,
                                     replacement=True, generator=generator)  # [T, K]

    # gather log p_t[y~] for each draw, mean over tokens and over K draws
    nll = -logp.gather(dim=-1, index=y_tilde)                   # [T, K]
    return nll.mean()


def get_gradients(
        analyzer,
        input_tokens,
        save_path: Optional[str] = None,
        saliency_path: Optional[str] = None,
        num_groups: Optional[int] = None,
        sub_saliency: Optional[Tuple[int, int]] = None,
        skip_save_gradients: bool = False,
        mc_fisher: Optional[bool] = None,
        mc_samples: Optional[int] = None,
        mc_seed: Optional[int] = None,
):
    """
    Calculates weight gradients for the given input tokens. Optionally also calculates
    'saliency' (mean absolute gradient w.r.t. each module's output activations, grouped
    by channel) if 'saliency_path' is provided. In that case, we save one file per layer
    under 'saliency_path' directory (e.g., l0.pt, l1.pt, ...).

    If 'sub_saliency' is provided (e.g. (start_layer, end_layer)), we only attach saliency
    hooks (and save files) for layers in [start_layer, end_layer). Layers outside that
    range won't generate saliency data or files.

    Args:
        analyzer:        Analyzer object with `.model`, `.get_layers()`, `.get_modules(layer)`.
        input_tokens:    Collection of token tensors, each shape [seq_len].
        save_path:       Path to save the final weight gradients (list of dicts).
                         If the file already exists, user is prompted before overwriting.
        saliency_path:   Directory in which to save the saliency files (one file per layer).
                         If None, no saliency is computed/saved.
        num_groups:      Number of groups to chunk the channel dimension for saliency.
                         E.g. if hidden_dim=4096 and num_groups=4, each group has 1024 channels.
        sub_saliency:    (start_layer, end_layer). If provided, only layers in that range
                         will collect saliency. Otherwise, collect for all layers.

    Returns:
        gradients (list of dict): The list of per-layer, per-module weight gradients.
    """

    # ----------------------------------------------------------------
    # 0) Resolve Hessian-estimator mode (default: true-label empirical Fisher)
    # ----------------------------------------------------------------
    # The estimator can be switched to Monte-Carlo GGN/Fisher (pseudo-labels drawn
    # from the model's own predictive distribution) either via the explicit arg or,
    # to avoid threading a flag through every entrypoint's argparse, via env vars:
    #     AP_MC_FISHER=1        enable MC pseudo-label estimator
    #     AP_MC_SAMPLES=<int>   number of pseudo-label draws K (default 1)
    #     AP_MC_SEED=<int>      RNG seed for reproducible sampling (optional)
    # Leaving all of these unset reproduces the original true-label path byte-for-byte.
    if mc_fisher is None:
        mc_fisher = os.environ.get("AP_MC_FISHER", "0") not in ("0", "", "false", "False")
    if mc_samples is None:
        mc_samples = int(os.environ.get("AP_MC_SAMPLES", "1"))
    if mc_seed is None and os.environ.get("AP_MC_SEED") is not None:
        mc_seed = int(os.environ["AP_MC_SEED"])
    mc_samples = max(1, int(mc_samples))

    if mc_fisher:
        logging.info(f"[hessian-estimator] Monte-Carlo GGN/Fisher "
                     f"(pseudo-label, K={mc_samples}, seed={mc_seed}).")
    else:
        logging.info("[hessian-estimator] true-label empirical Fisher (default).")

    # ----------------------------------------------------------------
    # 1) Possibly load from cache (gradients only)
    # ----------------------------------------------------------------
    # The early-out below must NOT fire when saliency is requested but not yet
    # complete on disk. The weight-gradient cache (save_path) is estimator-
    # AGNOSTIC (no _mc suffix), so a stale Fisher gradient file would otherwise
    # short-circuit an MC run and silently skip saliency generation entirely.
    # Only skip work when the gradient cache exists AND either no saliency was
    # asked for, or every saliency file already exists.
    def _saliency_complete():
        if saliency_path is None:
            return True
        layer_ids = list(analyzer.get_layers())
        n_layers = len(layer_ids)
        if sub_saliency is not None:
            s, e = sub_saliency
            needed = range(s, e)
        else:
            needed = range(n_layers)
        return all(os.path.isfile(os.path.join(saliency_path, f"l{i}.pt"))
                   for i in needed)

    if save_path is not None and os.path.isfile(save_path) and _saliency_complete():
        logging.info(f"Gradients already calculated and saved at {save_path}.")
        if saliency_path is not None:
            logging.info(f"Saliency cache also present at {saliency_path}.")
        logging.info(f"Loading cached gradients...")
        return torch.load(save_path)

    if save_path is not None and os.path.isfile(save_path) and saliency_path is not None:
        logging.info(f"Weight-gradient cache exists at {save_path}, but saliency "
                     f"cache {saliency_path} is incomplete -> running the pass to "
                     f"generate saliency (estimator: "
                     f"{'MC' if os.environ.get('AP_MC_FISHER','0') not in ('0','','false','False') else 'set below'}).")

    # ----------------------------------------------------------------
    # 2) Prepare model
    # ----------------------------------------------------------------
    model = analyzer.model
    if torch.cuda.device_count() > 1:
        model = dispatch_model(model)

    model = model.bfloat16()
    model.eval()

    if model.device.type != 'cuda' and torch.cuda.device_count() == 1:
        model.cuda()

    layers = analyzer.get_layers()

    # If sub_saliency is given, parse it
    # We'll use these to decide whether to hook/save a given layer
    if sub_saliency is not None:
        start_layer, end_layer = sub_saliency
    else:
        start_layer, end_layer = (None, None)

    # ----------------------------------------------------------------
    # 3) If we want saliency, set up forward hooks
    # ----------------------------------------------------------------
    # We'll store a list-of-dicts parallel to `layers`:
    #   saliency_data[i_layer][module_name] = list of [bsz, seq_len, num_groups]
    saliency_data = None
    saliency_hooks = []

    if saliency_path is not None:
        # We'll store chunk-lists for all layers, but only fill them
        # for the sub_saliency range
        saliency_data = [
            {module_name: [] for module_name in analyzer.get_modules(layer).keys()}
            for layer in layers
        ]

        def make_forward_hook(layer_idx, module_name):
            def forward_hook(module, inp, out):
                # We'll store gradient on 'out', so we must retain it
                out.retain_grad()

                def grad_hook(grad):
                    """
                    grad shape typically [bsz, seq_len, hidden_dim].
                    We group the channels, take abs, then average.
                    """
                    bsz, seq_len, hidden_dim = grad.shape
                    group_size = hidden_dim // num_groups

                    grad_squared = (grad.float() * 1e3).pow(2).view(bsz, seq_len, num_groups, group_size)
                    mean_squared_grad = grad_squared.mean(dim=-1)  # -> [bsz, seq_len, num_groups]

                    # Move to CPU and store
                    saliency_data[layer_idx][module_name].append(mean_squared_grad.bfloat16().cpu())

                # Attach the gradient hook to 'out'
                out.register_hook(grad_hook)
            return forward_hook

        # Attach hooks only for layers in [start_layer, end_layer) if set
        for layer_idx, layer in enumerate(layers):
            if (start_layer is not None) and (end_layer is not None):
                if not (start_layer <= layer_idx < end_layer):
                    # skip hooking this layer
                    continue

            # Register forward hooks for each module
            for module_name, module in analyzer.get_modules(layer).items():
                h = module.register_forward_hook(make_forward_hook(layer_idx, module_name))
                saliency_hooks.append(h)

    # ----------------------------------------------------------------
    # 4) Weight-gradient hook (square_grad_hook)
    # ----------------------------------------------------------------
    def square_grad_hook(grad):
        return grad.pow(2)

    weight_hooks = []
    for layer_idx in layers:
        for module in analyzer.get_modules(layer_idx).values():
            weight_hooks.append(module.weight.register_hook(square_grad_hook))

    # ----------------------------------------------------------------
    # 5) Forward/backward pass over data
    # ----------------------------------------------------------------
    # Optional reproducible RNG for MC pseudo-label sampling, placed on the model device.
    mc_generator = None
    if mc_fisher and mc_seed is not None:
        mc_generator = torch.Generator(device=model.device)
        mc_generator.manual_seed(int(mc_seed))

    for tokens in tqdm(input_tokens, desc="Calculating gradients"):
        tokens = tokens.to(model.device).unsqueeze(0)
        if mc_fisher:
            # MC GGN/Fisher: no label needed; sample pseudo-labels from the model.
            outputs = model(input_ids=tokens)
            loss = _mc_fisher_loss(outputs.logits, mc_samples, generator=mc_generator)
        else:
            # Default: true-label empirical Fisher (HF mean-over-tokens CE).
            outputs = model(input_ids=tokens, labels=tokens)
            loss = outputs.loss
        loss.backward()

    # ----------------------------------------------------------------
    # 6) Remove hooks
    # ----------------------------------------------------------------
    for h in weight_hooks:
        h.remove()

    for h in saliency_hooks:
        h.remove()

    # ----------------------------------------------------------------
    # 7) Move model back to CPU
    # ----------------------------------------------------------------
    model.cpu()

    # ----------------------------------------------------------------
    # 8) Harvest the weight gradients
    # ----------------------------------------------------------------
    gradients = []
    for layer_idx in layers:
        gradients_per_layer = {}
        for module_name, module in analyzer.get_modules(layer_idx).items():
            gradients_per_layer[module_name] = module.weight.grad
        gradients.append(gradients_per_layer)

    # ----------------------------------------------------------------
    # 9) Save saliency per layer, if computed
    # ----------------------------------------------------------------
    if saliency_path is not None:
        logging.info(f"Saving saliency files to {saliency_path}...")

        # Ensure directory exists
        os.makedirs(saliency_path, exist_ok=True)

        # For each layer, gather module data -> single dictionary, then save
        for layer_idx, layer in enumerate(layers):
            # If sub_saliency is set, only save if layer_idx in range
            if (start_layer is not None) and (end_layer is not None):
                if not (start_layer <= layer_idx < end_layer):
                    continue

            # Build dict of { module_name -> cat_tensor or None }
            layer_dict = {}
            for module_name, chunk_list in saliency_data[layer_idx].items():
                if len(chunk_list) > 0:
                    cat_tensor = torch.cat(chunk_list, dim=0)  # shape: [N, seq_len, num_groups]
                else:
                    cat_tensor = None
                layer_dict[module_name] = cat_tensor

            # If there's no data at all (empty?), you can choose to skip saving
            # But we'll save anyway for consistency
            filename = os.path.join(saliency_path, f"l{layer_idx}.pt")

            if os.path.exists(filename):
                logging.warning(f"[saliency] {filename} already exists; overwriting.")

            # Save each layer's dictionary to l{layer_idx}.pt
            torch.save(layer_dict, filename)

    # ----------------------------------------------------------------
    # 10) Save the gradients (if needed)
    # ----------------------------------------------------------------
    # In MC mode the harvested weight.grad comes from pseudo-labels and is NOT the
    # quantity anyone caches; more importantly save_path is estimator-agnostic
    # (no _mc suffix), so writing here would CLOBBER the Fisher weight-gradient
    # cache. Only the (suffixed) saliency files are the MC deliverable, so skip
    # the weight-gradient write entirely under MC.
    if save_path is not None and not skip_save_gradients and not mc_fisher:
        logging.info(f"Saving gradients to {save_path}...")
        if not save_path.endswith('.pt'):
            save_path = save_path + '.pt'
        if os.path.exists(save_path):
            logging.warning(f"[gradients] {save_path} already exists; overwriting.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(gradients, save_path)
    elif mc_fisher:
        logging.info("[hessian-estimator] MC mode: weight-gradient cache left "
                     "untouched (only saliency files were written).")

    # ----------------------------------------------------------------
    # 11) Return the gradients
    # ----------------------------------------------------------------
    return gradients