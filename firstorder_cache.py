"""
Build the signed first-order end-loss cache for LNQ-F.

This is the first-order companion to the squared-saliency Hessian pass. It writes
one file per layer:

    {cache_dir}/firstorder/{model_name}-{dataset}_s{N}_blk{seq}/l{L}.pt
        -> { module_name: g_bar tensor [out, in] }

The output directory is exactly what run_lnqf.sh passes to
`--lnqf_gbar_path`, so the two line up with no manual bookkeeping.

Usage (normally invoked from scripts/run_lnqf.sh):

    python firstorder_cache.py <model_path> \
        --model_name <ref> --dataset c4 --seq_len 2048 --num_examples 128 \
        --random_state 42
"""

import argparse
import functools
import logging
import os
import sys

import torch
torch.load = functools.partial(torch.load, weights_only=False)

from any_precision.analyzer import get_analyzer
from any_precision.quantization.datautils import get_tokens
from any_precision.quantization.gradients_firstorder import get_firstorder


def main():
    p = argparse.ArgumentParser(description="Build signed first-order cache for LNQ-F")
    p.add_argument("model", type=str)
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default="cache")
    p.add_argument("--dataset", type=str, default="c4")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--num_examples", type=int, default=128)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--sub_layer", nargs="+", type=int, default=None,
                   help="(start, end) layer range to restrict to")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s | %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    model_string = args.model
    model_name = args.model_name or model_string.split("/")[-1]

    tokens_cache_path = (f"{args.cache_dir}/tokens/"
                         f"{model_name}-{args.dataset}_s{args.num_examples}_blk{args.seq_len}.pt")
    gbar_path = (f"{args.cache_dir}/firstorder/"
                 f"{model_name}-{args.dataset}_s{args.num_examples}_blk{args.seq_len}")

    logging.info(f"[lnqf] first-order cache target: {gbar_path}")

    analyzer = get_analyzer(model_string, include_tokenizer=True)

    tokens = get_tokens(
        args.dataset, "train", analyzer.tokenizer,
        args.seq_len, args.num_examples, tokens_cache_path, args.random_state,
    )

    sub_layer = tuple(args.sub_layer) if args.sub_layer else None
    get_firstorder(
        analyzer, tokens, gbar_path,
        sub_layer=sub_layer, overwrite=args.overwrite,
    )

    # Emit the path on stdout's last line so the shell script can capture it.
    print(gbar_path)


if __name__ == "__main__":
    main()
