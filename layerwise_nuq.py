import argparse
from any_precision.quantization import layerwise_nuq
import torch, functools
torch.load = functools.partial(torch.load, weights_only=False)
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model to any precision")
    parser.add_argument("model", type=str, help="The model to quantize")
    parser.add_argument("--seed_precision", type=int, help="The precision to quantize the seed to")
    parser.add_argument("--mode", type=str, default="pack", help="The mode to run in")
    parser.add_argument("--yaml_path", type=str, help="The path to the architecture config yaml file")
    parser.add_argument("--cache_dir", type=str, help="The directory to cache results in")
    parser.add_argument("--dataset", type=str, help="The dataset to use")
    parser.add_argument("--seq_len", type=int, help="The sequence length to use")
    parser.add_argument("--num_examples", type=int, help="The number of examples to use")
    parser.add_argument("--cpu_count", type=int, help="The number of CPUs to use for parallelization")
    parser.add_argument("--overwrite_quantize", action="store_true",
                        help="Whether to overwrite the quantized model stored to disk")
    parser.add_argument("--overwrite_pack", action="store_true",
                        help="Whether to overwrite the packed model stored to disk")
    parser.add_argument("--random_state", type=int,
                        help="The random state to use for reproducibility\n"
                             "[WARNING] May not be reproducible across different machines")
    parser.add_argument("--sub_hessian", nargs='+', type=int, default=None,
                         help="(start, end) of layers to use for hessian saving")
    parser.add_argument("--num_groups", type=int, default=4,
                        help="Number of groups $g$ to use for GuidedQuant Hessian")
    parser.add_argument("--num_iterations", type=int, default=3,
                        help="Number of iterations to run")
    parser.add_argument('--cd_cycles', type=int, default=4,
                        help='Number of CD cycles to run')
    parser.add_argument("--sub_qlayer", nargs='+', type=int, default=None,
                        help="(start, end) of layers to use for quantization")
    parser.add_argument("--is_nosal", type=str2bool, default=False,
                        help="Do not use GuidedQuant Hessian")
    parser.add_argument("--model_name", type=str, default=None,
                    help="Stable short name for cache paths (defaults to basename of model path)")
    parser.add_argument("--solver", type=str, default="lnq",
                        choices=["lnq", "flexnu"])
    parser.add_argument("--flexnu_iters", type=int, default=300)
    parser.add_argument("--flexnu_lr_scale", type=float, default=3e-3)
    parser.add_argument("--flexnu_lr_cb", type=float, default=1e-5)
    parser.add_argument("--flexnu_row_block", type=int, default=64)
    parser.add_argument("--flexnu_tau_frac", type=float, default=0.5)
    parser.add_argument("--flexnu_stage_frac", type=float, default=0.0)
    parser.add_argument("--flexnu_eval_every", type=int, default=1)
    parser.add_argument("--flexnu_freeze_codebook", type=str2bool, default=False)
    parser.add_argument("--flexnu_freeze_scale", type=str2bool, default=False)
    args = parser.parse_args()
    args.sub_hessian = tuple(args.sub_hessian) if args.sub_hessian else None
    args.sub_qlayer = tuple(args.sub_qlayer) if args.sub_qlayer else None

    # only pass options that are not None
    fk = {k[len("flexnu_"):]: v for k, v in vars(args).items()
          if k.startswith("flexnu_") and v is not None}
    kw = {k: v for k, v in vars(args).items()
          if v is not None and not k.startswith("flexnu_")}
    kw["flexnu_kwargs"] = fk
    layerwise_nuq(**kw)