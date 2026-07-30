set -x

MODEL_REF=$1
BITS=$2
NUM_GROUPS=$3

MODEL_PATH=$(python resolve_model.py "$MODEL_REF") || exit $?

# Optional mode argument
MODE_OPT=""
if [[ "$4" == "-m" && -n "$5" ]]; then
  MODE_OPT="--mode $5"
fi

DATASET=${DATASET:-c4}
SEQ_LEN=${SEQ_LEN:-2048}
NUM_EXAMPLES=${NUM_EXAMPLES:-128}

# ---------------------------------------------------------------------------
# Optimizer selection (same entry point, just a different --solver).
#
# Default is plain LNQ (1-opt scalar CD within the nearest-codeword set).
# Set SOLVER=c2cd to swap the assignment optimizer for the correlation-matching
# M2-CD method (the "** corr-matching M2-CD **" runner in m2cd_verify.py):
# instead of scalar per-coordinate flips, coordinates are paired by a greedy
# max-weight matching over the top-nu correlation graph (|H_ik| normalised),
# and each pair is updated by an EXACT m^2 joint enumeration. This lets the
# search cross the pair-barriers that no single 1-opt flip can reach. The
# codebook solve (Eq. 9 / update_C) is untouched, so it drops straight into the
# existing LNQ alternating loop.
#
#   SOLVER=c2cd bash scripts/run_lnq.sh Llama-3.2-1B 3 1 -m hessians
#
# Correlation-matching knobs (only used when SOLVER=c2cd):
#   C2CD_NU      top-nu correlation neighbours per coord for the matching graph
#   C2CD_CYCLES  number of M2-CD sweeps per P-step (default: cd_cycles)
SOLVER=${SOLVER:-lnq}
SOLVER_OPT="--solver $SOLVER"

C2CD_OPT=""
if [[ "$SOLVER" == "c2cd" ]]; then
  C2CD_NU=${C2CD_NU:-16}
  C2CD_OPT="--c2cd_nu $C2CD_NU"
  [[ -n "$C2CD_CYCLES" ]] && C2CD_OPT="$C2CD_OPT --c2cd_cycles $C2CD_CYCLES"
fi

# MC / Fisher estimator passthrough. Set via env before the call, e.g.:
#   MC_SAMPLES=4 bash scripts/run_lnq.sh Llama-3.2-1B 3 1 -m hessians
# Setting MC_SAMPLES (to anything) turns on the MC estimator; unset = Fisher.
MC_OPT=""
if [[ -n "$MC_SAMPLES" ]]; then
  MC_OPT="--mc_fisher true --mc_samples $MC_SAMPLES"
  [[ -n "$MC_SEED" ]] && MC_OPT="$MC_OPT --mc_seed $MC_SEED"
fi

# MP trust-region (replaces damping in LNQ's train_least_squares). Enable with:
#   MP_TR=1 [MP_NEFF=<n_eff> MP_KMAX=256] bash scripts/run_lnq.sh ...
# MP_NEFF should come from measure_mp.py (§8). Consumed via env by layerwise_quantize.
if [[ -n "$MP_TR" ]]; then
  export AP_MP_TR="$MP_TR"
  [[ -n "$MP_NEFF" ]] && export AP_MP_NEFF="$MP_NEFF"
  [[ -n "$MP_KMAX" ]] && export AP_MP_KMAX="$MP_KMAX"
fi

python layerwise_nuq.py "$MODEL_PATH" \
  --model_name "$MODEL_REF" \
  --seed_precision "$BITS" \
  --dataset "$DATASET" --seq_len "$SEQ_LEN" --num_examples "$NUM_EXAMPLES" \
  --num_groups "$NUM_GROUPS" --random_state 42 $SOLVER_OPT $C2CD_OPT $MODE_OPT $MC_OPT
