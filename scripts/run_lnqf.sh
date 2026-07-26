#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LNQ-F : LNQ with the FIRST-ORDER (linear) end-loss term retained.
#
#   bash scripts/run_lnqf.sh <model_ref> <bits> <num_groups> [-m mode]
#   bash scripts/run_lnqf.sh meta-llama/Llama-2-7b-hf 3 1
#
# WHAT IT DOES
# ------------
# Plain LNQ / GuidedQuant minimises only the SECOND-order term of the change in
# end loss,
#         Delta_l ~= 1/2 (w_hat - w)^T H (w_hat - w),           (B2)
# dropping the first-order term  g_bar^T (w_hat - w)  under the "converged model
# => mean gradient ~ 0" assumption. On a small, distribution-shifted calibration
# set that residual is real (most LLM tokens are not saturated => d l/d z != 0),
# and it is what end-to-end fine-tuning is later seen to recover.
#
# LNQ-F keeps it:
#         Delta_l ~= g_bar^T(w_hat - w) + 1/2 (w_hat - w)^T H (w_hat - w).  (B1+B2)
#
# IMPLEMENTATION (option A -- NO global H^{-1})
# ---------------------------------------------
# The first-order term is folded DIRECTLY into LNQ's two closed-form updates,
# not via a Newton target shift w - H^{-1} g_bar (which would invert the
# ill-conditioned full H and blow up -- the earlier prototype's failure mode):
#   * codebook (given P):  c = (P^T H P)^{-1} (P^T H w - P^T g_bar)
#                          -- same small (m x m) inverse as vanilla LNQ; the RHS
#                          shift is one triangular solve with the existing L.
#   * assignment (given c): CD round target becomes  w_i - g_bar_i/H_ii - B_i
#                          -- division by the SCALAR diagonal H_ii, no inverse.
# Both remain exact minimizers along their block/coordinate, so LNQ's Prop. 4.1
# monotone-descent + convergence guarantee carries over unchanged.
#
# PREREQUISITES (identical to the other solvers, PLUS one extra pass)
# -------------------------------------------------------------------
#   bash scripts/run_sqllm.sh $MODEL $BITS $G           # init  -- REQUIRED
#   bash scripts/run_lnq.sh   $MODEL $BITS $G -m hessians   # H cache -- shared
# LNQ-F additionally needs the SIGNED first-order cache, which the squared
# saliency pass does NOT produce (it stores (d l/d z)^2, sign discarded). This
# script builds it automatically via firstorder_cache.py before quantizing.
#
# SANITY CHECK (do this first): run with SKIP_FIRSTORDER=1 after deleting the
# first-order cache, OR compare against run_lnq.sh directly. With g_bar absent
# LNQ-F reduces bit-for-bit to vanilla LNQ, so the two objectives/perplexity
# must match. Only once that holds should you trust the g_bar-on numbers.
#
# The B1+B2 objective is logged per iteration as  phi = B2 + lin, so you can
# watch the linear term's contribution directly.
#
# NOTE: MU / MAX_SHIFT are accepted for CLI compatibility but are UNUSED in
# option A (there is no Newton step to damp). They are kept only so old command
# lines do not break.
#
# Env overrides: DATASET, SEQ_LEN, NUM_EXAMPLES, NUM_ITER, CD_CYCLES,
#                SKIP_FIRSTORDER  (MU, MAX_SHIFT accepted but ignored)
# ---------------------------------------------------------------------------
set -x

MODEL_REF=$1
BITS=$2
NUM_GROUPS=$3

MODEL_PATH=$(python resolve_model.py "$MODEL_REF") || exit $?

# Optional mode argument (tokens | hessians | quantize | pack), same as run_lnq.sh
MODE_OPT=""
MODE_VAL=""
if [[ "$4" == "-m" && -n "$5" ]]; then
  MODE_OPT="--mode $5"
  MODE_VAL="$5"
fi

DATASET=${DATASET:-c4}
SEQ_LEN=${SEQ_LEN:-2048}
NUM_EXAMPLES=${NUM_EXAMPLES:-128}

# ---- LNQ stage --------------------------------------------------------------
NUM_ITER=${NUM_ITER:-3}
CD_CYCLES=${CD_CYCLES:-4}

# ---- first-order (B1) knobs -------------------------------------------------
# option A folds g_bar directly into update_P/update_C; there is no Newton step,
# so MU and MAX_SHIFT are UNUSED (kept only for CLI back-compat).
MU=${MU:-0.0}
MAX_SHIFT=${MAX_SHIFT:-0.0}
# Set SKIP_FIRSTORDER=1 to reuse an existing first-order cache without rebuilding.
SKIP_FIRSTORDER=${SKIP_FIRSTORDER:-0}

# ---------------------------------------------------------------------------
# Step 1: build (or locate) the signed first-order cache.
#         firstorder_cache.py prints the cache directory as its last stdout line.
#         Skip entirely when we are only running the -m tokens/hessians stages,
#         since those never reach the solver.
# ---------------------------------------------------------------------------
GBAR_ARG=""
if [[ "$MODE_VAL" != "tokens" && "$MODE_VAL" != "hessians" ]]; then
  if [[ "$SKIP_FIRSTORDER" != "1" ]]; then
    # NOTE: do NOT pipe python into tail -- a pipe hides python's exit code, so a
    # crash in firstorder_cache.py would go unnoticed and we'd run with an empty
    # path. Capture full stdout, check the exit code, then take the last line.
    FO_OUT=$(python firstorder_cache.py "$MODEL_PATH" \
      --model_name "$MODEL_REF" \
      --dataset "$DATASET" --seq_len "$SEQ_LEN" --num_examples "$NUM_EXAMPLES" \
      --random_state 42)
    FO_RC=$?
    echo "$FO_OUT"
    if [[ $FO_RC -ne 0 ]]; then
      echo "[run_lnqf] first-order cache build FAILED (rc=$FO_RC). Aborting." >&2
      exit $FO_RC
    fi
    GBAR_PATH=$(printf '%s\n' "$FO_OUT" | tail -n 1)
  else
    GBAR_PATH="cache/firstorder/${MODEL_REF##*/}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
  fi
  if [[ -z "$GBAR_PATH" ]]; then
    echo "[run_lnqf] could not determine first-order cache path. Aborting." >&2
    exit 1
  fi
  GBAR_ARG="--lnqf_gbar_path $GBAR_PATH"
fi

# ---------------------------------------------------------------------------
# Step 2: quantize with the LNQ-F solver.
# ---------------------------------------------------------------------------
python layerwise_nuq.py "$MODEL_PATH" \
  --model_name "$MODEL_REF" \
  --solver lnqf \
  --seed_precision "$BITS" \
  --dataset "$DATASET" --seq_len "$SEQ_LEN" --num_examples "$NUM_EXAMPLES" \
  --num_groups "$NUM_GROUPS" --random_state 42 \
  --num_iterations "$NUM_ITER" --cd_cycles "$CD_CYCLES" \
  --lnqf_mu "$MU" \
  --lnqf_max_shift "$MAX_SHIFT" \
  $GBAR_ARG \
  $MODE_OPT