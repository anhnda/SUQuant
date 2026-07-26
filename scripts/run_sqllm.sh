set -x

MODEL_REF=$1
BITS=$2
NUM_GROUPS=$3

MODEL_PATH=$(python resolve_model.py "$MODEL_REF") || exit $?

DATASET=${DATASET:-c4}
SEQ_LEN=${SEQ_LEN:-2048}
NUM_EXAMPLES=${NUM_EXAMPLES:-128}

MODE_OPT=""
if [[ "$4" == "-m" && -n "$5" ]]; then
  MODE_OPT="--mode $5"
fi

MC_OPT=""
if [[ -n "$MC_SAMPLES" ]]; then
  MC_OPT="--mc_fisher true --mc_samples $MC_SAMPLES"
  [[ -n "$MC_SEED" ]] && MC_OPT="$MC_OPT --mc_seed $MC_SEED"
fi

python quantize.py "$MODEL_PATH" \
  --model_name "$MODEL_REF" \
  --seed_precision "$BITS" --parent_precision "$BITS" \
  --dataset "$DATASET" --seq_len "$SEQ_LEN" --num_examples "$NUM_EXAMPLES" \
  --num_groups "$NUM_GROUPS" --random_state 42 $MODE_OPT $MC_OPT