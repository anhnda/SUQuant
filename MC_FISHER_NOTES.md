# MC Fisher / GGN — nhánh Hessian estimate (thêm vào, mặc định giữ nguyên)

## Cái gì đổi
Thêm nhánh Monte-Carlo GGN/Fisher cho việc tính saliency Hessian, cạnh nhánh
true-label empirical Fisher hiện có. **Mặc định = Fisher cũ, byte-for-byte.**

Estimator:
- Fisher (cũ, default):  backward CE(logits, true_labels) -> delta = p - onehot(y_true)
- MC (mới, --mc_fisher):  y~ ~ softmax(logits); backward mean_k -log p[y~_k]
                          -> delta = p - onehot(y~),  E[delta delta^T] = diag(p)-pp^T = C_t

Cả hai đi qua đúng cùng grad_hook (s = (1e3*grad)^2) và cùng H = X^T diag(s) X.
Khác biệt DUY NHẤT là loss được backward. Scale giữ nguyên: MC loss là MEAN
over tokens (khớp HF outputs.loss), nên 1e3/1e6 và LNQ damping không phải sửa.

## File đã sửa
- any_precision/quantization/gradients.py   : + _mc_fisher_loss(), + params, nhánh backward
- any_precision/quantization/main.py         : + params, resolve, suffix cache, forward
- any_precision/quantization/layerwise_main.py: + params, resolve, suffix saliency/hessian/quantized/packed
- layerwise_nuq.py        : + CLI --mc_fisher/--mc_samples/--mc_seed
- layerwise_nuq_seq.py    : + CLI + suffix saliency + forward

## Tách cache (quan trọng cho A/B)
Bật MC -> mọi path liên quan thêm suffix "_mc<K>":
  saliency/..._g{g}_mc{K},  hessians/..._mc{K},  layerwise_quantized/..._mc{K}, packed/..._mc{K}
=> Fisher và MC KHÔNG BAO GIỜ đè/đọc nhầm cache của nhau.

## Cách chạy so sánh (LNQ, Llama-2-7B, 2-bit, g=1)
# baseline Fisher (như cũ):
bash scripts/run_lnq.sh Llama-2-7b-hf 2 1 -m hessians

# MC, K=1 (unbiased, noisy nhất):
python layerwise_nuq.py <MODEL_PATH> --model_name Llama-2-7b-hf --seed_precision 2 \
    --num_groups 1 --dataset c4 --seq_len 2048 --num_examples 128 --random_state 42 \
    --mode hessians --mc_fisher true --mc_samples 1 --mc_seed 0

# MC, K=4 (giảm variance):
python layerwise_nuq.py <MODEL_PATH> ... --mode hessians --mc_fisher true --mc_samples 4 --mc_seed 0

Hoặc bật bằng env cho BẤT KỲ entrypoint nào (kể cả main.py any-precision):
  AP_MC_FISHER=1 AP_MC_SAMPLES=4 AP_MC_SEED=0  bash scripts/run_lnq.sh Llama-2-7b-hf 2 1 -m hessians

Lưu ý: run_lnq.sh -m hessians CHỈ đọc saliency; phải sinh saliency MC trước
(qua layerwise_nuq_seq.py ở clean mode, hoặc main.py --mode gradients) với CÙNG
--mc_fisher/--mc_samples để path khớp.

## Đã verify (không chạy pipeline)
- syntax toàn bộ file OK
- MC flags được forward đúng qua kw của entrypoint
- E[delta delta^T] -> C_t  (sai số 4e-4 @ N=4e5): estimator unbiased đúng như lý thuyết
- _mc_fisher_loss: loss finite, grad per-token sum=0 (đúng softmax-CE), shape đúng
