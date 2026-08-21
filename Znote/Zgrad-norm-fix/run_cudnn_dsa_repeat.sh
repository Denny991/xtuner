#!/usr/bin/env bash
set -euo pipefail

# Run from the H200 container. The shared environment is used by absolute path,
# so conda does not need to be installed in the container.
REPO_ROOT=${REPO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-glm52-repro}
ENV_PREFIX=${ENV_PREFIX:-/mnt/shared-storage-user/llmrazor-share/comm_env/glm52-pt121-cu132}
ENV_PYTHON=${ENV_PYTHON:-${ENV_PREFIX}/bin/python}
RESULT_ROOT=${RESULT_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/glm52-repro}
LOG_PATH=${LOG_PATH:-${RESULT_ROOT}/sparse-mla-backward-ab-repeat.log}

# Default to an A/B test in the same software environment. To test only cuDNN,
# invoke the script with BACKENDS=cudnn_dsa.
BACKENDS=${BACKENDS:-"tilelang cudnn_dsa"}
REPEATS=${REPEATS:-5}
SPARSE_MLA_SEQ_LEN=${SPARSE_MLA_SEQ_LEN:-4096}
TORCH_CUDNN_DETERMINISTIC=${TORCH_CUDNN_DETERMINISTIC:-0}

export REPO_ROOT ENV_PREFIX ENV_PYTHON RESULT_ROOT LOG_PATH
export BACKENDS REPEATS SPARSE_MLA_SEQ_LEN
export TORCH_CUDNN_DETERMINISTIC
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RESULT_ROOT}"
cd "${REPO_ROOT}"

if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "ERROR: Python is not executable: ${ENV_PYTHON}" >&2
  exit 1
fi

TMP_DIR=$(mktemp -d /tmp/glm52-sparse-mla-ab.XXXXXX)
export TMP_DIR
trap 'rm -rf -- "${TMP_DIR}"' EXIT

{
  echo "===== Environment preflight ====="
  echo "date: $(date --iso-8601=seconds)"
  echo "repo: ${REPO_ROOT}"
  echo "python: ${ENV_PYTHON}"
  echo "backends: ${BACKENDS}"
  echo "repeats: ${REPEATS}"
  echo "sequence length: ${SPARSE_MLA_SEQ_LEN}"

  CUDA_VISIBLE_DEVICES=0 "${ENV_PYTHON}" - <<'PY'
import torch
import xtuner
from xtuner.v1.ops.sparse_mla import (
    ensure_cudnn_dsa_runtime_available,
    ensure_tilelang_runtime_available,
)

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("xtuner:", xtuner.__file__)
ensure_tilelang_runtime_available()
print("TileLang runtime: OK")
ensure_cudnn_dsa_runtime_available()
print("cuDNN DSA runtime: OK")
PY

  read -r -a BACKEND_ARRAY <<< "${BACKENDS}"
  for BACKEND in "${BACKEND_ARRAY[@]}"; do
    case "${BACKEND}" in
      tilelang|cudnn_dsa) ;;
      *)
        echo "ERROR: unsupported backend in BACKENDS: ${BACKEND}" >&2
        exit 1
        ;;
    esac

    for REPEAT in $(seq 1 "${REPEATS}"); do
      echo
      echo "===== backend=${BACKEND} repeat=${REPEAT} ====="

      CUDA_VISIBLE_DEVICES=0 "${ENV_PYTHON}" - "${BACKEND}" "${REPEAT}" <<'PY'
import hashlib
import math
import os
import sys

import torch

from xtuner.v1.ops.sparse_mla import sparse_mla


backend = sys.argv[1]
repeat = int(sys.argv[2])
seq_len = int(os.environ["SPARSE_MLA_SEQ_LEN"])
tmp_dir = os.environ["TMP_DIR"]

num_heads = 64
head_dim = 576
value_dim = 512
topk = min(2048, seq_len)
seed = 20260814

if topk % 64 != 0:
    raise RuntimeError(f"topk must be divisible by 64, got {topk}")

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = os.environ["TORCH_CUDNN_DETERMINISTIC"] == "1"
torch.backends.cudnn.benchmark = False
print("torch deterministic:", torch.are_deterministic_algorithms_enabled())
print("cudnn deterministic:", torch.backends.cudnn.deterministic)
print("cudnn benchmark:", torch.backends.cudnn.benchmark)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda")
q = torch.randn(
    seq_len,
    num_heads,
    head_dim,
    device=device,
    dtype=torch.bfloat16,
    requires_grad=True,
)
kv = torch.randn(
    seq_len,
    1,
    head_dim,
    device=device,
    dtype=torch.bfloat16,
    requires_grad=True,
)
grad_output = torch.randn(
    seq_len,
    num_heads,
    value_dim,
    device=device,
    dtype=torch.bfloat16,
)

# Deterministic causal sparse indices. Invalid leading positions are padded
# with -1, matching the GLM training path handled by XTuner's cuDNN wrapper.
query_pos = torch.arange(seq_len, device=device, dtype=torch.int32)[:, None]
offset = torch.arange(topk, device=device, dtype=torch.int32)[None, :]
indices_2d = query_pos - offset
indices_2d.masked_fill_(indices_2d < 0, -1)
indices = indices_2d.unsqueeze(1).contiguous()

outputs = sparse_mla(
    q,
    kv,
    indices,
    scaling=1.0 / math.sqrt(head_dim),
    value_dim=value_dim,
    backend=backend,
)
dq, dkv = torch.autograd.grad(
    outputs.raw_output,
    (q, kv),
    grad_outputs=grad_output,
)
torch.cuda.synchronize()


def sha256_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


print("backend:", backend)
print("repeat:", repeat)
print("shape:", (seq_len, num_heads, head_dim, topk))
print(
    "input_probe:",
    [
        q.detach().flatten()[0].item(),
        kv.detach().flatten()[0].item(),
        grad_output.flatten()[0].item(),
        indices.flatten()[-1].item(),
    ],
)
print("output_sha256:", sha256_tensor(outputs.raw_output))
print("dQ_sha256:", sha256_tensor(dq))
print("dKV_sha256:", sha256_tensor(dkv))
print("dQ_norm:", dq.detach().float().norm().item())
print("dKV_norm:", dkv.detach().float().norm().item())

# dKV is small enough to retain temporarily for an exact elementwise summary.
torch.save(dkv.detach().cpu(), os.path.join(tmp_dir, f"{backend}-{repeat}.pt"))
PY
    done
  done

  echo
  echo "===== Exact repeat comparison ====="
  "${ENV_PYTHON}" - <<'PY'
import glob
import os

import torch


tmp_dir = os.environ["TMP_DIR"]
backends = os.environ["BACKENDS"].split()

first_by_backend = {}
for backend in backends:
    paths = sorted(glob.glob(os.path.join(tmp_dir, f"{backend}-*.pt")))
    if not paths:
        raise RuntimeError(f"No dKV files found for {backend}")

    tensors = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    reference = tensors[0]
    first_by_backend[backend] = reference

    print(f"backend={backend}")
    print(f"  repeat_count={len(tensors)}")
    print(f"  all_dKV_bitwise_equal={all(torch.equal(reference, x) for x in tensors[1:])}")

    reference_f32 = reference.float()
    reference_norm = reference_f32.norm().item()
    for index, tensor in enumerate(tensors[1:], start=2):
        tensor_f32 = tensor.float()
        delta = tensor_f32 - reference_f32
        changed = torch.count_nonzero(tensor != reference).item()
        max_abs = delta.abs().max().item()
        mean_abs = delta.abs().mean().item()
        rel_l2 = delta.norm().item() / reference_norm if reference_norm else float("nan")
        print(
            f"  repeat1_vs_repeat{index}: changed={changed}/{reference.numel()}, "
            f"max_abs={max_abs:.9g}, mean_abs={mean_abs:.9g}, rel_l2={rel_l2:.9g}"
        )

if "tilelang" in first_by_backend and "cudnn_dsa" in first_by_backend:
    tilelang = first_by_backend["tilelang"]
    cudnn = first_by_backend["cudnn_dsa"]
    delta = cudnn.float() - tilelang.float()
    base_norm = tilelang.float().norm().item()
    print("cross_backend_repeat1:")
    print("  dKV_bitwise_equal:", torch.equal(tilelang, cudnn))
    print("  changed:", f"{torch.count_nonzero(tilelang != cudnn).item()}/{tilelang.numel()}")
    print("  max_abs:", delta.abs().max().item())
    print("  mean_abs:", delta.abs().mean().item())
    print("  rel_l2:", delta.norm().item() / base_norm if base_norm else float("nan"))
PY

  echo
  echo "===== Experiment completed ====="
  echo "Log saved to: ${LOG_PATH}"
} 2>&1 | tee "${LOG_PATH}"
