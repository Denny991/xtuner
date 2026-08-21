#!/usr/bin/env bash
set -euo pipefail

# Fixed-input SparseMLA backward reproducibility test.
#
# Each repeat is a fresh Python process on one H200. The PyTorch correctness
# backend and the TileLang backend receive bitwise-identical q/kv/dOutput and
# causal sparse indices. The script compares output, dQ and dKV hashes, then
# reports elementwise dKV differences.
#
# Default shape is intentionally smaller than the 4096-token cuDNN experiment:
# PyTorch materializes [S, topk, D] and [S, H, topk] tensors. S=1024 preserves
# heavy many-to-one dKV accumulation while fitting comfortably on one H200.

REPO_ROOT=${REPO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-glm52-repro}
ENV_PYTHON=${ENV_PYTHON:-/usr/bin/python}
RESULT_ROOT=${RESULT_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/glm52-repro}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-$(date +%Y%m%d%H%M%S)}
LOG_PATH=${LOG_PATH:-${RESULT_ROOT}/torch-tilelang-sparse-mla-repeat-${EXPERIMENT_TAG}.log}

BACKENDS=${BACKENDS:-"torch tilelang"}
REPEATS=${REPEATS:-5}
SPARSE_MLA_SEQ_LEN=${SPARSE_MLA_SEQ_LEN:-1024}
SPARSE_MLA_TOPK=${SPARSE_MLA_TOPK:-1024}
STRICT_DETERMINISTIC=${STRICT_DETERMINISTIC:-1}

export REPO_ROOT ENV_PYTHON RESULT_ROOT LOG_PATH
export BACKENDS REPEATS SPARSE_MLA_SEQ_LEN SPARSE_MLA_TOPK STRICT_DETERMINISTIC
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -d "${REPO_ROOT}/.git" ]]; then
  echo "ERROR: repository not found: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "ERROR: Python is not executable: ${ENV_PYTHON}" >&2
  exit 1
fi
if [[ ! "${REPEATS}" =~ ^[2-9][0-9]*$ ]]; then
  echo "ERROR: REPEATS must be an integer >= 2" >&2
  exit 1
fi
if [[ ! "${SPARSE_MLA_SEQ_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SPARSE_MLA_SEQ_LEN must be positive" >&2
  exit 1
fi
if [[ ! "${SPARSE_MLA_TOPK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SPARSE_MLA_TOPK must be positive" >&2
  exit 1
fi
if (( SPARSE_MLA_TOPK > SPARSE_MLA_SEQ_LEN )); then
  echo "ERROR: SPARSE_MLA_TOPK cannot exceed SPARSE_MLA_SEQ_LEN" >&2
  exit 1
fi
if (( SPARSE_MLA_TOPK % 64 != 0 )); then
  echo "ERROR: SPARSE_MLA_TOPK must be divisible by 64 for TileLang" >&2
  exit 1
fi
if [[ "${STRICT_DETERMINISTIC}" != "0" && "${STRICT_DETERMINISTIC}" != "1" ]]; then
  echo "ERROR: STRICT_DETERMINISTIC must be 0 or 1" >&2
  exit 1
fi

mkdir -p "${RESULT_ROOT}"
cd "${REPO_ROOT}"

TMP_DIR=$(mktemp -d /tmp/glm52-torch-tilelang-repeat.XXXXXX)
export TMP_DIR
trap 'rm -rf -- "${TMP_DIR}"' EXIT

# Preserve all terminal output in one small log file.
exec > >(tee "${LOG_PATH}") 2>&1

echo "===== Environment preflight ====="
echo "date: $(date --iso-8601=seconds)"
echo "repo: ${REPO_ROOT}"
echo "commit: $(git rev-parse HEAD)"
echo "python: ${ENV_PYTHON}"
echo "backends: ${BACKENDS}"
echo "repeats: ${REPEATS}"
echo "sequence length: ${SPARSE_MLA_SEQ_LEN}"
echo "topk: ${SPARSE_MLA_TOPK}"
echo "strict deterministic algorithms: ${STRICT_DETERMINISTIC}"
echo "log: ${LOG_PATH}"

CUDA_VISIBLE_DEVICES=0 "${ENV_PYTHON}" - <<'PY'
import torch
import xtuner
from xtuner.v1.ops.sparse_mla import ensure_tilelang_runtime_available

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA device count:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("xtuner:", xtuner.__file__)

if torch.__version__.split("+")[0] != "2.9.1":
    raise RuntimeError(f"Expected Torch 2.9.1 CI environment, found {torch.__version__}")
if torch.version.cuda != "12.8":
    raise RuntimeError(f"Expected CUDA 12.8 Torch build, found {torch.version.cuda}")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"Expected one visible GPU, found {torch.cuda.device_count()}")

ensure_tilelang_runtime_available()
print("TileLang runtime: OK")
print("PREFLIGHT PASSED")
PY

read -r -a BACKEND_ARRAY <<< "${BACKENDS}"
for BACKEND in "${BACKEND_ARRAY[@]}"; do
  case "${BACKEND}" in
    torch|tilelang) ;;
    *)
      echo "ERROR: unsupported backend: ${BACKEND}" >&2
      exit 1
      ;;
  esac

  for REPEAT in $(seq 1 "${REPEATS}"); do
    echo
    echo "===== backend=${BACKEND} repeat=${REPEAT}/${REPEATS} ====="

    set +e
    CUDA_VISIBLE_DEVICES=0 "${ENV_PYTHON}" - "${BACKEND}" "${REPEAT}" <<'PY'
import hashlib
import math
import os
import sys
import time

import torch

from xtuner.v1.ops.sparse_mla import sparse_mla


backend = sys.argv[1]
repeat = int(sys.argv[2])
seq_len = int(os.environ["SPARSE_MLA_SEQ_LEN"])
topk = int(os.environ["SPARSE_MLA_TOPK"])
strict_deterministic = os.environ["STRICT_DETERMINISTIC"] == "1"
tmp_dir = os.environ["TMP_DIR"]

num_heads = 64
head_dim = 576
value_dim = 512
seed = 20260817

torch.use_deterministic_algorithms(
    True,
    warn_only=not strict_deterministic,
)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats(device)

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

# Deterministic causal indices, with -1 padding before sequence start.
query_pos = torch.arange(seq_len, device=device, dtype=torch.int32)[:, None]
offset = torch.arange(topk, device=device, dtype=torch.int32)[None, :]
indices_2d = query_pos - offset
indices_2d.masked_fill_(indices_2d < 0, -1)
indices = indices_2d.unsqueeze(1).contiguous()

torch.cuda.synchronize()
started = time.perf_counter()

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

elapsed = time.perf_counter() - started
peak_memory_gib = torch.cuda.max_memory_allocated(device) / (1024**3)


def sha256_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


output_hash = sha256_tensor(outputs.raw_output)
dq_hash = sha256_tensor(dq)
dkv_hash = sha256_tensor(dkv)

print("backend:", backend)
print("repeat:", repeat)
print("shape:", (seq_len, num_heads, head_dim, topk))
print("strict_deterministic:", strict_deterministic)
print(
    "input_probe:",
    [
        q.detach().flatten()[0].item(),
        kv.detach().flatten()[0].item(),
        grad_output.flatten()[0].item(),
        indices.flatten()[0].item(),
        indices.flatten()[-1].item(),
    ],
)
print("output_sha256:", output_hash)
print("dQ_sha256:", dq_hash)
print("dKV_sha256:", dkv_hash)
print("dQ_norm:", dq.detach().float().norm().item())
print("dKV_norm:", dkv.detach().float().norm().item())
print("elapsed_seconds:", elapsed)
print("peak_memory_GiB:", peak_memory_gib)

torch.save(
    {
        "backend": backend,
        "repeat": repeat,
        "output_hash": output_hash,
        "dq_hash": dq_hash,
        "dkv_hash": dkv_hash,
        "dkv": dkv.detach().cpu(),
    },
    os.path.join(tmp_dir, f"{backend}-{repeat}.pt"),
)
PY
    STATUS=$?
    set -e

    if [[ "${STATUS}" -ne 0 ]]; then
      echo "BACKEND_REPEAT_FAILED: backend=${BACKEND}, repeat=${REPEAT}, exit_code=${STATUS}"
      # One failed repeat is sufficient to classify this backend as unavailable
      # under strict deterministic mode. Continue so the other backend still runs.
      break
    fi
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
expected_repeats = int(os.environ["REPEATS"])

first_by_backend = {}
for backend in backends:
    paths = sorted(glob.glob(os.path.join(tmp_dir, f"{backend}-*.pt")))
    print(f"backend={backend}")
    print(f"  successful_repeats={len(paths)}/{expected_repeats}")
    if not paths:
        print("  result=UNAVAILABLE_OR_FAILED")
        continue

    payloads = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    reference = payloads[0]["dkv"]
    first_by_backend[backend] = reference

    print(
        "  all_output_bitwise_equal=",
        len(paths) == expected_repeats
        and all(item["output_hash"] == payloads[0]["output_hash"] for item in payloads[1:]),
        sep="",
    )
    print(
        "  all_dQ_bitwise_equal=",
        len(paths) == expected_repeats
        and all(item["dq_hash"] == payloads[0]["dq_hash"] for item in payloads[1:]),
        sep="",
    )
    print(
        "  all_dKV_bitwise_equal=",
        len(paths) == expected_repeats
        and all(torch.equal(reference, item["dkv"]) for item in payloads[1:]),
        sep="",
    )

    reference_f32 = reference.float()
    reference_norm = reference_f32.norm().item()
    for index, payload in enumerate(payloads[1:], start=2):
        tensor = payload["dkv"]
        delta = tensor.float() - reference_f32
        changed = torch.count_nonzero(tensor != reference).item()
        max_abs = delta.abs().max().item()
        mean_abs = delta.abs().mean().item()
        rel_l2 = delta.norm().item() / reference_norm if reference_norm else float("nan")
        print(
            f"  repeat1_vs_repeat{index}: changed={changed}/{reference.numel()}, "
            f"max_abs={max_abs:.9g}, mean_abs={mean_abs:.9g}, rel_l2={rel_l2:.9g}"
        )

if "torch" in first_by_backend and "tilelang" in first_by_backend:
    torch_dkv = first_by_backend["torch"]
    tilelang_dkv = first_by_backend["tilelang"]
    delta = tilelang_dkv.float() - torch_dkv.float()
    base_norm = torch_dkv.float().norm().item()
    print("cross_backend_repeat1:")
    print("  dKV_bitwise_equal:", torch.equal(torch_dkv, tilelang_dkv))
    print("  changed:", f"{torch.count_nonzero(torch_dkv != tilelang_dkv).item()}/{torch_dkv.numel()}")
    print("  max_abs:", delta.abs().max().item())
    print("  mean_abs:", delta.abs().mean().item())
    print("  rel_l2:", delta.norm().item() / base_norm if base_norm else float("nan"))
PY

echo
echo "===== Experiment completed ====="
echo "Log saved to: ${LOG_PATH}"
