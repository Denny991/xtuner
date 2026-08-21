#!/usr/bin/env bash
set -euo pipefail

# Three full 35-step SFT runs using the shared Torch 2.12.1/CUDA 13.2
# environment and the cuDNN DSA sparse-MLA backward.
REPO_ROOT=${REPO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-glm52-repro}
ENV_PREFIX=${ENV_PREFIX:-/mnt/shared-storage-user/llmrazor-share/comm_env/glm52-pt121-cu132}
ENV_PYTHON=${ENV_PYTHON:-${ENV_PREFIX}/bin/python}
REPRO_ROOT=${REPRO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/glm52-repro}
RUN_COUNT=${RUN_COUNT:-3}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-$(date +%Y%m%d%H%M%S)}

BASE_CONFIG=${BASE_CONFIG:-${REPO_ROOT}/autotest/config/glm5p2_30B_all2all_nosave.py}
DSA_CONFIG=${DSA_CONFIG:-${REPO_ROOT}/autotest/config/glm5p2_30B_all2all_cudnn_dsa_nosave.py}
EXPERIMENT_ROOT=${REPRO_ROOT}/cudnn-dsa-sft-${EXPERIMENT_TAG}
TRACKER_ROOT=${EXPERIMENT_ROOT}/trackers

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/llmrazor-share/model/GLM-5.2-30B}
export ALPACA_PATH=${ALPACA_PATH:-/mnt/shared-storage-user/llmrazor-share/data/alpaca}
export XTUNER_GC_ENABLE=${XTUNER_GC_ENABLE:-1}
export SWAP_OPTIMIZER=${SWAP_OPTIMIZER:-0}
export XTUNER_ACTIVATION_OFFLOAD=${XTUNER_ACTIVATION_OFFLOAD:-0}
export XTUNER_USE_CUTLASS_GROUP_GEMM=${XTUNER_USE_CUTLASS_GROUP_GEMM:-1}
export XTUNER_DETERMINISTIC=${XTUNER_DETERMINISTIC:-true}

if [[ ! -d "${REPO_ROOT}/.git" ]]; then
  echo "ERROR: repository not found: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "ERROR: shared-environment Python is not executable: ${ENV_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "ERROR: base config not found: ${BASE_CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -e "${ALPACA_PATH}" ]]; then
  echo "ERROR: Alpaca data path not found: ${ALPACA_PATH}" >&2
  exit 1
fi
if [[ ! "${RUN_COUNT}" =~ ^[2-9][0-9]*$ ]]; then
  echo "ERROR: RUN_COUNT must be an integer >= 2" >&2
  exit 1
fi

mkdir -p "${EXPERIMENT_ROOT}" "${TRACKER_ROOT}"
cd "${REPO_ROOT}"

# Generate the experiment config from the already-tested no-save/all2all config.
# Only sparse_mla_backend is changed.
cp "${BASE_CONFIG}" "${DSA_CONFIG}"

MATCH_COUNT=$(grep -Ec 'sparse_mla_backend[[:space:]]*=[[:space:]]*"tilelang"' "${DSA_CONFIG}" || true)
if [[ "${MATCH_COUNT}" -ne 1 ]]; then
  echo "ERROR: expected exactly one tilelang sparse_mla_backend assignment; found ${MATCH_COUNT}" >&2
  exit 1
fi

sed -i -E \
  's/(sparse_mla_backend[[:space:]]*=[[:space:]]*)"tilelang"/\1"cudnn_dsa"/' \
  "${DSA_CONFIG}"

if ! grep -Eq 'dispatcher[[:space:]]*=[[:space:]]*"all2all"' "${DSA_CONFIG}"; then
  echo "ERROR: generated config is not using dispatcher=all2all" >&2
  exit 1
fi
if ! grep -Eq 'sparse_mla_backend[[:space:]]*=[[:space:]]*"cudnn_dsa"' "${DSA_CONFIG}"; then
  echo "ERROR: generated config is not using sparse_mla_backend=cudnn_dsa" >&2
  exit 1
fi
if ! grep -Eq 'debug_skip_save[[:space:]]*=[[:space:]]*True' "${DSA_CONFIG}"; then
  echo "ERROR: debug_skip_save=True is missing; refusing to risk a large checkpoint write" >&2
  exit 1
fi

echo "===== Experiment configuration ====="
echo "repo: ${REPO_ROOT}"
echo "commit: $(git rev-parse HEAD)"
echo "python: ${ENV_PYTHON}"
echo "base config: ${BASE_CONFIG}"
echo "DSA config: ${DSA_CONFIG}"
echo "experiment root: ${EXPERIMENT_ROOT}"
echo "run count: ${RUN_COUNT}"
echo "model: ${MODEL_PATH}"
echo "data: ${ALPACA_PATH}"
grep -E 'dispatcher|sparse_mla_backend|debug_skip_save' "${DSA_CONFIG}"
df -h "${REPRO_ROOT}"

echo
echo "===== Runtime preflight ====="
"${ENV_PYTHON}" - <<'PY'
import torch
import xtuner
from xtuner.v1.ops.sparse_mla import (
    ensure_cudnn_dsa_runtime_available,
    ensure_tilelang_runtime_available,
)

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA device count:", torch.cuda.device_count())
print("GPU 0:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("xtuner:", xtuner.__file__)
if torch.cuda.device_count() != 8:
    raise RuntimeError(f"Expected 8 visible GPUs, found {torch.cuda.device_count()}")
ensure_tilelang_runtime_available()
ensure_cudnn_dsa_runtime_available()
print("TileLang forward runtime: OK")
print("cuDNN DSA backward runtime: OK")
PY

for RUN_INDEX in $(seq 1 "${RUN_COUNT}"); do
  RUN_NAME="cudnn-dsa-run${RUN_INDEX}"
  WORK_DIR="${EXPERIMENT_ROOT}/${RUN_NAME}"
  TRAIN_LOG="${WORK_DIR}/train.log"

  mkdir -p "${WORK_DIR}"
  export GITHUB_RUN_ID="manual-${EXPERIMENT_TAG}-${RUN_NAME}"
  export WORK_DIR

  echo
  echo "===== Starting ${RUN_NAME}/${RUN_COUNT} ====="
  echo "WORK_DIR=${WORK_DIR}"

  set +e
  "${ENV_PYTHON}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=8 \
    xtuner/v1/train/cli/sft.py \
    --config "${DSA_CONFIG}" \
    2>&1 | tee "${TRAIN_LOG}"
  TRAIN_STATUS=${PIPESTATUS[0]}
  set -e

  if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
    echo "ERROR: ${RUN_NAME} failed with exit code ${TRAIN_STATUS}" >&2
    echo "Log: ${TRAIN_LOG}" >&2
    exit "${TRAIN_STATUS}"
  fi

  TRACKER=$(find "${WORK_DIR}" \
    -type f \
    -path '*/logs/exp_tracking/rank0/tracker.jsonl' \
    | sort \
    | tail -1)

  if [[ -z "${TRACKER}" ]]; then
    echo "ERROR: tracker not found under ${WORK_DIR}" >&2
    exit 1
  fi

  TRACKER_LINES=$(wc -l < "${TRACKER}")
  if [[ "${TRACKER_LINES}" -ne 35 ]]; then
    echo "ERROR: expected 35 tracker records, found ${TRACKER_LINES}: ${TRACKER}" >&2
    exit 1
  fi

  cp "${TRACKER}" "${TRACKER_ROOT}/${RUN_NAME}.jsonl"
  echo "Completed ${RUN_NAME}: ${TRACKER_LINES} steps"
  echo "Tracker: ${TRACKER_ROOT}/${RUN_NAME}.jsonl"
done

echo
echo "===== Tracker summary ====="
wc -l "${TRACKER_ROOT}"/*.jsonl

echo
echo "===== Pairwise comparison ====="
"${ENV_PYTHON}" - "${TRACKER_ROOT}" <<'PY'
import itertools
import json
import sys
from pathlib import Path


tracker_root = Path(sys.argv[1])
paths = sorted(tracker_root.glob("cudnn-dsa-run*.jsonl"))
if len(paths) < 2:
    raise RuntimeError(f"Need at least two trackers, found {len(paths)}")

metrics = (
    "grad_norm",
    "loss/local_loss",
    "loss/reduced_llm_loss",
    "loss/reduced_mtp_loss",
    "lr",
    "runtime_info/text_tokens",
)


def load(path):
    rows = []
    with path.open() as stream:
        for position, line in enumerate(stream):
            row = json.loads(line)
            row["_position"] = position
            rows.append(row)
    return rows


runs = {path.stem: load(path) for path in paths}
for left_name, right_name in itertools.combinations(runs, 2):
    left = runs[left_name]
    right = runs[right_name]
    if len(left) != len(right):
        raise RuntimeError(f"Length mismatch: {left_name}={len(left)}, {right_name}={len(right)}")

    print(f"--- {left_name} vs {right_name} ---")
    for metric in metrics:
        comparisons = []
        for left_row, right_row in zip(left, right, strict=True):
            if left_row.get("step") != right_row.get("step"):
                raise RuntimeError(
                    f"Step mismatch at position {left_row['_position']}: "
                    f"{left_row.get('step')} != {right_row.get('step')}"
                )
            left_value = float(left_row[metric])
            right_value = float(right_row[metric])
            denominator = max(abs(left_value), 1e-12)
            relative_error = abs(left_value - right_value) / denominator
            comparisons.append(
                (
                    relative_error,
                    int(left_row["step"]),
                    int(left_row["_position"]),
                    left_value,
                    right_value,
                )
            )

        worst = max(comparisons, key=lambda item: item[0])
        failures = [item for item in comparisons if round(item[0], 2) > 1e-6]
        if failures:
            first = failures[0]
            first_failure = f"position={first[2]}, training_step={first[1]}"
        else:
            first_failure = "none"

        print(
            f"{metric}: max_rel={worst[0]:.6%}, max_step={worst[1]}, "
            f"first_CI_failure={first_failure}, CI_failure_steps={len(failures)}"
        )

    print()

print("Experiment trackers:")
for path in paths:
    print(path)
PY

echo
echo "===== Experiment completed ====="
echo "Experiment root: ${EXPERIMENT_ROOT}"
echo "Trackers: ${TRACKER_ROOT}"
