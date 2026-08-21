#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2 MTP-gradient ablation:
#   - same GLM-5.2-30B checkpoint
#   - same Torch 2.9.1/CUDA 12.8 CI environment
#   - same TileLang sparse MLA + all2all configuration
#   - only MTP loss_scaling_factor is changed from 0.1 to 0.0
#   - three independent 35-step runs, with checkpoint saving disabled

REPO_ROOT=${REPO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-glm52-repro}
ENV_PYTHON=${ENV_PYTHON:-/usr/bin/python}
REPRO_ROOT=${REPRO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/glm52-repro}
RUN_COUNT=${RUN_COUNT:-3}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-$(date +%Y%m%d%H%M%S)}

BASE_CONFIG=${BASE_CONFIG:-${REPO_ROOT}/autotest/config/glm5p2_30B_all2all_nosave.py}
TEST_CONFIG=${TEST_CONFIG:-${REPO_ROOT}/autotest/config/glm5p2_30B_all2all_mtp_loss0_nosave.py}
EXPERIMENT_ROOT=${REPRO_ROOT}/glm52-mtp-loss0-sft-${EXPERIMENT_TAG}
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
export TEST_CONFIG

if [[ ! -d "${REPO_ROOT}/.git" ]]; then
  echo "ERROR: repository not found: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "ERROR: CI-environment Python is not executable: ${ENV_PYTHON}" >&2
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

# Generate the test config from the previously validated TileLang/all2all/no-save
# config. The two inserted lines are the only intended experiment change.
cp "${BASE_CONFIG}" "${TEST_CONFIG}"

DISPATCHER_COUNT=$(grep -Ec 'moe_cfg\.dispatcher[[:space:]]*=[[:space:]]*"all2all"' "${TEST_CONFIG}" || true)
if [[ "${DISPATCHER_COUNT}" -ne 1 ]]; then
  echo "ERROR: expected exactly one moe_cfg.dispatcher=all2all assignment; found ${DISPATCHER_COUNT}" >&2
  exit 1
fi

sed -i -E '/moe_cfg\.dispatcher[[:space:]]*=[[:space:]]*"all2all"/a\
assert moe_cfg.mtp_config is not None\
moe_cfg.mtp_config.loss_scaling_factor = 0.0' "${TEST_CONFIG}"

if ! grep -Eq 'dispatcher[[:space:]]*=[[:space:]]*"all2all"' "${TEST_CONFIG}"; then
  echo "ERROR: generated config is not using dispatcher=all2all" >&2
  exit 1
fi
if ! grep -Eq 'sparse_mla_backend[[:space:]]*=[[:space:]]*"tilelang"' "${TEST_CONFIG}"; then
  echo "ERROR: generated config is not using sparse_mla_backend=tilelang" >&2
  exit 1
fi
if [[ $(grep -Ec 'mtp_config\.loss_scaling_factor[[:space:]]*=[[:space:]]*0\.0' "${TEST_CONFIG}" || true) -ne 1 ]]; then
  echo "ERROR: expected exactly one MTP loss_scaling_factor=0.0 assignment" >&2
  exit 1
fi
if ! grep -Eq 'debug_skip_save[[:space:]]*=[[:space:]]*True' "${TEST_CONFIG}"; then
  echo "ERROR: debug_skip_save=True is missing; refusing to risk a large checkpoint write" >&2
  exit 1
fi

echo "===== GLM-5.2 MTP-loss-zero configuration ====="
echo "repo: ${REPO_ROOT}"
echo "commit: $(git rev-parse HEAD)"
echo "python: ${ENV_PYTHON}"
echo "model: ${MODEL_PATH}"
echo "data: ${ALPACA_PATH}"
echo "base config: ${BASE_CONFIG}"
echo "test config: ${TEST_CONFIG}"
echo "experiment root: ${EXPERIMENT_ROOT}"
echo "run count: ${RUN_COUNT}"
grep -E 'dispatcher|sparse_mla_backend|mtp_config|loss_scaling_factor|debug_skip_save' "${TEST_CONFIG}"
df -h "${REPRO_ROOT}"

echo
echo "===== Runtime and config preflight ====="
# The XTuner autotest config reads WORK_DIR while the Python module is being
# imported. Give the import-only preflight its own harmless directory; every
# actual run overrides WORK_DIR below.
export WORK_DIR="${EXPERIMENT_ROOT}/preflight"
export GITHUB_RUN_ID="manual-${EXPERIMENT_TAG}-preflight"
mkdir -p "${WORK_DIR}"
"${ENV_PYTHON}" - <<'PY'
import os
import runpy

import torch
import xtuner
from xtuner.v1.ops.sparse_mla import ensure_tilelang_runtime_available

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA device count:", torch.cuda.device_count())
print("GPU 0:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("xtuner:", xtuner.__file__)

if torch.__version__.split("+")[0] != "2.9.1":
    raise RuntimeError(f"Expected Torch 2.9.1 CI environment, found {torch.__version__}")
if torch.version.cuda != "12.8":
    raise RuntimeError(f"Expected CUDA 12.8 Torch build, found {torch.version.cuda}")
if torch.cuda.device_count() != 8:
    raise RuntimeError(f"Expected 8 visible GPUs, found {torch.cuda.device_count()}")

ensure_tilelang_runtime_available()

namespace = runpy.run_path(os.environ["TEST_CONFIG"])
moe_cfg = namespace["moe_cfg"]
if moe_cfg.mtp_config is None:
    raise RuntimeError("MTP block is unexpectedly disabled; this experiment must keep it enabled")
if moe_cfg.mtp_config.loss_scaling_factor != 0.0:
    raise RuntimeError(
        f"Expected MTP loss scale 0.0, found {moe_cfg.mtp_config.loss_scaling_factor}"
    )
if moe_cfg.dispatcher != "all2all":
    raise RuntimeError(f"Expected all2all dispatcher, found {moe_cfg.dispatcher}")
if moe_cfg.attention.sparse_mla_backend != "tilelang":
    raise RuntimeError(
        f"Expected TileLang sparse MLA, found {moe_cfg.attention.sparse_mla_backend}"
    )

print("main layers:", moe_cfg.num_hidden_layers)
print("MTP layers:", moe_cfg.mtp_config.num_layers)
print("MTP loss scale:", moe_cfg.mtp_config.loss_scaling_factor)
print("dispatcher:", moe_cfg.dispatcher)
print("sparse MLA backend:", moe_cfg.attention.sparse_mla_backend)
print("TileLang runtime: OK")
print("PREFLIGHT PASSED")
PY

for RUN_INDEX in $(seq 1 "${RUN_COUNT}"); do
  RUN_NAME="mtp-loss0-run${RUN_INDEX}"
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
    --config "${TEST_CONFIG}" \
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
paths = sorted(tracker_root.glob("mtp-loss0-run*.jsonl"))
if len(paths) < 2:
    raise RuntimeError(f"Need at least two trackers, found {len(paths)}")

requested_metrics = (
    "grad_norm",
    "loss/local_loss",
    "loss/reduced_llm_loss",
    "loss/reduced_mtp_loss",
    "loss/maxvio",
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
metrics = [
    metric
    for metric in requested_metrics
    if all(metric in row for rows in runs.values() for row in rows)
]
print("Compared metrics:", ", ".join(metrics))

if "loss/reduced_mtp_loss" in metrics:
    mtp_values = [
        float(row["loss/reduced_mtp_loss"])
        for rows in runs.values()
        for row in rows
    ]
    print(
        "MTP logged loss: "
        f"all_zero={all(value == 0.0 for value in mtp_values)}, "
        f"max_abs={max(abs(value) for value in mtp_values):.12g}"
    )

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
        exact_equal = all(item[3] == item[4] for item in comparisons)
        failures = [item for item in comparisons if round(item[0], 2) > 1e-6]
        if failures:
            first = failures[0]
            first_failure = f"position={first[2]}, training_step={first[1]}"
        else:
            first_failure = "none"

        print(
            f"{metric}: exact_equal={exact_equal}, "
            f"max_rel={worst[0]:.6%}, max_step={worst[1]}, "
            f"first_CI_failure={first_failure}, CI_failure_steps={len(failures)}"
        )

    left_step16 = next((row for row in left if row.get("step") == 16), None)
    right_step16 = next((row for row in right if row.get("step") == 16), None)
    if left_step16 is not None and right_step16 is not None:
        left_grad = float(left_step16["grad_norm"])
        right_grad = float(right_step16["grad_norm"])
        relative_error = abs(left_grad - right_grad) / max(abs(left_grad), 1e-12)
        print(
            "Step 16 grad_norm: "
            f"{left_name}={left_grad:.8f}, "
            f"{right_name}={right_grad:.8f}, "
            f"relative_error={relative_error:.6%}, "
            f"CI_rounded_error={round(relative_error, 2):.2f}"
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
