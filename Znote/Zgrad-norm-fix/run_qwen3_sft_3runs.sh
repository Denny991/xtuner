#!/usr/bin/env bash
set -euo pipefail

# Qwen3-30B-A3B negative-control experiment for the GLM-5.2 grad_norm issue.
# Runs the same 35-step XTuner SFT three times on 8 H200 GPUs and compares
# trackers using the current CI checker's rounded relative-error behavior.
REPO_ROOT=${REPO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-glm52-repro}
ENV_PYTHON=${ENV_PYTHON:-/usr/bin/python}
REPRO_ROOT=${REPRO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/glm52-repro}
RUN_COUNT=${RUN_COUNT:-3}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-$(date +%Y%m%d%H%M%S)}

BASE_CONFIG=${BASE_CONFIG:-${REPO_ROOT}/autotest/config/qwen3_moe_30BA3_ep8.py}
TEST_CONFIG=${TEST_CONFIG:-${REPO_ROOT}/autotest/config/qwen3_moe_30BA3_ep4_gbs8_35step_nosave.py}
EXPERIMENT_ROOT=${REPRO_ROOT}/qwen3-control-sft-${EXPERIMENT_TAG}
TRACKER_ROOT=${EXPERIMENT_ROOT}/trackers

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export QWEN3_MOE_PATH=${QWEN3_MOE_PATH:-/mnt/shared-storage-user/llmrazor-share/model/Qwen3-30B-A3B}
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
  echo "ERROR: CI-environment Python is not executable: ${ENV_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "ERROR: base config not found: ${BASE_CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${QWEN3_MOE_PATH}" ]]; then
  echo "ERROR: Qwen model directory not found: ${QWEN3_MOE_PATH}" >&2
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

# Generate a deterministic 35-step/no-save control config from the official
# Qwen3 autotest config. EP, GBS, training length and seed match the GLM
# all2all experiment as closely as the model-specific config allows.
cp "${BASE_CONFIG}" "${TEST_CONFIG}"

EP_COUNT=$(grep -Ec 'Qwen3MoE30BA3Config\(ep_size=8,[[:space:]]*compile_cfg=False\)' "${TEST_CONFIG}" || true)
GBS_COUNT=$(grep -Ec 'global_batch_size[[:space:]]*=[[:space:]]*16,' "${TEST_CONFIG}" || true)
EPOCH_COUNT=$(grep -Ec 'total_epoch[[:space:]]*=[[:space:]]*1,' "${TEST_CONFIG}" || true)
SEED_COUNT=$(grep -Ec 'seed[[:space:]]*=[[:space:]]*0,' "${TEST_CONFIG}" || true)

if [[ "${EP_COUNT}" -ne 1 || "${GBS_COUNT}" -ne 1 || "${EPOCH_COUNT}" -ne 1 || "${SEED_COUNT}" -ne 1 ]]; then
  echo "ERROR: unexpected base-config structure" >&2
  echo "Qwen3 EP8 config count: ${EP_COUNT}" >&2
  echo "global_batch_size=16 count: ${GBS_COUNT}" >&2
  echo "total_epoch=1 count: ${EPOCH_COUNT}" >&2
  echo "seed=0 count: ${SEED_COUNT}" >&2
  exit 1
fi

sed -i -E 's/Qwen3MoE30BA3Config\(ep_size=8,/Qwen3MoE30BA3Config(ep_size=4,/' "${TEST_CONFIG}"
sed -i -E '/moe_cfg[[:space:]]*=[[:space:]]*Qwen3MoE30BA3Config/a\moe_cfg.dispatcher = "all2all"' "${TEST_CONFIG}"
sed -i -E 's/global_batch_size[[:space:]]*=[[:space:]]*16,/global_batch_size=8,/' "${TEST_CONFIG}"
sed -i -E 's/total_epoch[[:space:]]*=[[:space:]]*1,/total_step=35,/' "${TEST_CONFIG}"
sed -i -E '/seed[[:space:]]*=[[:space:]]*0,/a\    debug_skip_save=True,' "${TEST_CONFIG}"

if ! grep -Eq 'Qwen3MoE30BA3Config\(ep_size=4,[[:space:]]*compile_cfg=False\)' "${TEST_CONFIG}"; then
  echo "ERROR: generated config is not Qwen3-30B-A3B EP4" >&2
  exit 1
fi
if ! grep -Eq 'dispatcher[[:space:]]*=[[:space:]]*"all2all"' "${TEST_CONFIG}"; then
  echo "ERROR: generated config is not using dispatcher=all2all" >&2
  exit 1
fi
if ! grep -Eq 'global_batch_size[[:space:]]*=[[:space:]]*8,' "${TEST_CONFIG}"; then
  echo "ERROR: generated config is not global_batch_size=8" >&2
  exit 1
fi
if ! grep -Eq 'total_step[[:space:]]*=[[:space:]]*35,' "${TEST_CONFIG}"; then
  echo "ERROR: generated config is not total_step=35" >&2
  exit 1
fi
if ! grep -Eq 'debug_skip_save[[:space:]]*=[[:space:]]*True' "${TEST_CONFIG}"; then
  echo "ERROR: debug_skip_save=True is missing; refusing to risk a large checkpoint write" >&2
  exit 1
fi

echo "===== Qwen3 control configuration ====="
echo "repo: ${REPO_ROOT}"
echo "commit: $(git rev-parse HEAD)"
echo "python: ${ENV_PYTHON}"
echo "model: ${QWEN3_MOE_PATH}"
echo "data: ${ALPACA_PATH}"
echo "base config: ${BASE_CONFIG}"
echo "test config: ${TEST_CONFIG}"
echo "experiment root: ${EXPERIMENT_ROOT}"
echo "run count: ${RUN_COUNT}"
grep -E 'Qwen3MoE30BA3Config|dispatcher|global_batch_size|total_step|seed|debug_skip_save' "${TEST_CONFIG}"
df -h "${REPRO_ROOT}"

echo
echo "===== Runtime preflight ====="
"${ENV_PYTHON}" - <<'PY'
import torch
import xtuner

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA device count:", torch.cuda.device_count())
print("GPU 0:", torch.cuda.get_device_name(0))
print("xtuner:", xtuner.__file__)

if torch.__version__.split("+")[0] != "2.9.1":
    raise RuntimeError(f"Expected Torch 2.9.1 CI environment, found {torch.__version__}")
if torch.version.cuda != "12.8":
    raise RuntimeError(f"Expected CUDA 12.8 Torch build, found {torch.version.cuda}")
if torch.cuda.device_count() != 8:
    raise RuntimeError(f"Expected 8 visible GPUs, found {torch.cuda.device_count()}")

print("PREFLIGHT PASSED")
PY

for RUN_INDEX in $(seq 1 "${RUN_COUNT}"); do
  RUN_NAME="qwen3-run${RUN_INDEX}"
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
paths = sorted(tracker_root.glob("qwen3-run*.jsonl"))
if len(paths) < 2:
    raise RuntimeError(f"Need at least two trackers, found {len(paths)}")

requested_metrics = (
    "grad_norm",
    "loss/local_loss",
    "loss/reduced_llm_loss",
    "loss/reduced_balancing_loss",
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

    print()

print("Experiment trackers:")
for path in paths:
    print(path)
PY

echo
echo "===== Experiment completed ====="
echo "Experiment root: ${EXPERIMENT_ROOT}"
echo "Trackers: ${TRACKER_ROOT}"
