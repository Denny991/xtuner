#!/usr/bin/env bash
set -euo pipefail

# End-to-end determinism control for GLM-5.2 SFT.
#
# The correctness-first Torch SparseMLA backend is deterministic in the
# fixed-input microkernel test, but its O(S * topk * D) intermediates do not fit
# the original 16384-token job. Keep the real 8-GPU GLM SFT path and all other
# training knobs, shorten packed sequences to 1024, run exactly 35 steps three
# times, and compare the resulting trackers.
#
# This is a diagnostic control. It does not modify/fix the TileLang kernel.

REPO_ROOT=${REPO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-glm52-repro}
ENV_PYTHON=${ENV_PYTHON:-/usr/bin/python}
REPRO_ROOT=${REPRO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/glm52-repro}
RUN_COUNT=${RUN_COUNT:-3}
TRAIN_STEPS=${TRAIN_STEPS:-35}
SEQ_LEN=${SEQ_LEN:-1024}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-$(date +%Y%m%d%H%M%S)}

BASE_CONFIG=${BASE_CONFIG:-${REPO_ROOT}/autotest/config/glm5p2_30B_all2all_nosave.py}
TEST_CONFIG=${TEST_CONFIG:-${REPO_ROOT}/autotest/config/glm5p2_30B_all2all_torch_seq${SEQ_LEN}_nosave.py}
EXPERIMENT_ROOT=${REPRO_ROOT}/glm52-torch-sparse-mla-seq${SEQ_LEN}-sft-${EXPERIMENT_TAG}
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
export TEST_CONFIG TRAIN_STEPS SEQ_LEN

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${REPO_ROOT}/.git" ]] || fail "repository not found: ${REPO_ROOT}"
[[ -x "${ENV_PYTHON}" ]] || fail "Python is not executable: ${ENV_PYTHON}"
[[ -f "${BASE_CONFIG}" ]] || fail "base config not found: ${BASE_CONFIG}"
[[ -f "${MODEL_PATH}/config.json" ]] || fail "model config not found: ${MODEL_PATH}/config.json"
[[ -e "${ALPACA_PATH}" ]] || fail "Alpaca data path not found: ${ALPACA_PATH}"
[[ "${RUN_COUNT}" =~ ^[2-9][0-9]*$ ]] || fail "RUN_COUNT must be an integer >= 2"
[[ "${TRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "TRAIN_STEPS must be positive"
[[ "${SEQ_LEN}" =~ ^[1-9][0-9]*$ ]] || fail "SEQ_LEN must be positive"
(( SEQ_LEN % 64 == 0 )) || fail "SEQ_LEN must be divisible by 64"
(( SEQ_LEN <= 2048 )) || fail "Torch reference backend is memory-heavy; keep SEQ_LEN <= 2048"

mkdir -p "${EXPERIMENT_ROOT}" "${TRACKER_ROOT}"
cd "${REPO_ROOT}"

if [[ "$(git rev-parse HEAD)" != "51c775d8aa70a10f8e23092372b002ea9dbe3233" ]]; then
  fail "expected commit 51c775d8aa70a10f8e23092372b002ea9dbe3233, found $(git rev-parse HEAD)"
fi

cp "${BASE_CONFIG}" "${TEST_CONFIG}"

BACKEND_COUNT=$(grep -Ec 'sparse_mla_backend[[:space:]]*=[[:space:]]*"tilelang"' "${TEST_CONFIG}" || true)
TOKENIZE_COUNT=$(grep -Ec '(OpenaiTokenizeFunctionConfig|FTDPTokenizeFnConfig)\([^)]*max_length[[:space:]]*=[[:space:]]*[0-9]+' "${TEST_CONFIG}" || true)
PACK_COUNT=$(grep -Ec 'DataloaderConfig\([^)]*pack_max_length[[:space:]]*=[[:space:]]*[0-9]+' "${TEST_CONFIG}" || true)
EPOCH_COUNT=$(grep -Ec 'total_epoch[[:space:]]*=[[:space:]]*1,' "${TEST_CONFIG}" || true)

[[ "${BACKEND_COUNT}" -eq 1 ]] || fail "expected one TileLang backend assignment, found ${BACKEND_COUNT}"
[[ "${TOKENIZE_COUNT}" -eq 1 ]] || fail "expected one tokenizer max_length assignment, found ${TOKENIZE_COUNT}"
[[ "${PACK_COUNT}" -eq 1 ]] || fail "expected one pack_max_length assignment, found ${PACK_COUNT}"
[[ "${EPOCH_COUNT}" -eq 1 ]] || fail "expected one total_epoch=1 assignment, found ${EPOCH_COUNT}"

sed -i -E \
  's/(sparse_mla_backend[[:space:]]*=[[:space:]]*)"tilelang"/\1"torch"/' \
  "${TEST_CONFIG}"
sed -i -E \
  "s/((OpenaiTokenizeFunctionConfig|FTDPTokenizeFnConfig)\([^)]*max_length[[:space:]]*=[[:space:]]*)[0-9]+/\1${SEQ_LEN}/" \
  "${TEST_CONFIG}"
sed -i -E \
  "s/(DataloaderConfig\([^)]*pack_max_length[[:space:]]*=[[:space:]]*)[0-9]+/\1${SEQ_LEN}/" \
  "${TEST_CONFIG}"
sed -i -E \
  "s/total_epoch[[:space:]]*=[[:space:]]*1,/total_step=${TRAIN_STEPS},/" \
  "${TEST_CONFIG}"

grep -Eq 'sparse_mla_backend[[:space:]]*=[[:space:]]*"torch"' "${TEST_CONFIG}" \
  || fail "generated config is not using sparse_mla_backend=torch"
grep -Eq "(OpenaiTokenizeFunctionConfig|FTDPTokenizeFnConfig)\([^)]*max_length[[:space:]]*=[[:space:]]*${SEQ_LEN}" "${TEST_CONFIG}" \
  || fail "generated config tokenizer length is not ${SEQ_LEN}"
grep -Eq "DataloaderConfig\([^)]*pack_max_length[[:space:]]*=[[:space:]]*${SEQ_LEN}" "${TEST_CONFIG}" \
  || fail "generated config pack length is not ${SEQ_LEN}"
grep -Eq "total_step[[:space:]]*=[[:space:]]*${TRAIN_STEPS}," "${TEST_CONFIG}" \
  || fail "generated config total_step is not ${TRAIN_STEPS}"
grep -Eq 'debug_skip_save[[:space:]]*=[[:space:]]*True' "${TEST_CONFIG}" \
  || fail "debug_skip_save=True is missing; refusing to risk a large checkpoint write"
grep -Eq 'dispatcher[[:space:]]*=[[:space:]]*"all2all"' "${TEST_CONFIG}" \
  || fail "generated config is not using dispatcher=all2all"

echo "===== GLM-5.2 Torch SparseMLA SFT control ====="
echo "repo: ${REPO_ROOT}"
echo "commit: $(git rev-parse HEAD)"
echo "python: ${ENV_PYTHON}"
echo "model: ${MODEL_PATH}"
echo "data: ${ALPACA_PATH}"
echo "base config: ${BASE_CONFIG}"
echo "test config: ${TEST_CONFIG}"
echo "experiment root: ${EXPERIMENT_ROOT}"
echo "run count: ${RUN_COUNT}"
echo "training steps: ${TRAIN_STEPS}"
echo "sequence length: ${SEQ_LEN}"
grep -nE 'dispatcher|sparse_mla_backend|FTDPTokenizeFnConfig|DataloaderConfig|total_step|debug_skip_save' "${TEST_CONFIG}"
df -h "${REPRO_ROOT}"

echo
echo "===== Runtime and config preflight ====="
export WORK_DIR="${EXPERIMENT_ROOT}/preflight"
export GITHUB_RUN_ID="manual-${EXPERIMENT_TAG}-preflight"
mkdir -p "${WORK_DIR}"

CUDA_VISIBLE_DEVICES=0 "${ENV_PYTHON}" - <<'PY'
import os
import runpy

import torch
import xtuner

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA device count in preflight:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("xtuner:", xtuner.__file__)

if torch.__version__.split("+")[0] != "2.9.1":
    raise RuntimeError(f"Expected Torch 2.9.1 CI environment, found {torch.__version__}")
if torch.version.cuda != "12.8":
    raise RuntimeError(f"Expected CUDA 12.8 Torch build, found {torch.version.cuda}")

namespace = runpy.run_path(os.environ["TEST_CONFIG"])
moe_cfg = namespace["moe_cfg"]
trainer = namespace["trainer"]

expected_steps = int(os.environ["TRAIN_STEPS"])
expected_len = int(os.environ["SEQ_LEN"])

if moe_cfg.attention.sparse_mla_backend != "torch":
    raise RuntimeError(
        f"Expected Torch SparseMLA, found {moe_cfg.attention.sparse_mla_backend}"
    )
if moe_cfg.dispatcher != "all2all":
    raise RuntimeError(f"Expected all2all dispatcher, found {moe_cfg.dispatcher}")
if trainer.total_step != expected_steps or trainer.total_epoch is not None:
    raise RuntimeError(
        f"Expected total_step={expected_steps}, total_epoch=None; "
        f"found total_step={trainer.total_step}, total_epoch={trainer.total_epoch}"
    )
if trainer.debug_skip_save is not True:
    raise RuntimeError("debug_skip_save is not True")

tokenize_lengths = [entry["tokenize_fn"].max_length for entry in trainer.dataset_cfg]
if tokenize_lengths != [expected_len]:
    raise RuntimeError(f"Unexpected tokenizer max lengths: {tokenize_lengths}")
if trainer.dataloader_cfg.pack_max_length != expected_len:
    raise RuntimeError(
        f"Expected pack_max_length={expected_len}, "
        f"found {trainer.dataloader_cfg.pack_max_length}"
    )

torch.use_deterministic_algorithms(True)
print("strict deterministic algorithms:", torch.are_deterministic_algorithms_enabled())
print("SparseMLA backend:", moe_cfg.attention.sparse_mla_backend)
print("dispatcher:", moe_cfg.dispatcher)
print("tokenizer max lengths:", tokenize_lengths)
print("pack max length:", trainer.dataloader_cfg.pack_max_length)
print("training steps:", trainer.total_step)
print("debug_skip_save:", trainer.debug_skip_save)
print("PREFLIGHT PASSED")
PY

for RUN_INDEX in $(seq 1 "${RUN_COUNT}"); do
  RUN_NAME="torch-sparse-mla-run${RUN_INDEX}"
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
    if grep -Eqi 'out of memory|CUDA error: out of memory' "${TRAIN_LOG}"; then
      echo "Torch reference backend exhausted memory. Retry with SEQ_LEN=512; do not use the original 16384 length." >&2
    fi
    exit "${TRAIN_STATUS}"
  fi

  TRACKER=$(find "${WORK_DIR}" \
    -type f \
    -path '*/logs/exp_tracking/rank0/tracker.jsonl' \
    | sort \
    | tail -1)

  [[ -n "${TRACKER}" ]] || fail "tracker not found under ${WORK_DIR}"
  TRACKER_LINES=$(wc -l < "${TRACKER}")
  [[ "${TRACKER_LINES}" -eq "${TRAIN_STEPS}" ]] \
    || fail "expected ${TRAIN_STEPS} tracker records, found ${TRACKER_LINES}: ${TRACKER}"

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
paths = sorted(tracker_root.glob("torch-sparse-mla-run*.jsonl"))
if len(paths) < 2:
    raise RuntimeError(f"Need at least two trackers, found {len(paths)}")

requested_metrics = (
    "grad_norm",
    "loss/local_loss",
    "loss/reduced_llm_loss",
    "loss/reduced_mtp_loss",
    "loss/reduced_balancing_loss",
    "loss/maxvio",
    "lr",
    "runtime_info/text_tokens",
    "runtime_info/seqlen_tokens",
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

all_exact = True
for left_name, right_name in itertools.combinations(runs, 2):
    left = runs[left_name]
    right = runs[right_name]
    if len(left) != len(right):
        raise RuntimeError(
            f"Length mismatch: {left_name}={len(left)}, {right_name}={len(right)}"
        )

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
        all_exact = all_exact and exact_equal
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

print(f"ALL_COMPARED_METRICS_EXACT={all_exact}")
print("Experiment trackers:")
for path in paths:
    print(path)
PY

echo
echo "===== Experiment completed ====="
echo "Experiment root: ${EXPERIMENT_ROOT}"
echo "Trackers: ${TRACKER_ROOT}"
