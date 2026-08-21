#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2 complete-MTP-removal experiment:
#   - use the original GLM-5.2-30B checkpoint (5 main layers + 1 MTP layer)
#   - build only the original 5 main layers; do not build or execute MTP
#   - ignore the unused MTP checkpoint tensors with strict_load=False
#   - keep Torch 2.9.1/CUDA 12.8, TileLang, all2all and other knobs unchanged
#   - run three independent 35-step SFT jobs without saving checkpoints

REPO_ROOT=${REPO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-glm52-repro}
ENV_PYTHON=${ENV_PYTHON:-/usr/bin/python}
REPRO_ROOT=${REPRO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/glm52-repro}
RUN_COUNT=${RUN_COUNT:-3}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-$(date +%Y%m%d%H%M%S)}

BASE_CONFIG=${BASE_CONFIG:-${REPO_ROOT}/autotest/config/glm5p2_30B_all2all_nosave.py}
TEST_CONFIG=${TEST_CONFIG:-${REPO_ROOT}/autotest/config/glm5p2_30B_all2all_nomtp5_nosave.py}
EXPERIMENT_ROOT=${REPRO_ROOT}/glm52-nomtp5-sft-${EXPERIMENT_TAG}
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
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: model config not found: ${MODEL_PATH}/config.json" >&2
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

# Start from the validated all2all/TileLang/no-save config. Disable MTP after
# Glm52MoEConfig.from_hf() has read the original checkpoint configuration.
cp "${BASE_CONFIG}" "${TEST_CONFIG}"

DISPATCHER_COUNT=$(grep -Ec 'moe_cfg\.dispatcher[[:space:]]*=[[:space:]]*"all2all"' "${TEST_CONFIG}" || true)
STRICT_COUNT=$(grep -Ec 'strict_load[[:space:]]*=[[:space:]]*True,' "${TEST_CONFIG}" || true)
if [[ "${DISPATCHER_COUNT}" -ne 1 ]]; then
  echo "ERROR: expected exactly one moe_cfg.dispatcher=all2all assignment; found ${DISPATCHER_COUNT}" >&2
  exit 1
fi
if [[ "${STRICT_COUNT}" -ne 1 ]]; then
  echo "ERROR: expected exactly one strict_load=True assignment; found ${STRICT_COUNT}" >&2
  exit 1
fi

sed -i -E '/moe_cfg\.dispatcher[[:space:]]*=[[:space:]]*"all2all"/a\
assert moe_cfg.mtp_config is not None\
assert moe_cfg.num_hidden_layers == 5\
moe_cfg.mtp_config = None\
moe_cfg.num_nextn_predict_layers = 0\
if moe_cfg.attention.indexer_types is not None:\
    moe_cfg.attention.indexer_types = moe_cfg.attention.indexer_types[:moe_cfg.num_hidden_layers]' "${TEST_CONFIG}"

sed -i -E \
  's/strict_load[[:space:]]*=[[:space:]]*True,/strict_load=False,/' \
  "${TEST_CONFIG}"

if ! grep -Eq 'dispatcher[[:space:]]*=[[:space:]]*"all2all"' "${TEST_CONFIG}"; then
  echo "ERROR: generated config is not using dispatcher=all2all" >&2
  exit 1
fi
if ! grep -Eq 'sparse_mla_backend[[:space:]]*=[[:space:]]*"tilelang"' "${TEST_CONFIG}"; then
  echo "ERROR: generated config is not using sparse_mla_backend=tilelang" >&2
  exit 1
fi
if [[ $(grep -Ec 'moe_cfg\.mtp_config[[:space:]]*=[[:space:]]*None' "${TEST_CONFIG}" || true) -ne 1 ]]; then
  echo "ERROR: expected exactly one moe_cfg.mtp_config=None assignment" >&2
  exit 1
fi
if [[ $(grep -Ec 'moe_cfg\.num_nextn_predict_layers[[:space:]]*=[[:space:]]*0' "${TEST_CONFIG}" || true) -ne 1 ]]; then
  echo "ERROR: expected exactly one num_nextn_predict_layers=0 assignment" >&2
  exit 1
fi
if ! grep -Eq 'strict_load[[:space:]]*=[[:space:]]*False,' "${TEST_CONFIG}"; then
  echo "ERROR: generated config is not using strict_load=False" >&2
  exit 1
fi
if ! grep -Eq 'debug_skip_save[[:space:]]*=[[:space:]]*True' "${TEST_CONFIG}"; then
  echo "ERROR: debug_skip_save=True is missing; refusing to risk a large checkpoint write" >&2
  exit 1
fi

echo "===== GLM-5.2 five-layer/no-MTP configuration ====="
echo "repo: ${REPO_ROOT}"
echo "commit: $(git rev-parse HEAD)"
echo "python: ${ENV_PYTHON}"
echo "model: ${MODEL_PATH}"
echo "data: ${ALPACA_PATH}"
echo "base config: ${BASE_CONFIG}"
echo "test config: ${TEST_CONFIG}"
echo "experiment root: ${EXPERIMENT_ROOT}"
echo "run count: ${RUN_COUNT}"
grep -E 'dispatcher|sparse_mla_backend|mtp_config|num_nextn_predict_layers|indexer_types|strict_load|debug_skip_save' "${TEST_CONFIG}"
df -h "${REPRO_ROOT}"

echo
echo "===== Runtime and config preflight ====="
export WORK_DIR="${EXPERIMENT_ROOT}/preflight"
export GITHUB_RUN_ID="manual-${EXPERIMENT_TAG}-preflight"
mkdir -p "${WORK_DIR}"
"${ENV_PYTHON}" - <<'PY'
import json
import os
import runpy
from pathlib import Path

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

with (Path(os.environ["MODEL_PATH"]) / "config.json").open() as stream:
    hf_cfg = json.load(stream)
if hf_cfg.get("num_hidden_layers") != 5:
    raise RuntimeError(
        f"Expected original checkpoint num_hidden_layers=5, found {hf_cfg.get('num_hidden_layers')}"
    )
if hf_cfg.get("num_nextn_predict_layers") != 1:
    raise RuntimeError(
        "Expected original checkpoint num_nextn_predict_layers=1, "
        f"found {hf_cfg.get('num_nextn_predict_layers')}"
    )

ensure_tilelang_runtime_available()

namespace = runpy.run_path(os.environ["TEST_CONFIG"])
moe_cfg = namespace["moe_cfg"]
trainer = namespace["trainer"]

if moe_cfg.num_hidden_layers != 5:
    raise RuntimeError(f"Expected five main layers, found {moe_cfg.num_hidden_layers}")
if moe_cfg.mtp_config is not None:
    raise RuntimeError("MTP config is still enabled")
if moe_cfg.num_nextn_predict_layers != 0:
    raise RuntimeError(
        f"Expected num_nextn_predict_layers=0, found {moe_cfg.num_nextn_predict_layers}"
    )
indexer_types = moe_cfg.attention.indexer_types
if indexer_types is not None and len(indexer_types) != moe_cfg.num_hidden_layers:
    raise RuntimeError(
        f"Expected {moe_cfg.num_hidden_layers} main-layer indexer types, found {len(indexer_types)}"
    )
if trainer.strict_load is not False:
    raise RuntimeError(f"Expected strict_load=False, found {trainer.strict_load}")
if moe_cfg.dispatcher != "all2all":
    raise RuntimeError(f"Expected all2all dispatcher, found {moe_cfg.dispatcher}")
if moe_cfg.attention.sparse_mla_backend != "tilelang":
    raise RuntimeError(
        f"Expected TileLang sparse MLA, found {moe_cfg.attention.sparse_mla_backend}"
    )

print("checkpoint main layers:", hf_cfg["num_hidden_layers"])
print("checkpoint MTP layers:", hf_cfg["num_nextn_predict_layers"])
print("built main layers:", moe_cfg.num_hidden_layers)
print("built MTP config:", moe_cfg.mtp_config)
print("built next-token-prediction layers:", moe_cfg.num_nextn_predict_layers)
print("built indexer types:", indexer_types)
print("strict load:", trainer.strict_load)
print("dispatcher:", moe_cfg.dispatcher)
print("sparse MLA backend:", moe_cfg.attention.sparse_mla_backend)
print("TileLang runtime: OK")
print("PREFLIGHT PASSED")
PY

for RUN_INDEX in $(seq 1 "${RUN_COUNT}"); do
  RUN_NAME="nomtp5-run${RUN_INDEX}"
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

  if grep -q 'loss/reduced_mtp_loss' "${TRACKER}"; then
    echo "ERROR: tracker unexpectedly contains MTP loss: ${TRACKER}" >&2
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
paths = sorted(tracker_root.glob("nomtp5-run*.jsonl"))
if len(paths) < 2:
    raise RuntimeError(f"Need at least two trackers, found {len(paths)}")

requested_metrics = (
    "grad_norm",
    "loss/local_loss",
    "loss/reduced_llm_loss",
    "loss/maxvio",
    "lr",
    "runtime_info/text_tokens",
)


def load(path):
    rows = []
    with path.open() as stream:
        for position, line in enumerate(stream):
            row = json.loads(line)
            if "loss/reduced_mtp_loss" in row:
                raise RuntimeError(f"MTP loss unexpectedly found in {path} at position {position}")
            row["_position"] = position
            rows.append(row)
    return rows


runs = {path.stem: load(path) for path in paths}
metrics = [
    metric
    for metric in requested_metrics
    if all(metric in row for rows in runs.values() for row in rows)
]
print("MTP metric absent in every tracker: True")
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
