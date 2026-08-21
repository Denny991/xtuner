#!/usr/bin/env bash

# Patch the GLM-5.2 E2E metric gate to grad_norm P95 < 20%, run one complete
# 35-step SFT without saving the huge checkpoint, and validate the fresh
# tracker against the official CI baseline with XTuner's own checker.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-glm52-repro}"
CASE_NAME="${CASE_NAME:-glm5-2-sft-30B}"
THRESHOLD="${THRESHOLD:-0.20}"
AGGREGATE="${AGGREGATE:-95}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/mnt/shared-storage-user/ailab-sys/liutong/glm52-repro}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${RESULT_BASE}/glm52-ci-p95-${RUN_TAG}}"
RUN_ID="${RUN_ID:-manual-glm52-ci-p95-${RUN_TAG}}"

CASE_CONFIG="${REPO_ROOT}/autotest/config.yaml"
RESULT_TRACKER="${CURRENT_TRACKER:-}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${REPO_ROOT}" ]] || fail "repository does not exist: ${REPO_ROOT}"
[[ -f "${CASE_CONFIG}" ]] || fail "case config does not exist: ${CASE_CONFIG}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "python not found: ${PYTHON_BIN}"

mkdir -p "${EXPERIMENT_ROOT}"
cd "${REPO_ROOT}"

echo "===== Experiment ====="
echo "repo: ${REPO_ROOT}"
echo "commit: $(git rev-parse HEAD)"
echo "case: ${CASE_NAME}"
echo "grad_norm gate: P${AGGREGATE} < ${THRESHOLD}"
echo "experiment root: ${EXPERIMENT_ROOT}"

# Keep a recoverable copy of the exact configuration used before patching.
cp "${CASE_CONFIG}" "${EXPERIMENT_ROOT}/config.yaml.before"

# Patch only the grad_norm entry inside the requested case. This accepts both
# the original scalar form and a previously patched mapping form.
"${PYTHON_BIN}" - "${CASE_CONFIG}" "${CASE_NAME}" "${THRESHOLD}" "${AGGREGATE}" <<'PY'
import re
import sys
from pathlib import Path

import yaml


path = Path(sys.argv[1])
case_name = sys.argv[2]
threshold = float(sys.argv[3])
aggregate = int(sys.argv[4])
text = path.read_text(encoding="utf-8")

case_pattern = re.compile(
    rf"(?ms)^    {re.escape(case_name)}:\n.*?(?=^    \S|\Z)"
)
case_match = case_pattern.search(text)
if case_match is None:
    raise SystemExit(f"case not found in {path}: {case_name}")

block = case_match.group(0)
lines = block.splitlines(keepends=True)
metric_indices = [
    index
    for index, line in enumerate(lines)
    if re.match(r"^\s+grad_norm:\s*", line)
]
if len(metric_indices) != 1:
    raise SystemExit(
        f"expected exactly one grad_norm entry in {case_name}, found {len(metric_indices)}"
    )

index = metric_indices[0]
match = re.match(r"^(?P<indent>\s*)grad_norm:\s*(?P<value>.*?)\s*$", lines[index])
if match is None:
    raise SystemExit("failed to parse grad_norm entry")

indent = match.group("indent")
end = index + 1
if not match.group("value"):
    while end < len(lines):
        stripped = lines[end].strip()
        next_indent = len(lines[end]) - len(lines[end].lstrip())
        if stripped and next_indent <= len(indent):
            break
        end += 1

replacement = [
    f"{indent}grad_norm:\n",
    f"{indent}    threshold: {threshold:.2f}\n",
    f"{indent}    aggregate: {aggregate}\n",
]
lines[index:end] = replacement
patched_block = "".join(lines)
patched_text = text[: case_match.start()] + patched_block + text[case_match.end() :]
path.write_text(patched_text, encoding="utf-8")

config = yaml.safe_load(patched_text)
steps = config["case"][case_name]
metric = steps[0]["assert_info"]["check_metrics"]["grad_norm"]
expected = {"threshold": threshold, "aggregate": aggregate}
if metric != expected:
    raise SystemExit(f"patched metric mismatch: expected {expected}, got {metric}")
print(f"patched {path}: grad_norm={metric}")
PY

cp "${CASE_CONFIG}" "${EXPERIMENT_ROOT}/config.yaml.after"

echo
echo "===== Patched case ====="
rg -n -A45 "^    ${CASE_NAME}:" "${CASE_CONFIG}" || true
git diff --check -- autotest/config.yaml
git diff -- autotest/config.yaml || true

# Extract the training config, baseline path, GPU count, and the environment
# declared by this case. Default train envs are applied before case overrides.
mapfile -t CASE_METADATA < <(
  "${PYTHON_BIN}" - "${CASE_CONFIG}" "${CASE_NAME}" <<'PY'
import os
import sys

import yaml


with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
case_name = sys.argv[2]
step = config["case"][case_name][0]
base_root = config["base_path"]["base_baseline_path"]
base_metric = step["assert_info"]["base_metric"]
default_gpus = config["default_config"]["train"]["resource"].get("gpus_per_task", 8)
case_gpus = step.get("resource", {}).get("gpus_per_task", default_gpus)
print(step["parameters"]["config"])
print(os.path.join(base_root, base_metric))
print(case_gpus)
PY
)

[[ "${#CASE_METADATA[@]}" -eq 3 ]] || fail "failed to read case metadata"
TRAIN_CONFIG="${CASE_METADATA[0]}"
BASELINE_TRACKER="${CASE_METADATA[1]}"
CASE_GPUS="${CASE_METADATA[2]}"
[[ "${TRAIN_CONFIG}" = /* ]] || TRAIN_CONFIG="${REPO_ROOT}/${TRAIN_CONFIG}"

[[ -f "${TRAIN_CONFIG}" ]] || fail "training config does not exist: ${TRAIN_CONFIG}"
[[ -f "${BASELINE_TRACKER}" ]] || fail "official baseline does not exist: ${BASELINE_TRACKER}"
[[ "${NPROC_PER_NODE}" -eq "${CASE_GPUS}" ]] || fail \
  "NPROC_PER_NODE=${NPROC_PER_NODE}, but case requests ${CASE_GPUS} GPUs"

while IFS= read -r env_item; do
  [[ -z "${env_item}" ]] && continue
  export "${env_item}"
  echo "export ${env_item}"
done < <(
  "${PYTHON_BIN}" - "${CASE_CONFIG}" "${CASE_NAME}" <<'PY'
import sys

import yaml


with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
step = config["case"][sys.argv[2]][0]
default_envs = config["default_config"]["train"].get("resource", {}).get("envs", [])
case_envs = step.get("resource", {}).get("envs", [])
for item in [*default_envs, *case_envs]:
    print(item)
PY
)

echo
echo "===== Runtime inputs ====="
echo "python: $(command -v "${PYTHON_BIN}")"
"${PYTHON_BIN}" -c 'import torch; print("torch:", torch.__version__); print("CUDA:", torch.version.cuda); print("GPU count:", torch.cuda.device_count())'
echo "training config: ${TRAIN_CONFIG}"
echo "official baseline: ${BASELINE_TRACKER}"
echo "baseline records: $(wc -l < "${BASELINE_TRACKER}")"

if [[ -z "${RESULT_TRACKER}" ]]; then
  SAFE_TRAIN_CONFIG="${EXPERIMENT_ROOT}/glm5p2_30B_ci_p95_nosave.py"
  cp "${TRAIN_CONFIG}" "${SAFE_TRAIN_CONFIG}"

  # Metric validation does not need a hundreds-of-GB checkpoint. Only disable
  # the final save; model/data/optimizer/backend and all training steps stay the same.
  "${PYTHON_BIN}" - "${SAFE_TRAIN_CONFIG}" <<'PY'
import re
import sys
from pathlib import Path


path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if re.search(r"(?m)^\s*debug_skip_save\s*=", text):
    text = re.sub(
        r"(?m)^(\s*debug_skip_save\s*=\s*).*$",
        r"\1True,",
        text,
        count=1,
    )
else:
    text, count = re.subn(
        r"(?m)^(trainer\s*=\s*TrainerConfig\(\s*)$",
        r"\1\n    debug_skip_save=True,",
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("failed to add debug_skip_save=True")
path.write_text(text, encoding="utf-8")
if "debug_skip_save=True" not in text.replace(" ", ""):
    raise SystemExit("debug_skip_save=True verification failed")
print(f"no-save training config: {path}")
PY

  export PYTHONNOUSERSITE=1
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export GITHUB_RUN_ID="${RUN_ID}"
  export WORK_DIR="${EXPERIMENT_ROOT}/work"
  export GITHUB_STEP_SUMMARY="${EXPERIMENT_ROOT}/github-step-summary.md"
  mkdir -p "${WORK_DIR}"

  echo
  echo "===== Starting complete SFT ====="
  echo "WORK_DIR=${WORK_DIR}"
  set +e
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    xtuner/v1/train/cli/sft.py \
    --config "${SAFE_TRAIN_CONFIG}" \
    2>&1 | tee "${EXPERIMENT_ROOT}/train.log"
  TRAIN_STATUS=${PIPESTATUS[0]}
  set -e
  [[ "${TRAIN_STATUS}" -eq 0 ]] || fail "training failed with status ${TRAIN_STATUS}"

  RESULT_TRACKER="$(
    find "${WORK_DIR}" -type f -path '*/logs/exp_tracking/rank0/tracker.jsonl' \
      | sort | tail -1
  )"
fi

[[ -n "${RESULT_TRACKER}" ]] || fail "current tracker was not found"
[[ -f "${RESULT_TRACKER}" ]] || fail "current tracker does not exist: ${RESULT_TRACKER}"

BASELINE_LINES="$(wc -l < "${BASELINE_TRACKER}")"
CURRENT_LINES="$(wc -l < "${RESULT_TRACKER}")"
[[ "${CURRENT_LINES}" -eq "${BASELINE_LINES}" ]] || fail \
  "tracker length mismatch: baseline=${BASELINE_LINES}, current=${CURRENT_LINES}"

cp "${RESULT_TRACKER}" "${EXPERIMENT_ROOT}/current-tracker.jsonl"

echo
echo "===== Running XTuner CI metric checker ====="
echo "baseline: ${BASELINE_TRACKER}"
echo "current: ${RESULT_TRACKER}"

export GITHUB_RUN_ID="${RUN_ID}"
export GITHUB_STEP_SUMMARY="${EXPERIMENT_ROOT}/github-step-summary.md"
"${PYTHON_BIN}" - \
  "${REPO_ROOT}" \
  "${CASE_CONFIG}" \
  "${CASE_NAME}" \
  "${BASELINE_TRACKER}" \
  "${RESULT_TRACKER}" <<'PY'
import sys

import yaml


repo_root, config_path, case_name, baseline, current = sys.argv[1:]
sys.path.insert(0, f"{repo_root}/autotest")
sys.path.insert(0, repo_root)

from utils.check_metric import check_result


with open(config_path, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
metrics = config["case"][case_name][0]["assert_info"]["check_metrics"]
print("check_metrics:", metrics)
passed, detail = check_result(case_name, baseline, current, metrics)
print("CI_METRIC_CHECK_PASSED:", passed)
print(detail)
raise SystemExit(0 if passed else 2)
PY

echo
echo "===== Validation passed ====="
echo "experiment root: ${EXPERIMENT_ROOT}"
echo "tracker: ${EXPERIMENT_ROOT}/current-tracker.jsonl"
echo "train log: ${EXPERIMENT_ROOT}/train.log"
echo "metric report: ${REPO_ROOT}/${RUN_ID}/${CASE_NAME}_comparison.png"

