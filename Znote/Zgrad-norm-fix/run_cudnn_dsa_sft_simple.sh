#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/mnt/shared-storage-user/ailab-sys/liutong/xtuner-glm52-repro
ENV_PYTHON=/mnt/shared-storage-user/llmrazor-share/comm_env/glm52-pt121-cu132/bin/python
REPRO_ROOT=/mnt/shared-storage-user/ailab-sys/liutong/glm52-repro
BASE_CONFIG="$REPO_ROOT/autotest/config/glm5p2_30B_all2all_nosave.py"
DSA_CONFIG="$REPO_ROOT/autotest/config/glm5p2_30B_all2all_cudnn_dsa_nosave.py"

TAG=$(date +%Y%m%d%H%M%S)
EXP_ROOT="$REPRO_ROOT/cudnn-dsa-sft-$TAG"
TRACKER_ROOT="$EXP_ROOT/trackers"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

test -x "$ENV_PYTHON"
test -f "$BASE_CONFIG"
mkdir -p "$EXP_ROOT" "$TRACKER_ROOT"
cd "$REPO_ROOT"

# Generate a config by changing exactly one setting.
cp "$BASE_CONFIG" "$DSA_CONFIG"

OLD_LINE='moe_cfg.attention.sparse_mla_backend = "tilelang"'
NEW_LINE='moe_cfg.attention.sparse_mla_backend = "cudnn_dsa"'

if [[ $(grep -Fc "$OLD_LINE" "$DSA_CONFIG") -ne 1 ]]; then
  echo "ERROR: expected exactly this line in $DSA_CONFIG:" >&2
  echo "$OLD_LINE" >&2
  exit 1
fi

sed -i "s|$OLD_LINE|$NEW_LINE|" "$DSA_CONFIG"

grep -Fq 'moe_cfg.dispatcher = "all2all"' "$DSA_CONFIG"
grep -Fq "$NEW_LINE" "$DSA_CONFIG"

if ! grep -Eq 'debug_skip_save[[:space:]]*=[[:space:]]*True' "$DSA_CONFIG"; then
  echo "ERROR: debug_skip_save=True is missing; refusing to write checkpoints" >&2
  exit 1
fi

echo "===== Config ====="
echo "commit: $(git rev-parse HEAD)"
echo "config: $DSA_CONFIG"
echo "experiment: $EXP_ROOT"
grep -E 'dispatcher|sparse_mla_backend|debug_skip_save' "$DSA_CONFIG"
df -h "$REPRO_ROOT"

echo "===== Preflight ====="
"$ENV_PYTHON" -c '
import torch
import xtuner
from xtuner.v1.ops.sparse_mla import ensure_cudnn_dsa_runtime_available
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
print("xtuner:", xtuner.__file__)
assert torch.cuda.device_count() == 8
ensure_cudnn_dsa_runtime_available()
print("cuDNN DSA: OK")
'

for RUN in 1 2 3; do
  RUN_NAME="cudnn-dsa-run$RUN"
  WORK_DIR="$EXP_ROOT/$RUN_NAME"
  LOG_PATH="$WORK_DIR/train.log"

  mkdir -p "$WORK_DIR"
  export WORK_DIR
  export GITHUB_RUN_ID="manual-$TAG-$RUN_NAME"

  echo "===== Starting $RUN_NAME ====="

  set +e
  "$ENV_PYTHON" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=8 \
    xtuner/v1/train/cli/sft.py \
    --config "$DSA_CONFIG" \
    2>&1 | tee "$LOG_PATH"
  STATUS=${PIPESTATUS[0]}
  set -e

  if [[ $STATUS -ne 0 ]]; then
    echo "ERROR: $RUN_NAME failed with status $STATUS" >&2
    echo "log: $LOG_PATH" >&2
    exit "$STATUS"
  fi

  TRACKER=$(find "$WORK_DIR" \
    -type f \
    -path '*/logs/exp_tracking/rank0/tracker.jsonl' \
    | sort \
    | tail -1)

  if [[ -z "$TRACKER" ]]; then
    echo "ERROR: tracker not found below $WORK_DIR" >&2
    exit 1
  fi

  LINES=$(wc -l < "$TRACKER")
  if [[ $LINES -ne 35 ]]; then
    echo "ERROR: expected 35 tracker lines, found $LINES" >&2
    exit 1
  fi

  cp "$TRACKER" "$TRACKER_ROOT/$RUN_NAME.jsonl"
  echo "completed: $RUN_NAME, tracker lines: $LINES"
done

echo "===== Tracker summary ====="
wc -l "$TRACKER_ROOT"/*.jsonl

echo "===== Pairwise grad_norm comparison ====="
"$ENV_PYTHON" - "$TRACKER_ROOT" <<'PY'
import itertools
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = sorted(root.glob("cudnn-dsa-run*.jsonl"))
runs = {}

for path in paths:
    with path.open() as stream:
        runs[path.stem] = [json.loads(line) for line in stream]

for left_name, right_name in itertools.combinations(runs, 2):
    left = runs[left_name]
    right = runs[right_name]
    values = []

    for position, (a, b) in enumerate(zip(left, right, strict=True)):
        assert a["step"] == b["step"]
        av = float(a["grad_norm"])
        bv = float(b["grad_norm"])
        rel = abs(av - bv) / max(abs(av), 1e-12)
        values.append((rel, position, int(a["step"]), av, bv))

    worst = max(values)
    failed = [item for item in values if round(item[0], 2) > 1e-6]

    if failed:
        first = failed[0]
        first_text = f"position={first[1]}, training_step={first[2]}"
    else:
        first_text = "none"

    print(f"{left_name} vs {right_name}")
    print(
        f"  max_rel={worst[0]:.6%}, max_step={worst[2]}, "
        f"first_CI_failure={first_text}, CI_failure_steps={len(failed)}"
    )

print("trackers:")
for path in paths:
    print(path)
PY

echo "===== Completed ====="
echo "experiment: $EXP_ROOT"
echo "trackers: $TRACKER_ROOT"
