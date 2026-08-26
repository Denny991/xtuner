#!/usr/bin/env bash

# Run a same-topology 8-GPU A/B experiment at XTuner commit 2c60087:
#   A: the original AdamW-only clipping policy
#   B: the local three-line clip-all patch
#
# The script always restores the clip-all source file before it exits.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-qwen35-clip-debug-2c60087}"
ENV_PYTHON="${ENV_PYTHON:-/mnt/shared-storage-user/llmrazor-share/comm_env/glm52-pt121-cu132/bin/python}"
REPRO_ROOT="${REPRO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/qwen35-clip-ab}"

MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/llmrazor-share/model/Qwen3.5-35B-A3B}"
DATA_PATH="${DATA_PATH:-/mnt/shared-storage-user/llmrazor-share/data/ci_vl}"
MEDIA_ROOT="${MEDIA_ROOT:-/mnt/shared-storage-user/llmrazor-share/data/ci_vl}"

EXPECTED_HEAD="2c60087fe8d9380dab47d5b743e1784904c8360d"
SOURCE_CONFIG="autotest/config/qwen3_5_ep2_mtp4_vl.py"
OPTIM_REL="xtuner/v1/config/optim.py"

TAG="${TAG:-$(date +%Y%m%d%H%M%S)}"
EXPERIMENT_ROOT="$REPRO_ROOT/qwen35-clip-ab-8gpu-$TAG"
TRACKER_ROOT="$EXPERIMENT_ROOT/trackers"
RUNTIME_CONFIG="$EXPERIMENT_ROOT/qwen3_5_ep2_mtp4_vl_8gpu_nosave.py"
ORIGINAL_OPTIM="$EXPERIMENT_ROOT/optim.adamw_only.py"
CLIP_ALL_OPTIM="$EXPERIMENT_ROOT/optim.clip_all.py"
EXPECTED_CLIP_ALL_OPTIM="$EXPERIMENT_ROOT/optim.expected_clip_all.py"

cd "$REPO_ROOT"

if [[ ! -x "$ENV_PYTHON" ]]; then
    echo "ERROR: Python is not executable: $ENV_PYTHON" >&2
    exit 1
fi

if [[ "$(git rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
    echo "ERROR: expected HEAD $EXPECTED_HEAD, got $(git rev-parse HEAD)" >&2
    exit 1
fi

for path in "$SOURCE_CONFIG" "$OPTIM_REL" "$MODEL_PATH" "$DATA_PATH" "$MEDIA_ROOT"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: required path does not exist: $path" >&2
        exit 1
    fi
done

mkdir -p "$TRACKER_ROOT"

# HEAD contains the original AdamW-only policy. The working tree is expected
# to contain exactly the previously prepared clip-all policy.
other_tracked_changes=$(git diff HEAD --name-only \
    | grep -v -x "$OPTIM_REL" \
    | grep -v -x "$SOURCE_CONFIG" \
    | grep -v -x 'autotest/config.yaml' \
    || true)
if [[ -n "$other_tracked_changes" ]]; then
    echo "ERROR: tracked changes other than $OPTIM_REL would make the A/B experiment ambiguous:" >&2
    echo "$other_tracked_changes" >&2
    exit 1
fi

git show "HEAD:$OPTIM_REL" > "$ORIGINAL_OPTIM"
cp "$OPTIM_REL" "$CLIP_ALL_OPTIM"
sed 's/clip_grad=False/clip_grad=True/g' "$ORIGINAL_OPTIM" > "$EXPECTED_CLIP_ALL_OPTIM"

original_false_count=$(grep -c 'clip_grad=False' "$ORIGINAL_OPTIM" || true)
original_true_count=$(grep -c 'clip_grad=True' "$ORIGINAL_OPTIM" || true)
clip_all_false_count=$(grep -c 'clip_grad=False' "$CLIP_ALL_OPTIM" || true)
clip_all_true_count=$(grep -c 'clip_grad=True' "$CLIP_ALL_OPTIM" || true)

if [[ "$original_false_count" -ne 3 || "$original_true_count" -ne 1 ]]; then
    echo "ERROR: unexpected AdamW-only policy counts: false=$original_false_count true=$original_true_count" >&2
    exit 1
fi

if [[ "$clip_all_false_count" -ne 0 || "$clip_all_true_count" -ne 4 ]]; then
    echo "ERROR: current working tree is not the expected three-line clip-all patch." >&2
    echo "       expected false=0 true=4, got false=$clip_all_false_count true=$clip_all_true_count" >&2
    exit 1
fi

if ! cmp -s "$EXPECTED_CLIP_ALL_OPTIM" "$CLIP_ALL_OPTIM"; then
    echo "ERROR: the working-tree optim.py contains changes beyond the expected three clip_grad flags." >&2
    diff -u "$EXPECTED_CLIP_ALL_OPTIM" "$CLIP_ALL_OPTIM" || true
    exit 1
fi

restore_clip_all() {
    cp "$CLIP_ALL_OPTIM" "$OPTIM_REL"
}
trap restore_clip_all EXIT

git show "HEAD:$SOURCE_CONFIG" > "$RUNTIME_CONFIG"
if ! grep -q 'debug_skip_save=True' "$RUNTIME_CONFIG"; then
    sed -i '/trainer = TrainerConfig(/a\    debug_skip_save=True,' "$RUNTIME_CONFIG"
fi

echo "===== Runtime configuration ====="
grep -nE 'ep_size|num_layers|global_batch_size|sp_size|pack_max_length|total_step|debug_skip_save' "$RUNTIME_CONFIG"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XTUNER_DETERMINISTIC=true
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MODEL_PATH DATA_PATH MEDIA_ROOT

"$ENV_PYTHON" - <<'PY'
import torch
import xtuner

print("Python/Torch preflight")
print("  torch:", torch.__version__)
print("  CUDA:", torch.version.cuda)
print("  visible GPUs:", torch.cuda.device_count())
print("  xtuner:", xtuner.__file__)
if torch.cuda.device_count() != 8:
    raise SystemExit(f"ERROR: expected 8 visible GPUs, got {torch.cuda.device_count()}")
PY

run_one() {
    local run_name="$1"
    local optim_source="$2"
    local run_root="$EXPERIMENT_ROOT/$run_name"
    local tracker
    local train_status

    cp "$optim_source" "$OPTIM_REL"

    export WORK_DIR="$run_root/work"
    export GITHUB_RUN_ID="manual-qwen35-$run_name-$TAG"
    mkdir -p "$WORK_DIR"

    echo
    echo "===== Starting $run_name ====="
    echo "WORK_DIR=$WORK_DIR"
    grep -n 'clip_grad=' "$OPTIM_REL"

    set +e
    "$ENV_PYTHON" -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc-per-node=8 \
        xtuner/v1/train/cli/sft.py \
        --config "$RUNTIME_CONFIG" \
        2>&1 | tee "$run_root/train.log"
    train_status=${PIPESTATUS[0]}
    set -e

    if [[ "$train_status" -ne 0 ]]; then
        echo "ERROR: $run_name failed with status $train_status" >&2
        return "$train_status"
    fi

    tracker=$(find "$WORK_DIR" \
        -type f \
        -path '*/logs/exp_tracking/rank0/tracker.jsonl' \
        | sort \
        | tail -1)

    if [[ -z "$tracker" ]]; then
        echo "ERROR: tracker not found for $run_name" >&2
        return 1
    fi

    if [[ "$(wc -l < "$tracker")" -ne 20 ]]; then
        echo "ERROR: expected 20 tracker rows for $run_name: $tracker" >&2
        wc -l "$tracker"
        return 1
    fi

    cp "$tracker" "$TRACKER_ROOT/$run_name.jsonl"
    echo "Completed $run_name"
    echo "Tracker: $TRACKER_ROOT/$run_name.jsonl"
}

# A is the exact source at 2c60087. B differs only in the three Muon
# parameter-group clip_grad flags.
run_one "adamw-only" "$ORIGINAL_OPTIM"
run_one "clip-all" "$CLIP_ALL_OPTIM"

restore_clip_all

echo
echo "===== Tracker summary ====="
wc -l "$TRACKER_ROOT"/*.jsonl

"$ENV_PYTHON" - \
    "$TRACKER_ROOT/adamw-only.jsonl" \
    "$TRACKER_ROOT/clip-all.jsonl" <<'PY'
import json
import sys

import numpy as np


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            step = row.get("step")
            if not isinstance(step, int):
                raise SystemExit(f"{path}:{line_number}: invalid step={step!r}")
            if step in rows:
                raise SystemExit(f"{path}:{line_number}: duplicate step={step}")
            rows[step] = row
    return rows


def relative_error(base, current):
    if abs(base) < 1e-10:
        return 0.0 if abs(current) <= 1e-10 else float("inf")
    return abs(current - base) / abs(base)


left_path, right_path = sys.argv[1:3]
left = load(left_path)
right = load(right_path)

if set(left) != set(right):
    raise SystemExit(
        "tracker step mismatch: "
        f"only_adamw_only={sorted(set(left) - set(right))}, "
        f"only_clip_all={sorted(set(right) - set(left))}"
    )

steps = sorted(left)

control_metrics = [
    "lr",
    "runtime_info/text_tokens",
    "runtime_info/seqlen_tokens",
    "runtime_info/approximate_total_consumed_tokens",
]

print("\n===== Control metrics =====")
for metric in control_metrics:
    present = [metric in left[step] and metric in right[step] for step in steps]
    if not any(present):
        print(f"{metric}: skipped (not logged)")
        continue
    if not all(present):
        raise SystemExit(f"{metric}: missing from part of the aligned tracker rows")
    differing_steps = [step for step in steps if left[step][metric] != right[step][metric]]
    print(f"{metric}: exact_equal={not differing_steps}, differing_steps={differing_steps}")

left_loss_metrics = {key for step in steps for key in left[step] if key.startswith("loss/")}
right_loss_metrics = {key for step in steps for key in right[step] if key.startswith("loss/")}
metrics = ["grad_norm", *sorted(left_loss_metrics & right_loss_metrics)]

print("\n===== AdamW-only vs clip-all =====")
for metric in metrics:
    missing_steps = [step for step in steps if metric not in left[step] or metric not in right[step]]
    if missing_steps:
        raise SystemExit(f"{metric}: missing at aligned steps {missing_steps}")

    a = np.asarray([float(left[step][metric]) for step in steps], dtype=np.float64)
    b = np.asarray([float(right[step][metric]) for step in steps], dtype=np.float64)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise SystemExit(f"{metric}: non-finite value found")
    abs_error = np.abs(a - b)
    rel_error = np.asarray([relative_error(base, current) for base, current in zip(a, b)], dtype=np.float64)
    max_index = int(np.argmax(rel_error))
    max_step = steps[max_index]

    print(
        f"{metric}: exact_equal={bool(np.array_equal(a, b))}, "
        f"p50_rel={np.percentile(rel_error, 50) * 100:.6f}%, "
        f"p80_rel={np.percentile(rel_error, 80) * 100:.6f}%, "
        f"p95_rel={np.percentile(rel_error, 95) * 100:.6f}%, "
        f"max_rel={rel_error[max_index] * 100:.6f}%, "
        f"max_abs={abs_error[max_index]:.8g}, step={max_step}, "
        f"adamw_only={a[max_index]:.8g}, clip_all={b[max_index]:.8g}"
    )
PY

echo
echo "===== Experiment completed ====="
echo "Experiment root: $EXPERIMENT_ROOT"
echo "Trackers: $TRACKER_ROOT"
echo "Source restored to clip-all: $REPO_ROOT/$OPTIM_REL"
