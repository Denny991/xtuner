#!/usr/bin/env bash

# Offline runner for the pre-d1 Qwen3.5 reference (9e4e212) on the same 8-GPU topology
# used by run_qwen35_clip_ab_8gpu.sh, then compare all three trackers:
#   pre-d1 global clipping, 2c60087 AdamW-only, 2c60087 clip-all.
#
# The pre-d1 repository must already be prepared outside the training
# container. This script performs no network access and creates no worktree.

set -euo pipefail

SOURCE_REPO="${SOURCE_REPO:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-qwen35-clip-debug-2c60087}"
PRE_D1_WORKTREE="${PRE_D1_WORKTREE:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-qwen35-pre-d1-9e4e212}"
ENV_PYTHON="${ENV_PYTHON:-/mnt/shared-storage-user/llmrazor-share/comm_env/glm52-pt121-cu132/bin/python}"

AB_EXPERIMENT_ROOT="${AB_EXPERIMENT_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/qwen35-clip-ab/qwen35-clip-ab-8gpu-20260825060848}"

MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/llmrazor-share/model/Qwen3.5-35B-A3B}"
DATA_PATH="${DATA_PATH:-/mnt/shared-storage-user/llmrazor-share/data/ci_vl}"
MEDIA_ROOT="${MEDIA_ROOT:-/mnt/shared-storage-user/llmrazor-share/data/ci_vl}"

PRE_D1_COMMIT="9e4e21255511ec518821fd7eb9ca77db5cd15e7d"
FAIL_COMMIT="2c60087fe8d9380dab47d5b743e1784904c8360d"
CONFIG_REL="autotest/config/qwen3_5_ep2_mtp4_vl.py"

TAG="${TAG:-$(date +%Y%m%d%H%M%S)}"
REFERENCE_RUN_ROOT="$AB_EXPERIMENT_ROOT/pre-d1-9e4e212-$TAG"
RUNTIME_CONFIG="$REFERENCE_RUN_ROOT/qwen3_5_ep2_mtp4_vl_8gpu_nosave.py"
REFERENCE_TRACKER="$AB_EXPERIMENT_ROOT/trackers/pre-d1-9e4e212-$TAG.jsonl"
ADAMW_ONLY_TRACKER="$AB_EXPERIMENT_ROOT/trackers/adamw-only.jsonl"
CLIP_ALL_TRACKER="$AB_EXPERIMENT_ROOT/trackers/clip-all.jsonl"

if [[ ! -x "$ENV_PYTHON" ]]; then
    echo "ERROR: Python is not executable: $ENV_PYTHON" >&2
    exit 1
fi

for path in "$SOURCE_REPO" "$AB_EXPERIMENT_ROOT" "$ADAMW_ONLY_TRACKER" "$CLIP_ALL_TRACKER" \
    "$MODEL_PATH" "$DATA_PATH" "$MEDIA_ROOT"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: required path does not exist: $path" >&2
        exit 1
    fi
done

if [[ "$(git -C "$SOURCE_REPO" rev-parse HEAD)" != "$FAIL_COMMIT" ]]; then
    echo "ERROR: SOURCE_REPO is not at the expected failing commit $FAIL_COMMIT" >&2
    echo "       got $(git -C "$SOURCE_REPO" rev-parse HEAD)" >&2
    exit 1
fi

if [[ ! -d "$PRE_D1_WORKTREE" ]]; then
    echo "ERROR: pre-d1 repository was not prepared outside the container:" >&2
    echo "       $PRE_D1_WORKTREE" >&2
    exit 1
fi

if ! git -C "$PRE_D1_WORKTREE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: PRE_D1_WORKTREE is not a Git repository/worktree: $PRE_D1_WORKTREE" >&2
    exit 1
fi

if [[ "$(git -C "$PRE_D1_WORKTREE" rev-parse HEAD)" != "$PRE_D1_COMMIT" ]]; then
    echo "ERROR: PRE_D1_WORKTREE is at the wrong commit:" >&2
    echo "       expected $PRE_D1_COMMIT" >&2
    echo "       got      $(git -C "$PRE_D1_WORKTREE" rev-parse HEAD)" >&2
    exit 1
fi

if [[ -n "$(git -C "$PRE_D1_WORKTREE" status --porcelain)" ]]; then
    echo "ERROR: pre-d1 worktree is not clean: $PRE_D1_WORKTREE" >&2
    git -C "$PRE_D1_WORKTREE" status --short >&2
    exit 1
fi

pre_d1_config_sha=$(git -C "$PRE_D1_WORKTREE" show "HEAD:$CONFIG_REL" | sha256sum | awk '{print $1}')
fail_config_sha=$(git -C "$SOURCE_REPO" show "HEAD:$CONFIG_REL" | sha256sum | awk '{print $1}')
if [[ "$pre_d1_config_sha" != "$fail_config_sha" ]]; then
    echo "ERROR: pre-d1 and failing commits do not contain the same Qwen3.5 test config" >&2
    echo "       pre-d1: $pre_d1_config_sha" >&2
    echo "       failing: $fail_config_sha" >&2
    exit 1
fi

for tracker in "$ADAMW_ONLY_TRACKER" "$CLIP_ALL_TRACKER"; do
    if [[ "$(wc -l < "$tracker")" -ne 20 ]]; then
        echo "ERROR: expected 20 rows in existing tracker: $tracker" >&2
        wc -l "$tracker" >&2
        exit 1
    fi
done

mkdir -p "$REFERENCE_RUN_ROOT/work" "$AB_EXPERIMENT_ROOT/trackers"
git -C "$PRE_D1_WORKTREE" show "HEAD:$CONFIG_REL" > "$RUNTIME_CONFIG"
if ! grep -q 'debug_skip_save=True' "$RUNTIME_CONFIG"; then
    sed -i '/trainer = TrainerConfig(/a\    debug_skip_save=True,' "$RUNTIME_CONFIG"
fi

echo "===== Pre-d1 reference ====="
echo "commit: $(git -C "$PRE_D1_WORKTREE" rev-parse HEAD)"
echo "worktree: $PRE_D1_WORKTREE"
echo "config sha256: $pre_d1_config_sha"
grep -nE 'ep_size|num_layers|global_batch_size|sp_size|pack_max_length|total_step|debug_skip_save' "$RUNTIME_CONFIG"

export PYTHONNOUSERSITE=1
# Do not inherit a PYTHONPATH that could import XTuner from the 2c60087 worktree.
export PYTHONPATH="$PRE_D1_WORKTREE"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XTUNER_DETERMINISTIC=true
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MODEL_PATH DATA_PATH MEDIA_ROOT
export WORK_DIR="$REFERENCE_RUN_ROOT/work"
export GITHUB_RUN_ID="manual-qwen35-pre-d1-$TAG"

cd "$PRE_D1_WORKTREE"

"$ENV_PYTHON" - "$PRE_D1_WORKTREE" <<'PY'
import pathlib
import sys

import torch
import xtuner

expected_root = pathlib.Path(sys.argv[1]).resolve()
actual_xtuner = pathlib.Path(xtuner.__file__).resolve()

print("Python/Torch preflight")
print("  torch:", torch.__version__)
print("  CUDA:", torch.version.cuda)
print("  visible GPUs:", torch.cuda.device_count())
print("  xtuner:", actual_xtuner)

if torch.cuda.device_count() != 8:
    raise SystemExit(f"ERROR: expected 8 visible GPUs, got {torch.cuda.device_count()}")
if expected_root not in actual_xtuner.parents:
    raise SystemExit(f"ERROR: imported XTuner from the wrong worktree: {actual_xtuner}")
PY

echo
echo "===== Starting pre-d1-9e4e212 ====="
set +e
"$ENV_PYTHON" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=8 \
    xtuner/v1/train/cli/sft.py \
    --config "$RUNTIME_CONFIG" \
    2>&1 | tee "$REFERENCE_RUN_ROOT/train.log"
train_status=${PIPESTATUS[0]}
set -e

if [[ "$train_status" -ne 0 ]]; then
    echo "ERROR: pre-d1 reference failed with status $train_status" >&2
    exit "$train_status"
fi

tracker=$(find "$WORK_DIR" \
    -type f \
    -path '*/logs/exp_tracking/rank0/tracker.jsonl' \
    | sort \
    | tail -1)

if [[ -z "$tracker" ]]; then
    echo "ERROR: pre-d1 tracker not found under $WORK_DIR" >&2
    exit 1
fi

if [[ "$(wc -l < "$tracker")" -ne 20 ]]; then
    echo "ERROR: expected 20 rows in pre-d1 tracker: $tracker" >&2
    wc -l "$tracker" >&2
    exit 1
fi

cp "$tracker" "$REFERENCE_TRACKER"

echo
echo "===== Tracker summary ====="
wc -l "$REFERENCE_TRACKER" "$ADAMW_ONLY_TRACKER" "$CLIP_ALL_TRACKER"

"$ENV_PYTHON" - \
    "$REFERENCE_TRACKER" \
    "$ADAMW_ONLY_TRACKER" \
    "$CLIP_ALL_TRACKER" <<'PY'
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


def common_metrics(left, right, steps):
    left_losses = {key for step in steps for key in left[step] if key.startswith("loss/")}
    right_losses = {key for step in steps for key in right[step] if key.startswith("loss/")}
    return ["grad_norm", *sorted(left_losses & right_losses)]


def compare(name, left, right):
    if set(left) != set(right):
        raise SystemExit(
            f"{name}: step mismatch, only_left={sorted(set(left) - set(right))}, "
            f"only_right={sorted(set(right) - set(left))}"
        )
    steps = sorted(left)
    print(f"\n===== {name} =====")

    for metric in (
        "lr",
        "runtime_info/text_tokens",
        "runtime_info/seqlen_tokens",
        "runtime_info/approximate_total_consumed_tokens",
    ):
        present = [metric in left[step] and metric in right[step] for step in steps]
        if not any(present):
            print(f"CONTROL {metric}: skipped (not logged)")
            continue
        if not all(present):
            raise SystemExit(f"{name}: {metric} missing from part of the aligned rows")
        bad_steps = [step for step in steps if left[step][metric] != right[step][metric]]
        print(f"CONTROL {metric}: exact_equal={not bad_steps}, differing_steps={bad_steps}")

    all_exact = True
    result = {}
    for metric in common_metrics(left, right, steps):
        missing_steps = [step for step in steps if metric not in left[step] or metric not in right[step]]
        if missing_steps:
            raise SystemExit(f"{name}: {metric} missing at steps {missing_steps}")

        a = np.asarray([float(left[step][metric]) for step in steps], dtype=np.float64)
        b = np.asarray([float(right[step][metric]) for step in steps], dtype=np.float64)
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise SystemExit(f"{name}: non-finite {metric}")

        exact = bool(np.array_equal(a, b))
        all_exact = all_exact and exact
        abs_error = np.abs(a - b)
        rel_error = np.asarray([relative_error(base, current) for base, current in zip(a, b)])
        max_index = int(np.argmax(rel_error))
        result[metric] = float(np.percentile(rel_error, 95))
        print(
            f"{metric}: exact_equal={exact}, "
            f"p50_rel={np.percentile(rel_error, 50) * 100:.6f}%, "
            f"p80_rel={np.percentile(rel_error, 80) * 100:.6f}%, "
            f"p95_rel={result[metric] * 100:.6f}%, "
            f"max_rel={rel_error[max_index] * 100:.6f}%, "
            f"max_abs={abs_error[max_index]:.8g}, step={steps[max_index]}, "
            f"left={a[max_index]:.8g}, right={b[max_index]:.8g}"
        )

    print(f"ALL_TRAINING_METRICS_EXACT={all_exact}")
    return result


reference_path, adamw_path, clip_all_path = sys.argv[1:4]
reference = load(reference_path)
adamw_only = load(adamw_path)
clip_all = load(clip_all_path)

reference_vs_clip = compare("pre-d1 vs 2c-clip-all", reference, clip_all)
reference_vs_adamw = compare("pre-d1 vs 2c-adamw-only", reference, adamw_only)
compare("2c-adamw-only vs 2c-clip-all", adamw_only, clip_all)

print("\n===== Restoration verdict by p95 relative error =====")
for metric in sorted(reference_vs_clip):
    clip_error = reference_vs_clip[metric]
    adamw_error = reference_vs_adamw[metric]
    closer = "clip-all" if clip_error < adamw_error else ("adamw-only" if adamw_error < clip_error else "tie")
    print(
        f"{metric}: closer_to_pre_d1={closer}, "
        f"clip_all_p95={clip_error * 100:.6f}%, adamw_only_p95={adamw_error * 100:.6f}%"
    )
PY

echo
echo "===== Pre-d1 reference experiment completed ====="
echo "Experiment root: $AB_EXPERIMENT_ROOT"
echo "Reference tracker: $REFERENCE_TRACKER"
echo "Reference log: $REFERENCE_RUN_ROOT/train.log"
echo "Pre-d1 worktree retained at: $PRE_D1_WORKTREE"
