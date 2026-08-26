#!/usr/bin/env bash

# Fully offline repeat experiment for the Qwen3.5 clipping investigation.
# Existing run1 trackers are reused. This script adds exactly two runs:
#   1. pre-d1 run2
#   2. 2c60087 + clip-all run2
# It then estimates within-policy noise versus cross-version differences.

set -euo pipefail

CLIP_REPO="${CLIP_REPO:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-qwen35-clip-debug-2c60087}"
PRE_D1_REPO="${PRE_D1_REPO:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-qwen35-pre-d1-9e4e212}"
ENV_PYTHON="${ENV_PYTHON:-/mnt/shared-storage-user/llmrazor-share/comm_env/glm52-pt121-cu132/bin/python}"
AB_ROOT="${AB_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/qwen35-clip-ab/qwen35-clip-ab-8gpu-20260825060848}"

PRE_D1_RUN1="${PRE_D1_RUN1:-$AB_ROOT/trackers/pre-d1-9e4e212-20260825072240.jsonl}"
CLIP_ALL_RUN1="${CLIP_ALL_RUN1:-$AB_ROOT/trackers/clip-all.jsonl}"

MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/llmrazor-share/model/Qwen3.5-35B-A3B}"
DATA_PATH="${DATA_PATH:-/mnt/shared-storage-user/llmrazor-share/data/ci_vl}"
MEDIA_ROOT="${MEDIA_ROOT:-/mnt/shared-storage-user/llmrazor-share/data/ci_vl}"

PRE_D1_COMMIT="9e4e21255511ec518821fd7eb9ca77db5cd15e7d"
CLIP_COMMIT="2c60087fe8d9380dab47d5b743e1784904c8360d"
CONFIG_REL="autotest/config/qwen3_5_ep2_mtp4_vl.py"
OPTIM_REL="xtuner/v1/config/optim.py"

TAG="${TAG:-$(date +%Y%m%d%H%M%S)}"
REPEAT_ROOT="$AB_ROOT/noise-repeat-$TAG"
PRE_D1_RUN2="$AB_ROOT/trackers/pre-d1-run2-$TAG.jsonl"
CLIP_ALL_RUN2="$AB_ROOT/trackers/clip-all-run2-$TAG.jsonl"

if [[ ! -x "$ENV_PYTHON" ]]; then
    echo "ERROR: Python is not executable: $ENV_PYTHON" >&2
    exit 1
fi

for path in "$CLIP_REPO" "$PRE_D1_REPO" "$AB_ROOT" "$PRE_D1_RUN1" "$CLIP_ALL_RUN1" \
    "$MODEL_PATH" "$DATA_PATH" "$MEDIA_ROOT"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: required path does not exist: $path" >&2
        exit 1
    fi
done

if [[ "$(git -C "$CLIP_REPO" rev-parse HEAD)" != "$CLIP_COMMIT" ]]; then
    echo "ERROR: clip-all repository is not at $CLIP_COMMIT" >&2
    exit 1
fi

if [[ "$(git -C "$PRE_D1_REPO" rev-parse HEAD)" != "$PRE_D1_COMMIT" ]]; then
    echo "ERROR: pre-d1 repository is not at $PRE_D1_COMMIT" >&2
    exit 1
fi

if [[ -n "$(git -C "$PRE_D1_REPO" status --porcelain)" ]]; then
    echo "ERROR: pre-d1 repository is not clean" >&2
    git -C "$PRE_D1_REPO" status --short >&2
    exit 1
fi

unexpected_clip_changes=$(git -C "$CLIP_REPO" diff HEAD --name-only \
    | grep -v -x "$OPTIM_REL" \
    | grep -v -x "$CONFIG_REL" \
    | grep -v -x 'autotest/config.yaml' \
    || true)
if [[ -n "$unexpected_clip_changes" ]]; then
    echo "ERROR: unexpected tracked changes in clip-all repository:" >&2
    echo "$unexpected_clip_changes" >&2
    exit 1
fi

if [[ -e "$REPEAT_ROOT" ]]; then
    echo "ERROR: repeat output already exists; use a new TAG: $REPEAT_ROOT" >&2
    exit 1
fi

mkdir -p "$REPEAT_ROOT" "$AB_ROOT/trackers"

expected_clip_optim="$REPEAT_ROOT/optim.expected_clip_all.py"
git -C "$CLIP_REPO" show "HEAD:$OPTIM_REL" \
    | sed 's/clip_grad=False/clip_grad=True/g' \
    > "$expected_clip_optim"

if ! cmp -s "$expected_clip_optim" "$CLIP_REPO/$OPTIM_REL"; then
    echo "ERROR: clip repository does not contain exactly the expected three-line clip-all patch" >&2
    diff -u "$expected_clip_optim" "$CLIP_REPO/$OPTIM_REL" || true
    exit 1
fi

pre_config_sha=$(git -C "$PRE_D1_REPO" show "HEAD:$CONFIG_REL" | sha256sum | awk '{print $1}')
clip_config_sha=$(git -C "$CLIP_REPO" show "HEAD:$CONFIG_REL" | sha256sum | awk '{print $1}')
if [[ "$pre_config_sha" != "$clip_config_sha" ]]; then
    echo "ERROR: training configs differ: pre=$pre_config_sha clip=$clip_config_sha" >&2
    exit 1
fi

for tracker in "$PRE_D1_RUN1" "$CLIP_ALL_RUN1"; do
    if [[ "$(wc -l < "$tracker")" -ne 20 ]]; then
        echo "ERROR: expected 20 rows in existing tracker: $tracker" >&2
        wc -l "$tracker" >&2
        exit 1
    fi
done

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XTUNER_DETERMINISTIC=true
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MODEL_PATH DATA_PATH MEDIA_ROOT

run_one() {
    local run_name="$1"
    local repo="$2"
    local output_tracker="$3"
    local run_root="$REPEAT_ROOT/$run_name"
    local runtime_config="$run_root/qwen3_5_ep2_mtp4_vl_8gpu_nosave.py"
    local tracker
    local train_status

    mkdir -p "$run_root/work"
    git -C "$repo" show "HEAD:$CONFIG_REL" > "$runtime_config"
    if ! grep -q 'debug_skip_save=True' "$runtime_config"; then
        sed -i '/trainer = TrainerConfig(/a\    debug_skip_save=True,' "$runtime_config"
    fi

    export PYTHONPATH="$repo"
    export WORK_DIR="$run_root/work"
    export GITHUB_RUN_ID="manual-qwen35-$run_name-$TAG"

    cd "$repo"

    "$ENV_PYTHON" - "$repo" <<'PY'
import pathlib
import sys

import torch
import xtuner

expected = pathlib.Path(sys.argv[1]).resolve()
actual = pathlib.Path(xtuner.__file__).resolve()
print("preflight:", actual)
print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPUs:", torch.cuda.device_count())
if expected not in actual.parents:
    raise SystemExit(f"ERROR: imported XTuner from the wrong repository: {actual}")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"ERROR: expected 8 GPUs, got {torch.cuda.device_count()}")
PY

    echo
    echo "===== Starting $run_name ====="
    echo "repo=$repo"
    echo "WORK_DIR=$WORK_DIR"

    set +e
    "$ENV_PYTHON" -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc-per-node=8 \
        xtuner/v1/train/cli/sft.py \
        --config "$runtime_config" \
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

    if [[ -z "$tracker" || "$(wc -l < "$tracker")" -ne 20 ]]; then
        echo "ERROR: expected a 20-row tracker for $run_name, got: $tracker" >&2
        [[ -n "$tracker" ]] && wc -l "$tracker" >&2
        return 1
    fi

    cp "$tracker" "$output_tracker"
    echo "Completed $run_name: $output_tracker"
}

# Reverse the previous run order to reduce warm-up/order bias.
run_one "pre-d1-run2" "$PRE_D1_REPO" "$PRE_D1_RUN2"
run_one "clip-all-run2" "$CLIP_REPO" "$CLIP_ALL_RUN2"

echo
echo "===== Four tracker summary ====="
wc -l "$PRE_D1_RUN1" "$PRE_D1_RUN2" "$CLIP_ALL_RUN1" "$CLIP_ALL_RUN2"

"$ENV_PYTHON" - \
    "$PRE_D1_RUN1" "$PRE_D1_RUN2" \
    "$CLIP_ALL_RUN1" "$CLIP_ALL_RUN2" <<'PY'
import json
import math
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


def symmetric_relative(a, b):
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)


paths = sys.argv[1:5]
names = ["pre1", "pre2", "clip1", "clip2"]
runs = {name: load(path) for name, path in zip(names, paths)}

step_sets = {name: set(rows) for name, rows in runs.items()}
reference_steps = step_sets["pre1"]
for name, steps in step_sets.items():
    if steps != reference_steps:
        raise SystemExit(
            f"step mismatch for {name}: only_reference={sorted(reference_steps - steps)}, "
            f"only_{name}={sorted(steps - reference_steps)}"
        )
steps = sorted(reference_steps)

print("\n===== Control metrics across four runs =====")
for metric in (
    "lr",
    "runtime_info/text_tokens",
    "runtime_info/seqlen_tokens",
    "runtime_info/approximate_total_consumed_tokens",
):
    present = [metric in runs[name][step] for name in names for step in steps]
    if not any(present):
        print(f"{metric}: skipped (not logged)")
        continue
    if not all(present):
        raise SystemExit(f"{metric}: missing from part of the four trackers")
    values = [[runs[name][step][metric] for step in steps] for name in names]
    exact = all(values[index] == values[0] for index in range(1, len(values)))
    print(f"{metric}: all_four_exact={exact}")
    if not exact:
        raise SystemExit(f"ERROR: control metric differs across runs: {metric}")

loss_sets = []
for name in names:
    loss_sets.append({key for step in steps for key in runs[name][step] if key.startswith("loss/")})
metrics = ["grad_norm", *sorted(set.intersection(*loss_sets))]

arrays = {}
for name in names:
    arrays[name] = {}
    for metric in metrics:
        missing = [step for step in steps if metric not in runs[name][step]]
        if missing:
            raise SystemExit(f"{name}: {metric} missing at steps {missing}")
        values = np.asarray([float(runs[name][step][metric]) for step in steps], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise SystemExit(f"{name}: non-finite {metric}")
        arrays[name][metric] = values

pairs = [
    ("within-pre", "pre1", "pre2"),
    ("within-clip", "clip1", "clip2"),
    ("cross-pre1-clip1", "pre1", "clip1"),
    ("cross-pre1-clip2", "pre1", "clip2"),
    ("cross-pre2-clip1", "pre2", "clip1"),
    ("cross-pre2-clip2", "pre2", "clip2"),
]

stats = {}
for pair_name, left_name, right_name in pairs:
    stats[pair_name] = {}
    print(f"\n===== {pair_name} =====")
    for metric in metrics:
        left = arrays[left_name][metric]
        right = arrays[right_name][metric]
        abs_error = np.abs(left - right)
        rel_error = symmetric_relative(left, right)
        max_index = int(np.argmax(rel_error))
        stats[pair_name][metric] = {
            "p95": float(np.percentile(rel_error, 95)),
            "max_abs": float(np.max(abs_error)),
        }
        print(
            f"{metric}: exact={bool(np.array_equal(left, right))}, "
            f"p50_sym_rel={np.percentile(rel_error, 50) * 100:.6f}%, "
            f"p95_sym_rel={stats[pair_name][metric]['p95'] * 100:.6f}%, "
            f"max_sym_rel={rel_error[max_index] * 100:.6f}%@step{steps[max_index]}, "
            f"max_abs={np.max(abs_error):.8g}"
        )

print("\n===== Noise-floor assessment =====")
cross_names = [name for name, _, _ in pairs if name.startswith("cross-")]
for metric in metrics:
    within_ceiling = max(stats["within-pre"][metric]["p95"], stats["within-clip"][metric]["p95"])
    cross_values = [stats[name][metric]["p95"] for name in cross_names]
    cross_floor = min(cross_values)
    cross_ceiling = max(cross_values)
    if within_ceiling == 0:
        ratio = 0.0 if cross_floor == 0 else math.inf
    else:
        ratio = cross_floor / within_ceiling

    pre_mean = (arrays["pre1"][metric] + arrays["pre2"][metric]) / 2
    clip_mean = (arrays["clip1"][metric] + arrays["clip2"][metric]) / 2
    within_abs = np.maximum(
        np.abs(arrays["pre1"][metric] - arrays["pre2"][metric]),
        np.abs(arrays["clip1"][metric] - arrays["clip2"][metric]),
    )
    between_abs = np.abs(pre_mean - clip_mean)
    beyond_noise_steps = int(np.sum(between_abs > 2 * within_abs))

    ratio_text = "inf" if math.isinf(ratio) else f"{ratio:.3f}"
    print(
        f"{metric}: within_p95_ceiling={within_ceiling * 100:.6f}%, "
        f"cross_p95_range=[{cross_floor * 100:.6f}%, {cross_ceiling * 100:.6f}%], "
        f"cross_floor/within_ceiling={ratio_text}, "
        f"mean_difference_beyond_2x_within_noise_steps={beyond_noise_steps}/{len(steps)}"
    )

print("\nInterpretation: ratios near or below 1 indicate that pre-d1 vs clip-all differences are not larger than")
print("the observed same-policy run-to-run noise. This is evidence, not a hard statistical pass threshold.")
PY

echo
echo "===== Noise repeat completed ====="
echo "Repeat root: $REPEAT_ROOT"
echo "pre-d1 run2: $PRE_D1_RUN2"
echo "clip-all run2: $CLIP_ALL_RUN2"
