#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2 NoAux-router-bias ablation.
# Reuse the verified five-main-layer/no-MTP runner and change exactly one
# additional model option: router_bias_update_speed = 0.0.
#
# This keeps router parameters trainable and keeps normal top-k routing. It only
# disables the per-step in-place update of e_score_correction_bias.

REPO_ROOT=${REPO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/xtuner-glm52-repro}
REPRO_ROOT=${REPRO_ROOT:-/mnt/shared-storage-user/ailab-sys/liutong/glm52-repro}
RUN_COUNT=${RUN_COUNT:-3}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-$(date +%Y%m%d%H%M%S)}

ORIGINAL_BASE_CONFIG=${ORIGINAL_BASE_CONFIG:-${REPO_ROOT}/autotest/config/glm5p2_30B_all2all_nosave.py}
ROUTER0_BASE_CONFIG=${ROUTER0_BASE_CONFIG:-${REPO_ROOT}/autotest/config/glm5p2_30B_all2all_routerbias0_nosave.py}
TEST_CONFIG=${TEST_CONFIG:-${REPO_ROOT}/autotest/config/glm5p2_30B_all2all_nomtp5_routerbias0_nosave.py}
NOMTP_RUNNER=${NOMTP_RUNNER:-${REPO_ROOT}/run_glm52_nomtp5_sft_3runs.sh}
ROUTER0_REPRO_ROOT=${ROUTER0_REPRO_ROOT:-${REPRO_ROOT}/nomtp5-routerbias0}

if [[ ! -d "${REPO_ROOT}/.git" ]]; then
  echo "ERROR: repository not found: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${ORIGINAL_BASE_CONFIG}" ]]; then
  echo "ERROR: original all2all/no-save config not found: ${ORIGINAL_BASE_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${NOMTP_RUNNER}" ]]; then
  echo "ERROR: verified no-MTP runner not found: ${NOMTP_RUNNER}" >&2
  exit 1
fi
if ! bash -n "${NOMTP_RUNNER}"; then
  echo "ERROR: no-MTP runner has invalid Bash syntax: ${NOMTP_RUNNER}" >&2
  exit 1
fi
if [[ ! "${RUN_COUNT}" =~ ^[2-9][0-9]*$ ]]; then
  echo "ERROR: RUN_COUNT must be an integer >= 2" >&2
  exit 1
fi

cd "${REPO_ROOT}"

# Build a temporary experiment base config. The downstream no-MTP runner will
# copy this again, remove MTP, switch strict_load to False, and run all checks.
cp "${ORIGINAL_BASE_CONFIG}" "${ROUTER0_BASE_CONFIG}"

DISPATCHER_COUNT=$(grep -Ec 'moe_cfg\.dispatcher[[:space:]]*=[[:space:]]*"all2all"' "${ROUTER0_BASE_CONFIG}" || true)
if [[ "${DISPATCHER_COUNT}" -ne 1 ]]; then
  echo "ERROR: expected exactly one moe_cfg.dispatcher=all2all assignment; found ${DISPATCHER_COUNT}" >&2
  exit 1
fi

sed -i -E '/moe_cfg\.dispatcher[[:space:]]*=[[:space:]]*"all2all"/a\
assert hasattr(moe_cfg.router, "router_bias_update_speed")\
moe_cfg.router.router_bias_update_speed = 0.0' "${ROUTER0_BASE_CONFIG}"

if [[ $(grep -Ec 'moe_cfg\.router\.router_bias_update_speed[[:space:]]*=[[:space:]]*0\.0' "${ROUTER0_BASE_CONFIG}" || true) -ne 1 ]]; then
  echo "ERROR: expected exactly one router_bias_update_speed=0.0 assignment" >&2
  exit 1
fi
if ! grep -Eq 'debug_skip_save[[:space:]]*=[[:space:]]*True' "${ROUTER0_BASE_CONFIG}"; then
  echo "ERROR: debug_skip_save=True is missing; refusing to run" >&2
  exit 1
fi

echo "===== NoAux router-bias-update ablation ====="
echo "repo: ${REPO_ROOT}"
echo "commit: $(git rev-parse HEAD)"
echo "original base config: ${ORIGINAL_BASE_CONFIG}"
echo "router-bias-zero base config: ${ROUTER0_BASE_CONFIG}"
echo "generated no-MTP test config: ${TEST_CONFIG}"
echo "verified no-MTP runner: ${NOMTP_RUNNER}"
echo "result parent: ${ROUTER0_REPRO_ROOT}"
echo "experiment tag: ${EXPERIMENT_TAG}"
echo "run count: ${RUN_COUNT}"
grep -E 'dispatcher|router_bias_update_speed|debug_skip_save' "${ROUTER0_BASE_CONFIG}"

# The child runner performs the complete environment/config preflight, keeps
# five main layers, removes MTP, runs 3x35 steps, checks that MTP metrics are
# absent, and performs pairwise tracker comparison.
BASE_CONFIG="${ROUTER0_BASE_CONFIG}" \
TEST_CONFIG="${TEST_CONFIG}" \
REPRO_ROOT="${ROUTER0_REPRO_ROOT}" \
RUN_COUNT="${RUN_COUNT}" \
EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
bash "${NOMTP_RUNNER}"

echo
echo "===== Router-bias-update ablation completed ====="
echo "Expected experiment root: ${ROUTER0_REPRO_ROOT}/glm52-nomtp5-sft-${EXPERIMENT_TAG}"
