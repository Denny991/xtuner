#!/usr/bin/env bash
# Run inside the rlaunch container.
# Target image verified in this note:
# registry.h.pjlab.org.cn/ailab-sys-sys_gpu/lt_test:vllm0.15.1-torch2.10-cu128

set -euo pipefail

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

# Persistent inputs mounted by rlaunch.
SRC="${SRC:-/mnt/shared-storage-user/ailab-sys/liutong/Automodel}"
DEEP_EP_SRC="${DEEP_EP_SRC:-/mnt/shared-storage-user/ailab-sys/liutong/third_party/DeepEP}"
MODEL="${MODEL:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--zai-org--GLM-5.2}"

# Container-local work dirs. Override RESET_WORKSRC=1 if you want a fresh copy.
WORKSRC="${WORKSRC:-/tmp/Automodel-torch210}"
VENV="${VENV:-/opt/venv}"
WORK="${WORK:-/mnt/shared-storage-user/ailab-sys/liutong/glm_5_2_smoke_work}"
RESET_WORKSRC="${RESET_WORKSRC:-0}"
RESET_VENV="${RESET_VENV:-0}"
FORCE_DEEP_EP_BUILD="${FORCE_DEEP_EP_BUILD:-0}"
AUTO_INSTALL_RDMA_DEPS="${AUTO_INSTALL_RDMA_DEPS:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"

disable_external_nvidia_apt_sources() {
  [ -d /etc/apt/sources.list.d ] || return 0

  mkdir -p /etc/apt/disabled-sources
  while IFS= read -r source_file; do
    [ -n "$source_file" ] || continue
    disabled_file="/etc/apt/disabled-sources/$(basename "$source_file").disabled"
    if [ -e "$disabled_file" ]; then
      disabled_file="${disabled_file}.$(date +%s)"
    fi
    mv "$source_file" "$disabled_file"
    echo "disabled apt source: $source_file -> $disabled_file"
  done < <(grep -Rl "developer.download.nvidia.com" /etc/apt/sources.list.d 2>/dev/null || true)
}

configure_pjlab_apt_sources() {
  test -f /etc/os-release || fail "/etc/os-release missing; cannot select PJLab apt mirror"
  # shellcheck disable=SC1091
  . /etc/os-release

  case "${VERSION_CODENAME:-}" in
    jammy)
      cp -a /etc/apt/sources.list "/etc/apt/sources.list.bak.$(date +%s)" 2>/dev/null || true
      cat >/etc/apt/sources.list <<'EOF'
deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy main restricted universe multiverse
deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-security main restricted universe multiverse
deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-updates main restricted universe multiverse
deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-backports main restricted universe multiverse
EOF
      ;;
    noble)
      mkdir -p /etc/apt/sources.list.d
      cp -a /etc/apt/sources.list.d/ubuntu.sources \
        "/etc/apt/sources.list.d/ubuntu.sources.bak.$(date +%s)" 2>/dev/null || true
      cat >/etc/apt/sources.list.d/ubuntu.sources <<'EOF'
Types: deb
URIs: http://mirrors.i.h.pjlab.org.cn/repository/apt-noble-proxy
Suites: noble noble-updates noble-backports
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: http://mirrors.i.h.pjlab.org.cn/repository/apt-noble-proxy
Suites: noble-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
      ;;
    *)
      fail "Unsupported Ubuntu codename for automatic RDMA deps install: ${VERSION_CODENAME:-unknown}"
      ;;
  esac
}

ensure_rdma_build_deps() {
  if [ -f /usr/include/infiniband/mlx5dv.h ]; then
    return 0
  fi

  if [ "$AUTO_INSTALL_RDMA_DEPS" != "1" ]; then
    fail "DeepEP build needs /usr/include/infiniband/mlx5dv.h. Set AUTO_INSTALL_RDMA_DEPS=1 or bake rdma-core libibverbs-dev into the image."
  fi

  test "$(id -u)" = "0" || fail "RDMA deps are missing and automatic apt install requires root"
  command -v apt-get >/dev/null || fail "apt-get not found; bake rdma-core libibverbs-dev into the image"

  log "RDMA headers missing; configure PJLab apt mirror and install rdma-core/libibverbs-dev"
  configure_pjlab_apt_sources
  disable_external_nvidia_apt_sources

  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends rdma-core libibverbs-dev

  test -f /usr/include/infiniband/mlx5dv.h || \
    fail "rdma-core/libibverbs-dev installed, but /usr/include/infiniband/mlx5dv.h is still missing"
}

log "Set Python/CUDA paths for the PyTorch 2.10 cu128 image"
export PATH="/opt/ac2/bin:/opt/conda/bin:/usr/local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/bin:/usr/bin:/sbin:/usr/sbin:/opt/ssh-static/bin:${HOME}/.local/bin"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  NVCC_PATH="$(command -v nvcc || true)"
  if [ -n "$NVCC_PATH" ]; then
    CUDA_HOME="$(cd "$(dirname "$NVCC_PATH")/.." && pwd)"
  fi
fi
test -x "$CUDA_HOME/bin/nvcc" || fail "nvcc not found. DeepEP HybridEP runtime JIT needs nvcc; expected $CUDA_HOME/bin/nvcc"
test -d "$CUDA_HOME/include" || fail "CUDA include dir missing: $CUDA_HOME/include"
test -d "$CUDA_HOME/lib64" || fail "CUDA lib64 dir missing: $CUDA_HOME/lib64"
CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
CUDA_NVCC_EXECUTABLE="${CUDA_NVCC_EXECUTABLE:-$CUDA_HOME/bin/nvcc}"
export CUDA_HOME CUDA_PATH CUDACXX CUDA_NVCC_EXECUTABLE
export PATH="/opt/ac2/bin:/opt/conda/bin:/usr/local/bin:/usr/local/nvidia/bin:$CUDA_HOME/bin:/bin:/usr/bin:/sbin:/usr/sbin:/opt/ssh-static/bin:${HOME}/.local/bin"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$CUDA_HOME/lib64"

find_base_python() {
  if [ -n "${BASE_PYTHON:-}" ]; then
    test -x "$BASE_PYTHON" || fail "BASE_PYTHON is not executable: $BASE_PYTHON"
    "$BASE_PYTHON" -c "import torch" >/dev/null 2>&1 || \
      fail "BASE_PYTHON cannot import torch: $BASE_PYTHON"
    printf '%s\n' "$BASE_PYTHON"
    return 0
  fi

  for candidate in \
    /opt/ac2/bin/python \
    /opt/ac2/bin/python3 \
    /opt/ac2/bin/python3.12 \
    /opt/conda/bin/python \
    /opt/conda/bin/python3 \
    /usr/local/bin/python \
    /usr/local/bin/python3 \
    "$(command -v python || true)" \
    "$(command -v python3 || true)"; do
    [ -n "$candidate" ] || continue
    [ -x "$candidate" ] || continue
    if "$candidate" -c "import torch" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  fail "No Python with torch found. Checked /opt/ac2, /opt/conda, /usr/local, and PATH. Verify the rlaunch image tag."
}

BASE_PYTHON="$(find_base_python)"
echo "BASE_PYTHON=$BASE_PYTHON"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDACXX=$CUDACXX"
"$CUDA_HOME/bin/nvcc" --version | sed -n '1,4p'
echo "SCRIPT_VERSION=2026-07-13-torch210-cu128-ac2-uv-v7"

log "Validate CUDA before installing anything"
nvidia-smi -L
"$BASE_PYTHON" - <<'PY'
import inspect
import torch
from torch.distributed.device_mesh import init_device_mesh

print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("available=", torch.cuda.is_available())
print("count=", torch.cuda.device_count())
print("backend_override=", "backend_override" in inspect.signature(init_device_mesh).parameters)
major, minor = (int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
if (major, minor) < (2, 10):
    raise SystemExit(f"torch {torch.__version__} is too old for this DeepEP smoke stack; need torch >= 2.10")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
if "backend_override" not in inspect.signature(init_device_mesh).parameters:
    raise SystemExit("init_device_mesh backend_override is missing")
x = torch.empty(1, device="cuda")
print(x)
print(torch.cuda.get_device_name(0))
PY

log "Prepare AutoModel working copy"
test -d "$SRC" || fail "SRC does not exist: $SRC"
test -f "$SRC/tests/functional_tests/speculative/run_glm_5.2_smoke.sh" || \
  fail "SRC does not contain run_glm_5.2_smoke.sh: $SRC"

if [ "$WORKSRC" != "$SRC" ]; then
  if [ -d "$WORKSRC" ] && [ "$RESET_WORKSRC" = "1" ]; then
    rm -rf "$WORKSRC"
  fi
  if [ ! -d "$WORKSRC" ]; then
    cp -a "$SRC" "$WORKSRC"
  fi
fi
cd "$WORKSRC"
test -f tests/functional_tests/speculative/run_glm_5.2_smoke.sh

log "Configure pip/uv mirrors"
export PIP_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/pypi/simple/"
export PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export PIP_TRUSTED_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"

export UV_DEFAULT_INDEX="http://mirrors.i.h.pjlab.org.cn/pypi/simple/"
export UV_INDEX="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export UV_INSECURE_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export UV_INDEX_STRATEGY="unsafe-best-match"

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV_BIN" ] && [ -x /opt/ac2/bin/uv ]; then
  UV_BIN=/opt/ac2/bin/uv
fi
if [ -z "$UV_BIN" ] && [ -x /usr/local/bin/uv ]; then
  UV_BIN=/usr/local/bin/uv
fi
test -z "$UV_BIN" || test -x "$UV_BIN" || fail "UV_BIN is not executable: $UV_BIN"
test -n "$UV_BIN" || fail "uv not found; use the verified PyTorch 2.10 cu128 image or bake uv into the base image first"
log "Use uv from $UV_BIN"
"$UV_BIN" --version

log "Create or reuse venv with system torch from the PyTorch base image"
export UV_PROJECT_ENVIRONMENT="$VENV"
if [ -d "$VENV" ] && [ "$RESET_VENV" = "1" ]; then
  rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
  "$UV_BIN" venv "$VENV" --system-site-packages --python "$BASE_PYTHON"
fi
source "$VENV/bin/activate"

uv_pip_install() {
  (cd /tmp && "$UV_BIN" pip install --python "$VENV/bin/python" "$@")
}

check_torch_stack() {
  "$VENV/bin/python" - <<'PY'
import inspect
import torch
from torch.distributed.device_mesh import init_device_mesh

print("checked torch=", torch.__version__)
print("checked torch_cuda=", torch.version.cuda)
print("checked torch_file=", torch.__file__)
print("checked cuda_available=", torch.cuda.is_available())
print("checked gpu_count=", torch.cuda.device_count())
major, minor = (int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
if (major, minor) < (2, 10):
    raise SystemExit(f"torch changed to {torch.__version__}; expected torch >= 2.10")
if torch.version.cuda != "12.8":
    raise SystemExit(f"torch.version.cuda changed to {torch.version.cuda}; expected 12.8 from the base image")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() became False")
if "backend_override" not in inspect.signature(init_device_mesh).parameters:
    raise SystemExit("init_device_mesh backend_override disappeared")
PY
}

python - <<'PY'
import inspect
import torch
from torch.distributed.device_mesh import init_device_mesh

print("venv torch=", torch.__version__)
print("venv torch_cuda=", torch.version.cuda)
print("venv cuda_available=", torch.cuda.is_available())
print("venv gpu_count=", torch.cuda.device_count())
major, minor = (int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
if (major, minor) < (2, 10):
    raise SystemExit(f"torch {torch.__version__} is too old for this DeepEP smoke stack; need torch >= 2.10")
has_backend_override = "backend_override" in inspect.signature(init_device_mesh).parameters
print("init_device_mesh_backend_override=", has_backend_override)
if not has_backend_override:
    raise SystemExit(
        "This AutoModel checkout expects torch.distributed.device_mesh.init_device_mesh(..., "
        "backend_override=...). Upgrade to a PyTorch build that exposes this argument, "
        "while keeping torch.version.cuda compatible with the host driver."
    )
PY

log "Patch AutoModel metadata for a PyTorch base image"
cp -n pyproject.toml pyproject.toml.bak.glm52-smoke 2>/dev/null || true
bash docker/common/update_pyproject_pytorch.sh "$PWD"

# Avoid GitHub-only dev dependencies in this temporary container workflow.
sed -i "s#^deep_ep = { git = \"https://github.com/deepseek-ai/DeepEP.git\".*#deep_ep = { path = \"$DEEP_EP_SRC\" }#" pyproject.toml
sed -i '/^dev = \[/,/^]/d' pyproject.toml
sed -i '/^cut-cross-entropy = { git = /d' pyproject.toml
sed -i '/^liger-kernel = { git = /d' pyproject.toml
sed -i '/^deep_ep = \[{ requirement = "torch", match-runtime = true }\]$/d' pyproject.toml
sed -i '/^transformer-engine = \[{ requirement = "torch", match-runtime = true }\]$/d' pyproject.toml
sed -i '/^transformer-engine-torch = \[{ requirement = "torch", match-runtime = true }\]$/d' pyproject.toml

log "Install AutoModel runtime deps without full uv sync"
uv_pip_install -U "setuptools>=80.10.2" packaging wheel ninja --index-strategy unsafe-best-match || \
  uv_pip_install -U packaging wheel ninja --index-strategy unsafe-best-match
# Install the local package itself, but do not let uv resolve torch again.
# The PyTorch base image already provides the CUDA 12.8 torch build; resolving deps here may
# try the repo's CUDA-13 index and fail on a CUDA-12.8 host.
uv_pip_install --no-build-isolation --no-deps -e "$WORKSRC"
uv_pip_install \
  "datasets>=4.0.0" pybind11 pyyaml \
  "transformers==5.12.1" pytest ruamel.yaml nvidia-nvtx-cu12 \
  --index-strategy unsafe-best-match
uv_pip_install --no-deps torchdata --index-strategy unsafe-best-match
check_torch_stack
log "Install torchao into $VENV"
uv_pip_install --no-deps --reinstall-package torchao "torchao==0.14.0" --index-strategy unsafe-best-match
check_torch_stack
"$VENV/bin/python" - <<'PY'
import importlib.util

if importlib.util.find_spec("torchao") is None:
    raise SystemExit("torchao install command completed, but torchao is still not importable from /opt/venv")
PY
python - <<'PY'
import datasets
import torchdata
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

print("datasets=", datasets.__version__)
print("torchdata=", getattr(torchdata, "__version__", "unknown"))
print("glm_moe_dsa config=", GlmMoeDsaConfig.__name__)
print("torchao float8=", precompute_float8_dynamic_scale_for_fsdp.__name__)
PY

log "Prepare DeepEP runtime/linker paths"
ensure_rdma_build_deps

ARCH_LIB="$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || true)"
if [ -n "$ARCH_LIB" ]; then
  if [ ! -e "/usr/lib/$ARCH_LIB/libmlx5.so" ] && [ -e "/usr/lib/$ARCH_LIB/libmlx5.so.1" ]; then
    ln -sf "/usr/lib/$ARCH_LIB/libmlx5.so.1" "/usr/lib/$ARCH_LIB/libmlx5.so"
  fi
  export RDMA_CORE_HOME="${RDMA_CORE_HOME:-/opt/rdma-core/build}"
  mkdir -p "$RDMA_CORE_HOME"
  ln -sfn /usr/include "$RDMA_CORE_HOME/include"
  ln -sfn "/usr/lib/$ARCH_LIB" "$RDMA_CORE_HOME/lib"
fi

NVTX_DIR="$(python - <<'PY'
from pathlib import Path
import site

for root in site.getsitepackages():
    path = Path(root) / "nvidia" / "nvtx" / "lib"
    if (path / "libnvtx3interop.so.1").exists():
        print(path)
        break
else:
    raise SystemExit("libnvtx3interop.so.1 not found")
PY
)"
ln -sf "$NVTX_DIR/libnvtx3interop.so.1" "$NVTX_DIR/libnvtx3interop.so"

NVSHMEM_DIR="$(python - <<'PY'
from pathlib import Path
import site

for root in site.getsitepackages():
    path = Path(root) / "nvidia" / "nvshmem" / "lib"
    if path.exists():
        print(path)
        break
PY
)"

if [ -n "$NVSHMEM_DIR" ]; then
  export LD_LIBRARY_PATH="$NVTX_DIR:/usr/local/nvidia/lib64:$NVSHMEM_DIR:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="$NVTX_DIR:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
fi
export LIBRARY_PATH="$NVTX_DIR:$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:/usr/local/nvidia/lib64:${LIBRARY_PATH:-}"
export LDFLAGS="-L$NVTX_DIR -L$CUDA_HOME/lib64 -L$CUDA_HOME/lib64/stubs -L/usr/local/nvidia/lib64 ${LDFLAGS:-}"

log "Build/install DeepEP if needed"
test -d "$DEEP_EP_SRC" || fail "DeepEP source missing: $DEEP_EP_SRC"
test -f "$DEEP_EP_SRC/setup.py" || fail "DeepEP setup.py missing: $DEEP_EP_SRC"

if [ "$FORCE_DEEP_EP_BUILD" = "1" ] || ! python -c "import deep_ep" >/dev/null 2>&1; then
  rm -rf "$DEEP_EP_SRC/build" "$DEEP_EP_SRC/deep_ep.egg-info"
  uv_pip_install --no-build-isolation --no-deps "$DEEP_EP_SRC"
else
  log "DeepEP already importable; skip rebuild"
fi

log "Verify DeepEP and CUDA"
python - <<'PY'
import importlib.util
import torch

print("deep_ep=", importlib.util.find_spec("deep_ep") is not None)
print("cuda_available=", torch.cuda.is_available())
print("gpu_count=", torch.cuda.device_count())
assert importlib.util.find_spec("deep_ep") is not None
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
PY

log "Resolve GLM-5.2 HF cache snapshot"
test -f "$MODEL/refs/main" || fail "MODEL refs/main missing: $MODEL"
HASH="$(cat "$MODEL/refs/main")"
GLM52="$MODEL/snapshots/$HASH"
echo "GLM52=$GLM52"
test -d "$GLM52" || fail "GLM52 snapshot missing: $GLM52"
ls "$GLM52/config.json" "$GLM52/model.safetensors.index.json"

mkdir -p "$WORK"
if [ "$RUN_SMOKE" = "1" ]; then
  log "Run GLM-5.2 single-node 8-GPU reduced-layer smoke"
  PYTHON="$VENV/bin/python" WORK="$WORK" TARGET="$GLM52" bash tests/functional_tests/speculative/run_glm_5.2_smoke.sh
else
  log "RUN_SMOKE=0, skip smoke. To run manually:"
  echo "cd \"$WORKSRC\""
  echo "source \"$VENV/bin/activate\""
  echo "PYTHON=\"$VENV/bin/python\" WORK=\"$WORK\" TARGET=\"$GLM52\" bash tests/functional_tests/speculative/run_glm_5.2_smoke.sh"
fi
