# GLM-5.2 单机 8xH200 smoke：平台容器启动脚本

## 1. 通过平台启动容器

### 1.1 可选：2 卡镜像探针

```shell
rlaunch --gpu=2 --memory=512000 --cpu=32 \
--charged-group=kj_gpu --private-machine=group \
--image=registry.h.pjlab.org.cn/ailab-sys-sys_gpu/lt_test:vllm0.15.1-torch2.10-cu128 \
--mount=gpfs://gpfs1/zhaopenghao/slime0701/:/mnt/shared-storage-user/zhaopenghao/slime0701/ \
--mount=gpfs://gpfs1/ailab-sys/liutong:/mnt/shared-storage-user/ailab-sys/liutong \
--mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
--entrypoint= \
-d -- bash -c "sleep infinity"
```

进入容器后验证：

```shell
export PATH=/opt/ac2/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/bin:/usr/bin:/sbin:/usr/sbin:/opt/ssh-static/bin
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
hash -r
PY=$(command -v python)

"$PY" - <<'PY'
import inspect, torch
from torch.distributed.device_mesh import init_device_mesh

print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("gpu_count=", torch.cuda.device_count())
print("backend_override=", "backend_override" in inspect.signature(init_device_mesh).parameters)
print(torch.empty(1, device="cuda"))
PY

cat >/tmp/dist_probe.py <<'PY'
import inspect, torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

dist.init_process_group("nccl")
print(
    "rank", dist.get_rank(),
    "gpu_count", torch.cuda.device_count(),
    "backend_override", "backend_override" in inspect.signature(init_device_mesh).parameters,
)
dist.destroy_process_group()
PY

"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 /tmp/dist_probe.py
```

2 卡探针只验证基础镜像，不跑 GLM-5.2 smoke；真实 GLM-5.2 MoE smoke 仍需 8 卡。

### 1.2 正式：8 卡 GLM-5.2 smoke

```shell
rlaunch --gpu=8 --memory=1600000 --cpu=128 \
--charged-group=kj_gpu --private-machine=group \
--image=registry.h.pjlab.org.cn/ailab-sys-sys_gpu/lt_test:vllm0.15.1-torch2.10-cu128 \
--mount=gpfs://gpfs1/ailab-sys/liutong:/mnt/shared-storage-user/ailab-sys/liutong \
--mount=gpfs://gpfs1/zhaopenghao/slime0701/ckpts/GLM-5.2-30B-MTP:/mnt/shared-storage-user/zhaopenghao/slime0701/ckpts/GLM-5.2-30B-MTP \
--mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
--entrypoint= \
-d -- bash -c "sleep infinity"
```

说明：

- 当前先使用 `registry.h.pjlab.org.cn/ailab-sys-sys_gpu/lt_test:vllm0.15.1-torch2.10-cu128` 作为基础镜像；目标是同时满足 AutoModel 的 `init_device_mesh backend_override` 路径和 DeepEP 当前 README 的 `torch>=2.10` 下限。
- 本次 H200 节点是 `Driver 570.133.20 / CUDA 12.8`，容器里必须保持 `torch.version.cuda <= 12.8`。`torch.version.cuda=13.x` 会报 `The NVIDIA driver on your system is too old`。
- 该 vLLM/PyTorch 镜像的 Python 可能在 `/opt/ac2/bin/python3.12`，也可能在 `/opt/conda/bin/python`；平台进入容器后的 `PATH` 可能只有 `/bin:/usr/bin:...`，脚本会把 `/opt/ac2/bin`、`/opt/conda/bin` 放到 `PATH` 最前面，并选择第一个能 `import torch` 的 Python。
- `LD_LIBRARY_PATH` 使用宿主 driver 库 `/usr/local/nvidia/lib64`；不要把 `/usr/local/cuda/compat` 放在最前面。
- DeepEP HybridEP 会在训练第一个 step 做运行时 JIT 编译，脚本会显式设置 `CUDA_HOME=/usr/local/cuda`、`CUDACXX=$CUDA_HOME/bin/nvcc`，并提前检查 `nvcc`，避免 DeepEP 拼出 `/bin/nvcc`。
- 当前只把 `gpfs1/ailab-sys/liutong` 挂到 `/mnt/shared-storage-user/ailab-sys/liutong`；不要同时挂父目录和子目录，除非平台明确要求。
- 额外挂载赵鹏昊目录下的 `GLM-5.2-30B-MTP` checkpoint；如果平台真实源路径不是 `gpfs://gpfs1/zhaopenghao/...`，需按实际 fileset 改成对应源路径。
- 最后一条 mount 把 `gpfs2-shared-public` 这个 GPFS fileset 挂到容器内；后面的 `MODEL` 再指到其中的完整 `models--zai-org--GLM-5.2` cache 根目录。
- `sleep infinity` 只是让容器常驻；启动成功后再通过平台进入容器执行下面命令。

## 2. 进入容器后执行

```shell
bash /mnt/shared-storage-user/ailab-sys/liutong/xtuner/Znote/glm/glm52_h200_container_smoke.sh
```

脚本默认会做这些事：

- 使用 `/usr/local/nvidia/lib64` 修正 CUDA driver 库路径，不把 `/usr/local/cuda/compat` 放前面。
- 设置 `CUDA_HOME` / `CUDA_PATH` / `CUDACXX` / `CUDA_NVCC_EXECUTABLE`，并确认 `$CUDA_HOME/bin/nvcc` 存在。
- 从 `/mnt/shared-storage-user/ailab-sys/liutong/Automodel` 复制一份源码到 `/tmp/Automodel-torch210`，避免直接污染挂载源码。
- 创建或复用 `/opt/venv`，使用基础镜像自带的 `torch>=2.10` 且 `torch.version.cuda=12.8`。
- 检查 `torch.distributed.device_mesh.init_device_mesh` 是否支持 `backend_override`；当前 GLM smoke 路径需要这个 PyTorch API。
- 使用内网 PyPI 源安装 AutoModel 运行依赖；当前临时路径不用 `uv sync --all-groups --extra moe`。
- 复用基础镜像里的 `uv`；不再运行时强制升级，避免卡在镜像源超时。
- 运行 `uv pip install` 时切到 `/tmp`，避免当前项目 `pyproject.toml` 对临时包安装产生干扰。
- 安装 AutoModel 本体时使用 `--no-deps -e .`，复用镜像里的 CUDA 12.8 torch，避免重新解析到 CUDA 13 的 `torch` 源。
- 因为使用了 `--no-deps`，脚本会显式补 `datasets`、`torchdata`、`pyyaml`、`pybind11` 等基础运行时依赖；否则 8 卡 recipe 导入阶段会缺包。
- `torchdata`、`torchao`、DeepEP 安装时都禁止解析依赖，防止 `uv` 重新安装 CUDA 13 的 torch 污染 `/opt/venv`。
- 每次高风险依赖安装后都会检查 `torch>=2.10`、`torch.version.cuda == 12.8`、CUDA 可用和 `backend_override=True`，一旦 torch 被换掉会立刻停住。
- DeepEP 编译前检查 `/usr/include/infiniband/mlx5dv.h`；如果缺 RDMA/IB 开发头文件，脚本默认会按 Ubuntu 版本切到平台内网 apt 源、临时禁用外部 NVIDIA apt 源，并安装 `rdma-core libibverbs-dev`。如果不希望脚本改 apt，设置 `AUTO_INSTALL_RDMA_DEPS=0`。
- 额外安装项目要求的 `transformers==5.12.1`；基础 PyTorch 镜像不保证包含 GLM-5.2 需要的 `transformers.models.glm_moe_dsa`。
- 额外强制安装项目 lock 中的 `torchao==0.14.0`；DSpark recipe 启动时需要 `torchao.float8.precompute_float8_dynamic_scale_for_fsdp`，脚本会用 `--reinstall-package torchao` 显式写入 `/opt/venv`，避免 uv 误判已满足。
- smoke 启动多进程时显式使用 `/opt/venv/bin/python -m torch.distributed.run`，避免全局 `torchrun` 走 `/usr/bin/python3` 而看不到 venv 里的依赖。
- 从 `/mnt/shared-storage-user/ailab-sys/liutong/third_party/DeepEP` 编译安装 DeepEP。
- 解析 HF cache 的 `refs/main`，得到真正的 GLM-5.2 snapshot 路径。
- 执行 `tests/functional_tests/speculative/run_glm_5.2_smoke.sh`，即 8 卡、减层、真实权重 smoke。

常用开关：

```shell
# 只准备环境，不跑 smoke。
RUN_SMOKE=0 bash /mnt/shared-storage-user/ailab-sys/liutong/xtuner/Znote/glm/glm52_h200_container_smoke.sh

# 重新复制 /tmp/Automodel-torch210 工作副本。
RESET_WORKSRC=1 bash /mnt/shared-storage-user/ailab-sys/liutong/xtuner/Znote/glm/glm52_h200_container_smoke.sh

# 重建 /opt/venv，旧 venv 状态异常时使用。
RESET_VENV=1 bash /mnt/shared-storage-user/ailab-sys/liutong/xtuner/Znote/glm/glm52_h200_container_smoke.sh

# 强制重编 DeepEP。
FORCE_DEEP_EP_BUILD=1 bash /mnt/shared-storage-user/ailab-sys/liutong/xtuner/Znote/glm/glm52_h200_container_smoke.sh

# 禁止脚本自动改 apt 源/安装 RDMA 系统依赖。
AUTO_INSTALL_RDMA_DEPS=0 bash /mnt/shared-storage-user/ailab-sys/liutong/xtuner/Znote/glm/glm52_h200_container_smoke.sh
```

预期：

- `nvidia-smi -L` 能看到 8 张 H200
- `torch.cuda.is_available()` 为 `True`
- `torch.cuda.device_count()` 为 `8`
- 能成功打印 `tensor(..., device='cuda:0')`
- `deep_ep= True`
- smoke 最后打印 `SMOKE OK` 或至少成功写出 finite loss 日志。

## 3. 本次失败点记录

### `nvidia-smi` 找不到 `libnvidia-ml.so`

现象：

```text
NVIDIA-SMI couldn't find libnvidia-ml.so library in your system.
```

处理：

```shell
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/bin:/usr/bin:/sbin:/usr/sbin:/opt/ssh-static/bin
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
```

如果 `/dev/nvidia0-7` 存在，补上上面的路径后 `nvidia-smi -L` 通常能看到 8 张卡。

### `Error 803: system has unsupported display driver / cuda driver combination`

现象：

```text
torch= 2.7.1+cu128
torch_cuda= 12.8
available= False
count= 8
Error 803: system has unsupported display driver / cuda driver combination
```

本次根因：`/usr/local/cuda/compat` 被放在 `LD_LIBRARY_PATH` 最前面，PyTorch 优先加载了容器 compat driver 库，而不是宿主机挂入的 `/usr/local/nvidia/lib64`。

已验证可用处理：

```shell
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/bin:/usr/bin:/sbin:/usr/sbin:/opt/ssh-static/bin
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
```

验证成功输出：

```text
torch= 2.7.1+cu128
torch_cuda= 12.8
available= True
count= 8
tensor([0.], device='cuda:0')
NVIDIA H200
```

### `device_count=8` 但 `torch.cuda.is_available()=False`

这不是可用状态。必须继续用下面命令验证 CUDA tensor：

```shell
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.empty(1, device='cuda'))"
```

本次遇到的根因：

```text
宿主机：Driver 570.133.20 / CUDA 12.8
容器：torch.version.cuda = 13.2
错误：The NVIDIA driver on your system is too old
```

结论：宿主驱动不能升级时，需要换 `torch.version.cuda <= 12.8` 的 NeMo AutoModel 镜像。不要继续跑 smoke。

### `run_glm_5.2_smoke.sh: No such file or directory`

现象：

```text
bash: tests/functional_tests/speculative/run_glm_5.2_smoke.sh: No such file or directory
```

原因：当前容器的 `/opt/Automodel` 源码版本和本地 IDE 打开的 checkout 不一致，镜像自带源码里没有这个新脚本。

验证：

```shell
cd /opt/Automodel
find tests -name '*glm*5*' -o -name '*smoke*.sh'
```

处理：切到或挂载包含 `tests/functional_tests/speculative/run_glm_5.2_smoke.sh` 的 Automodel checkout，再跑 smoke。

### `uv` 去拉 `https://download.pytorch.org/whl/cu130/torch/`

现象：

```text
uv 0.8.3
unknown field `extra-build-dependencies`
error: Failed to fetch: `https://download.pytorch.org/whl/cu130/torch/`
```

原因：

- 旧 `vllm-openai:v0.10.0` 自带的 `uv 0.8.3` 太旧，不认识当前 `pyproject.toml` 里的 `[tool.uv.extra-build-dependencies]`；当前 PyTorch 基础镜像已经有 `uv 0.9.9`，脚本会直接复用，不再运行时强制升级。
- 临时环境里执行普通 `uv pip install -e .` 会重新解析 `torch` 依赖，可能走到 repo 默认的 CUDA 13 index。

处理：

```shell
UV_BIN="$(command -v uv || true)"
test -n "$UV_BIN" || { echo "uv missing; use the verified PyTorch 2.10 cu128 image or bake uv into the base image first"; exit 1; }
"$UV_BIN" --version

"$UV_BIN" pip install --no-build-isolation --no-deps -e .
```

本次脚本已经内置这个处理。

### `torch` 被换成 `2.13.0+cu130`

现象：

```text
+ torch==2.13.0
+ nvidia-cuda-runtime==13.0.96
RuntimeError: The detected CUDA version (12.8) mismatches the version that was used to compile PyTorch (13.0)
```

判断：`17.log` 中 `torchao` 已经安装成功，新的失败点在 DeepEP 编译。根因是临时安装 `torchdata` 等运行依赖时，`uv` 重新解析依赖，把基础镜像的 `torch==2.9.1+cu128` 替换成了 `torch==2.13.0+cu130`，导致宿主 Driver 570 / CUDA 12.8 无法继续。

处理：

```shell
cd /tmp/Automodel-torch210
source /opt/venv/bin/activate
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.__file__)
PY
```

如果看到 `2.13.0` 或 `cu130`，不要继续编 DeepEP，直接重建 venv：

```shell
RESET_WORKSRC=1 RESET_VENV=1 FORCE_DEEP_EP_BUILD=1 \
bash /mnt/shared-storage-user/ailab-sys/liutong/xtuner/Znote/glm/glm52_h200_container_smoke.sh
```

本次脚本已经内置处理：`torchdata`、`torchao`、DeepEP 都用 `--no-deps`，并在安装后检查 torch 仍是 CUDA 12.8。

### `infiniband/mlx5dv.h: No such file or directory`

现象：

```text
fatal error: infiniband/mlx5dv.h: No such file or directory
```

判断：`18.log` 里 `torch==2.9.1+cu128` 已经保住，`torchao` 也安装成功；新的失败点是 DeepEP 编译阶段缺系统 RDMA/IB 开发头文件。项目自己的 `docker/Dockerfile` 在构建 DeepEP 前也会安装：

```text
rdma-core libibverbs-dev
```

处理：当前脚本默认会自动补包；长期建议烘进 `vllm0.15.1-torch2.10-cu128` 派生镜像。

`20.log` 已验证的有效路径是：容器为 `jammy` / Ubuntu 22.04，切到平台内网 jammy 源后，先禁用 `developer.download.nvidia.com` 这个外部 NVIDIA apt 源，再安装：

```text
rdma-core libibverbs-dev
```

脚本现在已内置这套逻辑：如果 `/usr/include/infiniband/mlx5dv.h` 已存在，会直接跳过；如果缺失且 `AUTO_INSTALL_RDMA_DEPS=1`，会自动处理 apt 源并安装包；如果设置 `AUTO_INSTALL_RDMA_DEPS=0`，会在 DeepEP 编译前直接报错退出。

如果 `apt-get update` 仍访问 `archive.ubuntu.com` / `security.ubuntu.com` 并超时，先确认系统版本：

```shell
cat /etc/os-release
```

`19.log` 里的容器是 `jammy` / Ubuntu 22.04，因此应改 `/etc/apt/sources.list` 为平台内网 jammy 源：

```shell
cp -a /etc/apt/sources.list /etc/apt/sources.list.bak.$(date +%s) 2>/dev/null || true
cat >/etc/apt/sources.list <<'EOF'
deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy main restricted universe multiverse
deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-security main restricted universe multiverse
deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-updates main restricted universe multiverse
deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-backports main restricted universe multiverse
EOF
```

如果是 Ubuntu 24.04 / `noble`，不要写上面的 jammy 源，应改 `/etc/apt/sources.list.d/ubuntu.sources` 为 noble 源。

如果还有外网 CUDA 源报 `developer.download.nvidia.com` DNS 失败，可以临时禁用对应源文件；当前只安装 RDMA 头文件，不需要访问 NVIDIA apt 源：

```shell
grep -R "developer.download.nvidia.com" -n /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null || true
mkdir -p /etc/apt/disabled-sources
for f in $(grep -Rl "developer.download.nvidia.com" /etc/apt/sources.list.d 2>/dev/null); do
  mv "$f" "/etc/apt/disabled-sources/$(basename "$f").disabled"
done
```

```shell
apt-get update
apt-get install -y --no-install-recommends rdma-core libibverbs-dev

ARCH_LIB="$(dpkg-architecture -qDEB_HOST_MULTIARCH)"
test -f "/usr/lib/${ARCH_LIB}/libmlx5.so" || ln -sf "/usr/lib/${ARCH_LIB}/libmlx5.so.1" "/usr/lib/${ARCH_LIB}/libmlx5.so"
mkdir -p /opt/rdma-core/build
ln -sfn /usr/include /opt/rdma-core/build/include
ln -sfn "/usr/lib/${ARCH_LIB}" /opt/rdma-core/build/lib

ls /usr/include/infiniband/mlx5dv.h "/usr/lib/${ARCH_LIB}/libmlx5.so"
```

如果是手动补包路径，补完后不需要重建 venv，继续强制重编 DeepEP：

```shell
RESET_WORKSRC=1 FORCE_DEEP_EP_BUILD=1 \
bash /mnt/shared-storage-user/ailab-sys/liutong/xtuner/Znote/glm/glm52_h200_container_smoke.sh
```

本次脚本已经内置前置检查和自动安装逻辑：如果缺 `/usr/include/infiniband/mlx5dv.h`，会先尝试按平台内网 apt 源安装 RDMA 依赖，再进入 DeepEP 编译。

### `sh: 1: /bin/nvcc: not found`

现象：

```text
Training:   0%|          | 0/8 [00:00<?, ?step/s]sh: 1: /bin/nvcc: not found
RuntimeError: Failed to compile the code, compile command: /bin/nvcc ... -I/include -L/lib64 -lcudart ...
```

判断：`21.log` 已经越过 CUDA、`torch==2.9.1+cu128`、`torchao`、DeepEP 安装、GLM-5.2 snapshot、8 卡分布式初始化，并进入第一个训练 step。新的失败点是 DeepEP HybridEP 的运行时 JIT 编译。编译命令里出现 `/bin/nvcc`、`-I/include`、`-L/lib64`，说明 JIT 没拿到正确的 `CUDA_HOME`，而不是 DeepEP 没装上。

处理：脚本现在会显式设置：

```shell
CUDA_HOME=/usr/local/cuda
CUDA_PATH=/usr/local/cuda
CUDACXX=/usr/local/cuda/bin/nvcc
CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc
```

并提前检查：

```shell
$CUDA_HOME/bin/nvcc --version
test -d "$CUDA_HOME/include"
test -d "$CUDA_HOME/lib64"
```

重新执行：

```shell
RESET_WORKSRC=1 FORCE_DEEP_EP_BUILD=1 \
bash /mnt/shared-storage-user/ailab-sys/liutong/xtuner/Znote/glm/glm52_h200_container_smoke.sh
```

### 尝试旧 DeepEP 版本并不能直接解决 CUDA 12.8 适配

背景：当前 GLM-5.2 smoke 走 MoE，`ep_size=8`，第一步训练会进入 DeepEP / HybridEP 的 token dispatch。这里不是普通 Python 依赖问题，而是 DeepEP 源码、CUDA toolkit、NVSHMEM/RDMA、运行时 JIT 同时要匹配。

本次确认过两条 DeepEP 版本线：

```text
pyproject.toml / uv.lock:
  7febc6e25660af0f54d95dd781ecdcd62265ecca

docker/Dockerfile:
  42144303752422ade37f24bca9e2dde12df70e09
```

结论：

- `7febc6e...` 是项目 lock 中的旧版本，能安装并进入 8 卡训练 step，但 HybridEP 运行时 JIT 在 CUDA 12.8 下报 `cp_async_bulk` 签名不匹配；说明“换旧版 EP”不是直接解法。
- `4214430...` 是项目 Dockerfile 中的 DeepEP commit，包含 HybridEP padding 修复；但 Dockerfile 当前基础是 CUDA 13.x，并安装 `nvidia-nvshmem-cu13`。把它放到 CUDA 12.8 环境里编译时，会遇到 `cudaLaunchAttributeNvlinkUtilCentricScheduling` 这类 CUDA 13 launch attribute，仍需 CUDA 12.8 兼容补丁。
- 官方 `nemo-automodel:26.06.00` 容器里的 DeepEP / HybridEP 可能已经配好，但该镜像 `torch.version.cuda=13.2`，当前 H200 宿主 `Driver 570.133.20 / CUDA 12.8` 无法运行。

因此当前判断是：项目里还没有一个已经验证可直接用于 `CUDA 12.8 + torch>=2.10 + GLM-5.2 HybridEP smoke` 的 DeepEP commit。短期先切到 `vllm0.15.1-torch2.10-cu128` 基础镜像补齐 DeepEP 当前 README 的 torch 下限；如果仍然卡在 `cudaLaunchAttributeNvlinkUtilCentricScheduling` 或 `cp_async_bulk`，再对 DeepEP 做 CUDA 12.8 兼容 patch。长期应把 DeepEP commit、RDMA/NVSHMEM/NVTX 依赖和 patch 一起烘进平台镜像。

### `No module named 'torchao'`

现象：

```text
ModuleNotFoundError: No module named 'torchao'
```

判断：`14.log` / `15.log` / `16.log` 里 CUDA、8 卡、`torch=2.9.1+cu128`、`backend_override=True` 都已经通过；失败发生在脚本的依赖自检阶段。根因是临时环境用 `--no-deps -e .` 安装 AutoModel，本体装上了，但没有自动补齐 `torchao`。另外，从项目根目录直接执行 `uv pip install torchao...` 时，`uv` 会读取当前 `pyproject.toml`，日志只出现 `Audited` 而没有 `Installed`，因此脚本现在切到 `/tmp` 再执行临时依赖安装。

处理：

```shell
cd /tmp/Automodel-torch210
source /opt/venv/bin/activate
UV_BIN="$(command -v uv)"
(cd /tmp && "$UV_BIN" pip install --python /opt/venv/bin/python --reinstall-package torchao "torchao==0.14.0" --index-strategy unsafe-best-match)
python -c "from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp; print(precompute_float8_dynamic_scale_for_fsdp)"
```

本次脚本已经内置这个处理。

### `init_device_mesh() got an unexpected keyword argument 'backend_override'`

现象：

```text
TypeError: init_device_mesh() got an unexpected keyword argument 'backend_override'
```

判断：这已经进入 8 卡分布式初始化阶段，说明 CUDA、DeepEP、GLM-5.2 snapshot、`datasets`、`transformers.glm_moe_dsa`、`torchao` 都已经越过。根因不是 CUDA runtime，而是旧 `vllm-openai:v0.10.0` 镜像里的 `torch==2.7.1+cu128` 太旧；当前 AutoModel 源码会调用：

```python
init_device_mesh(..., backend_override=...)
```

官方 PyTorch 文档里，`backend_override` 从 PyTorch 2.9 的 `init_device_mesh` 签名开始出现；PyTorch 2.7/2.8 没有这个参数。

处理方向：不要改 AutoModel 源码，升级运行环境。目标环境应同时满足：

```text
torch >= 2.9
torch.version.cuda <= 12.8   # 当前 H200 宿主 Driver 570.133.20 / CUDA 12.8
```

优先找平台镜像：

```text
PyTorch >= 2.9
CUDA 12.8 / cu128
Python 3.12 可用
nvcc/gcc/g++/ninja/cmake 可用或可安装
```

更换基础镜像或升级 torch 后，需要先验证：

```shell
source /opt/venv/bin/activate
python - <<'PY'
import inspect
import torch
from torch.distributed.device_mesh import init_device_mesh

print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("gpu_count=", torch.cuda.device_count())
print("backend_override=", "backend_override" in inspect.signature(init_device_mesh).parameters)
print(torch.empty(1, device="cuda"))
PY
```

只有同时看到 `cuda_available=True`、`gpu_count=8`、`backend_override=True`，再继续跑 smoke。

### `No module named 'datasets'`

现象：

```text
ModuleNotFoundError: No module named 'datasets'
```

判断：这已经进入 `### 4/4  8-GPU real-weight training smoke`，并且日志里显示失败进程使用的是：

```text
binary: /opt/venv/bin/python
```

所以前面的 venv / `torchrun` 路径问题已经解决。新的根因是临时环境用 `uv pip install --no-deps -e .` 安装 AutoModel，本体装上了，但 `pyproject.toml` 的基础 runtime 依赖没有自动安装。

处理：

```shell
cd /tmp/Automodel-torch210
source /opt/venv/bin/activate
uv pip install "datasets>=4.0.0" pybind11 pyyaml torchdata --index-strategy unsafe-best-match
```

本次脚本已经内置这个处理。

### `No module named 'transformers.models.glm_moe_dsa'`

现象：

```text
ModuleNotFoundError: No module named 'transformers.models.glm_moe_dsa'
```

原因：基础镜像自带的 `transformers` 版本不一定包含 GLM-5.2 需要的 `glm_moe_dsa` 模块；而临时环境为了避免重装 torch，AutoModel 本体是用 `--no-deps -e .` 安装的，所以不会自动升级 `transformers`。

处理：

```shell
cd /tmp/Automodel-torch210
source /opt/venv/bin/activate
uv pip install "transformers==5.12.1" --index-strategy unsafe-best-match
python -c "from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig; print(GlmMoeDsaConfig)"
```

本次脚本已经内置这个处理。
