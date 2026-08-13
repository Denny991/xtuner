# GLM-5.2 + NeMo AutoModel 调研与单机试跑

> **GLM-5.2 技术汇报**见：[GLM-5.2-技术汇报.md](./GLM-5.2-技术汇报.md)
> 整理自 2026-07-03 对本仓库的调研。  
> 个人主副本：`~/ZmyCode/my-note20260120/notes/training/nemo-automodel/`

---

## 1. NeMo AutoModel 是什么

PyTorch 原生、YAML 驱动的 LLM/VLM/Diffusion 训练框架，HuggingFace 模型通过 `NeMoAutoModelForCausalLM` 桥接，分布式支持 FSDP2 / EP / PP / CP。

```text
automodel <recipe.yaml> --nproc-per-node=8
    → recipes/          训练入口
    → components/       model / dataset / distributed / checkpoint
    → _transformers/    NeMoAuto* 桥接层
```

官方文档：https://docs.nvidia.com/nemo/automodel/latest/index.html

---

## 2. GLM-5.2 支持现状（2026-06-21 官宣）

| 项 | 内容 |
|----|------|
| HF 模型 | [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) |
| 架构类 | `GlmMoeDsaForCausalLM` |
| 实现路径 | `nemo_automodel/components/models/glm_moe_dsa/` |
| 规模 | ~355B MoE，78 层，256 routed experts，hidden=6144 |
| 状态 | 正式 SFT recipe 已提供；model coverage 标记为进行中 |

### 三个关键技术

| 技术 | 作用 |
|------|------|
| **IndexShare DSA** | shared 层复用上一层 full 层的 top-k 稀疏选择；PP 需跨 stage 传递 `topk_indices` |
| **TileLang 稀疏 kernel** | `backend.attn: tilelang`，Indexer + Sparse MLA 融合 CUDA kernel |
| **Context Parallel** | 32K recipe：CP=8 + packed THD + TileLang |

详见：[GLM-5.2-DSA-MLA-技术附录.md](./GLM-5.2-DSA-MLA-技术附录.md)

---

## 3. 硬件需求

| 场景 | 并行配置 | 硬件 |
|------|----------|------|
| 正式 SFT（HellaSwag） | EP=64, PP=4 | **32 节点 x 8 卡** |
| 32K 长上下文 SFT | EP=64, PP=4, CP=8 | 多节点 + TileLang |
| **单机 H200 smoke（减层）** | EP=8, PP=1, CP=1, 6 层 target | **1 节点 x 8 卡（>=80GB）** |

单机 8 卡这里应先跑 **减层 smoke**，不要把它理解成完整 78 层正式 SFT。仓库里的正式/全量 GLM-5.2 路径按多节点规划（DSpark full target 建议 `ep_size>=32/64`）；瓶颈主要是 MoE expert 权重驻留内存，降 `seq_length` 或开 activation checkpointing 不能根治。

8 x H200（141GB）比文档验证的 8 x H100 80GB 更宽裕；减层 smoke 约 ~24 GiB/rank。

---

## 4. 官方 Recipe 索引

| 文件 | 用途 |
|------|------|
| `examples/llm_finetune/glm/glm_5.2_tulu3_4k_tilelang_100k.yaml` | 4K SFT，EP=64 PP=4 |
| `examples/llm_finetune/glm/glm_5.2_tulu3_32k_tilelang_cp8.yaml` | 32K + CP=8 |
| `examples/llm_finetune/glm/glm_5.2_hellaswag_pp.yaml` | HellaSwag benchmark |
| `examples/speculative/dspark/glm_5.2_dspark.yaml` | DSpark draft（多节点） |

文档：`docs/model-coverage/llm/thudm/glm5-moe-dsa.mdx`

---

## 5. 单机 8 卡：减层 Smoke 试跑

### Smoke 是什么

**Smoke test = 冒烟测试**：快速验证整条训练 pipeline 能跑通，不是训出可用模型。

| 维度 | Smoke | 正式 SFT |
|------|-------|----------|
| 层数 | 6 层（`target_num_hidden_layers=6`） | 78 层 |
| 数据 | 64 条假 jsonl | 真实数据集 |
| 成功标准 | loss 有限且略降 | 收敛 / 指标 |
| 耗时 | 数分钟 | 小时~天 |

### 前置依赖

- HybridEP 或 DeepEP（MoE token dispatch，缺则报 `HybridEP is not installed`；官方 `nemo-automodel:26.06.00` 容器已内置，本地 uv / bare-metal 才需要 `--extra moe`）
- GLM-5.2 checkpoint（Hub ID 或本地路径；H200 机器建议先用本地路径。若是 HF cache 结构 `blobs/refs/snapshots`，`TARGET` 要指到 `snapshots/<hash>` 那层）
- 8 GPU + NCCL（smoke 脚本固定 `torchrun --standalone --nproc_per_node=8`）

### H200 平台环境判定（2026-07-08）

这次平台排障结论：

| 检查项 | 结论 |
|--------|------|
| GLM-5.2 权重路径 | 正确，`refs/main` 指到 `snapshots/120edc221b4e5f6408384be245d7b49124012fcc`，且能看到 `config.json` / `model.safetensors.index.json` |
| DeepEP | 容器内可 import，`deep_ep=True` |
| GPU 设备节点 | `/dev/nvidia0-7` 存在 |
| driver 库路径 | 容器初始 `nvidia-smi` 找不到 `libnvidia-ml.so`，需补 `LD_LIBRARY_PATH` |
| 关键阻塞 | 当前镜像内 `torch.version.cuda=13.2`，但宿主机 `Driver 570.133.20 / CUDA 12.8`，PyTorch 报 driver too old |
| 源码版本 | 镜像自带 `/opt/Automodel` 可能没有 `run_glm_5.2_smoke.sh`，需挂载/切到包含该脚本的 checkout |

因此不能只看：

```bash
python -c "import torch; print(torch.cuda.device_count())"
```

`device_count=8` 仍可能同时伴随 `torch.cuda.is_available()=False`。真正的 smoke 前置判定必须能分配 CUDA tensor：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/nvidia/bin:$PATH

python -c "import torch; print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('available=', torch.cuda.is_available()); print('count=', torch.cuda.device_count()); x=torch.empty(1, device='cuda'); print(x); print(torch.cuda.get_device_name(0))"
```

合格输出必须满足：

```text
torch.version.cuda <= 12.8   # 对当前 Driver 570.133.20 节点
available=True
count=8
tensor(..., device='cuda:0')
NVIDIA H200
```

如果仍然出现：

```text
The NVIDIA driver on your system is too old
```

不要继续跑 smoke。宿主驱动不能升级时，选择 **PyTorch CUDA <= 12.8** 的 NeMo AutoModel 镜像；否则换到 driver 更新的 H200 节点。

### 命令

实际平台启动脚本见：[script.md](./script.md)。

```bash
# 进入平台容器后执行
cd /opt/Automodel
source /opt/venv/env.sh

# HF cache 结构用 refs/main 定位 snapshot；根目录不能直接当 TARGET。
MODEL=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--zai-org--GLM-5.2
HASH="$(cat "$MODEL/refs/main")"
GLM52="$MODEL/snapshots/$HASH"

# 确认这里能看到 config 和 safetensors index。
ls "$GLM52/config.json" "$GLM52/model.safetensors.index.json"

# 确认当前源码含有 smoke 脚本；镜像自带 /opt/Automodel 可能偏旧。
test -f tests/functional_tests/speculative/run_glm_5.2_smoke.sh

# 确认 CUDA 真的可用；不能只看 device_count。
export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/nvidia/bin:$PATH
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.empty(1, device='cuda'))"

# 平台内网镜像通常已内置 DeepEP / HybridEP；普通 smoke 不需要重新 uv sync。
TARGET="$GLM52" \
  bash tests/functional_tests/speculative/run_glm_5.2_smoke.sh
```

脚本：`tests/functional_tests/speculative/run_glm_5.2_smoke.sh`

### H200 单机减层配置是否对

- 对：`run_glm_5.2_smoke.sh` 会把派生 config 改成 `ep_size=8`、`activation_checkpointing=true`、`target_num_hidden_layers=6`、`draft_num_hidden_layers=2`、`target_layer_ids=[3, 4, 5]`、`seq_length=512`。
- 不要原样在单机跑 `examples/speculative/dspark/glm_5.2_dspark.yaml`：它默认是 full DSpark，多节点 `ep_size=64`。
- 不要把 32K TileLang/CP recipe 当作这次单机目标：那条是正式长上下文多节点 SFT 路径，不是 H200 单机减层 smoke。

### Smoke 脚本做了什么

1. CPU 单元测试 `tests/unit_tests/speculative/test_dspark_glm_5_2.py`
2. 生成 64 条 OpenAI messages 格式假数据
3. 从 `examples/speculative/dspark/glm_5.2_dspark.yaml` 派生临时 config（EP=8, 6 层, seq=512）
4. `torchrun --nproc_per_node=8` 跑 1 个 epoch
5. 检查 `dspark_train_metrics.jsonl` 里 loss 有限且下降

### Smoke 验证点（5 条）

1. FP8 checkpoint dequant 到 bf16，forward 有限
2. 跨 EP rank 捕获的 hidden states 完整
3. frozen target forward 用 2D mask + use_cache=False
4. ep_size=8 整除 n_routed_experts=256（每 rank 32 experts）
5. 每 rank 显存 ~24 GiB（仅 6 层）

---

## 6. 建议调研顺序

```text
1. docs/model-coverage/llm/thudm/glm5-moe-dsa.mdx
2. XTuner 仓库 `Znote/glm/GLM-5.2-DSA-MLA-技术附录.md`
3. 单机 smoke 验证环境
4. 正式 SFT → 规划多节点（至少 4 节点 EP=32，推荐 8 节点 EP=64）
```

---

## 7. 参考（本仓库）

- README GLM-5.2 新闻：`README.md` 2026-06-21 条目
- DSpark 说明：`examples/speculative/dspark/README_glm_5.2.md`
- 模型实现：`nemo_automodel/components/models/glm_moe_dsa/model.py`
