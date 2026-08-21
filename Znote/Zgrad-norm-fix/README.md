# GLM-5.2 30B SFT `grad_norm` 波动复现记录

## 背景

- Case：`glm5-2-sft-30B`
- 问题：相同配置多次训练的 `grad_norm` 存在明显波动。CI 输出中的 position 15 对应实际
  training Step 16，该位置曾出现约 7% 的相对误差。
- 目标：在 8 张 H200 上复现 CI 训练，收集三组 tracker 做两两比较。
- 镜像：`registry.h.pjlab.org.cn/ailab-llmrazor/xtuner:pt29_latest`
- Commit：`51c775d8aa70a10f8e23092372b002ea9dbe3233`

## 结论摘要

- [已完成] 使用 8 张 H200 完成原始配置及多组单变量对照。DeepEP、CUTLASS Grouped GEMM、
  MTP loss、整个 MTP block 和 NoAux router correction-bias 更新均不是问题成立的必要条件。
- [已完成] 固定输入微内核测试直接观察到：TileLang Sparse MLA 的 forward 和 `dQ` 可重复，
  只有 backward `dKV` 在独立进程间不一致；对应实现恰好通过 `T.atomic_addx4` 将多个 query
  tile 的结果并发累加到相同 KV 地址。
- [已完成] Torch reference SparseMLA 在单卡微内核中 output、`dQ`、`dKV` 五次逐位一致；在
  真实 8 卡 GLM-5.2 SFT（seq1024、MTP 开启、35 step）中三次训练的 loss、`grad_norm`、
  maxvio、学习率和 token 数也全部逐值一致。
- [已完成] 在 cuDNN DSA 微内核中对照了 `torch.backends.cudnn.deterministic=False/True`。
  开启该开关后 `dKV` 仍然不能逐位重复，只是相对 L2 波动均值从 `2.72e-5` 降到
  `2.38e-5`，因此该开关不是 cuDNN DSA backward 的确定性修复。
- [方案调整] 当前证据最充分的根因仍是 Sparse MLA backward `dKV` 并行归约带来的
  低位波动。专门实现确定性 kernel 会增加开发、显存和性能成本，而它主要服务于 CI；
  当前更合理的工程方案是保留高性能 TileLang 后端，将 `grad_norm` 从逐 step `1e-6`
  改为基于分位数的曲线门禁。`P95 < 20%` 已连续两次通过正式 baseline 的 `grad_norm`
  检查，其中第二次全部 CI 指标通过；当前将其作为候选门禁继续积累样本。

## 启动平台容器

```bash
rlaunch --gpu=8 --memory=1600000 --cpu=128 \
  --charged-group=kj_gpu --private-machine=group \
  --namespace=ailab-sys \
  --image=registry.h.pjlab.org.cn/ailab-llmrazor/xtuner:pt29_latest \
  --mount=gpfs://gpfs1/ailab-sys/liutong:/mnt/shared-storage-user/ailab-sys/liutong \
  --mount=gpfs://gpfs1/shipengcheng:/mnt/shared-storage-user/shipengcheng \
  --mount=gpfs://gpfs1/zhaopenghao/:/mnt/shared-storage-user/zhaopenghao/ \
  --mount=gpfs://gpfs1/llmrazor-share/:/mnt/shared-storage-user/llmrazor-share/ \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --entrypoint= -d -- bash -c "sleep infinity"
```

## 环境准备

```bash
cd /mnt/shared-storage-user/ailab-sys/liutong
git clone https://github.com/InternLM/xtuner.git xtuner-glm52-repro
cd xtuner-glm52-repro
git checkout 51c775d8aa70a10f8e23092372b002ea9dbe3233

# 配置内部 PyPI 源。
export PIP_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/pypi/simple/"
export PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export PIP_TRUSTED_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"

# 允许在该临时容器的系统 Python 环境中安装包。
export PIP_BREAK_SYSTEM_PACKAGES=1
python -m pip install -e '.[all]'
python -m pip install more-itertools pytest-xdist

# CI 针对 Torch 2.9.1 使用的 cuDNN 补丁；源码安装后需要最后执行。
python -m pip install nvidia-cudnn-cu12==9.15.1.9
```

`pip` 会提示 Torch 声明的 cuDNN 版本为 `9.10.2.21`，这是预期的依赖警告；CI 会主动覆盖为 `9.15.1.9`。

## 已验证环境

```text
torch: 2.9.1+cu128
cuDNN package: 9.15.1.9
cuDNN runtime: 91501
GPU count: 8
CUDA smoke test: 16.0
```

## 启动第一次训练

```bash
export MODEL_PATH=/mnt/shared-storage-user/llmrazor-share/model/GLM-5.2-30B
export ALPACA_PATH=/mnt/shared-storage-user/llmrazor-share/data/alpaca
export XTUNER_GC_ENABLE=1
export SWAP_OPTIMIZER=0
export XTUNER_ACTIVATION_OFFLOAD=0
export XTUNER_USE_CUTLASS_GROUP_GEMM=1
export XTUNER_DETERMINISTIC=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export GITHUB_RUN_ID=manual-glm52-run1
export WORK_DIR=/mnt/shared-storage-user/ailab-sys/liutong/glm52-repro/run1
mkdir -p "$WORK_DIR"
set -o pipefail

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  xtuner/v1/train/cli/sft.py \
  --config autotest/config/glm5p2_30B.py \
  2>&1 | tee "$WORK_DIR/train.log"
```

训练结束后将 `GITHUB_RUN_ID` 和 `WORK_DIR` 分别改为 `run2`、`run3`，每次启动新的 `torchrun` 进程。

## 结果位置

```bash
find /mnt/shared-storage-user/ailab-sys/liutong/glm52-repro \
  -path '*/logs/exp_tracking/rank0/tracker.jsonl' \
  -print
```

CI baseline：

```text
/mnt/shared-storage-user/llmrazor-share/qa-llm-cicd/xtuner-baselines/latest/glm5-2-sft-30B/tracker.jsonl
```

## 三次运行结果

三份 tracker 均包含完整的 35 step。学习率和每步文本 token 数完全一致，`loss/local_loss` 和
`loss/reduced_llm_loss` 的最大两两相对误差均低于 0.23%，但 `grad_norm` 从 Step 16 开始明显分叉。

| 对比 | 首次触发 CI 的 step | `grad_norm` 最大相对误差 | 最大误差 step |
|---|---:|---:|---:|
| run1 vs run2 | 16 | 20.14% | 32 |
| run1 vs run3 | 16 | 18.51% | 24 |
| run2 vs run3 | 16 | 26.52% | 32 |

Step 16 的三次 `grad_norm` 分别为：

```text
run1: 21.44484711
run2: 22.40070724
run3: 19.73213387
```

结果说明数据顺序和学习率调度一致，主要差异出现在反向传播。结合 TileLang Sparse MLA backward
中的浮点原子累加，下一步使用 `cudnn_dsa` backend 做单变量对照：它保留相同的 TileLang forward
和 top-k indexer，只替换 Sparse MLA backward。

### DeepEP 排除实验

将 `moe_cfg.dispatcher` 从 `deepep` 改为 `all2all`，其余训练配置保持不变。两次 all2all
运行仍在 Step 16 首次触发 CI 的 `grad_norm` 判定，最大相对误差为 8.00%（Step 32），共有
17 个 step 会触发当前 CI checker。`loss/local_loss` 和 `loss/reduced_llm_loss` 最大相对误差
均低于 0.1%。因此 DeepEP 不是该问题的必要条件，可以从主要嫌疑中排除。

### CUTLASS Grouped GEMM 排除实验

在 all2all 配置上设置 `XTUNER_USE_CUTLASS_GROUP_GEMM=0`，改用 Triton Grouped GEMM，连续
运行两次后仍然复现：首次触发 CI 判定的位置为 Step 15，`grad_norm` 最大相对误差为 13.54%
（Step 32），共有 20 个 step 触发判定。Step 16 的相对误差为 3.72%。loss、学习率和 token
数仍基本一致。因此 CUTLASS Grouped GEMM 也不是该问题的必要条件。

### TileLang Sparse MLA backward 微内核复现

在单张 H200 上使用完全相同的随机种子、输入和上游梯度，分别启动 5 个独立进程执行同一个
Sparse MLA forward/backward（序列长度 4096、64 heads、head dim 576、value dim 512、
top-k 2048）。结果如下：

- 5 次 `input_probe` 完全一致；
- 5 次 `dQ_sha256` 完全一致；
- 5 次 `dKV_sha256` 全部不同；
- 打印精度下的 `dQ_norm` 和 `dKV_norm` 一致，但哈希差异证明 `dKV` 并非逐位确定。

这直接证明非确定性存在于 TileLang Sparse MLA backward 的 `dKV` 路径。对应实现通过
`T.atomic_addx4` 将多个并行计算结果累加到 `dKV`；浮点加法不满足结合律，原子操作的实际
到达顺序变化会造成低位差异。训练中这些微小差异经过优化器状态持续累积，并可能被 MoE/DSA
的离散 top-k 选择放大，最终表现为 `grad_norm` 在某些 step 明显分叉。

因此 Step 16（或关闭 CUTLASS 后的 Step 15）只是差异首次超过 CI 有效阈值的位置，并不是
非确定性从该 step 才开始，也不是一个固定的特殊训练分支。

### cuDNN DSA 对照环境

共享环境 `/mnt/shared-storage-user/llmrazor-share/comm_env/glm52-pt121-cu132` 已确认包含
`nvidia-cudnn-frontend==1.26.0`，且可以精确导入
`cudnn.deepseek_sparse_attention.sparse_attention_backward`。环境同时包含
`nvidia-cutlass-dsl==4.5.0`、`torch==2.12.1+cu132` 和 `cuda-python==13.3.1`。

该环境可以用于验证替换 Sparse MLA backward 后 `dKV` 是否稳定，但它与原 CI 的
Torch 2.9.1/CUDA 12.8/frontend 1.10 环境不同，因此结果属于后端因果对照，不能解释为原 CI
环境下的严格等价复现。容器中没有 `conda` 命令时，直接使用该环境的 `bin/python`，避免误用
`/usr/bin/python`。

使用该环境中的 cuDNN DSA backward 对相同输入运行 5 个独立进程后，forward 输出和 `dQ`
的 SHA256 均完全一致，但 5 个 `dKV` SHA256 全部不同。相对于 repeat 1，其余运行分别有
707～767 个元素发生变化（总计 2,359,296 个元素，约 0.03%），最大绝对误差为
0.0078125～0.015625，相对 L2 误差为 `1.99e-5～2.93e-5`。因此 cuDNN DSA backward 也不是
逐位确定的，不能作为消除非确定性的直接修复；是否能让完整训练的差异保持在 CI 阈值以内，
仍需先与同一 cu132 环境下的 TileLang backward 做等输入、等指标的量级对照。

随后在相同 cu132 环境、相同输入和相同 shape 下重复 TileLang backward。TileLang 相对于
repeat 1 有 818～867 个 `dKV` 元素变化，最大绝对误差为 0.0078125～0.03125，相对 L2
误差为 `2.53e-5～3.71e-5`。四组对比的相对 L2 均值约为 `3.18e-5`；cuDNN DSA 对应均值
约为 `2.35e-5`，降低约 26%。cuDNN 的变化元素数均值约为 736，TileLang 约为 850，降低
约 13%。因此 cuDNN DSA 的波动一致地小于 TileLang，但仍处于同一数量级，是否足以让完整
训练通过 CI 需要用两到三次 35-step SFT 实测，不能从微内核结果直接保证。

### `torch.backends.cudnn.deterministic` 对照实验

为确认 PyTorch 的 cuDNN 确定性开关是否能约束 DSA backward，在相同 Torch 2.12.1/
CUDA 13.2/cuDNN frontend 1.26 环境、相同输入和 seq4096 shape 下，对
`TORCH_CUDNN_DETERMINISTIC=0/1` 分别运行 10 个独立进程。两组都同时开启：

```python
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = flag
```

| `cudnn.deterministic` | `dKV` 逐位一致 | 平均变化元素数 | 最大绝对误差 | 平均相对 L2 |
|---|---|---:|---:|---:|
| `False` | 否 | 711.3 / 2,359,296 | 0.015625 | `2.719e-5` |
| `True` | 否 | 704.6 / 2,359,296 | 0.0078125 | `2.381e-5` |

开启开关后，变化元素数均值仅下降约 0.95%，相对 L2 均值下降约 12.4%，但 10 次
`dKV` SHA256 仍然全部不同。这说明 `torch.backends.cudnn.deterministic=True` 主要影响
PyTorch/cuDNN 能够识别和选择的确定性算法，并没有让当前
`cudnn.deepseek_sparse_attention` backward 切换到逐位确定的 `dKV` 归约路径。

因此没有必要只因为该开关而重跑三次完整 SFT：微内核已经证明它不能提供
bitwise deterministic backward，而训练级差异还会继续被优化器和离散 top-k 选择放大。

### cuDNN DSA 完整 SFT 三次重复实验

在 Torch 2.12.1/CUDA 13.2/frontend 1.26 环境中，使用 all2all dispatcher、cuDNN DSA
backward 和 `debug_skip_save=True` 完成三次 35-step SFT。三个 tracker 均完整，学习率与每步
token 数完全一致。

三组 `grad_norm` 两两比较均失败：run1/run2 最大单向相对误差 68.36%（Step 32），run1/run3
最大 12.19%（Step 24），run2/run3 最大 39.43%（Step 32）；分别有 20、15、21 个 step 触发
CI checker。run2 后半程偏离最明显，但 run1/run3 单独比较也有 15 个失败 step，因此不是单个
异常 run。Step 32 三次实际值为 3.28968668、5.53853559、3.35464025，三次范围除以均值为
55.38%。

`grad_norm` 从 Step 1 已有约 `8.46e-6` 的三次相对 spread，Step 15 达 0.556%，Step 16
升至 13.89%。所以 Step 15/16 仍然只是越过 CI 阈值以及差异被明显放大的位置。local loss 和
LLM loss 的最大三次 spread 均低于 0.32%；MTP loss 在少数 step 略超 checker 的有效 0.5%
阈值；数据 token 数和学习率严格一致。

因此 cuDNN DSA 微内核中较小的 `dKV` 波动没有转化为完整训练的稳定性，直接替换 TileLang
backward 不能解决该 CI 问题。由于完整 TileLang 与 cuDNN 实验使用了不同的 Torch/CUDA
软件栈，不能据此声称 cuDNN 比 TileLang 更差；能确定的是 cuDNN DSA 同样无法通过重复训练
精度检查。

### Qwen3-30B-A3B 阴性对照

为区分 XTuner 通用训练链路和 GLM 特有路径，在原 CI 软件栈（Torch 2.9.1/CUDA 12.8）中
使用 Qwen3-30B-A3B 完成三次 35-step SFT。配置对齐为 8 卡、EP4、all2all、GBS8、pack
16K、seed 0、6e-5 cosine、CUTLASS Grouped GEMM，并设置 `debug_skip_save=True`。

三次运行的 `grad_norm`、local loss、LLM loss、balancing loss、maxvio、学习率、token 数以及
显存统计在全部 35 个 step 上逐值完全一致。三个 tracker 文件的 SHA256 不同，仅因为
data/step/train time、ETA 和由时间计算的吞吐率不同。`grad_norm` 从 Step 1 的
23.597126007080078 到 Step 35 的 0.8851279616355896，三次轨迹完全相同。

Qwen 日志同样出现 `aux_loss.py` 中 `_histc_cuda` 没有 deterministic implementation 的 warning，
但 balancing loss 和 `grad_norm` 仍完全一致。因此该 warning 本身不是 GLM 分叉的充分原因。
这个阴性对照说明 FSDP、AdamW、梯度裁剪、EP4 all2all、CUTLASS Grouped GEMM、标准 Qwen
MoE greedy router 和 balancing loss 在当前配置下能够产生确定结果。后续通过 MTP、NoAux
router 和 SparseMLA 对照进一步缩小了问题范围。

### MTP loss 置零实验

保持 MTP block 的构建和执行，仅将 MTP loss scaling factor 设为 0，连续完成三次 35-step
SFT。三份 tracker 中 `loss/reduced_mtp_loss` 全部严格为 0，但 `grad_norm` 仍然分叉：三组
两两比较的最大相对误差分别为 16.57%、11.11% 和 11.30%，最早在 Step 14/15 触发 CI；
local loss 和 LLM loss 的最大相对误差低于 0.24%。

这说明 MTP loss 及其反向梯度不是非确定性的必要来源。该实验仍保留了 MTP forward，因而
继续进行了完全移除 MTP block 的对照。

### 完全移除 MTP 实验

从原始 GLM-5.2-30B checkpoint 只构建 5 个 main layer，设置 `mtp_config=None`、
`num_nextn_predict_layers=0`，并使用 `strict_load=False` 忽略 checkpoint 中未使用的 MTP
张量。三次 35-step 训练的 tracker 均不再包含 MTP metric，但 `grad_norm` 最大两两相对误差
仍为 3.49%、3.81% 和 4.22%，最早在 Step 14 触发 CI。LLM/local loss 的最大相对误差低于
0.05%。

因此整个 MTP block 也不是问题成立的必要条件。MTP 可能改变误差传播或放大量级，但不能
解释非确定性的起点。

### NoAux router correction-bias 更新排除实验

在完全移除 MTP 的配置上进一步设置 `router_bias_update_speed=0.0`，关闭动态 correction-bias
更新。三组 `grad_norm` 最大两两相对误差分别达到 63.29%、57.84% 和 22.65%，最早在
Step 17/19 触发 CI；loss 仍相对接近。关闭 bias 更新后 `maxvio` 保持较高，说明该机制原本
用于改善专家负载，而不是制造当前非确定性。

因此 NoAux router correction-bias 更新不是根因，反而可能在部分运行中抑制已经存在的数值
分叉。

### Torch 与 TileLang Sparse MLA 固定输入对照

在原 CI 软件栈 Torch 2.9.1/CUDA 12.8、单张 H200 上，使用相同随机种子、相同 q/kv、相同
上游梯度和固定 causal indices，对 Torch 与 TileLang SparseMLA 各启动 5 个独立进程。测试
shape 为 seq1024、64 heads、head dim 576、value dim 512、top-k 1024，并开启 PyTorch
deterministic algorithms。

| 后端 | output | `dQ` | `dKV` |
|---|---|---|---|
| Torch | 5 次逐位一致 | 5 次逐位一致 | 5 次逐位一致 |
| TileLang | 5 次逐位一致 | 5 次逐位一致 | 5 个不同 SHA256 |

TileLang repeat 1 与其余运行相比，每组有 133～141/589824 个 `dKV` 元素变化，约占
0.023%；最大绝对误差为 0.015625～0.03125，相对 L2 误差为
`2.08e-5～4.47e-5`，均值约 `3.15e-5`。严格确定性开关没有拦截该问题，因为 TileLang
自定义 kernel 内部的原子更新不受 PyTorch 算子级 deterministic 检查约束。

TileLang backward 的 `dKV` 路径通过两个 `T.atomic_addx4` 将不同 query tile 的结果写入
相同 KV 槽位，而 `dQ` 使用直接 copy。GPU block 到达顺序不固定，加上浮点加法不满足结合律，
能够解释为什么只有 `dKV` 发生运行间低位波动。

Torch 与 TileLang repeat 1 的 `dKV` 相对 L2 差约 4.61%，这是两个实现的累计精度/顺序造成的
系统性数值差异，不能当作 TileLang 运行间非确定性的量级。跨后端 correctness 应另按数值容差
验证，运行间非确定性应比较同一后端的独立重复结果。

### Torch SparseMLA 真实 SFT 对照

Torch reference backend 会物化 `O(S × topk × D)` 中间张量，无法直接承载原始 seq16K
训练。因此保持 GLM-5.2 权重、MTP、EP4、all2all、GBS8、优化器、seed 和 8 卡训练链路，
只将 SparseMLA backend 切换为 Torch，并把 tokenize/pack 长度降至 1024，固定运行 35 step。

三次 tracker 的以下字段在全部 35 step 上逐值完全一致：

- `grad_norm`
- local、LLM 和 MTP loss
- `maxvio`
- 学习率
- text/seqlen token 数

此前容易触发 CI 的 Step 16，三次 `grad_norm` 均严格等于
`58.338050842285156`。Step 1、15、16 和 35 的所有训练精度指标也都逐值相等，最终输出
`ALL_COMPARED_METRICS_EXACT=True`。

三个 tracker 文件本身的 SHA256 不同，仅因为 data/step/train time、ETA、吞吐率和少量显存
统计不同；这些运行状态字段不属于精度轨迹。日志仍有 `_histc_cuda` deterministic warning，
但 MTP loss、maxvio 和 `grad_norm` 保持完全一致，因此该 warning 在本实验中没有造成可观察的
训练分叉。

这个结果为“确定性的 SparseMLA 路径可以消除 GLM 训练分叉”提供了真实 SFT 佐证。需要注意，
该对照同时将 DSA top-k 切换为 Torch 实现，并把序列长度改为 1024；它不能代替原始 seq16K
上的同实现替换实验。结合固定 indices 微内核已将局部非确定性直接定位到 TileLang `dKV`，
当前结论足以说明逐 step `1e-6` 不适合用于该高性能 backward，但不应表述成原始
seq16K CI 已经获得确定性修复。

## CI 指标与阈值建议

专门为 CI 实现两阶段、固定归约顺序的 `dKV` kernel 会增加开发维护成本，并且因为
partial buffer 和第二阶段 reduction 增加显存占用、kernel launch 和延迟。确定性路径如果只服务于
35-step CI，当前收益不足以覆盖这些代价。因此工程上调整为：保留高性能后端，改用能区分
“少数 step 尖峰”和“整段轨迹回归”的聚合指标。

### 已有数据的统一口径

下表统计每组实验两两比较中最差一对 run 的每步 `grad_norm` 相对误差。由于 CI
使用 `abs(base-current)/abs(base)`，相对误差不对称，这里将每对 run 的正反两个方向都纳入统计。
P80/P95 表示 80%/95% 的 step 不超过该误差；最大单步只是局部极值。

| 配置 | 实验数 | 最差 P80 | 最差 P95 | 最大单步误差 |
|---|---:|---:|---:|---:|
| TileLang + DeepEP（MTP） | 3 runs | 11.32% | 18.96% | 36.09% |
| TileLang + all2all + CUTLASS（MTP） | 2 runs | 3.53% | 7.07% | 8.70% |
| TileLang + all2all + Triton（MTP） | 2 runs | 5.47% | 10.07% | 15.67% |
| cuDNN DSA + all2all（MTP） | 3 runs | 15.97% | 41.82% | 68.36% |
| TileLang + DeepEP（NoMTP） | 3 runs | 1.66% | 2.96% | 4.40% |
| PR #1989 TileLang + NoMTP（200-step A/B） | 1 组 A/B | 0.95% | 2.61% | 6.57% |

PR #1989 的 control/selective 是 selective checkpoint 功能 A/B，不是两次完全相同的重复训练，
所以只能作为曲线级门禁的辅助证据，不单独用于标定随机波动上界。该实验的 loss MAE 为
`0.000417`、`grad_norm` MAE 为 `0.013563`，两条 200-step 曲线的整体趋势一致。

微内核中 cuDNN DSA 的 `dKV` 波动比 TileLang 略小，但完整 SFT 没有保持这个优势：
cuDNN DSA 三次完整训练的最差 P95 和最大单步误差反而更大。这不足以证明 cuDNN DSA
本身更差，因为两组实验的 Torch/CUDA 环境不同；但足以说明不能为了降低 CI 波动而直接切换
到 cuDNN DSA。当前建议继续使用 TileLang，也避免为 CI 额外引入 cuDNN DSA frontend 环境依赖。

### 当前数据能否通过 `P95 < 20%`

`Znote/Zgrad-norm-fix/trackers/run1.jsonl`、`run2.jsonl`、`run3.jsonl` 就是原始 TileLang +
DeepEP + MTP 配置的三份 35-step tracker。按 commit `51c775d` 的 checker 算法对全部六个有向组合
重新计算，结果如下：

| baseline → current | P95 | `P95 < 20%` |
|---|---:|---|
| run1 → run2 | 13.39% | 通过 |
| run1 → run3 | 11.96% | 通过 |
| run2 → run1 | 11.79% | 通过 |
| run2 → run3 | 15.93% | 通过 |
| run3 → run1 | 11.83% | 通过 |
| run3 → run2 | 18.96% | 通过 |

因此，`threshold: 0.20, aggregate: 95` **能通过目前本地三次训练的全部两两比较**。
但最差 P95 已经达到 18.96%，离 20% 只有 1.04 个百分点，作为长期 CI 阈值的余量不足。

### 正式 CI baseline 验证

使用 CI 镜像、原始 TileLang + DeepEP + seq16K 配置和正式 baseline：

```text
/mnt/shared-storage-user/llmrazor-share/qa-llm-cicd/xtuner-baselines/latest/glm5-2-sft-30B/tracker.jsonl
```

设置以下门禁后连续完成两次独立的 35-step 训练：

```yaml
grad_norm:
  threshold: 0.20
  aggregate: 95
```

| 运行 | `grad_norm` P95 | 最大单步误差 | TGS 结果 | 完整 CI 结果 |
|---|---:|---:|---|---|
| `glm52-ci-p95-20260821064041` | 4.47% | 5.53% | P80 约 5.21%，略高于 5% | 仅 TGS 失败 |
| `glm52-ci-p95-20260821064857` | 12.50% | 17.32% | 3.22%，通过 | **全部指标通过** |

两次的 loss、LR、显存和 token 数均通过原有门禁；第一次总结果失败只因为独立的性能指标
`runtime_info/tgs` 比 5% 门限高约 0.21 个百分点，不影响 `grad_norm` 阈值验证。第二次运行的
`CI_METRIC_CHECK_PASSED: True` 证明 `P95 < 20%` 至少能够在正式 baseline、完整 35-step
训练和原 CI 其他指标同时开启的情况下通过。

两次正式比较的 `grad_norm` P95 分别为 4.47% 和 12.50%，均低于 20%；结合本地三次运行
全部有向比较的最差 P95 18.96%，当前已有证据支持采用 20% 门禁。但 18.96% 与 20% 仍然
接近，后续应继续累计同环境重复运行，用于评估长期误报率。

### 推荐的暂定门禁

当目标分支的 SFT checker 支持 percentile aggregation 时，建议保留 TileLang，将原来“任意
单步相对误差小于 `1e-6`”改为“P95 相对误差小于 20%”：

```yaml
check_metrics:
  grad_norm:
    threshold: 0.20
    aggregate: 95
```

选择 20% 而不是此前按安全余量估算的 25%，是为了保持更强的回归敏感度；该值已经通过两次
正式 baseline 的 `grad_norm` 检查，并在第二次运行中通过全部 CI 指标。对 35-step 训练，P95
只容忍最差约 1～2 个 step；如果偏离在多个 step 持续扩大，CI 仍会失败。需要注意，本地
全部有向比较的历史最差 P95 为 18.96%，因此 20% 是经过实测但余量偏小的门限，而不是充分
保守的波动上界；如果后续同环境重复实验仍频繁越界，应根据新增样本重新标定，而不是反复重跑
直到偶然通过。

不建议只把旧的逐 step 标量阈值放宽。如果不改 checker，覆盖已观测最大值至少需要：

- TileLang：约 `0.40`（已观测最大 36.09%）；
- cuDNN DSA：约 `0.75`（已观测最大 68.36%）。

这种门禁会让一个或多个 step 的大幅回归直接通过，诊断价值很低。如果坚持使用 cuDNN DSA，
更可取的聚合门禁是 `P80 < 20%`，因为它的已观测最差 P80 为 15.97%；但 P80 会忽略
最差约 20% 的 step，仍比 TileLang 的 `P95 < 20%` 覆盖更少，不构成切换后端的理由。

`grad_norm` 之外仍需保留以下硬门禁：

- loss 和 `grad_norm` 全程 finite，不得出现 NaN/Inf；
- step 数、LR 和 token 数与 baseline 一致；
- loss 使用单独标定的严格阈值，不随 `grad_norm` 同比例放宽；
- 比较报告保留首个分叉 step、P95 和最大单步误差，但不再让一个原子归约尖峰单独阻断 CI。

当前 20% 已有三次本地重复训练和两次正式 baseline 比较作为依据，但正式 baseline 样本仍然
较少。合入前后应使用最终 CI 配置累计至少 5 次正式比较，统计通过率和 P95 分布；如果 20%
出现非偶发越界，再基于新增分布决定是否调整门限。权重、MTP、序列长度、
dispatcher、Sparse MLA backend 或软硬件环境发生变化时，必须重新标定，不能混用这些实验的分布。

## 实验结论矩阵

| 实验 | 结果 | 结论 |
|---|---|---|
| 原始 GLM-5.2 三次 SFT | `grad_norm` 从 Step 16 明显分叉 | 问题稳定复现 |
| DeepEP → all2all | 仍分叉 | 排除 DeepEP |
| CUTLASS → Triton Grouped GEMM | 仍分叉 | 排除 CUTLASS Grouped GEMM |
| cuDNN DSA backward | 微内核及完整 SFT 仍不确定 | 不能作为直接修复 |
| `torch.backends.cudnn.deterministic=True` | `dKV` 仍有 10 个不同 hash | 只略微减小波动，不提供 DSA backward 确定性 |
| Qwen3-30B-A3B 三次 SFT | 所有训练数值完全一致 | 排除通用 XTuner/FSDP/EP/优化器链路 |
| MTP loss = 0 | 仍分叉 | 排除 MTP loss/梯度 |
| 完全移除 MTP | 仍分叉 | 排除整个 MTP block |
| router bias update speed = 0 | 仍分叉且可能更大 | 排除动态 correction-bias 更新 |
| Torch vs TileLang 固定输入 | 只有 TileLang `dKV` 不一致 | 定位 TileLang backward 原子累加 |
| Torch SparseMLA seq1024 三次 SFT | 全部精度指标完全一致 | 真实训练链路支持上述定位 |
| TileLang `grad_norm` 分位数标定 | 本地最差 P95 18.96%；正式 baseline 两次为 4.47%/12.50% | 候选 CI 使用 `P95 < 20%` |
| `P95 < 20%` 完整 CI 验证 | 第一次仅 TGS 抖动失败；第二次全部指标通过 | 门禁格式与阈值可落地，继续累计样本 |

## 当前根因判断与 CI 落地方向

当前最有证据支持的传播链为：

```text
TileLang Sparse MLA backward dKV 原子累加顺序变化
                         ↓
                Step 1 即产生低位梯度差异
                         ↓
               参数与优化器状态逐步分叉
                         ↓
        DSA/NoAux MoE 的离散 top-k 边界放大差异
                         ↓
             Step 14～17 附近越过 CI 有效阈值
                         ↓
            grad_norm 出现百分比到数十百分比差异
```

根因定位与工程决策需要分开：确定性 `dKV` kernel 能够从根本上消除该波动，但对当前
仅有 35 step 的 CI 场景，它的开发、维护、显存和性能代价偏高。当前不将实现确定性 kernel
作为 CI 落地的前置条件，而是采用以下方案：

1. 保持原始 seq16K 和 TileLang 高性能路径，不因微内核的小幅差异换用显存成本过高的 Torch reference。
2. `grad_norm` 使用已完成正式 baseline 验证的 `P95 < 20%` 相对误差候选门禁，不再要求任意
   单步达到 `1e-6`。
3. loss、finite 状态、step、LR 和 token 数仍保持独立严格检查，避免放宽 `grad_norm` 后掩盖真实回归。
4. 已完成两次正式 baseline 比较；继续使用最终 CI 配置累计至少 5 次，根据通过率和 P95 分布
   复核 20% 阈值。配置或软硬件环境改变时重新标定。
5. 固定输入的 Torch/TileLang/cuDNN DSA 微内核测试继续作为诊断测试，记录 output、`dQ`、`dKV`
   的误差分布，但不对高性能 atomic backward 要求 bitwise equality。

## 当前进度

- [x] 使用 CI 镜像启动 8 张 H200
- [x] 固定失败任务对应 commit
- [x] 安装当前 XTuner 源码及 CI 依赖
- [x] 恢复 CI 使用的 cuDNN 版本
- [x] 通过 8 卡 CUDA smoke test
- [x] 完成 run1、run2、run3
- [x] 对三组 `grad_norm` 做两两比较
- [x] 使用 all2all dispatcher 排除 DeepEP
- [x] 关闭 CUTLASS Grouped GEMM，排除 CUTLASS Grouped GEMM
- [x] 通过单卡重复微内核实验定位到 TileLang Sparse MLA backward 的 `dKV` 路径
- [x] 找到包含 cuDNN DSA frontend 的共享 cu132 环境
- [x] 使用 cuDNN DSA backward 重复微内核实验，确认其 `dKV` 同样不是逐位确定
- [x] 在同一 cu132 环境中量化比较 TileLang 与 cuDNN DSA 的 `dKV` 波动
- [x] 使用 cuDNN DSA backend 重复三次完整 35-step SFT，确认仍会明显分叉
- [x] 使用 Qwen3-30B-A3B 做三次同配置阴性对照，全部训练数值完全一致
- [x] 将 MTP loss 置零后重复三次 SFT，确认仍会分叉
- [x] 完全移除 MTP block 后重复三次 SFT，排除整个 MTP 路径
- [x] 关闭 NoAux router correction-bias 更新，确认其不是根因
- [x] 在原 CI 软件栈中完成 Torch/TileLang 固定输入对照，确认只有 TileLang `dKV` 不可重复
- [x] 使用 Torch SparseMLA 完成三次 seq1024 真实 SFT，全部精度指标完全一致
- [x] 对照 `torch.backends.cudnn.deterministic=False/True`，确认开启后 cuDNN DSA `dKV` 仍不可重复
- [x] 汇总 TileLang、cuDNN DSA、MTP/NoMTP 及 PR #1989 数据，完成 `grad_norm` P80/P95 标定
- [x] 确定暂不为 CI 单独开发确定性 `dKV` kernel，候选门禁为 TileLang `P95 < 20%`
- [x] 完成两次正式 baseline 对新 tracker 的 P95 比较，`grad_norm` 均通过
- [x] 完成一次全部指标通过的 `P95 < 20%` 完整 CI 验证
- [ ] 用最终 CI 配置累计至少 5 次正式比较（当前 2/5），复核并固化 percentile 阈值

## 主要产物

| 产物 | 用途 |
|---|---|
| `run_cudnn_dsa_repeat.sh` | cuDNN DSA backward 固定输入重复测试，支持 `TORCH_CUDNN_DETERMINISTIC=0/1` 对照 |
| `run_cudnn_dsa_sft_3runs.sh` | cuDNN DSA 完整 SFT 三次对照 |
| `run_qwen3_sft_3runs.sh` | Qwen3 阴性对照 |
| `run_glm52_mtp_loss0_sft_3runs.sh` | MTP loss=0 消融 |
| `run_glm52_nomtp5_sft_3runs.sh` | 完全移除 MTP 消融 |
| `run_glm52_nomtp5_routerbias0_sft_3runs.sh` | router correction-bias 更新消融 |
| `run_torch_tilelang_sparse_mla_repeat.sh` | Torch/TileLang SparseMLA 固定输入对照 |
| `run_glm52_torch_sparse_mla_sft_3runs.sh` | Torch SparseMLA seq1024 真实 SFT 对照 |
| `run_glm52_ci_p95_validation.sh` | 将 GLM-5.2 `grad_norm` 门禁设为 `P95 < 20%`，执行完整 SFT 并与正式 baseline 比较 |
