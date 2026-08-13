# GLM-5.2 AutoModel / XTuner FSDP2 Mesh 调研与初步开发方案

> 文档状态：调研完成，方案待评审，功能尚未实现。
>
> 最后复核：2026-08-13。
>
> 源码版本：AutoModel `e2c48886`，XTuner `9dabb956`。
>
> 本文是统一入口，整合了完整 GLM-5.2 实现调研、EP/FSDP mesh 差异、显存实验和 XTuner
> 初步改造方案。文末列出的旧文档保留为实验明细和历史记录。

面向代码评审的精简设计与单文件伪代码见
[`docs/design/model/glm52_decoupled_fsdp.md`](../../docs/design/model/glm52_decoupled_fsdp.md)。

**阅读导航**：第一次阅读建议依次看第 1、3、6、7、8、12、17 节；准备开发时重点看第 9～16 节；
核对实现时查看第 18 节源码索引。

## 1. 调研目的

这次调研要回答四个问题：

1. AutoModel 和 XTuner 是否使用同一类 FSDP？
2. 为什么两边都设置 `EP=4`，XTuner 仍比 AutoModel 多占一部分显存？
3. AutoModel 的 dense/expert 解耦方式能否集成到 XTuner？
4. 如果要集成，哪些能力可以直接兼容，哪些必须改造，收益和代价分别是什么？

核心结论是：

> 两边都使用 PyTorch 官方 FSDP2、DeviceMesh 和 DTensor。主要区别不在 FSDP 实现来源，而在
> **dense 参数和 routed expert 参数是否共用同一个 FSDP/EP 拓扑**。

```text
AutoModel：dense 使用主 FSDP mesh；routed experts 使用独立 EP + expert-FSDP mesh
XTuner：   dense FSDP 与 expert EP 共用 [FSDP, EP] 二维 mesh
```

XTuner 当前实现不是错误实现。它结构简单，dense collective group 较小，在小 EP 或显存足够时可能更快；
但 EP 越大，dense FSDP 维度越小，dense 参数、梯度和 AdamW 状态会沿 EP 复制。完整大 MoE 场景下，
这种复制会逐渐成为显存瓶颈。

建议在 XTuner 中：

- 保留当前实现，命名为 `coupled`，继续作为默认路径；
- 新增 `decoupled` 实验路径；
- routed experts 继续按 `EP x expert-FSDP` 切分；
- attention、router、shared expert、norm、embedding、lm_head 等 dense 参数改为全 rank FSDP；
- HF load/export、DCP、梯度缩放和参数布局必须随 mesh 一起改，不能只改 `fully_shard()`。

## 2. 调研范围和证据边界

### 2.1 三种 GLM-5.2 场景

| 场景 | 模型状态 | 适合验证什么 | 不能证明什么 |
|---|---|---|---|
| 6 层 NoMTP checkpoint | 从完整模型裁剪的约 30B parity/smoke 权重 | 训练链路、loss/grad、显存、时间、checkpoint | 完整模型生成能力和多机容量 |
| 完整 backbone SFT | 78 层、256 routed experts、约 355B 总参数 | 完整权重训练与生产拓扑 | 本轮尚未实际跑完 |
| 完整 frozen target | 完整目标模型冻结，用于 DSpark hidden-state capture | 教师侧前向与特征捕获 | 不等价于完整模型 SFT |

本文的显存数字来自第一种场景；完整模型部分来自源码和 recipe 调研。6 层实验可以证明并行语义和显存方向，
不能替代完整 78 层、多节点 load/forward/backward/checkpoint 验证。

### 2.2 证据等级

为避免把推测写成结论，本文采用以下口径：

| 标记 | 含义 |
|---|---|
| 已确认：源码 | 当前版本源码可以直接证明 |
| 已确认：实验 | 已有日志或 memory snapshot 可以直接证明 |
| 方案 | 建议实现，尚未编码 |
| 待验证 | 原理可行，但必须通过测试后才能承诺 |

### 2.3 XTuner 更新后的复核结果

XTuner 已更新到 `9dabb956`。当前分支包含原生 GLM-5.2 训练支持，包括：

- GLM-5.2 配置、DSA/MLA、IndexShare、NoAux router 和 MTP；
- SP、EP、FSDP2、activation checkpoint 和 intra-layer micro-batch；
- HF round-trip、DCP model/optimizer round-trip；
- DeepEP/all2all/AGRS dispatcher；
- Triton/CUTLASS grouped GEMM；
- GLM-5.2 tiny SFT、EP4+MTP HF checkpoint 和训练数值测试。

本次重新阅读最新版代码后，下面这些关键事实仍成立：

| 检查项 | `9dabb956` 现状 | 对方案的影响 |
|---|---|---|
| FSDP 配置 | `FSDPConfig` 只有 `ep_size`，没有 dense/expert 解耦开关 | 需要新增配置 |
| MoE root mesh | `_init_device_mesh()` 仍创建 `[world_size / EP, EP]` | 当前仍为 coupled |
| Dense 参数 | `_replicate_other_params()` 仍使非 expert 参数沿 EP 复制 | EP 增大时 dense shard 变大 |
| Layer FSDP | decoder layer、embedding、norm、lm_head 仍使用 `self.fsdp_mesh` | 需要拆出 `dense_fsdp_mesh` |
| Gradient | 已支持按 Replicate placement 分组并 coalesced all-reduce | 新路径要去掉不再需要的 dense EP reduce |
| HF layout | `_init_load_spec()` 仍拒绝一个参数出现多个 `Shard` placement | 多 mesh load/export 仍是重点改造项 |
| HF gather | `_fsdp_foreach_allgather()` 仍统一读取 `self.fsdp_mesh` | 需要按参数角色选择 mesh |
| DCP | 使用 PyTorch distributed state-dict API | 有复用基础，但必须做 nested FSDP round-trip |
| HSDP | `hsdp_sharding_size` 与 `EP>1` 仍有硬限制 | 第一版不兼容 HSDP+EP |

已有 memory profile 没有记录可恢复的 XTuner commit，不能直接当成 `9dabb956` 的性能承诺。由于当前核心
mesh 代码仍保持 coupled 语义，历史数据仍能支撑改造方向；但合入新功能前必须在 `9dabb956` 或目标 PR
基线上重跑 EP2/EP4 数值与显存回归。

## 3. 先理解 Mesh、EP 和 FSDP

### 3.1 Mesh 是什么

DeviceMesh 可以理解成“给 GPU 排座位，并给每个方向起名字”。

单机 8 卡、`EP=4` 时，XTuner 当前把 GPU 排成：

```text
               EP0  EP1  EP2  EP3
FSDP row 0      0    1    2    3
FSDP row 1      4    5    6    7
```

这里有两种通信组：

```text
EP groups:          [0,1,2,3] / [4,5,6,7]
FSDP column groups: [0,4] / [1,5] / [2,6] / [3,7]
```

- EP group 决定 token 应该发给哪张卡上的专家；
- FSDP group 决定一份参数被切成几片；
- 同一个二维 mesh 可以派生多个一维 submesh。

### 3.2 Expert group 不是“把四个专家都拦腰切断”

假设只有 4 个专家、`EP=4`：

```text
EP0 负责专家0
EP1 负责专家1
EP2 负责专家2
EP3 负责专家3
```

这是先按“专家编号”分工，每个 EP rank 只拥有一部分专家。随后 expert-FSDP 再切每个本地专家的权重：

```text
rank0 + rank4：共同持有专家0的不同参数分片
rank1 + rank5：共同持有专家1的不同参数分片
rank2 + rank6：共同持有专家2的不同参数分片
rank3 + rank7：共同持有专家3的不同参数分片
```

所以是两层含义：

1. EP 把不同专家分给不同 GPU 位置；
2. expert-FSDP 再把某个位置上的专家权重切成多片。

### 3.3 Dense 参数指什么

本文的 dense 参数不是指“dense model”，而是指每个 token 都会经过、并且不按专家编号分配的参数，包括：

- attention；
- router/gate；
- shared expert；
- RMSNorm；
- embedding；
- lm_head。

routed expert 的 `fused_w1w3/fused_w2` 才属于 expert 参数。

## 4. AutoModel 当前怎么切

### 4.1 主链路

AutoModel 完整 GLM-5.2 训练链路可以概括为：

```text
HF checkpoint
  -> HF config与head_dim修正
  -> GLM DSA/IndexShare/MoE模型
  -> 构建主device_mesh和独立moe_mesh
  -> 沿EP分配routed experts
  -> experts沿EP_SHARD做内层FSDP2
  -> decoder block沿主FSDP mesh做外层FSDP2，并排除experts
  -> 分片后加载HF权重
  -> optimizer/train
  -> DCP或consolidated HF checkpoint
```

关键源码：

| 环节 | 文件和符号 |
|---|---|
| 主 mesh 和 MoE mesh | `nemo_automodel/components/distributed/mesh_utils.py::_create_fsdp2_device_mesh()` |
| MoE mesh 构造 | `mesh_utils.py::_create_moe_mesh()` |
| 应用 EP | `nemo_automodel/components/moe/parallelizer.py::apply_ep()` |
| Expert FSDP | `parallelizer.py::apply_fsdp()` 中对 `moe_module.experts` 的 `fully_shard()` |
| Dense FSDP | 同一函数对 decoder block 的外层 `fully_shard()` |
| 模型基础设施 | `nemo_automodel/_transformers/infrastructure.py` |
| Checkpoint | `nemo_automodel/components/checkpoint/checkpointing.py` |

### 4.2 完整 GLM-5.2 能力概览

AutoModel 中与完整模型训练有关的能力不只有 FSDP mesh：

| 能力 | AutoModel 实现 | 对 XTuner 的意义 |
|---|---|---|
| 模型注册和构造 | HF config 经 registry 进入自定义 `GlmMoeDsaForCausalLM` | XTuner 已有原生 GLM-5.2，不需迁移模型类 |
| 78 层 backbone | 前 3 层 dense，后续按配置使用 256 routed experts | 两边都已具备 |
| DSA/MLA | Q/KV LoRA、sparse top-k、TileLang 和 fallback | XTuner 已有 torch/TileLang/cuDNN DSA |
| IndexShare | shared 层复用前一个 full 层的 top-k | 两边都有；shared 层没有独立 indexer 权重是正常设计 |
| PP | PP stage 间同时传 `hidden_states` 和 `topk_indices` | XTuner 当前没有等价 GLM PP 链路 |
| CP/长序列 | 32K recipe 使用 CP8、THD packing 和 TileLang | XTuner 使用 SP，作用相近但实现不同 |
| Dispatcher | HybridEP，框架也支持 DeepEP 等实现 | XTuner 已有 DeepEP/all2all/AGRS |
| Expert GEMM | 默认 `torch._grouped_mm`，另有其它 backend | XTuner 保留已验证的 CUTLASS，不需照搬 |
| HF 权重适配 | split experts 与 grouped experts 双向转换 | XTuner 已有自己的 fused expert 转换 |
| 分片加载/checkpoint | meta 构建、先分片再加载；Checkpointer 接收 `moe_mesh` | 多 mesh checkpoint 是最值得借鉴的部分 |

AutoModel 当前实现也有明确边界：

- 自定义 GLM 目录当前没有 MTP auxiliary layer；XTuner 已有 MTP，不应删除或替换；
- custom MoE 路径要求 `TP=1`；
- 完整 355B recipe 和源码链路存在，但本轮没有完成 32 节点长训练验证；
- `PP4+EP64`、`PP4+EP64+CP8` 是整体方案，不能把其中一项单独等同于“完整模型已经对齐”。

### 4.3 AutoModel 的 mesh 公式

AutoModel 主 mesh 的 data-parallel 大小按下面公式推导：

```text
dp_size = world_size / (TP x CP x PP)
```

注意：这里没有把 EP 从 `dp_size` 中除掉。随后再从非 PP ranks 派生 MoE mesh：

```text
non_pp_size = DP x CP x TP
expert_fsdp_size = non_pp_size / EP
moe_mesh = [expert_fsdp_size, EP]
```

因此 dense 和 expert 拥有两种视图：

```text
dense:  主FSDP mesh
expert: [expert-FSDP, EP] mesh
```

### 4.4 Nested FSDP2

AutoModel 对每个 MoE block 采用 bottom-up 包裹：

```python
# 内层：只管理routed expert参数
fully_shard(
    moe_module.experts,
    mesh=expert_fsdp_mesh,
)

# 外层：管理attention、router、shared expert、norm等dense参数
fully_shard(
    decoder_block,
    mesh=dense_fsdp_mesh,
    ignored_params=set(moe_module.experts.parameters()),
)
```

这就是本文所说的 nested FSDP2：同一个 decoder block 里，routed experts 是内层 FSDP unit，block
其余参数是外层 FSDP unit，两者使用不同 submesh。

### 4.5 8 卡 EP4 的实际分组

```text
AutoModel dense FSDP group:
  [0,1,2,3,4,5,6,7]       -> dense切8份

AutoModel EP groups:
  [0,1,2,3]
  [4,5,6,7]

AutoModel expert-FSDP groups:
  [0,4] [1,5] [2,6] [3,7] -> 每个EP位置的expert再切2份
```

这里不是创建两套 GPU，而是同一批 GPU 同时拥有两种 mesh 视图。

### 4.6 完整模型 recipe 的意义

AutoModel 完整 4K SFT recipe 使用：

```text
world_size = 256
PP = 4
EP = 64
每个PP stage的非PP ranks = 64
dense FSDP = 64
moe_mesh = [expert-FSDP=1, EP=64]
```

32K recipe 在此基础上增加 `CP=8`。这说明完整模型能力不只是独立 mesh，还包括 PP、CP、packed sequence、
checkpoint 和多机通信。XTuner 只集成 decoupled mesh，并不等于已经具备 AutoModel 的完整 `PP4+EP64`
生产拓扑。

## 5. XTuner 当前怎么切

### 5.1 当前链路

最新版 XTuner 的 MoE FSDP 链路仍是：

```text
HF checkpoint
  -> meta device构建模型
  -> model_mesh = [world_size / EP, EP]
  -> expert权重沿EP Shard(0)
  -> 非expert参数沿EP Replicate()
  -> 完整decoder layer沿FSDP维fully_shard()
  -> HF权重加载
  -> optimizer/train
  -> DCP或HF checkpoint
```

关键源码：

| 行为 | 文件和符号 |
|---|---|
| 配置 | `xtuner/v1/config/fsdp.py::FSDPConfig` |
| 创建二维 mesh | `xtuner/v1/model/moe/moe.py::_init_device_mesh()` |
| 沿 EP 复制 dense | `moe.py::_replicate_other_params()` |
| 模型 FSDP | `moe.py::fully_shard()` |
| Expert 初始 EP shard | `xtuner/v1/module/grouped_linear/moe_group_linear.py::GroupedLinear` |
| 梯度缩放/归约 | `moe.py::scale_and_reduce_grad()` |
| HF load/export | `xtuner/v1/model/base.py` |
| DCP | `xtuner/v1/engine/train_engine.py` |

### 5.2 当前 mesh 公式

XTuner 直接计算：

```text
expert_fsdp_size = world_size / EP
model_mesh = [expert_fsdp_size, EP]
```

然后：

```text
self.fsdp_mesh = model_mesh[FSDP轴]
self.ep_mesh   = model_mesh[EP轴]
```

decoder layer、embedding、norm 和 lm_head 都使用 `self.fsdp_mesh`。非 expert 参数先沿 EP 标成
`Replicate()`，所以 dense 只沿 FSDP 轴切分，并在 EP 轴保留副本。

### 5.3 8 卡 EP4 的实际分组

```text
XTuner root mesh = [FSDP=2, EP=4]

Dense FSDP groups:
  [0,4] [1,5] [2,6] [3,7]

Dense placement:
  在FSDP轴Shard，在EP轴Replicate
```

用一份 dense 参数举例：

```text
rank0：dense第一半
rank1：dense第一半，和rank0是EP副本
rank2：dense第一半，和rank0是EP副本
rank3：dense第一半，和rank0是EP副本

rank4：dense第二半
rank5：dense第二半，和rank4是EP副本
rank6：dense第二半，和rank4是EP副本
rank7：dense第二半，和rank4是EP副本
```

8 张卡实际只把 dense 切成 2 份，每份复制 4 次。训练时 replicated dense gradient 需要额外在 EP
副本间做归约；最新版 `scale_and_reduce_grad()` 已把这些梯度按 process group 合并后 all-reduce。

## 6. 两种方案的相同点和差异

| 项目 | AutoModel | XTuner 当前 | 是否相同 |
|---|---|---|---|
| FSDP 来源 | PyTorch FSDP2 | PyTorch FSDP2 | 相同 |
| 分布式张量 | DeviceMesh + DTensor | DeviceMesh + DTensor | 相同 |
| Expert Parallel | 按 EP 分专家 | 按 EP 分专家 | 语义相同 |
| Expert 二次切分 | expert-FSDP | 当前 FSDP 轴 | 基本同类 |
| Dense FSDP | 主 mesh，不因 EP 缩小 | `world/EP` | 不同 |
| Dense 沿 EP | 不复制 | `Replicate()` | 不同 |
| FSDP 包裹 | expert 内层 + block 外层 | 整个 layer 使用同一 FSDP 轴 | 不同 |
| HF checkpoint | 基础设施知道主 mesh/MoE mesh | 多处默认单一 `self.fsdp_mesh` | 不同 |
| 完整模型 PP | recipe 使用 PP4 | 当前 GLM 路径没有等价 PP 方案 | 不同 |

更准确的表述不是“AutoModel 用官方 FSDP，XTuner 没用”，而是：

> 两边都基于官方 FSDP2。AutoModel 使用主 mesh 与 MoE mesh 的 parallel folding，并用 nested FSDP2
> 分别管理 dense/expert；XTuner 当前把 FSDP 与 EP 耦合在一个二维 mesh 中。

## 7. 实验为什么支持这个判断

### 7.1 EP2：先排除 dispatcher 和 expert GEMM

已有 6 层、8xH200、EP2、no-swap profile 得到：

| 实验 | snapshot峰值 | step结束active |
|---|---:|---:|
| XTuner all2all + Triton | `131.754 GiB` | `78.446 GiB` |
| XTuner DeepEP + Triton | `126.688 GiB` | `73.380 GiB` |
| XTuner DeepEP + CUTLASS | `104.209 GiB` | `50.901 GiB` |
| AutoModel HybridEP | `95.322 GiB` | `50.166 GiB` |
| AutoModel DeepEP | `95.278 GiB` | `50.180 GiB` |

控制变量结论：

- XTuner `all2all -> DeepEP`，active 减少 `5.066 GiB`；
- XTuner `DeepEP + Triton -> CUTLASS`，active 再减少 `22.479 GiB`；
- AutoModel `HybridEP -> DeepEP`，active 仅变化 `0.014 GiB`；
- XTuner CUTLASS 和 AutoModel DeepEP 的 step-end active 只差 `0.721 GiB`。

因此 dispatcher 优化有效，但不是全部原因；XTuner 默认 Triton expert GEMM 是 EP2 显存偏高的主要来源之一。

### 7.2 EP4：kernel 优化后暴露 mesh 差异

相同口径下的 EP4 profile：

| 实验 | snapshot峰值 | step结束active |
|---|---:|---:|
| XTuner all2all + Triton | `125.666 GiB` | `85.088 GiB` |
| XTuner DeepEP + Triton | `119.961 GiB` | `80.377 GiB` |
| XTuner DeepEP + CUTLASS | `98.848 GiB` | `61.239 GiB` |
| AutoModel HybridEP | `82.823 GiB` | `50.166 GiB` |
| AutoModel DeepEP | `82.779 GiB` | `50.180 GiB` |

XTuner 使用 DeepEP+Cutlass 后，仍比 AutoModel DeepEP 多 `11.059 GiB` step-end active。snapshot 中
存在与 dense shard degree 精确一致的 4 倍常驻块：

```text
GLM-5.2 embedding/lm_head完整FP32矩阵：3.544922 GiB

XTuner dense FSDP=2：3.544922 / 2 = 1.772461 GiB
AutoModel dense FSDP=8：3.544922 / 8 = 0.443115 GiB
```

embedding 和 lm_head 未绑定，各自有参数、AdamW 一阶动量 `m`、二阶动量 `v`：

```text
XTuner：   6 x 1.772461 = 10.635 GiB
AutoModel：6 x 0.443115 =  2.659 GiB
差异：                       7.976 GiB
```

仅这一组就解释了 `11.059 GiB` 常驻差异的 `72.1%`。其它 dense 块也出现相同 4 倍尺寸规律。

### 7.3 为什么 EP4 比 EP2 更明显

| EP | XTuner dense FSDP | AutoModel dense FSDP | XTuner单卡dense shard相对大小 |
|---:|---:|---:|---:|
| 2 | 4 | 8 | 约 2 倍 |
| 4 | 2 | 8 | 约 4 倍 |

AutoModel 的 dense FSDP 不随 EP 改变；XTuner 的 dense FSDP 为 `world/EP`。因此 EP 越大，两边 dense
参数和 optimizer state 的单卡差距越明显。

### 7.4 不能过度解读

这些实验不能证明：

- AutoModel 的 DeepEP 天然比 XTuner DeepEP 省全部差值；两边 expert GEMM 仍不同；
- reserved memory 都是活跃 tensor；reserved 包含 allocator 缓存；
- 4-step profile 可以代表吞吐；速度必须看关闭 profile 的长跑；
- 6 层结果可以代替完整 78 层验证；
- 旧 profile 数字在 XTuner `9dabb956` 上会逐字节复现。

## 8. 完整 GLM-5.2 应偏向哪种方案

### 8.1 当前 coupled 的优点

- 结构简单，mesh 和 checkpoint 假设统一；
- dense FSDP group 较小，参数 all-gather/reduce-scatter 通信域较小；
- 当前 HF、DCP、FP8、MTP、compile 和 RL 路径都围绕该布局建设；
- 已有测试和生产脚本覆盖更充分。

### 8.2 当前 coupled 的缺点

- EP 增大时 dense FSDP 度缩小；
- dense 参数、梯度和 optimizer state 沿 EP 复制；
- 完整大 MoE 通常需要较大 EP，显存问题会被放大；
- 为避免 OOM 使用 swap/offload 会引入额外 CPU 通信，并掩盖参数布局问题。

### 8.3 Decoupled 的优点

- dense 参数和 AdamW state 能使用更大的 FSDP shard degree；
- EP 只负责专家归属，不再决定 dense 每卡保存多少；
- 更适合专家数多、EP 大、dense/optimizer state 已成为瓶颈的完整 MoE；
- dense 和 expert 可以分别调整 reshard/prefetch，后续调优空间更大。

### 8.4 Decoupled 的缺点

- dense collective group 从 `world/EP` 扩大到主 FSDP group，通信可能变慢；
- nested FSDP 和多 mesh 增加参数生命周期、prefetch 和 process group 复杂度；
- HF load/export、FP8、MTP、Muon、RL weight update 等单 mesh 假设需要逐项改造；
- checkpoint 能保存不等于能跨 topology 恢复；
- XTuner 尚无 AutoModel 完整 recipe 使用的 PP4，对 355B 全参数训练仍可能缺少容量维度。

### 8.5 场景建议

| 场景 | 建议 |
|---|---|
| EP 小、显存足够、优先稳定或通信效率 | 继续使用 `coupled` |
| EP 大、dense/optimizer 常驻显存成为瓶颈 | 尝试 `decoupled` |
| 完整 355B 多机训练 | 优先调研 `decoupled`，同时评估 PP；只改 mesh 不一定够 |
| 未验证的 MTP/FP8/compile/RL 组合 | 先使用 `coupled` |

对完整 GLM-5.2，方向上更偏向 decoupled，但最终选择必须看两个数据：是否解决容量问题，以及 dense
collective 扩大后吞吐回退多少。

## 9. XTuner 初步目标设计

### 9.1 设计原则

1. 默认行为不变，避免影响现有模型和任务；
2. mesh 是显式对象，不让 `self.fsdp_mesh` 同时代表多个含义；
3. 参数必须有唯一 owner，外层 FSDP 不重复管理 expert；
4. checkpoint 与训练布局同时设计；
5. 第一版聚焦 GLM-5.2，不提前抽象所有 MoE 模型；
6. 不满足兼容条件时直接报错，不静默退回 coupled。

### 9.2 配置开关

建议在 `FSDPConfig` 增加：

```python
from typing import Literal

moe_fsdp_mesh: Literal["coupled", "decoupled"] = "coupled"
```

平台 A/B 可以临时提供：

```bash
export XTUNER_MOE_FSDP_MESH=decoupled
```

正式配置以 `FSDPConfig` 为准，环境变量只用于实验覆盖。启动时必须打印 requested/effective mode、
world size、EP size、dense FSDP size 和 expert-FSDP size。

### 9.3 统一 Mesh 对象

建议新增：

```python
@dataclass(frozen=True)
class MoEParallelMeshes:
    mode: Literal["coupled", "decoupled"]
    root_mesh: DeviceMesh
    dense_fsdp_mesh: DeviceMesh
    expert_fsdp_mesh: DeviceMesh
    ep_mesh: DeviceMesh
```

共享 root mesh：

```text
root_mesh = [EXPERT_FSDP = world_size / EP, EP]
```

`decoupled` 从同一个 root 派生：

```text
ep_mesh          = root_mesh[EP]
expert_fsdp_mesh = root_mesh[EXPERT_FSDP]
dense_fsdp_mesh  = flatten(root_mesh)  # 覆盖全部rank
```

`coupled` 也返回同一个对象，但：

```text
dense_fsdp_mesh = expert_fsdp_mesh
```

所有 submesh 必须来自同一个 root，不能分别调用多次 `init_device_mesh()`。PyTorch 的 `_flatten()`/`_unflatten()`
在不同版本存在接口差异，兼容代码应集中放在一个 mesh helper 中。

### 9.4 参数归属

| 参数类型 | Owner | 使用的 mesh |
|---|---|---|
| Routed experts | expert 内层 FSDP unit | `EP + expert_fsdp_mesh` |
| Attention | decoder block 外层 FSDP | `dense_fsdp_mesh` |
| Router/gate | decoder block 外层 FSDP | `dense_fsdp_mesh` |
| Shared expert | decoder block 外层 FSDP | `dense_fsdp_mesh` |
| Norm | decoder block 外层 FSDP | `dense_fsdp_mesh` |
| Embedding | 独立 dense FSDP unit | `dense_fsdp_mesh` |
| Final norm | 独立 dense FSDP unit | `dense_fsdp_mesh` |
| LM head | 独立 dense FSDP unit | `dense_fsdp_mesh` |
| MTP routed experts | 第一版暂不支持 | 后续单独设计 |

参数角色应由模块边界确定：`MoEDecoderLayer.experts` 是 routed expert root。不能只依赖
`".experts" in name` 判断 checkpoint owner。

### 9.5 Nested FSDP2 顺序

建议按 bottom-up 顺序：

```python
expert_params = set(layer.experts.parameters())

fully_shard(
    layer.experts,
    mesh=meshes.expert_fsdp_mesh,
    mp_policy=mp_policy,
    reshard_after_forward=expert_reshard_after_forward,
)

fully_shard(
    layer,
    mesh=meshes.dense_fsdp_mesh,
    mp_policy=mp_policy,
    reshard_after_forward=dense_reshard_after_forward,
    ignored_params=expert_params,
)
```

随后 embedding、norm、lm_head 和 model root 使用 `dense_fsdp_mesh`。`decoupled` 不再调用
`_replicate_other_params()`；`coupled` 完全保留当前顺序。

### 9.6 `BaseModel._fully_shard()`

当前 `_fully_shard()` 只会收集 FP32 pattern 对应的 ignored params。建议允许模型额外传入：

```python
def _fully_shard(
    ...,
    extra_ignored_params: set[nn.Parameter] | None = None,
) -> None:
    ignored_params = collect_fp32_ignored_params(...)
    ignored_params.update(extra_ignored_params or ())
    fully_shard(..., ignored_params=ignored_params or None)
```

`BaseModel` 不应硬编码 GLM 类型；MoE 模型负责识别并传入 routed expert params。

## 10. 运行时和梯度语义

### 10.1 Forward

```text
Dense layer:
  沿dense_fsdp_mesh all-gather完整参数
  -> 前向计算
  -> 按配置reshard

Routed experts:
  token沿ep_mesh dispatch
  -> 本地expert沿expert_fsdp_mesh all-gather
  -> grouped GEMM
  -> reshard
  -> token沿ep_mesh combine
```

DeepEP/all2all 只依赖 `ep_mesh`；CUTLASS/Triton 只消费当前 rank 的 local expert tensor。因此这两类优化
和 dense/expert mesh 解耦在概念上正交，不需要重写 kernel。

### 10.2 Backward

当前 coupled 布局中：

- routed expert gradient 需要按 EP loss-average 语义除以 `ep_size`；
- dense 参数沿 EP 复制，需要手工在 replicated group 做 all-reduce。

decoupled 后：

- routed expert 的 EP scaling 语义保持；
- dense 参数不再沿 EP Replicate，不应再做该手工 all-reduce；
- dense gradient 由 FSDP2 在 `dense_fsdp_mesh` 上 reduce-scatter。

这里不能凭“loss 看起来差不多”验收，必须比较固定参数切片的单步更新量，防止出现 `EP` 倍或 `1/EP`
倍的缩放错误。

## 11. Checkpoint 设计

Checkpoint 是本方案风险最高的部分。

### 11.1 当前单 mesh 假设

XTuner 当前存在以下假设：

1. `BaseModel` 主要通过一个 `self.fsdp_mesh` 处理 load/export；
2. `_init_load_spec()` 遇到多个 `Shard` placement 会报错；
3. `_fsdp_foreach_allgather()` 统一使用 `self.fsdp_mesh`；
4. fused experts 需要组合 EP shard 和 FSDP shard 后再转回 HF keys；
5. FP8 padding、Muon、RL weight update 等代码也读取单一 mesh 语义。

因此只修改 `MoE.fully_shard()` 会得到“可能能训练，但不能可靠加载和保存”的半成品。

### 11.2 参数布局描述

建议把两个概念拆开：

```text
LoadSpec：XTuner参数如何映射到HF key，是否fused/split/transpose
RuntimeShardSpec：参数当前属于哪个角色、在哪个mesh、有哪些placements
```

示意结构：

```python
class ParamRole(StrEnum):
    DENSE = "dense"
    ROUTED_EXPERT = "routed_expert"
    REPLICATED = "replicated"

@dataclass(frozen=True)
class RuntimeShardSpec:
    role: ParamRole
    fsdp_mesh: DeviceMesh | None
    ep_mesh: DeviceMesh | None
    placements: tuple[Placement, ...]
```

第一版若不引入 dataclass，至少提供统一 helper：

```python
get_param_fsdp_mesh(param)
get_param_ep_mesh(param)
get_param_shard_placements(param)
```

新的 checkpoint 代码不能继续假设所有参数都使用 `self.fsdp_mesh`。

### 11.3 HF 加载

```text
Dense：
  HF tensor
  -> 根据dense_fsdp_mesh计算当前rank offset
  -> 写入local shard

Routed expert：
  HF split expert keys
  -> 选择当前EP rank负责的experts
  -> 转为fused_w1w3/fused_w2布局
  -> 根据expert_fsdp_mesh计算当前rank offset
  -> 写入local shard
```

必须验证 256 experts 不重不漏、fused reshape/transpose 不变、`head_dim/qk_rope_head_dim` 修正不受影响，
并且 IndexShare 的 shared 层不会被误报为缺少 indexer。

### 11.4 HF 导出

不同参数不能放进同一个 gather bucket：

```text
Dense bucket：
  沿dense_fsdp_mesh gather

Expert bucket：
  沿expert_fsdp_mesh gather本地experts
  -> 沿ep_mesh收集全部expert组
  -> 转回HF split expert keys
```

验收必须重新加载导出目录并比较 config、attention、router、shared expert、首/中/末 routed expert、
embedding、lm_head，以及随机抽样权重 checksum。

### 11.5 DCP

XTuner 已使用 PyTorch `get_model_state_dict()`、`get_optimizer_state_dict()` 和 DCP API，理论上可以识别
FSDP2/DTensor。但“API 支持”不代表当前 nested topology 已通过验证。

第一版至少验证：

```text
decoupled EP2 save -> decoupled EP2 resume
decoupled EP4 save -> decoupled EP4 resume
model + AdamW state
resume后下一step的loss/grad/参数更新连续
```

第一版不承诺 `coupled EP2 -> decoupled EP4` 之类的跨拓扑 DCP 恢复。跨拓扑模型迁移优先通过
consolidated HF 权重完成，不复用 optimizer state。

Checkpoint metadata 至少记录：

```text
mesh_mode, world_size, ep_size, dense_fsdp_size, expert_fsdp_size,
model_variant, MTP, precision, XTuner_commit
```

## 12. 兼容性矩阵

### 12.1 第一版支持范围

| 能力 | 状态 | 说明 |
|---|---|---|
| GLM-5.2 6 层 NoMTP | 支持 | 开发和 parity 基线 |
| 完整 78 层 NoMTP | 目标支持 | 完成多机 2-step 后才算支持 |
| TP=1、PP=1、HSDP关闭 | 支持 | 第一版硬约束 |
| EP2/EP4 | 支持 | 单机主验证 |
| BF16 compute + FP32 master | 支持 | 当前数值基线 |
| AdamW | 支持 | 第一版唯一优化器 |
| DeepEP + CUTLASS | 支持 | 当前候选性能组合 |
| all2all + Triton | 回归支持 | 用于保证旧路径未破坏 |
| Reshard after forward | 支持 | dense/expert分别验证 |
| 同步 HF load/export | 必须支持 | 不支持则不能交付 |
| 同步 DCP model/optimizer | 必须支持 | 不支持则不能长训练 |

### 12.2 原理兼容，但必须单独回归

| 能力 | 判断 | 主要验证点 |
|---|---|---|
| Activation checkpoint | 可兼容 | nested FSDP unshard/reshard 生命周期 |
| SP2 | 可兼容 | data mesh 与参数 mesh 的 rank 组合、collective 顺序 |
| Packing/GBS | 可兼容 | 不改 loss mask，只改参数布局 |
| TileLang/cuDNN DSA | 可兼容 | dense 参数 unshard 后的输入/layout |
| Dispatcher 切换 | 可兼容 | 只应依赖 `ep_mesh` |
| Expert GEMM 切换 | 可兼容 | 只应读取 local expert tensor |
| CPU offload | 原理兼容 | PyTorch 版本、ignored params 和 checkpoint hooks |
| 多机 | 必须最终支持 | dense group 扩大后的 NCCL 通信成本 |

### 12.3 第一版不兼容或主动关闭

| 能力 | 原因 |
|---|---|
| MTP | shared weights、reentrant checkpoint 和 nested FSDP 生命周期需单独设计 |
| FP8 | Float8 handler/padding 依赖单一 `fsdp_mesh` |
| HSDP+EP | 当前 `FSDPConfig` 明确要求 HSDP 时 `EP=1` |
| TP>1 | XTuner GLM 和 AutoModel custom MoE 均没有本方案可直接复用的稳定 TP 路径 |
| PP>1 | XTuner 当前没有与 AutoModel GLM PP/IndexShare carry 等价的链路 |
| Muon | optimizer 多 placement 通信需适配 |
| model/torch compile | 先验证 eager nested FSDP，再处理 graph break |
| Async HF checkpoint | 先保证同步 round-trip 正确 |
| RL weight update | 当前在线权重路径依赖单一 mesh 语义 |
| optimizer swap | 会掩盖纯 mesh 的显存收益，首轮关闭 |
| 跨 mesh DCP resume | 第一版只支持相同 mode、world size 和 EP 恢复 |

这里的“不兼容”是第一版工程边界，不代表长期无法支持。

## 13. 预计改动范围

| 模块 | 预计改动 | 风险 |
|---|---|---|
| `xtuner/v1/config/fsdp.py` | 增加 mode、校验和序列化 | 低 |
| 新增 MoE mesh helper | 构建 root/dense/expert/EP submesh | 中 |
| `xtuner/v1/model/moe/moe.py` | 分流 coupled/decoupled、nested FSDP、梯度语义 | 高 |
| `xtuner/v1/model/base.py` | 扩展 ignored params、runtime layout、HF load/export | 很高 |
| `xtuner/v1/engine/train_engine.py` | DCP metadata 与恢复校验 | 中 |
| FP8/MTP/Muon/RL | 后续兼容 | 高，不放第一版 |
| Tests | mesh、数值、HF、DCP、显存、完整模型 | 高但不可省略 |

这不是只改十几行 mesh shape 的小功能。核心训练原型属于中等改造；真正的工作量主要在 HF/DCP
checkpoint、梯度语义和兼容测试。建议拆成多个可回滚 PR，不做一个大 PR。

## 14. 开发拆分

### PR 1：配置和 MeshContext

- 增加 `moe_fsdp_mesh`；
- 新增 `MoEParallelMeshes` 和 builder；
- 打印 topology；
- 增加 EP2/EP4 rank membership 测试；
- 只构建新 mesh，不改变参数布局。

### PR 2：Nested FSDP 和梯度

- routed experts 使用内层 expert-FSDP；
- decoder block 使用外层 dense FSDP并排除 experts；
- decoupled 停止 `_replicate_other_params()`；
- 调整 `scale_and_reduce_grad()`；
- 完成 tiny 2-step/20-step 参数更新对比。

### PR 3：HF Load/Export

- 分离 HF key mapping 与 runtime shard layout；
- 按参数实际 mesh load/gather；
- 完成 EP2/EP4 strict HF round-trip；
- 保证 coupled 现有 checkpoint 不回归。

### PR 4：DCP 和 200-step

- DCP model/optimizer round-trip；
- checkpoint metadata；
- 6 层 EP2/EP4 200-step loss/grad、显存和时间；
- 在目标 XTuner commit 重跑历史 profile。

### PR 5：完整模型

- 完整 78 层 NoMTP meta/frozen load；
- 完整 HF 权重 load；
- forward、backward、AdamW 2-step；
- DCP save/resume；
- 再决定是否建设 PP 和 EP64 生产拓扑。

### 后续 PR

```text
SP2 -> MTP -> compile -> FP8 -> offload/swap -> async checkpoint -> RL
```

每个组合单独打开和回归，不在一个实验中同时改变多个变量。

## 15. 测试和验收

### 15.1 Mesh 与参数归属

| 测试 | 验证内容 |
|---|---|
| 8卡EP2/EP4 mesh | shape、name、rank membership、root identity |
| 参数 owner | dense/expert/router/shared expert 归属唯一 |
| DTensor placement | local shape 与 Shard/Replicate 符合设计 |
| Coupled 回归 | 默认 mesh、参数布局和日志保持不变 |

### 15.2 数值

同一 seed、checkpoint、batch 和 optimizer，比较：

- 每步 loss；
- clip 前 grad_norm；
- 固定 dense/expert 参数切片的 gradient；
- optimizer.step 后的参数 delta；
- AdamW step、`m`、`v`。

只看 loss 曲线不够，因为梯度缩放错误可能短时间内看不出来。

### 15.3 Checkpoint

| 测试 | 通过标准 |
|---|---|
| HF load | 所有 key 加载完成，关键参数与源权重一致 |
| HF export/reload | config 和抽样权重一致，重新加载可运行 |
| DCP model resume | 下一步 loss/grad 连续 |
| DCP optimizer resume | AdamW step/m/v 连续 |
| EP2/EP4 round-trip | 两种拓扑分别通过 |

### 15.4 性能

至少记录：

- rank0 和全 rank 最大 `max_memory_allocated`；
- memory snapshot timeline peak、end active、reserved；
- warmup 后 steady-state step time；
- tokens/s；
- dense/expert FSDP group size 和通信时间。

性能验收不要求 decoupled 一定更快，但必须说明“节省多少显存、慢多少或快多少”。

### 15.5 第一版完成标准

1. 默认 coupled 行为不变；
2. decoupled EP2/EP4 200 step 无 OOM、NaN 或异常退出；
3. loss/grad/参数更新无系统性偏差；
4. EP4 dense/optimizer 常驻显存明显下降；
5. 同步 HF 和 DCP round-trip 通过；
6. 完整 78 层 NoMTP 至少完成 load + 2-step + DCP resume；
7. 通信和 step time 代价有实测数据。

## 16. 风险和待决问题

| 问题 | 当前判断 | 如何决定 |
|---|---|---|
| Dense group 扩大是否过慢 | 最大性能风险 | EP4/多机 NCCL profile |
| LoadSpec 如何表达多 mesh | 大概率需要扩展 | PR3 原型后确定 dataclass/helper |
| DCP 是否零改动支持 nested FSDP | 理论可行，未验证 | model+optimizer round-trip |
| 完整模型没有 PP 能否训练 | 未知 | 完整 load/显存估算后决定 |
| SP2 是否可直接组合 | 理论可行 | 核心链路稳定后单独回归 |
| MTP shared weights 如何包裹 | 高风险 | 单独设计 FSDP unit 和 checkpoint |
| 多机 rank 排列是否合理 | 需要实测 | EP 尽量留在高速域，dense 跨域 profile |
| 旧 profile 是否适用于新分支 | 不能直接承诺 | 在 `9dabb956`/目标 PR 重跑 |

## 17. 给评审者的最终建议

建议批准一个分阶段原型，而不是直接承诺替换 XTuner 当前实现：

1. 当前 coupled 路径保留且默认不变；
2. 先做 GLM-5.2 NoMTP 的 decoupled mesh、nested FSDP 和同步 checkpoint；
3. 用 6 层 EP2/EP4 证明数值正确和显存收益；
4. 再用完整 78 层验证容量和 DCP；
5. 只有当显存收益能覆盖通信代价时，才继续接 MTP、SP、compile、FP8 和生产拓扑。

这条路线能把风险控制在模型级实验功能内，同时保留现有稳定路径。即使最终因为通信成本不适合作为默认方案，
该原型也能给出一个明确结论：XTuner 的显存瓶颈中，哪些来自 kernel，哪些来自 mesh，分别能优化到什么程度。

## 18. 源码索引

### AutoModel

- `nemo_automodel/components/distributed/mesh_utils.py`
- `nemo_automodel/components/distributed/mesh.py`
- `nemo_automodel/_transformers/infrastructure.py`
- `nemo_automodel/components/moe/parallelizer.py`
- `nemo_automodel/components/checkpoint/checkpointing.py`
- `nemo_automodel/components/models/glm_moe_dsa/model.py`
- `nemo_automodel/components/models/glm_moe_dsa/layers.py`
- `nemo_automodel/components/models/glm_moe_dsa/state_dict_adapter.py`
- `examples/llm_finetune/glm/glm_5.2_tulu3_4k_tilelang_100k.yaml`
- `examples/llm_finetune/glm/glm_5.2_tulu3_32k_tilelang_cp8.yaml`

### XTuner `9dabb956`

- `xtuner/v1/config/fsdp.py`
- `xtuner/v1/model/moe/moe.py`
- `xtuner/v1/model/moe/glm52.py`
- `xtuner/v1/model/base.py`
- `xtuner/v1/module/decoder_layer/moe_decoder_layer.py`
- `xtuner/v1/module/grouped_linear/moe_group_linear.py`
- `xtuner/v1/engine/train_engine.py`
- `tests/model/test_glm52_moe.py`
- `tests/engine/test_glm52_moe_train_engine.py`
- `tests/train/test_glm52_sft_smoke.py`

## 19. 关联材料

下面材料用于追溯细节，新读者只阅读本文即可了解完整结论：

- [`glm52_full_automodel_implementation_and_xtuner_integration.md`](./glm52_full_automodel_implementation_and_xtuner_integration.md)：完整模型和 AutoModel 链路原始调研；
- [`glm52_decoupled_fsdp_initial_design.md`](../../../Automodel/notes/training/xtuner/glm52_decoupled_fsdp_initial_design.md)：早期方案记录；
- [`glm52_xtuner_automodel_mesh_analysis.md`](../../../Automodel/notes/training/xtuner/glm52_xtuner_automodel_mesh_analysis.md)：两边 mesh 源码对照；
- [`glm52_ep2_memory_profile_analysis.md`](../../../Automodel/notes/training/xtuner/glm52_ep2_memory_profile_analysis.md)：EP2 dispatcher/expert GEMM 实验；
- [`glm52_ep4_memory_profile_analysis.md`](../../../Automodel/notes/training/xtuner/glm52_ep4_memory_profile_analysis.md)：EP4 dense/optimizer 显存归因；
- [`glm52_fsdp_ep_sharding_summary.md`](../../../Automodel/notes/training/xtuner/glm52_fsdp_ep_sharding_summary.md)：面向初学者的 rank0-rank7 切分说明。
