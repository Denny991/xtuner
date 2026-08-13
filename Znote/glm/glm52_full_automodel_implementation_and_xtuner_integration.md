# AutoModel 完整 GLM-5.2 实现与 XTuner 集成分析

> 核对版本：AutoModel `abec122f`，XTuner `fc47628d`
>
> 目标：说明 AutoModel 当前如何支持完整 GLM-5.2，并判断 XTuner 已有能力、真正缺口和建议集成顺序。
>
> 说明：本文的“完整模型”主要指 78 层、256 routed experts、约 355B 总参数的 GLM-5.2 backbone；MTP 是否包含在内会单独标注。

## 1. 结论先说

AutoModel 对 GLM-5.2 的支持不是一个单独模型文件，而是一条完整链路：

```text
HF GLM-5.2 checkpoint
  -> GlmMoeDsaConfig / head_dim 修正
  -> GLM DSA + IndexShare + MoE 模型
  -> TileLang sparse MLA / HybridEP / torch._grouped_mm
  -> 主 device_mesh + 独立 moe_mesh
  -> dense FSDP2 + expert EP/expert-FSDP
  -> Pipeline Parallel / Context Parallel
  -> HF adapter + DCP/HF checkpoint
```

完整模型方面，AutoModel 仓库提供了两条不同路径：

1. **完整 backbone SFT**：示例使用 `PP=4 + EP=64`，预计 32 节点、256 GPU；另有 `CP=8` 的 32K 长序列配置。
2. **完整 frozen target**：DSpark 使用 `PP=1 + EP=64`，预计 8 节点、64 GPU；完整 78 层目标模型冻结，只训练小型 drafter。

但需要把“有代码和 recipe”与“已经实跑验证”分开：当前仓库有完整模型配置和执行链，我们本轮实际验证的是 6 层 NoMTP parity/smoke，**没有证据表明本地已经完成过 32 节点完整 SFT**。

对 XTuner 而言，GLM-5.2 模型本身不需要重新移植。最新 XTuner 已经具备：

- 78 层完整配置解析、DSA、IndexShare、NoAux router 和 MoE；
- MTP 物理层及其 checkpoint/recompute 生命周期；
- HF 加载和导出；
- SP、activation checkpoint、DeepEP/all2all/AGRS；
- Triton/CUTLASS expert GEMM、TileLang/cuDNN DSA；
- tiny 模型的 EP、SP、MTP、HF round-trip 和训练回归测试。

XTuner 真正需要从 AutoModel 借鉴的优先级是：

| 优先级 | 需要补的能力 | 为什么 |
|---|---|---|
| P0 | dense FSDP 与 expert EP/expert-FSDP 解耦 | 当前 EP 越大，XTuner dense FSDP 度越小；完整模型下会放大 dense 参数和 optimizer state 的单卡占用 |
| P0 | 多 mesh 下的 HF/DCP 加载、保存、optimizer resume | 只改训练 mesh 会出现“能跑但权重 shard 加载/导出错误”的高风险问题 |
| P1 | 完整模型多机 load + 2-step 验证 | tiny/6 层通过不能证明 78 层 FP8 checkpoint、256 experts 和 optimizer 能正确落盘与恢复 |
| P1 | Pipeline Parallel 与跨 stage IndexShare carry | 想复现 AutoModel `PP4+EP64` 的 32 节点 SFT 拓扑时必须具备 |
| P2 | 32K 长序列的 CP/THD 路径评估 | XTuner 已有 SP，先验证是否满足目标；不要直接把 AutoModel CP 当作同名功能复制 |

反过来，下面这些**不应从 AutoModel 重复移植**：GLM 模型结构、MTP、DeepEP、CUTLASS、SP。XTuner 已经有自己的实现，其中 MTP 和 CUTLASS 甚至比当前 AutoModel GLM 路径更完整或更适合现有生产配置。`head_dim` 方面 XTuner 已拆分内部值与 HF 导出值，但仍建议补一个 raw-config guard，避免旧版 HF `attribute_map` 再次覆盖 `qk_rope_head_dim`。

## 2. 先区分三种“GLM-5.2”

| 场景 | 模型规模/状态 | 用途 | 本轮验证状态 |
|---|---|---|---|
| 6 层 NoMTP checkpoint | 从完整模型裁出的约 30B parity 件 | loss、grad_norm、时间、显存和训练链路对齐 | 已在 8xH200 跑 200 step |
| 完整 78 层 backbone SFT | 约 355B、256 routed experts | 真正完整模型微调 | AutoModel 有 32 节点 recipe；本轮未实跑 |
| 完整 78 层 frozen target | 约 355B、目标模型冻结 | DSpark hidden-state capture，只训练 drafter | AutoModel 有 8 节点 recipe；单机只做 6 层 smoke |

6 层 checkpoint 可以验证训练数值和工程链路，但不能证明完整模型能生成，也不能替代完整模型的多机装载、checkpoint 和下游能力验证。

## 3. AutoModel 的 GLM-5.2 实现链路

### 3.1 模型注册与构造入口

注册入口位于：

- `nemo_automodel/_transformers/registry.py`
- `GlmMoeDsaForCausalLM -> nemo_automodel.components.models.glm_moe_dsa.model.GlmMoeDsaForCausalLM`

YAML 中的 `NeMoAutoModelForCausalLM.from_pretrained` 先读取 HF config，再通过 registry 找到 AutoModel 自定义 GLM 类。随后 `infrastructure.py` 负责模型低精度处理、并行化、meta 参数实体化和 checkpoint 加载。

主要调用关系：

```text
examples/llm_finetune/glm/*.yaml
  -> NeMoAutoModelForCausalLM.from_pretrained
  -> _transformers/registry.py
  -> components/models/glm_moe_dsa/model.py
  -> _transformers/infrastructure.py
  -> components/moe/parallelizer.py
  -> components/checkpoint/checkpointing.py
```

### 3.2 78 层 backbone、dense 前缀和 MoE 层

`nemo_automodel/components/models/glm_moe_dsa/model.py` 中：

- `GlmMoeDsaModel` 按 `range(config.num_hidden_layers)` 构建 decoder stack；完整 config 对应 78 层。
- 每层依据 `mlp_layer_types`，或者 `first_k_dense_replace`，选择普通 dense MLP 或 MoE。
- MoE config 从 HF config 中读取 `n_routed_experts`、`n_shared_experts`、`num_experts_per_tok`、routing scale 等字段。
- embedding、RMSNorm、LM head 与模型 dtype 一起构建，支持 FP32 master parameter + BF16 mixed precision。

完整模型默认结构参数在 XTuner 和 HF config 中也能看到：

```text
num_hidden_layers       = 78
first_k_dense_replace   = 3
hidden_size             = 6144
n_routed_experts        = 256
n_shared_experts        = 1
num_experts_per_tok     = 8
moe_intermediate_size   = 2048
```

### 3.3 DSA、MLA 与 IndexShare

GLM-5.2 不是普通全注意力。AutoModel 在 `glm_moe_dsa/layers.py` 中实现：

- Q-LoRA、KV-LoRA、`qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim` 的 MLA 投影；
- DSA indexer，根据 query/key 计算 sparse top-k token 索引；
- sparse MLA，只对 indexer 选中的 key/value 做注意力；
- eager/SDPA fallback 和 TileLang fused sparse kernel；
- indexer 的 FP32 score 计算与因果 mask。

IndexShare 的含义是：并非每层都拥有 indexer。`indexer_types[layer] == "shared"` 时，该层没有自己的 indexer，而是复用前一个 `full` 层产生的 top-k indices。

```text
full layer    -> 计算 top-k indices
shared layer  -> 复用上一份 top-k
shared layer  -> 继续复用
下一 full     -> 重新计算 top-k
```

这也解释了 6 层 checkpoint 中后几层没有 `indexer.*` 权重：如果对应层类型是 `shared`，缺少独立 indexer 是模型设计，不是 checkpoint 损坏。

### 3.4 Pipeline Parallel 下的 IndexShare carry

完整 SFT recipe 使用 `PP=4`。普通 PP 只需在 stage 之间传 hidden states，但 GLM-5.2 还可能在 stage 边界遇到：

```text
stage 0 最后一层：full，生成 top-k
stage 1 第一层：shared，需要复用 stage 0 的 top-k
```

AutoModel 在 `GlmMoeDsaForCausalLM.get_pipeline_stage_metas()` 和 `forward()` 中额外传递：

```text
(hidden_states, topk_indices)
```

top-k 原本是整数，但 PyTorch pipeline 接收 buffer 会按可求导 activation 处理，所以 AutoModel 临时用 FP32 传输，并用零权重 autograd link 给它定义零梯度；下一 stage 再转回 `int32/int64`。这是 GLM-5.2 能安全跨 PP stage 使用 IndexShare 的模型特定逻辑。

### 3.5 Context Parallel 与 32K THD packed sequence

AutoModel 的 GLM DSA 声明支持 CP，但当前只在 `backend.attn=tilelang` 时启用：

- `glm_moe_dsa/cp.py` 将 THD packed query 沿 token 维连续切到各 CP rank；
- K/V 使用可求导 all-gather 恢复全序列上下文；
- 保留全局 `cu_seqlens` 和 query index；
- token 数必须能被 CP size 整除；
- `packed_sequence_thd_collater` 必须正确生成 `cu_seqlens`。

完整 32K recipe 使用：

```text
PP=4, EP=64, CP=8, TP=1
packed_sequence_size=32768
backend.attn=tilelang
```

这条 CP 路径与 XTuner 的 `SP_SIZE` 作用相近但不是同一个实现，不能只按参数名一一对应。

### 3.6 MoE dispatcher 与 expert GEMM

AutoModel 的公共 MoE 层位于：

- `nemo_automodel/components/moe/layers.py`
- `nemo_automodel/components/moe/experts.py`

可选能力包括：

| 类别 | AutoModel GLM 当前 recipe/默认 |
|---|---|
| token dispatcher | `HybridEP`；框架也支持 DeepEP、torch、UCCL-EP |
| expert GEMM | 默认 `torch._grouped_mm`；也有 TE/GMM 等路径 |
| router precision | FP32 |
| activation checkpoint | 开启；可避免重算 router 导致 shape/路由不一致 |
| reshard after forward | 4K recipe 关闭，32K CP recipe 开启 |

这里要避免一个误解：HybridEP、DeepEP 负责的是 token dispatch/combine；`torch._grouped_mm`、CUTLASS、Triton 负责的是 expert GEMM。它们是两层不同的问题。

### 3.7 HF 权重适配与 `head_dim` 修复

HF checkpoint 中 experts 是逐 expert 的 `gate_proj/up_proj/down_proj`，AutoModel 运行时使用 grouped expert tensor。`GlmMoeDsaStateDictAdapter` 继承 GLM4-MoE adapter，完成：

```text
HF split experts <-> AutoModel grouped experts
```

它还特别处理 DSA indexer 中不应量化的权重。

GLM-5.2 config 同时存在：

```text
head_dim=192
qk_rope_head_dim=64
```

HF `attribute_map` 会让前者覆盖后者，导致 `kv_a_proj_with_mqa` 按 704 构造，而 checkpoint 实际是 576。AutoModel finetune YAML 用 `model.config.head_dim: 64` 修复；DSpark 使用 `repair_glm_5_2_qk_rope_head_dim()` 从原始 JSON 恢复真实的 `qk_rope_head_dim=64`。

### 3.8 分片后加载和 checkpoint

完整模型不能先在每张 GPU 上创建完整权重再切分。AutoModel 的 `infrastructure.py` 会根据 PP/TP/EP 判断加载顺序：

1. meta device 上构建模型；
2. 应用 PP、EP 和 FSDP2；
3. 实体化各 rank 需要的参数 shard；
4. 使用 checkpoint adapter 将 HF split experts 映射到运行时 grouped experts；
5. DCP 直接写入分片后的目标 tensor；
6. 保存时可生成 distributed checkpoint 或 consolidated HF safetensors。

`Checkpointer` 会收到 `moe_mesh`。这很重要，因为 expert 的 HF 转换和 checkpoint 不能假设所有参数都使用主 dense FSDP mesh。

## 4. AutoModel 完整模型如何切分

### 4.1 4K 完整 SFT：PP4 + EP64

配置文件：`examples/llm_finetune/glm/glm_5.2_tulu3_4k_tilelang_100k.yaml`

```text
world_size = 32 nodes * 8 GPU = 256
PP = 4
CP = 1
TP = 1
EP = 64

每个 PP stage 的非 PP rank 数 = 256 / 4 = 64
dense FSDP size              = 64
moe_mesh                     = [EP_SHARD=1, EP=64]
```

可以这样理解：

- 78 层 backbone 先按 PP 分给 4 个 pipeline stage，每个 stage 只持有一部分层；
- 每个 stage 内，dense 参数在 64 rank 上做 FSDP；
- 256 experts 在同一 stage 内按 EP64 分配，每个 EP rank 负责 4 个 routed experts；
- 因为 `EP_SHARD=1`，该配置主要依靠 PP 和 EP 分散 expert 参数，没有再对单个 EP shard 做额外 expert-FSDP。

### 4.2 32K 完整 SFT：PP4 + EP64 + CP8

配置文件：`examples/llm_finetune/glm/glm_5.2_tulu3_32k_tilelang_cp8.yaml`

```text
world_size = 256
PP = 4
CP = 8
TP = 1
DP = 256 / (PP4 * CP8) = 8
```

主 mesh 形状是 `PP4 x DP_SHARD8 x CP8`。AutoModel 将 `DP_SHARD` 和 `CP` 展平为 `DP_SHARD_CP=64` 给 dense FSDP 使用，因此：

```text
dense FSDP size = DP8 * CP8 = 64
moe_mesh        = [EP_SHARD=1, EP=64]
```

也就是说，CP8 切序列，但没有把 dense 参数的 FSDP 度从 64 降成 8。

### 4.3 DSpark 完整 frozen target：PP1 + EP64

配置文件：`examples/speculative/dspark/glm_5.2_dspark.yaml`

```text
world_size = 8 nodes * 8 GPU = 64
PP = 1
EP = 64
dense FSDP size = 64
moe_mesh        = [EP_SHARD=1, EP=64]
```

这里目标模型冻结，不创建完整 AdamW 状态，因此 64 GPU 能用于 full target hidden-state capture。它不能直接证明同样 64 GPU 足以进行完整 SFT。

### 4.4 为什么独立 mesh 对完整模型重要

AutoModel 同一批 rank 可以同时形成两种视图：

```text
dense view:  所有非 PP rank 一起切 dense
MoE view:    [EP_SHARD, EP] 切 experts
```

所以增大 EP 不必牺牲 dense FSDP 度。这一设计在完整模型上尤其重要，因为每层都有 attention、norm、router、shared expert 等非 routed-expert 参数，训练还会为它们创建 master weight、gradient 和 AdamW states。

### 4.5 两种切分各自的优缺点，完整模型偏向哪种

| 方案 | 优点 | 代价/限制 | 更适合 |
|---|---|---|---|
| XTuner 当前耦合 `[FSDP, EP]` | mesh、梯度和 checkpoint 逻辑较简单；dense FSDP group 较小，单次 dense all-gather/reduce-scatter 通信范围小；当前生产代码已经验证 | EP 增大时 dense FSDP 同步缩小，dense 参数、gradient、master weight 和 AdamW state 沿 EP 复制 | 小/裁剪模型、EP 较小、显存仍有余量、优先减少通信和改造风险 |
| AutoModel 独立 dense/expert mesh | EP 大小不影响 dense FSDP 度；可以分别为 dense 和 experts 选择最合适的切分；更容易装载超大 MoE | process group、nested FSDP、梯度缩放和 checkpoint 更复杂；dense FSDP group 变大后通信可能增加 | 完整超大 MoE、EP 很大、dense/optimizer state 已成为显存瓶颈 |

以完整 SFT 的 `PP4 + EP64` 为例，每个 PP stage 有 64 个 rank：

```text
AutoModel 独立 mesh:
  dense FSDP = 64
  expert EP  = 64

若沿用 XTuner 当前耦合公式:
  dense FSDP = 64 / EP64 = 1
  dense 参数会在 64 个 EP 位置复制
```

因此，对 6 层 EP2/EP4 实验，耦合实现并非“错误”，尤其 EP2 + CUTLASS 已经接近 AutoModel；但若目标是完整 355B、EP64 的 full finetune，**独立 mesh 明显更合适，甚至是避免 dense/optimizer state 大量复制的前置能力**。

它的代价主要是通信和工程复杂度，而不是数值精度：切分方式正确实现后不应改变 loss/gradient；实际速度是否更快必须实测，不能仅凭显存下降推断。

## 5. AutoModel 当前实现的边界

### 5.1 当前 GLM 自定义模型没有 MTP auxiliary layer

这是本次源码核对中最需要说清的一点。

`nemo_automodel/components/models/glm_moe_dsa/model.py` 只按 `num_hidden_layers` 构建 backbone layers，再接 norm 和 LM head；该目录没有 `mtp`、`num_nextn_predict_layers` 或 GLM MTP auxiliary head 的实现。

因此 AutoModel 文档中的“full GLM-5.2”应理解为：

> 完整 78 层 GLM-5.2 backbone 的训练或 frozen target 加载，不等于完整支持 GLM-5.2 checkpoint 中可能存在的 MTP auxiliary training path。

XTuner 已经实现 MTP config、物理 MTP 层、共享权重、IndexShare 生命周期以及 HF round-trip，所以这一项无需向 AutoModel 对齐，反而应保留 XTuner 自己的实现。

### 5.2 TP 不支持

AutoModel 的 GLM model capabilities 明确为：

```text
TP: false
CP: true
PP: true
EP: true
```

公共 MoE parallelizer 也会断言 custom MoE 的 TP size 必须为 1。完整模型示例通过 PP、CP、EP、FSDP2 扩展，而不是 TP。

### 5.3 完整 32 节点 recipe 尚未由本轮实验验证

当前可以确认的是源码路径、配置关系和 6 层训练结果。完整 355B SFT 还需要额外验证：

- 32 节点 rendezvous 和 topology；
- 完整 FP8 checkpoint 的多机 dequant/load；
- 78 层 PP stage 划分与跨 stage top-k carry；
- optimizer state 的显存与保存恢复；
- 长时间训练稳定性和最终 consolidated checkpoint。

因此汇报时应说“AutoModel 提供完整模型 recipe 和实现链路”，不要说“已经完成完整模型训练验证”。

## 6. XTuner 当前已经有什么

### 6.1 模型结构和配置比 6 层实验更完整

`/home/liutong/ZmyCode/xtuner/xtuner/v1/model/moe/glm52.py` 已实现：

- 完整 GLM-5.2 默认结构：78 层、256 routed experts；
- HF config -> XTuner config；
- `head_dim` 与 `qk_rope_head_dim` 分开保存：转换逻辑令内部 attention head dim 取 `cfg.qk_rope_head_dim`，HF 导出则通过独立 `hf_head_dim` 写回 192；
- dense/sparse layer 类型；
- DSA IndexShare；
- MTP physical layer、共享 MTP 和 indexer 类型校验；
- fused expert tensor 与 HF split expert tensor 双向转换。

所以 XTuner 不需要复制 AutoModel 的 `GlmMoeDsaForCausalLM`。

### 6.2 DSA 和 kernel

XTuner 已经提供：

| 模块 | 已有能力 |
|---|---|
| sparse MLA | torch、TileLang、cuDNN DSA |
| IndexShare | `SequenceContext` 中的跨层 top-k cache、释放计划和 optional offload |
| dispatcher | all2all、DeepEP、AGRS |
| expert GEMM | Triton，及 `XTUNER_USE_CUTLASS_GROUP_GEMM=1` 的 CUTLASS 路径 |
| compile | 对 GLM DSA graph break 和 MoE 子阶段做模型特定配置 |

本轮 profile 已证明 CUTLASS 能显著降低 XTuner Triton group GEMM 的 active memory。这个优化应继续保留，不属于从 AutoModel 迁移的内容。

### 6.3 SP、MTP 和测试

XTuner 当前测试覆盖：

- HF GLM config 语义转换；
- tiny GLM HF save/load round-trip；
- MTP shared weight 和 activation checkpoint；
- SP2 下 LM/MTP loss 与 gradient 对齐；
- EP1/EP4/EP8 的 engine smoke；
- DeepEP/all2all 和 FP8 组合；
- tiny SFT 训练。

这些测试说明 XTuner 已经具备较完整的单模型训练语义，但仍不等于完成 full 355B 多机训练。

## 7. 两边当前最关键的差异

| 能力 | AutoModel | XTuner 当前 | 判断 |
|---|---|---|---|
| 78 层 GLM backbone | 有 | 有 | 不迁移 |
| DSA/IndexShare | 有 | 有，且有更细的 cache 生命周期 | 不迁移 |
| MTP auxiliary path | 当前 GLM 目录未实现 | 已实现 | 保留 XTuner |
| HF head-dim 处理 | YAML/repair helper 从 raw JSON 恢复 64 | config 已分离内部/HF 字段，但 `from_hf()` 仍信任 HF 解析后的 `cfg.qk_rope_head_dim` | 保留结构，补 raw-config guard |
| sparse attention | TileLang、SDPA fallback | TileLang、cuDNN DSA、torch | 按性能选，不迁移 |
| dispatcher | HybridEP、DeepEP 等 | DeepEP、all2all、AGRS | 已有 |
| expert GEMM | 默认 `torch._grouped_mm` | Triton/CUTLASS | 已有，CUTLASS 已验证省显存 |
| PyTorch FSDP2 | 有 | 有 | 底层相同 |
| dense/expert mesh 解耦 | 有 | 无，当前为 `[FSDP=world/EP, EP]` | **P0 集成** |
| expert nested FSDP | 独立 `EP_SHARD` mesh | 与当前 FSDP 维组合，但 dense 同时被迫复制 | **P0 重构** |
| Pipeline Parallel | GLM 明确支持，含 top-k carry | v1 当前无训练 PP engine/config | 完整 SFT 拓扑需要 P1 集成 |
| 长序列切分 | CP + THD + TileLang | SP + DSA backend | 先评估，不做机械迁移 |
| 多 mesh checkpoint | Checkpointer 接收 `moe_mesh` | HF load/save 多处统一依赖 `self.fsdp_mesh` | **P0 集成** |
| DSpark | 有完整 target builder/recipe | 当前无同类功能 | 仅产品需要时考虑 |

## 8. XTuner 需要集成什么

### 8.1 P0：dense 与 expert 并行解耦

XTuner 当前在 `MoE._init_device_mesh()` 中创建：

```text
model_mesh = [FSDP=world_size/EP, EP]
```

并由 `_replicate_other_params()` 把非 expert 参数沿 EP 显式复制。8 卡 EP4 时：

```text
dense FSDP = 2
dense 参数每个 shard 在 4 个 EP 位置复制
```

目标拓扑应改为同一 root mesh 的三种子视图：

```text
dense_fsdp_mesh  = world_size
expert_fsdp_mesh = world_size / EP
ep_mesh          = EP
```

8 卡 EP4 示例：

```text
dense FSDP group: [0,1,2,3,4,5,6,7]
EP groups:        [0,1,2,3] / [4,5,6,7]
expert-FSDP:      [0,4] / [1,5] / [2,6] / [3,7]
```

建议改动：

| XTuner 文件 | 建议 |
|---|---|
| `xtuner/v1/config/fsdp.py` | 增加默认关闭的实验开关，例如 `decouple_dense_fsdp` |
| `xtuner/v1/model/moe/moe.py` | 从同一 root mesh 派生 dense-FSDP、expert-FSDP、EP mesh |
| `xtuner/v1/model/moe/moe.py` | experts 先做 EP，再单独 expert-FSDP；block dense FSDP 时排除 expert params |
| `xtuner/v1/model/moe/moe.py` | decoupled 路径停止 `_replicate_other_params()`，重新核对 gradient scaling |

第一版应限定：

```text
GLM-5.2
TP=1
HSDP disabled
EP>1
world_size % EP == 0
compile disabled
```

稳定后再推广为通用 MoE parallel folding 能力。

### 8.2 P0：HF/DCP checkpoint 必须同时改

XTuner `base.py` 的加载、导出和 all-gather 多处直接读取 `self.fsdp_mesh`。解耦后：

```text
dense 参数  -> dense_fsdp_mesh
expert 参数 -> expert_fsdp_mesh + ep_mesh
```

必须让每个参数根据自己的 DTensor placements 或显式参数类别选择通信 group 和 shard offset。至少验证：

1. 完整/裁剪 HF checkpoint strict load；
2. DCP model + optimizer save/resume；
3. HF export 后重新 load；
4. fused expert 拆分后 expert id、gate/up/down 顺序不变；
5. EP2 保存后是否允许 EP4 恢复，若不支持要明确报错。

这一步的风险高于 mesh 创建本身。只修改 `moe.py` 会留下静默加载错 shard 的可能。

### 8.3 P1：先做完整模型 load，不急着直接跑长训练

在扩大模型前，建议先给 `Glm52MoEConfig.from_hf()` 增加与 AutoModel repair helper 等价的保护：同时读取 raw `config.json`，若解析后的 `qk_rope_head_dim` 与 raw 值不一致，则以 raw `qk_rope_head_dim=64` 为准，并增加 `head_dim=192 / qk_rope_head_dim=64` 回归测试。这是小改动，但能避免完整 checkpoint 在不同 Transformers 版本下出现 704/576 shape mismatch。

建议按以下顺序扩大规模：

1. 6 层 EP2/EP4 decoupled，2 step；
2. 6 层 200 step，核对 loss/grad、snapshot 和 checkpoint round-trip；
3. 完整 78 层 frozen load + 1 次 forward；
4. 完整 78 层 optimizer build + 2 step；
5. 再决定 200 step 和生产数据。

完整模型阶段要记录每个 rank：

```text
rank -> dense FSDP group
rank -> expert-FSDP group
rank -> EP group
local expert ids
参数/gradient/optimizer state GiB
```

### 8.4 P1：是否集成 Pipeline Parallel

如果目标是复现 AutoModel 的完整 SFT recipe：

```text
PP4 + EP64 + dense FSDP64
```

XTuner 需要新增真正的 PP engine，而不仅是把 layers 放进 `ModuleDict`。GLM 特有工作包括：

- stage 切分 embedding、decoder layers、norm 和 LM head；
- interleaved 1F1B schedule；
- stage 间同时传 hidden states 和 IndexShare top-k；
- packed THD 的 shape metadata；
- top-k carry 的不可导/零梯度处理；
- PP stage 的 HF load/save key 范围和 checkpoint rank metadata；
- MTP 放置位置及是否跨 stage。

这是一项明显大于 mesh 解耦的工作。若短期只要求“完整模型能训”，可以先研究 `PP=1 + EP + expert-FSDP` 是否在目标 GPU 数上可行；若要求与 AutoModel 生产拓扑一致，再做 PP。

### 8.5 P2：CP 还是继续使用 XTuner SP

AutoModel CP8 和 XTuner SP2 都切 token/sequence，但通信和 attention kernel 接口不同。建议先用 XTuner 现有 SP 做：

- 4K/16K/32K loss 和 gradient parity；
- cuDNN DSA 与 TileLang 的显存/时间对比；
- SP 与 EP、decoupled FSDP、MTP 的组合测试。

只有现有 SP 无法支撑 32K 或性能明显不足时，再考虑移植 AutoModel 的 CP batch sharder 和 K/V all-gather 设计。

## 9. 推荐实施路线

### 阶段 A：并行语义正确

目标：证明 decoupled mesh 的参数归属、梯度和 optimizer 更新正确。

验收：

- 8 卡 EP2/EP4 group 与预期一致；
- loss/grad_norm finite；
- 固定参数在 coupled/decoupled 下更新方向一致；
- 无重复 all-reduce、无漏 reduce；
- EP4 CUTLASS active memory 向 AutoModel 靠近。

### 阶段 B：checkpoint 正确

目标：训练、恢复、HF 导出形成闭环。

验收：

- 2 step 保存，恢复后继续 2 step，与连续 4 step 对齐；
- optimizer step、momentum、variance 恢复；
- HF export reload 后随机抽查 dense、router、shared expert、routed expert 和 MTP 权重；
- 不同 rank 不产生重复或缺失 expert shard。

### 阶段 C：完整模型装载

目标：从“6 层算法正确”升级到“78 层工程可用”。

验收：

- 完整 checkpoint strict load；
- 所有 256 experts 映射唯一且完整；
- FP8 base dequant/load 峰值不 OOM；
- frozen forward 和 2-step full finetune 均成功；
- 保存目录大小与参数/optimizer 理论规模相符。

### 阶段 D：生产拓扑

根据资源和目标二选一：

| 路线 | 说明 | 工作量 |
|---|---|---|
| PP1 + decoupled EP/expert-FSDP | 先利用独立 mesh 扩大 expert-FSDP，验证完整模型训练 | 中 |
| PP4 + EP64 | 对齐 AutoModel 32 节点 recipe，增加 PP 和 IndexShare carry | 高 |

## 10. 建议新增的测试

| 测试 | 卡数 | 目的 |
|---|---:|---|
| mesh membership unit/functional | 8 | 核对 dense、EP、expert-FSDP group |
| coupled vs decoupled parameter update | 8 | 排除 gradient scale 错误 |
| EP2/EP4 HF round-trip | 8 | 验证多 mesh load/export |
| EP2 save -> EP2 resume | 8 | 验证 optimizer state |
| full 78-layer frozen load | 64 或规划资源 | 验证真实 checkpoint 和 256 experts |
| full 78-layer 2-step SFT | 依据内存估算 | 验证 optimizer/gradient 完整链路 |
| PP stage top-k carry | 至少 4 | 若引入 PP，验证 full/shared 跨 stage |
| SP/EP/FSDP/MTP 组合 | 8 | 防止生产组合回归 |

## 11. 最终建议

短期最值得做的不是继续增加 6 层配置组合，而是实现一个默认关闭的 XTuner decoupled mesh 原型，并把 checkpoint round-trip 一起纳入第一期。

推荐顺序：

```text
独立 mesh
  -> 梯度/optimizer 正确性
  -> HF/DCP checkpoint
  -> 6 层 200-step 性能
  -> 完整 78 层 load/2-step
  -> 决定是否建设 PP4
```

可以向领导概括为：

> AutoModel 对完整 GLM-5.2 的核心优势，不是模型层比 XTuner 多，而是已经把 78 层 GLM backbone 接入了独立 dense/expert mesh、FSDP2、PP、CP 和分片 checkpoint 链路。XTuner 当前的 GLM、MTP、DeepEP、CUTLASS 和 SP 已较完整，最需要补的是 dense FSDP 与 EP 解耦及其 checkpoint 适配；若要复现 AutoModel 的 32 节点完整 SFT 拓扑，再继续补 PP 和跨 stage IndexShare carry。

## 12. 关键源码索引

### AutoModel

| 内容 | 文件 |
|---|---|
| GLM registry | `nemo_automodel/_transformers/registry.py` |
| GLM backbone、IndexShare、PP carry | `nemo_automodel/components/models/glm_moe_dsa/model.py` |
| DSA indexer、MLA、TileLang | `nemo_automodel/components/models/glm_moe_dsa/layers.py` |
| GLM CP batch sharder | `nemo_automodel/components/models/glm_moe_dsa/cp.py` |
| HF state adapter | `nemo_automodel/components/models/glm_moe_dsa/state_dict_adapter.py` |
| 通用 MoE/expert backend | `nemo_automodel/components/moe/layers.py`、`experts.py` |
| EP + nested FSDP | `nemo_automodel/components/moe/parallelizer.py` |
| 主 mesh 与 moe_mesh | `nemo_automodel/components/distributed/mesh_utils.py`、`mesh.py` |
| 模型装载编排 | `nemo_automodel/_transformers/infrastructure.py` |
| HF/DCP checkpoint | `nemo_automodel/components/checkpoint/checkpointing.py` |
| head-dim repair / DSpark target | `nemo_automodel/recipes/llm/_dspark_target_build.py` |
| 4K full recipe | `examples/llm_finetune/glm/glm_5.2_tulu3_4k_tilelang_100k.yaml` |
| 32K CP8 recipe | `examples/llm_finetune/glm/glm_5.2_tulu3_32k_tilelang_cp8.yaml` |
| full frozen DSpark target | `examples/speculative/dspark/glm_5.2_dspark.yaml` |

### XTuner

| 内容 | 文件 |
|---|---|
| GLM config、MTP、HF mapping | `/home/liutong/ZmyCode/xtuner/xtuner/v1/model/moe/glm52.py` |
| MoE forward、FSDP/EP mesh | `/home/liutong/ZmyCode/xtuner/xtuner/v1/model/moe/moe.py` |
| HF load/save shard 逻辑 | `/home/liutong/ZmyCode/xtuner/xtuner/v1/model/base.py` |
| FSDP config | `/home/liutong/ZmyCode/xtuner/xtuner/v1/config/fsdp.py` |
| DSA/IndexShare | `/home/liutong/ZmyCode/xtuner/xtuner/v1/module/attention/dsa_mla.py`、`dsa_topk_sharing.py` |
| expert tensor | `/home/liutong/ZmyCode/xtuner/xtuner/v1/module/grouped_linear/moe_group_linear.py` |
| dispatcher | `/home/liutong/ZmyCode/xtuner/xtuner/v1/module/dispatcher/` |
| Triton/CUTLASS expert GEMM | `/home/liutong/ZmyCode/xtuner/xtuner/v1/ops/moe/cuda/` |
| 当前 GLM SFT config | `/home/liutong/ZmyCode/xtuner/examples/v1/config/sft_glm5p2.py` |
| GLM 模型测试 | `/home/liutong/ZmyCode/xtuner/tests/model/test_glm52_moe.py` |
| GLM engine 测试 | `/home/liutong/ZmyCode/xtuner/tests/engine/test_glm52_moe_train_engine.py` |

## 13. 相关实验文档

- [`glm52_xtuner_automodel_mesh_analysis.md`](../../../Automodel/notes/training/xtuner/glm52_xtuner_automodel_mesh_analysis.md)：8 卡 EP2/EP4 mesh 和改造风险。
- [`glm52_ep2_memory_profile_analysis.md`](../../../Automodel/notes/training/xtuner/glm52_ep2_memory_profile_analysis.md)：EP2 dispatcher、GEMM 和 active memory 分析。
- [`glm52_ep4_memory_profile_analysis.md`](../../../Automodel/notes/training/xtuner/glm52_ep4_memory_profile_analysis.md)：EP4 dense FSDP 差异和 snapshot 证据。
- [`glm52_fsdp_ep_sharding_summary.md`](../../../Automodel/notes/training/xtuner/glm52_fsdp_ep_sharding_summary.md)：面向初学者的 rank-by-rank 切分说明。
- [`glm5.2_readme.md`](../../../Automodel/notes/training/xtuner/glm5.2_readme.md)：当前 6 层 parity、生产配置和执行命令。
