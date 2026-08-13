# GLM-5.2 Indexer SFT 调研

> 更新时间：2026-08-13
>
> 核对版本：XTuner `9dabb956`，NeMo AutoModel `e2c48886`
>
> 调研范围：DSA Indexer 的训练语义、GLM-5.2 普通 SFT 中的实际行为、IndexShare、
> checkpoint，以及 XTuner 若要支持 Indexer warm-up / 联合训练需要补齐的能力。

## 1. 先说结论

当前最容易混淆的是“带 Indexer 跑 SFT”和“训练 Indexer”并不是一回事。

| 场景 | 主模型 | Indexer | 损失 | 当前 XTuner |
|---|---|---|---|---|
| 已发布 GLM-5.2 的普通下游 SFT | 训练 | 冻结 | LM cross entropy | **已支持** |
| DSA Indexer warm-up | 冻结 | 训练 | dense attention 蒸馏 KL | **未支持** |
| DSA sparse adaptation | 训练 | 训练 | 主模型用 LM loss，Indexer 用 KL loss | **未支持** |
| IndexShare-aware 训练 | 训练 | 仅训练 `full` 层 Indexer | 多层 attention 蒸馏 KL + LM loss | **未支持** |

最新 XTuner 中，`DSAIndexer` 会显式执行 `requires_grad_(False)`，其 `forward()` 也在
`torch.no_grad()` 下运行。普通 SFT 只使用预训练 checkpoint 中已有的 Indexer 产生
top-k token 位置，不会更新 Indexer。

因此：

1. 如果目标是对已发布的 GLM-5.2 做业务 SFT，当前“冻结 Indexer”是明确且合理的行为。
2. 如果目标是把 dense MLA 模型转换为 DSA，或者重新训练/适配 Indexer，当前配置无法做到。
3. 仅删除 `no_grad` 或把 `requires_grad` 改为 `True` 仍然不够。top-k 下标是整数，LM loss
   无法通过整数下标反传给 Indexer；必须增加 top-k 前的可微分 score 和独立蒸馏损失。

### 1.1 一眼看懂三种训练方式

如果只想先弄明白“谁在训练、谁没有训练”，看下面三张图即可。

**A. 当前 XTuner 已支持：冻结 Indexer 的普通 GLM-5.2 SFT**

```text
HF checkpoint 中的 Indexer 权重
        |
        v
加载到模型，但设置 requires_grad=False + no_grad
        |
        v
Indexer 正常计算 Top-K token 位置
        |
        v
Sparse Attention 只计算这些 Top-K 位置
        |
        v
LM cross entropy loss
        |
        +----> 更新 attention / MoE / embedding 等主模型参数
        |
        +--X-> Indexer 没有梯度，权重保持不变
```

一句话：**Indexer 在工作，但没有被训练。**

**B. 当前 XTuner 未支持：Indexer warm-up**

```text
冻结的 backbone -> hidden states
                         |
                         +-> Dense Attention teacher -> 分布 p
                         |
                         +-> Indexer scores -> softmax -> 分布 q

分布 p + 分布 q -> KL(p || q) -> 只更新 Indexer
backbone 和 Dense Attention teacher 均不更新
```

一句话：**主模型不动，让 Indexer 学会模仿完整 attention 的选点结果。**

**C. 当前 XTuner 未支持：Sparse adaptation 联合训练**

```text
hidden states
  |
  +-> Indexer -> 离散 Top-K -> Sparse Attention -> LM loss -> 更新 backbone
  |                 (这里不传梯度)
  |
  +-> detach -> Indexer scores + teacher 分布 -> KL loss -> 更新 Indexer
```

这里有两条独立梯度：

- LM loss 负责让主模型适应 sparse attention；
- KL loss 负责让 Indexer 继续逼近 attention teacher；
- 离散 Top-K 不负责传递梯度，Indexer 也不会被 LM loss 间接训练。

一句话：**backbone 和 Indexer 都训练，但各自使用不同的损失和梯度路径。**

## 2. Indexer 到底做什么

GLM-5.2 的 attention 分成两条支路：

```text
hidden states
  ├─ Indexer: 给每个 query 的所有历史 key 打分，选出 top-k 位置
  │              └─ topk_indices [token, 1, K]
  │
  └─ MLA: 只在 Indexer 选中的 K 个位置上计算 sparse attention
                 └─ attention output -> 后续 LM loss
```

以当前 GLM-5.2 配置为例：

| 参数 | 含义 | 典型值 |
|---|---|---:|
| `hidden_size` | 主模型 hidden 维度 | 6144 |
| `q_lora_rank` | Indexer 和 MLA 共用的低秩 Q 输入 | 2048 |
| `index_n_heads` | Indexer 的打分头数 | 32 |
| `index_head_dim` | 每个 Indexer head 的维度 | 128 |
| `index_topk` | 每个 query 最终保留的历史 token 数 | 2048 |

XTuner 的数据流可以简化为：

```text
q_resid [1,S,2048] --wq_b--> Q_index [1,S,32,128]
hidden  [1,S,6144] --wk+LN--> K_index [1,S,128]
hidden  [1,S,6144] --weights_proj--> head weights [1,S,32]

ReLU(Q_index @ K_index^T) * head weights
  -> index_scores [1,S,S_global]
  -> causal/packing mask
  -> top-k
  -> topk_indices [S,1,2048]
```

`topk_indices` 保存的是 token 位置，例如 query 位置 5000 可能选择
`[0, 37, 420, ..., 4999]`，不是选 2048 个 hidden 维度，也不是 MoE expert 路由。

相关代码：

- `xtuner/v1/module/attention/dsa_mla.py:60-167`：`DSAIndexer` 与 Q/K/weight 数据流。
- `xtuner/v1/module/attention/dsa_mla.py:279-383`：Indexer top-k 如何送入 SparseMLA。
- 详细 shape 图可继续阅读
  [GLM-5.2 DSA / MLA 技术附录](./GLM-5.2-DSA-MLA-技术附录.md)。

## 3. 为什么普通 LM loss 训练不到 Indexer

普通 SFT 的目标是 next-token cross entropy：

```text
logits -> cross entropy -> LM loss
```

Indexer 的输出却是离散整数：

```text
index_scores -> topk() -> integer topk_indices
```

`topk_indices` 只是告诉 SparseMLA 去取哪些 token。整数下标没有连续导数，因此下面这条
梯度路径在 top-k 处断开：

```text
LM loss -> SparseMLA -> topk_indices -X-> index_scores -> Indexer weights
```

这就是为什么“让 Indexer 参数 `requires_grad=True`”并不等于“Indexer 会被训练”。要训练
Indexer，损失必须直接作用在 top-k 之前的 `index_scores` 上：

```text
dense attention teacher distribution
                  │
                  v
             KL divergence
                  ^
                  │
         softmax(index_scores)
```

## 4. 需要分清的三类训练

### 4.1 Indexer warm-up：只训练 Indexer

这是把一个已有 dense MLA 模型转换为 DSA 时的第一阶段，不是常规业务 SFT。

```text
同一份 hidden states
  |                          |
  |                          +-> Indexer -> index_scores -> q
  |
  +-> dense MLA teacher -> dense attention distribution -> p

p 与 q -> KL loss -> 反向传播 -> 只更新 Indexer

backbone: 冻结
dense MLA: 只负责生成 teacher
LM loss: 不参与这一阶段
```

参考 DSA 方法会把各 attention head 的 dense attention 分布聚合并归一化，得到 teacher
分布 `p`，再最小化：

```text
L_indexer = KL(p_dense_attention || softmax(index_scores))
```

GLM-5 技术报告给出的 DSA warm-up 为 1000 step。报告中的小规模验证也明确说明 warm-up
阶段只训练 Indexer、冻结 base model；之后再进行联合训练。这里的 1000 step 是论文训练
方案，不应直接套到本项目 6 层 smoke checkpoint 上。

### 4.2 Sparse adaptation：主模型和 Indexer 联合训练

第二阶段开始真正使用 top-k sparse attention：

```text
路径 1: hidden -> Indexer Top-K -> Sparse Attention -> LM loss -> 主模型参数
路径 2: detach(hidden) -> Indexer scores -> KL loss -> Indexer 参数
```

参考 DSA 方案会 detach Indexer 的输入，使两套梯度彼此独立：

- LM loss 不借 Indexer 分支更新主模型；
- Indexer KL 不通过输入反向修改主模型；
- Indexer 只由 KL loss 更新；
- 主模型只由 LM loss 更新，并逐步适应 sparse attention。

这不是简单的 `LM loss + KL loss` 全图混合反传，而是两条受控的梯度路径。

### 4.3 已发布 GLM-5.2 的普通 SFT：冻结 Indexer

本项目现有 200-step 对齐实验属于这一类。起点已经是完成 DSA/IndexShare 适配的
GLM-5.2 checkpoint，训练目的只是让模型学习 SFT 数据，而不是重新发明 sparse pattern。

```text
Indexer 权重从 HF checkpoint 加载
        |
        v
Indexer 正常计算 Top-K
        |
        v
Sparse Attention 使用这些 Top-K
        |
        v
LM loss
        |
        +----> 更新 attention、MoE、embedding 等主模型参数
        |
        +--X-> Indexer 参数没有梯度，不发生更新
```

冻结的好处是行为清晰、减少 optimizer state，并避免业务 SFT 的短数据或短序列破坏已经
训练好的长上下文选择器。代价是主模型 hidden states 在长时间或大步长 SFT 后可能漂移，
而 Indexer 不会同步适配。因此大规模领域继续训练仍应检查 top-k 稳定性和长上下文能力。

## 5. 最新 XTuner 的真实实现

### 5.1 SFT recipe 只有 LM loss

`examples/v1/config/sft_glm5p2.py` 当前执行：

- 从 HF checkpoint 构建 GLM-5.2；
- 配置 `CELossConfig`；
- 选择 sparse MLA backend；
- 配置 AdamW/Muon、FSDP、EP、SP 和 checkpoint。

这里没有 Indexer KL loss、dense attention teacher、Indexer 专用 optimizer 或 warm-up/joint
stage scheduler。对应代码在 `examples/v1/config/sft_glm5p2.py:35-130`。

### 5.2 Indexer 被显式冻结

当前代码为：

```python
self.requires_grad_(False)

@torch.no_grad()
def forward(...):
    ...
```

位置：`xtuner/v1/module/attention/dsa_mla.py:86-91`。

AdamW 构建时只接收 `requires_grad=True` 的参数，因此 Indexer 不会进入 optimizer，也不会
创建 AdamW 的一阶/二阶动量状态：`xtuner/v1/config/optim.py:37-55`。

该行为由 XTuner commit `8e2f10f2` 明确引入，提交说明为：

```text
[GLM-5.2] Fix freezed Indexer dcp load.
[Fix] Freeze GLM-5.2 DSA indexer parameters
```

### 5.3 backend 当前只返回整数 top-k

`DSATopKIndicesProtocol` 的输出只有 `torch.Tensor` 类型的 top-k IDs。PyTorch fallback
虽然在内部计算了 `index_scores` 和 `topk_scores`，最终只返回 `topk_indices`：

- `xtuner/v1/ops/sparse_mla/protocol.py:37-55`
- `xtuner/v1/ops/sparse_mla/pytorch.py:50-71`

这对普通 SFT 足够，却无法计算 Indexer KL loss。Indexer 训练至少需要让 backend 提供
可微分 `index_scores`，或者提供一个专门的 training-score/recompute 接口。

### 5.4 IndexShare：`full` 层计算，`shared` 层复用

`indexer_types` 中：

- `full`：该层真实拥有 `wq_b/wk/k_norm/weights_proj`，并计算新 top-k；
- `shared`：该层不创建 Indexer 参数，复用前一个 `full` 层缓存的 top-k。

例如：

```text
layer 0: full    -> 计算 topk_0
layer 1: shared  -> 复用 topk_0
layer 2: shared  -> 复用 topk_0
layer 3: shared  -> 复用 topk_0
layer 4: full    -> 计算 topk_4，覆盖当前缓存
```

可以把它理解成：

```text
layer 0 (full，拥有 Indexer 权重)
        |
        +-> 计算并缓存 topk_0
                    |
                    +-> layer 1 (shared，无 Indexer 权重) 复用
                    +-> layer 2 (shared，无 Indexer 权重) 复用
                    +-> layer 3 (shared，无 Indexer 权重) 复用

layer 4 (full，拥有另一套 Indexer 权重)
        |
        +-> 重新计算 topk_4，并成为后续 shared 层的新来源
```

所以 `shared` 的含义不是“拥有一份相同的 Indexer 权重”，而是“根本没有 Indexer 参数，
直接复用前面 `full` 层计算出的 Top-K 结果”。

XTuner 会根据 `indexer_types` 或 `index_topk_freq/index_skip_topk_offset` 找到 source layer，
并为 activation checkpoint recompute、MTP 重复使用和可选 CPU offload 管理 top-k 生命周期：

- source layer 解析：`xtuner/v1/module/attention/dsa_topk_sharing.py:30-52`
- cache/recompute 复用：`dsa_topk_sharing.py:227-290`
- top-k offload 开关：`XTUNER_DSA_TOPK_OFFLOAD=1`

GLM-5.2 官方介绍称 IndexShare 会跨多个 sparse attention layer 复用 Indexer，从而降低长
上下文中的 Indexer 开销。它共享的是 top-k 下标，不是共享整层 attention 权重或 KV cache。

### 5.5 为什么 shared 层缺少 `indexer.*` 权重是正常的

如果 config 标记某层为 `shared`，该层在结构上就没有 Indexer，因此 checkpoint 中不会有：

```text
model.layers.<shared-layer>.self_attn.indexer.wq_b.weight
model.layers.<shared-layer>.self_attn.indexer.wk.weight
model.layers.<shared-layer>.self_attn.indexer.k_norm.*
model.layers.<shared-layer>.self_attn.indexer.weights_proj.weight
```

加载器若无视 `indexer_types`、强行给每层创建 Indexer，就会把这些“设计上不存在的权重”
误报为 missing keys。正确判断方式是先看该层为 `full` 还是 `shared`。

### 5.6 frozen Indexer 的 checkpoint 语义

XTuner 的默认配置 `dcp_ignore_frozen_params=True`，DCP 收集 state dict 时会忽略冻结参数：

- 默认值：`xtuner/v1/model/base.py:147`
- DCP options：`xtuner/v1/engine/train_engine.py:323-336`
- resume 前先加载 HF base：`xtuner/v1/train/trainer.py:1170-1176`

因此两类 checkpoint 要分开理解：

| 格式 | Indexer 权重 | 用途 |
|---|---|---|
| HF export | 应包含所有 `full` 层 Indexer 权重 | 独立部署、交换、下游评测 |
| DCP training checkpoint | 默认可省略 frozen Indexer | 依赖同一份 HF base 后再恢复训练状态 |

DCP 不是“丢了 Indexer”，而是采用“先从相同 base 恢复冻结参数，再从 DCP 恢复训练参数与
optimizer”的语义。若只搬走 DCP 而没有对应 base checkpoint，恢复链路就不完整。

恢复流程可以直接看成：

```text
原始 HF base checkpoint
  -> 恢复主模型初始权重
  -> 恢复被冻结的 Indexer 权重
                    |
                    v
            加载 DCP checkpoint
              -> 覆盖训练过的主模型参数
              -> 恢复 optimizer / scheduler / step
              -> 默认不覆盖 frozen Indexer
                    |
                    v
             得到完整的续训状态
```

因此，单独的 DCP 更像“相对于原始 HF base 的训练状态包”，不是一份能够完全替代 HF base
的独立模型目录。

## 6. XTuner 与 AutoModel 对照

| 项目 | XTuner `9dabb956` | AutoModel `e2c48886` | 判断 |
|---|---|---|---|
| DSA Indexer forward | 支持 | 支持 | 两边都有 |
| IndexShare full/shared | 支持，并管理 cache 生命周期 | 支持，通过 `prev_topk_indices` 逐层传递 | 语义一致，runtime 组织不同 |
| 普通 SFT 的 Indexer freeze | 显式 `requires_grad_(False)` + `no_grad` | 当前 Indexer 类未显式冻结，也没有 `no_grad` | 行为不完全一致 |
| LM CE 对 Indexer 的有效梯度 | 无 | 正常 top-k 路径也应为无 | 整数 index 切断梯度 |
| Indexer KL loss | 无 | 无 | 两边都不能直接做 Indexer warm-up |
| dense teacher attention | 无对应 recipe | 无对应 recipe | 两边都需开发 |
| HF/分片 checkpoint | 有 HF round-trip 与 DCP 测试 | 有 state-dict adapter | 都有基础能力 |
| Indexer RoPE | interleaved，测试 oracle 对齐 HF PR #46842 | 当前代码为 half-split | 需要做 top-k 数值复核 |

AutoModel 的 `GlmMoeDsaIndexer` 位于
`nemo_automodel/components/models/glm_moe_dsa/layers.py:155-364`。当前类中没有 XTuner
同类的显式冻结操作；标准 finetune recipe 也未发现 Indexer KL loss。由于其最终仍只把
整数 top-k 交给 attention，普通 CE loss 理论上不能训练 Indexer，但是否产生额外 FSDP、
optimizer 参数管理开销，应通过运行时 `requires_grad/grad/optimizer.state` 检查确认。

另一个需要单独验证的差异是 RoPE：

- XTuner `dsa_mla.py:134-136` 明确使用 interleaved Indexer RoPE，并在测试中 patch HF
  numeric oracle 对齐 HF PR #46842；
- AutoModel `layers.py:251-264` 当前明确使用 non-interleaved half-split RoPE。

RoPE 会改变 Indexer Q/K，从而可能改变 top-k。这里可以确认“源码路径不同”，但在没有用
同一权重和输入做 top-k overlap 前，不应直接写成 AutoModel 数值错误。

## 7. 如果要在 XTuner 真正支持 Indexer SFT，需要改什么

### 7.1 建议的模式开关

建议不要复用一个模糊的 `TRAIN_INDEXER=1`，而是明确区分：

```python
indexer_train_mode: Literal["frozen", "warmup", "joint"] = "frozen"
indexer_loss_weight: float = 1.0
indexer_lr: float | None = None
indexer_teacher: Literal["dense_attention"] = "dense_attention"
```

| 模式 | backbone | Indexer | attention | loss |
|---|---|---|---|---|
| `frozen` | train | freeze | sparse | LM |
| `warmup` | freeze | train | dense teacher | Indexer KL |
| `joint` | train | train | sparse | LM + 独立 Indexer KL |

默认必须保持 `frozen`，以兼容现有 GLM-5.2 下游 SFT 和 checkpoint。

### 7.2 模型与算子接口

当前 `DSATopKIndicesProtocol` 只返回 IDs，建议引入训练专用输出：

```python
class DSAIndexerOutputs(NamedTuple):
    topk_indices: torch.Tensor
    index_scores: torch.Tensor | None
```

需要处理：

1. PyTorch backend 返回带梯度的 `index_scores`。
2. TileLang/cuDNN backend 要么返回 score，要么提供可重算 score 的训练接口。
3. 普通 `frozen` 模式仍只返回 IDs，避免保存巨大的 `[S,S]` score。
4. packed sequence 的 causal mask、序列边界和 SP global-K 必须与现有 top-k 完全一致。

最大的工程风险是显存。`index_scores` 为 `[S,S]`，16K 序列的 FP32 单层矩阵约 1 GiB，
还没算 head 中间值和多层 teacher。不能直接让所有层同时保存完整 score，需要 chunk、重算、
逐层消费或专用 fused loss。

### 7.3 teacher attention 与 Indexer loss

warm-up 阶段需要 dense attention teacher；joint 阶段至少需要被选 token 上的 teacher
distribution。当前 absorbed SparseMLA 只返回 attention output 和 `softmax_lse`，不提供完整
teacher distribution，因此还需新增：

- dense MLA teacher 路径或等价 score recompute；
- 跨 head 聚合与 L1 normalize；
- causal/packing-aware KL；
- joint 阶段 selected-set KL；
- teacher target `detach()`；
- Indexer 输入与主模型计算图解耦。

### 7.4 IndexShare-aware loss

GLM-5.2 的 `full` Indexer 要服务后续多个 `shared` 层。只让它拟合自己的 attention
distribution，不一定能为 shared 层选好 token。

IndexCache 的 training-aware 方法对一个 `full` Indexer 使用多层蒸馏：

```text
full layer l 的 Indexer
  <- layer l 的 attention teacher
  <- layer l+1 shared 的 attention teacher
  <- layer l+2 shared 的 attention teacher
  <- layer l+3 shared 的 attention teacher
```

即让该 Indexer 拟合它所服务层的平均/多项 attention 分布。若目标是复现 GLM-5.2 的
IndexShare 训练语义，这部分不能省略。

### 7.5 optimizer 与 checkpoint

建议为 Indexer 建立独立 param group：

```text
backbone group: main_lr，仅接收 LM loss
indexer group:  indexer_lr，仅接收 KL loss
```

进入 `warmup/joint` 后，`dcp_ignore_frozen_params` 的语义也会变化：

- `warmup`：backbone frozen，可从 base 恢复；Indexer 必须进入 DCP。
- `joint`：两者都 trainable，都必须进入 DCP 与 optimizer state。
- 模式切换 resume 时必须检查 checkpoint 中的 param group 与训练阶段。

## 8. 推荐的验证顺序

### P0：确认现有 frozen SFT 没有回归

1. 所有 `*.indexer.*` 参数均 `requires_grad=False`。
2. optimizer param groups 中没有 Indexer。
3. 训练前后 Indexer 权重逐 tensor 完全相同。
4. HF export 后，所有 `full` 层 Indexer 权重存在，`shared` 层不存在。
5. HF round-trip 和 DCP resume 后 loss/grad 曲线连续。

### P1：做 Indexer 数值 parity

固定同一 checkpoint、同一 packed 输入，比较 XTuner、AutoModel 和 HF：

| 指标 | 用途 |
|---|---|
| `index_scores` max/mean error | 定位 Q/K/RoPE/scale 差异 |
| top-k overlap / Jaccard | 判断选中的 token 是否一致 |
| 每层 top-k 越界/跨 pack 比例 | 检查 causal 与 packing mask |
| full/shared cache 命中 | 检查 IndexShare 生命周期 |
| sparse attention output error | 判断 top-k 差异是否传到模型输出 |

应优先验证 interleaved 与 half-split RoPE 差异，不能只比较最终 200-step loss 后再猜原因。

### P2：最小 Indexer warm-up

先用 tiny 模型、短序列和 PyTorch backend 验证：

1. backbone grad 全为 `None`；
2. Indexer grad finite 且非零；
3. KL loss 连续下降；
4. Indexer top-k 对 dense teacher top-k 的 recall/overlap 上升；
5. DCP/HF round-trip 后指标不变。

### P3：joint 与 IndexShare

1. LM loss 只更新 backbone 路径，KL 只更新 Indexer 路径。
2. full layer 的多层蒸馏 loss 正确覆盖对应 shared layers。
3. activation checkpoint、compile、SP、MTP 下 cache 生命周期无泄漏。
4. 与 frozen SFT 对比长上下文 eval loss、RULER/NIAH 和吞吐。

## 9. 当前风险与建议

| 风险 | 影响 | 建议 |
|---|---|---|
| 把普通 SFT 误称为 Indexer SFT | 误以为 Indexer 已被训练 | 汇报时明确称“冻结 Indexer 的 GLM-5.2 SFT” |
| 只打开 `requires_grad` | Indexer 仍无有效梯度 | 先实现 score + KL teacher loss |
| 保存完整 `[S,S]` scores | 长序列显存爆炸 | chunk/recompute/fused loss |
| 忽略 IndexShare 的多层目标 | full Indexer 只适合自己层 | 使用多层蒸馏或先做训练-free pattern 验证 |
| XTuner/AutoModel RoPE 路径不同 | top-k 与最终 loss 可能偏离 | 建立 score/top-k parity 测试 |
| DCP 忽略 frozen 参数 | 单独搬 DCP 无法完整恢复 | 固定并记录 base HF checkpoint |

建议当前先保持 XTuner 的 frozen Indexer 行为，不直接改训练主链。下一步最有价值的工作不是
马上跑大规模 Indexer 训练，而是先补一套小模型的 `index_scores/top-k overlap` parity 工具，
确认 XTuner、HF 与 AutoModel 的 RoPE、mask、score scale 和 IndexShare 均一致。数值基线稳定
后，再按 `warmup -> joint -> IndexShare-aware` 的顺序开发。

## 10. 资料与源码索引

### 论文与官方资料

- [GLM-5 Technical Report](https://arxiv.org/abs/2602.15763)：GLM-5 DSA 两阶段训练、
  1000-step warm-up、sparse adaptation 和 SFT 对比。
- [DeepSeek-V3.2-Exp Technical Report](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)：
  Indexer KL 目标、dense warm-up、joint sparse training 和梯度解耦。
- [IndexCache / IndexShare](https://arxiv.org/abs/2603.12201)：full/shared 层、training-free pattern
  与多层蒸馏目标。
- [GLM-5.2 官方仓库](https://github.com/zai-org/GLM-5)：GLM-5.2 与 IndexShare 说明。

### XTuner `9dabb956`

| 内容 | 文件 |
|---|---|
| Indexer 与 SparseMLA 数据流 | `xtuner/v1/module/attention/dsa_mla.py` |
| IndexShare cache/recompute/offload | `xtuner/v1/module/attention/dsa_topk_sharing.py` |
| top-k backend protocol | `xtuner/v1/ops/sparse_mla/protocol.py` |
| PyTorch score/top-k 实现 | `xtuner/v1/ops/sparse_mla/pytorch.py` |
| GLM-5.2 HF config/key mapping/MTP | `xtuner/v1/model/moe/glm52.py` |
| 普通 GLM-5.2 SFT recipe | `examples/v1/config/sft_glm5p2.py` |
| optimizer 参数过滤 | `xtuner/v1/config/optim.py` |
| frozen-param DCP | `xtuner/v1/engine/train_engine.py` |
| HF/DCP/loss regression | `tests/engine/test_glm52_moe_train_engine.py` |
| HF numeric oracle patch | `xtuner/_testing/glm52_hf.py` |

### NeMo AutoModel `e2c48886`

| 内容 | 文件 |
|---|---|
| GLM DSA Indexer/MLA | `nemo_automodel/components/models/glm_moe_dsa/layers.py` |
| IndexShare 跨层/PP 传递 | `nemo_automodel/components/models/glm_moe_dsa/model.py` |
| HF state-dict 与 Indexer 量化处理 | `nemo_automodel/components/models/glm_moe_dsa/state_dict_adapter.py` |

## 11. 可直接用于汇报的一段话

> 目前 XTuner 已支持带 DSA/IndexShare 的 GLM-5.2 常规 SFT，但这里是“使用 Indexer”，
> 不是“训练 Indexer”。最新代码会显式冻结 Indexer，并让它在 no-grad 下产生离散 top-k；
> 主模型继续由 LM loss 更新。真正的 Indexer warm-up 需要保留 dense attention 作为 teacher，
> 用 top-k 前的可微分 score 计算 KL loss；联合 sparse adaptation 还要把主模型 LM loss 与
> Indexer KL 的梯度分开。当前 XTuner 和 AutoModel 都没有这套完整 recipe。XTuner 若要补齐，
> 核心改动是 score backend、teacher attention、KL loss、独立参数组、IndexShare 多层蒸馏和
> checkpoint stage 兼容，而不是简单解除冻结。
