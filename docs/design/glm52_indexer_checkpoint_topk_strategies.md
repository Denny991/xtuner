# GLM-5.2 Indexer：三种 checkpoint / Top-K 策略对照

本文只回答一个问题：开启 activation checkpoint 后，Indexer 的 `project`、Top-K、
fixed-ID KL 和 backward 分别会执行几次、使用哪一份 IDs。

本文只讨论 GLM-5.2 production recipe 采用的 **sparse Indexer KL**：Student 与 Teacher 在
Top-K IDs 指定的相同 support 上比较。Dense Indexer loss 会在全部合法 Key 上比较，IDs 不
决定它的 KL support，不属于本文范围。

为避免概念交织，下面把三种方案完全分开介绍：

1. Megatron 的 joint Indexer training；
2. PR #2039 的 frozen Indexer；
3. XTuner joint SFT 的建议方案。

完整算法设计见 [glm52_indexer_sft.md](./glm52_indexer_sft.md)，基础概念见
[glm52_indexer_sft_beginner_guide.md](./glm52_indexer_sft_beginner_guide.md)。本文只讲
activation checkpoint，不讨论保存到磁盘的 DCP/HF checkpoint。

---

## 0. 先固定五个术语

本文的调用次数统一按“每个 source/computing layer、每个 micro-batch、每次 checkpoint
invocation”统计。Shared layer 只消费 source IDs，不计入 Indexer project/Top-K 次数。

另外，下面两条运行路径互斥，不能把次数加在一起：

~~~text
不开 activation checkpoint：只走 normal forward
开启 activation checkpoint：走 original forward + replay + backward
~~~

`project` 次数只统计 Indexer 投影真正执行的次数，不包括主 MLA 投影；Top-K 次数只统计
selector/kernel 真正执行的次数。Replay 虽然可以再次进入复用 wrapper 的调用点，但若它只从
frame 取回 IDs，就不算再次执行 Top-K。Custom backward 重算连续 score 也不算 Top-K。

### 0.1 `project`

`project` 根据本层输入和 Indexer 参数生成连续特征。真实实现还会消费 position embeddings、
`SequenceContext`，并按并行布局准备或 gather global K：

~~~text
hidden_states / q_resid
          ↓
    Indexer project
          ↓
index_q、index_k、head_weights
~~~

它不接收 Top-K IDs。忽略 RoPE、packed/SP 和 global-K 通信后，可以简化理解为：

~~~python
index_q = wq_b(q_resid)
index_k = wk(hidden_states)
head_weights = weights_proj(hidden_states)
~~~

### 0.2 `select_topk`

`select_topk` 使用 `index_q/index_k/head_weights` 计算所有合法 Query-Key 分数，再挑出
每个 Query 分数最高的 K 个 Key：

~~~text
index_q / index_k / head_weights
              ↓
dense selection logits [S_q, S_k]
              ↓
            Top-K
              ↓
topk_ids [S_q, K]，整数、不可导
~~~

### 0.3 `fixed-ID KL`

IDs 已知后，sparse KL 的**数学 support**固定为这些位置：

~~~text
新的可导 index_q/index_k/head_weights + 固定 IDs
                         ↓
              Student logits [S_q,K]

同层主 MLA Q/K + 相同固定 IDs
                         ↓
              Teacher probability [S_q,K]

KL(Teacher.detach() || Student)
~~~

IDs 只负责指定位置；梯度经过 Student 的连续分数回到 Indexer 参数，不经过整数 IDs。

这里的 `[S_q,K]` 是最终参与 sparse KL 的逻辑形状，不保证每个 backend 的中间 workspace
都只有 `[S_q,K]`：production streaming/fixed-ID kernel 可以只算 selected positions；某些
PyTorch/reference/custom backward 也可能临时重算 `[S_q,S_k]` dense score，再按 IDs gather
或 mask。两种实现的 KL support 相同，但显存和计算量不同。

### 0.4 `original forward` 与 `checkpoint replay`

Reentrant activation checkpoint 的时间线是：

~~~text
original forward（no-grad，不保存内部计算图）
          ↓
后续网络与最终 loss
          ↓
backward 到达 checkpointed layer
          ↓
checkpoint replay（grad-enabled，重跑 forward 并建立计算图）
          ↓
backward
~~~

`original forward` 虽然是 no-grad，仍然必须算出 Layer 输出；MLA 和可训练参数会在 replay
中重新执行并正常获得梯度。

### 0.5 `custom backward`

某些 Indexer loss backend 使用自定义 autograd op。它的 backward 可能只重算 selected
score，也可能先重算 dense score 再按 IDs 限定 support，或者直接计算 `dQ/dK/dWeight`。
具体 workspace 取决于 backend；共同点是 fixed IDs 已确定 sparse KL support，所以它不会
再次决定 Top-K。

---

## 1. Megatron：replay 重新选择 Top-K

### 1.1 方案目标

这是可训练 Indexer 的方案。source/computing layer 的 Indexer 参数接收 KL 梯度。

### 1.2 不使用 activation checkpoint 时

~~~text
grad-enabled forward:
  project
  → select_topk，得到 IDs
  → 在这些 IDs 上计算 Student / Teacher / KL
  → backward 使用这份 IDs 计算 Indexer 梯度
~~~

每个 source/computing layer、每个 micro-batch：

| 操作 | 真正执行次数 |
|---|---:|
| `project` | 1 |
| Top-K selector | 1 |
| Indexer backward | 1 |

### 1.3 使用 reentrant activation checkpoint 时

#### Megatron original forward

~~~text
hidden_states
  → project（no-grad）
  → select_topk
  → original IDs
  → Sparse MLA
  → Layer output
~~~

这次 forward 用于得到 Layer 输出，不保留可导 Indexer 图。

#### Megatron checkpoint replay

~~~text
hidden_states
  → 重新 project（grad-enabled）
  → 重新计算 dense selection logits
  → 再执行一次 Top-K
  → replay IDs
  → Student / Teacher / KL
~~~

这里使用的是 replay 新产生的 IDs，而不是依赖 original IDs 构造可导图。

#### Megatron Indexer loss backward

~~~text
replay IDs 已经确定
  → 按 backend 实现重算连续 score / gradient
  → sparse KL support 仍由这些 IDs 固定
  → dQ、dK、dWeight
  → 更新 Indexer 参数
~~~

自定义 backward 不会再执行第三次 Top-K。它只处理 replay 已经选好的 support。

### 1.4 Megatron 完整时间线

~~~text
ORIGINAL（no-grad）
  project #1
  Top-K #1
  得到 IDs_A

REPLAY（grad-enabled）
  project #2
  Top-K #2
  得到 IDs_B
  fixed-ID KL（使用 IDs_B）

CUSTOM BACKWARD
  固定 IDs_B
  计算 sparse-loss gradient
  不再 Top-K
~~~

调用次数：

| 操作 | 真正执行次数 |
|---|---:|
| `project` | 2 |
| Top-K selector | 2 |
| Indexer backward | 1 |

理想情况下 `IDs_A == IDs_B`。但若存在 near-tie 或 kernel 数值波动，二者可能不同；若要
采用这项策略，应测试 original/replay IDs 是否逐值一致，并定义不一致时的失败语义。

### 1.5 一句话记忆

> Megatron 在 checkpoint replay 中把可训练 Indexer forward 整体重跑一遍，因此 project 和
> Top-K 都重算；随后 custom backward 固定 replay IDs，只计算连续梯度。

---

## 2. PR #2039：frozen Indexer 只执行一次

### 2.1 方案目标

PR #2039 当前面向 frozen Indexer：

~~~text
Indexer 参数 requires_grad=False
Indexer 不进 optimizer
LM loss 不更新 Indexer
~~~

它解决的是：checkpoint replay 时不要重复执行一个不需要梯度的 Indexer/Top-K。

### 2.2 显式 IDs 数据流

source layer 无论输入中是否带有旧 IDs，都必须执行本层 Indexer，产生并覆盖新的
`dsa_topk_ids`；shared layer 必须接收上一 source layer 的 IDs，不执行自己的 Indexer：

~~~text
source layer
  → Indexer / Top-K
  → 输出 dsa_topk_ids
  → shared layer #1
  → shared layer #2
  → 下一个 source layer 生成并覆盖新 IDs
~~~

IDs 是 decoder layer 的显式输入/输出，不再隐藏在 `SequenceContext` mutable cache 中。

### 2.3 不使用 activation checkpoint 时

~~~text
source layer:
  frozen Indexer project
  → Top-K
  → IDs
  → Sparse MLA

shared layer:
  直接使用 source IDs
  → Sparse MLA
~~~

没有 Indexer backward。

### 2.4 使用 reentrant activation checkpoint 时

#### PR #2039 original forward

~~~text
with torch.no_grad():
  reuse_during_recompute(self.indexer, ...)
      → 真正执行 frozen Indexer project
      → 真正执行 Top-K
      → 将 Indexer output / IDs 放入当前 checkpoint frame 的 FIFO
~~~

#### PR #2039 checkpoint replay

~~~text
reuse_during_recompute(self.indexer, ...)
  → 不调用 self.indexer
  → 从当前 checkpoint frame 的 FIFO 取回 original output / IDs
~~~

主 MLA 仍会 replay 并通过 LM loss 获得梯度；只有 frozen Indexer 被跳过。

#### PR #2039 backward

~~~text
LM loss
  → 主 MLA / 主模型参数梯度

Indexer:
  无 KL
  无 backward
  无参数更新
~~~

### 2.5 PR #2039 完整时间线

~~~text
ORIGINAL（no-grad）
  frozen project #1
  Top-K #1
  保存 IDs_A 到 frame

REPLAY（grad-enabled）
  从 frame 取回 IDs_A
  不 project
  不 Top-K
  主 MLA 正常 replay

BACKWARD
  主模型更新
  Indexer 不更新
~~~

调用次数：

| 操作 | 真正执行次数 |
|---|---:|
| Indexer `project` | 1 |
| Top-K selector | 1 |
| Indexer backward | 0 |

### 2.6 一句话记忆

> PR #2039 的 Indexer 是 frozen 的，所以 original 算一次并保存 IDs；replay 直接取回，不需要
> 为 Indexer 建图，也没有 Indexer backward。

---

## 3. XTuner joint SFT 建议：重算 project，复用 original IDs

### 3.1 方案目标

目标同时满足：

1. replay 中重新建立可导 Indexer project 图；
2. original/replay 使用完全相同的 sparse support；
3. dense selection logits 和 Top-K 只执行一次；
4. KL 只更新 source/full Indexer，不通过 Teacher 污染主模型。

### 3.2 不使用 activation checkpoint 时

~~~text
grad-enabled forward:
  可导 project
  → no-grad select_topk
  → fixed-ID Student / Teacher / KL
  → AuxLossScaler attach
  → backward 更新 Indexer
~~~

Top-K IDs 是整数、不可导，不妨碍 KL 通过 Student 的连续 score 回到 project 参数。

### 3.3 使用 reentrant activation checkpoint 时

#### 建议方案 original forward

~~~text
hidden_states
  → project（整层 original 为 no-grad）
  → reuse_during_recompute(selector, detached features, ...)
      → 真正执行一次 Top-K
      → 保存完整 original IDs 到 checkpoint frame
  → fixed-ID KL 只产生 detached 展示值
  → Layer output
~~~

这里 original 的 project 结果没有可导图，不长期保存；只保存离散 IDs。

#### 建议方案 checkpoint replay

~~~text
hidden_states
  → 重新 project（grad-enabled）
  → 得到新的可导 index_q/index_k/head_weights

checkpoint frame
  → 取回 original IDs

可导 index_q/index_k/head_weights + original IDs
  → 按 backend 计算 sparse Student / Teacher
  → 最终 KL support 为 original IDs [S_q,K]
  → fixed-ID KL
  → AuxLossScaler attach
~~~

上面描述的是数学数据流。目标 production adapter 应流式计算 selected positions；reference 或
custom backward 允许临时重算 dense score 后再按 IDs gather/mask，但不能重新选择 Top-K。

注意：不是“根据 IDs 重新 project”。`project` 还会使用 position/sequence context 和并行
信息，但不接收 IDs；IDs 在 project 之后用于限定 sparse KL support。

#### 建议方案 backward

~~~text
fixed-ID KL
  → Student sparse-loss gradient
  → replay index_q/index_k/head_weights
  → wq_b / wk / weights_proj / k_norm
  → 更新 Indexer

original IDs:
  只决定 support
  没有梯度
~~~

如果 loss backend 使用 custom backward，它仍然固定 original IDs。它可以只重算 selected
score，也可以重算 dense score 后按 IDs 限定 support，但不能再次执行 Top-K。

### 3.4 建议方案完整时间线

~~~text
ORIGINAL（no-grad）
  project #1
  Top-K #1
  保存 IDs_A

REPLAY（grad-enabled）
  project #2
  复用 IDs_A
  不再 Top-K
  fixed-ID KL（使用 IDs_A）

BACKWARD
  固定 IDs_A
  计算 sparse-loss gradient
  更新 Indexer
~~~

调用次数：

| 操作 | 真正执行次数 |
|---|---:|
| `project` | 2 |
| Top-K selector | 1 |
| Indexer backward | 1 |

### 3.5 为什么不复用整个 original Indexer output

Original 在 no-grad 下产生的 `index_q/index_k/head_weights` 没有 `grad_fn`，无法把 KL 梯度
传回 Indexer 参数。长期保存完整可导中间量也违背 activation checkpoint 的省显存目标。

所以 joint 模式只能复用无梯度、离散、相对紧凑的 IDs；连续 project 必须在 replay 重算。

### 3.6 为什么不在 replay 再做一次 Top-K

- 保证 original/replay 使用相同 sparse support；
- 避免 near-tie 或算子波动导致 IDs 改变；
- 少物化一次 `[S_q,S_k]` dense selection logits；
- 少执行一次 Top-K；
- Top-K 本身不可导，重算并不会提供额外梯度。

### 3.7 一句话记忆

> 建议的 XTuner joint 方案只重算“需要梯度的连续 project”，复用“不可导的离散 IDs”，然后
> 在 original IDs 指定的范围内计算可导 KL。

---

## 4. 三种方案横向对比

下面调用次数均指每个 source/computing layer、每个 micro-batch，并假设启用 reentrant
activation checkpoint。

| 项目 | Megatron joint | PR #2039 frozen | 建议的 XTuner joint |
|---|---|---|---|
| Indexer 是否训练 | 是 | 否 | 是 |
| original project | 执行 | 执行 | 执行 |
| original Top-K | 执行 | 执行并保存 IDs | 执行并保存 IDs |
| replay project | 重新执行 | 不执行 | 重新执行 |
| replay Top-K | 重新执行 | 不执行 | 不执行 |
| replay 使用的 IDs | replay 新 IDs | original IDs | original IDs |
| fixed-ID KL | replay 计算 | 无 | replay 计算 |
| Indexer backward | 有 | 无 | 有 |
| project 总次数 | 2 | 1 | 2 |
| Top-K 总次数 | 2 | 1 | 1 |
| original/replay support | 可能变化，需验证 | 相同 | 相同 |

最短记忆：

~~~text
Megatron joint：      project 两次，Top-K 两次，训练 Indexer
PR #2039 frozen：     project 一次，Top-K 一次，不训练 Indexer
建议 XTuner joint：  project 两次，Top-K 一次，训练 Indexer
~~~

---

## 5. 模型层 IDs 数据流与 backend 接口要分开

### 5.1 模型层：显式传 `dsa_topk_ids`

这一层负责：

~~~text
source layer 产生新 IDs
  → decoder 显式传给 shared layer
  → shared layer 原样透传
  → 下一个 source layer 覆盖为新 IDs
~~~

Checkpoint frame 也只负责在同一次 checkpoint invocation 的 original/replay 之间保存和取回
IDs。它不负责决定 Torch、TileLang 或 cuDNN 具体怎样算 Top-K/KL。

### 5.2 Backend 层：怎样产生 IDs、怎样计算 fixed-ID loss

不同 backend 仍需要统一抽象：

~~~python
class DSAIndexerBackendProtocol(Protocol):
    def select_topk(...) -> Tensor:
        """产生不可导的 int32 IDs。"""

    def loss_on_fixed_topk(
        ...,
        fixed_topk_ids: Tensor,
    ) -> DSAIndexerLossStats:
        """在固定 support 上计算 Student/Teacher/KL。"""
~~~

因此可以移除旧的 `DSATopKIndicesProtocol` 名称，但仍需要等价的 backend 分发能力。PR #2039
统一的是“模型怎样显式传 IDs”，不等于 Torch/TileLang/cuDNN 从此不需要算子接口。

---

## 6. 常见问题

### 6.1 “Megatron replay 重算 Top-K，但 custom backward 不重算”矛盾吗

不矛盾。Megatron replay 是重新执行整个可训练 Indexer forward；custom backward 接手时，
replay IDs 已经产生。Backward 可以按 backend 选择 sparse 或 dense-recompute 实现，但只计算
这份 sparse support 对应的 loss gradient，不再决定新的 Top-K。

### 6.2 “复用 IDs”是否表示“根据 IDs 重新 project”

不是。正确关系是：

~~~text
hidden/q_resid → replay project → 可导 q/k/weights ─┐
                                                    ├→ selected score / KL
original IDs ───────────────────────────────────────┘
~~~

IDs 在 project 后用于限定 sparse KL support；具体 backend 可以直接 gather selected K，也可以
先重算 dense score再 gather/mask。

### 6.3 为什么不能把 original `q/k/weights` 一起存下来

Original 是 no-grad 的，保存下来的值没有可导图；如果改成保存完整可导激活，又会失去
activation checkpoint 节省显存的主要意义。

### 6.4 如果不用 activation checkpoint 呢

没有 original/replay 两次 forward。Joint 模式只需：

~~~text
一次可导 project
→ 一次 no-grad Top-K
→ 一次 fixed-ID KL
→ backward
~~~

### 6.5 shared layer 也会重复这些流程吗

不会。上面的计数针对 source/computing layer。Shared layer 没有独立 Indexer，只接收并透传
source IDs；它自己的 Sparse MLA/主模型计算仍正常 forward/replay/backward。

---

## 7. 建议方案与待验证项

以下是设计建议，不是当前 checkout 已经实现或验证通过的能力。XTuner joint SFT 建议采用：

~~~text
PR #2039 风格的显式跨层 dsa_topk_ids
  + checkpoint-local FIFO 保存 original IDs
  + replay 重算可导 project
  + fixed-ID Student/Teacher/KL
  + backend facade 保留 select_topk / loss_on_fixed_topk 两项能力
~~~

至少验证：

- frozen：source Indexer/Top-K 调用一次，replay 取回 original IDs；
- joint：project original/replay 各一次，真正的 selector 只调用一次；
- original/replay IDs 的值与 storage 符合复用约定；
- replay fixed-ID KL 产生 finite、nonzero 的 Indexer gradient；
- shared layer 缺失 IDs 立即失败，并原样透传 source IDs；
- multi-micro-batch 中每个位置有独立 FIFO entry，original/replay 顺序完全一致；
- 日志只在 checkpoint wrapper 外聚合一次，不因 replay 重复计数。
