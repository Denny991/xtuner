# GLM-5.2 Indexer 联合 SFT 入门对照解说

> 本文是 [正式设计](./glm52_indexer_sft.md) 和
> [接口伪代码](./glm52_indexer_sft.py) 的伴读版，不定义新的实现语义。
> 如果三份材料存在表述差异，以正式设计和最终实现为准。

## 0. 建议怎么读

第一次接触时，不建议从 1000 多行伪代码顺序往下读。推荐顺序是：

1. 先读本文第 1～8 节，建立整体数据流；
2. 对照正式设计的第 4、5、9、13 节；
3. 再按本文第 16 节的顺序阅读伪代码；
4. backend、分布式校准和 checkpoint 恢复放到最后看。

先记住一句话：

> Indexer 是一个“候选 key 检索器”。它先给 Sparse MLA 选出 Top-K 个 key；
> 训练时再用主 MLA 生成的 teacher 分布，通过 KL loss 教它如何选得更准。

## 1. 这次到底要训练谁

正式设计的目标是同时训练两条相互隔离的路径：

~~~text
LM loss       -> 更新主模型/backbone
Indexer KL    -> 只更新 source/full Indexer
Top-K IDs     -> 只是整数地址，不承担梯度
~~~

这不是“删除 `requires_grad_(False)` 就结束”。因为当前 Indexer 最终只输出整数
Top-K IDs，LM loss 无法穿过整数选择自动训练 Indexer，所以还需要一条可导的 KL loss。

### frozen 与 joint

| 模式 | Indexer 参数 | Indexer KL | 用途 |
|---|---|---|---|
| `frozen` | 不训练 | 没有 | 保持当前行为，作为零回归基线 |
| `joint` | 只训练 source/full Indexer | 有 | LM 与 Indexer 辅助监督联合训练 |

对应正式设计：第 1、3、5 节。对应伪代码：`validate_indexer_training_config()`、
`DSAIndexer`。

## 2. 先认识主要角色

可以用“图书检索”来理解：

| 名称 | 通俗理解 | 更准确的含义 |
|---|---|---|
| GLM-5.2 | 整套问答系统 | 当前要训练的模型 |
| MLA | 精读候选资料的专家 | Multi-head Latent Attention |
| Sparse MLA | 只精读少量候选 | 只对 Top-K key 做注意力 |
| Indexer | 快速目录检索员 | 给 query-key 对打分并选 Top-K |
| Top-K IDs | K 个书架编号 | 每个 query 选中的 global key 整数下标 |
| support | 本次入选的候选集合 | 当前 IDs 指向的所有合法 key 位置 |
| teacher | 专家的参考判断 | 用主 MLA Q/K 计算并 detach 的目标分布 |
| prediction | 检索员的判断 | Indexer 在固定 support 上的分布 |
| KL loss | 比较两种判断的差异 | `KL(teacher || prediction)` |
| IndexShare | 后续层沿用检索结果 | shared 层复用 source 层的 IDs |

### Q、K、V 是什么

- Q（query）：当前位置想查什么；
- K（key）：每个历史位置提供什么索引信息；
- V（value）：真正被注意力读取的内容。

这里有两套不同的 Q/K：

1. 主 MLA Q/K：用于真实 attention，也用于生成 teacher；
2. Indexer q/k：更轻量的检索特征，用于选 Top-K 和计算 prediction。

两套张量不能混为一谈。

### 常见训练术语

| 名称 | 入门解释 |
|---|---|
| SFT | Supervised Fine-Tuning，使用监督样本微调模型 |
| DSA | 本文中的稀疏注意力方案；先选候选 key，再做 Sparse MLA |
| packed sequence | 把多条短样本拼进一个长序列，需要严格保留样本边界 |
| micro-batch（MB） | 一次实际送进模型的小批数据 |
| grad accumulation | 多个 MB 依次累积梯度，之后才执行一次 optimizer step |
| SP | Sequence Parallel，把序列维切到多个 rank/GPU 上计算 |

## 3. 跟着一次 source layer forward 走

下面是 joint 模式的一次完整数据流：

~~~text
hidden_states
    │
    ├──> 主 MLA 投影
    │       ├──> main query/key ──detach──> teacher
    │       └──> q_resid
    │
    └──detach──> Indexer.project(q_resid.detach())
                    ├──> 可导 indexer q/k/head_weights
                    │
                    ├──detach──> no-grad Top-K ──> int32 IDs
                    │                                  │
                    │                                  ├──> Sparse MLA ──> LM loss
                    │                                  │
                    └── teacher + fixed IDs ──> KL ───┘
                                                   │
                                                   └──> 只更新 Indexer
~~~

按真实控制流拆开是：

1. `_project_main_attention()` 生成主 MLA 的 query、key 和 q_resid；
2. `DSAIndexer.project()` 用 detach 后的输入生成可导的 Indexer q/k/head_weights；
3. `DSAIndexer.select_topk()` 在 `no_grad` 下产生整数 IDs；
4. 主 MLA query/key detach 后构造 teacher；
5. `loss_for_indices()` 在同一组固定 IDs 上比较 teacher 与 prediction；
6. Sparse MLA 使用这组 IDs 完成真正的 attention；
7. KL 经 step-global 缩放后，通过 `AuxLossScaler` 挂到 attention 输出；
8. layer 同时显式返回 `projected_output`、`dsa_topk_ids` 和 detached loss 日志值。

对应正式设计：第 5、9、11、12 节。对应伪代码：
`DSAMultiLatentAttention.forward()`。

## 4. 为什么 LM loss 不能直接训练 Indexer

Top-K 最终返回类似这样的整数：

~~~text
dsa_topk_ids = [3, 8, 12, ...]
~~~

分数发生很小变化时，ID 可能完全不变；越过临界点后，ID 又会突然换成另一个整数。
这种离散选择没有普通连续导数，因此下面这条路径会在 Top-K 处断开：

~~~text
LM loss -> Sparse MLA -> integer Top-K -X-> Indexer parameters
~~~

解决方法不是强行对整数求导，而是保留 Indexer 的连续 score，在固定的 Top-K support 上
增加 KL：

~~~text
Indexer continuous score -> prediction -> KL -> Indexer parameters
~~~

“fixed-ID”只表示本次 loss 把 IDs 当作常量。Indexer 参数更新后，未来 step 的 score 和
Top-K 仍然可以变化。

## 5. Teacher、prediction 和 sparse KL

### 5.1 prediction：Indexer 认为谁重要

GLM-5.2 默认：

- `H_index=32`：Indexer 有 32 个打分 head；
- `D_index=128`：每个 Indexer head 的向量维度是 128。

它们不是主 MLA 的 attention head 数，也不是模型 hidden size。

简化公式：

~~~text
head_weights = FP32(weights_proj(hidden)) / sqrt(H_index)
qk_scale     = 1 / sqrt(D_index)

每个 head：ReLU((q_index · k_index) * qk_scale)
最终 score：用 head_weights 汇总所有 Indexer head
prediction：在 key 维做 log_softmax
~~~

- `1/sqrt(H_index)`：避免多个 head 汇总后数值随 head 数膨胀；
- `1/sqrt(D_index)`：稳定点积尺度和 softmax 温度；
- 二者必须各乘一次。

`head_weights` 不是概率，也没有要求在 head 维求和为 1。

### 5.2 teacher：主 MLA 给出的参考答案

~~~text
main query/key --detach
  -> FP32 QK^T * attention scale
  -> 每个 attention head 做 softmax
  -> 汇总 attention heads
  -> 沿 key normalize
  -> teacher probability
~~~

teacher 不是另一个外部大模型，而是同一层主 MLA 的 Q/K 生成的 stop-gradient 软标签。
teacher 与 prediction 必须使用同一个 global K、同一 packed causal 合法范围和同一组
Top-K support；跨 packed sample 的 ID 即使数值没有越界，也属于非法候选。

### 5.3 sparse KL 到底监督了什么

~~~text
L_indexer = loss_coeff * mean_valid_query KL(teacher || prediction)
~~~

第一版默认 `loss_coeff=0.001`、`loss_type="sparse"`。sparse 表示 teacher 和
prediction 都只在当前 Top-K support 内重新归一化并比较。

因此它能校准“已经选中的 K 个 key 内部谁更重要”，但不会直接惩罚某个本应入选、
却落在 Top-K 外面的 key。覆盖全部合法 key 的 dense loss 是另一种目标，当前只规划为
tiny oracle/未来 warm-up，不能当成已实现能力。

### 5.4 三种系数不要混淆

| 名称 | 作用 | 是否训练超参 |
|---|---|---|
| `1/sqrt(H_index)` | 平衡多个 Indexer head 的汇总尺度 | 否，属于 score 定义 |
| `1/sqrt(D_index)` | 稳定 q·k 与 softmax 温度 | 否，属于 score 定义 |
| `loss_coeff` | 控制 Indexer KL 相对 LM loss 的强度 | 是，第一版为 0.001 |
| row coefficient | 把 local KL sum 换成正确的全局均值梯度 | 否，运行时计算 |

当 `global_average=True` 时，row coefficient 实际对应：

~~~text
grad_average_group_size / global_valid_rows_step
~~~

乘 group size 是为了抵消 FSDP/replica 最终的梯度平均；统计有效行的 reduce group 必须与
Indexer 参数真正的 gradient-average group 一致，不能看到“global”就默认猜成 WORLD。
只有显式使用 `global_average=False` 时，才采用 local mean-of-MB-means 语义。

cuDNN adapter 若使用 `sm_scale=1`，会临时把 `1/sqrt(D_index)` 折入
`effective_weights`。这不是第二套参数或第二种数学。

对应正式设计：第 4 节。对应伪代码：`DSAIndexerInputs`、
`torch_fixed_topk_indexer_loss()`。

## 6. detach 为什么不会让 Indexer 也失去梯度

看一个最小例子：

~~~python
y = W(x.detach())
~~~

这里：

- `x` 不接收来自 y 的梯度；
- 参数 `W` 仍然可以接收梯度。

所以：

~~~text
Indexer KL -> Indexer 参数       有梯度
Indexer KL -> hidden/backbone    被 detach 切断
~~~

teacher Q/K 也 detach，是为了避免 Indexer KL 反过来更新主 MLA。

最终梯度分工必须是：

~~~text
LM loss       -> backbone/main MLA
Indexer KL    -> source/full Indexer only
Top-K IDs     -> no gradient
~~~

## 7. IndexShare：source 与 shared 层

可以把 source 看成“负责选路线的层”，shared 看成“沿用路线的层”：

~~~text
source/full layer 0
  生成 IDs=A，计算本层 Indexer KL
          │
          ▼
shared layer 1
  复用 A，不计算 Indexer KL
          │
          ▼
shared layer 2
  继续复用 A，不计算 Indexer KL
          │
          ▼
source/full layer 3
  生成 IDs=B，覆盖 A，计算本层 Indexer KL
~~~

注意：

- `source/full` 的 full 表示“拥有并运行 Indexer”，不是 full/dense attention；
- source 无条件生成新 IDs 并覆盖传入的旧值；
- shared 如果没有收到 IDs 必须立即报错，并原样透传同一 IDs tensor；
- shared 层共享的是 IDs，不是 Q/K/V、attention 输出或 Indexer 参数；
- shared 层仍然用自己的 Q/K/V 跑 Sparse MLA，并由 LM loss 正常训练；
- shared 层不会把自己的 teacher 再回传给前面的 source Indexer。

PR #2039 风格的数据流会把 `dsa_topk_ids` 作为 layer 的显式输入/输出传递，而不是藏在
`SequenceContext` mutable cache 中。这样从函数签名就能看到 IDs 从哪里来、传到哪里。

对应正式设计：第 7.1、12、13.1 节。对应伪代码：`GLM52AttnOutputs`、
`call_glm52_decoder_layer()`。

## 8. Activation checkpoint：为什么会有两次 forward

这里的 checkpoint 是“激活检查点”，用于用额外计算换取更少显存，不是保存到磁盘的模型文件。

Megatron、PR #2039 与本文建议方案的完整独立时间线，见
[三种 checkpoint / Top-K 策略对照](./glm52_indexer_checkpoint_topk_strategies.md)。该文按三个
独立章节分别解释，不使用交叉的“多层重算”表述。

普通训练：

~~~text
forward（保存中间激活） -> backward
~~~

reentrant activation checkpoint：

~~~text
original forward（不保存内部计算图）
          -> 得到输出并继续算最终 loss
backward 到达该层
          -> replay forward（重新计算并建立图）
          -> backward
~~~

可以把它想成：第一次只“演出结果、不录像”；反向时再演一次，这次录像，然后马上倒着算梯度。

### 8.1 MLA 为什么 original 是 no-grad

因为后续层仍需要 MLA 输出，所以第一次 forward 必须计算 MLA；但为了省显存，它不保存
Q/K、attention 等中间计算图。backward replay 时 MLA 会在 grad-enabled 状态重新执行，
然后正常计算主模型梯度。

因此：

> “某一次 forward 在 no-grad 下运行”不等于“这个模块被永久 frozen”。

### 8.2 frozen Indexer 的 checkpoint 时间线

| 阶段 | Indexer/Top-K | MLA |
|---|---|---|
| original | 整个 frozen Indexer 执行一次，并保存 IDs | 计算输出，不保留图 |
| replay | 整个 Indexer 不重算，直接取原 IDs | 重算并建立图 |
| backward | 无 Indexer backward | LM loss 更新主模型 |

### 8.3 joint Indexer 的 checkpoint 时间线

joint 不能像 frozen 一样复用整个 Indexer，否则 replay 没有可导的 Indexer 投影图。

| 阶段 | Indexer project | Top-K selector | fixed-ID KL | MLA |
|---|---|---|---|---|
| original | 执行，但整层 no-grad | 真正执行并保存 IDs | 只产生展示值 | 计算但不存图 |
| replay | 重新执行并建立图 | 不重算，取原 IDs | 重算并挂载梯度 | 重算并建立图 |

所以对每个 source layer、每个 MB，joint + checkpoint 的调用次数是：

~~~text
Indexer project：2 次
Top-K selector：1 次
~~~

### 8.4 `reuse_during_recompute` 的直觉

可以先把它理解成下面的简化逻辑：

~~~python
if 没有 checkpoint:
    return function(*args)
if original_forward:
    output = function(*args)
    frame.enqueue(output)
    return output
if replay_forward:
    return frame.dequeue()
~~~

真实实现还要处理 PyTree、FIFO、multi-MB 和错误检查。frame 只保存无梯度结果；
对当前路径最重要的就是 contiguous int32 `dsa_topk_ids`，而不是 MLA 计算图。
同一次 multi-MB layer invocation 可以共享一个 frame，但每个 MB 必须拥有独立的 IDs、
loss context 和 FIFO entry；original/replay 的 MB 顺序必须一致，不能跨 MB 复用 entry。

对应正式设计：第 3.4、13 节。对应伪代码：`reuse_during_recompute()` 和
`DSAMultiLatentAttention.forward()`。

## 9. AuxLossScaler：怎样挂入一条辅助梯度

设计中使用：

~~~python
projected_output = AuxLossScaler.apply(projected_output, scaled_kl)
~~~

它可以理解为把一张“KL 收据”订在 attention 输出上：

- forward 返回的 `projected_output` 数值完全不变；
- backward 经过这里时，额外触发 KL 对 Indexer 的梯度；
- 不需要修改 LM loss 的数据格式，也不需要第二次 `backward()`。

它只在普通 grad-enabled forward 或 checkpoint replay 中挂载。checkpoint original 处于
no-grad，只返回 detached 展示值；外层只聚合 original 输出一次，replay 不再次累计日志。

`MoEModelOutputs.indexer_loss` 中的值已经 detach，只用于日志和展示，不是另一条梯度入口。

## 10. 为什么不能平均 micro-batch mean

假设同一个 optimizer step 有两个 accumulation micro-batch：

~~~text
MB1：100 个有效 query
MB2：900 个有效 query
~~~

错误算法：

~~~text
0.5 * (KL_sum_1/100 + KL_sum_2/900)
~~~

这会让只有 100 行的 MB1 和 900 行的 MB2 拥有同样权重。

正确算法：

~~~text
(KL_sum_1 + KL_sum_2) / 1000
~~~

多 replica/FSDP gradient-average 时，实际 local 挂载还要使用：

~~~text
scaled_kl =
    local_kl_sum
    * loss_coeff
    * grad_average_group_size
    / global_valid_rows_step
~~~

因此 `DSAIndexerLossContext` 要在一个 train step 开始前汇总：

- 所有 grad-acc micro-batch；
- SP 切分后的有效 query 行；
- 真正参与 Indexer 参数 gradient-average 的 replica group。

所有 context 共用同一个 `global_valid_rows_step`，之后不能再除一次 grad-acc batch size。
这里的有效数目是 query 行数，不是 Top-K slot 总数。

这里还要区分：grad-acc MB 共同决定一个 optimizer step 的统一分母；intra-layer multi-MB
则使用等长 list 传递，每个位置拥有独立的 IDs、loss context 和 FIFO entry。

对应正式设计：第 3.5、10 节。对应伪代码：`DSAIndexerLossContext`、
`build_indexer_contexts_for_step()`。

## 11. Query chunk：它节省了什么

当前 TileLang Top-K selection 会临时物化 FP32 `[S_q,S_k]` logits。以 `S=16K` 为例，
仅这张矩阵大约就是 1 GiB。

沿 query 分块后：

~~~text
for 每 C 个 query:
    仍读取完整 global K
    只产生 [C,S_k] 临时 logits
    立即取 Top-K，只保存 int32 IDs
~~~

如果 `C=2048、S_k=16K`，临时 logits 约为 128 MiB。

但必须明确：

- 峰值从 `O(S_q*S_k)` 降为 `O(C*S_k)`；
- 总计算量仍约为 `O(S_q*S_k)`；
- 最终 `[S_q,K_topk]` IDs 仍必须保存；
- 第一阶段只 chunk no-grad selection，不 chunk fixed-ID loss/backward；
- 它不是 `O(S_q*K_topk)` 的 streaming Top-K。

理论上每个 query 独立，full/chunk 应产生完全相同的 ordered IDs。实现仍要测试 packed
边界、global K 坐标、near-tie 和不足 `block_Q` 的尾块。

实现必须满足：

- 先基于完整 Q 计算一次 global `starts/ends`，再随 chunk 切片，不能让每块从 query 0 重算；
- 每块始终读取完整 global K，并返回 global ID 坐标；
- 非 `block_Q` 对齐的尾块必须有 tail guard，或用零 q/weights + 空 range 补齐后再裁回；
- P0 的非空 chunk size 只允许 TileLang selector，即主 backend 为 `tilelang/cudnn_dsa`；
  `torch + chunk` 必须启动即失败。

只要 full/chunk IDs 相同，后面的 fixed-ID KL 图就相同，query chunk 不会新增一条反向路径。

对应正式设计：第 3.6、7.3 节。对应伪代码：`tilelang_topk_query_chunked()`。

## 12. 两类 backend 不要混在一起

设计中有两个概念上不同的配置轴：

1. `sparse_mla_backend`：控制 Top-K selector 和主 Sparse MLA；
2. `indexer_loss_cfg.backend`：控制 fixed-ID Indexer KL。

这不表示它们可以任意做笛卡尔组合。主 Sparse MLA、解析后的 selector 和 loss adapter
必须同时通过 capability/preflight；任一项不可用都要启动即失败。

### 12.1 当前主 Sparse MLA 路径

| `sparse_mla_backend` | Top-K | MLA forward | MLA backward |
|---|---|---|---|
| `torch` | PyTorch | PyTorch | PyTorch |
| `tilelang` | TileLang | TileLang | TileLang |
| `cudnn_dsa` | TileLang | TileLang | cuDNN DSA |

所以当前 `cudnn_dsa` 是混合路径，不等于 full cuDNN DSA。

### 12.2 Indexer loss backend 的阶段

| loss backend | 定位 | 当前设计阶段 |
|---|---|---|
| `torch_reference` | 短序列可信参考/oracle | P1 correctness |
| `tilelang` | 长序列 production sparse KL | P3 待实现 adapter |
| `cudnn` | full DSA + Indexer-loss 目标 | 后续增强 |

oracle 是用于小规模验证数学的参考实现，不是 16K/128K production 实现。能力不可用时应
启动即报错，不能静默 fallback 到 Torch 后再 OOM。

## 13. 三种 checkpoint/保存概念

| 名称 | 目的 | 保存在哪里 |
|---|---|---|
| activation checkpoint | 用重算换训练显存 | 运行时 checkpoint frame |
| DCP | 保存 model + optimizer 训练状态 | 磁盘 checkpoint 目录 |
| HF checkpoint/export | 模型初始化和权重交换 | HF 格式权重目录 |

它们只是都叫 checkpoint，解决的问题完全不同。

- frozen 且忽略 frozen 参数时，DCP 可能不包含 Indexer，恢复前要从同一 HF base 补齐；
- joint 时 Indexer 已进入 optimizer，DCP 必须保存其 model/optimizer state；
- frozen 与 joint 的 optimizer 参数集合不同，禁止直接 full optimizer resume；
- 模式切换要走 model-only/HF 初始化，并重建 optimizer 与 scheduler。
- HF export 保存 full/source Indexer 权重，但不保存 optimizer；shared 本来就没有独立
  Indexer 权重；
- 即使 mode 相同，coeff/type/backend/global-average/Top-K/chunk/optimizer policy 改变，
  也不属于 exact-trajectory full resume；model-only 初始化可以变更策略，但必须重建
  optimizer/scheduler；
- 标准含 MTP 的 HF GLM-5.2 不能直接用于 P1：需要同时清除 `mtp_config` 和
  `num_nextn_predict_layers`，以 `strict_load=False` 做 model-only 初始化，且不能冒充
  原 checkpoint 的 full resume。

## 14. 第一版为什么暂不支持 MTP、Muon 和 compile

这些不是原理上永远不可能，而是 P1 为了先证明核心数学而设置的能力边界：

- NoMTP：公开实现没有可直接照搬的 MTP Indexer-loss objective；
- AdamW only：Indexer 在 Muon 下的参数分类、shape-dependent LR、coeff、clip 和
  optimizer-state/DCP resume 尚未标定；
- `compile_cfg=False`：第一版 loss context 仍有 Python 状态与运行时断言；
- reentrant checkpoint only：fixed-ID replay 语义先针对现有主 decoder 路径验证。

frozen 主模型仍可以使用 Muon；第一版禁止的是 `joint + Muon`。

## 15. P0～P4 表示“计划阶段”，不是已全部实现

| 阶段 | 主要目标 |
|---|---|
| P0 | 显式 IDs、frozen 零回归、可选 query-chunk selection |
| P1 | tiny Torch fixed-ID KL、梯度隔离、单 source checkpoint correctness |
| P2 | IndexShare、SP/grad-acc/multi-MB、DCP/HF；定义 MTP objective 后再决定扩展 |
| P3 | TileLang production fixed-ID/streaming sparse KL、16K 验证 |
| P4 | full cuDNN、Muon 策略、更多 objective、128K |

伪代码中出现一个类或函数，不代表对应 production kernel 已经存在。`...`、capability probe
和 fail-fast 分支正是在标记待实现边界。

## 16. 伪代码阅读路线

建议按下面顺序打开 [glm52_indexer_sft.py](./glm52_indexer_sft.py)：

| 顺序 | 符号 | 先看什么 |
|---:|---|---|
| 1 | `DSAIndexerInputs/Output/Teacher/LossStats` | 每种张量扮演什么角色 |
| 2 | `DSAIndexer.project()` | 可导投影与 H/D scale |
| 3 | `DSAIndexer.select_topk()` | 为什么只返回无梯度 IDs |
| 4 | `DSAIndexer.loss_for_indices()` | KL 梯度怎样回到 Indexer |
| 5 | `DSAMultiLatentAttention.forward()` | 主模型、Indexer、teacher 三条路径如何汇合 |
| 6 | `torch_fixed_topk_indexer_loss()` | sparse support、mask 和 KL 数学 |
| 7 | `DSAIndexerLossContext` | step-global denominator |
| 8 | `AuxLossScaler` | 辅助梯度怎样挂到主 backward |
| 9 | `reuse_during_recompute()` 分支 | original/replay 怎样复用 IDs |
| 10 | `tilelang_topk_query_chunked()` | selection 显存优化 |
| 11 | backend resolver/metadata | 能力门禁与恢复策略 |

不要把该 `.py` 当作当前可以直接执行的实现。它是接口和控制流伪代码。

## 17. 常见问题

### 17.1 Top-K 不可导，为什么 Indexer 还能训练

不对整数 IDs 求梯度，而是对产生这些选择的连续 Indexer score 计算 fixed-ID KL。

### 17.2 `select_topk()` 是 no-grad，joint Indexer 还会训练吗

会。no-grad 的只有离散 selection；`project()` 和 fixed-ID KL 在普通 grad-enabled forward
或 checkpoint replay 中可导。

### 17.3 checkpoint original 中 MLA no-grad，MLA 还会训练吗

会。original 只是不保存图；backward replay 会重算 MLA、建立计算图并求梯度。

### 17.4 输入 detach 后，Indexer 参数还有梯度吗

有。detach 阻断的是梯度回到 backbone，不会阻断对 Indexer 权重求导。

### 17.5 shared 层是不是不训练

shared 层只是不训练独立 Indexer KL；它自己的主 MLA 仍由 LM loss 正常训练。

### 17.6 query chunk 会不会改变 Indexer backward

第一阶段只 chunk no-grad selection。只要 full/chunk IDs 完全相同，fixed-ID loss/backward
不变；因此必须先做 exact-ID 测试。

### 17.7 `loss_coeff=0.001` 是所有模型的默认值吗

不是。它是公开 GLM-5.2 provider 使用的覆盖值，不是所有模型和 MCore 的通用默认。

### 17.8 当前 `cudnn_dsa` 能否理解为全套 cuDNN

不能。当前是 TileLang Top-K + TileLang forward + cuDNN DSA backward 的混合路径。

### 17.9 日志里的 `indexer_loss` 会不会再次 backward

不会。日志值已经 detach；真正的梯度入口只有挂在 `projected_output` 上的
`AuxLossScaler`。

## 18. 最小检查清单

读完后，如果能回答下面这些问题，就已经掌握了主干：

- 为什么 LM loss 不能穿过整数 Top-K 训练 Indexer？
- 为什么 detach 输入后 Indexer 参数仍然有梯度？
- teacher Q/K 为什么也要 detach？
- source 与 shared 层分别做什么？
- checkpoint original 与 replay 哪些东西重算、哪些东西复用？
- 为什么 joint 只能复用 IDs，不能复用整个 Indexer？
- 为什么两个 micro-batch 不能各求均值后再平均？
- query chunk 降低的是峰值显存还是总计算量？
- `sparse_mla_backend` 和 Indexer loss backend 有什么区别？
- activation checkpoint、DCP 和 HF checkpoint 有什么区别？

如果其中某一项仍不清楚，可以直接按本文对应章节回看，再跳到正式设计和伪代码的同名部分。
