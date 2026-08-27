# GLM-5.2 Indexer 联合 SFT 设计

> 第一次接触 MLA、Indexer、KL 或 activation checkpoint，建议先读
> [入门对照解说](./glm52_indexer_sft_beginner_guide.md)；若只想比较 Megatron、PR #2039 与
> XTuner joint 的重算行为，直接读 [checkpoint / Top-K 策略对照](./glm52_indexer_checkpoint_topk_strategies.md)。

## 1. 目标

这份设计用于指导 XTuner 在现有 GLM-5.2 SFT 上增量支持 DSA Indexer 辅助监督。

目标训练语义：

~~~text
LM loss       -> 更新主模型
Indexer KL    -> 只更新 full/source Indexer
离散 Top-K   -> 选择 Sparse MLA 的 KV，不承担梯度
~~~

第一版要求：

- 默认 frozen 的 Top-K、Sparse MLA、checkpoint 和数值路径不变；
- joint 模式下 LM 与 Indexer KL 的梯度严格隔离；
- 对 packed sequence、SP、grad accumulation、IndexShare、activation checkpoint 和 DCP 给出闭环语义；
- 采用显式 `dsa_topk_ids` 跨层数据流和 checkpoint-local fixed-ID replay，不再扩展旧的
  `SequenceContext` mutable cache；
- 先用短序列 PyTorch oracle 证明数学，再接长序列 TileLang streaming sparse KL；
- backend 不具备训练能力时立即报错，不允许静默 fallback 后再 OOM。

对应接口伪代码见 [glm52_indexer_sft.py](./glm52_indexer_sft.py)；前期调研见
[glm52_indexer_sft_research.md](../../Znote/glm/glm52_indexer_sft_research.md)。

本设计核对的本地源码根目录统一为 /home/liutong/ZmyCode/：

- XTuner：/home/liutong/ZmyCode/xtuner
- Megatron-LM / Megatron-Core：/home/liutong/ZmyCode/Megatron-LM

GLM-5.2 provider 和公开 recipe 参考 Megatron-Bridge 官方 main。本地没有
Megatron-Bridge checkout，所以本文不伪造它的本地文件链接。

### 1.1 实现基线

本文复用两个 PR 中不同层次的资产，不能把它们当成互斥方案：

- PR #2039：提供显式 `dsa_topk_ids` 跨层数据流和 checkpoint-local FIFO ownership；
- PR #2022：提供可直接借鉴的 cuDNN fixed-ID Indexer score-recompute、32→64 head padding、
  custom backward 与算子测试。只复用其 op/adapter 资产，不照搬 model-boundary source-loss
  mean、mutable cache 或按 MB 独立归一化。

本文模型与 checkpoint 数据流以 PR #2039 的方案（或等价实现）为基线。当前本地
checkout 仍包含 `xtuner/v1/module/attention/dsa_topk_sharing.py`、
`SequenceContext.dsa_topk_cache` 等旧 mutable-cache 路径，因此 P0 必须二选一：先 rebase
到 PR #2039，或在同一变更中完成等价迁移并删除旧 cache ownership。两套 runtime 不得并存，
本文后续 `xtuner/v1/model/moe/glm52/*` 文件路径均指迁移后的目标结构。

## 2. 第一版非目标

- dense MLA 到 DSA 的独立 Indexer warm-up；
- source Indexer 同时蒸馏多个 shared 层 teacher；
- 独立 Indexer optimizer 或 LR scheduler；
- joint + MTP；
- joint + Muon；
- 把当前 XTuner cudnn_dsa 混合路径称为 full cuDNN DSA；
- 要求 TileLang/cuDNN 原子归约下逐 step grad_norm bitwise 一致。

MTP 不是遗漏。PR #2022 已实现“保留标准 MTP LM objective、只训练 main source Indexer、
physical MTP Indexer frozen”的可选策略；但第一版为先锁定 KL、重算和分布式归一化，仍保守
使用 NoMTP 并启动即拒绝 joint+MTP。这是 P1 验证范围，不是原理上不支持。

## 3. 当前实现的关键约束

### 3.1 Indexer 被结构性冻结

当前 xtuner/v1/module/attention/dsa_mla.py 中的 DSAIndexer 同时使用：

~~~python
self.requires_grad_(False)

@torch.no_grad()
def forward(...): ...
~~~

optimizer 又只收集 requires_grad=True 参数，所以只改 recipe 无法训练 Indexer。

### 3.2 Top-K 协议只有整数 IDs

DSATopKIndicesProtocol 只返回 topk_indices。离散 Top-K 会切断 LM loss 到 Indexer
score 的梯度，因此必须新增显式 KL 路径，不能期待 LM loss 自动训练 Indexer。

### 3.3 teacher 必须在 Sparse MLA 之后、attention 内构造

teacher 复用主 MLA 已生成的 query/key。只有 DSAMultiLatentAttention 同时知道：

- main Q/K、softmax scale，以及本次 fixed-ID Sparse MLA 的 natural-log LSE；
- Indexer q/k/head_weights/qk_scale；
- packed causal 边界和 SP 布局；
- 当前层是 full/source 还是 shared。

执行顺序必须是 `select IDs -> Sparse MLA -> teacher(Q/K/LSE) -> fixed-ID KL`。其中 PyTorch
oracle 可以重算分母，但 PR #2022 的 cuDNN score-recompute adapter 需要 Sparse MLA LSE。
Trainer 不应重新跑模型或理解 DSA 数学。

### 3.4 activation checkpoint 需要区分投影与离散 IDs

PR #2039 的 frozen 路径通过 `reuse_during_recompute` 在 checkpoint-local frame 中保存
source Indexer 输出，replay 直接复用整数 IDs。这对 frozen 是正确优化，但 joint 不能复用
整个 trainable Indexer：replay 必须重算可导 q/k/head_weights 投影，只复用 original 的离散 IDs，
再用 fixed-ID KL 重建 Indexer autograd 图。

### 3.5 不能平均 micro-batch mean

packed 后不同 micro-batch 的有效 query 数可能不同。若每个 MB 先求均值，再除
grad-acc 数，目标会随 packing 改变。Indexer KL 必须像 CE loss 一样使用整个
train step 的统一有效 query 分母。

### 3.6 TileLang Top-K 会物化完整 logits

当前 `tilelang_indexer_fwd.py` 在 Python 侧分配 FP32 `[S_q,S_k]` logits。kernel 内部按
block 计算并不能降低这张输出的峰值显存。第一阶段可沿 query 维切块：每块读取完整 global K，
立即做 Top-K 并只拼接 int32 IDs，使峰值从 `O(S_q*S_k)` 降为 `O(C*S_k)`；总计算量仍是
`O(S_q*S_k)`，最终 `[S_q,K]` IDs 也不能省略。

query chunk 只解决 selection workspace。joint loss 的 `[S_q,K]` teacher/predict、custom
autograd saved tensors 以及多个同时存活的 source graph 不会因此消失；生产门禁必须另行估算。

## 4. 算法基线

数学以 Megatron-Core 的 DSA Indexer loss 为基线。

### 4.1 prediction

这里 `H_index` 是 Indexer head 数，GLM-5.2 默认 32；`D_index` 是每个 Indexer head
的维度，默认 128。它们不是主 MLA 的 attention head 数或 hidden size。

~~~text
raw_head_weights = weights_proj(detached_hidden)
head_weights     = FP32(raw_head_weights) / sqrt(H_index)
qk_scale         = 1 / sqrt(D_index)

head_score(q,k,h) = ReLU((q_index[q,h] · k_index[k]) * qk_scale)
score(q,k)        = sum_h head_weights[q,h] * head_score(q,k,h)
prediction        = log_softmax(score, key_dim)
~~~

公共 `DSAIndexerInputs` 只有上述一种 `head_weights + qk_scale` 语义，不同时暴露
`selection_weights` 和 `training_weights`。若 cuDNN adapter 的 API 固定使用
`sm_scale=1`，只允许它在边界临时生成：

~~~python
effective_weights = (head_weights * qk_scale).to(torch.bfloat16)
~~~

这只是 backend 的 dtype/scale 表示，不是第二套参数或第二套数学。Top-K 与 loss backend
必须共用同一口径。还要严格区分：

- `H_index/D_index` score scale：定义 logits 和 softmax 温度；
- `loss_coeff`：控制 Indexer KL 相对 LM loss 的权重；
- `row_coefficient`：sum/mean 和分布式 gradient average 的实现换算。

### 4.2 teacher

~~~text
fixed-ID Sparse MLA -> natural-log LSE
main query/key --detach
  -> FP32 QK^T * attention_softmax_scale
  -> 每个 attention head 独立 softmax
  -> 对 head 求和
  -> 沿 key 做 L1 normalize
  -> teacher probability
~~~

当前 XTuner DSA query 保持 SP local，compressed K 为 SP global，本 rank 保留全部
attention heads。第一版不照搬 Megatron 的 TP head-shard all-reduce，但 teacher 与
prediction 必须使用同一 global K 和 packed 边界。

### 4.3 loss

~~~text
L_indexer = coeff * mean_valid_query KL(teacher || prediction)
~~~

默认 coeff=0.001、loss_type=sparse，与公开 GLM-5.2 Bridge 配置一致。这里要区分：

- MCore 通用默认不是无条件 0.001；
- 0.001 + sparse 是 GLM-5.2 provider 的覆盖值。

PR #2022 的实验配置/单测可使用 1.0，但不能直接与这里的 production 0.001 曲线横比；日志应
同时记录未缩放 KL 与乘 `loss_coeff` 后的 objective。

sparse 只在当前 Top-K support 上同时计算 teacher 与 prediction。它能校准已选集合内
的排序，但不会直接惩罚集合外漏选 key。dense 覆盖全部合法 key，只适合 tiny oracle
或未来 warm-up。

精确 Indexer-logit 梯度为 prediction - teacher；离散 IDs 无梯度。

## 5. 梯度不变量

~~~text
main hidden ───────────────────────────────> Sparse MLA ─> LM loss ─> backbone grad
     │                                             ▲
     ├─ detach ─> Indexer ─> integer Top-K ────────┘
     │                  │
     │                  └─ differentiable score ───────────────┐
     │                                                         │
main MLA Q/K ─ detach ─> teacher distribution ─> KL ──────────┘
                                                               │
                                                               └─> Indexer grad only
~~~

必须成立：

- Indexer 的 hidden_states/q_resid 输入 detach；
- teacher 的 main Q/K detach；
- full/source Indexer 只由 KL 更新；
- backbone 只由原有 LM/MTP/MoE loss 更新；
- shared 层没有独立 Indexer 参数，也不计算 KL；
- `dsa_topk_ids` 是显式无梯度输出；checkpoint frame 只保存整数 IDs，不保存 autograd graph。

## 6. 配置 API

在 DSAMLAConfig 增加：

~~~python
indexer_train_mode: Literal["frozen", "joint"] = "frozen"
indexer_loss_cfg: DSAIndexerLossConfig | None = None
indexer_topk_query_chunk_size: int | None = None
indexer_selector_workspace_budget_bytes: int | None = None
~~~

`indexer_topk_query_chunk_size` 是 Top-K selection 的执行策略，不属于 loss 配置：

- `None` 保持旧 one-shot 路径，用作 frozen bitwise baseline；
- 非 None 必须大于 0；第一阶段仅允许 Top-K 实际走 TileLang 的 `tilelang/cudnn_dsa`；
- production recipe 显式选择 2048/4096 等值，不静默修改默认值；
- 它需要进入运行日志以便复现，但不会改变 checkpoint 参数 schema。

joint 使用 TileLang selector 时还必须显式设置 selector workspace budget。预算不能只算
`4*C*S_k` logits，还要覆盖 padded chunk、Top-K scores/int64 indices/scratch、chunk IDs 和最终
`[S,K_eff]` int32 IDs；wrapper 预分配最终 IDs 后逐块 copy，不能 append 后再 cat 出第二份。

DSAIndexerLossConfig 采用当前 ZLossConfig 一类的 Pydantic BaseModel 风格，而不是继承
LM-head 专用、带抽象 loss kwargs 接口的 BaseLossConfig：

~~~python
class DSAIndexerLossConfig(BaseModel):
    loss_coeff: float = 0.001
    loss_type: Literal["sparse", "dense"] = "sparse"
    backend: Literal["torch_reference", "tilelang", "cudnn_indexer"] = "torch_reference"
    global_average: bool = True
    reference_workspace_limit_bytes: int = 2 * 1024**3
    production_total_peak_budget_bytes: int | None = None
~~~

| 模式 | loss cfg | Indexer 参数 | 行为 |
|---|---|---|---|
| frozen | 必须为 None | 全部 frozen | 复用当前路径 |
| joint | 必须存在且 coeff > 0 | 仅 full/source trainable | LM + Indexer KL |

`loss_coeff` 必须 finite 且大于 0；NaN/Inf 不能通过普通 `<= 0` 判断漏掉。非 reference
backend 必须显式配置正数 `production_total_peak_budget_bytes`。它不是单个 op 的 workspace，
而是 Indexer 相对主模型可占用的总峰值 headroom。

DSAMLAConfig.build() 当前调用 self.model_dump()。nested config 会被 dump 成 dict，
所以实现时必须 exclude indexer_loss_cfg 后显式传对象。

第一版仍只支持 joint + AdamW。Muon 在原理上并非不可支持，但会把 Indexer 二维投影和
一维 norm 分到不同算法，并应用 shape-dependent LR multiplier；其 loss coeff、LR、grad clip
与 DCP 语义尚未标定。因此 P1 保留启动硬限制，后续单独完成参数分组和更新等价测试再开放。
该校验必须放在能同时看到 model、optimizer 和 checkpoint 配置的 Trainer build 层。
同一启动校验同时拒绝 joint + MTP，不能拖到首个 batch 才报错。当前 MoE main decoder
checkpoint 固定使用 reentrant；P1 不新增一个仓库中不存在的 checkpoint_impl 配置。
P1 还要求 model_cfg.compile_cfg=False。当前 loss context 有 Python 状态更新和 shape/row
断言，不能直接放进 decoder fullgraph；compile-safe context/backend 留到 P3。
`fsdp_cfg.requires_grad=False` 也必须拒绝，否则 fully_shard 会把刚解冻的 Indexer 再冻结。
非 reference production backend 还必须给出显式总峰值预算与可证明的 recompute 覆盖。

启动接线必须落在现有真实构建顺序中：Trainer `_resolve_config_conflicts()` 后用局部
`intra_layer_micro_batch` 和统一解析的 runtime 开关完成静态校验；`TrainEngine.build_model()`
构造模型并调用 `model.fully_shard()`，在其中的唯一 checkpoint-wrapper site 选择 GLM frame
wrapper；fully_shard 返回、`build_model()` 返回前校验/cache Indexer gradient-average group；随后
沿用 `TrainEngine.__init__` 的现有 `build_optimizer()`。不能调用不存在的
`trainer.build_optimizer_and_engine()`，也不能假设 Trainer 实例持有
`activation_offload_enabled/intra_layer_micro_batch` 属性。

## 7. backend、显式 IDs 与 query chunk

### 7.1 模型层显式 IDs 契约

采用 PR #2039 的数据流：source layer 输出 `GLM52AttnOutputs.dsa_topk_ids`，decoder 将它
显式传给后续 shared layer；到下一个 source layer 时重新生成并覆盖。IDs 不再隐藏在
`SequenceContext` mutable cache 中。

~~~python
class DSAIndexerOutput(TypedDict):
    dsa_topk_ids: Tensor  # contiguous int32 [S_q,1,K]
~~~

### 7.2 统一 Indexer backend facade

模型层只依赖一个统一对象；旧 `DSATopKIndicesProtocol` 可以在迁移期作为 facade 内部的
legacy selector adapter，但不再作为联合训练的高层契约：

~~~python
class DSAIndexerBackendProtocol(Protocol):
    def select_topk(indexer_inputs, seq_ctx, *, index_topk) -> Tensor: ...
    def loss_on_fixed_topk(
        indexer_inputs,
        detached_teacher_q_k_lse,
        fixed_topk_ids,
        seq_ctx,
        *,
        loss_type,
    ) -> DSAIndexerLossStats: ...
~~~

`select_topk` 只返回 IDs；`loss_on_fixed_topk` 返回未乘 coeff 的 FP32 `kl_sum` 和 local
有效行数，不负责训练缩放、日志或 checkpoint。两者必须消费同一个 canonical
`head_weights + qk_scale`。cuDNN 如需 `sm_scale=1`，由 facade 内的 adapter 临时折入
effective weights，Attention 不保存第二套 weights。

这里的 teacher contract 对生产 cuDNN adapter 明确包含同一次 fixed-ID Sparse MLA 产生的
natural-log LSE。因此 attention 的顺序不是“先算 KL 再跑 Sparse MLA”，而是：

~~~text
project -> no-grad select IDs -> Sparse MLA(Q,K,IDs) -> Q/K/LSE teacher
        -> fixed-ID student/teacher recompute -> KL
~~~

### 7.3 no-grad query-chunk selection

第一阶段只分块离散 selection，不分块 fixed-ID KL/custom backward：

~~~text
先基于完整 Q 生成 global starts/ends
for query chunk [lo:hi]:
    q/head_weights/starts/ends 取 [lo:hi]
    K 保持完整 global K
    计算临时 FP32 [C,S_k] logits
    立即 Top-K，只保留 int32 IDs
拼接为 [S_q,1,K]
~~~

不能在每块重新调用 `packed_causal_query_ranges(chunk_len)`，否则后续块会被错误地当成
从 query 位置 0 开始；必须先对完整 Q 计算 ranges，再切片，并保持 global K 坐标。
也不能照搬 slime 的 Top-K scores/softmax/autograd wrapper：当前 selection 保持 no-grad，
训练梯度只来自后续 fixed-ID KL。

当前 XTuner TileLang primitive 对不足 `block_Q` 的 query tail 没有完整越界保护。因此
P0 必须二选一：移植完整 tail guard，或把每个 chunk 补到 `block_Q` 的整数倍，补齐行使用
全零 q/weights 和空的 global range，得到 IDs 后再裁回真实行。配套伪代码采用第二种方案；
在任一方案落地前，不允许把任意长度的尾块直接送入现有 primitive。

理论上 query 行彼此独立，full/chunk IDs 应完全相同；near-tie、尾块、kernel grid 和 packed
边界仍必须做 exact-ID 测试。若未来再分块 loss/backward，所有 chunk 只能先累加 raw
KL numerator，最后除一次 step-global valid rows；共享 K 的梯度也必须完整求和。

### 7.4 三种独立 backend 能力

不能把 selection、主 Sparse MLA 和 Indexer loss 合并成一个“是否 full cuDNN”的判断：

| 能力 | 作用 | 当前/目标实现 |
|---|---|---|
| selector | dense/streaming score 后选 IDs | torch 或 TileLang；可做 query chunk |
| sparse attention | 按 IDs 计算主 MLA forward/backward/LSE | torch、TileLang、当前 hybrid cudnn_dsa |
| fixed-ID Indexer loss | 在相同 IDs 上重算 teacher/student 并返回 Indexer grad | torch oracle、目标 TileLang、PR #2022 cuDNN adapter |

| loss backend | 当前/目标能力 | 复杂度与用途 | 状态 |
|---|---|---|---|
| torch_reference | fixed-ID sparse；dense 后补 | sparse 约 O(Sq*K)，dense O(Sq*Sk) | tiny correctness |
| tilelang | streaming fixed-ID sparse KL + Indexer grad | 目标是不物化 dense SxS | 待实现 adapter |
| cudnn_indexer | cuDNN score-recompute + custom Indexer backward | 复用 PR #2022 的 O(Sq*K) op | production 候选，需移植/门禁 |

这里的 tilelang 是目标 adapter，不是 XTuner 当前已有能力。Megatron production path
通常融合 Top-K 与 sparse KL；XTuner 还需提供固定 IDs 可重算 KL 的入口，才能服务 replay。
因此 P1 的配置 helper 默认必须是 torch_reference；只有 TileLang 或 `cudnn_indexer` adapter
通过 capability、显存和梯度测试后，16K production recipe 才能显式启用。

当前 Sparse MLA 路径：

| sparse_mla_backend | Top-K | forward | backward |
|---|---|---|---|
| torch | PyTorch | PyTorch | PyTorch |
| tilelang | TileLang | TileLang | TileLang |
| cudnn_dsa | TileLang | TileLang | cuDNN DSA |

因此当前 `sparse_mla_backend=cudnn_dsa` 是 hybrid，但这不妨碍在其上叠加独立的
`loss_backend=cudnn_indexer`。后者不是 full cuDNN attention；P4 的 full cuDNN DSA 仍是另一项工作。

TileLang adapter 第一版只允许 sparse，并检查 CUDA、BF16、layout、heads、head dim、
Top-K、packed 和 SP。MCore 的 TileLang dIndexK 路径仍可能用 atomic add，所以
与 reference 容差对齐不等于 bitwise deterministic。

`DSAMLAConfig.build()` 还必须分别保留/新增 Sparse MLA、Top-K selector 与 Indexer-loss 的
runtime preflight；三者任何一项不可用都应在启动时
fail fast，不能拖到首个 batch。

### 7.5 production 显存门禁

query chunk 的 `[C,S_k]` selection workspace、source-only IDs guard 与 fixed-ID loss 的 saved
tensors 必须分别建模，再按真实时序合成到同一个总峰值。后者至少受 `S_q*K`、
teacher/student dtype、custom-autograd 保存项和同时存活 source graph 数影响。
无 activation checkpoint 或 partial checkpoint 时，多个 source 的 `[S_q,K]` state 可能一起活到
backward，不能只估一层。

Trainer build 必须从静态 `pack_max_length`、IndexShare source mapping、实际 recompute plan 和
backend capability 得到：

~~~text
previous source persistent state
  + max(current selector peak, current IDs-guard peak, current loss invocation peak)
~~~

同时检查 forward 结束时的全部 source persistent state，并将最终最大值与
`production_total_peak_budget_bytes` 比较。selector 子预算只用于更早暴露具体子系统问题，不能
代替总 headroom。无法证明
pack length 或 source graph lifetime 时直接拒绝 production backend；不能以“首 batch 看是否 OOM”
作为能力探测。运行时再用真实 shape 做第二层 guard。

## 8. PyTorch oracle 的安全边界

fixed-ID sparse oracle 至少处理：

- 0 <= id < S_global；
- 每个 query 的 packed start/end；
- seq_ctx.mask；
- -1 padding slots；
- 全 invalid row；
- valid query 内的 compact `-1` suffix；
- tiny/debug 模式下的重复 ID。

全 invalid row 不能直接执行全负无穷上的 softmax。应先放一个 benign sentinel，
softmax 后再用原始 mask 把整行 loss 和梯度归零。对于有效 query，如果没有任何合法
Top-K ID，应直接报错，而不是让 denominator 随 backend 结果变化。

要区分两种 padding：

- 数据/SP tail padding query：现有 selector 可能仍返回正常 causal IDs；Indexer loss 对整行
  无条件忽略，不能要求 attention 实际 IDs 必须是 -1；
- valid query 的 Top-K support padding：只有 `-1` 可被忽略。非 -1 的 global 越界、跨
  packed sample或非因果 ID，即使同一行还有合法 ID，也必须立即报错；所有 `-1` 必须构成
  suffix，使 cuDNN `topk_length` 与实际前缀严格一致。

query-chunk 为对齐 `block_Q` 人工补出的临时行则必须使用空 range，并在 wrapper 内裁掉；
它们不是数据 padding，不能泄漏到最终 IDs。

公共 guard 放进 source 的 `select-and-validate` callable：normal/original 真正选择后校验一次，
checkpoint replay 从 frame 取得已校验 IDs，不再做 O(S*K) guard；shared 只做 shape/dtype/identity
轻量检查。production IDs 来自 TopK，本身 unique；完整 sort-based duplicate check 只在 tiny/debug
capability 测试运行，避免 16K×2048 每层额外分配数百 MiB。

GLM-5.2 teacher KV head 为 1，可按 S,K,D gather，再执行：

~~~python
einsum("shd,skd->shk", teacher_q, teacher_k)
~~~

不能先 expand 成 S,K,H,D 再 materialize。

门禁不能只看 seq_len。峰值由 local Q、global K、Top-K、heads、head dims 和 dtype
共同决定。配置使用 reference_workspace_limit_bytes，op 拿到真实 shape 后做保守估算。
dense oracle 和 PyTorch Top-K selector 仍是二次复杂度。后者会在当前实现中物化近似
`[B,S_q,H_index,S_k]` 的 per-head score，因此必须在分配前使用自己的动态 workspace guard，
不能误用 fixed-ID loss 的 O(S*K) 估算。

## 9. DSAIndexer 职责调整

对齐显式数据流后，`DSAIndexer.forward()` 返回 `DSAIndexerOutput`，但其中仍只包含
原有 contiguous int32 IDs，不暴露 dense score。新增细粒度方法：

~~~python
project(detached_hidden, detached_q_resid) -> trainable q/k/head_weights/qk_scale
select_topk(detached_inputs) -> integer IDs
loss_for_indices(inputs, detached_teacher, fixed_ids) -> loss stats
~~~

联合训练路径：

~~~python
inputs = project(hidden.detach(), q_resid.detach())
with torch.no_grad():
    ids = select_topk(inputs.detach())
sparse_out = sparse_mla(main_q, main_k, ids)
teacher = build_teacher(main_q.detach(), main_k.detach(), sparse_out.softmax_lse)
stats = loss_for_indices(inputs, teacher, ids)
~~~

query chunk 只发生在 `select_topk` backend 内部；完整可导 q/k/head_weights 仍交给一次
fixed-ID loss，dense selection logits 和 Top-K scores 不进入 autograd。`dsa_topk_ids` 的
dtype/contiguous/范围检查必须先于 Sparse MLA 和 cuDNN loss 调用，避免 opaque kernel 错误。

只去掉“整个类永远 no-grad”的结构限制；eval/prefill/decode 仍不计算 KL。frozen source
通过 `reuse_during_recompute(self.indexer, ...)` 复用整个无梯度输出；joint source 只对稳定的
`select_topk` callable 使用 checkpoint reuse。joint training 如果 context 丢失则立即报错，
避免静默退化成 frozen。

## 10. step-global loss 校准

若两个 accumulation MB 分别有 100 和 900 个有效 query：

~~~text
错误：0.5 * (KL_sum_1/100 + KL_sum_2/900)
正确：(KL_sum_1 + KL_sum_2) / 1000
~~~

Trainer 先把 SequenceContext 做 SP split 供 forward 使用，但 build_loss_ctx_batch 仍收到
未切分 data。因此 DSAIndexerLossContext.build_batches() 应仿 CELossContext：

1. 将每个完整 query mask 按与 SequenceContext.split() 相同的规则切成本 SP rank 的 local mask；
2. 汇总本 step 所有 grad-acc contexts 的 local rows；
3. 将 CPU data_batch 得到的计数 scalar 移到与 Indexer/WORLD backend 兼容的当前
   accelerator，再在实际参与 Indexer 参数梯度平均的 data group 上 all-reduce 一次；
4. 将同一个 global_valid_rows_step 写进每个 context。

每个 source layer、每个 MB 挂载：

~~~text
scaled_kl =
    local_kl_sum
    * loss_coeff
    * grad_average_group_size
    / global_valid_rows_step
~~~

乘 group size 是为了抵消 XTuner/FSDP 最终对 replica gradient 的平均。初版 capability
check 必须证明 rows reduce group 和 Indexer 参数的有效 gradient-average group 一致，
不能笼统写成 SP、DP 或 WORLD 后靠猜测。若目前只可证明 WORLD 拓扑，第一版就只支持该拓扑。
P1 必须在 fully_shard 后、首 batch 前遍历所有 source Indexer 的 DTensor/FSDP placement，
验证 WORLD-average 并缓存结果；每 step 只做 O(1) 断言后直接使用 WORLD，不能等第一批数据
才发现不兼容，也不能每 step 重扫 placement。
不向现有 build_loss_ctx_batch() 伪造一个无人传入的 grad_average_group 参数。

所有 contexts 已共用 step denominator，不能再除 batch_size。只有显式
global_average=False 才采用 local mean-of-MB-means。

backend 的 valid_rows 必须和 context 的 local query count 一致。production 不应为每个 source
调用 `.item()` 做 GPU-host 同步；context mask 是权威计数，backend 对齐使用 device-side async
assert 或仅在 debug/capability 测试中同步检查。

## 11. loss 挂载与日志

逐 source layer执行：

~~~python
projected_output = AuxLossScaler.apply(projected_output, scaled_kl)
~~~

它让 projected output 数值不变，并在主 loss backward 到达该层时触发 KL backward，
避免把所有层的 KL 图保留到 finalize。

每个 source layer 的 AttnOutputs 返回 detached `indexer_loss` 数值；
`Glm52MoE._call_decoder_layer` 在 checkpoint wrapper 外只聚合 original forward 的返回值，
replay 不会再次经过该外层聚合。这样不需要在 checkpoint 内维护 phase marker 来防止重复日志。
后续可增加：

- indexer_aux_total：所有 source/full invocation 的目标总和；
- indexer_aux_source_mean：按 source layer 平均；
- 可选 indexer_aux_all_layer_mean：shared/no-loss 层记 0，用于对照 MCore tracker。

MCore tracker 按总层数平均；若 XTuner 记录 source loss 总和，两者不能直接横比。

MoEModelOutputs 可增加 indexer_loss: Tensor | None。joint 时它是 detached scalar，
用于现有日志和总 loss 数值展示；梯度只来自 AuxLossScaler，不会 double backward。
frozen 时字段为 None，且不创建 context、不统计 rows、不发通信。

single-MB 和 intra-layer multi-MB 都要汇总各自 layer output，再求和为同一个 model output 字段。
P1 暂不把 Python float 诊断塞进 ModelForwardExtraLogInfo；该类要求 Tensor 且新 key 还要注册
reduction 规则。以后增加诊断时，字段名也不要包含 loss，避免被 TrainEngine 当作额外 loss。

## 12. 真实调用链

context 不能只从一个虚构的 Glm52MoE helper 透传。实际需要覆盖：

- MoE.build_loss_ctx_batch
- MoE.forward、_forward、_micro_batch_forward
- Glm52MoE._call_decoder_layer
- GLM52DenseDecoderLayer / GLM52MoEDecoderLayer 的显式 `dsa_topk_ids` 输入输出
- DenseDecoderLayer.forward、_forward
- MoEDecoderLayer.forward、_forward、_micro_batch_forward、_pre_moe_forward
- DSAMultiLatentAttention.forward

同时扩展 GLM52 layer outputs、MoELossContextDict 和 MoEModelOutputs。source layer 无论收到
什么旧 IDs 都生成新的 IDs；shared layer 必须显式收到前一个 source 的 IDs，否则立即失败。

GLM52 decoder output 必须是对现有契约的增量，保留 `router_logits/router_weights/router_topk_ids`
（multi-MB 为等长 list）。`Glm52MoE` 仍用它们计算 balancing/z loss；不能为了加入
`dsa_topk_ids/indexer_loss` 把返回值缩成只含 hidden/IDs/loss 的新三元组。

DSAMultiLatentAttention 仍返回完整 AttnOutputs；只替换其中的 projected_output carrier，
raw_output 和 softmax_lse 必须原样保留，不能把返回类型缩成一个 Tensor。

P1 的伪代码只要求 single-MB scalar 契约。P2 的 `_micro_batch_forward` 必须把它逐 MB 提升为
等长 list：`hidden_states: list[Tensor]`、`dsa_topk_ids: list[Tensor]`、
`indexer_loss: list[Tensor | None]`。每个 MB 使用自己的 IDs、loss context 和 FIFO entry；
同一次 multi-MB layer invocation 共享一个 checkpoint-local frame，original/replay 必须按相同
MB 顺序 push/pop。source/shared 只在同一 list position 内传递，最后在 wrapper 外汇总所有
MB 的 detached loss。不得跨 MB 复用 IDs/FIFO entry。

当前 decoder public API 接收可变位置参数，因此 multi-MB 必须调用
`layer(*hidden_states, dsa_topk_ids=ids_list, ...)`。写成 `layer(hidden_states, ...)` 会把整个
list 当成一个 Tensor，错误进入 single-MB 分支；也不能直调 `_micro_batch_forward` 绕过
checkpoint wrapper。

P1 拒绝 MTP，所以 MTPBlock/MTPLayer 先不透传 Indexer context；P2 开放时必须补齐
mtp_block.py 和 mtp_layer.py 的 forward、micro-batch、checkpoint 调用链。

TrainEngine 继续使用现有 train_step、_get_total_loss、clip_grad_norm 和 step_optimizer。
不新增第二次 backward，也不能用简化脚本绕过 invalid-grad、zero-grad 或 skip-threshold。

## 13. IndexShare 与 checkpoint

这里的 checkpoint 专指 activation checkpoint；DCP/HF 训练状态持久化见第 15 节。

### 13.1 显式跨层 IDs

- full/source 层用本层 main Q/K teacher 训练本层 Indexer；
- source layer 输出 contiguous int32 `dsa_topk_ids`；
- shared layer 从 decoder 输入显式接收并透传同一 IDs tensor，不计算 KL；
- 到下一个 source layer 时生成新 IDs 并覆盖旧值；
- 不把 shared 层 teacher 聚合回 source；
- 跨 PP stage share 继续不支持。

IDs 生命周期由显式 model output、checkpoint frame 和普通 tensor 引用管理，不再在
`SequenceContext` 中维护第二套 residency/release 状态，也不能另外创建按 `id(seq_ctx)`
索引的 cache。

P1 明确让 IDs 常驻 device。当前 activation-offload hook 只匹配 hidden-state data pointer，
Python frame 中的 int32 IDs 也不会自动成为 SavedVariable，因此不能宣称“复用现有 hook 即可”。
P2 若要 offload，checkpoint frame 必须成为 IDs 的唯一 owner，按 tensor identity 去重，确保同一
source IDs 不会因多个 shared consumer 重复 pack，并定义 replay 前的唯一恢复时机。
该方案只在所有 effective source 都被 activation checkpoint 覆盖时成立；若 recompute ratio 让
任一 source 走 normal forward，它没有 frame owner，P2 启动校验必须拒绝 IDs offload（或未来
另行设计 normal-source owner），不能留下半定义状态。

当前 main MoE forward 直接读取环境变量 `XTUNER_ACTIVATION_OFFLOAD`，Trainer 并不存在可直接
读取的 `activation_offload_enabled` 属性。P0/P2 必须把该开关在 Trainer conflict-resolution
阶段解析为一个权威 runtime 值，并让启动校验、model forward 和 offload hook 共用；不能一边
校验伪造的 Trainer 字段、一边让实际 forward 继续读取环境变量而绕过 ownership 门禁。

### 13.2 checkpoint-local frame

采用 PR #2039 的 PyTree-aware、frame-aware reentrant wrapper。当前本地主 decoder 的默认
wrapper 只接受 Tensor/tuple output，且没有 original/replay frame；仅新增
`reuse_during_recompute()` helper 不会生效。P0 必须同时：

1. 为 nested input 与 TypedDict/list output 做 flatten/unflatten；
2. 在 `MoE.fully_shard()` 的主 decoder wrapper 处显式接入该 checkpoint function；
3. 为每个 checkpoint invocation 创建、验证并销毁 frame。

第 2 点必须替换 `MoE.fully_shard()` 现有的唯一 wrapper site，并同时覆盖 frozen/joint。不能先在
Trainer 外预包 frame wrapper，再由 `fully_shard()` 套一次默认 reentrant wrapper；双重 checkpoint
会让外层 replay 重新执行内层 frame，破坏 selector call-count 和 original-ID ownership，也可能先被
默认 Tensor/tuple output signature 检查拒绝。

是否需要 checkpoint 仍完全由现有 `MoE._should_recompute(layer_idx, mtp_idx)` 决定，因为它掌握
全局层数、物理 layer index 与 MTP 位置。新 helper 只在该分支已为真后选择“GLM frame wrapper”
还是“其他模型默认 wrapper”，不能用只看到 `(layer, fsdp_cfg)` 的第二套简化判定替代它。

完成接线后，每个 checkpoint invocation 独享一个 frame：

- original forward 对稳定 callable 的无梯度输出按 FIFO 保存；
- replay 对相同 callable 按 FIFO 取回；
- frame 输出不得包含 `requires_grad=True` tensor；
- replay 结束必须消费完，否则立即报错；
- frame 不跨 checkpoint invocation；一个 multi-MB layer invocation 可以在同一 frame 中为
  多个 MB 保存独立 FIFO entries。

### 13.3 frozen replay

frozen source 沿用一次执行语义：

~~~python
with torch.no_grad():
    indexer_output = reuse_during_recompute(
        self.indexer,
        hidden_states,
        q_resid,
        position_embeddings,
        seq_ctx,
    )
ids = indexer_output["dsa_topk_ids"]
~~~

original 计算整个 frozen Indexer；source replay 从 frame 取回相同 IDs，不重新投影或 Top-K。
shared layer 从显式 decoder 输入获取 IDs，不访问 checkpoint frame。

### 13.4 joint fixed-ID replay

joint 不能把整个 Indexer 放进 `reuse_during_recompute`，否则 replay 会跳过 trainable 投影，
Indexer 没有梯度。正确拆分为：

~~~text
NORMAL:
  可导 project -> no-grad select IDs -> Sparse MLA/Q-K-LSE teacher
  -> fixed-ID KL -> attach

CHECKPOINT ORIGINAL（no_grad）:
  project -> select IDs 并存入 frame -> Sparse MLA/Q-K-LSE teacher
  -> no-grad fixed-ID KL

CHECKPOINT REPLAY（grad）:
  重新执行可导 project -> 从 frame 读取 original IDs
  -> 重算 Sparse MLA/Q-K-LSE teacher -> fixed-ID KL -> AuxLossScaler attach
~~~

伪代码中的 `select_topk` 必须是稳定 callable；original/replay 都调用
`reuse_during_recompute(selector, detached_inputs, ...)`，但只有 original 真正执行 selector。
因此 query chunk 也只执行一次，frame 保存的是已经拼接完整的 IDs，而不是每块 logits/scores。

正常 forward 与 replay 都在 `torch.is_grad_enabled()` 为真时挂载 AuxLossScaler；original
在 no-grad 下只计算 detached 展示值。source layer 的 detached loss 作为结构化 layer output
返回，在 checkpoint wrapper 外聚合一次；不在 layer 内写 phase marker或重复日志。

Megatron 的 source replay 会重新做 Top-K，但本设计优先复用 original IDs，以保证 checkpoint
forward/backward 使用同一 sparse support。如果未来切换为 replay 重新选择，必须先证明
original/replay IDs exact，并给出 ID 不一致时的失败语义。

### 13.5 不变量与失败条件

- original/replay 的 output PyTree schema 完全相同；
- source selector 输出必须是 contiguous int32 且无梯度；
- frozen Indexer 调用一次；joint project original/replay 各一次、selector 只调用一次；
- replay 的 fixed-ID KL 产生 finite/nonzero Indexer gradient；
- shared layer 缺失 IDs、frame missing/unconsumed output、callable FIFO 顺序不匹配均立即失败；
- 第一版只支持 reentrant；non-reentrant 未定义前启动即拒绝。

## 14. MTP 边界

第一版 joint + MTP 直接 fail fast。

标准 HF GLM-5.2 config 会根据 num_nextn_predict_layers 自动创建 mtp_config，因此 P1
默认 recipe 使用明确的 GLM-5.2-30B-NoMTP 权重。若只有标准含 MTP 权重，转换流程必须显式：

1. 将 model config 的 mtp_config 设为 None，同时将 num_nextn_predict_layers 设为
   None/0；二者必须一起校验，否则重新 from_hf 会再次创建 MTP；
2. 以 strict_load=False 做一次 model-only 初始化，允许忽略 HF 中的 MTP keys；
3. 重建 optimizer/scheduler，并在 metadata 记录原始 MTP base、原始
   num_nextn_predict_layers 与 NoMTP 转换；
4. 导出物是 NoMTP 模型，不能冒充对原含 MTP checkpoint 的 full resume。

后续开放前要先选择：

1. main 1x + MTP-depth average 1x：MTP 是额外 objective；
2. main + N depths 全部乘 1/(N+1)：开关 MTP 前后总权重不变；
3. 只在 main 训练 Indexer，MTP 不提供 KL。

不能只给 MTP depth 乘 1/N 后声称总权重不变，因为 main 仍是 1x。还需定义
share_weights true/false、full/shared physical layer 和 mtp loss scaling factor 的关系。
这些都是 XTuner policy，不应冒充 Megatron 已有语义。

## 15. Checkpoint 与导出

### 15.1 DCP

这里的“DCP payload”当前主要指 model/optimizer；Trainer checkpoint root 还会单独保存
scheduler、dataloader 与 train_state，不能把二者混为一谈：

- frozen + dcp_ignore_frozen_params=True：Indexer 可省略，但恢复前必须从同一 HF base 补齐；
- joint：Indexer trainable，必须进入 model state 和 optimizer state。

frozen 与 joint 会改变 optimizer 参数集合，禁止跨模式 full optimizer resume。阶段切换必须走
“先导出完整 HF checkpoint，再启动新训练”的流程：新 run 重新构造 optimizer、scheduler、step、
epoch 和 dataloader position，不能把 DCP 的部分恢复伪装成 model-only 初始化。本文的 DCP 加载
入口只支持 exact Trainer resume，要求 model、optimizer state、optimizer args、scheduler、dataset
position 和 train_state 一起恢复；部分恢复任意一项均直接拒绝。

### 15.2 metadata

建议在 checkpoint root 新增 indexer_sft.json，记录 mode、完整 loss policy（含
`global_average`）、主 `sparse_mla_backend`、解析后的 selector backend、`index_topk`、
query chunk size、optimizer policy、HF base 的 immutable revision 与 config/weight-content fingerprint、
runtime/kernel fingerprint、distributed-training fingerprint、activation-checkpoint policy 和 IndexShare
pattern，以及 scheduler-policy fingerprint。sidecar 带 schema version、
拒绝未知字段，并校验 frozen/joint 的 loss 与 optimizer policy 组合。Trainer 保存时写入，
_load_checkpoint 在 engine.load_dcp 前校验。chunk size 不改变模型 tensor schema，但可能
改变 near-tie IDs，因此仍属于 exact-trajectory resume policy。

metadata 还必须记录：

- `dcp_indexer_state`：Indexer tensors 是随 DCP 保存，还是仅在 frozen 模式下因
  `dcp_ignore_frozen_params=True` 被省略；
- `indexer_schema_fingerprint`：effective source mapping，以及每个 Indexer 参数的完整名称、shape、
  dtype 和 requires_grad；
- HF base 的 resolved commit SHA，或本地 config/index 与每个实际 weight shard 内容的 canonical
  digest/LFS OID；只 hash manifest 文件本身不够；
- XTuner/torch/CUDA/cuDNN-frontend/TileLang 与解析后 kernel backend 的 runtime fingerprint。
- world size/named mesh、Indexer gradient-average group、SP/EP/FSDP placements、grad accumulation、
  global batch 与 deterministic packing policy 的 distributed-training fingerprint。
- scheduler class、完整 `lr_cfg`、resolved `total_step`/warmup 和 Lambda closure policy 的
  scheduler fingerprint；只加载 scheduler state_dict 不能恢复被更换的 Python 调度策略。

这些字段必须由 model/config/optimizer/DCP manifest 的唯一 builder 推导，不能交给 recipe 手填。

sidecar 由 rank0 原子写入，并接入现有 checkpoint barrier、async-DCP 完成通知、保留和删除
生命周期，不能留下半写文件或孤儿 metadata。解析后的 effective source/full layer mapping
必须保存，不能假设 indexer_types 总是非空。

要区分：

- mode/参数拓扑不匹配：state 与 optimizer 结构不兼容，必须拒绝；
- coeff/type/backend/global-average、Top-K/selection/chunk 或 optimizer policy 不匹配：schema 可能相同，
  但不属于精确轨迹 resume，默认拒绝；
- 策略切换：只能从完整 HF export 启动新 run；不进入本文 DCP exact-resume 入口。

字符串路径相同不能证明 HF base 相同：共享目录内容可能已被替换。exact resume 必须同时验证
revision 和实际权重 fingerprint。若 frozen DCP 因 `dcp_ignore_frozen_params=True` 省略 Indexer，
engine build 前必须已按 current HF base 预载；load 前再证明 current base 与 saved sidecar 的
revision/fingerprint 相同，并校验内存中的 Indexer 参数确实匹配该 base。仅有路径字符串或在
`engine.load_dcp` 前临时调用一个没有 model 的“补权重 helper”都不成立。无法证明来源时，只能先
导出完整 HF checkpoint，再启动新 run。

历史 frozen checkpoint 没有 sidecar 时无法称为 exact resume。只有 current 仍是 frozen、显式开启
`allow_legacy_frozen_best_effort` 且可验证当前 HF base 时才兼容；current 是 joint 或默认严格模式
均拒绝。

### 15.3 HF

HF export 始终保存 full/source Indexer 权重。joint 后验证权重相对 base 已变化；
shared 层仍无独立 indexer 参数。

## 16. 实施阶段

### P0：frozen-compatible 重构

- 新增配置与 Trainer 组合校验；
- 以 PR #2039 为前置，或在本阶段删除旧 `dsa_topk_cache/dsa_topk_sharing` ownership，采用
  `DSAIndexerOutput/GLM52AttnOutputs.dsa_topk_ids` 显式数据流；
- 将主 decoder 真正接到 PyTree/frame-aware reentrant wrapper；TypedDict/list output 必须能
  flatten/unflatten；在 `MoE.fully_shard()` 原有 wrapper site 单次替换，不能 double-wrap，也不能
  只新增一个未被 wrapper 调用的 FIFO helper；
- 抽出 project/select_topk/loss_for_indices；
- 新增可选 no-grad query-chunk Top-K；默认 None 严格走旧 one-shot 路径；
- frozen 不建 context、不发 row-count 通信；
- 原 Top-K/Sparse MLA/compile/checkpoint 回归 bitwise 不变。

### P1：tiny PyTorch joint correctness

- fixed-ID sparse oracle；
- safe packed mask、数据 padding row、重复 ID 与 all-invalid row；
- step-global loss context；
- 显式透传到 attention；
- AuxLossScaler 挂载；
- source/full self-teacher；
- NoMTP 权重、AdamW、compile_cfg=False、eager op + reentrant checkpoint；
- 单 source、单 MB 的 joint checkpoint-frame fixed-ID replay；
- helper 默认 backend=torch_reference；任一 production loss backend 只能在 P3 完成能力与显存
  门禁后由 recipe 显式开启。

### P2：runtime 组合

- IndexShare full/shared；
- SP、unequal packed grad accumulation、intra-layer micro-batch；
- 将 fixed-ID replay 扩展到多 source/shared、activation offload 和 multi-MB 组合；
- P1 IDs 默认常驻 device；若在 P2 offload，由 frame 唯一持有、去重并恢复；
- DCP/HF round-trip；
- 定义 MTP objective 后再扩 MTP 调用链。

### P3：TileLang production sparse KL

- 实现 XTuner layout 的 streaming/fixed-ID adapter；
- 移植/适配 PR #2022 的独立 `cudnn_indexer` score-recompute/custom backward，作为另一条
  production 候选；它不要求先完成 P4 full cuDNN attention；
- 后续再评估 loss/backward query chunk 或 key-tile online Top-K；不能把 P0 的
  no-grad query chunk 称为 `O(S*K)` streaming；
- capability probe；
- 与 torch oracle 比较 KL 和全部 Indexer 参数梯度；
- 按 pack length、K、每层 saved tensors 与同时存活 source graph 数计算显存预算；未知的
  partial-recompute lifetime 或超预算组合启动即拒绝；
- 16K 显存、吞吐、compile、activation offload；
- production recipe 禁止 torch reference。

### P4：增强

- full cuDNN DSA attention（与 P3 独立 `cudnn_indexer` loss adapter 分开）；
- dense warm-up；
- 独立 Indexer LR/param group；
- 定义 Indexer MuonSplit、参数分类、coeff/LR/clip policy；先开放“主干 Muon + Indexer
  AdamW group”，再评估 native Indexer-Muon；
- 多层 teacher；
- 128K packed SFT。

## 17. 测试矩阵

### frozen 与配置

- 默认不创建 context，optimizer 不含 Indexer；
- frozen Top-K、loss、主模型梯度与改动前一致；
- P1 joint + MTP/Muon/compile/无 backend 能力均启动即失败；若未来暴露 main non-reentrant 入口也必须拒绝；
- P1 `intra_layer_micro_batch` 必须为 1；只有注册 P2 list/FIFO capability 后才能放开；
- P1 helper 在 NoMTP 上默认选择 torch_reference；标准含 MTP HF 权重给出明确转换错误；
- joint + `fsdp_cfg.requires_grad=False` 启动即失败；
- nested config build 后仍是 config 对象。
- chunk_size=None 与改动前 one-shot IDs 完全一致；chunk<=0、未支持 backend 启动即失败；

### 数学与 mask

- KL 方向对齐独立 oracle；
- causal、packed、数据/SP tail padding、valid-support `-1`；
- global 越界和跨 packed sample ID 均拒绝；
- valid query 内重复 ID 拒绝；invalid query 即使含现有 selector 的非负 causal IDs 也整行忽略；
- all-invalid row forward/backward 无 NaN；
- 有效 query 无合法 ID 时失败；
- H/D scale 各乘一次；
- canonical score 与 cuDNN folded-weight score、Top-K、student probability、KL 和全部
  Indexer 参数梯度对齐；missing/double scale 用例必须失败；
- workspace guard 按实际 shape 生效。

### query-chunk selection

- full 与 chunk ordered IDs exact；覆盖 `C={1,block_Q-1,block_Q,block_Q+1}`、非整除尾块和 `C>=S_q`；
- packed 多样本且 chunk 跨 sample 边界，ranges 由完整 Q 生成后再切片；
- primitive 补齐行只能产生全 -1 IDs，裁回后不得污染任一真实 query 的 ordered IDs；
- global K/IDs、padding、all-invalid、短样本和 near-tie policy 正确；
- cuDNN support 要求 compact `-1` suffix；`S_k<index_topk` 导致的实际 `K_eff` 非 128 倍时，
  必须在 opaque kernel 前明确拒绝（或未来显式 pad 到合法物理 K），不能首批数据才产生隐晦错误；
- full/chunk IDs 相同后，KL 与 wq_b/wk/k_norm/weights_proj 梯度对齐；
- 无论 selection 分几块，fixed-ID loss/backward 第一阶段仍只调用一次；
- 16K `max_memory_allocated` 随 `C*S_k` 降低，并记录吞吐/launch 数。

### 梯度隔离

- KL-only：Indexer grad finite/nonzero，backbone grad 为 None；
- joint：固定 IDs 时 backbone grad 与 frozen baseline 对齐；
- teacher Q/K、Indexer hidden/q_resid 不接收 KL grad；
- AuxLossScaler 前后 carrier 数值相同。

### 分布式校准

- unequal packed MB + grad accumulation；
- multi-source × unequal MB 使用“各 source raw numerator 之和 / 同一 step rows”，不得在 model
  boundary 对 source 或 MB mean 再平均；
- SP1/SP2、intra-layer MB1/MB2；
- 参数更新与等价 unsplit global batch 对齐；
- rows reduce group 与 grad-average group 不匹配时拒绝。

### IndexShare/checkpoint

- source 有 KL、shared 无 KL且 IDs 相同；
- source 输出与所有 shared 输入使用显式同一 IDs tensor；
- frozen Indexer call_count=1；joint project call_count=2、selector call_count=1；
- original/replay IDs 完全相同，original 日志一次、replay 梯度一次；
- checkpoint frame 不跨 layer invocation；同一 multi-MB invocation 的 FIFO entries 和顺序
  相互隔离，missing/unconsumed 均失败；
- main decoder 确实使用 PyTree/frame-aware wrapper；TypedDict/list output round-trip；
- frozen/joint 都只经过 `MoE.fully_shard()` 的同一个 checkpoint wrapper；不存在 default wrapper
  外再套 frame wrapper的 double-checkpoint；
- multi-MB public call 使用 `layer(*hidden_states, ...)`，没有绕过 wrapper；
- P1 IDs 常驻 device；P2 offload 的唯一 ownership/去重/恢复正确，同一 source 不重复 pack；
- `XTUNER_ACTIVATION_OFFLOAD`/正式 config 只解析一次，validator、model forward 与 hook 读取同一
  resolved 值；partial checkpoint + IDs offload 在启动时失败；
- non-reentrant 明确失败。

### backend 与训练

- TileLang 对齐 torch 的 Top-K、KL 和全部 Indexer 参数梯度；
- `cudnn_indexer` 的 XTuner↔cuDNN Q/K/IDs/LSE layout 转换、natural-log LSE、teacher scale 与
  folded student scale 分别正确；32→64 Indexer-head pad 及 backward unpad 后，全部 Indexer 参数
  梯度与 oracle 对齐；
- real kernel 使用标定容差，不要求 atomic bitwise 一致；
- capability mismatch 不 fallback；
- tiny joint KL finite 且下降；
- selector budget 覆盖 padded `[C,S_k]` logits、Top-K scores/int64 indices/scratch、chunk/final
  int32 IDs；source 边界只做一次完整 IDs 校验，shared 不重复物化 `[S,K]` guard workspace；
- production 总峰值预算按真实时序合成 prior-source persistent state 与 selector/guard/loss transient；
  unknown partial-recompute plan 拒绝；
- frozen/joint DCP 是否包含预期 Indexer/model/optimizer state；
- uninterrupted N+1 与 N→save/resume→N+1 的 IDs、loss、Indexer 参数、optimizer moment 对齐；
- mode/topology/policy/HF fingerprint/runtime mismatch 在 `engine.load_dcp` 前拒绝；
- world/mesh/SP/EP/FSDP/grad-acc/global-batch/packing fingerprint 变化时不得称为 exact resume；
- scheduler class/lr_cfg/total-step/warmup policy 变化时在加载 state 前拒绝；
- exact DCP 要同时恢复 optimizer state/args、scheduler、dataset position 和 train state；任一 partial
  restore 均拒绝；legacy 无 sidecar的 strict/best-effort 分支及 async sidecar retention/delete 生命周期；
- 模式/策略切换先导出完整 HF，再以新 optimizer/scheduler/step/data position 启动新 run；
- HF round-trip 后 joint Indexer 权重变化、shared 无独立参数、NoMTP config/keys 一致。

## 18. 预计文件改动

| 文件 | 改动 |
|---|---|
| xtuner/v1/loss/dsa_indexer_loss.py | config、step-global context、公共数学 |
| xtuner/v1/loss/__init__.py | 导出 |
| xtuner/v1/ops/sparse_mla/protocol.py | 统一 Indexer backend facade；旧 IDs selector 降为内部 adapter |
| xtuner/v1/ops/sparse_mla/pytorch.py | tiny reference + workspace guard |
| xtuner/v1/ops/sparse_mla/tilelang.py | no-grad query-chunk Top-K adapter |
| xtuner/v1/ops/sparse_mla/tilelang_indexer_fwd.py | 保持单块 primitive，不保存跨块 logits/scores |
| xtuner/v1/ops/sparse_mla/tilelang*.py | production fixed-ID/streaming loss adapter |
| xtuner/v1/ops/sparse_mla/cudnn_dsa_indexer_loss.py | 复用 PR #2022 的 score-recompute/custom backward |
| xtuner/v1/model/utils/checkpointing.py | PyTree output、selection-only FIFO frame 与 wrapper 接线 |
| xtuner/v1/model/moe/glm52/dsa_mla.py | canonical scale、trainability、teacher、loss attach、显式 IDs |
| xtuner/v1/model/moe/glm52/{decoder_layer,glm52}.py | source/shared 显式 IDs、context 与 model output |
| xtuner/v1/model/moe/moe.py | context/output/single-multi-MB 编排；在 `fully_shard()` 的唯一 site 选择 frame wrapper，并消费统一 resolved offload 开关 |
| xtuner/v1/module/attention/{dsa_mla,dsa_topk_sharing}.py | 未先合入 PR #2039 时，迁移旧实现并删除 mutable-cache runtime |
| xtuner/v1/data_proto/sequence_context.py | 未先合入 PR #2039 时，删除旧 `dsa_topk_cache` ownership，并补等价的 valid-query `mask` 属性供 padding/SP 校准 |
| xtuner/v1/train/trainer.py | 组合/显存/分布式校验、metadata fingerprint、save/resume |
| xtuner/v1/engine/train_engine.py | `build_model()` fully_shard 后、optimizer 构建前校验/cache Indexer AVG group；接收统一 resolved runtime |
| xtuner/v1/module/mtp/{mtp_block,mtp_layer}.py | P2 开放 MTP 时才修改 |
| examples/v1/config/sft_glm5p2.py | joint recipe |
| tests/module/attention/test_dsa_mla.py | 数学、mask、source/shared、replay |
| tests/ops/test_dsa_topk_chunk.py | full/chunk IDs、packed/tail、显存与吞吐 |
| tests/model/test_glm52_moe.py | SP/grad-acc/intra-layer、HF |
| tests/engine/test_glm52_moe_train_engine.py | optimizer/DCP/metadata |
| tests/train/test_glm52_sft_smoke.py | tiny joint smoke |

## 19. 结论

这项工作不是简单删除 requires_grad_(False)。真正需要的是：

~~~text
detached main Q/K teacher
  -> fixed Top-K support KL
  -> step-global calibration
  -> AuxLossScaler
  -> Indexer-only gradient
~~~

最小正确版本先证明 frozen 零回归、短序列数学、梯度隔离和 checkpoint-frame fixed-ID replay，再接
TileLang 长序列 adapter。这样能把算法正确性、训练运行时和高性能 kernel 分层验证，
也不会把当前 hybrid cudnn_dsa 或尚未实现的 TileLang loss 能力说成已经存在。
