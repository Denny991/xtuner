# GLM-5.2 Indexer 联合 SFT 设计

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
- 先用短序列 PyTorch oracle 证明数学，再接长序列 TileLang streaming sparse KL；
- backend 不具备训练能力时立即报错，不允许静默 fallback 后再 OOM。

对应接口伪代码见 [glm52_indexer_sft.py](./glm52_indexer_sft.py)；前期调研见
[glm52_indexer_sft_research.md](../../Znote/glm/glm52_indexer_sft_research.md)。

本设计核对的本地源码根目录统一为 /home/liutong/ZmyCode/：

- XTuner：/home/liutong/ZmyCode/xtuner
- Megatron-LM / Megatron-Core：/home/liutong/ZmyCode/Megatron-LM

GLM-5.2 provider 和公开 recipe 参考 Megatron-Bridge 官方 main。本地没有
Megatron-Bridge checkout，所以本文不伪造它的本地文件链接。

## 2. 第一版非目标

- dense MLA 到 DSA 的独立 Indexer warm-up；
- source Indexer 同时蒸馏多个 shared 层 teacher；
- 独立 Indexer optimizer 或 LR scheduler；
- joint + MTP；
- 把当前 XTuner cudnn_dsa 混合路径称为 full cuDNN DSA；
- 要求 TileLang/cuDNN 原子归约下逐 step grad_norm bitwise 一致。

MTP 不是遗漏。公开 GLM5Bridge recipe 没有可直接照搬的 MTP Indexer-loss 口径，
第一版应明确拒绝，等目标函数定义后再开放。

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

### 3.3 teacher 必须在 attention 内构造

teacher 复用主 MLA 已生成的 query/key。只有 DSAMultiLatentAttention 同时知道：

- main Q/K 与 softmax scale；
- Indexer q/k/weights；
- packed causal 边界和 SP 布局；
- 当前层是 full/source 还是 shared。

Trainer 不应重新跑模型或理解 DSA 数学。

### 3.4 checkpoint replay 目前会跳过 Indexer

当前 reentrant replay 直接复用 original forward 的 Top-K cache。冻结模式下这是正确优化；
联合训练下若不补 fixed-ID KL，replay 没有 Indexer autograd 图，Indexer 会没有梯度。

### 3.5 不能平均 micro-batch mean

packed 后不同 micro-batch 的有效 query 数可能不同。若每个 MB 先求均值，再除
grad-acc 数，目标会随 packing 改变。Indexer KL 必须像 CE loss 一样使用整个
train step 的统一有效 query 分母。

## 4. 算法基线

数学以 Megatron-Core 的 DSA Indexer loss 为基线。

### 4.1 prediction

~~~text
head_score(q,k,h) = ReLU(q_index[q,h] · k_index[k] / sqrt(D_index))
score(q,k)        = sum_h weights[q,h] * head_score(q,k,h)
prediction        = log_softmax(score, key_dim)
~~~

scale ownership 必须唯一：

- weights projection 路径乘一次 H_index 的负二分之一次方；
- QK score 路径乘一次 D_index 的负二分之一次方；
- Top-K 与 loss backend 共用同一口径，不能重复乘 scale。

### 4.2 teacher

~~~text
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
- cache 只保存 IDs，不保存 autograd graph。

## 6. 配置 API

在 DSAMLAConfig 增加：

~~~python
indexer_train_mode: Literal["frozen", "joint"] = "frozen"
indexer_loss_cfg: DSAIndexerLossConfig | None = None
~~~

DSAIndexerLossConfig 采用当前 ZLossConfig 一类的 Pydantic BaseModel 风格，而不是继承
LM-head 专用、带抽象 loss kwargs 接口的 BaseLossConfig：

~~~python
class DSAIndexerLossConfig(BaseModel):
    loss_coeff: float = 0.001
    loss_type: Literal["sparse", "dense"] = "sparse"
    backend: Literal["torch_reference", "tilelang", "cudnn"] = "torch_reference"
    global_average: bool = True
    reference_workspace_limit_bytes: int = 2 * 1024**3
~~~

| 模式 | loss cfg | Indexer 参数 | 行为 |
|---|---|---|---|
| frozen | 必须为 None | 全部 frozen | 复用当前路径 |
| joint | 必须存在且 coeff > 0 | 仅 full/source trainable | LM + Indexer KL |

DSAMLAConfig.build() 当前调用 self.model_dump()。nested config 会被 dump 成 dict，
所以实现时必须 exclude indexer_loss_cfg 后显式传对象。

第一版只支持 joint + AdamW。Muon 会按二维矩阵自动分类 Indexer 参数，语义尚未对齐。
该校验必须放在能同时看到 model、optimizer 和 checkpoint 配置的 Trainer build 层。
同一启动校验同时拒绝 joint + MTP，不能拖到首个 batch 才报错。当前 MoE main decoder
checkpoint 固定使用 reentrant；P1 不新增一个仓库中不存在的 checkpoint_impl 配置。
P1 还要求 model_cfg.compile_cfg=False。当前 loss context 有 Python 状态更新和 shape/row
断言，不能直接放进 decoder fullgraph；compile-safe context/backend 留到 P3。

## 7. backend 与协议

### 7.1 保留 Top-K IDs-only 契约

~~~python
DSATopKIndicesProtocol(q, k, weights, seq_ctx, ...) -> topk_indices
~~~

它仍是 frozen、eval 和推理的稳定协议。

### 7.2 新增 fixed-ID loss 契约

~~~python
DSAIndexerLossProtocol(
    indexer_inputs,
    detached_teacher,
    fixed_topk_indices,
    seq_ctx,
    loss_type=...,
) -> DSAIndexerLossStats(kl_sum, valid_rows)
~~~

普通 forward 先产生 IDs 再算 KL；checkpoint replay 把 original IDs 交给同一 loss
protocol。backend 返回未乘 coeff 的 FP32 kl_sum 和 local 有效行数，不负责训练缩放、
日志、checkpoint 或层选择。

### 7.3 能力矩阵

| loss backend | 当前/目标能力 | 复杂度与用途 | 第一版状态 |
|---|---|---|---|
| torch_reference | fixed-ID sparse；dense 后补 | sparse 约 O(Sq*K)，dense O(Sq*Sk) | tiny correctness |
| tilelang | streaming sparse KL + Indexer grad | 不物化 dense SxS | 待实现 adapter |
| cudnn | full DSA + Indexer loss | 目标 128K | 后续 |

这里的 tilelang 是目标 adapter，不是 XTuner 当前已有能力。Megatron production path
通常融合 Top-K 与 sparse KL；XTuner 还需提供固定 IDs 可重算 KL 的入口，才能服务 replay。
因此 P1 的配置 helper 默认必须是 torch_reference；只有 P3 adapter 实现并通过 capability
probe 后，16K production recipe 才显式改成 tilelang。

当前 Sparse MLA 路径：

| sparse_mla_backend | Top-K | forward | backward |
|---|---|---|---|
| torch | PyTorch | PyTorch | PyTorch |
| tilelang | TileLang | TileLang | TileLang |
| cudnn_dsa | TileLang | TileLang | cuDNN DSA |

因此当前 cudnn_dsa 是 hybrid，不是本设计中的 full loss backend=cudnn。

TileLang adapter 第一版只允许 sparse，并检查 CUDA、BF16、layout、heads、head dim、
Top-K、packed 和 SP。MCore 的 TileLang dIndexK 路径仍可能用 atomic add，所以
与 reference 容差对齐不等于 bitwise deterministic。

## 8. PyTorch oracle 的安全边界

fixed-ID sparse oracle 至少处理：

- 0 <= id < S_global；
- 每个 query 的 packed start/end；
- seq_ctx.mask；
- -1 padding slots；
- 全 invalid row。

全 invalid row 不能直接执行全负无穷上的 softmax。应先放一个 benign sentinel，
softmax 后再用原始 mask 把整行 loss 和梯度归零。对于有效 query，如果没有任何合法
Top-K ID，应直接报错，而不是让 denominator 随 backend 结果变化。
其中 -1 是唯一允许静默忽略的 slot；任何非 -1 的 global 越界、跨 packed sample ID，
即使同一行还存在其他合法 ID，也必须立即报错。

GLM-5.2 teacher KV head 为 1，可按 S,K,D gather，再执行：

~~~python
einsum("shd,skd->shk", teacher_q, teacher_k)
~~~

不能先 expand 成 S,K,H,D 再 materialize。

门禁不能只看 seq_len。峰值由 local Q、global K、Top-K、heads、head dims 和 dtype
共同决定。配置使用 reference_workspace_limit_bytes，op 拿到真实 shape 后做保守估算。
dense oracle 和 PyTorch dense Top-K 仍是二次复杂度。

## 9. DSAIndexer 职责调整

为证明 frozen 兼容，保留 DSAIndexer.forward() 返回 Tensor。新增：

~~~python
project(detached_hidden, detached_q_resid) -> trainable q/k/weights
select_topk(detached_inputs) -> integer IDs
loss_for_indices(inputs, detached_teacher, fixed_ids) -> loss stats
~~~

联合训练路径：

~~~python
inputs = project(hidden.detach(), q_resid.detach())
with torch.no_grad():
    ids = select_topk(inputs.detach())
stats = loss_for_indices(inputs, teacher.detach(), ids)
~~~

只去掉“整个类永远 no-grad”的结构限制；eval/prefill/decode 仍不计算 KL。
attention 在 loss context 为 None 时必须原样走现有 DSAIndexer.forward() 和
CrossLayerTopKSharingRuntime.get_or_compute()，不能先构造 teacher 或进入新的 loss-aware
phase 分支。joint training 如果 context 丢失则立即报错，避免静默退化成 frozen。

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
P1 在 model 内从 Indexer DTensor/FSDP placement 验证 WORLD-average 后直接使用 WORLD；
不向现有 build_loss_ctx_batch() 伪造一个无人传入的 grad_average_group 参数。

所有 contexts 已共用 step denominator，不能再除 batch_size。只有显式
global_average=False 才采用 local mean-of-MB-means。

backend 的 valid_rows 必须和 context 的 local query count 一致。

## 11. loss 挂载与日志

逐 source layer执行：

~~~python
projected_output = AuxLossScaler.apply(projected_output, scaled_kl)
~~~

它让 projected output 数值不变，并在主 loss backward 到达该层时触发 KL backward，
避免把所有层的 KL 图保留到 finalize。

P1 context 只累计 detached indexer_loss。后续可增加：

- indexer_aux_total：所有 source/full invocation 的目标总和；
- indexer_aux_source_mean：按 source layer 平均；
- 可选 indexer_aux_all_layer_mean：shared/no-loss 层记 0，用于对照 MCore tracker。

MCore tracker 按总层数平均；若 XTuner 记录 source loss 总和，两者不能直接横比。

MoEModelOutputs 可增加 indexer_loss: Tensor | None。joint 时它是 detached scalar，
用于现有日志和总 loss 数值展示；梯度只来自 AuxLossScaler，不会 double backward。
frozen 时字段为 None，且不创建 context、不统计 rows、不发通信。

single-MB 和 intra-layer multi-MB 都要 finalize 各自 context，再求和为同一个 output 字段。
P1 暂不把 Python float 诊断塞进 ModelForwardExtraLogInfo；该类要求 Tensor 且新 key 还要注册
reduction 规则。以后增加诊断时，字段名也不要包含 loss，避免被 TrainEngine 当作额外 loss。

## 12. 真实调用链

context 不能只从一个虚构的 Glm52MoE helper 透传。实际需要覆盖：

- MoE.build_loss_ctx_batch
- MoE.forward、_forward、_micro_batch_forward
- DenseDecoderLayer.forward、_forward
- MoEDecoderLayer.forward、_forward、_micro_batch_forward、_pre_moe_forward
- DSAMultiLatentAttention.forward

同时扩展 MoELossContextDict 和 MoEModelOutputs。

DSAMultiLatentAttention 仍返回完整 AttnOutputs；只替换其中的 projected_output carrier，
raw_output 和 softmax_lse 必须原样保留，不能把返回类型缩成一个 Tensor。

P1 拒绝 MTP，所以 MTPBlock/MTPLayer 先不透传 Indexer context；P2 开放时必须补齐
mtp_block.py 和 mtp_layer.py 的 forward、micro-batch、checkpoint 调用链。

TrainEngine 继续使用现有 train_step、_get_total_loss、clip_grad_norm 和 step_optimizer。
不新增第二次 backward，也不能用简化脚本绕过 invalid-grad、zero-grad 或 skip-threshold。

## 13. IndexShare 与 checkpoint

### 13.1 IndexShare

- full/source 层用本层 main Q/K teacher 训练本层 Indexer；
- shared 层只复用 source IDs，不计算 KL；
- 不把 shared 层 teacher 聚合回 source；
- 跨 PP stage share 继续不支持。

### 13.2 cache ownership

现有 CrossLayerTopKSharingRuntime 继续独占 seq_ctx.dsa_topk_cache、residency、offload、
released_sources、release plan 和 MTP counters。

不能新建按 id(seq_ctx) 索引的第二份 cache。runtime 只新增 loss-aware resolve 或只读
execution phase；get_or_compute 维持现有 cache 实现。after_sparse_mla_use 已由 decoder
post-hook 调用，attention 不能再调用一次，否则会双减计数或提前释放。

### 13.3 reentrant checkpoint

~~~text
NORMAL:
  计算/复用 IDs -> 可导 fixed-ID KL -> attach -> 记录一次日志

CHECKPOINT_ORIGINAL（no_grad）:
  计算并缓存 IDs -> no-grad fixed-ID KL -> 只记录 detached 日志

CHECKPOINT_REPLAY（grad）:
  读取同一 IDs -> 重算可导 fixed-ID KL -> attach -> 不重复日志
~~~

phase 必须在 runtime 内用现有阶段判断解析。调用方不能传一个默认 NORMAL 参数，
因为 reentrant original/replay 捕获相同的外层参数。

当前 non-reentrant 无法可靠区分 original/replay，第一版启动时拒绝。

现有单个 checkpoint_active 还不够。默认 78 层、freq=4、offset=3 时，source layer 74
服务到 layer 77；最后一层 77 不做 checkpoint，是 grad-enabled consumer。若只在
“no-grad 的最后 consumer”处设置全局 flag，source 74 replay 会被误判 NORMAL，重新计算
IDs 并重复日志。

因此 DSATopKCacheState 需要 per-source checkpoint_original_sources：

1. source original 在 no-grad 下生成 IDs 后立刻写 source marker；
2. 后续普通 shared consumer 即使 grad-enabled，也只读 cache，不清 marker；
3. source 在 grad-enabled 再次进入时，依据自身 marker 判定 replay并读取同一 IDs；
4. 更高层 backward 已结束后，source replay 复用现有 residency/recompute release
   清理 cache，同步 released_sources，再删除 marker。

该状态仍放在现有 SequenceContext cache 中，不创建第二套 runtime cache。测试必须覆盖
“checkpoint source + 非 checkpoint last consumer”这个默认配置边界。
marker 只由 joint 的 loss-aware wrapper 写入；frozen/eval 继续调用原 get_or_compute，
不能因为修 joint phase 而改变 frozen cache 状态机。

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

真实 DCP state 当前只有 model/optimizer：

- frozen + dcp_ignore_frozen_params=True：Indexer 可省略，但恢复前必须从同一 HF base 补齐；
- joint：Indexer trainable，必须进入 model state 和 optimizer state。

frozen 与 joint 会改变 optimizer 参数集合，禁止 full optimizer resume。阶段切换必须走
显式 HF/model-only 初始化并重建 optimizer。

### 15.2 metadata

建议在 checkpoint root 新增 indexer_sft.json，记录 mode、loss policy、HF base 和
IndexShare pattern。Trainer 保存时写入，_load_checkpoint 在 engine.load_dcp 前校验。

sidecar 由 rank0 原子写入，并接入现有 checkpoint barrier、async-DCP 完成通知、保留和删除
生命周期，不能留下半写文件或孤儿 metadata。解析后的 effective source/full layer mapping
必须保存，不能假设 indexer_types 总是非空。

要区分：

- mode/参数拓扑不匹配：state 与 optimizer 结构不兼容，必须拒绝；
- coeff/type/backend 不匹配：schema 可能相同，但不属于精确轨迹 resume，默认拒绝；
- 显式 model-only 初始化：可允许策略变化，但必须新建 optimizer/scheduler。

历史 frozen checkpoint 没有该 sidecar：current mode 仍是 frozen 且同一 HF base/load_from
可用时按 legacy frozen 恢复；current mode 是 joint 时必须拒绝 full optimizer resume。

### 15.3 HF

HF export 始终保存 full/source Indexer 权重。joint 后验证权重相对 base 已变化；
shared 层仍无独立 indexer 参数。

## 16. 实施阶段

### P0：frozen-compatible 重构

- 新增配置与 Trainer 组合校验；
- 保留 DSAIndexer.forward 返回 Tensor；
- 抽出 project/select_topk/loss_for_indices；
- frozen 不建 context、不发 row-count 通信；
- 原 Top-K/Sparse MLA/compile/checkpoint 回归 bitwise 不变。

### P1：tiny PyTorch joint correctness

- fixed-ID sparse oracle；
- safe packed mask 与 all-invalid row；
- step-global loss context；
- 显式透传到 attention；
- AuxLossScaler 挂载；
- source/full self-teacher；
- NoMTP 权重、AdamW、compile_cfg=False、eager op + reentrant checkpoint；
- helper 默认 backend=torch_reference，TileLang 只能在 P3 recipe 显式开启。

### P2：runtime 组合

- IndexShare full/shared；
- SP、unequal packed grad accumulation、intra-layer micro-batch；
- checkpoint cached-ID replay；
- DCP/HF round-trip；
- 定义 MTP objective 后再扩 MTP 调用链。

### P3：TileLang production sparse KL

- 实现 XTuner layout 的 streaming/fixed-ID adapter；
- capability probe；
- 与 torch oracle 比较 KL 和全部 Indexer 参数梯度；
- 16K 显存、吞吐、compile、activation offload；
- production recipe 禁止 torch reference。

### P4：增强

- full cuDNN DSA + Indexer loss；
- dense warm-up；
- 独立 Indexer LR/param group；
- 多层 teacher；
- 128K packed SFT。

## 17. 测试矩阵

### frozen 与配置

- 默认不创建 context，optimizer 不含 Indexer；
- frozen Top-K、loss、主模型梯度与改动前一致；
- joint + MTP/Muon/compile/无 backend 能力均启动即失败；若未来暴露 main non-reentrant 入口也必须拒绝；
- P1 helper 在 NoMTP 上默认选择 torch_reference；标准含 MTP HF 权重给出明确转换错误；
- nested config build 后仍是 config 对象。

### 数学与 mask

- KL 方向对齐独立 oracle；
- causal、packed、padding、-1；
- global 越界和跨 packed sample ID 均拒绝；
- all-invalid row forward/backward 无 NaN；
- 有效 query 无合法 ID 时失败；
- H/D scale 各乘一次；
- workspace guard 按实际 shape 生效。

### 梯度隔离

- KL-only：Indexer grad finite/nonzero，backbone grad 为 None；
- joint：固定 IDs 时 backbone grad 与 frozen baseline 对齐；
- teacher Q/K、Indexer hidden/q_resid 不接收 KL grad；
- AuxLossScaler 前后 carrier 数值相同。

### 分布式校准

- unequal packed MB + grad accumulation；
- SP1/SP2、intra-layer MB1/MB2；
- 参数更新与等价 unsplit global batch 对齐；
- rows reduce group 与 grad-average group 不匹配时拒绝。

### IndexShare/checkpoint

- source 有 KL、shared 无 KL且 IDs 相同；
- original 与 replay IDs 完全相同；
- original 日志一次，replay 梯度一次；
- checkpoint source 的最后 consumer 不 checkpoint 时仍复用 IDs，覆盖默认 layer 74 -> 77；
- cache/offload/release 无泄漏；
- non-reentrant 明确失败。

### backend 与训练

- TileLang 对齐 torch 的 Top-K、KL 和全部 Indexer 参数梯度；
- real kernel 使用标定容差，不要求 atomic bitwise 一致；
- capability mismatch 不 fallback；
- tiny joint KL finite 且下降；
- DCP resume、optimizer state、metadata 和 HF round-trip。

## 18. 预计文件改动

| 文件 | 改动 |
|---|---|
| xtuner/v1/loss/dsa_indexer_loss.py | config、step-global context、公共数学 |
| xtuner/v1/loss/__init__.py | 导出 |
| xtuner/v1/ops/sparse_mla/protocol.py | fixed-ID loss protocol |
| xtuner/v1/ops/sparse_mla/pytorch.py | tiny reference + workspace guard |
| xtuner/v1/ops/sparse_mla/tilelang*.py | production adapter |
| xtuner/v1/module/attention/dsa_mla.py | trainability、teacher、loss attach |
| xtuner/v1/module/attention/dsa_topk_sharing.py | phase/loss-aware resolve，不新建 cache |
| xtuner/v1/data_proto/sequence_context.py | cache 增加 per-source checkpoint marker |
| xtuner/v1/module/decoder_layer/dense_decoder_layer.py | optional context 透传 |
| xtuner/v1/module/decoder_layer/moe_decoder_layer.py | optional context 透传 |
| xtuner/v1/model/moe/moe.py | context build、output、single/multi-MB 编排 |
| xtuner/v1/model/moe/glm52.py | GLM-5.2 配置/HF 行为 |
| xtuner/v1/train/trainer.py | 组合校验、metadata save/resume |
| xtuner/v1/module/mtp/{mtp_block,mtp_layer}.py | P2 开放 MTP 时才修改 |
| examples/v1/config/sft_glm5p2.py | joint recipe |
| tests/module/attention/test_dsa_mla.py | 数学、mask、source/shared、replay |
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

最小正确版本先证明 frozen 零回归、短序列数学、梯度隔离和 cached-ID replay，再接
TileLang 长序列 adapter。这样能把算法正确性、训练运行时和高性能 kernel 分层验证，
也不会把当前 hybrid cudnn_dsa 或尚未实现的 TileLang loss 能力说成已经存在。
