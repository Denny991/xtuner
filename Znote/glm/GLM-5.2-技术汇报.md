# GLM-5.2 技术汇报：关键架构与单机验证

> **定位**：GLM-5.2 架构与验证策略说明。  
> **附录**：[GLM-5.2-DSA-MLA-技术附录.md](./GLM-5.2-DSA-MLA-技术附录.md)  
> **更新**：2026-07-09

---

## 1. 核心结论

GLM-5.2 是约 **355B 参数 MoE 模型**，在 NeMo AutoModel 中对应 `GlmMoeDsaForCausalLM`。它不是普通的 “标准 Attention + Dense MLP” 结构，核心差异集中在三点：

```text
DSA 决定“看哪些 token”
MLA 决定“怎么低成本表示 Q/KV”
MoE 决定“参数规模很大，但每个 token 只激活一小部分”
```

| 维度 | 结论 |
|------|------|
| 模型规模 | 78 层，hidden_size=6144，约 355B MoE |
| Attention | DSA Indexer + MLA，不是标准 MHA |
| 稀疏选择 | 每个 query 从序列位置中选 `index_topk=2048` 个 key |
| FFN | 前 3 层 dense MLP，后 75 层 MoE |
| 单机 8xH200 | 适合减层 smoke，不能代表完整 78 层正式训练 |

工程判断：

```text
单机 8xH200 的合理目标是验证“代码路径 + 权重加载 + EP/MoE + loss 有限”，
不是直接完成完整 GLM-5.2 训练。
```

---

## 2. 架构定位

标准 Transformer Block 通常可以简化为：

```text
Attention 看全序列 → Dense MLP 做逐 token 变换
```

GLM-5.2 的 Block 更接近：

```text
DSA 先选 key 位置
  ↓
MLA 用低秩 Q/KV 做 sparse attention
  ↓
MoE 让 token 只经过少量专家
```

| 模块 | 常规做法 | GLM-5.2 做法 | 目的 |
|------|----------|--------------|------|
| Attention 范围 | 每个 token 看全部历史 token | DSA 先选 top-k key 位置 | 降低长上下文 attention 成本 |
| Q/K/V 表示 | hidden 直接投影到全维 Q/K/V | MLA 低秩 Q 与 KV latent | 降低 KV 表示和缓存压力 |
| FFN | 每层一个 dense MLP | 前 3 层 dense，后 75 层 MoE | 扩大参数规模，同时控制激活计算 |
| 长上下文 | 受 `O(S^2)` 限制明显 | sparse + latent + CP/TileLang | 支撑 32K 及更长上下文路线 |

---

## 3. 关键技术

| 技术 | 一句话说明 | 关键数字 |
|------|------------|----------|
| DSA Indexer | 为每个 query 预先选择可看的 key 位置 | 32 heads，head_dim=128，topk=2048 |
| MLA | Q/KV 走低秩表示，再展开或直接 sparse attention | q_lora=2048，kv_lora=512 |
| DSA + MLA 协同 | DSA 产位置索引，MLA 在这些位置上做 attention | MLA heads=64，qk/v head_dim=256 |
| IndexShare | 部分层复用上一层 topk，减少重复 indexer 成本 | full/shared 层交替 |
| MoE FFN | 每个 token 只走 top-8 routed experts + shared expert | 256 routed + 1 shared |
| 长上下文训练 | 依赖 sparse attention、latent KV、CP/TileLang | 32K recipe 使用 CP=8 |

### 3.1 DSA：先决定看哪里

DSA 不是 attention 本身，而是一个 **Indexer**。它为每个 query token 选出可 attend 的 key 位置：

```text
hidden x
  → Indexer 打分
  → topk_indices [B, S, 2048]
  → 后续 attention 只看这些 key 位置
```

关键点：

| 问题 | 回答 |
|------|------|
| topk 选的是什么 | 序列里的 key 位置编号，不是向量维度 |
| topk 是否筛 Q | 不筛 Q，只限制 K/V 或 KV latent |
| Indexer 的 K 是否等于 MLA 的 K | 不是，Indexer 有独立 `wk(x)` |

### 3.2 MLA：让 Attention 表示更省

MLA 使用低秩结构表示 Q/KV，避免所有路径都直接物化完整 Q/K/V：

```text
Q 路径：
x [6144] → q_a_proj → q_resid [2048] → q_b_proj → Q [64 × 256]

KV 路径：
x [6144] → kv_a_proj → kv_latent [512] + k_pe [64]
```

| 路径 | 做法 | 适用 |
|------|------|------|
| Dense / SDPA | 展开完整 K/V，再使用 topk mask | 短序列 smoke、排障 |
| TileLang sparse MLA | 不展开全量 K/V，在 `512+64=576` latent 上 gather topk | 长上下文、正式 recipe |

### 3.3 MoE：参数规模与激活计算解耦

GLM-5.2 的 FFN 部分是 **前稠后稀**：

| 层范围 | FFN 类型 | 形态 |
|--------|----------|------|
| 0-2 | Dense MLP | 6144 → 12288 → 6144 |
| 3-77 | MoE | 256 routed experts + 1 shared expert |

MoE 层中，每个 token 的路径是：

```text
router 给 256 个专家打分
  → 选择 top-8 routed experts
  → 每个专家内部是 6144 → 2048 → 6144
  → 再加 shared expert
```

因此，355B 参数规模不等价于每个 token 都计算 355B 参数。

---

## 4. 一层 Block 的数据流

一层 Attention 可以简化为三条支路：

```text
输入 x [B,S,6144]
  │
  ├─ q_a_proj + RMSNorm
  │      ↓
  │   q_resid [B,S,2048]
  │      ├─ Indexer：wq_b(q_resid) + wk(x) → topk_indices [B,S,2048]
  │      └─ MLA Q：q_b_proj(q_resid) → Q [B,S,64,256]
  │
  ├─ MLA KV：kv_a_proj(x) → kv_latent [B,S,512] + k_pe [B,S,64]
  │
  └─ Sparse Attention：只看 topk_indices 指向的 key 位置
         ↓
      o_proj → 输出 [B,S,6144]
```

注意三点：

1. `q_resid [2048]` 是 Indexer 与 MLA 的共享分叉点。
2. `topk_indices [B,S,2048]` 存的是 key 位置编号，不是 hidden 维度。
3. Indexer 的 `Q/K` 与 MLA 的 `Q/KV` 是两套支路，不能混写。

完整矩阵流水和 Mermaid 图见：[GLM-5.2-DSA-MLA-技术附录.md](./GLM-5.2-DSA-MLA-技术附录.md)。

---

## 5. 单机 H200 验证边界

| 场景 | 单机 8xH200 是否适合 | 说明 |
|------|----------------------|------|
| 减层 smoke | 适合 | 6 层 target，验证 pipeline |
| 完整 78 层 target | 不建议 | MoE expert 权重驻留压力大 |
| 32K TileLang/CP 正式 SFT | 不适合单机 | 目标是多节点 recipe |

smoke 验证的是工程链路：

```text
本地 GLM-5.2 checkpoint
  → FP8 权重 dequant 到 bf16
  → EP=8 分片 MoE experts
  → 6 层 target forward
  → DSpark draft 训练
  → loss 有限且能跑到 step
```

smoke 不代表：

- 不代表完整 78 层能在单机训练。
- 不代表 32K TileLang 长上下文已经验证。
- 不代表训练出了可用模型。

---

## 6. 关键数字速查

| 类别 | 参数 | 值 |
|------|------|----|
| 模型 | hidden_size | 6144 |
| 模型 | num_hidden_layers | 78 |
| DSA | index_n_heads / index_head_dim | 32 / 128 |
| DSA | index_topk | 2048 |
| MLA | q_lora_rank / kv_lora_rank | 2048 / 512 |
| MLA | attention heads | 64 |
| MLA | qk_head_dim / v_head_dim | 256 / 256 |
| MoE | routed / shared experts | 256 / 1 |
| MoE | num_experts_per_tok | 8 |
| Dense FFN | intermediate_size | 12288 |
| MoE FFN | moe_intermediate_size | 2048 |
| 长上下文 | max_position_embeddings | 1,048,576 |

---

## 7. 结论与下一步

当前阶段应把 GLM-5.2 工作拆成两层目标：

| 阶段 | 目标 | 结果判定 |
|------|------|----------|
| 单机减层 smoke | 跑通 GLM-5.2 target + DSpark draft 的训练链路 | 权重加载成功、loss 有限、step 可持续 |
| 多节点正式验证 | 跑完整 78 层 target，验证 EP/TileLang/CP 长上下文路径 | target forward 稳定、显存可控、长上下文 recipe 可执行 |

优先级建议：

1. 先固化单机 8xH200 smoke 记录，形成可复现实验基线。
2. 再准备多节点 EP 配置，验证完整 78 层 target。
3. 最后推进 32K TileLang sparse MLA / CP 长上下文路线。
