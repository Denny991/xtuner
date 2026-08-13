# GLM-5.2 DSA / MLA 技术附录

> **主文**：[GLM-5.2-技术汇报.md](./GLM-5.2-技术汇报.md)  
> **代码入口**：`nemo_automodel/components/models/glm_moe_dsa/layers.py`

---

## 1. MLA 基础资料

建议有相关的基础(**MHA->MLA**)

如果对 MLA 本身不熟，建议先看这两篇，再继续看后面的矩阵流水；如果已经理解 DeepSeek MLA，可以直接跳过。

| 资料 | 链接 |
|------|------|
| 了解 MLA：MLA | <https://zhuanlan.zhihu.com/p/19585986234> |
| MLA 计算流全图解 & 吸收矩阵对比分析 | <https://zhuanlan.zhihu.com/p/1954817905456808453> |

---

## 2. 读图方式

- **节点**：算子名 → `in → out` shape → 示例数值（B=1, S=4096）
- **箭头文字**：这一步做什么 + 维度怎么变
- **线条颜色**：蓝=Block，黄=Q 下投影，橙=Indexer，绿=MLA，紫=Attention
- `GlmMoeDsaMLA.forward()` 收到的 `x` 已是 `input_layernorm(h)`；完整 Block 还包括 attention residual、`post_attention_layernorm`、MLP/MoE residual。

一句话版：

```text
Indexer 负责选 key 位置，MLA 负责在这些位置上做低秩 attention。
两者共享 q_resid 起点，但 Indexer 的 K 和 MLA 的 KV 是两条不同支路。
```

---

## 3. 完整数据流图

```mermaid
flowchart TB
    subgraph BLOCKIN[Block 输入]
        HRAW[① block hidden h<br/>in/out: B x S x h_dim<br/>例 1 x 4096 x 6144]
        HRAW -->|6144维 上一层输出| LNIN[② input_layernorm<br/>in/out: 1 x 4096 x 6144<br/>RMSNorm 稳定每 token 尺度]
        LNIN -->|6144→6144 归一化后不变| H0[③ x 送入 Attention<br/>in/out: 1 x 4096 x 6144<br/>GlmMoeDsaMLA 入口]
    end

    subgraph QDOWN[Q 下投影 两条路共用]
        H0 -->|6144→2048 把 hidden 压到 Q 低秩| QA[④ q_a_proj + q_a_layernorm<br/>in 6144 → out 2048<br/>例 1 x 4096 x 2048]
        QA -->|2048→2048 RMSNorm| QR[⑤ q_resid 共用残差<br/>in/out: 1 x 4096 x 2048<br/>Indexer 与 MLA 分叉点]
    end

    subgraph IDX[Indexer 支路 只产 topk_indices]
        QR -->|2048→32x128=4096 扩成 Indexer Q| WQB[⑥ wq_b 线性<br/>in 2048 → out 4096 flat<br/>再 view 成 32 x 128]
        WQB -->|4096→32x128 reshape| QIS[⑦ split + Indexer RoPE<br/>in 128 → 64 rope + 64 nope<br/>Indexer 布局 rope 在前]
        QIS -->|128→128 加位置信息| QI[⑧ Q_index<br/>out: 1 x 4096 x 32 x 128<br/>32 head 各 128 维]
        H0 -->|6144→128 从 x 算共享 K| WK[⑨ wk + LayerNorm<br/>in 6144 → out 128<br/>所有 idx head 共用一条 K]
        WK -->|128→128 LN + split| KIS[⑩ split + Indexer RoPE<br/>64 rope + 64 nope]
        KIS -->|128→128| KI[⑪ K_index<br/>out: 1 x 4096 x 128<br/>每个 key 位置 128 维]
        H0 -->|6144→32 学每个 head 权重| WH[⑫ weights_proj<br/>out: 1 x 4096 x 32]
        QI -->|matmul Q K转置<br/>4096 query 对 4096 key| MAT[⑬ per-head scores<br/>out: 1 x 4096 x 32 x 4096]
        KI -->|K 广播到 32 head| MAT
        WH -->|32 维 head 权重| WSUM[⑭ ReLU + 加权求和<br/>32 x 4096 → 4096 x 4096<br/>合成最终打分矩阵]
        MAT -->|ReLU 激活后加权| WSUM
        WSUM -->|4096x4096→4096x4096| CAU[⑮ mask<br/>padding 置 finfo.min<br/>causal: 只看 j 小于等于 i]
        CAU -->|每行取 top-k| TK[⑯ topk<br/>out: 1 x 4096 x 2048<br/>每 query 2048 个 key 下标]
        TK -->|整数索引 非向量| TOPK[⑰ topk_indices<br/>out: 1 x 4096 x 2048 int<br/>内容是 key 位置 j]
    end

    subgraph MLA[MLA 主支路]
        QR -->|2048→64x256=16384 扩成 MLA Q| QBP[⑱ q_b_proj<br/>in 2048 → out 16384<br/>view 成 64 head x 256]
        QBP -->|256→192 nope + 64 pe| QS[⑲ split + MLA RoPE<br/>MLA 布局 nope 在前]
        QS -->|192+64→256| QD[⑳ Q_dense<br/>out: 1 x 4096 x 64 x 256<br/>dense/SDPA 路径用]
        H0 -->|6144→512+64 压 KV 到 latent| KVA[㉑ kv_a_proj_with_mqa<br/>out: 1 x 4096 x 576<br/>512 kv + 64 k_pe]
        KVA -->|576→512+64 split| KVS[㉒ split kv 与 k_pe]
        KVS -->|512→512 RMSNorm| CKV[㉓ kv latent<br/>out: 1 x 4096 x 512<br/>低秩 KV 表示]
        KVS -->|64→64 RoPE| KPE[㉔ k_pe 带位置<br/>out: 1 x 4096 x 64]
        CKV -->|512→64x448 dense 上投影| KVB[㉕ kv_b_proj<br/>out 64 x 192 K + 64 x 256 V]
        KVB -->|448→192+256 split| DKV[㉖ split k_nope 与 v]
        DKV -->|192+64→256 拼 K| KD[㉗ K_dense<br/>out: 1 x 4096 x 64 x 256]
        KPE -->|64 维 broadcast 到 64 head| KD
        DKV -->|256 维 value| VD[㉘ V_dense<br/>out: 1 x 4096 x 64 x 256]
        QS -->|192→512 吸收 W_kc 仅 TileLang| QABS[㉙ q_nope absorb<br/>q_abs = q_nope x W_kc<br/>192→512 per head]
        QABS -->|512+64→576 latent Q| QTL[㉚ Q_latent THD<br/>out: 4096 x 64 x 576<br/>TileLang 路径用]
        CKV -->|512+64→576 拼 latent| KVLAT[㉛ KV_latent THD<br/>out: 4096 x 1 x 576<br/>不先展开全量 K/V]
        KPE -->|64 维并入 latent| KVLAT
    end

    subgraph ATTN[Attention 计算]
        TOPK -->|2048 个位置→S x S mask| SDPA[㉜ SDPA/TE dense<br/>非 topk 填 finfo.min<br/>仍算全 S 但大部分被 mask]
        QD -->|Q 1x4096x64x256| SDPA
        KD -->|K 1x4096x64x256| SDPA
        VD -->|V 1x4096x64x256| SDPA
        TOPK -->|只 gather 2048 个 latent 位| TL[㉝ TileLang sparse MLA<br/>O 约 S x k 而非 S x S]
        QTL -->|4096x64x576| TL
        KVLAT -->|4096x1x576| TL
        SDPA -->|64x256 per head 输出| OAT[㉞ attn_out<br/>dense: 1 x 4096 x 64 x 256<br/>tilelang: 4096 x 64 x 256]
        TL -->|W_vc 还原 v 维| OAT
    end

    OAT -->|16384→6144 合并 head| OP[㉟ o_proj<br/>in 64x256 → out 6144<br/>head 维拍平后线性]
    OP -->|6144→6144| ATTOUT[㊱ attention 输出<br/>out: 1 x 4096 x 6144<br/>与 block hidden 同维]

    subgraph BLOCKOUT[Block 输出]
        HRAW -->|6144 skip 直连| ADD1[㊲ residual add<br/>in h 6144 + attn 6144<br/>out: 1 x 4096 x 6144]
        ATTOUT -->|6144 加回主路径| ADD1
        ADD1 -->|6144→6144| LNPOST[㊳ post_attention_layernorm<br/>in/out: 1 x 4096 x 6144]
        LNPOST -->|dense: 6144→12288→6144<br/>MoE expert: 6144→2048→6144| FFN[㊴ MLP 或 MoE<br/>前 3 层 dense MLP<br/>其余层 MoE: 256 routed + 1 shared, top-8<br/>out 仍 6144]
        FFN -->|6144 FFN 输出| ADD2[㊵ block output<br/>in/out: 1 x 4096 x 6144<br/>送下一层 Block]
        ADD1 -->|第二个残差支路| ADD2
    end

    linkStyle 0,1,45,47,48,49,50 stroke:#1565C0,stroke-width:2.5px
    linkStyle 2,3 stroke:#F9A825,stroke-width:2.5px
    linkStyle 4,5,6,7,8,9,10,11,12,13,14,15,16,17 stroke:#EF6C00,stroke-width:2.5px
    linkStyle 18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33 stroke:#2E7D32,stroke-width:2.5px
    linkStyle 34,35,36,37,38,39,40,41,42,43,44,46 stroke:#7B1FA2,stroke-width:2.5px
```

---

## 4. 按颜色读主线

| 颜色 | 起点 | 维度变化链 | 最终产出 |
|------|------|------------|----------|
| 蓝 | `h` skip + `attn_out` | `6144 + 6144 → 6144`（残差）→ dense MLP / MoE → 再残差 | 下一层 Block 输入 |
| 黄 | `x` | `6144 → 2048`（q_a）→ `2048`（q_resid） | 供 Indexer / MLA 共用的 Q 低秩表示 |
| 橙 | `x` + `q_resid` | `6144 → 128`（K_index）+ `2048 → 32×128`（Q_index）→ 打分 `4096×4096` → topk | `topk_indices [1,4096,2048]` 整数位置 |
| 绿 | `x` + `q_resid` | `6144 → 512+64`（KV 压缩）+ `2048 → 64×256`（Q_mla）→ dense 展开 K/V 或 TileLang latent | `Q/K/V` 或 `Q_latent/KV_latent` |
| 紫 | `topk_indices` + MLA 输出 | topk → mask / gather → `64×256` head 输出 → `16384 → 6144` | `attn_out [1,4096,6144]` |

**Indexer vs MLA 的 Q 别混**：同源 `q_resid [1,4096,2048]`，但 Indexer 走 `wq_b → 32×128`，MLA 走 `q_b_proj → 64×256`；Indexer 的 K 来自 `wk(x)` 而非 MLA 的 K。

---

## 5. 关键维度

| 类别 | 参数 | 值 |
|------|------|----|
| 输入 | hidden_size | 6144 |
| Q 低秩 | q_lora_rank | 2048 |
| KV 低秩 | kv_lora_rank | 512 |
| MLA | num_attention_heads | 64 |
| MLA | qk_nope_head_dim | 192 |
| MLA | qk_rope_head_dim | 64 |
| MLA | qk_head_dim | 256 |
| MLA | v_head_dim | 256 |
| Indexer | index_n_heads | 32 |
| Indexer | index_head_dim | 128 |
| Indexer | index_topk | 2048 |

> HF config 中有 `head_dim=192` 字段；画 MLA 主路时应以 `qk_head_dim=256 = 192 + 64` 为准。

---

## 6. 分支解释

### 6.1 Indexer 支路

```text
q_resid [B,S,2048]
  → wq_b
  → Q_index [B,S,32,128]

x [B,S,6144]
  → wk + LayerNorm
  → K_index [B,S,128]

Q_index @ K_index^T
  → scores [B,S,S]
  → topk_indices [B,S,2048]
```

`topk_indices` 是整数位置编号，表示每个 query 可以看的 key 位置集合。

### 6.2 MLA 支路

```text
q_resid [B,S,2048]
  → q_b_proj
  → Q_mla [B,S,64,256]

x [B,S,6144]
  → kv_a_proj_with_mqa
  → kv_latent [B,S,512] + k_pe [B,S,64]
```

Dense 路径会通过 `kv_b_proj` 展开完整 `K/V`；TileLang 路径会在 latent 空间按 topk 做 sparse MLA。

---

## 7. Dense / SDPA 与 TileLang 路径

| 路径 | 计算方式 | 适用 |
|------|----------|------|
| Dense / SDPA | 展开 `K_dense/V_dense`，用 topk mask 屏蔽非选中位置 | smoke、短序列、排障 |
| TileLang sparse MLA | `q_nope` 吸收到 latent 维度，直接 gather selected latent | 长上下文、正式 recipe |

TileLang 关键形态：

```text
q_nope [T,64,192]
  → absorb W_kc
  → q_absorbed [T,64,512]
  → concat q_pe [64]
  → Q_latent [T,64,576]

kv_latent [T,512] + k_pe [T,64]
  → KV_latent [T,1,576]
```

---

## 8. 代码对应

| 概念 | 代码位置 / 名称 |
|------|-----------------|
| Block | `nemo_automodel/components/models/glm_moe_dsa/model.py::Block` |
| Attention | `GlmMoeDsaMLA` |
| Indexer | `GlmMoeDsaIndexer` |
| Q 低秩支路 | `q_a_proj`、`q_a_layernorm`、`q_b_proj` |
| KV latent 支路 | `kv_a_proj_with_mqa`、`kv_a_layernorm`、`kv_b_proj` |
| Dense sparse mask | `_build_sparse_mask` |
| TileLang sparse MLA | `tilelang_sparse_attention` |
| IndexShare | `indexer_types`、`skip_topk`、`prev_topk_indices` |

---

## 9. 容易讲错的点

| 误区 | 正确口径 |
|------|----------|
| `topk=2048` 是选 2048 个维度 | 是选 2048 个 key 位置 |
| DSA 替代 attention | DSA 只选位置，MLA 才做 attention |
| Indexer K 就是 MLA K | 两条支路，投影不同 |
| Indexer head 数应等于 MLA head 数 | Indexer 是 32 heads，MLA 是 64 heads |
| `head_dim=192` 就是 MLA QK head | MLA 主路应看 `qk_head_dim=256` |
