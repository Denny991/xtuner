"""GLM-5.2 DSA Indexer 联合 SFT 设计伪代码。

设计要点：
- frozen 采用显式 DSAIndexerOutput/GLM52AttnOutputs.dsa_topk_ids 数据流。
- joint 在固定 Top-K support 上计算 attention-teacher KL，只更新 Indexer。
- source/shared 通过 decoder 显式传 IDs，不扩展 SequenceContext mutable cache。
- reentrant checkpoint frame 只复用 original IDs；replay 重算可导投影与 fixed-ID KL。
- canonical head_weights 只含 H^-1/2；qk_scale 只含 D^-1/2。
- 可选 query chunk 只优化 no-grad Top-K selection，不分块第一版 loss/backward。
- loss context 使用整个 train step 的有效 query 分母，不做 micro-batch mean 的平均。
- PyTorch 只作有显存门禁的 oracle；长序列 TileLang/cuDNN adapter 未接通前 fail fast。

说明：
- 这是接口和控制流伪代码，不是可直接合入的实现。
- 名称尽量贴近 XTuner 当前模块；省略 HF mapping、FSDP 和 kernel 样板代码。
- 入门伴读见 docs/design/glm52_indexer_sft_beginner_guide.md。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, TypedDict

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict, model_validator
from torch import nn
from typing_extensions import NotRequired

from xtuner.v1.config import AdamWConfig


IndexerTrainMode = Literal["frozen", "joint"]
IndexerLossType = Literal["sparse", "dense"]
IndexerLossBackend = Literal["torch_reference", "tilelang", "cudnn_indexer"]


class DSAIndexerLossConfig(BaseModel):
    """采用 ZLossConfig 风格，而不是继承 LM-head 的 BaseLossConfig。"""

    model_config = ConfigDict(extra="forbid")

    loss_coeff: float = 1e-3
    loss_type: IndexerLossType = "sparse"
    backend: IndexerLossBackend = "torch_reference"
    global_average: bool = True
    reference_workspace_limit_bytes: int = 2 * 1024**3
    # production Indexer 相对主模型可占用的总峰值预算；同时覆盖此前 source
    # 已保存的 autograd state 与当前 selector/guard/loss transient。
    production_total_peak_budget_bytes: int | None = None

    def validate_joint(self) -> None:
        if not math.isfinite(self.loss_coeff) or self.loss_coeff <= 0:
            raise ValueError("joint Indexer SFT requires a finite loss_coeff > 0")
        if self.reference_workspace_limit_bytes <= 0:
            raise ValueError("reference_workspace_limit_bytes must be positive")
        if (
            self.production_total_peak_budget_bytes is not None
            and self.production_total_peak_budget_bytes <= 0
        ):
            raise ValueError("production_total_peak_budget_bytes must be positive")
        if (
            self.backend != "torch_reference"
            and self.production_total_peak_budget_bytes is None
        ):
            raise ValueError(
                "production Indexer backend requires an explicit total-peak budget"
            )

    def build(
        self,
        *,
        local_valid_query_mask: torch.Tensor,
    ) -> "DSAIndexerLossContext":
        return DSAIndexerLossContext(
            self,
            local_valid_query_mask=local_valid_query_mask,
        )


def validate_indexer_training_config(
    mode: IndexerTrainMode,
    loss_cfg: DSAIndexerLossConfig | None,
) -> None:
    if mode == "frozen" and loss_cfg is not None:
        raise ValueError("frozen mode must not build an Indexer loss")
    if mode == "joint" and loss_cfg is None:
        raise ValueError("joint mode requires indexer_loss_cfg")
    if loss_cfg is not None:
        loss_cfg.validate_joint()


class DSAMLAConfig(BaseModel):
    """真实 DSAMLAConfig(MLAConfig) 的字段 diff 与 build 边界。"""

    # 伪代码省略既有 MLA 字段，但新增字段仍按真实 extra='forbid' 语义设计，
    # 不能用 extra='allow' 掩盖 recipe 拼写错误。
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    indexer_train_mode: IndexerTrainMode = "frozen"
    indexer_loss_cfg: DSAIndexerLossConfig | None = None
    indexer_topk_query_chunk_size: int | None = None
    indexer_selector_workspace_budget_bytes: int | None = None
    index_topk: int
    sparse_mla_backend: Literal["torch", "tilelang", "cudnn_dsa"] = "torch"

    def build(self, **runtime_args) -> "DSAMultiLatentAttention":
        validate_indexer_training_config(self.indexer_train_mode, self.indexer_loss_cfg)
        if (
            self.indexer_topk_query_chunk_size is not None
            and self.indexer_topk_query_chunk_size <= 0
        ):
            raise ValueError("indexer_topk_query_chunk_size must be positive")
        if (
            self.indexer_topk_query_chunk_size is not None
            and self.sparse_mla_backend not in {"tilelang", "cudnn_dsa"}
        ):
            raise ValueError("P0 query-chunk Top-K requires a TileLang selector")
        if (
            self.indexer_selector_workspace_budget_bytes is not None
            and self.indexer_selector_workspace_budget_bytes <= 0
        ):
            raise ValueError("selector workspace budget must be positive")
        if (
            self.indexer_train_mode == "joint"
            and self.sparse_mla_backend in {"tilelang", "cudnn_dsa"}
            and self.indexer_selector_workspace_budget_bytes is None
        ):
            raise ValueError(
                "joint TileLang selector requires an explicit workspace budget"
            )

        # 保留现有 main SparseMLA + selector preflight；不能只 probe 新 loss adapter。
        ensure_sparse_mla_and_selector_runtime_available(self.sparse_mla_backend)

        # 当前真实 build 使用 **self.model_dump()。nested BaseModel 会变成 dict，
        # 所以新增配置必须 exclude 后显式传对象，不能假定 module 收到 config 实例。
        kwargs = self.model_dump(
            exclude={
                "indexer_train_mode",
                "indexer_loss_cfg",
                "indexer_topk_query_chunk_size",
                "indexer_selector_workspace_budget_bytes",
            }
        )
        indexer_backend = build_indexer_backend(
            selection_backend=self.sparse_mla_backend,
            query_chunk_size=self.indexer_topk_query_chunk_size,
            selector_workspace_budget_bytes=(
                self.indexer_selector_workspace_budget_bytes
            ),
            index_topk=self.index_topk,
            loss_cfg=self.indexer_loss_cfg,
        )
        return DSAMultiLatentAttention(
            **kwargs,
            **runtime_args,
            indexer_train_mode=self.indexer_train_mode,
            indexer_loss_cfg=self.indexer_loss_cfg,
            indexer_backend=indexer_backend,
            indexer_selector_workspace_budget_bytes=(
                self.indexer_selector_workspace_budget_bytes
            ),
        )


class SequenceContext(Protocol):
    mask: torch.Tensor
    sequence_parallel_mesh: Any

    def packed_causal_query_ranges(
        self, query_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def ensure_sparse_mla_and_selector_runtime_available(backend: str) -> None: ...


class DSAIndexerInputs(NamedTuple):
    """可导 Indexer 投影；输入 hidden/q_resid 已 detach。"""

    q: torch.Tensor  # [B, S_q_local, H_index, D_index]
    k: torch.Tensor  # [B, S_k_global, D_index]
    # canonical FP32 表示：weights_proj / sqrt(H_index)，不含 D_index scale。
    head_weights: torch.Tensor  # [B, S_q_local, H_index]
    index_head_dim: int

    @property
    def qk_scale(self) -> float:
        return self.index_head_dim**-0.5

    def effective_weights(self, dtype: torch.dtype) -> torch.Tensor:
        """仅供 sm_scale=1 backend adapter；保留 autograd。"""

        return (self.head_weights * self.qk_scale).to(dtype)

    def detached(self) -> "DSAIndexerInputs":
        return DSAIndexerInputs(
            self.q.detach(),
            self.k.detach(),
            self.head_weights.detach(),
            self.index_head_dim,
        )


class DSAIndexerOutput(TypedDict):
    dsa_topk_ids: torch.Tensor  # contiguous int32 [S_q,1,K]


class DSAIndexerTeacher(NamedTuple):
    # 与当前 absorbed DSA-MLA 的真实 layout 一致：teacher 不带 batch 维。
    query: torch.Tensor  # [S_q_local, H_attention, D_attention]
    key: torch.Tensor  # [S_k_global, H_kv, D_attention]
    softmax_scale: float
    # natural-log LSE [S_q_local,H_attention] from the same fixed-ID SparseMLA.
    # torch reference can recompute the denominator; PR #2022's cuDNN adapter needs it.
    softmax_lse: torch.Tensor | None = None

    def detached(self) -> "DSAIndexerTeacher":
        return DSAIndexerTeacher(
            self.query.detach(),
            self.key.detach(),
            self.softmax_scale,
            None if self.softmax_lse is None else self.softmax_lse.detach(),
        )


class DSAIndexerLossStats(NamedTuple):
    # backend 不乘 coeff、不做跨 MB/rank 平均。
    kl_sum: torch.Tensor
    valid_rows: torch.Tensor


class DSAIndexerBackendProtocol(Protocol):
    """模型层统一 facade；legacy IDs selector 只作为内部 adapter。"""

    def select_topk(
        self,
        indexer: DSAIndexerInputs,
        seq_ctx: SequenceContext,
        *,
        index_topk: int,
    ) -> torch.Tensor: ...

    def loss_on_fixed_topk(
        self,
        indexer: DSAIndexerInputs,
        teacher: DSAIndexerTeacher,
        topk_indices: torch.Tensor,
        seq_ctx: SequenceContext,
        valid_query_mask: torch.Tensor,
        *,
        loss_type: IndexerLossType,
    ) -> DSAIndexerLossStats: ...


def normalize_indexer_target(
    target: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    return target / target.sum(dim=-1, keepdim=True).clamp_min(eps)


def indexer_kl_sum(
    target: torch.Tensor,
    predict_log_probs: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    terms = target * (torch.log(target.clamp_min(eps)) - predict_log_probs)
    return terms.masked_fill(~valid, 0.0).sum()


def validate_fixed_topk_ids_before_sparse(
    topk_indices: torch.Tensor,
    seq_ctx: SequenceContext,
    valid_query_mask: torch.Tensor,
    *,
    kv_len: int,
    workspace_budget_bytes: int | None = None,
) -> torch.Tensor:
    """joint/production 的公共 pre-kernel guard；返回 device-local query mask。

    frozen 仍信任既有 selector 以保持零侵入 baseline。joint 在进入任一 opaque
    SparseMLA/cuDNN kernel 前验证 shape、global bounds、valid-query packed/causal
    boundary、support 非空与 compact padding。数据 padding 行忽略 packed 语义，
    但仍拒绝 OOB；高开销 duplicate 检查只在 tiny/debug 测试执行。
    """

    if topk_indices.ndim != 3 or topk_indices.size(1) != 1:
        raise RuntimeError("dsa_topk_ids must have shape [S_q,1,K_eff]")
    if topk_indices.dtype != torch.int32 or not topk_indices.is_contiguous():
        raise RuntimeError("dsa_topk_ids must be contiguous int32")
    ids = topk_indices[:, 0, :]  # 保持 int32，避免 production 额外物化 [S,K] int64
    q_len = ids.size(0)
    guard_peak = estimate_fixed_topk_guard_peak_bytes(q_len, ids.size(1))
    if workspace_budget_bytes is not None and guard_peak > workspace_budget_bytes:
        raise RuntimeError(
            "fixed-ID pre-kernel guard exceeds its workspace budget: "
            f"required={guard_peak}, budget={workspace_budget_bytes}"
        )
    query_valid = valid_query_mask.reshape(-1)
    if query_valid.numel() != q_len:
        raise RuntimeError("valid_query_mask length disagrees with local query length")
    query_valid = query_valid.to(ids.device, torch.bool)
    starts, ends = seq_ctx.packed_causal_query_ranges(q_len, ids.device)
    padding = ids == -1
    topk_length = (~padding).sum(dim=-1)
    compact_prefix = (~padding) == (
        torch.arange(ids.size(-1), device=ids.device)[None, :]
        < topk_length[:, None]
    )
    torch._assert_async(
        compact_prefix.all(),
        "Top-K -1 padding must be a compact suffix for cuDNN topk_length",
    )
    in_global = (ids >= 0) & (ids < kv_len)
    in_packed = (ids >= starts[:, None]) & (ids < ends[:, None])
    illegal = (~padding & ~in_global) | (
        query_valid[:, None] & ~padding & ~in_packed
    )
    torch._assert_async((~illegal).all(), "Top-K contains OOB/cross-sample IDs")
    legal = in_global & in_packed & query_valid[:, None]
    torch._assert_async(
        ((~query_valid) | legal.any(dim=-1)).all(),
        "a valid query row has no legal Top-K support",
    )
    # TopK selector 本身保证 unique。完整 sort-based duplicate check 会为 16K×2048
    # 额外分配数百 MiB，只放在 tiny/debug capability test；production 不每层执行。
    return query_valid


def estimate_fixed_topk_guard_peak_bytes(q_len: int, k_eff: int) -> int:
    """保守覆盖 source-only guard 的多张 [S,K] bool 与辅助向量。"""

    bool_planes = 10 * q_len * k_eff
    int_vectors = 8 * (2 * q_len + k_eff)
    final_ids = 4 * q_len * k_eff
    return bool_planes + int_vectors + final_ids


def torch_fixed_topk_indexer_loss(
    indexer: DSAIndexerInputs,
    teacher: DSAIndexerTeacher,
    topk_indices: torch.Tensor,
    seq_ctx: SequenceContext,
    valid_query_mask: torch.Tensor,
    *,
    loss_type: IndexerLossType,
) -> DSAIndexerLossStats:
    """短序列 fixed-ID sparse oracle。

    第一阶段 query chunk 只属于 no-grad selection；本函数仍保持一次完整调用。
    loss query/head chunk 是后续独立优化。这里假定 GLM-5.2 的 batch=1、
    teacher KV head=1，只为表达数学和安全 mask。
    """

    if loss_type == "dense":
        return torch_dense_indexer_loss(
            indexer,
            teacher,
            seq_ctx,
            valid_query_mask,
        )
    if indexer.q.size(0) != 1:
        raise NotImplementedError("design oracle assumes packed batch size 1")
    if teacher.key.size(1) != 1:
        raise NotImplementedError("design oracle assumes one shared teacher KV head")

    ids = topk_indices[:, 0, :].to(torch.long)
    q_len, topk = ids.shape
    kv_len = indexer.k.size(1)
    if topk == 0 or kv_len == 0:
        raise RuntimeError("Indexer loss requires non-empty Top-K and KV")

    starts, ends = seq_ctx.packed_causal_query_ranges(q_len, ids.device)
    query_mask = valid_query_mask.reshape(-1)
    if query_mask.numel() != q_len:
        raise RuntimeError("valid_query_mask length disagrees with local query length")
    query_mask = query_mask.to(ids.device, torch.bool)
    query_valid = query_mask & (ends > starts)
    padding = ids == -1
    legal = (
        (ids >= 0)
        & (ids < kv_len)
        & (ids >= starts[:, None])
        & (ids < ends[:, None])
        & query_valid[:, None]
    )
    # -1 是 valid query 内唯一允许的 support padding。数据/SP tail padding query
    # 仍可能由现有 selector 产生正常 causal IDs；这些行不参与 Indexer loss，不能把它们
    # 误判成 backend bug。仅对 valid query 严查越界、跨 packed sample 和非因果 ID。
    illegal = query_valid[:, None] & ~padding & ~legal
    if illegal.any():
        raise RuntimeError("Top-K contains an out-of-range or cross-sample ID")
    valid = legal
    safe_ids = ids.clamp(min=0, max=kv_len - 1)
    valid_rows = valid.any(dim=-1)
    missing_support = query_valid & ~valid_rows
    if missing_support.any():
        raise RuntimeError(
            "a valid query row has no legal Top-K ID; denominator must not depend on backend output"
        )

    # Top-K support 必须是集合。重复合法 ID 会让 gather 实现重复计概率，而 scatter/mask
    # 实现可能折叠，导致 backend 间语义不一致；tiny oracle 直接拒绝。
    ids_for_unique = ids.masked_fill(~valid, kv_len).sort(dim=-1).values
    duplicate = (ids_for_unique[:, 1:] == ids_for_unique[:, :-1]) & (
        ids_for_unique[:, 1:] < kv_len
    )
    if duplicate.any():
        raise RuntimeError("a valid query row contains duplicate Top-K IDs")

    # 全 invalid 行放一个 benign sentinel，使 softmax 输入至少有一个有限值；
    # 最终仍用原 valid mask 把该行的值和梯度严格归零。
    safe_support = valid.clone()
    safe_support[~valid_rows, 0] = True

    q_i = indexer.q[0].float()  # [S,H_i,D_i]
    k_i = indexer.k[0][safe_ids].float()  # [S,K,D_i]，不 expand 到 head 维
    index_logits_h = torch.einsum("shd,skd->shk", q_i, k_i)
    index_logits_h = torch.relu(index_logits_h * indexer.qk_scale)
    index_logits = torch.einsum(
        "shk,sh->sk", index_logits_h, indexer.head_weights[0]
    )
    index_logits = index_logits.masked_fill(~safe_support, float("-inf"))
    predict_log_probs = torch.log_softmax(index_logits, dim=-1)

    teacher_q = teacher.query.float()  # [S,H_a,D_a]
    teacher_k_base = teacher.key[:, 0, :].float()  # [S_k,D_a]
    teacher_k = teacher_k_base[safe_ids]  # [S,K,D_a]
    teacher_logits = torch.einsum("shd,skd->shk", teacher_q, teacher_k)
    teacher_logits = (teacher_logits * teacher.softmax_scale).masked_fill(
        ~safe_support[:, None, :], float("-inf")
    )
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    target = normalize_indexer_target(teacher_probs.sum(dim=1))

    kl_sum = indexer_kl_sum(target, predict_log_probs, valid)
    return DSAIndexerLossStats(kl_sum, valid_rows.sum())


def torch_dense_indexer_loss(
    indexer: DSAIndexerInputs,
    teacher: DSAIndexerTeacher,
    seq_ctx: SequenceContext,
    valid_query_mask: torch.Tensor,
) -> DSAIndexerLossStats:
    # 只作为 tiny-test oracle；未实现前 resolver 不得宣称 dense 可用。
    raise NotImplementedError


def estimate_reference_workspace_bytes(
    indexer: DSAIndexerInputs,
    teacher: DSAIndexerTeacher,
    topk_indices: torch.Tensor,
    *,
    loss_type: IndexerLossType,
) -> int:
    """按实际 local-Q/global-K/head/top-k 做保守运行时估算。"""

    batch, q_len, index_heads, index_dim = indexer.q.shape
    kv_len = indexer.k.size(1)
    attention_heads = teacher.query.size(1)
    attention_dim = teacher.query.size(2)
    support = kv_len if loss_type == "dense" else topk_indices.size(-1)
    fp32_scores = batch * q_len * support * (
        index_heads + attention_heads + 4
    ) * 4
    gathered_vectors = batch * q_len * support * (
        index_dim + attention_dim
    ) * max(indexer.q.element_size(), teacher.query.element_size())
    return 2 * (fp32_scores + gathered_vectors)  # 给 autograd/临时量留安全系数


class TorchReferenceIndexerLoss:
    def __init__(self, config: DSAIndexerLossConfig) -> None:
        self.config = config

    def __call__(
        self,
        indexer,
        teacher,
        topk_indices,
        seq_ctx,
        valid_query_mask,
        *,
        loss_type,
    ):
        required = estimate_reference_workspace_bytes(
            indexer, teacher, topk_indices, loss_type=loss_type
        )
        if required > self.config.reference_workspace_limit_bytes:
            raise RuntimeError(
                "torch_reference Indexer loss exceeds its workspace guard: "
                f"required={required}, "
                f"limit={self.config.reference_workspace_limit_bytes}"
            )
        return torch_fixed_topk_indexer_loss(
            indexer,
            teacher,
            topk_indices,
            seq_ctx,
            valid_query_mask,
            loss_type=loss_type,
        )


class DSAIndexerLossContext(nn.Module):
    """按整个 train step 校准的 layer auxiliary loss context。"""

    def __init__(
        self,
        config: DSAIndexerLossConfig,
        *,
        local_valid_query_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.local_valid_query_mask = local_valid_query_mask.to(torch.bool)
        self.local_valid_rows = self.local_valid_query_mask.sum()
        self.global_valid_rows_step: torch.Tensor | None = None
        self.grad_average_group_size = 1
        self.num_contexts = 1

    @staticmethod
    def build_batches(
        contexts: list["DSAIndexerLossContext"],
        *,
        grad_average_group: dist.ProcessGroup | None,
        collective_device: torch.device,
    ) -> list["DSAIndexerLossContext"]:
        """在 forward 前一次性计算 grad-acc × SP/DP 的 step 分母。

        local_valid_rows 必须由未切分 data 中的 query mask，按与
        SequenceContext.split() 相同的 sp_split 规则得到；不能把完整 mask 在每个 SP
        rank 重复计数。
        """

        if not contexts:
            raise ValueError("Indexer loss context list must not be empty")
        # data_batch/seq_ctx 此时仍可能在 CPU；NCCL collective 必须接收 accelerator
        # tensor。每 step 只搬运一个计数 scalar，不把完整 mask 搬到 GPU。
        rank_rows = torch.stack([ctx.local_valid_rows for ctx in contexts]).sum()
        rank_rows = rank_rows.to(device=collective_device, dtype=torch.int64)
        global_rows = rank_rows.clone()
        group_size = 1
        global_average = contexts[0].config.global_average
        if any(ctx.config.global_average != global_average for ctx in contexts):
            raise ValueError("all Indexer contexts in one step must share global_average")
        if global_average and grad_average_group is not None:
            dist.all_reduce(global_rows, op=dist.ReduceOp.SUM, group=grad_average_group)
            group_size = dist.get_world_size(grad_average_group)
        if global_rows.item() <= 0:
            raise RuntimeError("joint Indexer SFT has no valid query rows in this step")

        for ctx in contexts:
            ctx.global_valid_rows_step = global_rows
            ctx.grad_average_group_size = group_size
            ctx.num_contexts = len(contexts)
        return contexts

    def scale(
        self,
        *,
        stats: DSAIndexerLossStats,
    ) -> torch.Tensor:
        rows_local = stats.valid_rows.to(stats.kl_sum.device, torch.float32)
        expected_rows = self.local_valid_rows.to(rows_local.device, torch.float32)
        # 不能在每个 source/MB 调 .item()，否则 production replay 会反复 GPU-host
        # 同步。真实实现使用 device-side async assert；release build 也可只在 debug
        # capability test 中保留该检查，以 context mask 的计数为权威分母。
        torch._assert_async(
            rows_local == expected_rows,
            "backend valid-row count disagrees with loss calibration",
        )

        if self.config.global_average:
            if self.global_valid_rows_step is None:
                raise RuntimeError("Indexer step-global denominator was not calibrated")
            # XTuner/FSDP 最终对 replica gradient 求平均，因此乘相同 group size。
            # 所有 accumulation contexts 共用同一个 step denominator，不能再除 batch_size。
            scaled = (
                stats.kl_sum
                * self.config.loss_coeff
                * self.grad_average_group_size
                / self.global_valid_rows_step.to(stats.kl_sum.device).clamp_min(1)
            )
        else:
            # 显式选择 local 语义时才采用 mean-of-microbatch-means。
            scaled = (
                stats.kl_sum
                / rows_local.clamp_min(1)
                * self.config.loss_coeff
                / self.num_contexts
            )

        return scaled


class AuxLossScaler(torch.autograd.Function):
    """真实实现复用 xtuner.v1.loss.aux_loss.AuxLossScaler。"""

    @staticmethod
    def forward(ctx, carrier: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(aux_loss)
        return carrier

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (aux_loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(aux_loss)


class DSAIndexer(nn.Module):
    """显式 IDs 输出；joint 使用新增的细粒度方法。"""

    def __init__(
        self,
        *,
        train_mode: IndexerTrainMode,
        loss_cfg: DSAIndexerLossConfig | None,
        backend: DSAIndexerBackendProtocol,
        index_topk: int,
        index_head_dim: int,
        index_n_heads: int,
    ) -> None:
        super().__init__()
        validate_indexer_training_config(train_mode, loss_cfg)
        self.train_mode = train_mode
        self.loss_cfg = loss_cfg
        self.backend = backend
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.wq_b: nn.Module = ...
        self.wk: nn.Module = ...
        self.k_norm: nn.Module = ...
        self.weights_proj: nn.Module = ...
        self.requires_grad_(train_mode == "joint")

    def forward(
        self, hidden_states, q_resid, position_embeddings, seq_ctx
    ) -> DSAIndexerOutput:
        """frozen/eval 返回显式 IDs-only TypedDict。"""

        if self.train_mode != "frozen" and self.training:
            raise RuntimeError("joint training must use the attention-orchestrated path")
        with torch.no_grad():
            inputs = self.project(
                hidden_states, q_resid, position_embeddings, seq_ctx
            )
            ids = self.select_topk(inputs, seq_ctx)
        return {"dsa_topk_ids": ids.to(torch.int32).contiguous()}

    def project(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings,
        seq_ctx: SequenceContext,
    ) -> DSAIndexerInputs:
        q = project_indexer_q(self.wq_b, q_resid, position_embeddings)
        k = project_and_gather_indexer_k(
            self.wk, self.k_norm, hidden_states, position_embeddings, seq_ctx
        )
        # 公共唯一表示：FP32 head_weights 只含 H_index^-1/2。
        head_weights = (
            self.weights_proj(hidden_states).float()
            * (self.index_n_heads**-0.5)
        )
        return DSAIndexerInputs(q, k, head_weights, self.index_head_dim)

    @torch.no_grad()
    def select_topk(
        self, inputs: DSAIndexerInputs, seq_ctx: SequenceContext
    ) -> torch.Tensor:
        inputs = inputs.detached()
        return self.backend.select_topk(
            inputs,
            seq_ctx,
            index_topk=self.index_topk,
        )

    def loss_for_indices(
        self,
        inputs,
        teacher,
        topk_indices,
        seq_ctx,
        valid_query_mask,
    ):
        if self.loss_cfg is None:
            raise RuntimeError("joint Indexer loss backend is missing")
        return self.backend.loss_on_fixed_topk(
            inputs,
            teacher.detached(),
            topk_indices,
            seq_ctx,
            valid_query_mask,
            loss_type=self.loss_cfg.loss_type,
        )


def project_indexer_q(module, q_resid, position_embeddings): ...


def project_and_gather_indexer_k(
    module, norm, hidden_states, position_embeddings, seq_ctx
): ...


def reuse_during_recompute(function, /, *args, **kwargs):
    """真实实现复用 PR #2039 的 checkpoint-local FIFO helper。

    该 helper 只有在主 decoder layer 确实由下面的 PyTree-aware/frame-aware
    checkpoint wrapper 调用时才有 original/replay 状态；单独新增函数不会生效。
    """

    ...


def wrap_glm52_decoder_layer_for_reentrant_checkpoint(layer: nn.Module) -> nn.Module:
    """P0 的硬前置接线，不是普通 torch checkpoint_wrapper 的原样调用。

    目标 wrapper 必须同时：
    1. flatten/unflatten nested input 和 TypedDict/list output；
    2. 为每次 invocation 创建、校验并销毁 checkpoint-local FIFO frame；
    3. original/replay 共享 frame，结束时拒绝 missing/unconsumed entry；
    4. 只允许 frame 保存 requires_grad=False 的 IDs-only PyTree。

    若直接 rebase PR #2039，这里由其 wrapper 提供；否则必须在 P0 等价实现并让
    MoE.fully_shard() 的主 decoder wrapper 显式调用它。
    """

    return pytree_frame_checkpoint_wrapper(layer)


def pytree_frame_checkpoint_wrapper(layer: nn.Module) -> nn.Module: ...


def wrap_decoder_layer_at_moe_fully_shard_site(
    layer: nn.Module,
) -> nn.Module:
    """在 MoE._should_recompute(layer_idx, mtp_idx) 已为真时选择唯一 wrapper。

    真实容器是 ``model.layers``，不是 ``model.decoder_layers``。recompute coverage
    必须继续由现有、拥有 layer_idx/mtp_idx/全局层数的 ``MoE._should_recompute``
    决定；本 helper 不做第二次简化判定。不能先在 Trainer外包一层 frame wrapper，
    再让 MoE.fully_shard() 套一次默认 REENTRANT wrapper；
    double-checkpoint 会破坏 frame ownership。当前 fully_shard 循环必须在原来的
    ``if self._should_recompute(layer_idx, mtp_idx):`` 分支内调用本函数，GLM-5.2
    frozen/joint 都走同一个 PyTree/frame-aware wrapper，其他模型仍走既有默认 wrapper。

    P1 的 IDs 常驻 device。P2 若开启 activation offload，frame 必须成为 IDs 的唯一
    offload owner，并按 tensor identity 去重；当前只匹配 hidden-state data_ptr 的 hook
    不会自动 offload int32 IDs。
    """

    if is_glm52_decoder_layer(layer):
        return wrap_glm52_decoder_layer_for_reentrant_checkpoint(layer)
    return existing_reentrant_checkpoint_wrapper(layer)


def is_glm52_decoder_layer(layer) -> bool: ...


def existing_reentrant_checkpoint_wrapper(layer): ...


class GLM52AttnOutputs(TypedDict):
    raw_output: torch.Tensor
    projected_output: torch.Tensor
    softmax_lse: torch.Tensor
    dsa_topk_ids: torch.Tensor
    indexer_loss: NotRequired[torch.Tensor]


class GLM52DecoderLayerOutputs(TypedDict):
    """现有 decoder payload 的兼容扩展，不能丢掉 MoE router 三元组。"""

    hidden_states: torch.Tensor
    router_logits: NotRequired[torch.Tensor]
    router_weights: NotRequired[torch.Tensor]
    router_topk_ids: NotRequired[torch.Tensor]
    dsa_topk_ids: torch.Tensor
    indexer_loss: NotRequired[torch.Tensor]


class GLM52MicroBatchDecoderOutputs(TypedDict):
    """P2 list 契约；每个位置拥有独立 IDs/loss context/FIFO entry。"""

    hidden_states: list[torch.Tensor]
    router_logits: NotRequired[list[torch.Tensor]]
    router_weights: NotRequired[list[torch.Tensor]]
    router_topk_ids: NotRequired[list[torch.Tensor]]
    dsa_topk_ids: list[torch.Tensor]
    indexer_loss: NotRequired[list[torch.Tensor | None]]


class DSAMultiLatentAttention(nn.Module):
    """只展示 source Indexer loss 的编排；Sparse MLA 主路径沿用当前实现。"""

    layer_idx: int
    source_layer_idx: int
    softmax_scale: float

    def __init__(
        self,
        *,
        indexer_train_mode: IndexerTrainMode,
        indexer_loss_cfg: DSAIndexerLossConfig | None,
        indexer_backend: DSAIndexerBackendProtocol,
        indexer_selector_workspace_budget_bytes: int | None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.indexer_train_mode = indexer_train_mode
        self.indexer: DSAIndexer | None = ...
        self.indexer_backend = indexer_backend
        self.indexer_selector_workspace_budget_bytes = (
            indexer_selector_workspace_budget_bytes
        )
        # checkpoint frame 若按 callable 分桶，必须保存同一个 bound-method 对象，
        # 不能在 original/replay 临时重新取属性后依赖 object identity。
        self._selector_for_recompute = (
            self._select_and_validate_topk if self.indexer is not None else None
        )

    @torch.no_grad()
    def _select_and_validate_topk(
        self,
        inputs: DSAIndexerInputs,
        seq_ctx: SequenceContext,
        valid_query_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self.indexer is not None
        ids = self.indexer.select_topk(inputs, seq_ctx)
        validate_fixed_topk_ids_before_sparse(
            ids,
            seq_ctx,
            valid_query_mask,
            kv_len=inputs.k.size(1),
            workspace_budget_bytes=self.indexer_selector_workspace_budget_bytes,
        )
        return ids

    def forward(
        self,
        hidden_states,
        position_embeddings,
        seq_ctx,
        dsa_topk_ids: torch.Tensor | None = None,
        *,
        indexer_loss_ctx: DSAIndexerLossContext | None = None,
    ) -> GLM52AttnOutputs:
        q_resid, query, key, value_dim = self._project_main_attention(
            hidden_states, position_embeddings, seq_ctx
        )
        is_source = self.source_layer_idx == self.layer_idx
        if self.training and self.indexer_train_mode == "joint" and indexer_loss_ctx is None:
            raise RuntimeError("joint Indexer training lost its loss context")

        scaled_indexer_loss: torch.Tensor | None = None
        joint_indexer_inputs: DSAIndexerInputs | None = None
        valid_query_mask = (
            None
            if indexer_loss_ctx is None
            else indexer_loss_ctx.local_valid_query_mask
        )
        if is_source:
            assert self.indexer is not None
            # source 总是覆盖上一 source 的 IDs。
            if indexer_loss_ctx is None:
                with torch.no_grad():
                    indexer_output = reuse_during_recompute(
                        self.indexer,
                        hidden_states,
                        q_resid,
                        position_embeddings,
                        seq_ctx,
                    )
                dsa_topk_ids = indexer_output["dsa_topk_ids"]
            else:
                # original 在 no_grad 下执行；replay 在 grad 下重新建立这条投影图。
                joint_indexer_inputs = self.indexer.project(
                    hidden_states.detach(),
                    q_resid.detach(),
                    position_embeddings,
                    seq_ctx,
                )
                # 固定 call-site/FIFO：original 真正 select，replay 从 frame 取 IDs。
                assert self._selector_for_recompute is not None
                assert valid_query_mask is not None
                dsa_topk_ids = reuse_during_recompute(
                    self._selector_for_recompute,
                    joint_indexer_inputs.detached(),
                    seq_ctx,
                    valid_query_mask,
                )
        elif dsa_topk_ids is None:
            raise RuntimeError(
                f"shared DSA layer {self.layer_idx} requires explicit dsa_topk_ids"
            )

        assert dsa_topk_ids is not None
        if dsa_topk_ids.dtype != torch.int32 or not dsa_topk_ids.is_contiguous():
            raise RuntimeError("dsa_topk_ids must be contiguous int32")

        # 先运行 fixed-ID SparseMLA，teacher 才能携带与本次 support 完全对应的
        # natural-log softmax_lse。PR #2022 的 cuDNN score-recompute adapter依赖该值；
        # torch reference 虽可重算 denominator，也沿用同一 teacher contract。
        sparse_mla_outputs = self._sparse_mla(
            query, key, dsa_topk_ids, value_dim
        )
        if joint_indexer_inputs is not None:
            assert self.indexer is not None and indexer_loss_ctx is not None
            assert valid_query_mask is not None
            teacher = DSAIndexerTeacher(
                query.detach(),
                key.detach(),
                self.softmax_scale,
                sparse_mla_outputs.softmax_lse.detach(),
            )
            stats = self.indexer.loss_for_indices(
                joint_indexer_inputs,
                teacher,
                dsa_topk_ids,
                seq_ctx,
                valid_query_mask,
            )
            scaled_indexer_loss = indexer_loss_ctx.scale(stats=stats)

        attn_outputs = self._finish_attention(sparse_mla_outputs)
        projected_output = attn_outputs["projected_output"]

        # NORMAL 和 replay 为 grad-enabled，会把 KL 挂进主 backward；checkpoint
        # original 为 no-grad，只返回 detached 展示值。外层 decoder 只聚合 original 输出。
        if scaled_indexer_loss is not None and torch.is_grad_enabled():
            projected_output = AuxLossScaler.apply(
                projected_output, scaled_indexer_loss
            )
        attn_outputs["projected_output"] = projected_output
        attn_outputs["dsa_topk_ids"] = dsa_topk_ids
        if scaled_indexer_loss is not None:
            attn_outputs["indexer_loss"] = scaled_indexer_loss.detach()
        return attn_outputs

    def _project_main_attention(self, hidden_states, position_embeddings, seq_ctx): ...

    def _sparse_mla(self, query, key, ids, value_dim): ...

    def _finish_attention(self, output) -> dict[str, torch.Tensor]:
        # 保留 raw_output / softmax_lse，并完成 absorbed w_vc 与 o_proj。
        ...


def call_glm52_decoder_layer(
    layer,
    hidden_states,
    *,
    dsa_topk_ids: torch.Tensor | None,
    indexer_loss_ctx: DSAIndexerLossContext | None,
):
    """checkpoint wrapper 外聚合 original 输出；replay 不再次经过这里。

    为突出 IDs/loss 契约省略 position_embeddings/seq_ctx 等既有参数，真实调用必须原样透传。
    """

    layer_output: GLM52DecoderLayerOutputs = layer(
        hidden_states,
        dsa_topk_ids=dsa_topk_ids,
        indexer_loss_ctx=indexer_loss_ctx,
    )
    # Glm52MoE 外层继续消费 router logits/weights/top-k IDs 计算 balancing/z loss；
    # 这里只增加 DSA IDs/indexer_loss，不能缩成三元组后丢弃 router payload。
    return layer_output


def call_glm52_decoder_layer_micro_batches(
    layer,
    hidden_states: list[torch.Tensor],
    *,
    dsa_topk_ids: list[torch.Tensor | None],
    indexer_loss_ctx: list[DSAIndexerLossContext | None],
) -> GLM52MicroBatchDecoderOutputs:
    """P2 目标契约；一次 layer invocation 共享 frame，MB 按序使用独立 FIFO entry。

    为突出 list 契约省略 position_embeddings/seq_ctx 等既有参数，真实调用必须原样透传。
    """

    if not (
        len(hidden_states) == len(dsa_topk_ids) == len(indexer_loss_ctx)
    ):
        raise RuntimeError("micro-batch hidden/IDs/loss-context lists must align")
    # 必须调用 checkpoint wrapper 的 public layer boundary；layer.forward 再在 wrapper
    # 内部分派到 _micro_batch_forward。不能直调私有方法，也不能把每个 MB 拆成多个
    # checkpoint invocation。frame 内按 list 顺序保存/恢复 IDs。
    output: GLM52MicroBatchDecoderOutputs = layer(
        *hidden_states,
        dsa_topk_ids=dsa_topk_ids,
        indexer_loss_ctx=indexer_loss_ctx,
    )
    if not (
        len(output["hidden_states"])
        == len(output["dsa_topk_ids"])
        == len(hidden_states)
    ):
        raise RuntimeError("micro-batch output lists must preserve input cardinality")
    for router_key in ("router_logits", "router_weights", "router_topk_ids"):
        if router_key in output and len(output[router_key]) != len(hidden_states):
            raise RuntimeError(f"{router_key} must preserve micro-batch cardinality")
    return output


class MoELossContextDict(TypedDict):
    lm: Any
    balancing: Any
    z_loss: Any
    mtp: Any
    indexer: DSAIndexerLossContext | None


class MoEModelOutputsFieldPatch(TypedDict):
    """只表达对现有 MoEModelOutputs(ModelOutputs) 的字段增量。

    真实实现不能用这个类型替换原类；原有 router/balancing/z/MTP、
    tokens_per_expert_global 和 free_nongrad_feature/post-forward 契约全部保留。
    """

    indexer_loss: NotRequired[torch.Tensor | None]


def build_indexer_contexts_for_step(
    *,
    model,
    data_batch: list[dict],
    sp_mesh,
) -> list[DSAIndexerLossContext] | None:
    """从 Trainer 传入的未切分 data 构造 SP-local mask 与 step-global分母。"""

    cfg = model.config.attention.indexer_loss_cfg
    if cfg is None:
        return None  # frozen：无通信、无 context、无输出 schema 行为变化
    if model.config.mtp_config is not None:
        raise RuntimeError("P1 does not support joint Indexer SFT with MTP")

    # P1 只开放已验证为 WORLD-average 的拓扑。该不变量必须在 fully_shard 后、
    # 首 batch 前验证并缓存；这里仅确认 startup validation 已经完成，不能每 step
    # 重新遍历 DTensor placements 或等到首批数据才发现不兼容。
    require_cached_indexer_world_average_validation(model)
    grad_average_group = dist.group.WORLD if dist.is_initialized() else None
    collective_device = indexer_collective_device(model, grad_average_group)

    contexts = []
    for data in data_batch:
        full_mask = data["seq_ctx"].mask
        local_mask = split_like_sequence_context(full_mask, sp_mesh)
        contexts.append(cfg.build(local_valid_query_mask=local_mask))
    return DSAIndexerLossContext.build_batches(
        contexts,
        grad_average_group=grad_average_group,
        collective_device=collective_device,
    )


def split_like_sequence_context(mask: torch.Tensor, sp_mesh) -> torch.Tensor: ...


def validate_and_cache_indexer_distributed_layout_after_shard(model) -> None:
    """逐 source 验证 FSDP shard × EP/ETP/HSDP replica 的有效 AVG group。

    P1 只有在 rows reduce group 与每个 Indexer 参数最终梯度平均 group 都可证明为
    WORLD 时通过；验证结果存入 model，供每 step 的 context builder O(1) 检查。
    """

    ...


def finalize_glm52_joint_runtime_after_fully_shard(model) -> None:
    """Trainer 在 model.fully_shard() 返回后、构建首个 data batch 前必须调用。"""

    if model.config.attention.indexer_train_mode == "joint":
        validate_and_cache_indexer_distributed_layout_after_shard(model)


def require_cached_indexer_world_average_validation(model) -> None: ...


def indexer_collective_device(
    model, grad_average_group: dist.ProcessGroup | None
) -> torch.device:
    """返回与 Indexer 参数及 WORLD collective backend 兼容的 accelerator。"""
    ...


def finalize_indexer_output(
    *,
    output: dict[str, Any],
    layer_indexer_losses: list[torch.Tensor | None],
    device: torch.device,
) -> None:
    """wrapper 外汇总 original layer outputs；replay 不调用本函数。"""

    values = [value.detach() for value in layer_indexer_losses if value is not None]
    if not values:
        return
    output["indexer_loss"] = torch.stack(
        [value.to(device) for value in values]
    ).sum()


class _TopKSelector(Protocol):
    """facade 内部 SPI；不向 model/attention 暴露。"""

    def __call__(
        self,
        indexer: DSAIndexerInputs,
        seq_ctx: SequenceContext,
        *,
        index_topk: int,
    ) -> torch.Tensor: ...


class _FixedIDLossOp(Protocol):
    def __call__(
        self,
        indexer: DSAIndexerInputs,
        teacher: DSAIndexerTeacher,
        topk_indices: torch.Tensor,
        seq_ctx: SequenceContext,
        valid_query_mask: torch.Tensor,
        *,
        loss_type: IndexerLossType,
    ) -> DSAIndexerLossStats: ...


class ComposedDSAIndexerBackend:
    """统一 facade；selection/loss adapter 的 dtype/layout 差异留在内部。"""

    def __init__(
        self,
        selector: _TopKSelector,
        loss_op: _FixedIDLossOp | None,
    ) -> None:
        self.selector = selector
        self.loss_op = loss_op

    def select_topk(self, indexer, seq_ctx, *, index_topk):
        return self.selector(indexer, seq_ctx, index_topk=index_topk)

    def loss_on_fixed_topk(
        self,
        indexer,
        teacher,
        topk_indices,
        seq_ctx,
        valid_query_mask,
        *,
        loss_type,
    ):
        if self.loss_op is None:
            raise RuntimeError("frozen Indexer backend has no loss adapter")
        return self.loss_op(
            indexer,
            teacher,
            topk_indices,
            seq_ctx,
            valid_query_mask,
            loss_type=loss_type,
        )


class TileLangTopKSelector:
    """第一阶段只 chunk no-grad selection，不 chunk fixed-ID loss/backward。"""

    def __init__(
        self,
        query_chunk_size: int | None,
        workspace_budget_bytes: int | None,
    ) -> None:
        self.query_chunk_size = query_chunk_size
        self.workspace_budget_bytes = workspace_budget_bytes

    @torch.no_grad()
    def __call__(self, indexer, seq_ctx, *, index_topk):
        return tilelang_topk_query_chunked(
            indexer,
            seq_ctx,
            index_topk=index_topk,
            query_chunk_size=self.query_chunk_size,
            workspace_budget_bytes=self.workspace_budget_bytes,
        )


class TorchTopKSelector:
    """joint tiny-only selector；在物化 [B,Sq,H,Sk] 前做动态 workspace guard。"""

    def __init__(self, workspace_limit_bytes: int | None) -> None:
        # frozen + None 保持历史行为；joint 必须传 loss config 的显存预算。
        self.workspace_limit_bytes = workspace_limit_bytes

    @torch.no_grad()
    def __call__(self, indexer, seq_ctx, *, index_topk):
        if self.workspace_limit_bytes is not None:
            batch, q_len, heads, _ = indexer.q.shape
            kv_len = indexer.k.size(1)
            # 当前 torch selector 会物化 per-head score 与 reduced logits；给临时量
            # 留 2x 系数。不能只用 fixed-ID loss 的 O(S*K) estimator。
            required = 2 * batch * q_len * kv_len * (heads + 1) * 4
            if required > self.workspace_limit_bytes:
                raise RuntimeError(
                    "torch Top-K selector exceeds its workspace guard: "
                    f"required={required}, limit={self.workspace_limit_bytes}"
                )
        return torch_topk_selector_adapter(
            indexer, seq_ctx, index_topk=index_topk
        )


@torch.no_grad()
def tilelang_topk_query_chunked(
    indexer: DSAIndexerInputs,
    seq_ctx: SequenceContext,
    *,
    index_topk: int,
    query_chunk_size: int | None,
    workspace_budget_bytes: int | None,
) -> torch.Tensor:
    """沿 Q 分块；每块使用完整 global K，只拼 int32 IDs。"""

    if indexer.q.size(0) != 1 or indexer.k.size(0) != 1:
        raise NotImplementedError("GLM packed Top-K selector assumes batch=1")
    q = indexer.q[0].contiguous()
    k = indexer.k[0].contiguous()
    # TileLang primitive 没有独立 sm_scale 参数，在 adapter 边界折入 D^-1/2。
    effective_weights = indexer.effective_weights(torch.float32)[0].contiguous()

    # 必须基于完整 Q 生成一次 global K ranges；不能在每个 chunk 从位置 0 重建。
    starts, ends = seq_ctx.packed_causal_query_ranges(q.size(0), q.device)
    chunk_rows = q.size(0) if query_chunk_size is None else min(
        query_chunk_size, q.size(0)
    )
    block_q = 128 // q.size(1)
    padded_chunk_rows = round_up(chunk_rows, block_q)
    k_eff = min(index_topk, k.size(0))
    required_workspace = estimate_tilelang_selector_peak_bytes(
        q_len=q.size(0),
        kv_len=k.size(0),
        padded_chunk_rows=padded_chunk_rows,
        k_eff=k_eff,
    )
    if (
        workspace_budget_bytes is not None
        and required_workspace > workspace_budget_bytes
    ):
        raise RuntimeError(
            "TileLang selector workspace exceeds budget before logits allocation: "
            f"required={required_workspace}, budget={workspace_budget_bytes}"
        )
    if query_chunk_size is None or query_chunk_size >= q.size(0):
        # None/>=S 严格走现有 one-shot primitive。
        return _tilelang_topk_from_ranges(
            q, k, effective_weights, starts, ends, index_topk
        )

    # 预分配最终 int32 IDs，避免保留所有 chunk 后 torch.cat 再产生第二份 [S,K]。
    final_ids = torch.empty(
        (q.size(0), 1, k_eff),
        dtype=torch.int32,
        device=q.device,
    )
    for lo in range(0, q.size(0), query_chunk_size):
        hi = min(lo + query_chunk_size, q.size(0))
        q_chunk, weights_chunk, starts_chunk, ends_chunk, valid_rows = (
            _pad_tilelang_query_chunk(
                q[lo:hi],
                effective_weights[lo:hi],
                starts[lo:hi],
                ends[lo:hi],
            )
        )
        # 低层 op 内部完成 logits -> topk；dense logits/topk scores 不跨块返回。
        ids = _tilelang_topk_from_ranges(
            q_chunk,
            k,
            weights_chunk,
            starts_chunk,
            ends_chunk,
            index_topk,
        )
        final_ids[lo:hi].copy_(ids[:valid_rows])
    return final_ids.contiguous()


def estimate_tilelang_selector_peak_bytes(
    *,
    q_len: int,
    kv_len: int,
    padded_chunk_rows: int,
    k_eff: int,
) -> int:
    """保守计 logits、topk score/int64 index、chunk/final IDs 与 backend scratch。"""

    logits = 4 * padded_chunk_rows * kv_len
    topk_scores = 4 * padded_chunk_rows * k_eff
    topk_indices = 8 * padded_chunk_rows * k_eff
    ids = 4 * (padded_chunk_rows + q_len) * k_eff
    scratch = tilelang_topk_scratch_upper_bound(
        padded_chunk_rows=padded_chunk_rows,
        kv_len=kv_len,
        k_eff=k_eff,
    )
    return logits + topk_scores + topk_indices + ids + scratch


def tilelang_topk_scratch_upper_bound(**kwargs) -> int: ...


def _pad_tilelang_query_chunk(q, weights, starts, ends):
    """Pad 到 primitive 的 block_Q 倍数；padding row 使用空 global range。

    当前 kernel 没有完整 q-tail guard。真实实现也可选择先移植 MCore guard；在其中
    任一方案落地前，不能直接把非对齐 chunk 送入 primitive。
    """

    block_q = 128 // q.size(1)
    valid_rows = q.size(0)
    padded_rows = round_up(valid_rows, block_q)
    return pad_q_and_weights_with_zero_and_empty_ranges(
        q, weights, starts, ends, padded_rows
    ) + (valid_rows,)


def round_up(value: int, multiple: int) -> int: ...


def pad_q_and_weights_with_zero_and_empty_ranges(*args): ...


def _tilelang_topk_from_ranges(q, k, weights, starts, ends, index_topk):
    """现有 one-shot primitive 的 IDs-only adapter；不返回 logits/scores。"""

    ...


def build_indexer_backend(
    *,
    selection_backend: Literal["torch", "tilelang", "cudnn_dsa"],
    query_chunk_size: int | None,
    selector_workspace_budget_bytes: int | None,
    index_topk: int,
    loss_cfg: DSAIndexerLossConfig | None,
) -> DSAIndexerBackendProtocol:
    if selection_backend == "torch":
        if query_chunk_size is not None:
            raise RuntimeError("P0 torch selector does not support query chunk")
        selector = TorchTopKSelector(
            None
            if loss_cfg is None
            else loss_cfg.reference_workspace_limit_bytes
        )
    elif selection_backend in {"tilelang", "cudnn_dsa"}:
        # current cudnn_dsa 的 Top-K 仍走 TileLang。
        selector = TileLangTopKSelector(
            query_chunk_size,
            selector_workspace_budget_bytes,
        )
    else:
        raise AssertionError(selection_backend)

    loss_op = (
        build_indexer_loss_backend(loss_cfg, index_topk=index_topk)
        if loss_cfg is not None
        else None
    )
    return ComposedDSAIndexerBackend(selector, loss_op)


def torch_topk_selector_adapter(indexer, seq_ctx, *, index_topk):
    # Torch selector 接收同一 canonical head_weights + qk_scale，不能再按 head_dim
    # 隐式重复缩放。
    return torch_topk_selector_impl(
        indexer.q,
        indexer.k,
        indexer.head_weights,
        seq_ctx,
        qk_scale=indexer.qk_scale,
        index_topk=index_topk,
    )


def torch_topk_selector_impl(*args, **kwargs) -> torch.Tensor: ...


def build_indexer_loss_backend(
    config: DSAIndexerLossConfig,
    *,
    index_topk: int,
) -> _FixedIDLossOp:
    """静态 resolver；动态 shape/workspace 由 op 调用时继续 guard。"""

    if config.backend == "torch_reference":
        if config.loss_type == "dense":
            raise RuntimeError("dense torch oracle is not implemented yet")
        return TorchReferenceIndexerLoss(config)

    if config.backend == "tilelang":
        if config.loss_type != "sparse":
            raise RuntimeError("TileLang Indexer-loss target supports sparse KL only")
        # 当前 XTuner 尚无该 adapter。只有 BF16/layout/head/top-k/packed/SP capability
        # probe 全部通过后才返回，不能借 current TileLang Top-K 名义静默启用。
        ensure_tilelang_fixed_id_loss_available()
        return tilelang_fixed_topk_indexer_loss

    if config.backend == "cudnn_indexer":
        # 这是独立的 fixed-ID Indexer score-recompute/custom-backward adapter，
        # 可复用 PR #2022，并不要求主 SparseMLA 已是 full cuDNN DSA。
        if config.loss_type != "sparse":
            raise RuntimeError("cuDNN Indexer loss supports sparse KL only")
        ensure_cudnn_indexer_loss_available(index_topk=index_topk)
        return cudnn_fixed_topk_indexer_loss

    raise AssertionError(config.backend)


def ensure_tilelang_fixed_id_loss_available() -> None: ...


def ensure_cudnn_indexer_loss_available(*, index_topk: int) -> None:
    """静态检查依赖、SM90/BF16 计划与 K%128；layout/SP 继续运行时 guard。"""

    if index_topk % 128 != 0:
        raise RuntimeError("PR #2022 cuDNN Indexer adapter requires index_topk % 128 == 0")
    ...


def tilelang_fixed_topk_indexer_loss(*args, **kwargs) -> DSAIndexerLossStats: ...


def cudnn_fixed_topk_indexer_loss(
    indexer,
    teacher,
    topk_indices,
    seq_ctx,
    valid_query_mask,
    *,
    loss_type,
) -> DSAIndexerLossStats:
    # cuDNN adapter 使用 sm_scale=1，因此只在边界临时折入 D^-1/2；梯度仍会
    # 经 cast/乘法回到 canonical head_weights 和 weights_proj。
    effective_weights = indexer.effective_weights(torch.bfloat16)
    if teacher.softmax_lse is None:
        raise RuntimeError(
            "cuDNN Indexer loss requires natural-log SparseMLA softmax_lse"
        )
    if loss_type != "sparse":
        raise RuntimeError("cuDNN Indexer adapter supports fixed-ID sparse KL only")
    ensure_cudnn_indexer_runtime_shapes(
        indexer=indexer,
        teacher=teacher,
        topk_indices=topk_indices,
        seq_ctx=seq_ctx,
    )

    # 公共协议采用 current XTuner layout；只在 adapter 边界转为 PR #2022/cuDNN layout。
    # 32->64 Indexer head padding 与 backward unpad 必须封装在 custom op 内，并通过
    # 全部 wq_b/wk/k_norm/weights_proj grad parity 测试。
    ids_bsk = topk_indices[:, 0, :].unsqueeze(0)
    valid_mask_bs = valid_query_mask.reshape(1, -1).to(ids_bsk.device, torch.bool)
    topk_length = (ids_bsk != -1).sum(dim=-1).to(torch.int32)
    kl_sum = cudnn_indexer_loss_impl(
        indexer.q,
        indexer.k,
        effective_weights,
        teacher.query.unsqueeze(0),
        teacher.key[:, 0, :].unsqueeze(0),
        teacher.softmax_lse.unsqueeze(0),
        ids_bsk,
        topk_length,
        valid_mask_bs,
        seq_ctx,
        student_sm_scale=1.0,
        teacher_sm_scale=teacher.softmax_scale,
        row_coefficient=1.0,  # raw local KL sum；step-global coeff/mean 在 context 做
        pad_index_heads_to=64,
    )
    return DSAIndexerLossStats(
        kl_sum=kl_sum,
        valid_rows=valid_mask_bs.sum(),
    )


def cudnn_indexer_loss_impl(*args, **kwargs) -> torch.Tensor: ...


def ensure_cudnn_indexer_runtime_shapes(
    *,
    indexer: DSAIndexerInputs,
    teacher: DSAIndexerTeacher,
    topk_indices: torch.Tensor,
    seq_ctx: SequenceContext,
) -> None:
    """PR #2022 首版 capability：SM90/BF16/B=1/Hkv=1/K%128/SP=1/contiguous。"""

    bf16_tensors = (indexer.q, indexer.k, teacher.query, teacher.key)
    if any(t.dtype != torch.bfloat16 for t in bf16_tensors):
        raise RuntimeError("cuDNN Indexer adapter requires BF16 q/k tensors")
    if indexer.head_weights.dtype != torch.float32:
        raise RuntimeError("canonical Indexer head_weights must remain FP32")
    assert teacher.softmax_lse is not None
    if teacher.softmax_lse.dtype != torch.float32:
        raise RuntimeError("SparseMLA softmax_lse must use natural-log FP32")
    if indexer.q.ndim != 4 or indexer.k.ndim != 3 or indexer.head_weights.ndim != 3:
        raise RuntimeError("cuDNN Indexer adapter received an invalid Indexer layout")
    if teacher.query.ndim != 3 or teacher.key.ndim != 3:
        raise RuntimeError("cuDNN Indexer adapter received an invalid teacher layout")
    if indexer.q.size(0) != 1 or indexer.k.size(0) != 1 or teacher.key.size(1) != 1:
        raise RuntimeError("cuDNN Indexer P3 adapter supports B=1 and teacher Hkv=1 only")
    q_len = indexer.q.size(1)
    if not (
        indexer.head_weights.shape[:2] == (1, q_len)
        and teacher.query.size(0) == q_len
        and teacher.softmax_lse.shape[:1] == (q_len,)
        and topk_indices.shape[:2] == (q_len, 1)
        and indexer.k.size(1) == teacher.key.size(0)
    ):
        raise RuntimeError("cuDNN Indexer Q/K/weights/LSE/IDs shapes disagree")
    if teacher.softmax_lse.size(1) != teacher.query.size(1):
        raise RuntimeError("SparseMLA LSE head count disagrees with teacher query")
    if indexer.q.size(2) != 32:
        raise RuntimeError("P3 32->64 pad adapter is specific to GLM-5.2 H_index=32")
    if topk_indices.size(-1) % 128 != 0:
        raise RuntimeError("cuDNN Indexer adapter requires K%128==0")
    if sequence_parallel_world_size(seq_ctx) != 1:
        raise RuntimeError("P3 cuDNN Indexer adapter is not yet validated for SP>1")
    runtime_tensors = (*bf16_tensors, indexer.head_weights, teacher.softmax_lse)
    if not all(t.is_cuda and t.is_contiguous() for t in runtime_tensors):
        raise RuntimeError("cuDNN Indexer adapter requires contiguous CUDA tensors")
    if not topk_indices.is_cuda:
        raise RuntimeError("cuDNN Indexer IDs must be on CUDA")
    if torch.cuda.get_device_capability(indexer.q.device) != (9, 0):
        raise RuntimeError("P3 cuDNN Indexer adapter is validated on SM90 only")


def sequence_parallel_world_size(seq_ctx: SequenceContext) -> int: ...


class IndexerBackendMemoryContract(NamedTuple):
    # 一次 source forward 结束后、直到 backward 前持续存活的 custom-autograd state。
    saved_per_source_bytes: int
    # 当前 source loss 调用自身的总峰值增量，已经包含该 source 新建的 saved state。
    loss_invocation_peak_bytes: int


def estimate_production_total_peak_bytes(
    *,
    selector_and_guard_peak_bytes: int,
    num_source_graphs_alive: int,
    loss_memory: IndexerBackendMemoryContract,
) -> int:
    """合成此前 source 持久 state 与当前 source transient，而不是分开过门禁。"""

    if num_source_graphs_alive <= 0:
        raise ValueError("num_source_graphs_alive must be positive")
    previous_saved = (
        num_source_graphs_alive - 1
    ) * loss_memory.saved_per_source_bytes
    during_last_source = previous_saved + max(
        selector_and_guard_peak_bytes,
        loss_memory.loss_invocation_peak_bytes,
    )
    after_last_source = (
        num_source_graphs_alive * loss_memory.saved_per_source_bytes
    )
    return max(during_last_source, after_last_source)


def validate_production_indexer_memory_plan(
    *,
    model_cfg,
    fsdp_cfg,
    dataloader_cfg,
) -> None:
    loss_cfg = model_cfg.attention.indexer_loss_cfg
    if loss_cfg is None:
        return
    q_len = resolve_static_pack_max_length(dataloader_cfg)
    if q_len is None:
        raise RuntimeError("joint Indexer SFT needs a provable global pack length")

    attention_cfg = model_cfg.attention
    k_eff = min(attention_cfg.index_topk, q_len)
    if attention_cfg.sparse_mla_backend in {"tilelang", "cudnn_dsa"}:
        chunk_rows = min(
            attention_cfg.indexer_topk_query_chunk_size or q_len,
            q_len,
        )
        block_q = 128 // attention_cfg.index_n_heads
        selector_required = estimate_tilelang_selector_peak_bytes(
            q_len=q_len,
            kv_len=q_len,  # SP global K；用 global pack upper bound 保守估计
            padded_chunk_rows=round_up(chunk_rows, block_q),
            k_eff=k_eff,
        )
        guard_required = estimate_fixed_topk_guard_peak_bytes(q_len, k_eff)
        selector_and_guard_required = max(selector_required, guard_required)
        selector_budget = attention_cfg.indexer_selector_workspace_budget_bytes
        if selector_budget is None or selector_and_guard_required > selector_budget:
            raise RuntimeError(
                "TileLang selector/ID guard exceeds or omits its static workspace budget: "
                f"required={selector_and_guard_required}, budget={selector_budget}"
            )
    else:
        shape_contract = resolve_static_indexer_attention_shapes(model_cfg)
        selector_required = estimate_torch_selector_peak_bytes(
            q_len=q_len,
            kv_len=q_len,
            index_heads=shape_contract["index_heads"],
        )
        guard_required = estimate_fixed_topk_guard_peak_bytes(q_len, k_eff)
        selector_and_guard_required = max(selector_required, guard_required)
    if loss_cfg.backend == "torch_reference":
        return

    live_sources = resolve_max_concurrent_source_graphs(
        model_cfg=model_cfg,
        fsdp_cfg=fsdp_cfg,
    )
    if live_sources is None:
        raise RuntimeError(
            "production Indexer SFT needs a provable pack length and activation-"
            "checkpoint coverage; refusing an unknown saved-tensor lifetime"
        )
    shape_contract = resolve_static_indexer_attention_shapes(model_cfg)
    loss_memory = backend_memory_contract(
        backend=loss_cfg.backend,
        q_len=q_len,
        index_topk=attention_cfg.index_topk,
        **shape_contract,
    )
    required = estimate_production_total_peak_bytes(
        selector_and_guard_peak_bytes=selector_and_guard_required,
        num_source_graphs_alive=live_sources,
        loss_memory=loss_memory,
    )
    budget = loss_cfg.production_total_peak_budget_bytes
    assert budget is not None
    if required > budget:
        raise RuntimeError(
            "combined Indexer persistent/transient peak exceeds the production budget: "
            f"required={required}, budget={budget}, live_sources={live_sources}"
        )


def backend_memory_contract(**kwargs) -> IndexerBackendMemoryContract: ...


def estimate_torch_selector_peak_bytes(
    *, q_len: int, kv_len: int, index_heads: int
) -> int:
    return 2 * q_len * kv_len * (index_heads + 1) * 4


def resolve_static_pack_max_length(dataloader_cfg) -> int | None: ...


def resolve_static_indexer_attention_shapes(model_cfg) -> dict[str, Any]: ...


def resolve_max_concurrent_source_graphs(*, model_cfg, fsdp_cfg) -> int | None: ...


def validate_trainer_combination(
    model_cfg,
    optim_cfg,
    fsdp_cfg,
    dataloader_cfg,
    *,
    intra_layer_micro_batch: int,
    activation_offload_enabled: bool,
) -> None:
    """Trainer build 层校验；fully_shard 后另验证真实 gradient-average group。"""

    mode = model_cfg.attention.indexer_train_mode
    if mode == "joint" and not isinstance(optim_cfg, AdamWConfig):
        raise RuntimeError("P1 joint Indexer SFT supports AdamW only")
    if mode == "joint" and (
        model_cfg.mtp_config is not None
        or bool(model_cfg.num_nextn_predict_layers)
    ):
        raise RuntimeError(
            "P1 joint Indexer SFT requires mtp_config=None and "
            "num_nextn_predict_layers in {None, 0}"
        )
    if (
        mode == "joint"
        and model_cfg.compile_cfg is not False
        and not joint_compile_capability_available(model_cfg)
    ):
        raise RuntimeError(
            "joint Indexer compile is unavailable until the P3 compile-safe "
            "context/backend capability is registered"
        )
    if mode == "joint" and getattr(fsdp_cfg, "requires_grad", True) is False:
        raise RuntimeError(
            "joint Indexer SFT cannot use fsdp_cfg.requires_grad=False; "
            "it would freeze the Indexer again during fully_shard"
        )
    if mode == "joint":
        if (
            intra_layer_micro_batch > 1
            and not joint_multi_mb_capability_available(model_cfg)
        ):
            raise RuntimeError(
                "P1 joint Indexer SFT requires intra_layer_micro_batch=1; "
                "P2 must register the list/FIFO capability before enabling it"
            )
        if activation_offload_enabled:
            validate_indexer_ids_offload_plan(model_cfg=model_cfg, fsdp_cfg=fsdp_cfg)
        validate_production_indexer_memory_plan(
            model_cfg=model_cfg,
            fsdp_cfg=fsdp_cfg,
            dataloader_cfg=dataloader_cfg,
        )
    # 当前 MoE main decoder 的 recompute 固定使用 REENTRANT；P1 无需虚构一个
    # checkpoint_impl 配置。未来开放 MTP 时再校验 fsdp_cfg.mtp_checkpoint_use_reentrant。


def joint_compile_capability_available(model_cfg) -> bool: ...


def joint_multi_mb_capability_available(model_cfg) -> bool: ...


def validate_indexer_ids_offload_plan(*, model_cfg, fsdp_cfg) -> None:
    """P2 frame-owned IDs offload 要求所有 source 都在 activation checkpoint 内。"""

    if not all_effective_source_layers_are_checkpointed(model_cfg, fsdp_cfg):
        raise RuntimeError(
            "Indexer IDs offload currently requires every source layer to be checkpointed; "
            "normal-forward sources have no frame owner"
        )


def all_effective_source_layers_are_checkpointed(model_cfg, fsdp_cfg) -> bool: ...


class ResolvedMoERuntime(NamedTuple):
    activation_offload_enabled: bool


def resolve_moe_runtime_switches_once() -> ResolvedMoERuntime:
    """Trainer conflict-resolution 阶段统一解析当前环境/配置。

    当前 main MoE forward 直接读取 ``XTUNER_ACTIVATION_OFFLOAD``。落地时要么把它
    正式提升为 TrainerConfig 字段，要么在这里一次性解析环境变量，并让 validator、
    model forward 和 activation hooks 都只读同一个 resolved runtime；禁止双数据源。
    """

    ...


def install_resolved_moe_runtime(model, runtime: ResolvedMoERuntime) -> None:
    """让 model forward/offload hooks 使用与启动校验相同的权威值。"""

    ...


def validate_glm52_joint_before_build_engine(
    *,
    model_cfg,
    optim_cfg,
    fsdp_cfg,
    dataloader_cfg,
    intra_layer_micro_batch: int,
) -> ResolvedMoERuntime:
    """接入 Trainer.__init__ 的 conflict-resolution 后、build_engine 前。

    ``intra_layer_micro_batch`` 是 Trainer.__init__ 的真实局部/config 值，不是假设
    ``trainer.activation_offload_enabled`` 或其他不存在的实例属性。返回的 runtime
    由 Trainer.build_engine 显式传给 TrainEngine。
    """

    runtime = resolve_moe_runtime_switches_once()
    validate_trainer_combination(
        model_cfg,
        optim_cfg,
        fsdp_cfg,
        dataloader_cfg,
        intra_layer_micro_batch=intra_layer_micro_batch,
        activation_offload_enabled=runtime.activation_offload_enabled,
    )
    return runtime


def trainer_construct_train_engine_after_conflict_resolution(
    *,
    model_cfg,
    optim_cfg,
    fsdp_cfg,
    dataloader_cfg,
    intra_layer_micro_batch: int,
):
    """展示对现有 Trainer.build_engine -> TrainEngine(...) 的最小签名扩展。"""

    runtime = validate_glm52_joint_before_build_engine(
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        fsdp_cfg=fsdp_cfg,
        dataloader_cfg=dataloader_cfg,
        intra_layer_micro_batch=intra_layer_micro_batch,
    )
    # 对真实 TrainEngine.__init__ 新增 resolved_moe_runtime 参数；其现有
    # build_model() -> build_optimizer() 顺序保持不变。
    return TrainEngine(
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        fsdp_cfg=fsdp_cfg,
        intra_layer_micro_batch=intra_layer_micro_batch,
        resolved_moe_runtime=runtime,
    )


def train_engine_build_model_with_indexer_hooks(engine) -> nn.Module:
    """接入现有 TrainEngine.build_model，而不是调用虚构的 Trainer builder。

    真实顺序是 ``TrainEngine.__init__: build_model() -> build_optimizer()``。因此：
    1. 在 meta device 上由 model_cfg.build() 构造 model；
    2. 安装 Trainer 传入的 resolved runtime；
    3. 调 model.fully_shard()；其现有唯一 wrapper site 对 GLM-5.2 调
       wrap_decoder_layer_at_moe_fully_shard_site()；
    4. fully_shard 返回、build_model 返回前验证/cache Indexer AVG group；
    5. 保持既有 TrainEngine.__init__ 随后调用 build_optimizer()。
    """

    with torch.device("meta"):
        model = engine.model_cfg.build()
    install_resolved_moe_runtime(model, engine.resolved_moe_runtime)
    model = model.fully_shard(engine.fsdp_cfg)
    finalize_glm52_joint_runtime_after_fully_shard(model)
    return model


class IndexerCheckpointMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    mode: IndexerTrainMode
    loss_type: IndexerLossType | None
    loss_coeff: float | None
    loss_backend: IndexerLossBackend | None
    global_average: bool | None
    # 同时记录主 SparseMLA backend 与解析后的 selector；cudnn_dsa 当前 selector=tilelang。
    sparse_mla_backend: Literal["torch", "tilelang", "cudnn_dsa"]
    resolved_selector_backend: Literal["torch", "tilelang"]
    index_topk: int
    indexer_topk_query_chunk_size: int | None
    # frozen 仍兼容主模型 Muon；只有 joint P1 validator 强制 all_adamw。
    optimizer_family: Literal["adamw", "muon"]
    indexer_optimizer_policy: Literal["not_applicable", "all_adamw"]
    hf_base: str
    # 路径字符串不足以证明 frozen 参数来源相同；exact resume 必须记录并校验
    # immutable revision + config/weight manifest digest。
    hf_base_revision: str
    hf_base_fingerprint: str
    # kernel/frontend/torch/XTuner commit 会影响 near-tie IDs 与 custom backward。
    runtime_fingerprint: str
    # world/mesh/grad-average group、SP/EP/FSDP、grad accumulation/global batch
    # 以及 packing policy 会改变 step-global denominator 和精确训练轨迹。
    distributed_training_fingerprint: str
    # scheduler state_dict 不能恢复已更换的 scheduler class/Lambda closure；必须同时
    # 固化 lr_cfg、resolved total_step、warmup 等构造策略。
    scheduler_policy_fingerprint: str
    activation_checkpoint_policy: str
    dcp_indexer_state: Literal["included", "omitted_frozen"]
    indexer_schema_fingerprint: str
    converted_from_mtp_hf_base: str | None = None
    source_num_nextn_predict_layers: int | None = None
    # 保存解析后的物理 source/full 层，兼容 indexer_types=None 的 freq/offset 配置。
    effective_source_layers: tuple[int, ...]

    @model_validator(mode="after")
    def validate_policy(self):
        loss_fields = (
            self.loss_type,
            self.loss_coeff,
            self.loss_backend,
            self.global_average,
        )
        if self.mode == "frozen":
            if any(value is not None for value in loss_fields):
                raise ValueError("frozen metadata must not contain Indexer loss policy")
            if self.indexer_optimizer_policy != "not_applicable":
                raise ValueError("frozen Indexer optimizer policy must be not_applicable")
        else:
            if any(value is None for value in loss_fields):
                raise ValueError("joint metadata requires complete Indexer loss policy")
            assert self.loss_coeff is not None
            if not math.isfinite(self.loss_coeff) or self.loss_coeff <= 0:
                raise ValueError("joint loss_coeff must be finite and positive")
            if (
                self.optimizer_family != "adamw"
                or self.indexer_optimizer_policy != "all_adamw"
            ):
                raise ValueError("P1 joint metadata requires all-AdamW policy")
        if self.index_topk <= 0:
            raise ValueError("index_topk must be positive")
        if (
            self.indexer_topk_query_chunk_size is not None
            and self.indexer_topk_query_chunk_size <= 0
        ):
            raise ValueError("query chunk size must be positive")
        expected_selector = (
            "torch" if self.sparse_mla_backend == "torch" else "tilelang"
        )
        if self.resolved_selector_backend != expected_selector:
            raise ValueError(
                "resolved selector disagrees with current SparseMLA backend mapping"
            )
        if (
            self.indexer_topk_query_chunk_size is not None
            and self.resolved_selector_backend != "tilelang"
        ):
            raise ValueError("query chunk requires a TileLang selector")
        if not self.hf_base or not self.hf_base_revision or not self.hf_base_fingerprint:
            raise ValueError("HF base identity must include path/id, revision and fingerprint")
        if (
            not self.runtime_fingerprint
            or not self.distributed_training_fingerprint
            or not self.scheduler_policy_fingerprint
            or not self.activation_checkpoint_policy
            or not self.indexer_schema_fingerprint
        ):
            raise ValueError(
                "exact-resume runtime/distributed/scheduler/checkpoint policy must not be empty"
            )
        if self.mode == "joint" and self.dcp_indexer_state != "included":
            raise ValueError("trainable joint Indexer state must be included in DCP")
        if not self.effective_source_layers:
            raise ValueError("effective_source_layers must not be empty")
        if (
            any(layer < 0 for layer in self.effective_source_layers)
            or tuple(sorted(set(self.effective_source_layers)))
            != self.effective_source_layers
        ):
            raise ValueError(
                "effective_source_layers must be sorted, unique and non-negative"
            )
        converted_fields = (
            self.converted_from_mtp_hf_base,
            self.source_num_nextn_predict_layers,
        )
        if (converted_fields[0] is None) != (converted_fields[1] is None):
            raise ValueError("NoMTP conversion provenance fields must appear together")
        if (
            self.source_num_nextn_predict_layers is not None
            and self.source_num_nextn_predict_layers <= 0
        ):
            raise ValueError("converted MTP source must have a positive next-token depth")
        return self


def build_indexer_checkpoint_metadata(
    *,
    model,
    model_cfg,
    optim_cfg,
    fsdp_cfg,
    dataloader_cfg,
    lr_cfg,
    resolved_total_step: int,
    training_mesh,
    global_batch_size: int,
    intra_layer_micro_batch: int,
    hf_load_from: str,
) -> IndexerCheckpointMetadata:
    """唯一 metadata builder；字段不能由 recipe 任意手填。

    - Hub base 使用 resolved commit SHA；本地 base 对 config、index JSON 与每个实际
      weight shard 内容（或可信 LFS OID）做 canonical digest，不能只 hash manifest；
    - Indexer schema digest 覆盖 effective source mapping 及每个 Indexer parameter 的
      fully-qualified name/shape/dtype/requires_grad；
    - runtime digest 覆盖 XTuner commit、torch/CUDA/cuDNN-frontend/TileLang 版本与
      resolved kernel backend；
    - distributed digest 覆盖 world/named mesh、Indexer gradient-average group、
      SP/EP/FSDP placements、grad-acc/global batch 与 deterministic packing policy；
    - scheduler digest 覆盖 scheduler class、完整 lr_cfg、resolved total_step/warmup 与
      Lambda closure 的 canonical policy，而不只相信 scheduler state_dict；
    - DCP inclusion 从真实 state manifest/dcp_ignore_frozen_params 推导。
    """

    return IndexerCheckpointMetadata(
        **resolve_indexer_policy_fields(model, model_cfg, optim_cfg, fsdp_cfg),
        **canonical_hf_base_identity(hf_load_from),
        runtime_fingerprint=canonical_indexer_runtime_fingerprint(model_cfg),
        distributed_training_fingerprint=(
            canonical_distributed_training_fingerprint(
                model=model,
                model_cfg=model_cfg,
                fsdp_cfg=fsdp_cfg,
                dataloader_cfg=dataloader_cfg,
                training_mesh=training_mesh,
                global_batch_size=global_batch_size,
                intra_layer_micro_batch=intra_layer_micro_batch,
            )
        ),
        scheduler_policy_fingerprint=canonical_scheduler_policy_fingerprint(
            lr_cfg=lr_cfg,
            resolved_total_step=resolved_total_step,
        ),
        indexer_schema_fingerprint=canonical_indexer_schema_fingerprint(model),
        dcp_indexer_state=resolve_dcp_indexer_state(model, model_cfg),
    )


def resolve_indexer_policy_fields(*args, **kwargs) -> dict[str, Any]: ...


def canonical_hf_base_identity(load_from: str) -> dict[str, str]: ...


def canonical_indexer_runtime_fingerprint(model_cfg) -> str: ...


def canonical_distributed_training_fingerprint(**kwargs) -> str: ...


def canonical_scheduler_policy_fingerprint(**kwargs) -> str: ...


def canonical_indexer_schema_fingerprint(model) -> str: ...


def resolve_dcp_indexer_state(model, model_cfg) -> str: ...


def save_indexer_metadata(checkpoint_root: Path, metadata: IndexerCheckpointMetadata):
    # 嵌入 Trainer._maybe_save：仅 rank0 原子写，并复用现有 barrier、async-DCP
    # completion 和 checkpoint retention 生命周期，不能成为孤立 sidecar。
    if not dist.is_initialized() or dist.get_rank() == 0:
        path = checkpoint_root / "indexer_sft.json"
        tmp = checkpoint_root / "indexer_sft.json.tmp"
        tmp.write_text(metadata.model_dump_json(indent=2))
        tmp.replace(path)


def validate_indexer_resume_before_load_dcp(
    checkpoint_root: Path,
    current: IndexerCheckpointMetadata,
    load_checkpoint_cfg,
    model,
    *,
    allow_legacy_frozen_best_effort: bool = False,
) -> None:
    """插在 Trainer._load_checkpoint() 调 engine.load_dcp() 之前。

    本入口只允许 exact Trainer resume：model/optimizer/optimizer args/scheduler、dataset
    position 与 train_state 一起恢复。策略/模式切换不走 DCP model-only；先导出完整 HF，
    再启动一个重建 optimizer/scheduler/step/data-position 的新 run。
    """

    full_resume = (
        load_checkpoint_cfg.load_optimizer_states
        and load_checkpoint_cfg.load_optimizer_args
        and load_checkpoint_cfg.load_scheduler
        and load_checkpoint_cfg.load_dataset
    )
    if not full_resume:
        raise RuntimeError(
            "Indexer DCP resume must restore optimizer, optimizer args, scheduler, "
            "dataset position and train_state together. For model-only changes, export "
            "a full HF checkpoint and start a new run."
        )

    metadata_path = checkpoint_root / "indexer_sft.json"
    if not metadata_path.exists():
        if current.mode == "frozen" and allow_legacy_frozen_best_effort:
            # 无 sidecar 无法证明 exact trajectory；只提供显式 best-effort 兼容。
            verify_hf_base_identity(current)
            assert_model_indexer_matches_verified_hf_base(model, current)
            return
        raise RuntimeError(
            "legacy checkpoint has no Indexer sidecar; exact resume is unprovable"
        )
    saved = IndexerCheckpointMetadata.model_validate_json(
        metadata_path.read_text()
    )
    # saved.hf_base 可能是已搬迁的旧路径；以 sidecar 中的 immutable digest 为记录，
    # 只对 current 可访问位置重新计算 fingerprint，随后比较两份 policy payload。
    verify_hf_base_identity(current)
    if saved.dcp_indexer_state == "omitted_frozen":
        # engine/model 已在此入口前构造；必须证明它确实从 current（且与 saved digest
        # 相同）的 HF base 预载了被 DCP 省略的 Indexer，而不能只相信配置字符串。
        assert_model_indexer_matches_verified_hf_base(model, current)

    if saved.effective_source_layers != current.effective_source_layers:
        raise RuntimeError(
            "Indexer parameter topology changed; exact DCP resume is unsafe"
        )
    # HF 路径可搬迁；exact identity 由 revision + manifest fingerprint 决定。
    saved_policy = saved.model_dump(exclude={"hf_base"})
    current_policy = current.model_dump(exclude={"hf_base"})
    if saved_policy != current_policy:
        raise RuntimeError(
            "Indexer mode/loss/backend/runtime/HF identity differs from the checkpoint; "
            "exact full resume is forbidden. Export a complete HF checkpoint and start "
            "a new run instead."
        )


def verify_hf_base_identity(metadata: IndexerCheckpointMetadata) -> None:
    """校验 revision 与 config/weight manifest digest，而不只检查路径存在。"""

    ...


def assert_model_indexer_matches_verified_hf_base(
    model,
    metadata: IndexerCheckpointMetadata,
) -> None: ...


def configure_glm52_joint_sft(
    model_cfg,
    *,
    loss_backend: IndexerLossBackend = "torch_reference",
    indexer_topk_query_chunk_size: int | None = None,
    indexer_selector_workspace_budget_bytes: int | None = None,
    production_total_peak_budget_bytes: int | None = None,
) -> None:
    if model_cfg.mtp_config is not None or bool(model_cfg.num_nextn_predict_layers):
        raise RuntimeError(
            "P1 requires a NoMTP GLM-5.2 checkpoint. Converting a standard HF "
            "checkpoint requires explicit mtp_config=None, "
            "num_nextn_predict_layers=None, and a strict_load=False model-only "
            "initialization flow."
        )
    if (
        model_cfg.attention.sparse_mla_backend in {"tilelang", "cudnn_dsa"}
        and indexer_selector_workspace_budget_bytes is None
    ):
        raise RuntimeError(
            "TileLang Top-K selection requires an explicit selector workspace budget"
        )
    model_cfg.compile_cfg = False
    model_cfg.attention.indexer_train_mode = "joint"
    model_cfg.attention.indexer_topk_query_chunk_size = (
        indexer_topk_query_chunk_size
    )
    model_cfg.attention.indexer_selector_workspace_budget_bytes = (
        indexer_selector_workspace_budget_bytes
    )
    model_cfg.attention.indexer_loss_cfg = DSAIndexerLossConfig(
        loss_coeff=1e-3,
        loss_type="sparse",
        backend=loss_backend,
        production_total_peak_budget_bytes=production_total_peak_budget_bytes,
    )


# TrainEngine.train_step / _get_total_loss / step_optimizer 保持现有实现：
# - projected_output 上的 AuxLossScaler 是唯一可导 Indexer KL 入口；
# - MoEModelOutputs.indexer_loss 是 detached scalar，只补充数值与日志，不会 double backward；
# - 不在设计里伪造 optimizer.step()，避免绕过 invalid-grad 和 zero_grad 逻辑。
