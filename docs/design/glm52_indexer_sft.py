"""GLM-5.2 DSA Indexer 联合 SFT 设计伪代码。

设计要点：
- frozen 是兼容路径：沿用当前 DSAIndexer.forward() -> Top-K Tensor。
- joint 在固定 Top-K support 上计算 attention-teacher KL，只更新 Indexer。
- 当前 CrossLayerTopKSharingRuntime 仍是 Top-K cache 的唯一 owner。
- reentrant checkpoint replay 复用 original IDs，并重新建立可导 KL 图。
- loss context 使用整个 train step 的有效 query 分母，不做 micro-batch mean 的平均。
- PyTorch 只作有显存门禁的 oracle；长序列 TileLang/cuDNN adapter 未接通前 fail fast。

说明：
- 这是接口和控制流伪代码，不是可直接合入的实现。
- 名称尽量贴近 XTuner 当前模块；省略 HF mapping、FSDP 和 kernel 样板代码。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Protocol, TypedDict

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict
from torch import nn

from xtuner.v1.config import AdamWConfig


IndexerTrainMode = Literal["frozen", "joint"]
IndexerLossType = Literal["sparse", "dense"]
IndexerLossBackend = Literal["torch_reference", "tilelang", "cudnn"]


class DSAIndexerLossConfig(BaseModel):
    """采用 ZLossConfig 风格，而不是继承 LM-head 的 BaseLossConfig。"""

    model_config = ConfigDict(extra="forbid")

    loss_coeff: float = 1e-3
    loss_type: IndexerLossType = "sparse"
    backend: IndexerLossBackend = "torch_reference"
    global_average: bool = True
    reference_workspace_limit_bytes: int = 2 * 1024**3

    def validate_joint(self) -> None:
        if self.loss_coeff <= 0:
            raise ValueError("joint Indexer SFT requires loss_coeff > 0")
        if self.reference_workspace_limit_bytes <= 0:
            raise ValueError("reference_workspace_limit_bytes must be positive")

    def build(self, *, local_valid_rows: torch.Tensor) -> "DSAIndexerLossContext":
        return DSAIndexerLossContext(self, local_valid_rows=local_valid_rows)


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
    """只展示拟新增字段和真实 build 边界。"""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    indexer_train_mode: IndexerTrainMode = "frozen"
    indexer_loss_cfg: DSAIndexerLossConfig | None = None
    sparse_mla_backend: Literal["torch", "tilelang", "cudnn_dsa"] = "torch"

    def build(self, **runtime_args) -> "DSAMultiLatentAttention":
        validate_indexer_training_config(self.indexer_train_mode, self.indexer_loss_cfg)

        # 当前真实 build 使用 **self.model_dump()。nested BaseModel 会变成 dict，
        # 所以新增配置必须 exclude 后显式传对象，不能假定 module 收到 config 实例。
        kwargs = self.model_dump(
            exclude={"indexer_train_mode", "indexer_loss_cfg"}
        )
        loss_op = (
            build_indexer_loss_backend(self.indexer_loss_cfg)
            if self.indexer_loss_cfg is not None
            else None
        )
        return DSAMultiLatentAttention(
            **kwargs,
            **runtime_args,
            indexer_train_mode=self.indexer_train_mode,
            indexer_loss_cfg=self.indexer_loss_cfg,
            indexer_loss_op=loss_op,
        )


class SequenceContext(Protocol):
    mask: torch.Tensor
    sequence_parallel_mesh: Any

    def packed_causal_query_ranges(
        self, query_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class DSAIndexerInputs(NamedTuple):
    """可导 Indexer 投影；输入 hidden/q_resid 已 detach。"""

    q: torch.Tensor  # [B, S_q_local, H_index, D_index]
    k: torch.Tensor  # [B, S_k_global, D_index]
    weights: torch.Tensor  # [B, S_q_local, H_index]
    index_head_dim: int

    def detached(self) -> "DSAIndexerInputs":
        return DSAIndexerInputs(
            self.q.detach(),
            self.k.detach(),
            self.weights.detach(),
            self.index_head_dim,
        )


class DSAIndexerTeacher(NamedTuple):
    # 与当前 absorbed DSA-MLA 的真实 layout 一致：teacher 不带 batch 维。
    query: torch.Tensor  # [S_q_local, H_attention, D_attention]
    key: torch.Tensor  # [S_k_global, H_kv, D_attention]
    softmax_scale: float

    def detached(self) -> "DSAIndexerTeacher":
        return DSAIndexerTeacher(
            self.query.detach(), self.key.detach(), self.softmax_scale
        )


class DSAIndexerLossStats(NamedTuple):
    # backend 不乘 coeff、不做跨 MB/rank 平均。
    kl_sum: torch.Tensor
    valid_rows: torch.Tensor


class DSATopKIndicesProtocol(Protocol):
    """现有 IDs-only 协议保持不变。"""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        seq_ctx: SequenceContext,
        *,
        index_head_dim: int,
        index_topk: int,
    ) -> torch.Tensor: ...


class DSAIndexerLossProtocol(Protocol):
    """针对 original forward 已确定的 IDs 计算 KL。"""

    def __call__(
        self,
        indexer: DSAIndexerInputs,
        teacher: DSAIndexerTeacher,
        topk_indices: torch.Tensor,
        seq_ctx: SequenceContext,
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


def torch_fixed_topk_indexer_loss(
    indexer: DSAIndexerInputs,
    teacher: DSAIndexerTeacher,
    topk_indices: torch.Tensor,
    seq_ctx: SequenceContext,
    *,
    loss_type: IndexerLossType,
) -> DSAIndexerLossStats:
    """短序列 sparse oracle；生产实现还需要按 query/head chunk。

    这里假定 GLM-5.2 的 batch=1、teacher KV head=1，只为表达数学和安全 mask。
    """

    if loss_type == "dense":
        return torch_dense_indexer_loss(indexer, teacher, seq_ctx)
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
    query_mask = seq_ctx.mask.reshape(-1)[:q_len].to(torch.bool)
    query_valid = query_mask & (ends > starts)
    padding = ids == -1
    legal = (
        (ids >= 0)
        & (ids < kv_len)
        & (ids >= starts[:, None])
        & (ids < ends[:, None])
        & query_valid[:, None]
    )
    # -1 是唯一允许的 padding。只要出现非 -1 的越界、跨 packed sample，
    # 或 padding query 上的 ID，即使同一行还有其他合法 ID，也必须暴露 backend bug。
    illegal = ~padding & ~legal
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

    # 全 invalid 行放一个 benign sentinel，使 softmax 输入至少有一个有限值；
    # 最终仍用原 valid mask 把该行的值和梯度严格归零。
    safe_support = valid.clone()
    safe_support[~valid_rows, 0] = True

    q_i = indexer.q[0].float()  # [S,H_i,D_i]
    k_i = indexer.k[0][safe_ids].float()  # [S,K,D_i]，不 expand 到 head 维
    index_logits_h = torch.einsum("shd,skd->shk", q_i, k_i)
    index_logits_h = torch.relu(
        index_logits_h * (indexer.index_head_dim**-0.5)
    )
    index_logits = torch.einsum(
        "shk,sh->sk", index_logits_h, indexer.weights[0].float()
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

    def __call__(self, indexer, teacher, topk_indices, seq_ctx, *, loss_type):
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
            indexer, teacher, topk_indices, seq_ctx, loss_type=loss_type
        )


class DSAIndexerLossContext(nn.Module):
    """按整个 train step 校准的 layer auxiliary loss context。"""

    def __init__(
        self, config: DSAIndexerLossConfig, *, local_valid_rows: torch.Tensor
    ) -> None:
        super().__init__()
        self.config = config
        self.local_valid_rows = local_valid_rows
        self.global_valid_rows_step: torch.Tensor | None = None
        self.grad_average_group_size = 1
        self.num_contexts = 1
        self._running_total: torch.Tensor | None = None

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

    def accumulate(
        self,
        *,
        source_layer_idx: int,
        stats: DSAIndexerLossStats,
        record: bool,
    ) -> torch.Tensor:
        rows_local = stats.valid_rows.to(stats.kl_sum.device, torch.float32)
        expected_rows = self.local_valid_rows.to(rows_local.device, torch.float32)
        if int(rows_local.item()) != int(expected_rows.item()):
            raise RuntimeError("backend valid-row count disagrees with loss calibration")

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

        if record:
            detached = scaled.detach()
            self._running_total = (
                detached.clone()
                if self._running_total is None
                else self._running_total + detached
            )
        return scaled

    def finalize(self, device: torch.device) -> torch.Tensor:
        total = self._running_total
        self._running_total = None
        if total is None:
            total = torch.tensor(0.0, device=device, dtype=torch.float32)
        return total


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
    """保留现有 forward 兼容层；joint 使用新增的细粒度方法。"""

    def __init__(
        self,
        *,
        train_mode: IndexerTrainMode,
        loss_cfg: DSAIndexerLossConfig | None,
        topk_op: DSATopKIndicesProtocol,
        loss_op: DSAIndexerLossProtocol | None,
        index_topk: int,
        index_head_dim: int,
    ) -> None:
        super().__init__()
        validate_indexer_training_config(train_mode, loss_cfg)
        self.train_mode = train_mode
        self.loss_cfg = loss_cfg
        self.topk_op = topk_op
        self.loss_op = loss_op
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.wq_b: nn.Module = ...
        self.wk: nn.Module = ...
        self.k_norm: nn.Module = ...
        self.weights_proj: nn.Module = ...
        self.requires_grad_(train_mode == "joint")

    def forward(self, hidden_states, q_resid, position_embeddings, seq_ctx):
        """frozen/eval 兼容入口仍返回 Tensor，不引入复合类型。"""

        if self.train_mode != "frozen" and self.training:
            raise RuntimeError("joint training must use the attention-orchestrated path")
        with torch.no_grad():
            inputs = self.project(
                hidden_states, q_resid, position_embeddings, seq_ctx
            )
            return self.select_topk(inputs, seq_ctx)

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
        # H_index^-1/2 只在 weights projection 路径乘一次。
        weights = self.weights_proj(hidden_states).float() * (q.size(2) ** -0.5)
        return DSAIndexerInputs(q, k, weights, self.index_head_dim)

    @torch.no_grad()
    def select_topk(
        self, inputs: DSAIndexerInputs, seq_ctx: SequenceContext
    ) -> torch.Tensor:
        inputs = inputs.detached()
        # D_index^-1/2 由现有 Top-K op 内部乘一次；loss backend 必须同口径，不能重复缩放。
        return self.topk_op(
            inputs.q,
            inputs.k,
            inputs.weights,
            seq_ctx,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
        )

    def loss_for_indices(self, inputs, teacher, topk_indices, seq_ctx):
        if self.loss_op is None or self.loss_cfg is None:
            raise RuntimeError("joint Indexer loss backend is missing")
        return self.loss_op(
            inputs,
            teacher.detached(),
            topk_indices,
            seq_ctx,
            loss_type=self.loss_cfg.loss_type,
        )


def project_indexer_q(module, q_resid, position_embeddings): ...


def project_and_gather_indexer_k(
    module, norm, hidden_states, position_embeddings, seq_ctx
): ...


class IndexerExecutionPhase(Enum):
    NORMAL = auto()
    CHECKPOINT_ORIGINAL = auto()
    CHECKPOINT_REPLAY = auto()


@dataclass(frozen=True)
class DSAIndexerSelection:
    topk_indices: torch.Tensor
    loss_stats: DSAIndexerLossStats | None = None
    attach_loss: bool = False
    record_loss: bool = False


class CrossLayerTopKSharingRuntime:
    """伪扩展：真实 seq_ctx.dsa_topk_cache 仍是唯一 cache。

    DSATopKCacheState 新增 checkpoint_original_sources: set[int]。source 的
    original 当场写 marker，不能等最后 consumer 再设置一个全局 checkpoint_active；
    默认 GLM-5.2 的最后 consumer 可能是不 checkpoint 的最后一层。
    """

    def get_or_compute(self, *, layer, seq_ctx, compute_source_topk):
        # frozen/eval 完整沿用当前实现，不写新的 marker。
        return self._existing_get_or_compute(layer, seq_ctx, compute_source_topk)

    def get_or_compute_for_joint(
        self, *, layer, seq_ctx, compute_source_topk, phase: IndexerExecutionPhase
    ):
        # joint-only wrapper：source original 当场标记；source replay 读取原 IDs。
        if phase is IndexerExecutionPhase.CHECKPOINT_REPLAY:
            return self._read_existing_cache(seq_ctx, layer.source_layer_idx)
        topk = self.get_or_compute(
            layer=layer,
            seq_ctx=seq_ctx,
            compute_source_topk=compute_source_topk,
        )
        if (
            phase is IndexerExecutionPhase.CHECKPOINT_ORIGINAL
            and layer.layer_idx == layer.source_layer_idx
        ):
            seq_ctx.dsa_topk_cache.checkpoint_original_sources.add(layer.source_layer_idx)
        return topk

    def after_sparse_mla_use(self, *, layer, seq_ctx) -> None:
        # 仍只由 decoder post-hook 调用。grad-enabled 的普通 shared last-consumer
        # 不能清掉 marker/cache；更高层 backward 完成后，source replay 按现有
        # recompute_release plan 释放 residency，并删除该 source marker。
        if self._is_checkpoint_recompute(layer, seq_ctx):
            if (
                layer.dsa_topk_recompute_release.get(layer.source_layer_idx)
                == layer.layer_idx
            ):
                self._release_existing_cache(seq_ctx, layer.source_layer_idx)
                seq_ctx.dsa_topk_cache.checkpoint_original_sources.discard(
                    layer.source_layer_idx
                )
            return
        self._existing_after_sparse_mla_use(layer, seq_ctx)

    def execution_phase(self, *, layer, seq_ctx) -> IndexerExecutionPhase:
        if self._is_checkpoint_original_forward(layer):
            return IndexerExecutionPhase.CHECKPOINT_ORIGINAL
        if self._is_checkpoint_recompute(layer, seq_ctx):
            return IndexerExecutionPhase.CHECKPOINT_REPLAY
        return IndexerExecutionPhase.NORMAL

    def resolve_with_loss(
        self,
        *,
        layer,
        seq_ctx,
        need_loss: bool,
        select_topk: Callable[[], torch.Tensor],
        loss_for_indices: Callable[[torch.Tensor], DSAIndexerLossStats],
    ) -> DSAIndexerSelection:
        phase = self.execution_phase(layer=layer, seq_ctx=seq_ctx)
        topk = self.get_or_compute_for_joint(
            layer=layer,
            seq_ctx=seq_ctx,
            compute_source_topk=select_topk,
            phase=phase,
        )
        is_source = layer.source_layer_idx == layer.layer_idx
        if not need_loss or not is_source:
            return DSAIndexerSelection(topk)

        if phase is IndexerExecutionPhase.CHECKPOINT_ORIGINAL:
            with torch.no_grad():
                stats = loss_for_indices(topk)
            return DSAIndexerSelection(
                topk, stats, attach_loss=False, record_loss=True
            )

        stats = loss_for_indices(topk)
        return DSAIndexerSelection(
            topk,
            stats,
            attach_loss=True,
            record_loss=phase is IndexerExecutionPhase.NORMAL,
        )

    def _is_checkpoint_original_forward(self, layer) -> bool: ...

    def _is_checkpoint_recompute(self, layer, seq_ctx) -> bool:
        return (
            layer.layer_idx == layer.source_layer_idx
            and layer.source_layer_idx
            in seq_ctx.dsa_topk_cache.checkpoint_original_sources
            and torch.is_grad_enabled()
        )

    def _existing_get_or_compute(self, layer, seq_ctx, compute_source_topk): ...

    def _existing_after_sparse_mla_use(self, layer, seq_ctx) -> None: ...

    def _read_existing_cache(self, seq_ctx, source_layer_idx) -> torch.Tensor: ...

    def _release_existing_cache(self, seq_ctx, source_layer_idx) -> None:
        """复用现有 residency/recompute-release 清理，并同步 released_sources。"""
        ...


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
        indexer_loss_op: DSAIndexerLossProtocol | None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.indexer_train_mode = indexer_train_mode
        self.indexer: DSAIndexer | None = ...
        self.topk_runtime: CrossLayerTopKSharingRuntime = ...

    def forward(
        self,
        hidden_states,
        position_embeddings,
        seq_ctx,
        *,
        indexer_loss_ctx: DSAIndexerLossContext | None = None,
    ):
        q_resid, query, key, value_dim = self._project_main_attention(
            hidden_states, position_embeddings, seq_ctx
        )
        is_source = self.source_layer_idx == self.layer_idx
        if self.training and self.indexer_train_mode == "joint" and indexer_loss_ctx is None:
            raise RuntimeError("joint Indexer training lost its loss context")

        if indexer_loss_ctx is None:
            # frozen 和 joint-eval 完整保留当前入口/compile/checkpoint 图形。
            topk_indices = self.topk_runtime.get_or_compute(
                layer=self,
                seq_ctx=seq_ctx,
                compute_source_topk=lambda: self.indexer(
                    hidden_states, q_resid, position_embeddings, seq_ctx
                ),
            )
            selection = DSAIndexerSelection(topk_indices)
        else:
            inputs: DSAIndexerInputs | None = None

            def get_inputs() -> DSAIndexerInputs:
                nonlocal inputs
                assert is_source and self.indexer is not None
                if inputs is None:
                    inputs = self.indexer.project(
                        hidden_states.detach(),
                        q_resid.detach(),
                        position_embeddings,
                        seq_ctx,
                    )
                return inputs

            teacher = DSAIndexerTeacher(
                query.detach(), key.detach(), self.softmax_scale
            )
            selection = self.topk_runtime.resolve_with_loss(
                layer=self,
                seq_ctx=seq_ctx,
                need_loss=True,
                select_topk=lambda: self.indexer.select_topk(get_inputs(), seq_ctx),
                loss_for_indices=lambda ids: self.indexer.loss_for_indices(
                    get_inputs(), teacher, ids, seq_ctx
                ),
            )

        sparse_output = self._sparse_mla(query, key, selection.topk_indices, value_dim)
        # 不在 attention 内调用 after_sparse_mla_use。当前真实实现由 decoder
        # post-hook 在正确边界调用；重复调用会双减 MTP counter 或提前释放 cache。
        attn_outputs = self._finish_attention(sparse_output)
        projected_output = attn_outputs["projected_output"]

        if selection.loss_stats is not None:
            assert indexer_loss_ctx is not None
            scaled = indexer_loss_ctx.accumulate(
                source_layer_idx=self.source_layer_idx,
                stats=selection.loss_stats,
                record=selection.record_loss,
            )
            if selection.attach_loss:
                projected_output = AuxLossScaler.apply(projected_output, scaled)
        attn_outputs["projected_output"] = projected_output
        return attn_outputs

    def _project_main_attention(self, hidden_states, position_embeddings, seq_ctx): ...

    def _sparse_mla(self, query, key, ids, value_dim): ...

    def _finish_attention(self, output) -> dict[str, torch.Tensor]:
        # 保留 raw_output / softmax_lse，并完成 absorbed w_vc 与 o_proj。
        ...


class MoELossContextDict(TypedDict):
    lm: Any
    balancing: Any
    z_loss: Any
    mtp: Any
    indexer: DSAIndexerLossContext | None


class MoEModelOutputs(BaseModel):
    """真实类新增 Optional 字段；frozen 返回时保持 None/不参与 loss。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    loss: torch.Tensor
    indexer_loss: torch.Tensor | None = None
    extra_info: Any = None


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

    # P1 只开放已验证为 WORLD-average 的拓扑。该 resolver 必须核对 Indexer
    # DTensor/FSDP placements；不能从 sp_mesh 大小猜测 gradient group。
    assert_indexer_uses_world_gradient_average(model)
    grad_average_group = dist.group.WORLD if dist.is_initialized() else None
    collective_device = indexer_collective_device(model, grad_average_group)

    contexts = []
    for data in data_batch:
        full_mask = data["seq_ctx"].mask
        local_mask = split_like_sequence_context(full_mask, sp_mesh)
        contexts.append(cfg.build(local_valid_rows=local_mask.sum()))
    return DSAIndexerLossContext.build_batches(
        contexts,
        grad_average_group=grad_average_group,
        collective_device=collective_device,
    )


def split_like_sequence_context(mask: torch.Tensor, sp_mesh) -> torch.Tensor: ...


def assert_indexer_uses_world_gradient_average(model) -> None: ...


def indexer_collective_device(
    model, grad_average_group: dist.ProcessGroup | None
) -> torch.device:
    """返回与 Indexer 参数及 WORLD collective backend 兼容的 accelerator。"""
    ...


def finalize_indexer_output(
    *,
    output: dict[str, Any],
    loss_ctx: MoELossContextDict | list[MoELossContextDict],
    device: torch.device,
) -> None:
    """single-MB 与 intra-layer multi-MB 都归约成一个 detached 字段。"""

    loss_ctx_list = loss_ctx if isinstance(loss_ctx, list) else [loss_ctx]
    indexer_contexts = [
        ctx["indexer"]
        for ctx in loss_ctx_list
        if ctx.get("indexer") is not None
    ]
    if not indexer_contexts:
        return
    values = [ctx.finalize(device) for ctx in indexer_contexts]
    output["indexer_loss"] = torch.stack(values).sum().detach()


def build_indexer_loss_backend(
    config: DSAIndexerLossConfig,
) -> DSAIndexerLossProtocol:
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

    if config.backend == "cudnn":
        # current sparse_mla_backend='cudnn_dsa' 是 hybrid，不满足这里的 full adapter。
        ensure_full_cudnn_indexer_loss_available()
        return cudnn_fixed_topk_indexer_loss

    raise AssertionError(config.backend)


def ensure_tilelang_fixed_id_loss_available() -> None: ...


def ensure_full_cudnn_indexer_loss_available() -> None: ...


def tilelang_fixed_topk_indexer_loss(*args, **kwargs) -> DSAIndexerLossStats: ...


def cudnn_fixed_topk_indexer_loss(*args, **kwargs) -> DSAIndexerLossStats: ...


def validate_trainer_combination(model_cfg, optim_cfg, fsdp_cfg) -> None:
    """必须在 Trainer build 层校验，因为这里才能同时看到三类配置。"""

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
    if mode == "joint" and model_cfg.compile_cfg is not False:
        raise RuntimeError("P1 joint Indexer SFT requires compile_cfg=False")
    # 当前 MoE main decoder 的 recompute 固定使用 REENTRANT；P1 无需虚构一个
    # checkpoint_impl 配置。未来开放 MTP 时再校验 fsdp_cfg.mtp_checkpoint_use_reentrant。


class IndexerCheckpointMetadata(BaseModel):
    mode: IndexerTrainMode
    loss_type: IndexerLossType | None
    loss_coeff: float | None
    loss_backend: IndexerLossBackend | None
    hf_base: str
    converted_from_mtp_hf_base: str | None = None
    source_num_nextn_predict_layers: int | None = None
    # 保存解析后的物理 source/full 层，兼容 indexer_types=None 的 freq/offset 配置。
    effective_source_layers: tuple[int, ...]


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
) -> None:
    # 插在 Trainer._load_checkpoint() 调 engine.load_dcp() 之前。
    metadata_path = checkpoint_root / "indexer_sft.json"
    if not metadata_path.exists():
        if current.mode == "frozen":
            # legacy frozen checkpoint：调用方仍必须保证 current.hf_base/load_from 可用，
            # 以便补齐 dcp_ignore_frozen_params 省略的 Indexer。
            require_hf_base_available(current.hf_base)
            return
        raise RuntimeError("legacy checkpoint has no joint Indexer metadata")
    saved = IndexerCheckpointMetadata.model_validate_json(
        metadata_path.read_text()
    )
    if (
        saved.mode != current.mode
        or saved.effective_source_layers != current.effective_source_layers
    ):
        raise RuntimeError(
            "Indexer mode/parameter topology changed; full optimizer resume is forbidden. "
            "Use an explicit HF/model-only initialization flow instead."
        )
    if saved != current:
        raise RuntimeError(
            "Indexer loss policy differs from the checkpoint; exact trajectory resume is forbidden."
        )


def require_hf_base_available(hf_base: str) -> None: ...


def configure_glm52_joint_sft(
    model_cfg, *, loss_backend: str = "torch_reference"
) -> None:
    if model_cfg.mtp_config is not None or bool(model_cfg.num_nextn_predict_layers):
        raise RuntimeError(
            "P1 requires a NoMTP GLM-5.2 checkpoint. Converting a standard HF "
            "checkpoint requires explicit mtp_config=None, "
            "num_nextn_predict_layers=None, and a strict_load=False model-only "
            "initialization flow."
        )
    model_cfg.compile_cfg = False
    model_cfg.attention.indexer_train_mode = "joint"
    model_cfg.attention.indexer_loss_cfg = DSAIndexerLossConfig(
        loss_coeff=1e-3,
        loss_type="sparse",
        backend=loss_backend,
    )


# TrainEngine.train_step / _get_total_loss / step_optimizer 保持现有实现：
# - projected_output 上的 AuxLossScaler 是唯一可导 Indexer KL 入口；
# - MoEModelOutputs.indexer_loss 是 detached scalar，只补充数值与日志，不会 double backward；
# - 不在设计里伪造 optimizer.step()，避免绕过 invalid-grad 和 zero_grad 逻辑。
