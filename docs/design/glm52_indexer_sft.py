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
    indexer_topk_query_chunk_size: int | None = None
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

        # 保留现有 main SparseMLA + selector preflight；不能只 probe 新 loss adapter。
        ensure_sparse_mla_and_selector_runtime_available(self.sparse_mla_backend)

        # 当前真实 build 使用 **self.model_dump()。nested BaseModel 会变成 dict，
        # 所以新增配置必须 exclude 后显式传对象，不能假定 module 收到 config 实例。
        kwargs = self.model_dump(
            exclude={
                "indexer_train_mode",
                "indexer_loss_cfg",
                "indexer_topk_query_chunk_size",
            }
        )
        indexer_backend = build_indexer_backend(
            selection_backend=self.sparse_mla_backend,
            query_chunk_size=self.indexer_topk_query_chunk_size,
            loss_cfg=self.indexer_loss_cfg,
        )
        return DSAMultiLatentAttention(
            **kwargs,
            **runtime_args,
            indexer_train_mode=self.indexer_train_mode,
            indexer_loss_cfg=self.indexer_loss_cfg,
            indexer_backend=indexer_backend,
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

    def detached(self) -> "DSAIndexerTeacher":
        return DSAIndexerTeacher(
            self.query.detach(), self.key.detach(), self.softmax_scale
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
    """短序列 fixed-ID sparse oracle。

    第一阶段 query chunk 只属于 no-grad selection；本函数仍保持一次完整调用。
    loss query/head chunk 是后续独立优化。这里假定 GLM-5.2 的 batch=1、
    teacher KV head=1，只为表达数学和安全 mask。
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

    def loss_for_indices(self, inputs, teacher, topk_indices, seq_ctx):
        if self.loss_cfg is None:
            raise RuntimeError("joint Indexer loss backend is missing")
        return self.backend.loss_on_fixed_topk(
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


def reuse_during_recompute(function, /, *args, **kwargs):
    """真实实现复用 PR #2039 的 checkpoint-local FIFO helper。"""

    ...


class GLM52AttnOutputs(TypedDict):
    raw_output: torch.Tensor
    projected_output: torch.Tensor
    softmax_lse: torch.Tensor
    dsa_topk_ids: torch.Tensor
    indexer_loss: NotRequired[torch.Tensor]


class GLM52DecoderLayerOutputs(TypedDict):
    hidden_states: torch.Tensor
    dsa_topk_ids: torch.Tensor
    indexer_loss: NotRequired[torch.Tensor]


class GLM52MicroBatchDecoderOutputs(TypedDict):
    """P2 list 契约；每个位置拥有独立 IDs/loss context/FIFO entry。"""

    hidden_states: list[torch.Tensor]
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
        **kwargs,
    ) -> None:
        super().__init__()
        self.indexer_train_mode = indexer_train_mode
        self.indexer: DSAIndexer | None = ...
        self.indexer_backend = indexer_backend
        # checkpoint frame 若按 callable 分桶，必须保存同一个 bound-method 对象，
        # 不能在 original/replay 临时重新取属性后依赖 object identity。
        self._selector_for_recompute = (
            self.indexer.select_topk if self.indexer is not None else None
        )

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
                inputs = self.indexer.project(
                    hidden_states.detach(),
                    q_resid.detach(),
                    position_embeddings,
                    seq_ctx,
                )
                # 固定 call-site/FIFO：original 真正 select，replay 从 frame 取 IDs。
                assert self._selector_for_recompute is not None
                dsa_topk_ids = reuse_during_recompute(
                    self._selector_for_recompute,
                    inputs.detached(),
                    seq_ctx,
                )
                teacher = DSAIndexerTeacher(
                    query.detach(), key.detach(), self.softmax_scale
                )
                stats = self.indexer.loss_for_indices(
                    inputs, teacher, dsa_topk_ids, seq_ctx
                )
                scaled_indexer_loss = indexer_loss_ctx.scale(stats=stats)
        elif dsa_topk_ids is None:
            raise RuntimeError(
                f"shared DSA layer {self.layer_idx} requires explicit dsa_topk_ids"
            )

        assert dsa_topk_ids is not None
        if dsa_topk_ids.dtype != torch.int32 or not dsa_topk_ids.is_contiguous():
            raise RuntimeError("dsa_topk_ids must be contiguous int32")

        sparse_output = self._sparse_mla(query, key, dsa_topk_ids, value_dim)
        attn_outputs = self._finish_attention(sparse_output)
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
    return (
        layer_output["hidden_states"],
        layer_output["dsa_topk_ids"],
        layer_output.get("indexer_loss"),
    )


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
        hidden_states,
        dsa_topk_ids=dsa_topk_ids,
        indexer_loss_ctx=indexer_loss_ctx,
    )
    if not (
        len(output["hidden_states"])
        == len(output["dsa_topk_ids"])
        == len(hidden_states)
    ):
        raise RuntimeError("micro-batch output lists must preserve input cardinality")
    return output


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
        self, indexer, teacher, topk_indices, seq_ctx, *, loss_type
    ):
        if self.loss_op is None:
            raise RuntimeError("frozen Indexer backend has no loss adapter")
        return self.loss_op(
            indexer,
            teacher,
            topk_indices,
            seq_ctx,
            loss_type=loss_type,
        )


class TileLangTopKSelector:
    """第一阶段只 chunk no-grad selection，不 chunk fixed-ID loss/backward。"""

    def __init__(self, query_chunk_size: int | None) -> None:
        self.query_chunk_size = query_chunk_size

    @torch.no_grad()
    def __call__(self, indexer, seq_ctx, *, index_topk):
        return tilelang_topk_query_chunked(
            indexer,
            seq_ctx,
            index_topk=index_topk,
            query_chunk_size=self.query_chunk_size,
        )


@torch.no_grad()
def tilelang_topk_query_chunked(
    indexer: DSAIndexerInputs,
    seq_ctx: SequenceContext,
    *,
    index_topk: int,
    query_chunk_size: int | None,
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
    if query_chunk_size is None:
        # None 严格保留现有 one-shot baseline。
        return _tilelang_topk_from_ranges(
            q, k, effective_weights, starts, ends, index_topk
        )

    chunks: list[torch.Tensor] = []
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
        chunks.append(ids[:valid_rows])
    return torch.cat(chunks, dim=0).to(torch.int32).contiguous()


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
    loss_cfg: DSAIndexerLossConfig | None,
) -> DSAIndexerBackendProtocol:
    if selection_backend == "torch":
        if query_chunk_size is not None:
            raise RuntimeError("P0 torch selector does not support query chunk")
        selector = torch_topk_selector_adapter
    elif selection_backend in {"tilelang", "cudnn_dsa"}:
        # current cudnn_dsa 的 Top-K 仍走 TileLang。
        selector = TileLangTopKSelector(query_chunk_size)
    else:
        raise AssertionError(selection_backend)

    loss_op = build_indexer_loss_backend(loss_cfg) if loss_cfg is not None else None
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

    if config.backend == "cudnn":
        # current sparse_mla_backend='cudnn_dsa' 是 hybrid，不满足这里的 full adapter。
        ensure_full_cudnn_indexer_loss_available()
        return cudnn_fixed_topk_indexer_loss

    raise AssertionError(config.backend)


def ensure_tilelang_fixed_id_loss_available() -> None: ...


def ensure_full_cudnn_indexer_loss_available() -> None: ...


def tilelang_fixed_topk_indexer_loss(*args, **kwargs) -> DSAIndexerLossStats: ...


def cudnn_fixed_topk_indexer_loss(
    indexer, teacher, topk_indices, seq_ctx, *, loss_type
) -> DSAIndexerLossStats:
    # cuDNN adapter 使用 sm_scale=1，因此只在边界临时折入 D^-1/2；梯度仍会
    # 经 cast/乘法回到 canonical head_weights 和 weights_proj。
    effective_weights = indexer.effective_weights(torch.bfloat16)
    return cudnn_indexer_loss_impl(
        indexer.q,
        indexer.k,
        effective_weights,
        teacher,
        topk_indices,
        seq_ctx,
        sm_scale=1.0,
        loss_type=loss_type,
    )


def cudnn_indexer_loss_impl(*args, **kwargs) -> DSAIndexerLossStats: ...


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
        return self


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
    model_cfg,
    *,
    loss_backend: str = "torch_reference",
    indexer_topk_query_chunk_size: int | None = None,
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
    model_cfg.attention.indexer_topk_query_chunk_size = (
        indexer_topk_query_chunk_size
    )
    model_cfg.attention.indexer_loss_cfg = DSAIndexerLossConfig(
        loss_coeff=1e-3,
        loss_type="sparse",
        backend=loss_backend,
    )


# TrainEngine.train_step / _get_total_loss / step_optimizer 保持现有实现：
# - projected_output 上的 AuxLossScaler 是唯一可导 Indexer KL 入口；
# - MoEModelOutputs.indexer_loss 是 detached scalar，只补充数值与日志，不会 double backward；
# - 不在设计里伪造 optimizer.step()，避免绕过 invalid-grad 和 zero_grad 逻辑。
