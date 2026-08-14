"""GLM-5.2 MoE 解耦 FSDP2 设计伪代码。

本文件用于代码评审，集中展示拟修改的接口和控制流，不是可直接运行的实现。
真实代码应按文件边界拆分提交，并复用 XTuner 已有类型、日志和错误处理。

设计目标：
1. 默认 coupled 路径保持现有行为。
2. decoupled 路径让 dense 与 routed expert 使用不同 FSDP mesh。
3. EP、expert-FSDP、dense-FSDP 都从同一个 root DeviceMesh 派生。
4. 第一版不改变 expert 的 EP/FSDP Shard(0) 布局。
5. HF load/save 必须根据参数角色选择正确 FSDP mesh。
6. SFT CLI、Trainer 和 TrainEngine 的公开调用方式保持不变。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Literal, Protocol


# =============================================================================
# xtuner/v1/config/fsdp.py
# =============================================================================


class FSDPConfigPseudo:
    """在现有 FSDPConfig 上增加一个字段。"""

    ep_size: int = 1
    tp_size: int = 1
    hsdp_sharding_size: int | None = None
    cpu_offload: bool = False
    torch_compile: bool = False
    moe_fsdp_mesh: Literal["coupled", "decoupled"] = "coupled"


class ModelConfigPseudo(Protocol):
    ep_size: int
    mesh_prefix: str
    mtp_config: object | None
    float8_cfg: object | None


class TrainerConfigPseudo(Protocol):
    model_cfg: ModelConfigPseudo
    fsdp_cfg: FSDPConfigPseudo
    dataloader_cfg: Any
    optim_cfg: Any
    lr_cfg: Any
    load_from: Any
    load_checkpoint_cfg: Any
    global_batch_size: int | None
    intra_layer_micro_batch: int
    total_step: int | None
    total_epoch: int | None
    sp_size: int


def validate_decoupled_v1(
    fsdp: FSDPConfigPseudo,
    model: ModelConfigPseudo,
    trainer: TrainerConfigPseudo,
) -> None:
    """第一版在统一入口做 fail-fast 校验。

    这些限制分为两类：
    - 当前实现明确不兼容：HSDP+EP、FP8 单一 padding mesh/load spec、MoE TP。
    - 原理上可兼容但第一版暂缓：SP、compile、offload、MTP。
    """

    if fsdp.moe_fsdp_mesh != "decoupled":
        return

    assert fsdp.ep_size == model.ep_size
    assert fsdp.ep_size > 1
    assert fsdp.tp_size == 1
    assert fsdp.hsdp_sharding_size is None
    assert model.mtp_config is None
    assert model.float8_cfg is None
    assert trainer.sp_size == 1
    assert fsdp.cpu_offload is False
    assert fsdp.torch_compile is False


# =============================================================================
# xtuner/v1/train/cli/sft.py -> Trainer.__init__ -> Trainer.fit 调用链
# =============================================================================


def sft_main_pseudo(trainer_cfg: TrainerConfigPseudo) -> None:
    """SFT CLI 保持薄入口，不在这里创建 dense/expert mesh。"""

    trainer = TrainerLifecyclePseudo.from_config(trainer_cfg)
    trainer.fit()
    existing_destroy_process_group_if_initialized()


class TrainEngineLifecyclePseudo:
    """对应 TrainEngine.__init__ 与 build_model 的现有时序。"""

    def __init__(
        self,
        *,
        model_cfg: ModelConfigPseudo,
        optim_cfg: Any,
        fsdp_cfg: FSDPConfigPseudo,
        intra_layer_micro_batch: int,
    ) -> None:
        self.model_cfg = model_cfg
        self.fsdp_cfg = fsdp_cfg

        # 关键顺序：meta 构造 -> FSDP 包装 -> optimizer。
        self.model = self.build_model()
        self.optimizer = existing_build_optimizer(optim_cfg, self.model)
        self.intra_layer_micro_batch = intra_layer_micro_batch

    def build_model(self) -> Any:
        model = existing_build_model_on_meta(self.model_cfg)

        # coupled/decoupled 的唯一训练入口仍是 BaseModel.fully_shard。
        # Trainer 和 TrainEngine 不直接查找或包装 experts。
        return model.fully_shard(self.fsdp_cfg)

    def from_hf(self, model_path: Any, strict: bool) -> None:
        self.model.from_hf(hf_path=model_path, strict=strict)

    def init_model_weights(self) -> None:
        self.model.init_weights()

    def train_step(self, engine_input: Any) -> Any:
        """保持现有 forward、loss.backward 和 micro-batch 控制流。"""

        return existing_train_step(self.model, engine_input)

    def clip_grad_norm(self) -> Any:
        # 解耦后唯一相关变化发生在 model.scale_and_reduce_grad() 内部。
        self.model.scale_and_reduce_grad()
        return existing_cal_grad_norm(self.model)

    def step_optimizer(self, grad_norm: Any) -> None:
        existing_step_optimizer(self.optimizer, grad_norm)

    def save_hf(self, output_dir: Any) -> None:
        # 公开接口不改；model.save_hf 内部按参数角色选择 FSDP mesh。
        self.model.save_hf(output_dir)

    def save_dcp(self, output_dir: Any) -> None:
        existing_save_dcp(self.model, self.optimizer, output_dir)


class TrainerLifecyclePseudo:
    """只展示解耦 mesh 相关的 Trainer 初始化和训练主链。"""

    @classmethod
    def from_config(cls, config: TrainerConfigPseudo) -> TrainerLifecyclePseudo:
        # 真实 Trainer.from_config 继续逐字段转发；不新增顶层 trainer 参数。
        return cls(
            model_cfg=config.model_cfg,
            fsdp_cfg=config.fsdp_cfg,
            optim_cfg=config.optim_cfg,
            dataloader_cfg=config.dataloader_cfg,
            lr_cfg=config.lr_cfg,
            load_from=config.load_from,
            load_checkpoint_cfg=config.load_checkpoint_cfg,
            sp_size=config.sp_size,
            global_batch_size=config.global_batch_size,
            intra_layer_micro_batch=config.intra_layer_micro_batch,
            total_step=config.total_step,
            total_epoch=config.total_epoch,
        )

    def __init__(
        self,
        *,
        model_cfg: ModelConfigPseudo,
        fsdp_cfg: FSDPConfigPseudo,
        optim_cfg: Any,
        dataloader_cfg: Any,
        lr_cfg: Any,
        load_from: Any,
        load_checkpoint_cfg: Any,
        sp_size: int,
        global_batch_size: int | None,
        intra_layer_micro_batch: int,
        total_step: int | None,
        total_epoch: int | None,
    ) -> None:
        self._init_dist()
        self._sp_size = sp_size
        self._fsdp_config = fsdp_cfg
        self.tokenizer = existing_build_tokenizer()

        # 这是数据使用的 (dp, sp, tp) mesh；不能拿它代替模型的 MoE root mesh。
        self.data_mesh = self._init_data_mesh(
            tp_size=fsdp_cfg.tp_size,
            sp_size=sp_size,
        )
        self.sp_mesh = self.data_mesh["sp"]

        # 先执行当前 EP/compile 冲突处理，再校验 decoupled 第一阶段边界。
        self._resolve_config_conflicts(self.tokenizer, model_cfg, dataloader_cfg, fsdp_cfg)
        validate_decoupled_v1(fsdp_cfg, model_cfg, self)

        self._dataloader = existing_build_dataloader(
            dataloader_cfg,
            data_mesh=self.data_mesh,
            tokenizer=self.tokenizer,
            global_batch_size=global_batch_size,
            total_step=total_step,
        )

        self._engine = self.build_engine(
            model_path=load_from,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
            load_checkpoint_cfg=load_checkpoint_cfg,
            intra_layer_micro_batch=intra_layer_micro_batch,
        )
        self._lr_scheduler = existing_build_lr_scheduler(
            lr_cfg,
            self._engine.optimizer,
            total_step=total_step,
            total_epoch=total_epoch,
        )

        # DCP 恢复必须在 model、optimizer 和 scheduler 都存在后执行。
        if load_checkpoint_cfg.checkpoint_path is not None:
            self._load_checkpoint(load_checkpoint_cfg)

        self._setup_existing_hooks_profiler_and_metrics()

    def build_engine(
        self,
        *,
        model_path: Any,
        model_cfg: ModelConfigPseudo,
        optim_cfg: Any,
        fsdp_cfg: FSDPConfigPseudo,
        load_checkpoint_cfg: Any,
        intra_layer_micro_batch: int,
    ) -> TrainEngineLifecyclePseudo:
        engine = TrainEngineLifecyclePseudo(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
            intra_layer_micro_batch=intra_layer_micro_batch,
        )

        if existing_should_load_hf(model_path, model_cfg, load_checkpoint_cfg):
            engine.from_hf(model_path, strict=True)
        elif load_checkpoint_cfg.checkpoint_path is None:
            engine.init_model_weights()
        return engine

    def fit(self) -> None:
        """训练主循环不增加 coupled/decoupled 分支。"""

        for data_batch in self._data_iter():
            engine_input = self._prepare_model_input(data_batch)
            train_step_info = self._engine.train_step(engine_input)

            grad_norm = self._engine.clip_grad_norm()
            self._engine.step_optimizer(grad_norm)

            self._log_step(train_step_info, grad_norm)
            self._lr_scheduler.step()
            self._maybe_save_hf()  # 间接进入 mesh-aware model.save_hf
            self._maybe_save_dcp()  # 继续复用 PyTorch DCP
            self._maybe_collect_python_gc_every_50_steps()

    def _init_dist(self) -> None: ...

    @property
    def sp_size(self) -> int:
        return self._sp_size

    def _init_data_mesh(self, *, tp_size: int, sp_size: int) -> Any: ...

    def _resolve_config_conflicts(
        self,
        tokenizer: Any,
        model_cfg: ModelConfigPseudo,
        dataloader_cfg: Any,
        fsdp_cfg: FSDPConfigPseudo,
    ) -> None: ...

    def _load_checkpoint(self, load_checkpoint_cfg: Any) -> None: ...

    def _setup_existing_hooks_profiler_and_metrics(self) -> None: ...

    def _data_iter(self): ...

    def _prepare_model_input(self, data_batch: Any) -> Any: ...

    def _log_step(self, train_step_info: Any, grad_norm: Any) -> None: ...

    def _maybe_save_hf(self) -> None: ...

    def _maybe_save_dcp(self) -> None: ...

    def _maybe_collect_python_gc_every_50_steps(self) -> None: ...


def existing_destroy_process_group_if_initialized() -> None: ...


def existing_build_model_on_meta(model_cfg: ModelConfigPseudo) -> Any: ...


def existing_build_tokenizer() -> Any: ...


def existing_build_optimizer(optim_cfg: Any, model: Any) -> Any: ...


def existing_train_step(model: Any, engine_input: Any) -> Any: ...


def existing_cal_grad_norm(model: Any) -> Any: ...


def existing_step_optimizer(optimizer: Any, grad_norm: Any) -> None: ...


def existing_save_dcp(model: Any, optimizer: Any, output_dir: Any) -> None: ...


def existing_build_dataloader(
    dataloader_cfg: Any,
    *,
    data_mesh: Any,
    tokenizer: Any,
    global_batch_size: int | None,
    total_step: int | None,
) -> Any: ...


def existing_build_lr_scheduler(
    lr_cfg: Any,
    optimizer: Any,
    *,
    total_step: int | None,
    total_epoch: int | None,
) -> Any: ...


def existing_should_load_hf(
    model_path: Any,
    model_cfg: ModelConfigPseudo,
    load_checkpoint_cfg: Any,
) -> bool: ...


# =============================================================================
# xtuner/v1/model/moe/parallel_mesh.py（可新建，也可先放在 moe.py）
# =============================================================================


class DeviceMeshPseudo(Protocol):
    mesh_dim_names: tuple[str, ...]

    def __getitem__(self, names: str | tuple[str, ...]) -> DeviceMeshPseudo: ...

    def size(self) -> int: ...

    def _flatten(self, mesh_dim_name: str) -> DeviceMeshPseudo: ...


def init_device_mesh_pseudo(
    device: str,
    shape: tuple[int, ...],
    mesh_dim_names: tuple[str, ...],
) -> DeviceMeshPseudo: ...


@dataclass(frozen=True)
class MoEParallelMeshes:
    """同一个 root mesh 的不同视图。"""

    root: DeviceMeshPseudo
    ep: DeviceMeshPseudo
    expert_fsdp: DeviceMeshPseudo
    dense_fsdp: DeviceMeshPseudo


def build_moe_root_once(
    *,
    device: str,
    world_size: int,
    ep_size: int,
    prefix: str,
) -> tuple[DeviceMeshPseudo, DeviceMeshPseudo, DeviceMeshPseudo]:
    """模型构造时只创建一次 root。

    保留现有 ``prefix.fsdp`` 名称，降低 coupled 路径兼容风险。
    ``prefix.fsdp`` 在 decoupled 模式下承担 expert-FSDP 轴的职责。
    """

    assert world_size % ep_size == 0
    fsdp_name = f"{prefix}.fsdp"
    ep_name = f"{prefix}.ep"

    root = init_device_mesh_pseudo(
        device,
        (world_size // ep_size, ep_size),
        mesh_dim_names=(fsdp_name, ep_name),
    )
    return root, root[ep_name], root[fsdp_name]


def select_fsdp_meshes(
    *,
    root: DeviceMeshPseudo,
    expert_fsdp: DeviceMeshPseudo,
    prefix: str,
    mode: Literal["coupled", "decoupled"],
) -> MoEParallelMeshes:
    """根据开关选择 dense mesh，expert mesh 始终不变。"""

    ep_name = f"{prefix}.ep"
    fsdp_name = f"{prefix}.fsdp"

    if mode == "coupled":
        dense_fsdp = expert_fsdp
    else:
        dense_fsdp = root[(fsdp_name, ep_name)]._flatten(
            mesh_dim_name=f"{prefix}.dense_fsdp"
        )

    return MoEParallelMeshes(
        root=root,
        ep=root[ep_name],
        expert_fsdp=expert_fsdp,
        dense_fsdp=dense_fsdp,
    )


# 8 卡 EP4 的预期通信组：
#
# root:
#                 EP0  EP1  EP2  EP3
# FSDP 第 0 行      0    1    2    3
# FSDP 第 1 行      4    5    6    7
#
# ep groups:          [0,1,2,3] / [4,5,6,7]
# expert-FSDP groups: [0,4] / [1,5] / [2,6] / [3,7]
# dense-FSDP group:   [0,1,2,3,4,5,6,7]（仅 decoupled）


# =============================================================================
# 参数角色索引：以模块 ownership 为事实来源
# =============================================================================


class ParamRole(str, Enum):
    DENSE = "dense"
    ROUTED_EXPERT = "routed_expert"


def clean_param_name(name: str) -> str:
    """真实实现复用 BaseModel._clean_param_name。"""

    return name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")


@dataclass
class ParamRoleIndex:
    """同时服务 HF 名称查询和训练期对象查询。"""

    role_by_name: dict[str, ParamRole]
    expert_param_ids: set[int]

    @classmethod
    def before_fsdp(cls, model: Any, routed_expert_modules: Iterable[Any]) -> ParamRoleIndex:
        """FSDP 前建立稳定 canonical name 索引。

        ``model.named_parameters()`` 给出根路径；模块引用只用于确定哪些参数属于
        ``MoEDecoderLayer.experts``，不在其他路径重复解析字符串。
        """

        routed_ids = {
            id(param)
            for module in routed_expert_modules
            for param in module.parameters()
        }
        role_by_name = {
            clean_param_name(name): (
                ParamRole.ROUTED_EXPERT if id(param) in routed_ids else ParamRole.DENSE
            )
            for name, param in model.named_parameters(remove_duplicate=False)
        }
        return cls(role_by_name=role_by_name, expert_param_ids=set())

    def refresh_runtime_ids(self, routed_expert_modules: Iterable[Any]) -> None:
        """FSDP 和 to_empty() 完成后重建对象索引。"""

        self.expert_param_ids = {
            id(param)
            for module in routed_expert_modules
            for param in module.parameters()
        }

    def role_for_name(self, name: str) -> ParamRole:
        return self.role_by_name[clean_param_name(name)]

    def is_runtime_expert(self, param: Any) -> bool:
        return id(param) in self.expert_param_ids


# =============================================================================
# xtuner/v1/model/base.py：支持 outer FSDP 排除 inner experts
# =============================================================================


class FSDPModulePseudo:
    """代表 torch.distributed.fsdp.FSDPModule。"""


def fully_shard_pseudo(
    module: Any,
    *,
    mesh: DeviceMeshPseudo,
    ignored_params: set[Any] | None,
    mp_policy: Any,
    reshard_after_forward: bool,
    offload_policy: Any,
) -> None: ...


def collect_existing_fp32_ignored_params(
    *,
    root_model: Any,
    target: Any,
) -> set[Any]:
    """复用现有 fp32_keys_pattern 逻辑，但不进入 nested FSDP child。

    真实实现需要保留 BaseModel._fully_shard 现有 HF name mapping 行为；这里只展示
    新增的“跳过已管理子模块”约束。
    """

    ignored: set[Any] = set()

    def visit(module: Any) -> None:
        if module is not target and isinstance(module, FSDPModulePseudo):
            return
        for name, param in module.named_parameters(recurse=False):
            if existing_fp32_pattern_matches(root_model, module, name, param):
                ignored.add(param)
        for child in module.children():
            visit(child)

    visit(target)
    return ignored


def existing_fp32_pattern_matches(
    root_model: Any,
    module: Any,
    name: str,
    param: Any,
) -> bool: ...


def fully_shard_with_ignored(
    root_model: Any,
    *,
    module: Any,
    mesh: DeviceMeshPseudo,
    mp_policy: Any,
    reshard_after_forward: bool,
    offload_policy: Any,
    extra_ignored_params: set[Any] | None = None,
) -> None:
    """对现有 BaseModel._fully_shard 的最小接口扩展。"""

    ignored = collect_existing_fp32_ignored_params(
        root_model=root_model,
        target=module,
    )
    ignored.update(extra_ignored_params or ())

    fully_shard_pseudo(
        module,
        mesh=mesh,
        ignored_params=ignored or None,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        offload_policy=offload_policy,
    )


# =============================================================================
# xtuner/v1/model/moe/moe.py
# =============================================================================


class MoEDecoderLayerPseudo:
    experts: Any


class MoEDesignPseudo:
    """MoE 类拟修改的关键控制流。"""

    def __init__(self, config: ModelConfigPseudo) -> None:
        self.config = config
        self.layers: dict[str, Any] = {}
        self.embed_tokens: Any = None
        self.norm: Any = None
        self.lm_head: Any = None
        self.mtp_block: Any = None

        if config.ep_size > 1:
            root, ep, expert_fsdp = build_moe_root_once(
                device=existing_device(),
                world_size=existing_world_size(),
                ep_size=config.ep_size,
                prefix=config.mesh_prefix,
            )
            self._world_mesh = root
            self.ep_mesh = ep
            self.expert_fsdp_mesh = expert_fsdp
        else:
            # EP=1 继续走现有路径，避免第一版扩大修改面。
            self._world_mesh = None
            self.ep_mesh = None
            self.expert_fsdp_mesh = None

        # 真实 MoE.__init__ 应先按现有顺序构造 norm、lm_head、layers、rotary_emb、
        # embed_tokens 等全部模块，再建立参数角色索引。这里只收起与本设计无关的构造细节。
        self.build_all_existing_modules()
        self.routed_expert_modules = [
            layer.experts
            for layer in self.layers.values()
            if isinstance(layer, MoEDecoderLayerPseudo)
        ]
        self.param_roles = ParamRoleIndex.before_fsdp(
            self,
            self.routed_expert_modules,
        )

        # 保持当前时机：load spec 在 FSDP 前只观察到 EP placement。
        self._init_load_spec()

    def build_all_existing_modules(self) -> None:
        self.norm = self.build_existing_norm()
        self.lm_head = self.build_existing_lm_head()
        self.layers = self.build_layers_with_existing_ep_mesh()
        self.rotary_emb = self.build_existing_rotary_embedding()
        self.embed_tokens = self.build_existing_embeddings()

    def build_existing_norm(self) -> Any: ...

    def build_existing_lm_head(self) -> Any: ...

    def build_layers_with_existing_ep_mesh(self) -> dict[str, Any]: ...

    def build_existing_rotary_embedding(self) -> Any: ...

    def build_existing_embeddings(self) -> Any: ...

    def named_parameters(self, remove_duplicate: bool = True): ...

    def _init_load_spec(self) -> None: ...

    def _fully_shard_coupled_existing(self, fsdp_config: FSDPConfigPseudo) -> None:
        """调用当前已验证实现；默认模式不重写控制流。"""

        ...

    def fully_shard(self, fsdp_config: FSDPConfigPseudo) -> MoEDesignPseudo:
        assert fsdp_config.ep_size == self.config.ep_size

        if fsdp_config.moe_fsdp_mesh == "coupled":
            # 默认路径必须保持 _replicate_other_params、整层 fully_shard、梯度和 HF 行为。
            self._fully_shard_coupled_existing(fsdp_config)
            return self

        assert self._world_mesh is not None
        assert self.ep_mesh is not None
        assert self.expert_fsdp_mesh is not None

        meshes = select_fsdp_meshes(
            root=self._world_mesh,
            expert_fsdp=self.expert_fsdp_mesh,
            prefix=self.config.mesh_prefix,
            mode="decoupled",
        )
        self.dense_fsdp_mesh = meshes.dense_fsdp

        # BaseModel 旧接口仍读取 self.fsdp_mesh；在 decoupled 下它代表 dense mesh。
        # 所有 expert HF 路径必须改为 runtime_fsdp_mesh(...)，不能继续盲读该字段。
        self.fsdp_mesh = self.dense_fsdp_mesh

        self.promote_trainable_params_to_fp32_as_existing()
        mp_policy = self.build_existing_mixed_precision_policy(fsdp_config)

        # 关键差异：decoupled 不调用 _replicate_other_params(self)。
        for layer_idx, raw_layer in list(self.layers.items()):
            idx = int(layer_idx)
            current_expert_params: set[Any] = set()

            if isinstance(raw_layer, MoEDecoderLayerPseudo):
                fully_shard_with_ignored(
                    self,
                    module=raw_layer.experts,
                    mesh=meshes.expert_fsdp,
                    mp_policy=mp_policy,
                    reshard_after_forward=self.expert_reshard_after_forward(idx),
                    offload_policy=None,
                    # 第一版不传 AutoModel 的 Shard(1)，保留 XTuner 默认 Shard(0)。
                )
                # 必须在 inner FSDP 后重新读取，不能使用 FSDP 前缓存的 parameter id。
                current_expert_params = set(raw_layer.experts.parameters())

            wrapped_layer = self.maybe_checkpoint_wrap(raw_layer, idx)
            self.layers[layer_idx] = wrapped_layer

            fully_shard_with_ignored(
                self,
                module=wrapped_layer,
                mesh=meshes.dense_fsdp,
                mp_policy=mp_policy,
                reshard_after_forward=self.layer_reshard_after_forward(idx),
                offload_policy=None,
                extra_ignored_params=current_expert_params,
            )

        self.preserve_existing_layer_prefetch_order()

        fully_shard_with_ignored(
            self,
            module=self.embed_tokens,
            mesh=meshes.dense_fsdp,
            mp_policy=mp_policy,
            reshard_after_forward=self.config.embed_reshard_after_forward,
            offload_policy=None,
        )
        fully_shard_with_ignored(
            self,
            module=self.norm,
            mesh=meshes.dense_fsdp,
            mp_policy=mp_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=None,
        )
        fully_shard_with_ignored(
            self,
            module=self.lm_head,
            mesh=meshes.dense_fsdp,
            mp_policy=self.existing_lm_head_mp_policy(fsdp_config),
            reshard_after_forward=False,
            offload_policy=None,
        )

        # 根模块只补管尚未归属于 child FSDP unit 的参数。
        fully_shard_with_ignored(
            self,
            module=self,
            mesh=meshes.dense_fsdp,
            mp_policy=mp_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=None,
        )

        self.preserve_existing_root_prefetch_order()
        self._to_empty_meta()

        # 参数对象可能被 FSDP/to_empty 替换；最后再重建训练期 id 索引。
        self.param_roles.refresh_runtime_ids(self.routed_expert_modules)
        return self

    def promote_trainable_params_to_fp32_as_existing(self) -> None: ...

    def build_existing_mixed_precision_policy(self, fsdp_config: FSDPConfigPseudo) -> Any: ...

    def existing_lm_head_mp_policy(self, fsdp_config: FSDPConfigPseudo) -> Any: ...

    def expert_reshard_after_forward(self, layer_idx: int) -> bool: ...

    def layer_reshard_after_forward(self, layer_idx: int) -> bool: ...

    def maybe_checkpoint_wrap(self, layer: Any, layer_idx: int) -> Any: ...

    def preserve_existing_layer_prefetch_order(self) -> None: ...

    def preserve_existing_root_prefetch_order(self) -> None: ...

    def _to_empty_meta(self) -> None: ...

    def scale_and_reduce_grad(self) -> None:
        """保留 expert 缩放，只对真实 Replicate placement 做平均。"""

        for _name, param in self.trainable_parameters():
            if param.grad is None:
                continue

            if self.param_roles.is_runtime_expert(param):
                # expert 不跨 EP 保存副本；除 EP 保持当前 loss 平均语义。
                param.grad.div_(self.ep_mesh.size())
                continue

            # coupled dense 仍可能带 EP Replicate；decoupled dense 通常没有。
            reduce_only_over_actual_replicate_placements(param)

    def trainable_parameters(self): ...


def existing_device() -> str: ...


def existing_world_size() -> int: ...


def reduce_only_over_actual_replicate_placements(param: Any) -> None: ...


# =============================================================================
# xtuner/v1/model/base.py：mesh-aware HF load/save
# =============================================================================


class LoadEnumPseudo(str, Enum):
    SAME = "same"
    FUSED = "fused"
    SHARD = "shard"


@dataclass(frozen=True)
class LoadSpecPseudo:
    name: str
    load_enum: LoadEnumPseudo
    hf_keys: list[str]
    group: Any = None


def runtime_fsdp_mesh(
    model: MoEDesignPseudo,
    param_name: str,
) -> DeviceMeshPseudo:
    """按参数角色选择 FSDP group，而不是统一读取 self.fsdp_mesh。"""

    role = model.param_roles.role_for_name(param_name)
    if role is ParamRole.ROUTED_EXPERT:
        return model.expert_fsdp_mesh
    return model.dense_fsdp_mesh


def load_one_hf_param_mesh_aware(
    model: MoEDesignPseudo,
    param: Any,
    load_spec: LoadSpecPseudo,
    checkpoint_loader: Any,
) -> None:
    """三个现有 load helper 都要显式使用参数对应的 FSDP mesh。"""

    fsdp_mesh = runtime_fsdp_mesh(model, load_spec.name)

    if load_spec.load_enum is LoadEnumPseudo.SAME:
        model._load_same_hf_param(  # type: ignore[attr-defined]
            param,
            load_spec,
            checkpoint_loader,
            fsdp_mesh=fsdp_mesh,
        )
    elif load_spec.load_enum is LoadEnumPseudo.FUSED:
        model._load_fused_hf_param(  # type: ignore[attr-defined]
            param,
            load_spec,
            checkpoint_loader,
            fsdp_mesh=fsdp_mesh,
        )
    else:
        model._load_shard_hf_param(  # type: ignore[attr-defined]
            param,
            load_spec,
            checkpoint_loader,
            fsdp_mesh=fsdp_mesh,
        )


def group_hf_params_by_load_and_role(
    model: MoEDesignPseudo,
    state_dict: dict[str, Any],
    load_specs: dict[str, LoadSpecPseudo],
) -> dict[tuple[LoadEnumPseudo, ParamRole], list[tuple[Any, LoadSpecPseudo]]]:
    """同一个 gather bucket 中的参数必须使用同一个 FSDP mesh。"""

    buckets: dict[
        tuple[LoadEnumPseudo, ParamRole],
        list[tuple[Any, LoadSpecPseudo]],
    ] = {}

    for raw_name, tensor in state_dict.items():
        name = clean_param_name(raw_name)
        load_spec = load_specs[name]
        role = model.param_roles.role_for_name(name)
        buckets.setdefault((load_spec.load_enum, role), []).append(
            (tensor, load_spec)
        )
    return buckets


def fsdp_foreach_allgather_mesh_aware(
    tensor_list: list[Any],
    load_spec_list: list[LoadSpecPseudo],
    *,
    fsdp_mesh: DeviceMeshPseudo,
) -> list[Any]:
    """现有 _fsdp_foreach_allgather 的算法保持不变，只把 mesh 改为显式参数。"""

    return existing_pad_gather_unpad_algorithm(
        tensor_list,
        load_spec_list,
        process_group=fsdp_mesh,
    )


def iter_hf_save_chunks_mesh_aware(model: MoEDesignPseudo):
    """同步保存按 (LoadEnum, ParamRole) 依次生成 chunk。

    真实实现还必须同步修改：
    - _get_safe_tensor_num
    - _get_same_hf_param
    - _get_fused_hf_param
    - _get_shard_hf_param
    - _iter_hf_save_chunks
    - _save_hf

    异步 HF save 和 RL weight update 第一版不承诺，但不能遗忘它们复用了这些 helper。
    """

    buckets = group_hf_params_by_load_and_role(
        model,
        model.state_dict(),  # type: ignore[attr-defined]
        model.load_spec_mapping,  # type: ignore[attr-defined]
    )

    for (_load_enum, role), params in buckets.items():
        fsdp_mesh = (
            model.expert_fsdp_mesh
            if role is ParamRole.ROUTED_EXPERT
            else model.dense_fsdp_mesh
        )
        yield from existing_hf_generator_for_one_bucket(
            params,
            fsdp_mesh=fsdp_mesh,
        )


def existing_pad_gather_unpad_algorithm(
    tensor_list: list[Any],
    load_spec_list: list[LoadSpecPseudo],
    *,
    process_group: DeviceMeshPseudo,
) -> list[Any]: ...


def existing_hf_generator_for_one_bucket(
    params: list[tuple[Any, LoadSpecPseudo]],
    *,
    fsdp_mesh: DeviceMeshPseudo,
): ...


# =============================================================================
# xtuner/v1/engine/train_engine.py：DCP 不另造格式
# =============================================================================


def save_and_resume_same_topology_with_existing_dcp(engine: Any) -> None:
    """公开 DCP 接口不变，第一版只定义相同 decoupled topology 的边界。"""

    state = engine._get_dcp_state_dict(
        cpu_offload=True,
        save_optimizer=True,
    )
    engine.existing_dcp_save(state)
    engine.existing_dcp_load_and_set_state(state)

    # Trainer 现有逻辑另外恢复 scheduler 与 train_state.json。
