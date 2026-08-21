# GLM-5.2：XTuner 解耦 FSDP2 Mesh 初步设计

> 状态：设计评审稿，尚未实现。
>
> 核对基线：XTuner `9dabb956`；AutoModel `e2c48886`。
>
> 第一阶段目标：单机 8 卡、GLM-5.2 6 层 NoMTP、TP=SP=PP=1、EP=2/4、BF16、eager 模式。
> 完整模型、多机、MTP、FP8、compile 等能力必须在后续阶段单独验证，不能由 6 层实验直接外推。

## 1. 先给结论

XTuner 集成 AutoModel 类似的“dense 与 routed expert 使用不同 FSDP mesh”在技术上可行，因为两边都基于
PyTorch FSDP2、`DeviceMesh` 和 `DTensor`。但它不是把 `self.fsdp_mesh` 从 2 卡改成 8 卡这么简单，至少需要
同时处理以下四件事：

1. **Mesh**：所有 EP、expert-FSDP、dense-FSDP 子 mesh 必须来自同一个 root mesh。
2. **参数归属**：routed expert 使用内层 FSDP；attention、router、shared expert、norm 等使用外层 FSDP。
3. **梯度语义**：保留 expert 的 `1 / EP` 缩放，去掉解耦后 dense 参数不再需要的 EP 副本规约。
4. **Checkpoint**：HF 加载/导出不能再假设所有参数都使用同一个 `self.fsdp_mesh`。

因此，建议采用**显式开关、默认保持旧行为、分阶段提交**的方案。第一版只完成 NoMTP、非 FP8、非
compile 的正确性闭环，不直接复制 AutoModel 的所有并行能力。

## 2. 两边当前到底怎么切

### 2.1 AutoModel：nested FSDP2

AutoModel 先创建主 `device_mesh`：

```text
(pp, dp_replicate, dp_shard, cp, tp)
```

然后从主 mesh 的非 PP 维度派生 MoE mesh：

```text
(ep_shard, ep)
```

训练 MoE 层时执行两次 `fully_shard`：

1. 内层：`fully_shard(moe.experts, mesh=ep_shard_mesh)`，只管理 routed expert。
2. 外层：`fully_shard(decoder_block, mesh=dense_fsdp_mesh, ignored_params=expert_params)`，管理其余参数。

在本次 8 卡、TP=CP=PP=1、EP=4 的实验里，实际拓扑是：

```text
MoE root mesh [expert-FSDP=2, EP=4]

                    EP0  EP1  EP2  EP3
expert-FSDP 第 0 份   0    1    2    3
expert-FSDP 第 1 份   4    5    6    7

EP groups:          [0,1,2,3] / [4,5,6,7]
expert-FSDP groups: [0,4] / [1,5] / [2,6] / [3,7]
dense-FSDP group:   [0,1,2,3,4,5,6,7]
```

这里的“expert-FSDP group `[0,4]`”不是把四个专家拦腰切成两组，而是：GPU 0 和 GPU 4 持有**同一批
本地专家权重的不同参数分片**；EP 决定一张卡负责哪些专家，expert-FSDP 再决定该卡负责的专家权重如何
继续分片。

### 2.2 XTuner：当前 FSDP 与 EP 耦合

XTuner 当前创建 `(world_size / EP, EP)` 的二维 mesh：

```text
(fsdp, ep)
```

然后：

- `GroupedLinear` 先把 routed expert 权重沿 EP 做 `Shard(0)`；
- `_replicate_other_params()` 把非 expert 参数沿 EP 做 `Replicate()`；
- 整个 decoder layer 只沿 `fsdp` 轴执行一次 `fully_shard()`。

所以同样在 8 卡、EP=4 时：

```text
EP groups:        [0,1,2,3] / [4,5,6,7]
FSDP groups:      [0,4] / [1,5] / [2,6] / [3,7]
dense 切分份数:   2
dense EP 副本数:  4
```

例如一份 80 GB 的 dense 参数只考虑参数本身：

```text
当前 XTuner EP4：80 / 2 = 40 GB/卡，再沿 EP 保存 4 份相同 shard
目标解耦 EP4：    80 / 8 = 10 GB/卡，不再沿 EP 复制 dense shard
```

实际训练显存还包含梯度、AdamW 状态、激活、通信 buffer 和 allocator reserved memory，不能直接用这个
例子预测最终显存；它只解释了为什么 EP 增大时，XTuner 当前 dense 参数和 optimizer state 会变重。

### 2.3 相同点与关键差异

| 项目 | AutoModel | XTuner 当前 | 解耦后目标 |
|---|---|---|---|
| 基础能力 | PyTorch FSDP2 / DeviceMesh / DTensor | PyTorch FSDP2 / DeviceMesh / DTensor | 继续复用现有能力 |
| EP 权重分布 | routed expert 沿 EP 切分 | routed expert 沿 EP 切分 | 保持 XTuner 现状 |
| expert-FSDP | 内层 `fully_shard(experts)` | 整层 FSDP 顺带切 expert | 新增内层 expert FSDP |
| dense-FSDP | 外层 block 使用主 dense mesh | 只使用 `world / EP` 大小的 FSDP 轴 | 外层 block 使用全 rank dense mesh |
| dense 沿 EP | 不复制 | 显式 `Replicate()` | 不再复制 |
| routed expert 排除 | 外层 block 显式忽略 | 无独立外层/内层归属 | 外层 block 显式忽略 |

## 3. 目标拓扑

第一版仍使用 XTuner 熟悉的二维 root mesh，只增加不同的视图：

```text
root mesh = (expert_fsdp, ep)

ep_mesh          = root[ep]
expert_fsdp_mesh = root[expert_fsdp]
dense_fsdp_mesh  = expert_fsdp_mesh                         # coupled
                 = flatten(root[expert_fsdp, ep])            # decoupled
```

三者必须来自**同一个 root mesh**。不能重新创建一个互不关联的 1D world mesh 给 dense 使用，否则 FSDP2
和已有 EP DTensor 可能认为两套子 mesh 没有相同 parent，导致 placement 组合或 collective 出错。
full-rank `dense_fsdp_mesh` 应在确认启用 `decoupled` 后延迟创建，避免默认 `coupled` 模式无故多创建进程组。

8 卡下的预期大小如下：

| 配置 | 当前 dense-FSDP | 解耦 dense-FSDP | expert-FSDP | EP |
|---|---:|---:|---:|---:|
| EP=2 | 4 | 8 | 4 | 2 |
| EP=4 | 2 | 8 | 2 | 4 |

### 3.1 参数归属必须按模块确定

| 参数 | 所属 FSDP mesh | 是否参与 EP | 说明 |
|---|---|---|---|
| attention / DSA indexer | dense-FSDP | 否 | 普通 dense 参数 |
| router / gate | dense-FSDP | 否 | router 不是 routed expert 权重 |
| shared expert | dense-FSDP | 否 | 每个 token 都可能执行，语义上属于 dense 路径 |
| norm / embedding / LM head | dense-FSDP | 否 | 模型级 dense 参数 |
| `MoEDecoderLayer.experts` | expert-FSDP | 是 | 唯一进入内层 FSDP 的 routed expert |

实现时应以 `MoEDecoderLayer.experts` 的模块边界作为唯一 ownership 来源，并生成两份索引：

- FSDP 完成后，按当前 `Parameter` 对象重建对象索引，供梯度路径使用；
- 把 wrapper 前后的参数名统一清洗成 canonical name，建立名称索引，供 `state_dict()`、HF 加载和导出使用。

不能只在模型构造时缓存参数 `id`：FSDP2 可能替换受管参数，而 `state_dict()` 中的 Tensor 也不保证与原
`Parameter` 同一对象。也不要在加载、保存、梯度代码里到处新增 `".experts" in name`，因为 activation
checkpoint 和 compile wrapper 会改变参数路径。

## 4. 第一版设计决策

### 4.1 配置开关

在 `FSDPConfig` 增加：

```python
moe_fsdp_mesh: Literal["coupled", "decoupled"] = "coupled"
```

- `coupled`：默认值，保持当前拓扑、参数 placement 和数值行为。
- `decoupled`：启用新的 nested FSDP2 路径。

第一版 `decoupled` 必须校验：`ep_size > 1`、TP=SP=PP=1、未启用 HSDP、MTP、FP8、CPU offload 和
compile。它们不是永远不兼容，而是不能在核心路径尚未稳定时同时改变太多变量。

### 4.2 第一版不复制 AutoModel 的 expert `Shard(1)`

AutoModel 对二维及以上 expert 权重使用 `_moe_shard_placement()`，通常沿参数维度 1 做 FSDP 分片；XTuner
当前 FSDP 默认沿参数维度 0 分片，而且 HF fused expert 加载逻辑明确假设 EP 和 FSDP 都沿维度 0。

因此第一版应该：

- 保留 XTuner 当前 EP `Shard(0)`；
- 内层 expert FSDP 也先保留当前默认 `Shard(0)`；
- 只改变 dense/expert 使用哪一个 FSDP mesh，不同时改变 expert 参数布局。

后续若验证 `Shard(1)` 对完整模型更合适，应作为独立优化提交，并同时修改 HF 转换和 round-trip 测试。

### 4.3 nested FSDP 与 activation checkpoint 的顺序

当前 XTuner 先用 `checkpoint_wrapper(layer)` 包装 layer，再对整个 wrapper 执行 FSDP。解耦后需要先找到
原始 `layer.experts`，否则 wrapper 会增加模块路径和 ownership 识别难度。建议顺序：

1. 取得原始 decoder layer；
2. 对 `raw_layer.experts` 执行内层 FSDP；
3. 根据 recompute 配置包装整个 raw layer；
4. 对包装后的 layer 执行外层 dense FSDP，并忽略 expert 参数；
5. 保持当前 layer-level forward prefetch 与最后一层 `reshard_after_forward` 逻辑。

这个顺序需要专门测试 checkpoint replay、expert unshard/reshard 和梯度是否一致。第一版关闭 compile，避免
`_orig_mod` 与 checkpoint wrapper 同时干扰模块发现和 hook 顺序。

### 4.4 梯度处理

当前 `scale_and_reduce_grad()` 做两类工作：

- routed expert 不跨 EP all-reduce，只除以 `EP size`，保持 loss 平均语义；
- 对具有 `Replicate` placement 的普通参数，在实际副本组上做平均。

解耦后 dense 参数沿全 rank FSDP 使用 `Shard`，通常不再有 EP `Replicate`，因此不会进入第二类手工规约；
expert 的 `1 / EP` 缩放仍必须保留。通用的 `Replicate` 检查也不能删除，因为被 FSDP 忽略的 FP32 参数等
特殊参数仍可能真的存在副本。

`cal_grad_norm()` 已按 `(device_mesh, placements)` 对 DTensor 梯度分组，再沿每个 `Shard` mesh 维做
all-reduce；从机制上可以同时处理 dense 和 expert 两种 mesh。第一版先不改这段实现，但必须用 coupled/
decoupled 同 batch 测试确认没有漏算或重复计算 expert norm。Optimizer 在 FSDP 后构建，也要确认 AdamW 对
两类 DTensor 参数都创建了正确的本地状态。

### 4.5 HF checkpoint 是最大改动点

当前 HF 相关方法多处直接使用 `self.fsdp_mesh`：

- `_load_same_hf_param()`；
- `_load_fused_hf_param()`；
- `_load_shard_hf_param()`；
- `_fsdp_foreach_allgather()`。

解耦后 dense 参数应使用 `dense_fsdp_mesh`，expert 参数应使用 `expert_fsdp_mesh`。保存时不能只按
`LoadEnum.SAME/FUSED/SHARD` 分桶，因为一个 `FUSED` 桶里可能同时包含 dense fused 参数和 expert fused
参数。必须至少按下面两个维度分桶：

```text
(load_enum, runtime_fsdp_mesh)
```

同时要复核 `_get_safe_tensor_num()`、`_iter_hf_save_chunks()`、同步/异步 HF save plan 和 fused 参数的
save-rank 分配：mesh 分桶后每个 rank 生成 chunk 的数量和顺序可能变化，最终 index 必须确认没有重复 key、
缺失 key 或同名文件覆盖。只修 `_fsdp_foreach_allgather()` 还不算完成 checkpoint 兼容。

`_init_load_spec()` 应继续在 FSDP 前构建：此时 expert 只有 EP 的一个 `Shard`，仍符合它当前“最多一个
Shard placement”的限制。第一版不要在 nested FSDP 后重建 load spec；FP8 padding 会触发重建，因此 FP8
暂缓支持。

### 4.6 DCP 先复用官方接口，但必须实测

`TrainEngine` 已使用 PyTorch 官方：

- `get_model_state_dict()` / `set_model_state_dict()`；
- `get_optimizer_state_dict()` / `set_optimizer_state_dict()`；
- `torch.distributed.checkpoint.save()` / `load()`。

其中 DCP 负责 model 和 optimizer；Trainer 另外保存 `lr_scheduler` 与 `train_state.json`。第一版不另造
checkpoint 格式。需要验证同一解耦拓扑下，DCP 恢复 model/AdamW，并由 Trainer 恢复 scheduler/step 后，
下一步与未中断 run 连续。`coupled -> decoupled` 或 EP2 -> EP4 的 DCP 跨拓扑恢复暂不承诺，必要时先通过
consolidated HF 权重转换。

## 5. 单文件伪代码

下面只展示关键控制流，便于评审改动边界；它不是可以直接复制运行的实现。

```python
from dataclasses import dataclass
from typing import Literal

from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


# xtuner/v1/config/fsdp.py
class FSDPConfig(BaseModel):
    ep_size: int = 1
    moe_fsdp_mesh: Literal["coupled", "decoupled"] = "coupled"

    def model_post_init(self, context) -> None:
        if self.moe_fsdp_mesh == "decoupled":
            assert self.ep_size > 1
            assert self.hsdp_sharding_size is None
            # 第一版还需在训练入口校验 TP/SP/PP、MTP、FP8、compile 和 offload 均关闭。


# xtuner/v1/model/moe/parallel_mesh.py
@dataclass(frozen=True)
class MoEParallelMeshes:
    root: DeviceMesh
    ep: DeviceMesh
    expert_fsdp: DeviceMesh


def build_moe_meshes_once(device: str, world_size: int, ep_size: int, prefix: str) -> MoEParallelMeshes:
    # root 只创建一次，所有子 mesh 都从这个 root 派生。
    root = init_device_mesh(
        device,
        (world_size // ep_size, ep_size),
        mesh_dim_names=(f"{prefix}.expert_fsdp", f"{prefix}.ep"),
    )
    ep = root[f"{prefix}.ep"]
    expert_fsdp = root[f"{prefix}.expert_fsdp"]
    return MoEParallelMeshes(root, ep, expert_fsdp)


def get_dense_fsdp_mesh(meshes: MoEParallelMeshes, mode: str, prefix: str) -> DeviceMesh:
    if mode == "coupled":
        return meshes.expert_fsdp

    # 仅 decoupled 延迟创建 full-rank group，避免默认路径多创建进程组。
    return flatten_same_root(
        meshes.root,
        dims=(f"{prefix}.expert_fsdp", f"{prefix}.ep"),
        name=f"{prefix}.dense_fsdp",
    )


# xtuner/v1/model/base.py
def _fully_shard(
    self,
    *,
    module: nn.Module,
    mesh: DeviceMesh,
    extra_ignored_params: set[nn.Parameter] | None = None,
    **existing_options,
) -> None:
    ignored = collect_existing_fp32_ignored_params(
        module,
        skip_nested_fsdp_children=True,
    )
    ignored.update(extra_ignored_params or set())
    fully_shard(module, mesh=mesh, ignored_params=ignored or None, **existing_options)


# xtuner/v1/model/moe/moe.py
class MoE(BaseModel):
    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.parallel_meshes = build_moe_meshes_once(
            DEVICE,
            dist.get_world_size(),
            config.ep_size,
            config.mesh_prefix,
        )
        self.ep_mesh = self.parallel_meshes.ep

        # GroupedLinear 继续使用 ep_mesh 沿 expert 维做 Shard(0)。
        self.layers = self.build_layers(config)

        # 模块引用是 ownership 的唯一事实来源；不能在这里永久缓存参数 id。
        self.routed_expert_modules = [
            layer.experts
            for layer in self.layers.values()
            if isinstance(layer, MoEDecoderLayer)
        ]

        # 保持当前时机：load spec 在 FSDP 前只看到 EP placement。
        self._init_load_spec()

    def fully_shard(self, fsdp_config: FSDPConfig):
        meshes = self.parallel_meshes
        dense_fsdp_mesh = get_dense_fsdp_mesh(
            meshes,
            fsdp_config.moe_fsdp_mesh,
            self.config.mesh_prefix,
        )
        self.dense_fsdp_mesh = dense_fsdp_mesh

        if fsdp_config.moe_fsdp_mesh == "coupled":
            # 复用当前已验证的参数 placement 和整层 FSDP 控制流。
            # 虽然 root mesh 创建代码被收口，但默认拓扑和数值语义必须不变。
            return self._fully_shard_coupled_compatible(fsdp_config, meshes)

        self.fsdp_mesh = dense_fsdp_mesh

        promote_trainable_params_to_fp32_as_today()

        # 关键：解耦模式不再调用 _replicate_other_params(self)。
        for layer_idx, raw_layer in self.layers.items():
            expert_params: set[nn.Parameter] = set()

            if isinstance(raw_layer, MoEDecoderLayer):
                expert_params = set(raw_layer.experts.parameters())
                if meshes.expert_fsdp.size() > 1:
                    # 第一版保留 XTuner 当前 FSDP 默认 Shard(0)，不复制 AutoModel 的 Shard(1)。
                    self._fully_shard(
                        module=raw_layer.experts,
                        mesh=meshes.expert_fsdp,
                        mp_policy=self.mp_policy,
                        reshard_after_forward=expert_reshard_after_forward(layer_idx),
                        offload_policy=None,
                    )

            layer = maybe_checkpoint_wrap(raw_layer, layer_idx)
            self.layers[str(layer_idx)] = layer

            # 外层只管理 dense 参数；routed expert 已归内层 FSDP 管理。
            self._fully_shard(
                module=layer,
                mesh=dense_fsdp_mesh,
                extra_ignored_params=expert_params,
                mp_policy=self.mp_policy,
                reshard_after_forward=layer_reshard_after_forward(layer_idx),
                offload_policy=None,
            )

        # embedding、norm、lm_head 都使用 dense mesh。
        shard_embedding_norm_lm_head(dense_fsdp_mesh)

        # 根模块继续使用 dense mesh。FSDP2 会排除已经归属于子 FSDP unit 的参数。
        self._fully_shard(
            module=self,
            mesh=dense_fsdp_mesh,
            mp_policy=self.mp_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=None,
        )

        preserve_current_layer_prefetch_and_materialization_order()

        # FSDP 与 _to_empty_meta() 都完成后，再按最终参数对象重建两种查询索引。
        self.refresh_runtime_param_ownership()
        return self

    def refresh_runtime_param_ownership(self) -> None:
        expert_ids = {
            id(param)
            for module in self.routed_expert_modules
            for param in module.parameters()
        }
        self.routed_expert_param_ids = expert_ids
        self.param_role_by_canonical_name = {
            clean_wrapper_name(name): (
                "routed_expert" if id(param) in expert_ids else "dense"
            )
            for name, param in self.named_parameters(remove_duplicate=False)
        }

    @torch.no_grad()
    def scale_and_reduce_grad(self) -> None:
        for name, param in self.trainable_parameters():
            if param.grad is None:
                continue
            if id(param) in self.routed_expert_param_ids:
                param.grad.div_(self.ep_mesh.size())
                continue
            reduce_only_over_actual_replicate_placements(param)


# xtuner/v1/model/base.py：HF 加载与导出
def runtime_fsdp_mesh(self, param_name: str) -> DeviceMesh:
    canonical_name = clean_wrapper_name(param_name)
    # 非参数 buffer 默认属于 dense 路径；只有 experts 模块中的参数走 expert-FSDP。
    if self.param_role_by_canonical_name.get(canonical_name, "dense") == "routed_expert":
        return self.parallel_meshes.expert_fsdp
    return self.dense_fsdp_mesh


def grouped_hf_params(self):
    # LoadEnum 不足以区分两种 FSDP group，必须再按运行时 mesh 分桶。
    buckets = {}
    for param_name, tensor in self.state_dict().items():
        canonical_name = clean_wrapper_name(param_name)
        load_spec = self.load_spec_mapping[canonical_name]
        fsdp_mesh = runtime_fsdp_mesh(self, canonical_name)
        buckets.setdefault((load_spec.load_enum, fsdp_mesh), []).append(
            (canonical_name, tensor, load_spec)
        )
    return buckets


def load_or_export_hf(self):
    for (_, fsdp_mesh), bucket in grouped_hf_params(self):
        load_or_all_gather_with_existing_hf_conversion(bucket, fsdp_mesh=fsdp_mesh)


# xtuner/v1/engine/train_engine.py：DCP 格式先不改
def save_or_resume_dcp(self):
    state = get_model_and_optimizer_state_dict_with_official_pytorch_api()
    existing_dcp_save_or_load(state)
```

## 6. 真实文件改动边界

| 文件 | 第一阶段最小职责 |
|---|---|
| `xtuner/v1/config/fsdp.py` | 增加模式开关与不兼容配置校验 |
| `xtuner/v1/model/moe/parallel_mesh.py` | 创建一次 root，并派生 EP、expert-FSDP、dense-FSDP |
| `xtuner/v1/model/moe/moe.py` | 保留 coupled 路径；新增 decoupled nested FSDP 与 ownership |
| `xtuner/v1/model/base.py` | 支持额外 ignored params；HF load/save 按实际 mesh 分桶 |
| `xtuner/v1/engine/train_engine.py` | 初期不改格式，只补充 DCP 恢复断言或测试 |
| `tests/model/test_moe_decoupled_fsdp.py` | topology、placement、梯度、HF round-trip |
| `tests/engine/test_glm52_moe_train_engine.py` | DCP、优化器连续性和 GLM-5.2 数值测试 |

第一阶段不修改 dispatcher、DeepEP、DSA、IndexShare、Triton/CUTLASS grouped GEMM 的实现。

## 7. 兼容性边界

| 能力 | 第一版状态 | 原因或验收方式 |
|---|---|---|
| 默认 coupled 模式 | 必须兼容 | 现有 recipe 不改配置即可保持旧行为 |
| 6 层 GLM-5.2 NoMTP，EP2/EP4 | 必须支持 | 核心开发与数值目标 |
| BF16 计算、FP32 master/AdamW | 必须支持 | 保持当前参数提升和 mixed precision 顺序 |
| activation checkpoint | 必须支持 | 需要验证 inner/outer FSDP 与 replay 顺序 |
| `reshard_after_forward` | 必须支持 | inner expert 与 outer layer 分别控制 |
| DeepEP / all2all | 第二阶段回归 | 不改变参数 ownership，但要验证 collective 顺序 |
| CUTLASS / Triton expert GEMM | 第二阶段回归 | 算子选择与 mesh 正交，仍需实测显存和数值 |
| 同拓扑 DCP save/resume | 必须支持 | DCP 恢复 model/optimizer；Trainer 恢复 scheduler/step |
| HF load/export/reload | 必须支持 | 两种 FSDP mesh 下的关键正确性闭环 |
| 异步 HF export | 暂缓 | 复用相同 gather 生成器，先保证同步路径正确 |
| coupled/decoupled 跨模式 DCP | 暂不承诺 | 先使用 consolidated HF 作为转换格式 |
| CPU offload / optimizer swap | 暂缓 | nested FSDP 增加 stream、生命周期和状态迁移风险 |
| compile | 暂缓 | `_orig_mod`、checkpoint wrapper 和 FSDP hook 组合需单测 |
| FP8 | 暂缓 | padding 后会重建只允许一个 Shard 的 load spec |
| MTP | 暂缓 | 共享权重、reentrant checkpoint 与多次调用更复杂 |
| HSDP | 暂缓 | XTuner 当前已经限制 HSDP 与 EP 不能同时开启 |
| SP / TP / PP | 暂缓 | XTuner data mesh 与 model mesh 目前相互独立，不能直接套用 AutoModel 5D mesh |
| 完整 GLM-5.2 / 多机 | 最终必须验证 | 6 层只能证明链路，不能代表完整模型的通信收益 |

## 8. 风险与容易出错的地方

1. **HF checkpoint 最容易静默出错**：用错 FSDP group 可能不立即报错，却会加载错误 slice 或导出重复权重。
2. **一个参数只能归一个 FSDP unit**：outer block 必须排除 routed expert；根模块依赖 FSDP2 的 nested ownership。
3. **不能顺手改变 expert shard dim**：否则无法判断问题来自 mesh 解耦还是权重布局变化。
4. **梯度不能重复平均**：dense 已由全 rank FSDP reduce-scatter 后，不能再按 EP 做一次手工 all-reduce。
5. **包装顺序必须所有 rank 一致**：有无 MoE 的 layer 也要按确定顺序创建 FSDP unit，避免 collective 次序不一致。
6. **完整模型不一定一定更快**：dense 显存会下降，但 dense all-gather/reduce-scatter 的 group 从 `world/EP`
   扩大到全 rank；多机时可能增加跨节点通信。
7. **全量 flatten 只适合第一版约束**：引入 TP/SP/PP 后，dense mesh 不能简单 flatten 所有 rank，需要像
   AutoModel 一样统一设计完整多维 root mesh。

## 9. 验收标准

### 9.1 结构断言

- EP2：`dense_fsdp=8`、`expert_fsdp=4`、`ep=2`；
- EP4：`dense_fsdp=8`、`expert_fsdp=2`、`ep=4`；
- attention、indexer、router、shared expert、norm、embedding、LM head 只归 dense FSDP；
- `MoEDecoderLayer.experts` 只归 expert-FSDP，并保留 EP placement；
- 每个 trainable parameter 恰好归属于一个 FSDP unit。

### 9.2 数值与 checkpoint

- coupled 默认路径的参数 shape、placement、loss 和 grad 行为不回归；
- 同一 batch 下，coupled/decoupled 一步 loss、grad、参数 delta 和 AdamW 状态在容差内一致；
- 20 step 的 loss、grad_norm 不出现系统性比例差异；
- decoupled EP2/EP4 均完成严格 HF load -> train -> export -> reload；
- 导出的 `model.safetensors.index.json` 不缺 key、不重复 key，所有 shard 文件名唯一；
- 同拓扑 DCP 恢复 model/optimizer，Trainer 恢复 scheduler/step，并与未中断 run 的下一步一致。

### 9.3 性能与规模

- memory snapshot 能看到 dense 参数和 optimizer state 的单卡 active memory 下降；
- 分别记录 steady-state step time、通信占比和 max allocated memory，不能只看 reserved memory；
- 6 层通过后，再做完整 GLM-5.2 NoMTP 的 load、2 step、HF export 和 DCP resume；
- 多机验证后才能判断更大的 dense collective 是否值得。

## 10. 建议提交顺序

1. `feat(fsdp): add configurable MoE mesh views`
   只增加单 root 派生视图、配置开关和 topology 测试，不改变参数布局。
2. `feat(fsdp): add decoupled nested FSDP for MoE experts`
   新增 decoupled eager 路径、ownership、gradient 和 coupled 回归测试。
3. `fix(checkpoint): make MoE HF conversion mesh-aware`
   按 `(LoadEnum, fsdp_mesh)` 分桶，完成 EP2/EP4 严格 round-trip。
4. `test(checkpoint): validate decoupled MoE DCP resume`
   验证 DCP 的 model/AdamW，以及 Trainer scheduler/step 和下一步连续性。
5. `test(glm52): validate full-model decoupled training`
   完整 NoMTP 模型的显存、通信、保存和恢复报告。
6. 后续独立提交 compile、FP8、MTP、offload、SP/TP/PP，避免一次 PR 难以定位回归。

## 11. 源码依据

XTuner：

- `xtuner/v1/model/moe/moe.py::MoE.__init__`
- `xtuner/v1/model/moe/moe.py::MoE._init_device_mesh`
- `xtuner/v1/model/moe/moe.py::MoE._replicate_other_params`
- `xtuner/v1/model/moe/moe.py::MoE.fully_shard`
- `xtuner/v1/model/moe/moe.py::MoE.scale_and_reduce_grad`
- `xtuner/v1/module/grouped_linear/moe_group_linear.py::GroupedLinear`
- `xtuner/v1/module/decoder_layer/moe_decoder_layer.py::MoEDecoderLayer`
- `xtuner/v1/model/base.py::BaseModel._fully_shard`
- `xtuner/v1/model/base.py::BaseModel._init_load_spec`
- `xtuner/v1/model/base.py::BaseModel._fsdp_foreach_allgather`
- `xtuner/v1/engine/train_engine.py`

AutoModel：

- `nemo_automodel/components/distributed/mesh_utils.py::_create_fsdp2_device_mesh`
- `nemo_automodel/components/distributed/mesh_utils.py::_create_moe_mesh`
- `nemo_automodel/components/moe/parallelizer.py::ExpertParallel`
- `nemo_automodel/components/moe/parallelizer.py::_moe_shard_placement`
- `nemo_automodel/components/moe/parallelizer.py::apply_ep`
- `nemo_automodel/components/moe/parallelizer.py::apply_fsdp`

PyTorch 官方资料：

- [FSDP2 `fully_shard` API](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)

相关调研：

- `Znote/glm/glm52_xtuner_decoupled_fsdp_developer_design.md`
- `Znote/glm/glm52_full_automodel_implementation_and_xtuner_integration.md`