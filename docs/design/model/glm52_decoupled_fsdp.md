# GLM-5.2：XTuner 解耦 FSDP2 Mesh 设计

> 状态：设计评审稿，尚未实现。
>
> 对应伪代码：`glm52_decoupled_fsdp.py`。
>
> 核对基线：XTuner `d66cf9e9`；AutoModel `e2c48886`。

## 1. 设计结论

XTuner 可以基于现有 PyTorch FSDP2、`DeviceMesh` 和 `DTensor` 实现与 AutoModel 类似的 MoE 解耦分片：

```text
dense 参数          -> 主 dense-FSDP mesh
routed expert 参数  -> EP + expert-FSDP mesh
```

这项改动主要解决 EP 增大后 dense 参数、dense 梯度和 AdamW 状态在单卡上切分不足的问题。它不改变模型
公式、router 结果、dispatcher、expert GEMM 或 loss 定义。

第一版必须遵守四条原则：

1. 增加显式配置开关，默认继续走当前 `coupled` 路径。
2. EP、expert-FSDP、dense-FSDP 必须从同一个 root mesh 派生。
3. 第一版只改 mesh 和 FSDP ownership，不改 XTuner 当前 expert `Shard(0)` 布局。
4. HF 加载与导出必须改成 mesh-aware；只改训练 wrap 无法形成完整闭环。

## 2. 为什么要改

当前 XTuner 把 8 卡排成 `(world_size / EP, EP)`：

```text
EP=4：root shape = (2, 4)

                EP0  EP1  EP2  EP3
FSDP 第 0 行      0    1    2    3
FSDP 第 1 行      4    5    6    7
```

当前 dense 参数只沿纵向 2 卡 FSDP，横向 4 个 EP 位置保存相同副本：

```text
rank 0/1/2/3：dense 前半部分的 4 份副本
rank 4/5/6/7：dense 后半部分的 4 份副本
```

因此，8 卡 EP4 时 dense 只切成 2 份。解耦后的目标是：

```text
dense-FSDP：      [0,1,2,3,4,5,6,7]，切成 8 份
EP groups：       [0,1,2,3] / [4,5,6,7]
expert-FSDP：     [0,4] / [1,5] / [2,6] / [3,7]
```

假设 dense 参数共 80 GB，只看参数本身：

| 模式 | dense 分片份数 | 单卡 dense 参数 |
|---|---:|---:|
| 当前 XTuner EP4 | 2 | 约 40 GB |
| 解耦 XTuner EP4 | 8 | 约 10 GB |

实际训练显存还包含梯度、AdamW 状态、激活、通信 buffer 和 allocator reserved memory，上表不能直接预测最终
显存；它只说明解耦能降低哪一部分常驻显存。代价是 dense collective 从 2 卡扩大到 8 卡，通信时间可能上升。

## 3. 当前源码事实

### 3.1 XTuner 当前实现

当前 `MoE` 有以下行为：

1. `MoE.__init__()` 为 EP 创建一次 `(dp, ep)` root，但只保存 `ep` 子 mesh。
2. `MoE._init_device_mesh()` 在 FSDP 阶段再次创建 `(fsdp, ep)` root，并依赖 PyTorch mesh 的哈希与父子映射
   行为让新旧 `ep_mesh` 被识别为相关 mesh。
3. `GroupedLinear` 在构造时把 routed expert 权重沿 EP 维执行 `Shard(0)`。
4. `_replicate_other_params()` 把非 routed-expert 参数沿 EP 维执行 `Replicate()`。
5. decoder layer 整体只沿 `self.fsdp_mesh` 调用一次 `fully_shard()`，没有独立的 expert FSDP unit。
6. `scale_and_reduce_grad()` 对 expert 梯度除以 EP size，对带 `Replicate` placement 的 dense 梯度补做平均。

当前参数布局可以简化成：

```text
dense：  [Shard(fsdp), Replicate(ep)]
expert： [Shard(fsdp), Shard(ep)]
```

源码位置：

- `xtuner/v1/model/moe/moe.py::MoE.__init__`
- `xtuner/v1/model/moe/moe.py::MoE._init_device_mesh`
- `xtuner/v1/model/moe/moe.py::MoE._replicate_other_params`
- `xtuner/v1/model/moe/moe.py::MoE.fully_shard`
- `xtuner/v1/model/moe/moe.py::MoE.scale_and_reduce_grad`
- `xtuner/v1/module/grouped_linear/moe_group_linear.py::GroupedLinear`

### 3.2 AutoModel 当前实现

AutoModel 的 FSDP2 主 mesh 为：

```text
(pp, dp_replicate, dp_shard, cp, tp)
```

MoE mesh 从主 mesh 的非 PP 维度派生并重新解释为：

```text
(ep_shard, ep)
```

其执行顺序是：

1. `apply_ep()`：routed experts 沿 EP 执行 `Shard(0)`。
2. `apply_fsdp()` 内层：`fully_shard(moe.experts, mesh=ep_shard_mesh)`。
3. `apply_fsdp()` 外层：`fully_shard(block, mesh=fsdp_mesh, ignored_params=expert_params)`。

AutoModel 对二维及以上 expert 权重默认使用 `Shard(1)` 做 expert-FSDP；这是存储布局选择，不是解耦本身的
必要条件。

在本次单机 8 卡、TP=CP=PP=1、EP4 场景下：

```text
dense-FSDP size  = 8
expert-FSDP size = 8 / 4 = 2
EP size          = 4
```

“AutoModel dense 始终全 world 分片”不是通用结论。准确说法是：dense 使用主 mesh 的 DP/FSDP 轴；只有在
TP、CP、PP 均为 1 的当前场景下，该轴才覆盖 8 张卡。

源码位置：

- `nemo_automodel/components/distributed/mesh_utils.py::_create_fsdp2_device_mesh`
- `nemo_automodel/components/distributed/mesh_utils.py::_create_moe_mesh`
- `nemo_automodel/components/moe/parallelizer.py::ExpertParallel`
- `nemo_automodel/components/moe/parallelizer.py::_moe_shard_placement`
- `nemo_automodel/components/moe/parallelizer.py::apply_ep`
- `nemo_automodel/components/moe/parallelizer.py::apply_fsdp`

### 3.3 两边相同点和差异

| 项目 | AutoModel | XTuner 当前 | XTuner 目标 |
|---|---|---|---|
| 基础能力 | PyTorch FSDP2 / DeviceMesh / DTensor | 同左 | 继续复用 |
| EP placement | expert 沿 EP `Shard(0)` | 同左 | 保持不变 |
| expert-FSDP | 独立内层 FSDP unit | 整层 FSDP 顺带管理 | 增加独立内层 unit |
| dense-FSDP | 主 DP/FSDP mesh | `world / EP` 轴 | 当前约束下使用全 rank flat mesh |
| dense 沿 EP | 不复制 | `Replicate()` | 不再复制 |
| 外层 ownership | 显式忽略 experts | 无内外层区分 | 显式忽略 experts |
| expert FSDP shard dim | 二维权重通常 `Shard(1)` | 当前默认 `Shard(0)` | 第一版保持 `Shard(0)` |

## 4. 目标和非目标

### 4.1 第一阶段目标

- 单机 8 卡；
- GLM-5.2 6 层 NoMTP；
- TP=SP=PP=1；
- EP=2/4；
- BF16 前向/反向、FP32 master 参数和 AdamW 状态；
- activation checkpoint 与 `reshard_after_forward`；
- eager 模式；
- strict HF load、同步 HF export/reload、同拓扑 DCP save/resume；
- coupled 与 decoupled 的 loss、grad_norm、参数 delta 对齐。

### 4.2 第一阶段不做

- 不改变 dispatcher（all2all、DeepEP）；
- 不改变 expert GEMM（Triton、CUTLASS）；
- 不改变 router、DSA、Indexer、IndexShare；
- 不改变数据、packing、loss 或优化器公式；
- 不复制 AutoModel 的 expert `Shard(1)`；
- 不承诺 MTP、FP8、compile、CPU offload、optimizer swap、HSDP、TP/SP/PP；
- 不根据 6 层结果直接宣称完整 GLM-5.2 或多机一定更快。

## 5. 详细方案

### 5.1 配置开关

在 `FSDPConfig` 增加：

```python
moe_fsdp_mesh: Literal["coupled", "decoupled"] = "coupled"
```

- `coupled`：默认值，保持现有 placement、梯度和 checkpoint 行为。
- `decoupled`：启用 dense/expert 两套 FSDP mesh 和 nested FSDP2。

第一版进入 `decoupled` 时应主动拒绝尚未验证的组合，避免“看似能启动，保存或恢复时才出错”。校验应放在
能同时看到 model config、FSDP config 和 trainer config 的入口，而不是全部塞进 `FSDPConfig`。

### 5.2 root mesh 只创建一次

EP>1 时，在 `MoE.__init__()` 创建并保存一个 root：

```text
root shape = (world_size / ep_size, ep_size)
axis names = (prefix.fsdp, prefix.ep)
```

然后派生：

```text
ep_mesh          = root[prefix.ep]
expert_fsdp_mesh = root[prefix.fsdp]

coupled:
    dense_fsdp_mesh = expert_fsdp_mesh

decoupled:
    dense_fsdp_mesh = flatten(root[prefix.fsdp, prefix.ep])
```

保留 `prefix.fsdp` 这个轴名可以降低默认路径兼容风险；代码变量把它称为 `expert_fsdp_mesh`，只是说明它在
解耦模式下的职责。`dense_fsdp_mesh` 的 flat group 只在启用 decoupled 时创建。

不能重新调用 `init_device_mesh()` 创建一个独立 1D world mesh。rank 列表相同不代表 PyTorch 认为两套 mesh
具有相同 parent；expert 已经是 EP DTensor，后续 FSDP mesh 必须与 EP mesh 来自同一 root。

### 5.3 参数 ownership

| 参数或模块 | FSDP unit | EP placement |
|---|---|---|
| attention / DSA indexer | outer dense-FSDP | 无 |
| router / gate | outer dense-FSDP | 无 |
| shared experts / shared expert gate | outer dense-FSDP | 无 |
| norm / embedding / LM head | dense-FSDP | 无 |
| `MoEDecoderLayer.experts` | inner expert-FSDP | `Shard(0)` |

核心不变量：

```text
每个 trainable parameter 必须恰好由一个 FSDP unit 管理。
```

ownership 的事实来源必须是 `MoEDecoderLayer.experts` 模块，而不是在加载、保存、梯度代码中重复写多份
`".experts" in name`。建议维护两类索引：

- canonical parameter name -> `dense` 或 `routed_expert`，用于 HF load/save；
- FSDP 与 meta materialization 完成后的 parameter id 集合，用于训练期梯度路径。

canonical name 统一复用 `_clean_param_name()`，去掉 activation checkpoint 和 compile wrapper 引入的前缀。

### 5.4 nested FSDP 包装顺序

decoupled 路径按 decoder layer 自底向上处理：

```text
raw_layer.experts
    -> inner fully_shard(expert_fsdp_mesh)

raw_layer
    -> activation checkpoint wrapper（按现有比例）
    -> outer fully_shard(dense_fsdp_mesh, ignored_params=current_expert_params)

embedding / norm / lm_head / root
    -> dense_fsdp_mesh
```

注意事项：

1. `ignored_params` 要在 inner `fully_shard()` 后重新从 `raw_layer.experts.parameters()` 获取，不能永久缓存
   FSDP 前的 parameter id。
2. `BaseModel._fully_shard()` 需要接受额外 ignored params，并在扫描 FP32 特例时跳过已经成为
   `FSDPModule` 的子模块。
3. 保持现有 layer prefetch、最后一层 `reshard_after_forward=False` 和 meta materialization 时序。
4. 第一版拒绝 MTP，避免共享权重、reentrant checkpoint 和 nested FSDP 同时进入首个提交。

### 5.5 expert placement 暂不改变

XTuner 当前：

```text
EP：          Shard(0)
FSDP：        默认沿 FSDP_SHARD_DIM=0
HF fused load 假设 EP 与 FSDP 沿同一维继续切分
```

AutoModel 对二维 expert 权重的 expert-FSDP 通常使用 `Shard(1)`。第一版不能顺手照搬，否则同时改变了：

- mesh ownership；
- expert 本地 tensor 形状；
- HF fused expert 的 slice 算法；
- grouped GEMM 输入布局。

第一版只验证解耦 mesh，继续使用 XTuner 当前 `Shard(0)`。若后续要评估 `Shard(1)`，应单独提交并完成 HF
round-trip 与 Triton/CUTLASS 回归。

### 5.6 梯度、grad norm 和优化器

当前规则继续保留：

- routed expert 不跨 EP 做参数副本平均，只把梯度除以 `EP size`，维持当前全局 batch 平均语义；
- dense 参数在 coupled 模式有 EP `Replicate`，继续按实际副本 group 平均；
- dense 参数在 decoupled 模式沿 flat dense mesh 分片，不再有 EP `Replicate`，不能额外再做一次 EP all-reduce。

`cal_grad_norm()` 已按 `(device_mesh, placements)` 分桶，并沿所有 `Shard` placement 做规约；机制上能够处理
dense 与 expert 两种布局，但必须通过同 batch 测试证明没有漏算或重复规约。

AdamW 必须在 FSDP 完成后构建。验收时不仅比较 loss，还要比较：

- expert 与 dense 的局部 grad shape；
- global grad_norm；
- 一步参数 delta；
- AdamW `exp_avg`、`exp_avg_sq` 的 shape、dtype 和 placement。

### 5.7 HF 加载与导出

这是改动量和静默错误风险最大的部分。

当前 `_load_same_hf_param()`、`_load_fused_hf_param()`、`_load_shard_hf_param()` 和
`_fsdp_foreach_allgather()` 都直接使用 `self.fsdp_mesh`。解耦后：

```text
dense 参数  -> dense_fsdp_mesh
expert 参数 -> expert_fsdp_mesh
```

设计要求：

1. `_init_load_spec()` 继续在 FSDP 前执行。此时 expert 只有一个 EP `Shard`，符合当前“最多一个 Shard”限制。
2. 根据 canonical parameter name 查询参数角色，再选择 runtime FSDP mesh。
3. load helper 处理单个参数时显式拿到正确 mesh。
4. save 不能只按 `LoadEnum` 分桶，至少按 `(LoadEnum, parameter_role)` 分桶。
5. `_fsdp_foreach_allgather()` 显式接收 `fsdp_mesh`，不能读取全局 `self.fsdp_mesh`。
6. `LoadSpec.group` 仍保留原有 EP gather 语义，不能误替换成 FSDP group。
7. 同步和异步 HF save 共用的 chunk/gather 逻辑都要检查；第一版只承诺同步路径。
8. 导出后检查 index 中无缺 key、重复 key和文件名覆盖，再由独立 HF loader strict reload。

FP8 当前在 FSDP 前调用 `Float8Handler.pad_for_fsdp(self, self.fsdp_mesh, callback_after_pad=_init_load_spec)`。
该接口只接收一个全局 FSDP mesh；解耦后 dense 与 expert 的 padding 应分别按 dense-FSDP 和 expert-FSDP
计算，继续统一传 `self.fsdp_mesh` 会使用错误的分片规模。另一方面，如果后续选择在 nested FSDP 后重建
load spec，现有 `_init_load_spec()` 又无法表示 expert 的 EP + expert-FSDP 多个 `Shard` placement。因此 FP8
是第一版明确不兼容项，需要单独改造 padding 和 load-spec 表达，不是简单“还没测”。

### 5.8 DCP save/resume

XTuner 已使用 PyTorch 官方：

- `get_model_state_dict()` / `set_model_state_dict()`；
- `get_optimizer_state_dict()` / `set_optimizer_state_dict()`；
- `torch.distributed.checkpoint.save()` / `load()`。

第一版不设计新 DCP 格式，先验证同一 decoupled 拓扑下的保存恢复：

```text
decoupled EP4 保存 -> decoupled EP4 恢复 -> 下一步与未中断 run 对齐
```

以下暂不承诺：

```text
coupled -> decoupled
decoupled EP2 -> decoupled EP4
```

跨模式或跨 EP 恢复优先通过 consolidated HF 权重中转，不能默认 DCP 会自动完成所有重分片。

## 6. 兼容性矩阵

| 能力 | 第一版结论 | 说明 |
|---|---|---|
| 默认 coupled | 必须兼容 | 不改 recipe 即保持现有行为 |
| GLM-5.2 6 层 NoMTP EP2/EP4 | 必须支持 | 首个数值与显存闭环 |
| BF16 compute + FP32 master/AdamW | 必须支持 | 当前 parity 基线 |
| activation checkpoint | 必须支持 | 验证 inner/outer hook 与 replay |
| `reshard_after_forward` | 必须支持 | inner 和 outer 分别配置 |
| strict HF load | 必须支持 | 训练前置条件 |
| 同步 HF export/reload | 必须支持 | checkpoint 正确性闭环 |
| 同拓扑 DCP resume | 必须支持 | model、optimizer、scheduler、step 连续 |
| DeepEP / all2all | 原理上兼容，第二阶段验证 | dispatcher 不管理参数 ownership |
| Triton / CUTLASS expert GEMM | 原理上兼容，第二阶段验证 | 计算 kernel 与 mesh 设计正交 |
| 完整 GLM-5.2 NoMTP | 最终必须验证 | 6 层只证明链路 |
| SP | 暂缓验证 | XTuner SP 使用独立 data mesh，要核对 collective 顺序 |
| compile | 暂缓验证 | wrapper 名称、graph capture、FSDP hook 组合复杂 |
| CPU offload / optimizer swap | 暂缓验证 | 参数与状态生命周期发生变化 |
| 异步 HF export / RL weight update | 暂缓验证 | 共享 mesh-aware gather 逻辑，涉及后台进程/通信 |
| MTP | 第一版拒绝 | 共享权重、reentrant checkpoint、多次调用 |
| FP8 | 当前实现不兼容 | padding 只接收单一 FSDP mesh，load spec 也不能表达多个 Shard |
| HSDP + EP | 当前实现不兼容 | `FSDPConfig` 已明确限制 EP=1 |
| MoE TP | 当前实现不支持 | 不是本 PR 顺带解决的能力 |
| coupled/decoupled 跨模式 DCP | 暂不支持 | 先使用 HF 中转 |

## 7. 文件改动边界

| 文件 | 计划改动 |
|---|---|
| `xtuner/v1/config/fsdp.py` | 增加 `moe_fsdp_mesh` 配置 |
| `xtuner/v1/model/moe/moe.py` | 单 root mesh、coupled/decoupled 分支、nested FSDP、ownership、梯度 |
| `xtuner/v1/model/base.py` | 合并 ignored params；HF load/save 按参数角色选择 mesh |
| `xtuner/v1/engine/train_engine.py` | 原则上不改格式，只补 DCP 验证和必要断言 |
| `tests/model/test_moe_decoupled_fsdp.py` | topology、placement、ownership、grad、HF round-trip |
| `tests/engine/test_moe_decoupled_checkpoint.py` | DCP model/AdamW/scheduler/step 连续性 |
| GLM-5.2 recipe | 增加 decoupled 实验配置，不替换默认 recipe |

第一阶段不修改 dispatcher、DSA、Indexer、router 或 grouped GEMM 实现。

## 8. 风险

1. **HF checkpoint 静默错误**：mesh 用错可能不立刻报错，却会加载错误 slice 或导出重复权重。
2. **重复 ownership**：expert 同时被 inner 与 outer FSDP 管理会导致重复 hook 或 collective 错序。
3. **旧 parameter id 失效**：FSDP 与 `to_empty()` 可能替换参数对象，运行期索引必须最后重建。
4. **dense 重复规约**：decoupled dense 已在 8 卡 FSDP reduce-scatter，不能再按 EP 平均。
5. **通信变重**：EP4 时 dense collective 从 2 卡扩大到 8 卡，显存下降不代表 step time 一定更快。
6. **完整模型外推风险**：6 层模型的参数比例、通信占比和完整 GLM-5.2 不相同。
7. **多维并行扩展**：未来加入 TP/SP/PP 后，dense mesh 不能简单 flatten 所有 rank，需要统一 root 设计。

## 9. 验收标准

### 9.1 静态 topology 与 ownership

- EP2：`dense_fsdp=8`、`expert_fsdp=4`、`ep=2`；
- EP4：`dense_fsdp=8`、`expert_fsdp=2`、`ep=4`；
- coupled 的 mesh 名、placement 和 local shape 不回归；
- attention、indexer、router、shared expert、norm、embedding、LM head 只归 dense unit；
- `MoEDecoderLayer.experts` 只归 expert unit；
- 每个 trainable parameter 恰好归一个 FSDP unit。

### 9.2 数值

- 固定 seed、固定 pack、固定 batch；
- coupled/decoupled 第 1 步 loss、grad_norm、参数 delta 在约定容差内；
- 20 步 loss 与 grad_norm 无系统性比例偏差；
- EP2 与 EP4 的有效 global batch、token 数、学习率和 optimizer 设置一致；
- expert 梯度仍只做 `1 / EP` 缩放，dense 不重复 all-reduce。

### 9.3 checkpoint

- decoupled EP2/EP4 strict HF load；
- train -> sync HF export -> 独立 loader strict reload；
- `model.safetensors.index.json` 无缺 key、重复 key和覆盖文件；
- 同拓扑 DCP 恢复 model、AdamW、scheduler、train step；
- resume 后下一步与未中断 run 对齐。

### 9.4 显存和性能

- PyTorch memory snapshot 能看到 dense 参数和 optimizer state 的 active memory 下降；
- 同时记录 max allocated、reserved、steady-state step time 和 collective 时间；
- 先验证 6 层 EP2/EP4，再验证完整 NoMTP 模型 load、2 step、HF export、DCP resume；
- 多机结果出来前，不宣称该设计一定提升吞吐。

## 10. 建议提交顺序

1. `refactor(moe): build EP and FSDP views from one root mesh`
   只收口 root 创建，保持 coupled placement 与数值行为。
2. `feat(fsdp): add decoupled nested FSDP for MoE experts`
   增加开关、inner/outer ownership、梯度和基础测试。
3. `fix(checkpoint): make MoE HF conversion mesh-aware`
   完成 strict load、同步 export 和 EP2/EP4 round-trip。
4. `test(checkpoint): validate decoupled DCP resume`
   验证 model、AdamW、scheduler、step 和下一步连续性。
5. `test(glm52): validate full NoMTP model`
   完整模型显存、时间、HF 与 DCP 报告。
6. 后续分别补 DeepEP/CUTLASS、SP、compile、offload、FP8、MTP，避免一个 PR 同时改变过多变量。

## 11. 评审时需要确认的问题

1. 第一版开关名是否采用 `moe_fsdp_mesh`，还是项目希望统一为更通用的 parallel strategy 配置。
2. 第一版是否只承诺 AdamW；Muon 对 mesh 维名称有字符串匹配逻辑，需要单独回归。
3. 同步 HF export 是否必须和首个训练 PR 同时落地。本文建议必须，否则无法验证权重闭环。
4. 完整模型验证使用 EP2 还是 EP4 作为第一条生产链路。EP4 dense 通信增量更大，更能暴露问题。

## 12. 伪代码说明

`glm52_decoupled_fsdp.py` 按真实改动文件分段，展示：

- 配置与兼容性校验；
- 单 root mesh 及三种 mesh view；
- 参数角色索引；
- coupled 保持逻辑和 decoupled nested FSDP 主流程；
- 梯度处理；
- mesh-aware HF load/save；
- DCP 与测试骨架。

它是设计伪代码，不可直接复制运行；真实实现必须拆成上述提交并逐步通过测试。
