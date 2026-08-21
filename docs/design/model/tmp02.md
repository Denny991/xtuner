# GLM-5.2 解耦 FSDP2 学习讲解

> 这是一份面向初学者的辅助材料，用来解释
> `glm52_decoupled_fsdp.md` 中的概念、拓扑和伪代码。
>
> 原设计文档用于开发评审，本文件用于学习理解。这里出现的“目标实现”和伪代码尚未在 XTuner 中落地，
> 不能当作当前代码已经支持的功能。

## 1. 先用一句话理解这项改动

GLM-5.2 同时包含两类参数：

- **dense 参数**：attention、indexer、router、shared expert、norm、embedding、LM head 等；
- **routed expert 参数**：token 经 router 选择后才执行的专家，也就是 `MoEDecoderLayer.experts`。

XTuner 当前让这两类参数共用同一个较小的 FSDP 组。EP 越大，留给 FSDP 的维度越小，dense 参数在单卡上
切得越少、复制得越多。解耦方案想做的是：

```text
dense 参数          -> 始终使用全 8 卡 FSDP
routed expert 参数  -> 继续使用 EP + expert-FSDP
```

它主要解决的是**参数、梯度和优化器状态的单卡显存**，并不直接改变模型公式、router 结果或 loss 定义。

## 2. 阅读设计稿前必须认识的词

### 2.1 rank、world size 和 process group

单机 8 卡训练通常启动 8 个进程：

```text
GPU 0 对应 rank 0
GPU 1 对应 rank 1
...
GPU 7 对应 rank 7
world_size = 8
```

一个 process group 是“哪些 rank 要一起通信”的名单。例如 `[0, 4]` 表示 rank 0 和 rank 4 会在这个组内执行
all-gather、reduce-scatter 等 collective。

### 2.2 DeviceMesh 是什么

`DeviceMesh` 可以理解为“给 GPU 排座位，并给每个方向起名字”。例如：

```text
root mesh shape = (2, 4)
axis 0 = expert_fsdp
axis 1 = ep

                    ep0  ep1  ep2  ep3
expert_fsdp 第 0 行    0    1    2    3
expert_fsdp 第 1 行    4    5    6    7
```

从这张二维座位表可以得到不同方向的通信组：

```text
横着看 EP：          [0,1,2,3]、[4,5,6,7]
竖着看 expert-FSDP： [0,4]、[1,5]、[2,6]、[3,7]
全部拉平 dense-FSDP： [0,1,2,3,4,5,6,7]
```

mesh 本身不执行训练，它只是告诉 DTensor 和 FSDP：“这个张量沿哪个方向切，在哪些 rank 之间通信”。

### 2.3 DTensor、Shard 和 Replicate

DTensor 是带有分布式布局信息的 Tensor。最常见的 placement 有：

| placement | 含义 | 8 GB 参数在 4 卡上的直观例子 |
|---|---|---|
| `Shard(0)` | 沿张量第 0 维切开 | 每卡保存不同的约 2 GB |
| `Shard(1)` | 沿张量第 1 维切开 | 每卡保存不同的约 2 GB |
| `Replicate()` | 每卡保存完整副本 | 每卡都保存 8 GB |

`Shard(0)` 中的 `0` 是**张量维度**，不是 GPU 0，也不是 mesh 的第 0 个方向。

### 2.4 EP 是什么

EP 是 Expert Parallel。它把 routed experts 分给多个 EP 位置。

假设模型有 64 个 routed experts，`EP=4`，为了便于理解先忽略负载不均和底层 flatten 布局：

```text
EP0 负责专家 0~15
EP1 负责专家 16~31
EP2 负责专家 32~47
EP3 负责专家 48~63
```

所以 `EP=4` 的意思不是“模型只有 4 个专家”，而是“全部专家被分到 4 个 EP 位置”。token 被 router 选中后，
dispatcher 会把它发送到持有所需专家的 EP rank。

### 2.5 FSDP2 是什么

FSDP2 会把一个模块的参数、梯度和优化器状态分片保存。以一个被两卡 FSDP 管理的层为例：

```text
平时：rank 0 保存参数前一半，rank 1 保存参数后一半
前向前：两卡 all-gather，临时恢复计算所需的完整参数
前向后：若 reshard_after_forward=true，再拆回两半
反向：需要时再次 all-gather，计算梯度后 reduce-scatter
更新：每卡只更新自己持有的参数和 AdamW 状态分片
```

FSDP 节省常驻显存，但会增加参数 all-gather 和梯度 reduce-scatter 通信。

### 2.6 nested FSDP2 是什么

“nested”就是一个大模块内部还有一个独立的 FSDP unit：

```text
decoder layer（外层 FSDP，管理 dense 参数）
├── attention
├── router
├── shared experts
└── routed experts（内层 FSDP，管理 routed expert 参数）
```

一个参数只能由一个 FSDP unit 管理。外层如果再次管理 routed expert，就会发生重复分片、重复 hook 或
collective 顺序错误。因此外层必须忽略已经交给内层的 expert 参数。

### 2.7 “dense”在本文里是什么意思

这里的 dense 是**并行归属分类**，不等于“这个模块一定是普通全连接层”。例如 router 和 shared expert 在模型
结构上仍属于 MoE 层，但在本设计中不沿 EP 分配，因此归到 dense-FSDP。

## 3. 固定一个例子：8 卡、EP=4

后文始终使用这张表：

```text
                    EP0  EP1  EP2  EP3
expert-FSDP 第 0 份   0    1    2    3
expert-FSDP 第 1 份   4    5    6    7
```

这里满足：

```text
expert_fsdp_size = world_size / ep_size = 8 / 4 = 2
expert_fsdp_size * ep_size = 2 * 4 = 8
```

这个乘法只适用于当前 `TP=SP=PP=1` 的简化场景。未来加入 TP、SP、PP 或多维数据并行后，不能继续只用
`FSDP * EP = world_size` 推导完整拓扑。

### 3.1 EP 组在做什么

横向的两个组是：

```text
[0,1,2,3]
[4,5,6,7]
```

每一行包含 4 个 EP 位置。router 得到 top-k expert id 后，dispatcher 在这一行内把 token 发到对应 EP 位置。

### 3.2 expert-FSDP 组在做什么

纵向的四个组是：

```text
[0,4]、[1,5]、[2,6]、[3,7]
```

以 `[0,4]` 为例，两张卡属于同一个 EP 列，负责 EP0 对应的 local expert tensor；这个 tensor 又被 FSDP
分成两份保存，计算前再在 `[0,4]` 内恢复。

这里不要急着理解为“每个专家矩阵都拦腰切成两半”。物理分法取决于 `Shard(dim)` 和权重布局：

- AutoModel 对二维 expert 权重的 expert-FSDP 使用 `Shard(1)`，更接近把每个本地专家矩阵沿特征维切开；
- XTuner 当前 expert 权重和 FSDP 默认主要沿 `Shard(0)`，可能是把 flatten 后的 local expert tensor 行继续
  切分，物理上可能表现为不同 expert/output 行由不同 rank 保存；
- 无论存储维怎么切，FSDP 都会在 expert 计算前恢复该 FSDP unit 所需的完整参数。

第一版设计刻意保留 XTuner 的 `Shard(0)`，只验证 mesh 解耦，避免同时修改权重布局和 HF 转换。

## 4. 当前 XTuner 为什么叫“耦合”

当前 XTuner 使用同一个 `(fsdp=2, ep=4)` root mesh：

```text
dense 参数：  沿 fsdp 轴 Shard，沿 ep 轴 Replicate
expert 参数：沿 ep 轴 Shard，同时被 fsdp 轴继续管理
```

### 4.1 dense 参数在 8 张卡上怎么放

假设全部 dense 参数为 80 GB，只看参数本身：

```text
rank 0：dense 前 40 GB
rank 1：dense 前 40 GB，和 rank 0 是 EP 副本
rank 2：dense 前 40 GB，和 rank 0 是 EP 副本
rank 3：dense 前 40 GB，和 rank 0 是 EP 副本

rank 4：dense 后 40 GB
rank 5：dense 后 40 GB，和 rank 4 是 EP 副本
rank 6：dense 后 40 GB，和 rank 4 是 EP 副本
rank 7：dense 后 40 GB，和 rank 4 是 EP 副本
```

虽然有 8 张卡，dense 实际只切成 2 份；每一份又沿 EP 复制 4 次。因此每卡参数约 40 GB。

这就是“FSDP 与 EP 耦合”：`EP=4` 占用了 mesh 的一个维度，dense FSDP 只剩 `8/4=2`。

### 4.2 为什么 EP 变大可能让显存反而升高

在 8 卡下：

| EP | 当前 XTuner dense-FSDP | dense 每卡参数比例 |
|---:|---:|---:|
| 2 | 4 | `1/4` |
| 4 | 2 | `1/2` |
| 8 | 1 | `1` |

EP 变大让每卡持有的 routed experts 变少，但也让 dense 参数、dense 梯度和 AdamW 状态切得更浅。最终显存是
两部分此消彼长，不能只用“EP 越大越省专家显存”判断。

## 5. 解耦后的目标布局

解耦后仍保留相同的 expert 拓扑：

```text
EP groups:          [0,1,2,3] / [4,5,6,7]
expert-FSDP groups: [0,4] / [1,5] / [2,6] / [3,7]
```

只把 dense-FSDP 改成全 8 卡：

```text
dense-FSDP group: [0,1,2,3,4,5,6,7]
```

同样用 80 GB dense 参数举例：

```text
rank 0：第 1 个 10 GB shard
rank 1：第 2 个 10 GB shard
...
rank 7：第 8 个 10 GB shard
```

每卡 dense 参数从约 40 GB 降到约 10 GB。梯度和 AdamW 状态通常也跟随这种分片，因此收益不只参数本身。

代价是 dense 层的通信组从 2 卡扩大到 8 卡。单机 NVLink 内可能值得，多机跨节点时则要重新测通信开销。

## 6. 为什么三种 mesh 必须来自同一个 root

设计稿中的代码是：

```python
root = (expert_fsdp, ep)
ep_mesh = root[ep]
expert_fsdp_mesh = root[expert_fsdp]
dense_fsdp_mesh = flatten(root[expert_fsdp, ep])
```

即使两个独立创建的 mesh 恰好包含同样的 rank，PyTorch 也需要知道它们的父子关系，才能把已有 EP DTensor
placement 与后续 FSDP placement 组合起来。共同 root 相当于告诉 PyTorch：

```text
EP 横轴、expert-FSDP 纵轴、dense 全平面，都是同一张二维座位表的不同视图。
```

如果单独再创建一个无关的 1D world mesh，rank 名单看似正确，FSDP2 仍可能认为 DP mesh 和已有 DTensor
mesh 不属于同一个 parent，进而在 placement 组合或 collective 中报错。

## 7. 参数到底交给谁管理

目标 ownership 如下：

| 模块 | 目标 FSDP unit | 原因 |
|---|---|---|
| attention / DSA indexer | dense-FSDP | 每层所有 token 都会经过 |
| router / gate | dense-FSDP | 它负责选专家，但自身不是 routed expert |
| shared experts | dense-FSDP | 所有 token 都可能执行，不按 EP 独占分配 |
| norm / embedding / LM head | dense-FSDP | 模型级 dense 参数 |
| `MoEDecoderLayer.experts` | expert-FSDP | routed expert 的唯一模块边界 |

最重要的规则是：

```text
每个 trainable parameter 必须恰好归属于一个 FSDP unit。
```

不能归零个，否则它可能被完整复制；不能归两个，否则可能重复分片和重复规约。

## 8. 一层训练时实际发生什么

以一个 MoE decoder layer 为例。

### 8.1 前向

```text
1. 外层 dense-FSDP 在 8 卡间 all-gather 本层 dense 参数
2. attention / indexer / norm 计算
3. router 给每个 token 选择 top-k routed experts
4. dispatcher 在 EP 组内把 token 发到对应 EP 位置
5. 内层 expert-FSDP 在本列两卡间 all-gather local expert 参数
6. grouped GEMM 执行 routed expert 计算
7. dispatcher 把 expert 输出送回原 token 所在位置并 combine
8. 若开启 reshard，inner expert 和 outer dense 各自拆回本地 shard
```

### 8.2 反向

```text
1. 按 autograd 的逆序进入 expert 和 dense 模块
2. 对需要重算的参数再次 all-gather
3. 计算局部梯度
4. FSDP 在各自 mesh 内执行 reduce-scatter
5. dense shard 留在 8 卡各自位置；expert shard 留在对应 expert-FSDP 组
6. XTuner 保留 expert 梯度的 1 / EP 缩放，维持当前 loss 平均语义
7. AdamW 只更新本卡持有的参数 shard 和状态 shard
```

dense 和 expert 会使用不同通信组，这正是 nested FSDP2 的核心。

## 9. 挨个解释设计伪代码

### 9.1 配置开关

```python
moe_fsdp_mesh: Literal["coupled", "decoupled"] = "coupled"
```

含义：

- 默认 `coupled`，现有 YAML 不改就继续走旧路径；
- 只有显式设置 `decoupled` 才进入新实现；
- 这是为了便于回滚、A/B 对比和控制兼容范围。

为什么第一版限制 TP/SP/PP、MTP、FP8、compile 和 offload：不是证明这些能力永远不兼容，而是先只改变一个
核心变量。否则出错时无法判断是 mesh、wrapper、参数 padding、offload 生命周期还是 compile graph 导致。

### 9.2 `MoEParallelMeshes`

```python
@dataclass(frozen=True)
class MoEParallelMeshes:
    root: DeviceMesh
    ep: DeviceMesh
    expert_fsdp: DeviceMesh
```

它只是一个装 mesh 的只读容器：

- `root` 保存完整二维关系；
- `ep` 给 dispatcher 和 expert 权重的 EP placement 使用；
- `expert_fsdp` 给 routed expert 的内层 FSDP 使用。

`frozen=True` 表示创建后不应随意替换字段，避免训练中不同模块拿到不一致的 mesh。

### 9.3 `build_moe_meshes_once()`

```python
root = init_device_mesh(
    device,
    (world_size // ep_size, ep_size),
    mesh_dim_names=("expert_fsdp", "ep"),
)
```

输入 8 卡、EP4 时得到 `(2,4)`。函数名中的 `once` 强调 root 应统一创建，再派生子 mesh，而不是模型构造时
先建一份 EP mesh、FSDP 时又建一份“看起来一样”的 mesh。

### 9.4 `get_dense_fsdp_mesh()`

```python
if mode == "coupled":
    return meshes.expert_fsdp
return flatten_same_root(meshes.root, dims=("expert_fsdp", "ep"))
```

它是模式差异的核心：

```text
coupled：   dense 只使用纵轴，EP4 时 size=2
decoupled：dense 把二维 root 拉平，size=8
```

`flatten_same_root()` 在设计稿中是语义化占位函数，不是当前 XTuner 已存在、可以直接调用的 API。真实开发时
需要按当前 PyTorch 版本选择 `DeviceMesh._flatten()` 或封装兼容层，并测试进程组创建顺序。

### 9.5 扩展 `_fully_shard()` 的 `ignored_params`

```python
def _fully_shard(..., extra_ignored_params=None):
    ignored = collect_existing_fp32_ignored_params(...)
    ignored.update(extra_ignored_params or set())
    fully_shard(..., ignored_params=ignored or None)
```

XTuner 现有 `_fully_shard()` 已会忽略需要保持 FP32 的特殊参数。新设计不能覆盖这套行为，而是把 routed expert
参数加入已有 ignored 集合：

```text
原有 ignored：特殊 FP32 参数
新增 ignored：已经交给内层 FSDP 的 routed expert 参数
最终 ignored：两者并集
```

### 9.6 `MoE.fully_shard()` 的整体顺序

伪代码的关键顺序是：

```text
原始 layer
  -> 内层 fully_shard(raw_layer.experts)
  -> activation checkpoint 包装整个 layer
  -> 外层 fully_shard(layer, ignored_params=expert_params)
```

为什么先找 `raw_layer.experts`：checkpoint wrapper 会在模块路径中增加包装层；如果先包装再找 expert，ownership
识别、参数名清洗和 hook 顺序都会更复杂。

为什么外层仍包整个 layer：外层需要管理 attention、router、shared expert、norm 等 dense 参数，但通过
`ignored_params` 排除 routed expert。

为什么最后还要 shard 根模型：embedding、norm、LM head 和没有被子 FSDP unit 管理的模型级参数仍要有归属。
PyTorch nested FSDP 会跳过已经属于子 FSDP unit 的参数。

### 9.7 为什么 decoupled 不调用 `_replicate_other_params()`

当前 XTuner 的 `_replicate_other_params()` 会给非 expert 参数增加 EP `Replicate()` placement。这正是 dense shard
沿 EP 复制的来源之一。

解耦后 dense 参数已经在全 8 卡 FSDP 上 `Shard`，因此不能再人为增加 EP 副本。coupled 路径则必须保留当前
调用，才能保持旧行为和数值回归基线。

### 9.8 `refresh_runtime_param_ownership()`

伪代码同时建立两种索引：

```text
Parameter 对象 id -> 给训练热路径判断 expert 梯度
canonical 参数名 -> 给 state_dict / HF load / HF export 判断参数角色
```

为什么不在模型刚构造时永久保存参数 `id`：FSDP、meta materialization 或 wrapper 可能替换 Parameter 对象。
因此应保存 expert **模块引用**，等 FSDP 和 `_to_empty_meta()` 完成后，再从最终参数对象重建 id 集合。

为什么还要 canonical name：`state_dict()` 返回的 Tensor 不一定和训练中的 Parameter 是同一个对象，而且
checkpoint/compile wrapper 可能在名字中插入额外前缀。保存路径更适合按清洗后的稳定名字判断角色。

### 9.9 `scale_and_reduce_grad()`

核心逻辑是：

```python
if param 是 routed expert:
    grad /= ep_size
else:
    只对真实 Replicate placement 做规约
```

expert 在不同 EP 位置持有不同权重，不能像普通数据并行副本一样跨 EP 把不同专家的梯度相加；但 XTuner 当前
loss 平均语义需要保留 `1/EP` 缩放。

dense 参数解耦后由全 8 卡 FSDP 自己 reduce-scatter，不再是 EP 副本，因此不能额外按 EP all-reduce。否则会
重复平均，导致梯度系统性缩小。

通用 `Replicate` 规约仍不能删，因为某些被 FSDP 忽略的 FP32 参数可能真的保留了副本。

### 9.10 `runtime_fsdp_mesh()`

```python
if 参数角色 == routed_expert:
    return expert_fsdp_mesh
return dense_fsdp_mesh
```

当前 XTuner 的 HF 路径多处默认所有参数都使用 `self.fsdp_mesh`。解耦后这个假设不成立：

```text
dense 本地 shard 要在 8 卡组内 gather
expert 本地 shard 要在对应的 2 卡 expert-FSDP 组内 gather
```

如果选错 group，代码可能直接报 shape 错误，也可能更危险地拼出尺寸看似合理但内容错误的权重。

### 9.11 为什么 HF 参数按 `(LoadEnum, fsdp_mesh)` 分桶

`LoadEnum.SAME/FUSED/SHARD` 描述的是 XTuner 参数和 HF 参数之间如何映射，不描述参数在哪个 FSDP mesh 上。
一个 `FUSED` 桶里可能同时出现 dense fused 权重和 expert fused 权重，因此只按 `LoadEnum` 分桶不够。

正确思路是：

```text
先按转换类型分类
再按实际 gather mesh 分类
```

即 `(load_enum, runtime_fsdp_mesh)`。每个桶使用对应 process group all-gather，之后再复用原有 HF 转换逻辑。

### 9.12 为什么 DCP 初期可以不改格式

XTuner 的 TrainEngine 已通过 PyTorch 官方 distributed state-dict 和 DCP API 保存 model/optimizer。理论上这些 API
理解 DTensor placement，因此第一版优先复用，不另造格式。

但“API 支持”不等于“无需测试”。第一版只承诺**同一拓扑恢复**：

```text
decoupled EP4 保存 -> decoupled EP4 恢复
```

下面这些暂不承诺：

```text
coupled EP4 -> decoupled EP4
decoupled EP2 -> decoupled EP4
```

跨拓扑转换先走 consolidated HF 权重更稳妥。Trainer 另外保存 scheduler 和 step，也必须一起验证连续性。

## 10. activation checkpoint 和 reshard 不是一回事

这两个功能都省显存，但省的是不同对象：

| 功能 | 省什么 | 代价 |
|---|---|---|
| activation checkpoint | 前向激活 | 反向时重算前向 |
| `reshard_after_forward` | 前向后临时 all-gather 的完整参数 | 反向前再次 all-gather |

解耦后内层 expert 和外层 dense 都有自己的 unshard/reshard 生命周期，checkpoint replay 又会再次执行前向。
所以 wrapper 顺序和所有 rank 的 collective 顺序必须一致，否则可能出现 hang、重复 all-gather 或 metadata 错误。

## 11. mixed precision 在这里做什么

本设计要求保留现有 BF16 混合精度语义：

```text
FP32 master 参数和 AdamW 状态：保证更新精度
BF16 前向参数视图：降低计算和通信成本
BF16 梯度规约：按当前配置执行
部分算子内部：可能使用 FP32 累加
```

mesh 解耦只改变“在哪些 rank 上分片和通信”，不应顺手改变 dtype。否则 loss/grad 差异无法确定来自 mesh 还是
精度策略。

## 12. 这项改动的收益与代价

### 12.1 预期收益

- EP2、EP4 下 dense 参数都保持 8 路 FSDP；
- 降低 dense 参数、梯度和 AdamW 状态的单卡常驻显存；
- EP 大小时不再被迫让 dense-FSDP 变浅；
- 更接近 AutoModel 对 dense 和 routed expert 分别管理的思路。

### 12.2 代价

- dense collective 从 `world/EP` 扩大到全 rank，通信量和通信范围可能增加；
- nested FSDP 增加 FSDP unit、hook 和进程组管理复杂度；
- HF load/export 必须识别两种 runtime mesh；
- activation checkpoint、compile、offload、MTP 和 FP8 的组合风险增大；
- 多机时 dense all-gather 可能跨节点，显存更省但速度未必更快。

所以“解耦”首先是显存和拓扑能力优化，不应在实测前直接宣称一定更快。

## 13. 如何判断实现是否正确

建议按四层验收，不要一上来只跑 200 step 看 loss：

### 13.1 拓扑

```text
EP2：dense_fsdp=8，expert_fsdp=4，ep=2
EP4：dense_fsdp=8，expert_fsdp=2，ep=4
```

打印每个关键参数的 `device_mesh` 和 `placements`，确认 dense/expert 归属。

### 13.2 ownership

- 每个 trainable parameter 恰好被一个 FSDP unit 管理；
- routed expert 只归内层；
- attention、router、shared expert、norm 等只归外层；
- 没有因为 wrapper 前缀导致参数漏识别。

### 13.3 数值

固定同一 batch 比较 coupled/decoupled：

```text
一步 loss
每类参数 grad
grad_norm
optimizer.step 后的参数 delta
AdamW m/v 状态
```

然后再跑 20 step 看曲线，不要用 6 层模型的下游生成能力判断 mesh 正确性。

### 13.4 checkpoint

```text
HF strict load -> train -> export -> strict reload
DCP save -> resume -> 下一步与不中断训练一致
```

同时检查 safetensors index 不缺 key、不重复 key、文件名不覆盖。

## 14. 原设计文档每章在解决什么

| 原设计章节 | 真正要回答的问题 |
|---|---|
| 1. 先给结论 | 这不是只换一个 mesh，还会影响哪些系统 |
| 2. 两边当前怎么切 | AutoModel 和 XTuner 的参数布局差在哪里 |
| 3. 目标拓扑 | 8 卡上三种通信组应该怎么组成 |
| 4. 第一版设计决策 | 哪些行为这次改、哪些先保持不动 |
| 5. 单文件伪代码 | 代码控制流和改动落点是什么 |
| 6. 真实文件边界 | PR 预计会碰哪些模块 |
| 7. 兼容性边界 | 第一版承诺支持什么 |
| 8. 风险 | 最容易静默出错的地方是什么 |
| 9. 验收标准 | 怎么证明不是“能跑就算正确” |
| 10. 提交顺序 | 怎么拆 PR 才容易定位回归 |
| 11. 源码依据 | 上述判断可以去哪里核实 |

## 15. 推荐的源码阅读顺序

不要从 500 行伪代码硬啃，按下面顺序更容易建立联系：

1. `xtuner/v1/module/decoder_layer/moe_decoder_layer.py::MoEDecoderLayer`
   先认出 attention、gate、shared experts 和 routed experts 的模块边界。
2. `xtuner/v1/module/grouped_linear/moe_group_linear.py::GroupedLinear`
   看 routed expert 权重如何通过 EP `Shard(0)` 变成 DTensor。
3. `xtuner/v1/model/moe/moe.py::MoE._init_device_mesh`
   看当前 `(fsdp, ep)` mesh 如何建立。
4. `xtuner/v1/model/moe/moe.py::MoE._replicate_other_params`
   看 dense 参数为何沿 EP 形成副本。
5. `xtuner/v1/model/moe/moe.py::MoE.fully_shard`
   看 layer、embedding、norm、LM head 当前如何被 FSDP 包装。
6. `xtuner/v1/model/moe/moe.py::MoE.scale_and_reduce_grad`
   看 expert 缩放和 Replicate 梯度规约。
7. `xtuner/v1/model/base.py::_fully_shard` 与 HF load/save 方法
   理解 ignored params 和 checkpoint 为什么是主要改动点。
8. AutoModel `nemo_automodel/components/distributed/mesh_utils.py`
   看主 mesh 如何派生 MoE mesh。
9. AutoModel `nemo_automodel/components/moe/parallelizer.py::apply_fsdp`
   看 inner experts FSDP 和 outer block FSDP 如何组合。

## 16. 自测题

### 问题 1

8 卡 EP4 时，为什么当前 XTuner dense-FSDP 只有 2，而不是 8？

答：root 是 `(fsdp=2, ep=4)`；dense 沿 fsdp 切、沿 ep 复制，所以只切成 2 份。

### 问题 2

`[0,4]` 是不是表示 GPU 0 和 GPU 4 各负责一个完全不同的 EP 专家组？

答：不是。它们在同一个 EP 列，负责同一份 local expert tensor 的不同 FSDP 存储分片。具体切到哪些行或
特征维由权重布局和 `Shard(dim)` 决定。

### 问题 3

为什么外层 block 要忽略 routed expert？

答：routed expert 已经归内层 FSDP 管理；一个参数不能同时归两个 FSDP unit。

### 问题 4

为什么解耦后不能继续对 dense 梯度按 EP all-reduce？

答：dense 已经在全 8 卡 FSDP 中完成 reduce-scatter，不再是 EP 副本；再次规约会重复平均。

### 问题 5

为什么 HF checkpoint 比 DCP 更需要显式改代码？

答：HF 导出要把不同 mesh 上的本地 shard 重新拼成全局权重，必须明确选择 dense 或 expert 的 gather group；
DCP 本身能记录 DTensor 分布状态，但同拓扑恢复仍然必须实测。

### 问题 6

解耦后显存一定下降、速度一定提升吗？

答：dense 参数和 optimizer state 的单卡显存预期下降；速度不一定提升，因为 dense collective 组变大，尤其
多机时可能增加跨节点通信。

## 17. 最后记住这张图

```text
                    EP0  EP1  EP2  EP3
expert-FSDP 第 0 份   0    1    2    3
expert-FSDP 第 1 份   4    5    6    7

横向：EP，负责“专家分给谁、token 发给谁”
纵向：expert-FSDP，负责“本地 routed expert 权重怎么分片保存”
全平面：dense-FSDP，负责“attention 等 dense 参数怎么在 8 卡分片”
```

解耦并不是删除 EP，也不是把所有专家复制到每张卡；它只是让 dense 和 routed expert 分别选择更合适的
FSDP 通信组。
