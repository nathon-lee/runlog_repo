## AutoEP + ZeRO-3 技术设计方案

### 一、为什么原型不支持 ZeRO-3

原型的根本矛盾来自两个机制的**正交冲突**：

| 维度 | AutoEP (ZeRO-2) | ZeRO-3 |
|---|---|---|
| **专家权重归宿** | 每个 EP rank 只持有本地 `E_local` 个专家的 `[E_local, ffn, hidden]` 参数（**EP-partitioned**） | 所有参数按 DP 维度 reduce-scatter 成 `flat_shard`，每个 rank 持有 `1/dp_world_size` 的权重片段，与 EP 切分维度**正交且互斥** |
| **前向 gather** | 无需 gather（本地 expert 权重已完整）；仅对 token 做 AllToAll | ZeRO-3 会在 pre-forward hook 触发 `all_gather` 把参数从各 DP rank 拼回，**打破 EP 分片** |
| **梯度 reduce** | 只对 non-expert 参数做 DP all-reduce；expert 参数只在 EP-data-parallel 组内 reduce（`configure_moe_param_groups`）| ZeRO-3 对**所有**参数执行 reduce-scatter，expert 梯度会被按 DP rank 拆散，与 EP-data-parallel 组语义不匹配 |
| **优化器状态** | 每个 rank 只维护 `E_local` 个专家的优化器状态 | ZeRO-3 将优化器状态也按 DP rank 分片，与 EP 分片叠加后会出现状态对应错乱 |

---

### 二、解决思路：四层解耦

#### 层 1 — 参数分区语义分离：让 ZeRO-3 跳过 expert 参数

ZeRO-3 通过 `zero.Init()` 上下文在模型构建时给参数打上分区标记。核心思路是**在 AutoEP inject 时，对已被分区到该 EP rank 的 expert 参数，取消 ZeRO-3 的 all_gather 钩子，改为不分区（或按 EP 维度分区）**。

DeepSpeed 已有两个可复用的钩子点：
- `register_external_parameter(module, param)`：告诉 ZeRO-3 某参数在本 module 外部管理，跳过自动 all_gather。
- `ds_param_id` / `ds_status` / `ZeroParamStatus`：每个 ZeRO-3 参数的状态机，可强制设置为 `AVAILABLE`（不需要 gather）。

**具体做法**：在 AutoEP 的 `inject_auto_ep()` 替换 MoE 层之后，遍历 `AutoEPLayer` 中的 expert 参数（`GroupedExperts.w1/w2/w3`），对它们调用：

```python
# 伪代码，实际需在 zero.Init 上下文里或初始化后处理
param.ds_status = ZeroParamStatus.AVAILABLE       # 告知 ZeRO-3 已在本地完整
param.ds_numel = param.numel()                    # 设置完整尺寸（非分片）
param.ds_tensor = param.data                      # 本地视图即为完整权重
```

或者更稳健的方式：在 `zero.Init()` 的 `enabled` 上下文中，对 expert 子模块用 `zero.Init(enabled=False)` 构造（**嵌套 Init 上下文**）：

```python
# engine.py 中 inject 阶段
with deepspeed.zero.Init(enabled=False):
    new_ep_layer = AutoEPLayer(spec, config, ep_rank, ep_size)
    module.mlp = new_ep_layer  # 替换原始 MoE block
```

这使得 expert `w1/w2/w3` 从一开始就不进入 ZeRO-3 的参数分区图谱。

---

#### 层 2 — 梯度 reduce 正确性：expert 梯度走 EP-data-parallel 组

原型已有 `configure_moe_param_groups()` 将 expert 参数放入单独的 optimizer param group，并指定 `expert_dp_group`（而非全 DP group）做 all-reduce。ZeRO-3 不走这条路，而是直接 reduce-scatter 到 ZeRO partition group。

**解决方案**：在 `DeepSpeedZeroOptimizer_Stage3.__init__` 中，检测 `is_moe_param(p)` / `is_autoep_expert_param(p)` 标记，把这类参数从主 reduce-scatter bucket 中**排除**，转而注册一个独立的 `backward post-hook`，在梯度计算完成后直接做 `dist.all_reduce(grad, group=expert_dp_group)`（仿照 ZeRO-1/2 对 expert 参数的处理方式）。

参考本仓库 `engine.py:1421` 中已有的 `is_moe_param` 分支：

```python
# 现有 ZeRO-1/2 逻辑（engine.py ~L1420）
if is_moe_param(p):
    dist.all_reduce(p.grad, group=self.expert_data_parallel_group[p.group_name])
```

ZeRO-3 需要在 stage3.py 的 `reduce_ipg_grads` / `reduce_ready_partitions_and_reset` 中加入同样的**旁路分支**。

---

#### 层 3 — 优化器状态：expert 参数不参与 ZeRO partition

ZeRO-3 的优化器状态分片逻辑在 stage3.py 的 `_partition_parameters_among_ranks`。需要在分片之前，将 expert 参数**从分片列表中剔除**，交给独立的 full-precision optimizer group 管理（每个 rank 只持有本地 expert 的优化器状态）。

实现上类似 ZeRO++ 的 `zero_hpz_partition_size` 参数组机制，为 expert 参数创建一个 `hpz=1`（不跨 DP 分片）的子 group。

---

#### 层 4 — checkpoint save/load：EP × ZeRO-3 状态字典对齐

由于 expert 参数不走 ZeRO-3 分片，checkpoint 逻辑需要分别处理：
- `non-expert`：走标准 ZeRO-3 的 `consolidate_state_dict`。
- `expert`：直接按 EP rank gather，转为全局 `[E_global, ffn, hidden]` 再落盘（或分 EP rank 存储）。

---

### 三、工作分解（Phase 计划）

```
Phase 1 — Expert 参数豁免（核心，约 2 周）
├── 1a. 给 AutoEP expert 参数打标记（如 param._autoep_expert = True）
├── 1b. zero.Init 中用 enabled=False 子上下文构造 GroupedExperts
└── 1c. 在 partition_parameters.py 的 _convert_to_zero_parameters 中跳过标记参数

Phase 2 — 梯度 reduce 旁路（约 1.5 周）
├── 2a. stage3.py: reduce_ipg_grads 中检测 _autoep_expert，走 EP-DP all_reduce 而非 reduce-scatter
└── 2b. 注册 expert 参数的 grad accumulation hook（配合 grad_accum_steps）

Phase 3 — 优化器状态不分片（约 1.5 周）
├── 3a. stage3.py: _partition_parameters_among_ranks 排除 expert 参数
└── 3b. 为 expert param group 创建不分片的 full-precision 优化器实例（或复用现有 expert_param_group 逻辑）

Phase 4 — Checkpoint 对齐（约 1 周）
├── 4a. save_checkpoint: expert 参数 gather per EP rank → 按 ep_rank 落盘
└── 4b. load_checkpoint: 按 ep_rank 切片恢复 GroupedExperts 权重

Phase 5 — 测试与验收（约 1 周）
├── 5a. 单元测试：expert 参数 ZeRO-3 skip 验证
├── 5b. 集成测试：8xH100 Mixtral forward/backward 正确性（loss 对齐 AutoEP+ZeRO-2）
└── 5c. 内存/吞吐基准对比（AutoEP+ZeRO-3 vs AutoEP+ZeRO-2 vs ZeRO-3 leaf）
```

---

### 四、关键设计决策

| 决策点 | 推荐方案 | 备选 | 理由 |
|---|---|---|---|
| Expert 参数豁免方式 | `zero.Init(enabled=False)` 嵌套上下文（在 inject 时） | 初始化后修改 `ds_status` | 前者更干净，不依赖 ZeRO-3 内部状态机细节 |
| 梯度 reduce 旁路 | 在 `reduce_ipg_grads` 加分支，expert grad 直接 `all_reduce(group=expert_dp_group)` | 注册 autograd hook | 与现有 ZeRO-1/2 对 expert 的处理保持一致，复用 `configure_moe_param_groups` 的 param group 信息 |
| Expert 优化器状态 | 独立的 full optimizer group（不走 ZeRO partition） | ZeRO++ HPZ 机制扩展 | 简单直接；expert 参数本身已按 EP 分片，不需要再 ZeRO 分片 |
| Non-expert 参数 | 走标准 ZeRO-3 全流程（attention/FFN-dense/embedding） | — | 充分利用 ZeRO-3 内存节省，抵消 expert 不分片带来的额外内存占用 |

---

### 五、预期收益与权衡

**收益**：
- Non-expert 参数（通常占 attention/embedding 等）被 ZeRO-3 充分分片 → 对超大模型（attention 参数量大）有显著内存优势（vs AutoEP+ZeRO-2）
- Expert 参数继续走高效 EP AllToAll + grouped GEMM → 保持吞吐优势（vs ZeRO-3 leaf）

**权衡**：
- Expert 参数不被 ZeRO-3 分片，内存占用 ≈ AutoEP+ZeRO-2 的 expert 侧（但 attention 侧大幅节省）
- 实现复杂度中等：需要修改 stage3.py、partition_parameters.py、engine.py 三个核心文件

---

如果你希望，我可以立刻开始 **Phase 1（Expert 参数标记 + ZeRO-3 豁免）** 的实际编码实现。
