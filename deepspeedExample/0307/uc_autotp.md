# PR #7908：Add Universal Checkpoint for AutoTP — 完整分析

## 一、修改目的

**核心目标：让 AutoTP 替换出来的 TP 层，能被 Universal Checkpoint (UC) 正确描述、正确保存、正确恢复。**

旧框架的问题：

| 旧问题 | 说明 |
|---|---|
| TP 分片信息不显式 | UC 恢复依赖启发式规则，不知道每个参数"原本怎么切的" |
| sub-parameter 场景没 UC 支持 | fused QKV / GQA 等参数的逻辑子块边界无法被 UC 感知 |
| 分片语义没有随 checkpoint 落盘 | 转换工具 ds_to_universal.py 拿不到足够信息 |

---

## 二、涉及文件与改动规模

```
deepspeed/checkpoint/universal_checkpoint.py   +106 / -45
deepspeed/module_inject/layers.py              +263 / -0
deepspeed/runtime/bf16_optimizer.py            +17  / -0
deepspeed/runtime/engine.py                    +11  / -0
deepspeed/runtime/zero/stage_1_and_2.py        +17  / -0
tests/unit/checkpoint/test_autotp_universal_checkpoint.py   (新增)
tests/unit/runtime/tensor_parallel/test_autotp_universal_checkpoint.py  (新增)
```

---

## 三、实现方式

这次改动打通了一条**四环联动的链路**：

```
TP 层初始化
    ↓ _mark_uc_metadata()         【layers.py 新增】
参数对象携带 UC metadata
    ↓ collect_autotp_uc_info()    【layers.py 新增】
model 对象携带 UNIVERSAL_CHECKPOINT_INFO
    ↓ engine.py / optimizer       【engine.py / bf16 / zero 修改】
checkpoint 文件落盘
    ↓ _resolve_autotp_partition()  【universal_checkpoint.py 新增】
恢复时按 metadata 精确切分 TP shard
```

---

### 环节 1 — layers.py：每个 TP 层初始化后打元数据

每种 TP 层增加了 `_mark_uc_metadata()` 方法，并在 `__init__` 末尾调用：

```python
self._mark_uc_metadata()
```

覆盖的 TP 层：

| 类 | partition_type | 说明 |
|---|---|---|
| `LinearAllreduce` | `row` | weight 按 dim=1 切，bias replicated |
| `LinearLayer` | `column` | weight 按 dim=0 切，bias 随之切 |
| `SubParamLinearLayer` | `column` | 支持子参数边界，记录 `sub_param_sizes` |
| `SubParamLinearAllreduce` | `row` | 支持子参数边界，bias replicated |

---

### 关键设计：metadata 分两视图

这是整次改动设计上最值得肯定的点。

**Restore metadata（挂在参数对象上，恢复侧用）：**
```
partition_type, partition_dim, logical_shape, output_shape,
sub_param_shape, sub_param_sizes, target_partition_shape,
original_shape, is_bias, replicated
+ 嵌套 conversion 字段
```

**Conversion metadata（嵌套在 `conversion` 里，转换侧用）：**
```
partition_type, partition_dim, sub_param_shape, original_shape,
is_bias, replicated
```

两类消费者需求不同，明确隔离：

- 转换侧只需要 schema
- 恢复侧需要 runtime 级别的细节（`sub_param_sizes`、`target_partition_shape` 等）

---

### 环节 2 — layers.py：新增模型级信息收集

新增 `collect_autotp_universal_checkpoint_info(model)` 函数。

它遍历模型，读取每个参数的 conversion metadata，汇总成：

```python
{
    PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [...],
    TP_REPLICATED_PARAMETER_PATTERNS:        [...],
    VOCABULARY_PARAMETER_PATTERNS:           [...],
    PARAMETER_WITH_SUB_PARAMS:               [...],
    ORIGINAL_VOCAB_SIZE:                     ...,
}
```

格式与已有 UC 框架完全兼容，直接用 regex pattern 描述每个参数。

---

### 环节 3 — engine.py + 两个 optimizer：把 UC info 写入 checkpoint

**engine.py**：AutoTP 替换完成后立刻收集并挂到 model：

```python
setattr(model, UNIVERSAL_CHECKPOINT_INFO, collect_autotp_universal_checkpoint_info(model))
```

保存 checkpoint 时写入 state：

```python
autotp_uc_info = getattr(self.module, UNIVERSAL_CHECKPOINT_INFO, None)
if autotp_uc_info is not None:
    state[UNIVERSAL_CHECKPOINT_INFO] = autotp_uc_info
```

**bf16_optimizer.py / stage_1_and_2.py**：在 `_enable_universal_checkpoint()` 时缓存，避免参数属性被回收后丢失：

```python
autotp_uc_info = getattr(param, UNIVERSAL_CHECKPOINT_INFO, None)
if autotp_uc_info is not None:
    self._universal_checkpoint_info = autotp_uc_info
    break
```

optimizer state dict 保存时也带上：

```python
autotp_uc_info = self._get_universal_checkpoint_info()
if autotp_uc_info is not None:
    state_dict[UNIVERSAL_CHECKPOINT_INFO] = autotp_uc_info
```

---

### 环节 4 — universal_checkpoint.py：恢复时按 metadata 精确切分

新增 `_resolve_autotp_partition()`，在 `load_hp_checkpoint_state()` 里优先调用：

```python
autotp_tp_hp_slice = _resolve_autotp_partition(self, ckpt_dict, full_hp_param, tp_rank, tp_world_size)
if autotp_tp_hp_slice is not None:
    tp_hp_slice = autotp_tp_hp_slice
else:
    ...  # 旧逻辑兼容路径
```

`_resolve_autotp_partition()` 的核心逻辑：

```
有 metadata?
  ├─ replicated → 直接复制全量
  ├─ partition_dim is None → 同上
  └─ 有 sub_param_sizes / sub_param_shape
       ├─ 逐子块 narrow + chunk(tp_world_size)
       └─ cat 拼回来
  └─ 普通情况
       └─ full_view.chunk(tp_world_size, dim=partition_dim)[tp_rank]

最后按 target_partition_shape reshape，返回 flatten
```

---

## 四、是否合理

**整体合理，设计质量不低。** 具体亮点：

| 亮点 | 原因 |
|---|---|
| **双视图 metadata 设计** | 恢复侧和转换侧关注点不同，分开表达而不是用一个结构同时服务 |
| **兼容旧逻辑** | `_resolve_autotp_partition()` 返回 `None` 时自动 fallback，不影响旧 checkpoint |
| **信息尽早缓存** | optimizer 在 `_enable_universal_checkpoint()` 时缓存 UC info，防止参数属性被回收 |
| **测试覆盖两端** | 转换侧（pattern 收集）和恢复侧（partition 切分）都有独立测试 |
| **格式与已有 UC 框架兼容** | 复用 regex pattern 设计，不需要改 ds_to_universal.py 主逻辑 |

---

## 五、存在的问题与风险

| 问题 | 严重程度 | 说明 |
|---|---|---|
| vocabulary 参数识别仍是命名启发式 | 低 | `'embed' in name or 'lm_head' in name` 对非标准命名可能误判 |
| metadata 挂在参数动态属性上 | 低-中 | 大部分场景已通过 optimizer 缓存兜底，但 deepcopy / 重新 materialize 场景需关注 |
| 只覆盖 AutoTP linear 层家族 | 低 | 非线性 TP 层如果将来需要 UC 支持，需要继续接入 `_mark_uc_metadata()` |
| 缺少 end-to-end 集成测试 | 中 | 单测很全，但没有完整走 save → convert → restore 的端到端验证 |

---

## 六、是否满足要求

**满足**。这次改动已经完整实现了以下功能链路：

```
✅ AutoTP 参数分片语义显式化（layers.py）
✅ 模型级 UC schema 自动汇总（collect_autotp_uc_info）
✅ UC info 随 checkpoint 落盘（engine.py + optimizer）
✅ 恢复时按 AutoTP metadata 精确切分（universal_checkpoint.py）
✅ 兼容旧 checkpoint 和旧 UC 路径
✅ 单元测试覆盖转换侧和恢复侧
```

---

## 七、如果做 review 最值得提的 3 个 comment

1. **vocabulary pattern 的启发式识别** — 建议用显式 flag 替代 name pattern 判断，增强健壮性

2. **缺少 end-to-end 测试** — 建议补一个：AutoTP model → save checkpoint → `ds_to_universal` → 按不同 TP degree restore 的完整流程测试

3. **`_mark_uc_metadata()` 的扩展点** — 可以考虑在 base class `TensorParallel_Layer` 里把 `_mark_uc_metadata()` 做成抽象方法或默认空方法（已经是 `return`），但最好在文档里说明"子类需要实现此方法以支持 UC"
