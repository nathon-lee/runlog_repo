## AutoSP 集成方案：多模态模型序列并行

### 核心问题分析

现有 `UlyssesSPAttentionHF` 只针对纯文本 Transformer（Decoder-only LLM）设计，其假设：
1. 模型只有一种 attention 结构（causal attention）
2. 序列维度在整个前向传播中保持一致分片
3. 没有跨模态（图像 token + 文本 token）的序列拼接场景

多模态模型（如 LLaVA、InternVL、Qwen-VL 等）的挑战：

| 挑战 | 描述 |
|---|---|
| **双编码器结构** | ViT encoder + LLM decoder，两者 attention head 数/结构不同 |
| **序列拼接点** | 视觉 token 与文本 token 在某一层合并，此处序列维度发生变化 |
| **自动检测** | 用户无需手动指定哪些层用 SP、哪些层不用 |
| **位置编码兼容** | ViT 的 position embedding 基于固定 patch 网格，序列切分后需重构 |

---

### 总体架构

```
┌─────────────────────────────────────────────────────────┐
│                   AutoSP 入口                            │
│          auto_wrap_model_for_sp(model, sp_group)         │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
    ┌──────────▼──────────┐    ┌──────────▼──────────┐
    │  ViT 编码器分支      │    │   LLM 解码器分支     │
    │  UlyssesSPViTAttn   │    │  UlyssesSPAttentionHF│
    │  (patch-aware SP)   │    │  (已有实现)           │
    └──────────┬──────────┘    └──────────┬──────────┘
               │                          │
    ┌──────────▼──────────────────────────▼──────────┐
    │           ModalityFusionSPAdapter               │
    │  (在 vision-language projection 处做序列重组)    │
    └─────────────────────────────────────────────────┘
```

---

### 实现步骤

#### 步骤 1：模型架构自动探测（`autosp_detector.py`）

```python
# deepspeed/runtime/sequence_parallel/autosp_detector.py
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""
自动探测多模态模型中的 ViT encoder 与 LLM decoder 组件，
为 AutoSP 注入提供元信息。
"""

import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional

# 已知的 ViT attention 类名（可扩展）
_VIT_ATTN_CLASSNAMES = {
    "ViTAttention", "CLIPAttention", "SiglipAttention",
    "InternVisionAttention", "Qwen2VLVisionAttention",
}

# 已知的 LLM attention 类名
_LLM_ATTN_CLASSNAMES = {
    "LlamaAttention", "MistralAttention", "Qwen2Attention",
    "InternLM2Attention", "GemmaAttention",
}

@dataclass
class SPModelInfo:
    vit_attn_modules: List[nn.Module] = field(default_factory=list)
    llm_attn_modules: List[nn.Module] = field(default_factory=list)
    # 视觉-语言投影层（序列 gather/scatter 发生处）
    vision_projection_module: Optional[nn.Module] = None
    vit_num_heads: int = 0
    llm_num_heads: int = 0

def detect_model_sp_info(model: nn.Module) -> SPModelInfo:
    """递归扫描模型，识别需要 SP 包装的模块。"""
    info = SPModelInfo()
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name in _VIT_ATTN_CLASSNAMES:
            info.vit_attn_modules.append((name, module))
        elif cls_name in _LLM_ATTN_CLASSNAMES:
            info.llm_attn_modules.append((name, module))
        # 探测投影层（常见命名）
        if any(k in name for k in ["visual_projection", "mm_projector", "vision_proj"]):
            info.vision_projection_module = (name, module)
    return info
```

---

#### 步骤 2：ViT SP Attention 包装器（`autosp_vit.py`）

ViT 的关键差异：patch token 是空间有序的，序列切分需要按 **patch 块** 均匀分配而非随机分配。

```python
# deepspeed/runtime/sequence_parallel/autosp_vit.py
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import torch
import torch.nn as nn
import deepspeed.comm as dist
from deepspeed.sequence.layer import _DimZeroAllToAll


class UlyssesSPViTAttention(nn.Module):
    """
    为 ViT encoder 的 attention 层提供 Ulysses 序列并行包装。

    与 UlyssesSPAttentionHF 的核心区别：
    - ViT 输入形状为 [bs, num_patches, hidden_dim]，无 causal mask
    - 支持 cls_token：cls token 不参与序列切分，在每个 rank 上保留完整副本
    - num_patches 必须被 sp_world_size 整除（不满足时自动 padding）
    """

    def __init__(self, attn: nn.Module, process_group: dist.ProcessGroup,
                 has_cls_token: bool = True):
        super().__init__()
        self.attn = attn
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.has_cls_token = has_cls_token

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # hidden_states: [bs, seq_len, hidden_dim]
        # seq_len = 1(cls) + num_patches  (若 has_cls_token)
        bs, seq_len, hidden_dim = hidden_states.shape

        if self.has_cls_token:
            # cls token 不做序列并行，单独保留
            cls_token = hidden_states[:, :1, :]       # [bs, 1, hidden_dim]
            patch_tokens = hidden_states[:, 1:, :]    # [bs, num_patches, hidden_dim]
        else:
            patch_tokens = hidden_states

        num_patches = patch_tokens.shape[1]
        # padding 使 num_patches 能被 world_size 整除
        pad_len = (self.world_size - num_patches % self.world_size) % self.world_size
        if pad_len > 0:
            patch_tokens = torch.nn.functional.pad(patch_tokens, (0, 0, 0, pad_len))

        # 序列切分：每个 rank 持有本地 patch 子集
        local_patches = self._scatter_seq(patch_tokens)  # [bs, local_patches, hidden_dim]

        if self.has_cls_token:
            local_input = torch.cat([cls_token, local_patches], dim=1)
        else:
            local_input = local_patches

        # 调用原始 attn（此处 Q/K/V 投影在各 rank 本地完成）
        # 然后通过 all-to-all 交换获得完整 heads
        out = self._ulysses_attn_forward(local_input, **kwargs)

        # 反向 gather 序列
        if self.has_cls_token:
            cls_out = out[:, :1, :]
            patch_out = out[:, 1:, :]
        else:
            patch_out = out

        patch_out = self._gather_seq(patch_out)
        if pad_len > 0:
            patch_out = patch_out[:, :num_patches, :]

        if self.has_cls_token:
            return torch.cat([cls_out, patch_out], dim=1)
        return patch_out

    def _scatter_seq(self, x: torch.Tensor) -> torch.Tensor:
        """将序列维度按 rank 切分。"""
        rank = dist.get_rank(self.process_group)
        seq_len = x.shape[1]
        local_len = seq_len // self.world_size
        return x[:, rank * local_len:(rank + 1) * local_len, :].contiguous()

    def _gather_seq(self, x: torch.Tensor) -> torch.Tensor:
        """all-gather 序列维度。"""
        gathered = [torch.zeros_like(x) for _ in range(self.world_size)]
        dist.all_gather(gathered, x, group=self.process_group)
        return torch.cat(gathered, dim=1)

    def _ulysses_attn_forward(self, x, **kwargs):
        # 委托给原始 attn，all-to-all 由 _DimZeroAllToAll 处理
        return self.attn(x, **kwargs)
```

---

#### 步骤 3：模态融合适配器（序列重组）

视觉 token 经过 ViT 后需与文本 token 拼接。此处需要 **gather 视觉序列 → 拼接 → 重新 scatter**：

```python
# deepspeed/runtime/sequence_parallel/autosp_fusion.py
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import torch
import torch.nn as nn
import deepspeed.comm as dist


class ModalityFusionSPAdapter(nn.Module):
    """
    在视觉投影层之后、LLM 输入之前插入，负责：
    1. gather 所有 rank 上的视觉 token
    2. 与文本 token 拼接
    3. 按新的 (visual+text) 总长度重新 scatter 给各 rank

    这样 LLM decoder 的 UlyssesSPAttentionHF 可以直接使用已有逻辑。
    """

    def __init__(self, projection: nn.Module, process_group: dist.ProcessGroup):
        super().__init__()
        self.projection = projection  # 原始 vision projection 模块
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)

    def forward(self, visual_features, text_input_ids, text_embeds, **kwargs):
        # 1. 投影视觉特征
        visual_embeds = self.projection(visual_features)  # [bs, num_visual, hidden]

        # 2. 全局 gather 视觉 token（来自各 rank 的分片）
        visual_list = [torch.zeros_like(visual_embeds) for _ in range(self.world_size)]
        dist.all_gather(visual_list, visual_embeds, group=self.process_group)
        full_visual = torch.cat(visual_list, dim=1)  # [bs, total_visual, hidden]

        # 3. 将视觉 token 插入文本序列对应位置（由 image token 占位符标记）
        # 这里假设 text_embeds 中已预留 image placeholder
        fused = _insert_visual_tokens(text_embeds, full_visual, text_input_ids)

        # 4. 重新 scatter 给各 rank（供 LLM SP 使用）
        total_len = fused.shape[1]
        pad = (self.world_size - total_len % self.world_size) % self.world_size
        if pad:
            fused = torch.nn.functional.pad(fused, (0, 0, 0, pad))
        rank = dist.get_rank(self.process_group)
        local_len = fused.shape[1] // self.world_size
        return fused[:, rank * local_len:(rank + 1) * local_len, :].contiguous()


def _insert_visual_tokens(text_embeds, visual_embeds, input_ids, image_token_id=-200):
    """将视觉 token 替换 text_embeds 中的图像占位符位置（标准做法）。"""
    # 实现略，与 LLaVA 等框架的 prepare_inputs_embeds 逻辑一致
    ...
```

---

#### 步骤 4：AutoSP 主入口（`auto_sp.py`）

```python
# deepspeed/runtime/sequence_parallel/auto_sp.py
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""
AutoSP: 一行代码为多模态模型启用序列并行。
用法：
    from deepspeed.runtime.sequence_parallel.auto_sp import auto_wrap_model_for_sp
    model = auto_wrap_model_for_sp(model, sp_group=sp_group)
"""

import torch.nn as nn
import deepspeed.comm as dist
from .autosp_detector import detect_model_sp_info
from .autosp_vit import UlyssesSPViTAttention
from .ulysses_sp import UlyssesSPAttentionHF
from .autosp_fusion import ModalityFusionSPAdapter


def auto_wrap_model_for_sp(
    model: nn.Module,
    process_group: dist.ProcessGroup,
    seq_length_is_variable: bool = True,
) -> nn.Module:
    """
    自动扫描模型并注入序列并行包装器：
    - ViT attention → UlyssesSPViTAttention
    - LLM attention → UlyssesSPAttentionHF
    - 视觉投影层   → ModalityFusionSPAdapter
    """
    info = detect_model_sp_info(model)

    # 包装 ViT attention 层
    for name, module in info.vit_attn_modules:
        sp_attn = UlyssesSPViTAttention(module, process_group)
        _set_module_by_name(model, name, sp_attn)

    # 包装 LLM attention 层
    for name, module in info.llm_attn_modules:
        sp_attn = UlyssesSPAttentionHF(
            attn=module,
            process_group=process_group,
            seq_length_is_variable=seq_length_is_variable,
            # 其余参数由 detector 自动提取
            **_extract_llm_attn_params(module),
        )
        _set_module_by_name(model, name, sp_attn)

    # 包装视觉投影层
    if info.vision_projection_module:
        name, proj = info.vision_projection_module
        fusion = ModalityFusionSPAdapter(proj, process_group)
        _set_module_by_name(model, name, fusion)

    return model


def _set_module_by_name(model, name, new_module):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _extract_llm_attn_params(module):
    """从 LLM attention 模块中自动提取头数等参数。"""
    config = getattr(module, "config", None)
    if config is None:
        return {}
    return {
        "attn_head_count": getattr(config, "num_attention_heads", 32),
        "attn_head_size": getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32),
        "kv_head_count": getattr(config, "num_key_value_heads",
                                 getattr(config, "num_attention_heads", 32)),
        "num_hidden_layers": getattr(config, "num_hidden_layers", 32),
        "batch_size": 1,
    }
```

---

### 文件结构总结

```
deepspeed/runtime/sequence_parallel/
├── __init__.py
├── parallel_state_sp.py        (已有)
├── ulysses_sp.py               (已有，LLM SP)
├── autosp_detector.py          ← 新增：模型架构自动探测
├── autosp_vit.py               ← 新增：ViT SP attention 包装
├── autosp_fusion.py            ← 新增：模态融合序列重组适配器
└── auto_sp.py                  ← 新增：AutoSP 主入口（一行启用）
```

---

### 用户侧使用方式（目标体验）

```python
from deepspeed.runtime.sequence_parallel.auto_sp import auto_wrap_model_for_sp
from deepspeed.utils import groups

# 初始化 DeepSpeed（与现有流程完全一致）
model, _, _, _ = deepspeed.initialize(config=ds_config, model=model, ...)

# 一行启用多模态 SP，无需手动修改模型
sp_group = groups._get_sequence_parallel_group()
model = auto_wrap_model_for_sp(model, process_group=sp_group)
```

---

### 关键技术难点与解决策略

| 难点 | 策略 |
|---|---|
| **ViT cls_token 处理** | 不参与序列切分，在每个 rank 保持完整副本，最后 gather 取均值或只取 rank 0 的副本 |
| **num_patches 不整除 world_size** | 自动 padding 后 forward，backward 前截断，保证梯度正确 |
| **模态融合处序列长度变化** | `ModalityFusionSPAdapter` 负责统一 gather 再 scatter，LLM 感知不到变化 |
| **位置编码** | ViT 的 positional embedding 在 patch embed 阶段已加入，序列切分不影响；RoPE 在 LLM 侧按全局 position index 计算 |
| **变长序列（多图）** | `seq_length_is_variable=True` 配合 `UlyssesSPDataLoaderAdapter` 处理 |

---

### 开发优先级建议

1. **Phase 1（核心）**：实现 `autosp_detector.py` + `autosp_vit.py`，先支持 LLaVA/InternVL 系列
2. **Phase 2（融合）**：实现 `autosp_fusion.py`，解决模态边界的序列重组
3. **Phase 3（自动化）**：实现 `auto_sp.py` 一键注入，并扩展 `_VIT_ATTN_CLASSNAMES` 支持更多架构
4. **Phase 4（测试）**：在 ulysses_alst 下补充多模态 SP 测试，参照现有 test_ulysses_sp_hf.py 模式
