# Z-Image Turbo → XDNA2 NPU 适配调研笔记

> 目标：在 AMD Ryzen AI (XDNA2, STX) NPU 上跑通 Z-Image Turbo 生图。
> 环境：`C:\RyzenAI\1.8.0` + conda env `ryzen-ai-1.8.0` (Ryzen AI Software 1.8.0)
> 状态：**调研完成，未开始实现**

---

## 0. 结论摘要

| 项 | 判断 |
|---|---|
| 可行性 | **中等**，有明确路径 |
| 主要风险 | ONNX 导出（complex RoPE 必须改）+ 拓扑切片器命名 |
| 推荐路线 | **先 bf16**，int8_convrot 留作优化 |
| 工作量估计 | 2 个新文件 + 3 处注册 + 1 处导出改造 |

---

## 1. Z-Image Turbo 架构（来自 `genai-sd-cpp\z_image.hpp` + 移植报告）

| 参数 | 值 |
|---|---|
| 生成器 | **6B**, S3-DiT (Single-Stream) |
| diffusers 类 | `ZImageTransformer2DModel` (diffusers 0.36.0.dev0) |
| `dim` (hidden_size) | **3840** |
| `n_layers` | **30** |
| `n_refiner_layers` | **2** (noise_refiner + context_refiner 各 2) |
| `n_heads` / `n_kv_heads` | **30 / 30** ← **标准 MHA，无 GQA**（已对照官方 config 修正） |
| head_dim | 128 |
| `in_channels` | **16** |
| `all_patch_size` | **[2]** |
| RoPE | **3 轴** `axes_dims = [32, 48, 48]`（合计 128）, `axes_lens = [1536, 512, 512]` |
| `rope_theta` | **256.0** ← 不是标准 10000 |
| FFN | SwiGLU (`w1`/`w2`/`w3`) |
| `qk_norm` | **true** — Q/K 各过 `RMSNorm(head_dim)` |
| adaLN | **仅 noise_refiner 有, context_refiner 无** |
| `ADALN_EMBED_DIM` | 256 |
| `t_scale` | 1000.0 |
| `norm_eps` | 1e-05 |
| 文本编码器 | **`Qwen3Model`**（纯文本，非 VL）, `cap_feat_dim = 2560` |
| tokenizer | `Qwen2Tokenizer` |
| VAE | **`AutoencoderKL`** 16ch（与 FLUX `ae.sft` 同构） |
| scheduler | `FlowMatchEulerDiscreteScheduler` |
| 步数 | 8 (Turbo) |
| 精度 | **bf16 必须** (见 §5) |

> **权威来源**：`Tongyi-MAI/Z-Image-Turbo` 的 `transformer/config.json` 与 `model_index.json`。
> 本机 `z_image.hpp` 的 `JointAttention` 带 `num_kv_heads` 参数，**看起来像 GQA 但实际是 30/30**。

块命名（关键，决定切片器）：
```
noise_refiner.{0,1}      ← 带 adaLN
context_refiner.{0,1}    ← 无 adaLN
layers.{0..29}           ← 主 joint blocks
final_layer
```

---

## 2. 模型资产

| 资产 | 位置 | 体积 |
|---|---|---|
| Z-Image Turbo **int8_convrot** | `obsxrver/ComfyUI-Native-INT8_ConvRot` → `diffusion_models/z_image_turbo_int8_convrot.safetensors` | **6,299,127,240 B = 5.87 GiB** |
| Z-Image Turbo bf16 | `Tongyi-MAI/Z-Image-Turbo` | ~12 GB |
| Z-Image Turbo GGUF | `leejet/Z-Image-Turbo-GGUF` | 多档 |
| 文本编码器 | `unsloth/Qwen3-4B-Instruct-2507-GGUF` 等 | — |
| VAE | `black-forest-labs/FLUX.1-schnell` → `ae.sft` | — |

**体积交叉验证**：5.87 GiB ÷ 6B 参数 ≈ **0.98 字节/参数** → 确认为真 INT8 存储（非 weight-only 伪量化）。

**注意**：本机 `GenAI-SD\genai-sd-cpp` 是**旧版**，缺 `docs/int8_convrot.md` 和 `docs/qwen_image_2.1.md`。上游 master 已有。

---

## 3. AMD 侧扩展点（已核实到行号）

### 3.1 模型注册枚举
`Lib\site-packages\ryzenai_onnx_utils\strategy_builder\sd\__init__.py:12-23`
```python
class SDModel(enum.StrEnum):
    AUTO = "infer_from_path"
    SD_UNET = "sd-unet"
    SD_DIT = "sd-dit"
    SD15_VAE_ENCODER / SD15_VAE_DECODER
    SD3_VAE_ENCODER / SD3_VAE_DECODER
    SD_FLUX = "flux"
    SD_FLUX2 = "flux2"
```
`SUPPORTED_SD_MODELS = [e.value for e in SDModel]` 自动成为 CLI `--model-type` 的 choices。

### 3.2 策略选择器
`...\strategy_builder\sd\builder.py`
| 函数 | 行 | 作用 |
|---|---|---|
| `infer_model_type()` | 26-72 | 按**路径字符串**猜模型类型 |
| `preprocess_options()` | 75-97 | 返回 `(optimize_profile, save_as_external)` |
| `default_partition_strategy()` | 100-114 | 返回分区 YAML 文件名 |

**路径匹配优先级**（`infer_model_type` 40-52 行）：`flux2` → `flux` → `transformer`/sd3 → `unet`/sd15

⚠️ Z-Image 的目录若含 `transformer`，会被判为 `SD_DIT`，自动走 `mmdit` + `sd3_mmdit_bfp.yaml`。

### 3.3 预处理 profile（`--optimize` 的值）
`...\model_preprocessing\` 下的模块名即 `--optimize` 取值：

```
sd15_unet.py  sdxl_turbo_unet.py  mmdit.py  flux1.py  flux2.py  vae.py
sd15_controlnet.py  sd3_controlnet.py
llm.py  llm_vision.py  qwen3_vision.py  phi4_vision.py  phi35_vision.py  transgemma_vision.py
```

**每个模块只有 3 个函数**：`pre_optimize_passes()` / `finalize()` / `optimize()`

**实测：`mmdit.py`(162行) 和 `flux1.py`(152行) 的 `optimize()` 函数逐字相同**，都用
`FusionOptions(model_type="unet")` 且 `optimizer.optimize_model(..., model_type="unet")`。
→ **新模型基本只需改 `finalize()` 的 pass 列表。**

### 3.4 DD 分区 pass
`...\passes\dd\`
| 文件 | 作用 |
|---|---|
| `dynamic_sd_nodes_to_dd.py` | **基类**。hint = `{"SD"}`，即「op_type 含 SD 的节点」 |
| `dynamic_flux_nodes_to_dd.py` | FLUX.1 拓扑切片（354 行，注释极详细） |
| `dynamic_flux2_nodes_to_dd.py` | FLUX.2 拓扑切片（114 行，用断点前缀） |
| `dynamic_llmvision_nodes_to_dd.py` | LLM/VLM 切片 |

`dynamic_sd_nodes_to_dd.py:50-88` 的 `replacement()` 被所有切片器复用。
`dynamic_flux2_nodes_to_dd.py:113` → `REPLACEMENT = sd_dd_replacement`

### 3.5 `ModelType` 枚举（⚠️ 同步 C++）
`...\passes\model_type.py`
```python
# this should be kept in sync with the C++ version in
# onnx_custom_ops/dynamic_dispatch/external_buffers.hpp
class ModelType(enum.IntEnum):
    UNET=0, DECODER, UNET_BFP, DECODER_BFP, SD15_UNET, SD15_DECODER,
    SD30_MMDIT, SD30_VAE, LLM_PREFILL, LLM_TOKEN, LLM_VISION,
    FLUX_TRANSFORMER, UNKNOWN
```
**没有 ZIMAGE 项。** YAML 里 `model_type: <名>` 会被 `ModelType[model_type]` 查表并写成 int。

**缓解**：可复用 `FLUX_TRANSFORMER`（同为 Single-Stream DiT），避免动 C++。
风险未验证 —— 若 NPU 运行时按此值分派不同内核，可能出错。

---

## 4. 分区策略结构（已解构）

以 `flux2_transformer_bfp.yaml`（18 行）为例：
```yaml
attributes:
    domains: com.ryzenai
    xclbins: "stx_SD_unified_fastpm_2x4x4"      # STX overlay，与本机匹配
    op_namespaces: "sd3"
    model_type: FLUX_TRANSFORMER
    flux2: flux2
    op_config: "ops-config.json"
inherit_passes_before: !include [sd3_mmdit_no_dd_bfp.yaml]   # ← 114 行算子级通用链
passes:
    - dd.optimize_toposort
    - dd.dynamic_flux2_nodes_to_dd              # ← 唯一的架构专属 pass
```

`sd3_mmdit_no_dd_bfp.yaml` 是**算子级通用转换链**，覆盖：
```
gemm / gemm_mul_add / slice_gemm / ditfusion / slice / layernorm /
resize / mha(多种变体) / rope / rmsnorm / concat / mul_add / mul /
add / gelu / matmul / groupnorm / whitebox_checker / weights_shuffle
```

**策略文件目录**（共 70+ 个）：
`Lib\site-packages\ryzenai_onnx_utils\data\partition_strategies\`

---

## 5. 三个致命坑（来自 Core AI 移植报告，每个都会产出错图）

来源：`john-rocky/coreai-model-zoo` → `knowledge/zimage-port.md`

### 5.1 编码器取**倒数第二层** hidden state
```python
# 正确
caption_embeds = encoder_output.hidden_states[-2]
# 错误（默认做法）
caption_embeds = encoder_output.last_hidden_state
```

### 5.2 CFG 是**反号**的
```python
noise_pred = -(pos + g * (pos - neg))    # Z-Image
# 不是教科书的
noise_pred = neg + g * (pos - neg)
```

### 5.3 caption padding 是计算的一部分
- 用**学习出来的 pad token** 填充到 `SEQ_MULTI_OF = 32` 的倍数
- **这些 pad token 是真实 attention 上下文**（不做 mask）
- `n_cap` 必须等于 `round_up(L, 32)`
- 截断会让相关系数从 1.000000 掉到 0.994
- **推论**：cond 和 uncond 的 `n_cap` 通常不同（45 token prompt → 64；空负向 → 32），
  所以**每个分支需要各自的 caption RoPE 和 pad mask**

### 5.4 fp16 不可用，必须 bf16
- 残差流达 **~6.7e3**；`feed_forward.w2` / `attention.to_out` 输出达 **3.1e5**
- fp16 上限 65504 → **第 2 步全 NaN**（移植方在 LiteRT 和 Core AI 上两次撞到）
- bf16 动态范围同 fp32 → 可用
- **好消息**：AMD 的 SD3/FLUX 策略全是 bf16（`*_bfp.yaml`），方向一致

---

## 6. ⚠️ 已识别的导出级阻塞点：complex RoPE

**这是开始实现前必须先解决的第一件事。**

Z-Image 参考实现（diffusers）的 RoPE 用 `view_as_complex`：

```python
# 参考实现
q_complex = torch.view_as_complex(q.float().reshape(..., -1, 2))
q_rot = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
```

**ONNX 没有复数类型。** 移植报告的解法：

> The only change: the attention processor's `view_as_complex` RoPE swapped for a
> **bit-exact real interleaved form** fed precomputed cos/sin.

**所以导出前必须改模型代码**，把复数 RoPE 换成实数交错形式。这是 `export_dit.py` 阶段
的工作，不是 `onnx_utils` 阶段。

### 6.1 ✅ 已解决并数值验证（实现见 `zimage-port/rope_real.py`）

原实现（`diffusers/models/transformers/transformer_z_image.py:113-118`）：
```python
x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
freqs_cis = freqs_cis.unsqueeze(2)
x_out = torch.view_as_real(x * freqs_cis).flatten(3)
return x_out.type_as(x_in)
```

**实测确认的形状链**（不是推断 —— 我最初的推断错了，被测试抓出来）：
```
x_in        (B, S, H, D)          # attention 布局, D = head_dim = 128
freqs_cis   (B, S, D//2)          # complex64
  .unsqueeze(2) → (B, S, 1, D//2)
complex x   (B, S, H, D//2)
product     (B, S, H, D//2)
return      (B, S, H, D)          # 与输入同形
```
**没有 permute。** RoPE 直接在 attention 布局上按头独立应用。

**Rotary 参数**（`z_image.hpp:256-258` + 官方 config）：
```cpp
int theta = 256;                        // ⚠️ 不是标准的 10000
std::vector<int> axes_dim = {32, 48, 48};
```
`axes_lens = [1536, 512, 512]` 是**位置索引上界**，不是频率维度。

**实数改写（final）**：
```python
xf    = x.float().reshape(B, S, H, D//2, 2)   # 相邻对 = (实, 虚)
x_even, x_odd = xf[..., 0], xf[..., 1]
c     = cos.unsqueeze(0).unsqueeze(2)          # (1, S, 1, D//2)
s     = sin.unsqueeze(0).unsqueeze(2)
out_even = x_even * c - x_odd * s
out_odd  = x_even * s + x_odd * c
out   = torch.stack([out_even, out_odd], -1).flatten(-2).type_as(x)
```

**验证结果（32 项，exit 0）**：

| 检查 | 结果 |
|---|---|
| cos/sin 表 vs diffusers `freqs_cis.real/.imag` | ✅ <1e-6 |
| fp32 端到端 | ✅ **max err < 1e-6** |
| bf16 端到端 | ✅ **22/3,932,160 差异（5.6e-6），max 2.88 ULP** |
| **负对照：rotate-half 必须不匹配** | ✅ 误差 **6.288** vs 交错版 4.8e-7 |

**负对照是关键** —— 它证明交错语义正确。用 Llama 的 rotate-half 约定会错开 6.288。
这个检查排除了「公式看着对但语义错」这一整类静默 bug。

**ONNX 算子映射**（全部在 AMD 支持表内）：

| 步骤 | 算子 |
|---|---|
| 拆实虚 | `Reshape` + `Split`/`Slice` |
| 预计算 cos/sin | 常量 initializer 或图输入 |
| 旋转 | `Mul` + `Sub` + `Add` |
| 交错合并 | `Unsqueeze` + `Concat` + `Reshape` |
| 精度还原 | `Cast` → bf16 |

**而且正好对上现有 pass**：`flux1.replace_rope` 内部就是
`add_position_ids` / `add_rope_cache_concat` / `add_rotary_embedding`
—— AMD 已有「把预计算 RoPE 表接进图」的机制，Z-Image 复用同一模式。

---

## 7. 关键未知（按重要性排序）

| # | 未知 | 影响 | 如何验证 |
|---|---|---|---|
| 1 | 导出后节点命名能否被现有切片器正则覆盖 | **决定是否需要写新 pass** | 实际导出一次 |
| 2 | int8_convrot 能否被 Quark/Vitis AI 接受 | 决定 int8 路线可行性 | 查 Quark 文档 / 试 |
| 3 | XDNA2 是否有高效 I8×I8→I32 GEMM | 决定 int8 是否真能加速 | 无现成办法，需实测 |
| 4 | `sd3_mmdit_no_dd_bfp.yaml` 的算子 pass 能否匹配 Z-Image 图 | 决定 offload 比例 | `onnx_utils report dd_offload` |
| 5 | 复用 `FLUX_TRANSFORMER` 作 model_type 是否安全 | 影响是否要动 C++ | 无现成办法 |

---

## 8. 执行顺序

### 阶段 0：环境（未做）
```cmd
pip install hf_transfer
conda create -n convert_onnx python=3.10
conda activate convert_onnx
pip install torch==2.6.0 diffusers==0.32.2 onnx==1.17.0 safetensors onnxconverter-common
```

### 阶段 1：先拿 DiT 基线（**未做，零风险，最该先做**）
```cmd
conda activate ryzen-ai-1.8.0
cd /d D:\ryzenai\GenAI-SD
python scripts\run_pipelines.py --benchmark --pipelines flux1_schnell
```
FLUX.1-schnell 是 DiT + bf16 + 4 步，是 Z-Image 最近亲。**它的每步耗时决定 Z-Image 的性能上限。**

### 阶段 2：导出 ONNX（需先解决 §6 的 complex RoPE）
三个组件：DiT / Qwen3-4B 编码器 / VAE

### 阶段 3：注册 + 预处理模块
1. `strategy_builder/sd/__init__.py` 加 `SD_ZIMAGE = "zimage"`
2. `strategy_builder/sd/builder.py` 三处加分支
3. 新建 `model_preprocessing/zimage.py`（模板：`mmdit.py`，改 `finalize()`）

### 阶段 4：分区策略
新建 `data/partition_strategies/zimage_transformer_bfp.yaml`：
```yaml
inherit_passes_before: !include [sd3_mmdit_no_dd_bfp.yaml]
passes:
    - dd.optimize_toposort
    - dd.dynamic_zimage_nodes_to_dd      # ← 可能需新写
```

### 阶段 5：验证
```cmd
onnx_utils report dd_offload <path>\optimized.onnx --summarize
```
目标 offload > 80%

---

## 9. 已有的可复用资产

| Z-Image 需要 | 现成资产 |
|---|---|
| VAE 分区 | `sd3_vae.yaml`（FLUX 的 VAE 已在用） |
| Qwen3 文本编码器上 NPU 的先例 | `src/trigger/StableDiffusionFlux2KleinONNXPipelineTrigger.py:55`<br>`qwen3_text_encoder_prompt_embeds_matmulnbits.onnx` |
| LLM 量化上 NPU | `hybrid_llm_npu_matmulnbits.yaml` |
| 大模型多子图切片参考 | `dynamic_flux_nodes_to_dd.py`（FLUX 12.6GB 超 4GB 限制的完整解法） |

**XRT 单缓冲区上限 = 4 GB**（`dynamic_flux_nodes_to_dd.py` 文档字符串明写）。
Z-Image bf16 12GB → 需 ≥3 子图；int8 6GB → 需 2 子图。

---

## 10. 版本信息

- Ryzen AI Software: **1.8.0**
- `ryzenai_onnx_utils`: 1.8.0
- `onnxruntime-vitisai`: 1.27.0
- 目标芯片: STX (Strix Point), Radeon 880M, XDNA2
- overlay: `stx_SD_unified_fastpm_2x4x4`
