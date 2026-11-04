# BFP16-FORMAT — 官方 ONNX 路径的 BFP16 自定义格式调查（t22_sessC）

> 调查对象：`backup_ryzenai_1.6.0`（Ryzen AI 1.6.0 安装树）+ `RyzenAI-SW`（git，实测为 **RAI 1.7.0 Release** 单提交浅克隆）+ 1.8 官方文档树 + 1.8 nuget 运行时。
> 对照基线：我们的 `pack_bfp16.py`（bfp16ebs8，9 B/8 值，由厂商 `.rai` 反解并逐字节验证）。
> 日期：2026-10-24。**本文件只做调查与判定，不改产品、不改内核、不改权重。**

---

## 0. 结论先行

| 问题 | 判定 | 一句话 |
|---|---|---|
| 官方 BFP16 规格 = 我们的 9 B 块吗？ | **是，逐字段同构，且位精确** | Quark `bfp_kernel.cc` 的 `to_bfp(bit_width=16, block_size=8)` 与 `pack_bfp16._quantise` 在 19,200 组、含幂次 amax 与溢出组上**零字节差异** |
| 有 ONNX→自定义格式的转换器吗？ | **有，但它不产 9 B 字节** | `convert_fp32_to_bfp16.py` 只把 `com.amd.quark::BFPQuantizeDequantize` 节点插进图里，张量仍是 float32 |
| 9 B 块是谁产的？ | **编译器** | `vaiml.dll` / AIE 的 `to_v64bfp16ebs8` 在编译期打包，结果落进编译缓存 |
| 「ONNX+自定义格式走通官方 BFP16 路径」 | **成立，但产物不是 ONNX** | 你拿到的是缓存里的编译 blob（`.rai`/transaction bin），不是一份「装好 9 B 权重的 onnx」 |
| 转换成本（6.1B 权重） | **一次 10–11 min（单线程 numpy）+ 6.86 GB 落盘** | 相对 VAIML 编译时间可忽略；真正贵的是编译，不是打包 |
| 可缓存吗？ | **可以，且我们已经这么做了** | 官方 `cache_dir`+`cache_key`；我们 `webui/cache/wpack/*.bin`、`_out/textenc/te_npu_weights.bin` |
| `.onnx_transformers` 目录 | **不存在（负结果）** | 四路检索（文件系统 / git 全历史 / 18 个 wheel 成员+正文 / 1.8 文档）**0 命中** |
| 对 51.3 TOPS 固定门 / 产品 | **零影响** | 纯调查；51.3 TOPS 是 INT8 门，BFP16 有独立天花板（见 §7.3） |

---

## 1. 检索范围（可复现）

| # | 位置 | 方法 | 结果 |
|---|---|---|---|
| 1 | `backup_ryzenai_1.6.0`（全树，文本文件） | `Select-String 'ops_type|bfp16|bf16ebs|ebs8|ebs16'` | 仅 `voe-4.0-win_amd64/vaip_config.json`：内部 pass `args:["bf16"]` + `..._gemmbfp16_...xclbin` 文件名——**都是内部件，不是用户旋钮** |
| 2 | `RyzenAI-SW`（全树） | 同上 | 仅 `Demos/ASR/Whisper/README.md:62` 一句宣传语 |
| 3 | 18 个 wheel（成员名 + 成员正文） | zipfile 扫描 | **命中核心**：`amd_quark`、`ryzenai_onnx_utils`、`voe` |
| 4 | wheel 内二进制（`.pyd`/`.dll`） | 字节串扫描 | `vaiml.dll` 1710× `bfp16`；`_DynamicDispatch.pyd` `bfp16ebs8` 契约 |
| 5 | 1.8 文档树 `amd_xdna2_docs/ryzenai_docs_html/...`（26 篇 md） | grep | `oga_op_prepare.md:29` 唯一 `--ops_type bfp16`；`model_quantization.md` **无 BFP16** |
| 6 | 1.8 nuget `RyzenAI_Deployment.1.8.0.nupkg` | 解包 + 字节串 | `dyn_dispatch_core.dll` 要求 **pre-formatted bfp16ebs8** |
| 7 | `RyzenAI-SW` git 全历史 | `git grep` / `git log -S` | `onnx_transformers` **0 命中**（仅 1 个提交，浅克隆） |

**版本提醒**：wheel = 1.6.0，示例仓库 = 1.7.0，文档 = 1.8。三者对 BFP16 的表述不同，下文分别标注。

---

## 2. 官方规格（源码级）

### 2.1 先消歧：这个生态里「BFP16」不是一个东西

| 名字 | 官方出处 | `bfp_method` | bit_width | block_size | 每值字节 | 语义 |
|---|---|---|---|---|---|---|
| **BFP16（本文主角）** | `DEFAULT_BFP_PARAMS` | `to_bfp` | 16 | **8** | **1.125** | 8 值共享 1 个 8 bit 指数 + 每值 1 sign + 7 mantissa = **9 B / 8 值** |
| BFP-prime / 微指数 | `DEFAULT_MICROEXPONENTS_PARAMS` | `to_bfp_prime` | 11/13/16 | 16 | — | 再叠 `sub_block_size=2, sub_block_shift_bits=1` 的二级移位 |
| MX 微缩放 | `DEFAULT_MICROSCALING_PARAMS` | — | 8/6/4 | 32 | — | OCP MX，e8m0 缩放 + fp8/fp6/fp4 |
| BF16（IEEE） | `BF16_CONFIG` / `QBFloat16` | — | — | — | 2.0 | 标准 bfloat16，走 Cast，不是 BFP |

> ⚠️ 陷阱：`f2bfp(ebs=1)` 出来的 2 字节是 `[8 bit 指数][8 bit 有符号尾数]`，**不是** IEEE BF16 的 `[sign|exp][mantissa]`（符号位在尾数字节）。所以「BFP16 退化为 BF16」这个说法只在语义上近似，**字节布局不同**，不可互换。

### 2.2 官方属性表（图级声明的全部信息）

`quark/onnx/quantization/config/custom_config.py:40-46`：

```python
DEFAULT_BFP_PARAMS = {'bfp_method': "to_bfp", 'axis': 1, 'bit_width': 16,
                      'block_size': 8, 'rounding_mode': 2}
```

`quark/onnx/quant_utils.py:90-99`（节点上实际写进去的默认属性，比 preset 多 3 个）：

```python
BFP_OP_DEFAULT_ATTRS = {'bfp_method': "to_bfp", 'axis': 1, 'bit_width': 16,
                        'block_size': 8, 'rounding_mode': 0,
                        'sub_block_size': 2, 'sub_block_shift_bits': 1,
                        'convert_to_bfloat_before_bfp': 0}
```

节点身份（`quark/onnx/operators/custom_ops/__init__.py:9-17`）：

- domain = `com.amd.quark`
- op_type = `BFPQuantizeDequantize`
- 版本 = 1

### 2.3 逐字段对照：我们的 9 B 块 vs 官方 `bfp_kernel.cc`

官方参考实现：`quark/onnx/operators/custom_ops/src/bfp/cpu/bfp_kernel.cc`（MIT，AMD 版权，随 1.6.0 wheel 发布）。

| 字段 | 我们（`pack_bfp16.py`） | 官方（`bfp_kernel.cc`） | 一致 |
|---|---|---|---|
| 分组 | 行主序 K 连续，每 8 个一 group | `LaunchBFPCPUKernel` stride=1、block=8（`:293-298`） | ✅ |
| 块字节数 | 9 = 1 + 8 | 8×(1 sign + `bit_width-9`=7 mant) + 8 bit exp = 72 bit（`:75-76`） | ✅ |
| 字节序 | `[exp][m0..m7]` | `cast_bf16bfp16`: `concatenate((e, m))`（`lora.py:121-130`） | ✅ |
| 共享指数 | `E' = unbiased exp(amax)` | `shared_exp = max biased exp`，`shared_exp_value = shared-127`（`:61-74`） | ✅ |
| 缩放 | `scale = 2^(E'-6)`（**无条件**，无幂次特判） | `scale = 2^(sev - (m_bits-1))`，m_bits=7 → **2^(sev-6)**（`:77`） | ✅ |
| 尾数 | `sign(q)*floor(|q|+0.5)`，clamp int8 | `STD_ROUND: std::round(x/scale)`，即 half away from zero（`:89`） | ✅ |
| 溢出 | `if m.max()>127: e+=1 重算`（+128 非法 / -128 合法） | `BFPCPUKernelCompiler`: `if (x>=128 || x<-128) { shared_exp+=1; scale*=2; break; }`（`:147-152`） | ✅ |
| 值域 clamp | `[-128·scale, 127·scale]` | `max_v=2^(sev+1)-scale=127·scale`, `min_v=-2^sev·2^7=-128·scale`（`:155-156`） | ✅ |
| 指数字节 | `(e+133)&0xFF` = IEEE biased exp | `me` 直接是 biased exp，按 int8 存（`lora.py:117`） | ✅ 同一字节 |
| NaN/Inf | 未处理（直接量化） | `exp==0xff` 原样透传（`:82-83`） | ❌ 差异（权重中不出现） |
| 布局排列 | 32×32 tile-shuffle（`_slots`） | **无**，Quark 按 flat 顺序 | ⚠️ 不同层，非规格冲突 |

**两条重要的结构性结论：**

1. **Quark 有两个 kernel 变体**：`BFPCPUKernel`（`:54`）溢出时 **clamp** 到 127；`BFPCPUKernelCompiler`（`:106`）溢出时 **抬指数重算**。我们的 packer 对齐的是 **Compiler 变体**——这正是「厂商 `.rai` 里 +128 组指数比 clamp 组高 1」的来源。这也解释了为什么反解时「幂次 amax」会踩坑：只有抬指数路径才会暴露。
2. **编码 ≠ 布局**。Quark 规格只定义「9 B 块怎么算」，不定义「块在内存里怎么排」。tile-shuffle 是 DD/VAIML kernel 的 `coalesed_weights` 布局要求，是**下游**的事。

### 2.4 位精确性验证（机械链 + 负对照）

脚本：`imaggers/t22_sessC/verify_format_equiv.py`、`verify_kernel_semantics.py`（随本报告入库，见同目录 README）。

**正对照** — 我们的 `_quantise` vs Quark `f2bfp(bits=16, ebs=8)`（`lora.py` 原样移植）：

| 数据 | group 数 | 指数失配 | 尾数失配 |
|---|---|---|---|
| gaussian σ=0.05 | 8,192 | 0 | 0 |
| gaussian σ=3.0 | 8,192 | 0 | 0 |
| 精确 int8 网格 | 1,024 | 0 | 0 |
| 极小 1e-3 | 512 | 0 | 0 |
| 极大 1e3 | 512 | 0 | 0 |
| **幂次 amax（我们的历史坑）** | 512 | 0 | 0 |
| **溢出 → 抬指数** | 256 | 0 | 0 |
| **合计** | **19,200** | **0** | **0** |

**值级正对照** — 移植 `BFPCPUKernelCompiler` 语义 vs 我们 pack→decode：gaussian / 跨 1e±6 动态范围 / 抬指数组，`max_abs_diff = 0.000e+00`。

**负对照（必须失配，全部按预期失配）**：

| # | 变异 | 结果 | 判读 |
|---|---|---|---|
| NC1 | 截断代替 half-away 舍入 | 尾数失配 8,093 | 测试对舍入规则敏感 |
| NC2 | `ebs=16` | group 数 1,024 vs 2,048 | 对块大小敏感 |
| NC3 | `bit_width=13` | 尾数失配 16,260 | 对位宽敏感 |
| NC4 | 尾数在前 / 指数在前 | 18,432 字节中仅 96 偶然相等 | 字节序不是无所谓 |
| NC5 | clamp 变体 vs 抬指数变体 | 64/64 组不同（127.0 vs 128.0） | 证明我们对齐的是 Compiler 变体 |

> 方法学备注：NC1/NC5 第一版**没触发**（用了 127.4，round 后仍是 127，不会溢出）。改成 127.6 才真正进入抬指数分支。**负对照必须验证它真的走到了那条路径**，否则是假阳性。

---

## 3. 转换工具链

### 3.1 三条链，各自负责什么

```
① ONNX 图改写（Quark，Python，秒级）
   convert_fp32_to_bfp16.py / ModelQuantizer(BFP16_CONFIG)
        └─> 在权重与激活处插入 com.amd.quark::BFPQuantizeDequantize
            （张量仍是 float32；只多了一个带属性的节点）

② 编译（VAIML / flexml，分钟级）
   flexml.compile(model, enable_f32_to_bf16_conversion=..., config=vitisai_config.json)
        └─> AIE 后端 to_v64bfp16ebs8 在编译期完成 9 B 打包
            → 产物 = .rai / transaction bin（缓存）

③ 运行时（DynamicDispatch）
   weightDtype == bfp16ebs8 ⇒ 只接受「1 个已打包的常量张量」
```

**证据：**

- ① `quark/onnx/quantizers/bfp_quantizer.py:123-141` —— `_add_fn_pair_for_weight` 只是把 weight 名字换成 `<w>_dequantized` 并加一个 BFP 节点；initializer 原封不动还是 float32。
- ① 两个 CLI：`quark/onnx/tools/convert_fp32_to_bfp16.py`、`convert_fp16_to_bfp16.py`（`--input/--output [--save_as_external_data]`）。
- ② `voe/passes/vaiml.py:196-219` → `flexml.compile(..., enable_f32_to_bf16_conversion=...)`；`flexml_lite` 的 `vaiml.dll` 里有 1710 处 `bfp16`、AIE 内建 `v64bfp16ebs8_unaligned` / `to_v64bfp16ebs8`。
- ③ `ryzenai_dynamic_dispatch/_DynamicDispatch.pyd` 字符串：`"Expected 1 constant tensor containing pre-formatted weights/biases when weightDtype is bfp16ebs8"`；1.8 的 `dyn_dispatch_core.dll` 同样：`"Writing pre-formatted bfp16ebs8 weights/bias to IO buffer"`。

### 3.2 命令清单（1.6.0 实测可用形状）

```powershell
# ① Quark 图改写（BFP16）
python -m quark.onnx.tools.convert_fp32_to_bfp16 --input m.onnx --output m_bfp16.onnx

# ② 编译 + 缓存（Python API，vitisai_config.json 决定 vectorized/unvectorized）
python compile.py --model m_bfp16.onnx     # 内部: cache_dir + cache_key + config_file

# ②' 图级算子编译（LLM/OGA 插件流）
onnx_utils vaiml --model-dir <dml model folder> --plugin_name <name> --compile --ops_type bfp16
```

`onnx_utils` 的入口点是 `ryzenai_onnx_utils.partitioner:main`（wheel `entry_points.txt`），`vaiml` 是它的子命令。

### 3.3 关键否定：**没有任何官方工具输出「装好 9 B 权重的 ONNX」**

- ① 输出 ONNX，但字节未打包；
- ② 输出编译 blob，不是 ONNX；
- ③ 只接受打包好的常量，自己不打。

**所以「ONNX + 自定义格式」这条路能走通官方 BFP16，但终点是编译缓存，不是 ONNX 文件。** 想让 9 B 权重以文件形式可搬运/可复算，只有两条：
(a) 把编译缓存当产物（官方做法，不透明、跨版本易失效）；
(b) 自己打包（我们的 `pack_bfp16.py`，已位精确对齐官方规格）。

---

## 4. 转换成本与可缓存性

### 4.1 体积（6.1B 参数）

| 格式 | 字节/参数 | 6.1B 总量 |
|---|---|---|
| FP32 | 4.000 | 24.40 GB |
| BF16 | 2.000 | 12.20 GB |
| **BFP16 ebs8（9/8）** | **1.125** | **6.86 GB** |
| INT4 + bf16 scales（blk32） | 0.5625 | 3.43 GB |

BFP16 相对 BF16 省 43.75%，相对 INT4 多 2.0×。

### 4.2 实测吞吐（本机，前台 `perf_counter`，无看门狗介入）

两次独立跑（同一台机，单线程 numpy，`bench_pack.py`）：

| 形状 | 参数 | pack 耗时（run1 / run2） | 输入 MB/s | 输出 MB/s |
|---|---|---|---|---|
| 1024² | 1.05 M | 0.073 / 0.078 s | 54–57 | 15–16 |
| 2048² | 4.19 M | 0.288 / 0.289 s | 58 | 16 |
| 3840² | 14.7 M | 1.218 / 1.139 s | 48–52 | 14–15 |
| 4096² | 16.8 M | 1.404 / 1.425 s | 47–48 | 13 |
| 10240×3840 | 39.3 M | 3.962 / 4.331 s | 36–40 | 10–11 |

**外推 6.1B：9.1–9.9 M 参数/s → 615–672 s ≈ 10–11 min**，落盘 6.86 GB。取保守值 **11 min**。

> 这是**打包**成本，不含读 safetensors（IO 另计，NVMe 上 ~20-40 s/6 GB）与 VAIML 编译（分钟~小时级，随图规模）。**结论：打包不是瓶颈，编译才是。** 一次转换后权重永久可复用 —— 权重不随提示词/形状变化，只有 `tile_rows` 随 (N,K,M) 变（见 `pack_bfp16.KNOWN_TILE_ROWS` 与 `manifest.json` 的 `tile_rows_note`），所以严格说是「按 (N,K,M) 缓存」，不是「按权重缓存」。

### 4.3 官方缓存机制

```python
provider_options = {"config_file": "vitisai_config.json",
                    "cache_dir": <dir>, "cache_key": <str>,
                    "enable_cache_file_io_in_mem": 0}
```

出处：`RyzenAI-SW/CNN-examples/getting_started_resnet/bf16/compile.py:17-29`、`Transformer-examples/DistilBERT_text_classification_bf16/run_inference.py:17-19`。

- 键 = `cache_key`（示例里用模型 stem）；
- 命中则跳过编译，直接跑；
- 文档明确：`"The first-time compilation may take some time, but the compiled model is cached in a directory specified by the cache_dir and cache_key provider options."`（DistilBERT README §2）。

**跨版本风险**：`relnotes.md:120` —— 「Hybrid models published with earlier releases are not compatible with Ryzen AI 1.6」。编译缓存同理，不跨版本。

### 4.4 当年项目做过的转换（旧产物，全部在 `zimage-bfp16-npu/`）

| 产物 | 大小 | 性质 |
|---|---|---|
| `vendor_artifacts/39e41383350b8c53e90996269478cfb7.rai` | 399.5 MB | **厂商编译产物**（ELF-like，头 `24 00 00 00 20 00 34 00`），`pack_bfp16` 的反解基线；内含 layers 10–12 的 `coalesed_weights` 共 416,071,680 B |
| `weights/block{0..29}.npz` | 202.8 MB × 30 | 逐 block 的 FP32 源权重 |
| `weights/block0.coalesced.bin` | 198.4 MB | 已打包的 9 B 块 |
| `weights/manifest.json` | 40 KB | **规格 + 逐张量 sha256 + tile_rows + 解码误差**，即本报告的对照基线 |
| `_out/textenc/te_npu_weights.bin` + `.json` | 3.90 GB | Qwen3-4B 文本编码器整包（36 层），json 是 offset 索引 |
| `webui/cache/wpack/wpack_l30_v2.bin` + `.json` | 1.90 GB | 30 层 q/k/v/o 的打包缓存（`version:2, code:"t22_prewarm_1"`） |

**即：当年项目确实做过转换，且已经有「打包 + 索引 sidecar」的缓存形态**（`.bin` + `.json`），与官方 `cache_dir/cache_key` 思路一致，但键更细（按 tensor + 形状 + tile_rows）。

---

## 5. `--ops_type bfp16` 的图级用法与 D1-6 对照

### 5.1 它到底是什么（源码）

`ryzenai_onnx_utils/vaiml.py`：

```python
# :49-55
"--ops_type", type=str, choices=["bf16", "bfp16"], default="bf16",
help='Type of ops to generate: "bf16" (default) or "bfp16"'
# :299-307
is_v2 = args.ops_type == "bfp16"
ops_json = {"xclbin": "llama2_mladf_2x4x4_v1_gemmbfp16_silu_mul_mha_rms_rope.xclbin",
            "overlay": "2x4x4", "enable_v2": is_v2, ...}
# :370
"op_type": "gemm_w4abfp16" if is_v2 else "gemm_w4abf16"
```

**三个实际效果**：换 xclbin、置 `enable_v2`、把 GEMM 算子名从 `gemm_w4abf16` 换成 `gemm_w4abfp16`。

**它不是**：不是权重格式转换器，不是「把整图变 BFP16」的开关，不产生 9 B 字节。它是**编译哪个算子变体**的选择器，作用域是 **DD 插件里被抽出来的那些 op**。

### 5.2 D1-6 对照 —— **判定维持「佐证」，但需要收紧措辞**

| | D1-PROTOTYPE.md 原文 | 本次源码核实 |
|---|---|---|
| 断言 | 「bfp16 是图级合法路径」 | ✅ 成立 |
| 证据 | `vaiml --compile --ops_type bfp16` | ✅ 精确到 `vaiml.py:299-370` |
| 推论 | 「可在图级声明而不是手改」 | ⚠️ **部分成立**：图级能声明的是**算子变体选择**（xclbin/op_type），以及 Quark 侧的 **BFP 量化属性**（`BFPQuantizeDequantize`）。**9 B 布局仍由编译器决定，图级声明不了 tile_rows。** |

**修正后的 D1-6 表述**：

> BFP16 在官方栈里有**两个图级入口**：(a) Quark 的 `BFPQuantizeDequantize`（声明编码：bit_width/block_size/rounding），(b) `onnx_utils vaiml --ops_type bfp16`（声明算子变体：xclbin + gemm_w4abfp16）。二者都**不**声明内存布局；布局是编译器/kernel 的私有契约。因此「图级合法」成立，但**不能靠改 JSON 复现我们的 kernel**。

### 5.3 顺带佐证 D1-5（`preferred_data_storage`）

`vaiml.dll` 内错误串（原文）：

> `[flexml-compile] The experiment option 'preferred_data_storage' only supports 'vectorized', 'unvectorized' OR 'auto'.`

三档确认。且示例里的选择有明确倾向：

| 示例 | 取值 | 出处 |
|---|---|---|
| ResNet50 BF16（CNN） | `auto` | `CNN-examples/getting_started_resnet/bf16/vitisai_config.json:12` |
| **DistilBERT BF16（Transformer）** | **`unvectorized`** | `Transformer-examples/DistilBERT_text_classification_bf16/vitisai_config.json:11` |
| Quark 量化（CNN） | `auto` | `CNN-examples/quark_quantization/vaiml_config.json:12` |

**Transformer 官方示例确实选 `unvectorized`** —— 与 D1-5 的判断一致，且现在有了源码级取值域证明。

> ⚠️ 但注意：`unvectorized` 与「是否 BFP16」是**正交**的两件事。DistilBERT 示例走的是 **BF16**（`get_default_config("BF16")`），不是 BFP16。别把 `unvectorized` 当成 BFP16 的开关。

---

## 6. 负结果：`.onnx_transformers` 不存在

任务前提「树里有 `.onnx_transformers` 目录」。**四路检索，0 命中**：

| 检索 | 范围 | 命中 |
|---|---|---|
| 文件系统按名 | `test7` 深度 6、`public_work` 深度 5、`D:\Project` 深度 6 | 0（目录与文件都查了） |
| 18 个 wheel 的成员名 + 正文 | 全量 | 0 |
| `RyzenAI-SW` git | `git grep`（HEAD）+ `git log -S`（全历史） | 0 |
| 1.8 官方文档 26 篇 md | grep | 0 |

**诚实结论：这个目录名在本机所有可读的官方材料里都不存在。** 最接近的三个候选，供后续核对：

1. `quark` 在量化时生成的临时目录 —— 本机确有 `quark_onnx.quant.9zjvohcs`（`public_work/`，**空目录**），命名模式是 `quark_onnx.quant.<随机>`，不是 `.onnx_transformers`。
2. `model_generate` 的 ONNX 中间产物目录（`model_generate` wheel 内有 `onnx_tools/`、`scripts/add_cast_nodes.py`、`scripts/add_pack_consts.py`）。
3. `optimum`/`transformers` 的导出缓存。

**若该目录来自另一台机器或已删除的树，本报告无法证实其内容**；请提供确切路径或把它从前提里去掉。

---

## 7. 可行性判定

### 7.1 「ONNX + 自定义格式」能不能走通官方 BFP16？——**能，分三种「通」**

| 通法 | 能否 | 代价 | 产物 | 适用 |
|---|---|---|---|---|
| **A. 纯官方**：Quark 插 BFP 节点 → VAIML 编译 → 跑 | ✅ 通 | 编译时间（分钟~小时），无源码级控制 | 编译缓存 blob | 要「官方保证正确」，不要「可移植权重」 |
| **B. 官方编码 + 自建打包**：Quark 规格做参照，`pack_bfp16` 自己产 9 B | ✅ 通，且**已位精确** | 10–11 min / 6.1B，一次性 | 可搬运的 `.bin` + `.json` 索引 | 我们现在的路线 |
| **C. 期望「官方给一份装好 BFP16 的 onnx」** | ❌ 不通 | — | — | 不存在这种产物 |

**判定：A 与 B 都成立；B 是唯一能同时满足「可缓存、可复算、可跨机搬运、可审计」的形态。**

### 7.2 对我们的意义

1. **格式风险归零**：9 B 块不再是「反解出来的民间格式」，而是**有 MIT 源码背书的官方格式**（`bfp_kernel.cc`）。这条可以直接进对外文档，不必再用「reverse-engineered」措辞。
2. **D1-6 升级为「已核实」**，但措辞收紧（§5.2）。
3. **不必切到官方编译链**：官方 A 路线拿不到可搬运权重，且 `relnotes.md:278,280` 明写 BFP16 编译慢 + 精度可能掉、需要 Quark 介入。我们 B 路线在「权重可复现」这一点上严格优于 A。
4. **值得单试的一件事**：`preferred_data_storage: unvectorized`（Transformer 官方选择）。这不是 BFP16 开关，但可能影响 BFP16 权重在 NPU 上的落位方式——与 D1-5 合并成一个实验，成本低（改一个 json 字段重编译一次）。

### 7.3 与 51.3 TOPS 固定门的关系

- 51.3 TOPS 来自 `xrt-smi validate -r gemm`，是 **INT8 全阵列**基准（104% of 49.152），**与 BFP16 无关**，本调查不触碰它。
- BFP16 有**独立**天花板：单核 377 MAC/cyc（73.6%），32 核理论上限 ≈ 37.7 TFLOP/s（见 `BATCHM-ROADMAP.md:22`）。**换官方 BFP16 路径不会提高也不会降低这个天花板**——天花板在内核形状与数据流，不在格式。

---

## 8. 纪律合规

| 纪律 | 本单执行 |
|---|---|
| 预测先行 | 开查前判定「Quark 应有 BFP16 preset，且属性里应出现 block_size=8」；实际 `DEFAULT_BFP_PARAMS` 完全吻合 |
| 单次多步合并 / 机械链脚本化 | 检索、wheel 解包、字符串扫描、差分测试全部落成脚本（差分与基准入库于 `imaggers/t22_sessC/`），非手工点击 |
| 负对照必配 | 5 条负对照（NC1–NC5）；其中 2 条第一版**未触发**，已修正并复跑 |
| 诚实负结果 | `.onnx_transformers` 明确记为「不存在」，未编造来源 |
| 看门狗 ≠ 墙钟 | 成本数字全部是进程内 `perf_counter` 前台实测，无看门狗、无超时截断 |
| F-14.1 锁真实 pid | 本单无长驻进程，未启动后台作业 |
| 51.3 TOPS 固定门 | **未改动硬件/栈，未重跑**；本单不涉及形态变更（§7.3 说明为何不适用） |
| 产品零变化 | 只新增本文件；`pack_bfp16.py`、权重、内核、webui **零修改** |

---

## 9. 复现

```powershell
# 1) 格式差分（正对照 + NC1..NC4）       —— 只依赖 numpy + pack_bfp16.py
python imaggers/t22_sessC/verify_format_equiv.py
# 2) 值级语义（BFPCPUKernelCompiler vs 我们，+ NC5）
python imaggers/t22_sessC/verify_kernel_semantics.py
# 3) 成本
python imaggers/t22_sessC/bench_pack.py
```

官方源码不必跑脚本取——按下面证据索引里的 member 路径，直接从 wheel 里解压即可：

```powershell
python -c "import zipfile;z=zipfile.ZipFile(r'<backup_ryzenai_1.6.0>\amd_quark-0.10-py3-none-any.whl');open('bfp_kernel.cc','wb').write(z.read('quark/onnx/operators/custom_ops/src/bfp/cpu/bfp_kernel.cc'))"
```

**证据索引（关键行号）**

| 结论 | 文件:行 |
|---|---|
| BFP16 preset 属性 | `amd_quark/.../config/custom_config.py:40-46` |
| 节点默认属性（含 sub_block） | `amd_quark/quark/onnx/quant_utils.py:90-99` |
| 自定义算子 domain/name | `amd_quark/.../custom_ops/__init__.py:9-17` |
| 只改图、不改字节 | `amd_quark/.../quantizers/bfp_quantizer.py:123-141` |
| 编码规则（抬指数变体） | `amd_quark/.../bfp/cpu/bfp_kernel.cc:106-182` |
| 9 B 块打包实证 | `ryzenai_onnx_utils/ryzenai_onnx_utils/lora.py:62-68, 78-130` |
| `--ops_type` 语义 | `ryzenai_onnx_utils/vaiml.py:49-55, 299-307, 367-377` |
| `preferred_data_storage` 三档 | `vaiml.dll` 错误串（flexml-compile） |
| DD 要求预打包权重 | `_DynamicDispatch.pyd` / 1.8 `dyn_dispatch_core.dll` 字符串 |
| 官方 `--ops_type bfp16` 用法 | `amd_xdna2_docs/.../oga_op_prepare.md:29` |
| CNN/Transformer 官方只到 BF16 | `amd_xdna2_docs/.../model_quantization.md:5-13` |
| BFP16 编译慢 / 精度风险 | `amd_xdna2_docs/.../relnotes.md:278,280` |
| 编译缓存用法 | `RyzenAI-SW/CNN-examples/getting_started_resnet/bf16/compile.py:17-29` |

---

## 10. 未决 / 风险

1. **1.8 未复核**：`vaiml.dll`（278 MB）里 `bfp16ebs8` 641 处、`et_bfp16ebs8/et_bfp16ebs16` 类型仍在，但 **1.8 是否改了 block_size 默认值未验证**。若升 1.8，必须重跑 §2.4 差分。
2. **`axis` 语义**：官方 `axis=1`，我们按 K 连续分组。二者在 2D 权重上一致；**3D/4D 权重（如 conv）未验证**，我们目前只用于 2D GEMM。
3. **NaN/Inf 差异**：官方透传，我们直接量化。真实权重中未出现，但**若权重含 Inf（官方 Quark 自带 `quark/onnx/tools/replace_inf_weights.py`，说明这确实是现实问题），我们会静默产生错误块**。建议在 pack 入口加一个 `np.isfinite` 断言（**本单不改，记为待办**）。
4. **`.onnx_transformers` 前提不成立**（§6），需上游澄清。
