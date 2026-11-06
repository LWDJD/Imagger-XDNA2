[English](README.md) | **简体中文**

# Imagger-XDNA2

> [!WARNING]
> **这是一个 VibeCoding 出来的 MVP，不是生产软件。**
>
> 它由 AI 智能体在人的指挥下、在几周内、在一台机器上端到端完成。它能在真实硬件上
> 生成真实图像，本文档里的每个数字都是**实测**而非估算——但它**没有测试套件、没有 CI、
> 没有安装器、没有打包、也不保证 API 稳定**。换一台机器仍有四个绝对路径需要手工修改
> （§10.6）。请把它当作一份可运行的参考实现和一份测量记录，**而不是可以直接部署的东西**。

---

**在 AMD XDNA2 NPU 上运行 Z-Image-Turbo，支持 512²–1080p 出图**，附带 WebUI、
BlockFP16 权重格式，以及一份实测性能记录。

本仓库是**干净、可运行的核心**，不是研究用的原始工作树：这里每个文件都能从 `webui.py`
或 [docs/guides/BUILD.md](docs/guides/BUILD.md) 里的构建脚本追溯到。

> **状态：** 在 Ryzen AI 9 H 365 / Radeon 880M（XDNA2 "Strix"，32 个 AIE 核）上运行。
> 已验证字节级可复现：512² × 8 步 → latents `2abf91331200d069…`，搬迁到**本仓库**后重新复现通过。

---

## 1. 这是什么

| | |
|---|---|
| 模型 | Z-Image-Turbo，30 层 DiT + 文本编码器 + VAE，蒸馏到 8 步 |
| 权重 | **BlockFP16（BFP16）**——**每 8 个值 9 字节**：1 个 scale 字节加 8 个 int8 范围的值，`value = m × 2^(byte−133)`。设备峰值 **49.152 TOPS** = 32 核 × 512 MAC/周期 × 2 × 1.5 GHz |
| 后端 | AMD XDNA2 NPU，经 XRT + MLIR-AIE 编译出的 `xclbin` 镜像 |
| 主机 | CPU 负责注意力、refiner 预阶段、norm/rope/packing |
| 分辨率 | 512²、1024²、1920×1080、1080×1920，以及其它 64 的倍数（见 §6） |
| 批处理 | `--batch B`；双链引擎下同时有两张图在流水 |

**BFP16 是 8 位格式，不是 bf16。** 见 [docs/guides/BFP16.md](docs/guides/BFP16.md)。

## 2. 实测性能（512²，8 步，交流供电）

| 配置 | 主循环 | 说明 |
|---|---:|---|
| 出厂默认 | ~90–94 s | 本文档各表用的就是这组数 |
| + `ZIMAGE_SKIP_STEPS=7` | ~78–81 s | 跳过一步，latents 相关度 0.9999748 |
| + 快速注意力（`T22_ATTN_*`） | ~81 s | 相关度 0.9975444 |
| + `--batch 2` 双链 | 70.5 s/张 | −19.7 %，与串行逐字节相同 |

墙钟时间只能分辨到 ±2 s，真正的证据是日志里的**分段计时列**（[docs/guides/MEASUREMENTS.md](docs/guides/MEASUREMENTS.md)）。
设备利用率：单链 45 %，双链 56–59 %。

## 3. 环境要求

### 硬件
* AMD Ryzen AI（XDNA2）NPU，至少 4 列。已在 Ryzen AI 9 H 365（32 核）上验证。
* 512² 需要约 13 GB 可用提交内存；1080p 需要 **17 GB**。

### 本仓库**不含**的软件

下面这些都不随仓库分发，请自行安装。

| 依赖 | 作用 | 位置 |
|---|---|---|
| **AMD NPU 驱动 + XRT** | 设备运行时 | [docs/guides/BUILD.md](docs/guides/BUILD.md) §2 |
| **MLIR-AIE / Peano** | 编译 `xclbin` 镜像 | 仅在重建内核时需要；本仓库已附带编译好的 |
| **Python 3.10 与 3.12** | 产品硬编码了两个解释器 | `webui/config.py`，可用环境变量覆盖，见 §5 |
| `torch`（CPU 版） | 主机侧数学运算 | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `numpy`、`numba` | 主机数学、融合 softmax | pip（numba 只在 3.10 上） |
| **Vulkan SDK**（可选） | 仅用于 GPU 实验 | `glslangValidator`、`glslc` |

这里的 `torch` 是 **CPU 版**。本平台没有 CUDA/HIP 后端；GPU 只能通过 `ggml`+Vulkan 或
`torch-directml` 触达，产品目前两者都未使用。见 [docs/guides/GPU.md](docs/guides/GPU.md)。

### 需要自备的产物（不进 git——太大）

| 产物 | 大小 | 获取方式 |
|---|---:|---|
| `webui/models/zimage-te-qwen3-4b-bf16.gguf` | 8.05 GB | **下载**——Qwen3-4B 文本编码器 |
| `weights.bfp16`（在 `zimage-port/` 里，仓库之外） | 6.91 GB | **构建或下载**——wpack 的唯一输入 |
| `weights/block*.npz`（30 个） | 6.38 GB | 从上游模型**构建** |
| `webui/cache/wpack/wpack_l30_v2.bin` | 1.90 GB | 从 `weights.bfp16` **构建**，约 33 s |
| `brepack_cache/brepack_30_k32_*.bin` | 5.31 GB | **构建一次**，内容寻址 |

这些就位之前，全新的 clone 无法出图。其余内容——包括编译好的 NPU 镜像（**66 个文件，约 8.8 MB**）——
都在本仓库里。见 [docs/guides/WEIGHTS.md](docs/guides/WEIGHTS.md)，那里也记录了参考机器上
上游 transformer 分片**已被删除**这件事。

## 4. 快速开始

```powershell
git clone https://github.com/LWDJD/Imagger-XDNA2
cd Imagger-XDNA2
git checkout dev

# 1. 不生成任何东西，先证明路径是通的（纯主机侧，不需要设备）
.\imaggers\repro\t22_run.ps1 -Action preflight -Size 512 -Steps 8

# 2. 出图
.\imaggers\repro\t22_run.ps1 -Size 512 -Steps 8 -Tag first
# 或者用 WebUI：
python webui.py     # 然后打开 http://127.0.0.1:4168
```

实际会用到的三条命令见 [scripts/README.md](scripts/README.md)。

## 5. 配置

硬编码的路径都可以用环境变量覆盖：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `IMAGGER_PY310` | 出厂绝对路径 | 设备 worker 的解释器 |
| `IMAGGER_PY312` | 出厂绝对路径 | 规划器的解释器 |
| `IMAGGER_XRT` | 出厂绝对路径 | XRT 驱动栈 |
| `IMAGGER_PEANO` | `_pylibs/llvm-aie/llvm-aie` | MLIR-AIE 工具链（仅构建用） |

运行时开关都是环境变量，逐条记录在 [docs/reference/KNOBS.md](docs/reference/KNOBS.md)——
共 177 个，按「能否改变输出字节」分类。重要的几个：

| 开关 | 默认 | 作用 |
|---|---|---|
| `T22_FASTGLUE` | 产品配方中为 1 | 装上 8 个主机/设备钩子 |
| `T22_BREPACK_CACHE` | 产品配方中为 1 | 映射 5.31 GB 的 B-repack 缓存；启动 47.2 → 17.2 s |
| `T22_PROJ_BLOCK` | `1` = 1056 行 | 投影分块大小；**超过 1056 行必须保持开启**（§7） |
| `ZIMAGE_SKIP_STEPS` | 未设 | 例如 `7`；−9…−12.5 s，相关度 0.9999748 |
| `T22_ATTN_REUSE` | 0 | 快速注意力；相关度 0.9975444 |
| `T22_CHAIN2B` | 0 | 双链批处理引擎；逐字节相同，−19.7 % |

## 6. 分辨率支持

NPU 镜像是按固定的行块数（RT）编译的。**固定切片的镜像无法计算超过其切片数的行**，
因此投影按镜像自身的切片数分块运行（1056 行 = RT 11）。这一点是强制校验的——
守护逻辑会拒绝运行，而不是去读陈旧内存。

| 分辨率 | 行数（M） | 路径 | 状态 |
|---|---:|---|---|
| 512×512 | 1056 | 1 次投影发射 | 字节级可复现 |
| 1024×1024 | 4128 | 4 块 | 与参照逐字节相同 |
| 1920×1080 / 1080×1920 | 8256 | 8 块 | 生成正确图像 |
| 其它 | `ceil(h/8)*ceil(w/8)+32` | 同一规则 | 每边须为 64..2048 |

## 7. 会咬人的地方

* **行数超过切片数。** 把一个 11 切片镜像一次性喂进 86 切片的行，会留下 75 切片的**陈旧**
  缓冲——图像看起来合理，但 1056 行以上是错的。这是实测到的真实故障模式，投影分块守护
  就是为它存在的。
* **B-repack 缓存是按载荷 k 块顺序内容寻址的。** 用不同顺序构建的缓存会改变最后几位的
  算术，每层放大约 1.55 倍，到第 8 步达到 4.19e-2。**不要混用缓存。**
* **两条链需要共享同一套只读上下文。** 每个 `pyxrt.kernel()` 都是独立的硬件上下文；
  一条链需要 18 个，第 ~27 个会被以 `0xc01e0009` 拒绝。两个进程、两个各自注册的 `Dev`
  都不可行——见 [docs/guides/ARCHITECTURE.md](docs/guides/ARCHITECTURE.md) §4。
* **没送达进程的开关，和一个什么都没做的开关，看起来一模一样。**
  见 [docs/guides/MEASUREMENTS.md](docs/guides/MEASUREMENTS.md) §5。
## 8. 文档

**[docs/README.md](docs/README.md) 是整个文档树的索引。**

| 目录 | 文件数 | 内容 |
|---|---:|---|
| [docs/guides/](docs/guides/) | 6 | 为本仓库撰写，随时保持最新 |
| [docs/reference/](docs/reference/) | 2 | [KNOBS.md](docs/reference/KNOBS.md)（177 个开关）与 [API.md](docs/reference/API.md) |
| [docs/reports/](docs/reports/) | 24 | 精选工程记录 + [INDEX.md](docs/reports/INDEX.md) |
| [docs/archive/](docs/archive/) | 450 | 其余全部，路径与原始树对应 |

指南：

| 文件 | 回答什么 |
|---|---|
| [ARCHITECTURE.md](docs/guides/ARCHITECTURE.md) | 20 段流水线、时间花在哪、设备模型 |
| [BUILD.md](docs/guides/BUILD.md) | 两个解释器、NPU 工具链、`xclbin` 镜像是怎么编出来的 |
| [WEIGHTS.md](docs/guides/WEIGHTS.md) | 上游模型 → `block*.npz` → `wpack` → `brepack`，**以及那个阻塞点** |
| [BFP16.md](docs/guides/BFP16.md) | 9 字节/8 值的格式，以及为什么它不是 bf16 |
| [MEASUREMENTS.md](docs/guides/MEASUREMENTS.md) | 如何测量而不自欺 |
| [GPU.md](docs/guides/GPU.md) | iGPU 的真实情况 |

## 9. 验证状态

这是一个由研究驱动的移植，所以仓库只记录**被证明过**的东西：

* 512²×8 → `2abf91331200d069…`，跨会话复现，**搬迁后从本树重新复现**
  （preflight 12/12 PASS，主循环 94.09 s，设备干净释放）。
* 1024²×1 → `af993cadfecbc419…`，与独立的另一条分块路径逐字节相同。
* 1920×1080 → 真实图像，设备峰值 12.71 GB。
* 快速注意力、跳步、双链三个分支各自带 8 步质量门，以及**被证明会失败的负对照**。

## 10. 第三方分发物（不在本仓库内）

下面没有任何东西随仓库分发。每一行给出**它是什么、确切版本、从哪里获取、必须放在哪里、
以及产品是在运行时需要它，还是仅在重建内核时需要它。**

### 10.1 模型及其衍生物

| 产物 | 大小 | 获取方式 | 目标位置 | 用途 |
|---|---:|---|---|---|
| `Tongyi-MAI/Z-Image-Turbo` transformer，3 个 safetensors 分片 | 24,619,690,888 B | `modelscope download --model Tongyi-MAI/Z-Image-Turbo --include 'transformer/*'` | `zimage-port/_models/Z-Image-Turbo/transformer/` | **重建 `weights.bfp16`** |
| 同一仓库的 `text_encoder/` 3 分片 | 8,042,086,080 B | 同上，`--include 'text_encoder/*'` | `zimage-port/_models/Z-Image-Turbo/text_encoder/` | 重建文本编码器 GGUF |
| 同一仓库的 `vae/`、`tokenizer/`、`scheduler/` | ~180 MB | 同上 | `zimage-port/_models/Z-Image-Turbo/` | **运行时**——VAE 解码、分词 |
| 文本编码器 **GGUF** `zimage-te-qwen3-4b-bf16.gguf` | 8,051,284,800 B | **社区已有发布版**——见下文 | 任意位置；用 `ZIMAGE_TE_GGUF` 指过去 | **运行时**——生成 caption |

**你不需要自己转换文本编码器。** 它就是 Qwen3-4B（36 层、hidden 2560、词表 151,936），
而加载器只读取它——逐层抽头是 DLL 的运行时特性，不是文件的属性。上游在魔搭的
`Qwen/Qwen3-4B-GGUF` 里发布了同一座塔的五个量化版本：

| 文件 | 大小 |
|---|---:|
| `Qwen3-4B-Q4_K_M.gguf` | 2.33 GiB |
| `Qwen3-4B-Q5_K_M.gguf` | 2.69 GiB |
| `Qwen3-4B-Q6_K.gguf` | 3.08 GiB |
| `Qwen3-4B-Q8_0.gguf` | 3.99 GiB |

```bash
modelscope download --model Qwen/Qwen3-4B-GGUF --include 'Qwen3-4B-Q8_0.gguf'
```

这些版本带一个本管线从不读取的 `lm_head`，代价是几百 MB，别无其他。
**但量化版不是交付基线。** 它会改变 `pe`，而 `pe` 是全部 30 个块的条件张量，
所以量化 GGUF 是一个新基线：引用它的任何数字之前，先重跑 8 步门。
交付的 8.05 GB 文件是 bf16 且裁掉了输出头（398 个张量），由
`convert_hf_to_gguf.py --outtype bf16` 从 `text_encoder/` 生成；
转换配方在 `webui/rebuild_te_gguf.ps1`。

`weights.bfp16` **可以从 transformer 分片逐字节重建**：

```bash
modelscope download --model Tongyi-MAI/Z-Image-Turbo \
    --include 'transformer/*' --local_dir ./zit
python weights/build_bfp16_weights.py \
    --index  ./zit/transformer/diffusion_pytorch_model.safetensors.index.json \
    --shards ./zit/transformer \
    --manifest weights/skeleton_manifest.json \
    --out ./weights.bfp16
sha256sum ./weights.bfp16   # 4ff5b6a66ea3b3ae5a29ecc601097d50eade61979c222f8e0c58d3a272cb8042
```

6,913,965,970 B，506 个张量（270 个 packed，236 个 raw），64 核上约 9 分钟，约 25 GB 临时空间。
细节见 [weights/README.md](weights/README.md)。

### 10.2 NPU 工具链

| 组件 | 版本 | 获取方式 | 目标位置 | 用途 |
|---|---|---|---|---|
| **XRT**——`xrt_core.dll`、`xrt_coreutil.dll`、`xrt-smi.exe`、`pyxrt.pyd` | **2.21.0**，驱动 376 | AMD Ryzen AI **1.8.0** 安装器里的 `RyzenAI_Deployment.1.8.0.nupkg` | 任意目录；`IMAGGER_XRT` 覆盖 `webui/config.py:24` | **运行时**——每次设备调用 |
| **MLIR-AIE**（`mlir_aie`） | **1.4.4** | `https://github.com/Xilinx/mlir-aie/releases/download/v1.4.4/mlir_aie-1.4.4-cp312-cp312-win_amd64.whl` | 解压到 `_pylibs/mlir_aie/` | 编译内核 |
| **Peano / llvm-aie**——`aiecc`、`aie-opt`、`aie-translate`、`bootgen` | **1.8.0** | 随 mlir_aie 的 wheel 附带 | `_pylibs/mlir_aie/bin/` | 编译内核 |

这个 wheel **不在 PyPI 上**。请解压而不是 `pip install`，并把目录本身和 `<dir>\mlir_aie\python`
都放进 `sys.path`——wheel 里带了一个指向后者的 `aie.pth`。

**硬性约束：** `pyxrt.pyd` 链接 `python310.dll`。设备侧**必须**是 Python 3.10 解释器；
3.12 无论怎么设 `sys.path` 都导入不了它。

### 10.3 文本编码器运行时

| 组件 | 版本 | 获取方式 | 目标位置 | 用途 |
|---|---|---|---|---|
| **llama.cpp** 的 Vulkan 构建——`llama.dll`、`ggml.dll`、`ggml-base.dll`、`ggml-cpu-*.dll`、`ggml-vulkan.dll` | **llama.cpp 主线 0.9.0 或更新**（ggml 0.29.0+），用 `-DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON` 构建 | `https://github.com/ggml-org/llama.cpp` | `<checkout>/third_party/llama.cpp/build/bin/`，或设 `ZIMAGE_LLAMA_DIR` | **运行时**——由 `ctypes.WinDLL` 加载 |
| **Vulkan SDK**——`glslangValidator`、`glslc` | 1.4.357.0 | LunarG | — | 仅用于构建 Vulkan 后端 |

**请从主线构建，不要用别人留下的树。** 之前有个智能体在
`D:\Project\openhanako\workbench\prism-vulkan` 留了一份可用的副本；那只是那一台机器上的
便利，**不是本项目的依赖**。任何你打算复现的东西都必须来自 llama.cpp 的上游构建。

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -G "Visual Studio 17 2022" -A x64 \
      -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build --config Release --target llama
# -> build/bin/Release/{llama,ggml,ggml-base,ggml-cpu-*,ggml-vulkan}.dll
```

**这里有两处会绊住你。**

`-DBUILD_SHARED_LIBS=ON` 不是可选项：不加它得到的是静态库，没有 `llama.dll` 供
`ctypes.WinDLL` 加载。

**版本下限是硬性的。** 本项目用的不是普通嵌入接口——`llama_get_embeddings()` 返回的是
**最后一层、且过了 `output_norm`** 的张量，那是另一个东西。它需要的是逐层输入抽头，
声明在 `src/llama-ext.h`（一个 staging 头文件，不在 `include/llama.h` 里）：

```c
LLAMA_API void    llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);
```

这两个是 **C++ 链接**，所以它们在 DLL 里以修饰名出现，而
`imaggers/textenc/encode_api.py` 正是按这两个字符串查找的：

```
?llama_set_embeddings_layer_inp@@YAXPEAUllama_context@@I_N@Z
?llama_get_embeddings_layer_inp@@YAPEAMPEAUllama_context@@I@Z
```

已在 **llama.cpp 0.9.0**（ggml 0.29.0）上验证。没有这个抽头的构建会在 import 阶段
以未解析符号的错误明确失败，而不是悄悄返回错误的张量。

**主线构建与已退役的 fork 端到端逐字节一致。** 同一配方、同一个 512x512 / 8 步任务，
两套 DLL：

| 构建 | `latents` sha256 | 输出 PNG sha256 | `loop_s` |
|---|---|---|---|
| llama.cpp 0.9.0 主线 | `17f7dd1a6201772a` | `b862cd72...3145a87f` | 91.92 |
| `prism-vulkan`（fork） | `17f7dd1a6201772a` | `b862cd72...3145a87f` | 87.77 |

`pe` 张量本身在最后几位上有差异（相对 fork 余弦 0.9999999，最大相对误差 5.3e-05），
原因是不同版本的 llama.cpp 累加顺序不同。这个差异过不了 DiT：latents 和 PNG 完全相同。
`loop_s` 每次运行有约 4 秒的波动，不能归因于 DLL。

`encode_api.py` 按下面的顺序查找 DLL 目录，先命中者优先：
`ZIMAGE_LLAMA_DIR` → `<checkout>/third_party/llama.cpp/build/bin` → 旧的 `prism-vulkan` 路径
→ 文档路径（让加载失败时报出正确的放置位置）。

`webui/te_worker.py` 有一个 `cpu` 模式，只加载 `ggml-cpu-*.dll` 并设 `n_gpu_layers=0`，
用于 iGPU 显存被别的程序占用的情况。

### 10.4 Python

刻意保留两个完整解释器——两套科学栈无法共存：

| 角色 | 版本 | 关键包 |
|---|---|---|
| **运行**（设备侧） | **3.10.9** | numpy 1.26.4、torch 2.3.1+cpu、**numba 0.62.1**、llvmlite 0.45.1、pyxrt |
| **编译 / 规划** | **3.12.11** | numpy 1.26.4、torch 2.4.1+cpu、**ml_dtypes 0.5.4**、onnx 1.22.0、llvm-aie 1.8.0、diffusers 0.36.0 |

`numba` 只在 3.10 上有；`ml_dtypes` 与 `llvm-aie` 只在 3.12 上有。用
`IMAGGER_PY310` / `IMAGGER_PY312` 覆盖路径；preflight **P01** 会检查两者都能跑。
两个解释器上的 `torch` 都是 CPU 版——这里没有 CUDA/HIP 后端，`torch-directml` 也装不上。

### 10.5 全新 clone 能做什么

| | 仅 clone | + 模型 | + NPU 工具链 | + llama.cpp + GGUF |
|---|:---:|:---:|:---:|:---:|
| 读代码、跑规划器 | ✅ | ✅ | ✅ | ✅ |
| `weights/build_bfp16_weights.py` | ❌ | ✅ | | |
| 重建内核（`.xclbin`） | ❌ | | ✅ | |
| **生成图像** | ❌ | ✅（权重） | ✅ | ✅ |

**66 个编译好的内核镜像（约 8.8 MB）就在本仓库里**（`webui/cache/<shape>/` 与
`imaggers/*/out/`），所以运行**不需要**编译器，只需要 XRT 运行时。

### 10.6 换机器时需要改的绝对路径

| 文件 | 常量 | 当前指向 | 覆盖方式 |
|---|---|---|---|
| `webui/config.py:24` | `XRT_DIR` | `D:\Project\deepseek_harness\test7\WStmp\driver376\npu_mcdm_stack_prod` | `IMAGGER_XRT` |
| `webui/config.py:22-23` | `PY312`、`PY310` | 两个解释器 | `IMAGGER_PY312` / `IMAGGER_PY310` |
| `imaggers/textenc/encode_api.py` | `LLAMA_DIR` | 解析顺序：`ZIMAGE_LLAMA_DIR` → `third_party/llama.cpp/build/bin` → 旧路径 | `ZIMAGE_LLAMA_DIR` |
| `imaggers/textenc/encode_api.py` | `GGUF` | 解析顺序：`ZIMAGE_TE_GGUF` → `<checkout>/webui/models/` → 工作区内的副本 | `ZIMAGE_TE_GGUF` |

`webui/config.py:27` 从 `__file__` 解析 `PORT`，会跟着 checkout 走，无需修改。
两处文本编码器路径现在是导入时解析而非硬编码；新机器上只剩 `XRT_DIR` 和两个解释器
需要手工改，而它们各自都有环境变量。

## 11. 许可证

MIT——见 [LICENSE](LICENSE)。
