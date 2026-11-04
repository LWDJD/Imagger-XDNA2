# GRAPHOPT-47-REPORT — 基线出图复跑 + §4.7 判定实验（t22_graphrun）

> 2026-10-30 · 工作区 `zimage-bfp16-npu` · 接 fca628c7 的离线交付（GRAPHOPT-REPORT.md）
> **本轮有器件、有进程、有出图。** 基线逐字节复跑通过；§4.7 判定实验跑完，P3 的上界被证伪但幅度不足以翻案。
> 器件锁 `_scratch/device.lock` 全程 F-14.1：**pid = 真正 dispatch 的进程**（webui 用自锁启动器写自己的 os.getpid()；每个 D1 臂由 py310 进程自持自放）。

---

## 0. 结论（先给答案）

1. **基线复跑通过（F-23 口径，逐字节）**：7874 新实例、128²/8 步/seed 7、产品默认（fs_ffn_j5 + qkv/p10 开 + CPU VAE）
   → latents `9f86aee91de3dad09bf888b15fe560e16f9c3079920118d714697f25421dbfe2`、
   PNG `104ff564709c017848f2b433fe37e602558178da32729035c27552820500219a`（21,318 B），960 dispatch / 0 retry。
   **稳态墙钟 32.795 s**（loop 22.659 / device 11.921 / startup 8.166 / VAE 0.430）；冷发 77.636 s（其中 CPU VAE 首载 41.68 s）。
2. **§4.7 判定实验（只测判定点）**：ctrl 19.983 ms（n=9）；pass 斜率 **0.255 ms/pass = 382 GFLOP/s 微基准的 96.9 %**；
   A 读路径全删 **−1.1…−1.3 ms（−6.6 %）**；sw 读路径全删 **−0.3…−0.4 ms（−1.9 %）**。全部 state=COMPLETED。
3. **P3（A 驻留 ≤0.45 ms，置信 0.80）作为"上界"被证伪**——但要看清口径：aiso 同时删了 2,360 个 per-slice fill 任务，
   S1 只去掉 DRAM 那一段。S1 的捕获 ≤1.3 ms，现实 0.4–0.7 ms/block-step。
4. **换算到产品墙钟**：1.3 ms × 240 block-step = **0.26–0.32 s/图 = 0.8–1.0 %**。
   **建议：S1 不值得再投 0.5 天，更不值得 3–5 天全量重写**（决策页 `GRAPHOPT-47-DECISION.md`）。
5. **同一段真正的杠杆**（按证据强度）：**D1 产品化 −3.34 s/图（−10.2 %，卡 kgu64 链对 ref64 的数值门）**
   > GU 核数 4→8（−1.25 s，撞 shim/memtile 通道墙）> down k=8→64（−1.17 s，换参考已锁死）> GU 内核形态（−0.08 s，已到天花板）。
6. **交付缺口一个（已修）**：`zimage-bfp16-repo/_models/Z-Image-Turbo/host_tensors.safetensors` 不在树里，
   t18_run.py 的 device 腿直接 `FileNotFoundError`。用 `_scratch/image/t3real_host_tensors.npz` 重建（15 键、max|diff|=0.0、47,410,936 B）。

---

## 1. 基线出图（第一步）

### 1.1 口径与命令

    py312 webui.py --host 127.0.0.1 --port 7874      # 7870 全程未碰（事先确认无监听）
    POST /api/jobs {"prompt":"a mountain lake at sunrise","height":128,"width":128,
                    "steps":8,"seed":7,"vae":true,"mode":"trial"}

* 服务：py312 3.12.11；`ZIMAGE_WPACK_PREWARM=job`（默认）、wpack `reused`（verify 1.3–1.5 s）、memory_mode=fast。
* 锁：`_scratch/t22_graphrun/launch_webui.py` 用 `os.open(O_CREAT|O_EXCL)` 建锁并写入**自己的 pid**，再 `runpy` 起 webui —— 没有 helper pid。
* TE：两发 textenc_s 都是 0.009 s（server 侧常驻 worker 已在），所以 32.795 s 是**不含 TE 冷加载**的稳态口径。

### 1.2 结果

| 项 | B1（冷） | **B2（稳态）** |
|---|---|---|
| job id | 20261030-202719-6ea3eb | **20261030-202838-94969a** |
| status | done | **done** |
| total_s | 77.636 | **32.795** |
| loop_s / device_s / startup_s | 23.769 / 12.573 / 10.274 | **22.659 / 11.921 / 8.166** |
| textenc_s / vae_s | 0.009 / 41.676 | **0.009 / 0.430** |
| dispatches / retries | 960 / 0 | **960 / 0** |
| latents sha256 | `9f86aee9…dbfe2` ✓ | `9f86aee9…dbfe2` ✓ |
| PNG sha256 / bytes | `104ff564…219a` / 21,318 ✓ | `104ff564…219a` / 21,318 ✓ |
| closure.pass / unaccounted | true / 0.044 | **true / 0.047** |

* PNG 用落盘文件独立重算：`Get-FileHash webui/outputs/zimage_20283894969a_p0_s7.png` = `104FF564…219a`；
  参考件 `_scratch/image/w14465543340cp0_latents.npy` 重算 = `9F86AEE9…DBFE2`。
* 步时：step0 6.95 s（首层冷），step1–7 2.42–2.46 s。

### 1.3 与报告 §3 引用的 40.82 s 的口径差

报告 §3 引的 steady 40.82 s 在本快照的 `_scratch/t22_e2e_measure/SUMMARY.md` 里**找不到**（该文件写的是 cold 138.43 / steady 113.67，
是 **NPU VAE** 口径，vae_npu_s 61.88）。本轮是 **CPU VAE** 口径、且 TE 常驻，所以 32.795 s 不能与 40.82 或 113.67 直接比；
同口径比较只能用本轮 B1/B2 自己的成对值。

### 1.4 交付缺口与修复（非产品代码改动）

t18_run.py（产品驱动）在 device 腿上读 `zimage-bfp16-repo/_models/Z-Image-Turbo/host_tensors.safetensors`：

    FileNotFoundError: ...zimage-bfp16-repo\_models\Z-Image-Turbo\host_tensors.safetensors

该文件与它来源的 transformer 权重都不在快照里（`extract_host_tensors.py` 无法重生成，因为 22.93 GiB checkpoint 不在）。
修法：`_scratch/t22_graphrun/restore_host_tensors.py` 从 `_scratch/image/t3real_host_tensors.npz`（15 个非 block 张量，float32）
重建 safetensors 容器，回读 **15/15 键、worst |diff| = 0.0**，47,410,936 B。写的是模型目录，不是产品代码；`zimage-bfp16-repo` 的 git status 干净（_models 被 ignore）。

---

## 2. §4.7 判定实验（第二步）

**预测先写死**：`_scratch/t22_graphrun/GRAPHOPT-47-PREDICTIONS.md`（R0–R5，在任何器件动作之前落盘）。
**原始数据**：`_scratch/t22_graphrun/S47-RESULTS.md`。**每臂一进程一 xclbin**；`xrt-smi validate -r gemm` 4/4 `TOPS 51.3 [PASSED]`；
`t3d_baseline --check` 窗前后各一次 `ALL REFERENCE BYTES UNCHANGED`。

### 2.1 三个判定点

| 臂 | 改什么（不写新内核，只改数据面） | 墙钟中位 | 相对 ctrl | state |
|---|---|---:|---:|---|
| **ctrl** | `d1_ffn` 原样 | **19.983 ms**（n=9，19.746–20.219） | — | 全 COMPLETED，Y 全 `e2c5b24e37ce4907` |
| **p20** | passes=20 | 14.882 ms（n=6） | **斜率 0.255 ms/pass** | 全 COMPLETED |
| **aiso** | 每 pass 只取 1 个 A 对象（12,288 B）复用 60 次：A DRAM 29.49→0.49 MB，fill 2,400→40 | **18.668 ms**（n=6） | **−1.32 ms（−6.6 %）** | 全 COMPLETED，Y `49ce6230…` |
| **swiso** | 每 round 只取 1 个 sw 对象（1,536 B）复用 1,280 次：sw DRAM 19.66 MB→1.5 KB，任务 12,800→10 | 19.609 ms（n=6） | −0.37 ms（−1.9 %） | 全 COMPLETED，Y `33abd829…` |

### 2.2 读法（这是本单的核心）

* **pass 斜率 = 计算本身**：每 pass 每核 94.37 MFLOP ÷ 0.255 ms = **370 GFLOP/s/核**，是 kgu=64 常驻微基准 382 GFLOP/s 的 **96.9 %**。
  A 的 0.74 MB/pass 与 BG 的 2.21 MB/pass 已经被计算盖住，**没有出现在斜率里**。
  ⇒ GU 段剩余的全部头寸 = 3.1 % ≈ **0.32 ms/block-step**——这就是"内核形态率"这条杠杆的天花板。
* **A 读路径全删只值 1.1–1.3 ms**（不是字节账的 3.09 ms，也不是 0.45 ms）。这同时证明：
  (a) 报告"DRAM 只跑到 9.32 GB/s、离饱和很远"成立；(b) 但"A 重读完全不值钱"也不成立——它值约 6 % 的 FFN dispatch。
* **sw 全删只值 0.3–0.4 ms**，与"down 是 k=8 的每调用开销、不是带宽"一致（P4 基本成立）。
* **整图换算**：1.1–1.3 ms × 240 = **0.26–0.32 s/图**，占 32.795 s 的 **0.8–1.0 %**。

### 2.3 负结果（必须记账）

* **零步长 tap 做隔离 = 死**：首版 aflat/sflat 把 tap 的 slice 步长改成 0，编译期就警告
  `repeat count 2399 needs 10 queue pushes, more than the 4 the channel's queue holds ... the wait never ends`，
  器件 3/3 发 `ERT_CMD_STATE_TIMEOUT`（7,006–7,029 ms，Y 全 `c2cd1bef7dc6c18e`，与 U4 的 UNROLL≥3 同一个死 Y）。
  改成"每 pass/round 只取 1 个对象、核内复用"才跑通（insts 仍 2,908 B，与 ctrl 同尺寸）。
* **aiso ≠ S1**：aiso 是上界。S1（memtile 驻留）仍要 2,400 次 per-slice 重放（报告 §4.5 R1），只去掉 DRAM 29 MB。
  S1 能拿到 1.3 ms 的多少，取决于"DRAM 时间 vs 任务数"的拆分——本轮没测（见决策页 §3 的唯一建议）。
* **P2（737,280 B 单个 memtile 对象编不过）本轮没测**：没有写 S1，未撞编译墙；这是"未验证"，不是"已验证"。

---

## 3. 决策（第三步）

见 **`GRAPHOPT-47-DECISION.md`**（一页）。核心：

* **S1 不值 0.5 天**：上界 0.26–0.32 s/图（<1 %），现实 0.1–0.2 s/图。
* 若非要花 0.5 天：只做"DRAM 时间 vs 任务数"的拆分臂（保留 2,400 任务、只把源地址收敛到 12 KB，需 memtile 中转）。
* 替代杠杆：**D1 产品化 −3.34 s/图（−10.2 %）** ≫ GU 核数（−1.25 s）> down k 形状（−1.17 s，锁死）> GU 形态（−0.08 s）。
* 95.5 % 微基准算术（报告口径）：382 GFLOP/s/核 → 每 pass 0.247 ms → 40 pass 9.88 ms；实测 GU ≈10.4 ms → 95 %；
  本轮斜率口径 96.9 %。两条都指向"GU 已到天花板"。

---

## 4. 复现（本单全部命令）

    # 0 门
    xrt-smi validate -r gemm                                  # TOPS 51.3 [PASSED]
    py312 _scratch/image/t3d_baseline.py --check              # ALL REFERENCE BYTES UNCHANGED

    # 1 缺件修复（只需一次）
    py312 _scratch/t22_graphrun/restore_host_tensors.py

    # 2 基线（自带锁、自带 2 发、自动核对）
    powershell -NoProfile -ExecutionPolicy Bypass -File _scratch/t22_graphrun/run_baseline.ps1

    # 3 编译对照/实验臂（py312；~20 s/个）
    py312 _scratch/t22_graphrun/gcompile.py --design d1_ffn       --tag d1_ctrl --kw passes=40 --kw rounds=10
    py312 _scratch/t22_graphrun/gcompile.py --design d1_ffn       --tag d1_p20  --kw passes=20 --kw rounds=10
    py312 _scratch/t22_graphrun/gcompile.py --design d1_ffn_aiso  --tag d1_aiso
    py312 _scratch/t22_graphrun/gcompile.py --design d1_ffn_swiso --tag d1_swiso

    # 4 器件臂（窗 B/C：ctrl/p20/aiso/swiso 交替，每发一进程）
    powershell -NoProfile -ExecutionPolicy Bypass -File _scratch/t22_graphrun/run_s47b.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File _scratch/t22_graphrun/run_s47c.ps1

> `gcompile.py` 只修了 `d1_compile.py` 的工具链路径：本快照 `_pylibs` 在 workspace 根，不在 `zimage-bfp16-npu/_pylibs`。

---

## 5. 文件清单 / 提交

新增（**pathspec** 提交，未动产品文件）：

    GRAPHOPT-47-REPORT.md
    GRAPHOPT-47-DECISION.md
    _scratch/t22_graphrun/*.py / *.ps1 / *.md / *.json / *.log
    _scratch/t22_graphrun/designs/d1_ffn_{aiso,swiso,aflat,sflat,anorm,snorm}.py

另把上一单的离线交付一并入树：`GRAPHOPT-REPORT.md`、`_scratch/t22_graphopt/{GRAPHOPT-PREDICTIONS.md,byte_budget.py}`。

**产品零变化**：`webui/`、`webui.py`、`_scratch/image/*.py` 一个字节没改；
`t3d_baseline --check` 全程 4 次全绿；默认路径仍是 fs_ffn_j5 + qkv/p10 + CPU VAE。

---

## 6. 诚实边界

1. 本轮只测了 **128²/8 步/seed 7** 一个形状、**一个窗口**的 D1 原型；没有产品路径的 A/B（S1 没实现）。
2. `aiso/swiso` 是**计时臂**，数值是垃圾（Y 与链不同）——它们只证明"删掉这条读路径值多少墙钟"。
3. P2/P6（memtile 容量与通道墙）**未测**：没写 S1，不能把"9/8 撞墙"从预算结论升级成实测结论。
4. 32.795 s 是 **CPU VAE + TE 常驻**口径；与报告 §3 的 40.82 s、SUMMARY 的 113.67 s 都不可直接比。
5. 器件窗口有 0.5–1.2 ms 漂移（约 2–6 %），所有差值取同窗口成对中位；绝对秒数不跨会话比较。
