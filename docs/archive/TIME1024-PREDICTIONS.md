# TIME1024-PREDICTIONS — 1024² 计时（t22_time1024）

写于 2026-10-31 22:05 (+08:00)，**在任何 1024² job 之前**。owner：t22_time1024。
口径：1024²（M=4128=43×96）/ seed 7 / VAE off（计时臂）；ZIMAGE_DEVICE_TIMEOUT_S=7200；
ZIMAGE_BIGM=1、ZIMAGE_BIGM_PROJ=1、T22_HOSTKILL=1、T22_DEVKILL=1、T22_BIGM_DEVHANDOFF=1、
T22_FINAL110=1、host f32 attention。器件：NPU Strix（XRT 2.21.0 / driver 32.0.20102.3930，单实例）。

## 0. 量纲与锚（全部实测，来自 DEVKILL-REPORT / BIGPIPE3 / 1024_dbg2）

| 锚 | 值 | 来源 |
|---|---|---|
| 512² 稳态 30 层/步 | 10.75 s（358 ms/层） | DEVKILL 8g/8h |
| 其中 投影 153 / FFN 103 / attention 90 / rope 19 ms/层 | — | DEVKILL §2 |
| 1024² 旧 big-M 逐层（dbg2） | proj 1.59 / FFN 0.68 / attn 1.19 / rope 0.07 = 3.53 s/层 | 1024_dbg2 日志 30 层中位 |
| 1024² loop 132.37 s 里层外 | 132.37 − 30×3.53 = **26.5 s**（refiner pre.* + misc + step 边界） | 同上（与 512² 的 4.2 s 不成比例，按实测取 26.5） |

## 1. 预测 A：1024² 1 步，最优栈（本单主臂）

**分段模型（每层，M 线性项 ×3.909；器件 run 按 11-tile 块固定价）**

| 段 | 计算 | 预测 ms/层 |
|---|---|---:|
| 投影器件 | 4 投影 × 4 块（11+11+11+10 tile）× 15.4 ms | 246（最后一块 10 tile 仍按 11 计，+9% 浪费） |
| 投影 host（F110 zero-copy） | 512² 的 91 ms × 3.909（16 次 write/read/cv） | 355 |
| FFN（RT=11 对） | 4 块（11+11+11+10）× 102.9 ms | 412 |
| attention host f32 | 1024² 实测（dbg2） | 1220 |
| rope/rms | 19 × 3.909 | 74 |
| 逐层合计 | — | **2307** |
| ×30 | — | **69.2 s/步** |
| 层外（refiner+misc） | 实测 26.5 s | 26.5 |
| **loop** | — | **95.7 s** |

**预测（先落盘）**：1 步 **loop 90–105 s（点 96）**；step0 = loop（1 步就是 step0，含首次 JIT/首触）；
**total 118–138 s（点 128）**（startup 30–35 + loop + post 3）。
分段预测：proj_qkv ≈ 0.52 s、proj_o ≈ 0.17 s、FFN ≈ 0.41 s、attention ≈ 1.22 s、rope 0.07 s/层。

## 2. 预测 B：1024² 8 步，最优栈

step0 ≈ loop₁ + 5（首步 JIT）≈ 101；稳态步 95.7 → **loop₈ ≈ 770 s（带 720–830）**；
**total ≈ 810 s（带 760–870）**。PSNR：对同场默认 1 步（臂 C）**corr 0.985–0.995 / PSNR 28–33 dB**
（DEVKILL 已证全流程 latents 不可逐字节复现；单跑 corr 是分布）。latents 单跑，不取中位。

## 3. 预测 C：同场默认配置 1 步（对照）

默认 = ZIMAGE_BIGM 不设、ZIMAGE_CHUNK_PIPE 默认 1、全部 T22 旗标不设（交付 chunked 96 行流水线）。
历史同口径 1024² 1 步：100.78 / 111.46 / 123.76 s（BIGPIPE→FINAL110 各轮）。同场热机预测
**loop 105–125 s（点 115）、total 135–160 s**；对本单最优栈应 **不慢于** 最优栈（若最优栈在 1024² 反而更慢，是
1056 分块的 4 次固定价 + host 16 次往返造成的，属正常）。

## 4. 预测 D：MFU

口径沿用项目既有：**512² 8 步 50% = 1.92 s** ⇒ 1024² 8 步 50% = **7.68 s**（token ×4，线性）。
- 1 步 50% 基准 = 0.96 s ⇒ MFU₁ = 0.96 / step_s；8 步 MFU₈ = 7.68 / loop₈。
- 预测：MFU₁ ≈ 0.96/95.7 = **1.0%**；MFU₈ ≈ 7.68/770 = **1.0%**（对 51.3 TOPS 固定门约 50×）。
- 1024² 1 步 FLOP = 6e9×4096×2 = 49.15 TFLOP ⇒ 实测吞吐 ≈ 0.51 TFLOPS。

## 5. 风险 / 先验坑（跑前写下）

1. **1056 行分块坑**：devkill 的投影/FFN 镜像都是 RT=11（M=1056）。投影 proj4 已按 max_tiles=11 分块；
   **FFN 的 devkill_ffn.bigm_ffn 仍按 1632 行（RT=17）分块**，对 RT=11 镜像会越界（BO 8.1 MB）——本单在
   T22_DEVKILL 分支内把 FFN 分块改为 1056 行；产品默认分支零改动。
2. 43 tile = 11+11+11+10：最后一块 10 tile 仍付 11 tile 的固定价（预测已计 +9%）。
3. attention 4128² 的 host 峰值内存（qblock 512 × 4128 × 30 × 4 B ≈ 253 MB/块）——dbg2 已跑通。
4. 若 FFN 分块后某一块 state≠4（7 s 看门狗）→ 记负结果，不冒充墙钟。
5. 质量：单跑 corr 是分布；本单只跑 1 次 8 步（任务书"1 跑注明即可"）。
