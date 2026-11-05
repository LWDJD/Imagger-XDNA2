# FINAL110-REPORT — 512² 8 步 110 s 收官单（t22_final110）

日期：2026-10-31 20:40 (+08:00)｜owner：t22_final110｜器件：NPU Strix（XRT 2.21.0 / driver 32.0.20102.3930，单实例）
承接：DEVKILL-REPORT.md（512² 8 步 121.50 s 最好 / 123.2 中位）、OPENSTACK-REPORT.md（pvmap/qkmap 全表）、HOSTKILL-REPORT.md（X_tap 先例）
口径：512² / 8 步 / seed 7 / VAE off；ZIMAGE_BIGM=1、ZIMAGE_BIGM_PROJ=1、T22_HOSTKILL=1、T22_DEVKILL=1、T22_BIGM_DEVHANDOFF=1、host f32 attention；ZIMAGE_DEVICE_TIMEOUT_S=7200。

## 0. 判词

**两项都做了；110 s 仍未过，且 attention 下器件被证明是负收益（+0.37 s/层）。**

| 任务项 | 目标 | 实测 | 判 |
|---|---|---:|---|
| ① 向量化 MMA attention 装配（表驱动 pack/accum） | 单头 33.6 ms 形态接上 → G=8~15 → **<90 ms/层** | 单头 **62.1–62.8 ms**（corr **0.999945**）；8 核整层 **462.8 ms**（逐头 min corr **0.999745**） | **FAIL（5.1× 于 90 ms）** |
| ② 投影 host cv/read 压半 | 107 ms/层 → ≤50 | read+cv **34 → 17 ms/层**（zero-copy map + 预分配） | **FAIL（口径只覆盖 read+cv；107 ms 里 70 ms 是器件 run/写）** |
| ③ 512² 8 步 A/B | loop ≤110 s；质量 3 跑中位 ≥0.99/28 dB | **142.36 s**（同场基线 143.98 / 154.62）；3 跑 corr 中位 **0.9908** | **FAIL（1.29×）** |
| ④ 1024² | ≤75 s | **未跑**（L1 未绿，按纪律不启动） | — |

一句话：**表驱动 pvmap/qkmap 把 vectorized MMA attention 从 v9 的 96.4 ms/头打到 62 ms/头（1.55×，数值绿），但整层不是算力墙而是 host 逐对象 fill 墙：2244 次/头 × 30 头 = 6.73 万次/层 × ~6 µs ≈ 404 ms，8 核整层 462.8 ms，和 af9m8 的 468.7 ms 同档。hostkill 的 tap 法在 KV 形状上撞两条硬墙（逐字见 §2）：shim 生产者不支持 dimensionsToStream，memtile 1:1 link 放不下 540,672 B 的整头 KV。投影 zero-copy 只省 ~17 ms/层（~0.5 s/步）。两者合计仍离 110 s 差 30+ s。**

## 1. ① 向量化 MMA attention：装配成功、数值绿、墙钟红

### 1.1 做法（表驱动，不猜布局）

pvmap/qkmap 的 A/B/C 三张表先做闭式化验证（全部与 JSON 全表逐元素相等）：

- A(m,k) = 4K·(m//4) + 8·(m%4) + 32·(k//8) + (k%8)
- B(k,n) = 8K·(n//8) + 8·(n%8) + 64·(k//8) + (k%8)
- C(m,n) = 4N·(m//4) + 8·(m%4) + 32·(n//8) + (n%8)

于是 host 侧 pack 是纯 reshape/transpose（无 gather），device 侧 softmax 用 QKC 表读、P 用 PVA 表写、PV 累加用 PVC 表读；SRAW/OBLK 在消费者末尾自清零，内层只剩 4 个 dispatch（qk mm → softmax_pack → pv mm → accum），无 zero kernel。

实现：`_scratch/final110/attn_mma.py`、`attn_mma8.py`、`attn_mma_kernels.cc`、`attn_maps.h`（QKC/PVA/PVC 由 JSON 生成）。

### 1.2 单头（512²，seq=1056，head0，真实 layer0 dump）

| 臂 | 形态 | wall/头（3 跑） | corr | 判 |
|---|---|---:|---:|---|
| v9（attnfuse） | 融合核 bf16 | 96.4 ms | 0.999996 | 基线 |
| **am16f** | **表驱动 vectorized MMA** | **62.8 / 62.5 / 62.1 ms** | **0.999945**（rel 0.018） | **PASS 数值 / 1.55× 于 v9** |
| am16d | 同上 + 显式 OBLK zero | 62.4 / 61.9 / 61.8 ms | 0.999945 | 同档（自清零版同速） |
| am16（负结果） | f32 PV 输出、无 OBLK zero | 271 ms | 0.999967 | 慢 4.4×（f32 PV 微核 path） |
| v18（旧证） | vectorized、布局错 | 33.6 ms | 0.0128 | 只作速率参考，不作正确性 |

单头 62 ms 构成：MMA 计算地板 10.0 ms（2,178 block × 2 × 2.3 µs）+ 搬运/调用 ~13 ms（NOP 下限）+ 表 gather/softmax/accum ~39 ms。

### 1.3 8 核整层（30 头，4 组×8）

| 组 | heads | wall ms | state |
|---|---:|---:|---:|
| 0 | 0–7 | 118.4 | 4 |
| 1 | 8–15 | 117.5 | 4 |
| 2 | 16–23 | 107.2 | 4 |
| 3 | 24–29 | 119.7 | 4 |
| **整层** | 30 | **462.8** | **4/4** |

逐头最小 corr **0.999745**（逐头全 ≥0.9997）。对 host f32 的 90 ms/层是 **5.14×**；对 v9 的 468.7 ms 只快 1.3%。**目标 <90 ms/层 FAIL。**

### 1.4 fill 账（为什么 8 核不提速）

每头：Q 66 次 + KV 33×66 = 2,178 次 → 2,244 次；30 头 = **67,320 次/层**。v18 NOP 实测单次 fill **5–6.6 µs**，67,320 × 6 µs = **404 ms/层**，与 462.8 ms 同量级。MMA 算力（62 ms/头 / 8 核 ≈ 233 ms/层）被 fill 税盖住。

## 2. tap 法在 KV 形状上的逐字墙（按裁决试过）

### 2.1 一次大对象 + shim tap 自走

`inKV = ObjectFifo(kv_big_ty, depth=1, to_stream=kv_walk, consumer_obj_type=kv_ty)`（kv_big_ty = 33×16 KB = 540,672 B，一个 fill）：

```
aie._mlir_libs._mlir.ir.MLIRError: Verification failed:
error: unknown: dimensionsToStream data layout transformations are not supported on shim tile producers
```

shim 生产者（host fill）不能做 layout 变换/切片，必须走 memtile 1:1 link。

### 2.2 memtile 1:1 link + corner walk

`ObjectFifoLink(inKV.cons(), midKV.prod(), Tile(0,1), [], [])`，midKV.to_stream=kv_walk、consumer_obj_type=kv_ty：

```
mlir:10:5: error: 'aie.objectfifo.pool' op could not place buffers in accessible memory with available capacity
aiecc: edge 'input_with_symbols.mlir' (key 'input.mlir') failed
aiecc: pipeline failed
```

容量账：一个头 K+V = 33×16,384 B = **540,672 B > memtile 524,288 B**（超 16,384 B = 32 keys，attnfuse §5 同一笔账）。K/V 拆两个 memtile：每 worker 2 个 × 8 worker = **16 > 8 个 memtile**；4 worker 则 30 头要 8 轮，按 62 ms/头 ≈ 496 ms/层，仍 >90。

**判：tap 法对 KV 形状不可用（通道/容量双杀），fill 税在当前 memtile 数下不可归零；8 核 attention 地板 ≈ 460 ms/层。**

## 3. ② 投影 host read/cv：zero-copy 实测

改法（`T22_FINAL110=1`，默认 0）：输出 BO 用 `bo.map()` 零拷贝 + 每 (tag,tiles) 预分配 f32 缓冲，`np.left_shift` 原地提升；不再 `bytes(bo.read(...))` 复制。

| 每投影（11 tile） | 基线 read / cv | F110 read（含 cv） | 省 |
|---|---:|---:|---:|
| q | 2.4 / 6.1 ms | 3.6–5.3 ms | ~4 ms |
| k | 2.4 / 5.1 ms | 3.1–4.1 ms | ~4 ms |
| v | 3.1 / 6.8 ms | 3.6–5.1 ms | ~4.5 ms |
| o | 2.8 / 5.2 ms | 2.6–5.0 ms | ~2.5 ms |
| **4 投影/层** | **~34 ms** | **~17 ms** | **~17 ms/层（~0.51 s/步）** |

**口径对账**：任务书 107 ms/层 是 `proj_qkv 0.105–0.116 s + proj_o 0.050–0.073 s` 的整段（含器件 run 4×15–16 ms + 写 + Python/BO），其中真正的 host read+cv 只有 ~34 ms/层。本轮把 read+cv 压半到 ~17 ms，整段只从 ~170→153 ms/层（~0.5 s/步），**不是 3.2 s/步**；3.2 s/步口径把器件 run wait 计入了 host，需更正。

## 4. ③ 512² 8 步 A/B（同场邻臂，2026-10-31 20:20–20:33）

| arm | job | 变的是什么 | loop_s | total_s | step0 | 稳态中位 | corr / PSNR（对 512_base_8b） |
|---|---|---|---:|---:|---:|---:|---|
| 8a2 | 20261031-202202-6e1c7d | **+T22_FINAL110** | **142.36** | 180.25 | 23.9 | ~17.6 | **0.995455 / 33.31 dB** |
| 8base | 20261031-202600-f1eb72 | 同 8h，无 F110 | 143.98 | 184.36 | 21.4 | ~18.0 | 0.952546 / 23.15 dB（离群） |
| 8base2 | 20261031-203105-52470e | 同 8h，无 F110 | 154.62 | 192.06 | — | ~19–20 | 0.990849 / 30.27 dB |

- 同场邻臂 A/B：**142.36 vs 143.98 = −1.62 s（−1.1%）**，与 ~17 ms/层×30 ≈ 0.51 s/步同量级；但 F110 臂跑在前、机器持续热漂移（8base2 已到 154.62 s），**−1.62 s 不能全记作净收益**，只作方向证据。
- 质量：3 跑 corr 0.995455 / 0.952546 / 0.990849，**中位 0.990849 / 30.27 dB → PASS（≥0.99/28）**；8base 的 0.9525 是离群（DEVKILL 同配置 0.9911–0.9951；F110 与 8base2 互比 **0.991201**，属正常跑间），**质量是分布不是点**。
- 对 110 s 门：142.36 / 110 = **1.29×**；DEVKILL 冷态最好 121.50 s 也是 **1.10×**。
- MFU（1.92 s = 50% 口径）：142.36 s → **0.675%**；121.5 s → 0.79%；110 s 门本身 0.87%。

**attention 未接产品**：8 核 462.8 ms/层 > host 90 ms，接进去会把稳态拖慢 ~0.37 s/层（+11 s/步），按纪律“绿后才接”不接；本 A/B 只含投影 zero-copy。

## 5. ④ 1024²

**未跑。** 512² L1 未绿（1.29×），按纪律不启动。旧账：1024² 1 步最好 128.20 s，对 75 s 门 1.71×；attention host 37.9 s/步，8 核 v9 外推 61.7 s/步，均无正收益。

## 6. 归零表（两项）

| 项 | 起点 | 本轮 | 目标 | 差 | 性质 |
|---|---:|---:|---:|---:|---|
| attention host f32 | 90 ms/层 | 90（未接） | — | — | 保留 |
| attention 器件（8 核 vectorized） | v9 468.7 ms/层 | **462.8 ms/层**（单头 62 ms、corr 0.999945） | <90 ms/层 | **5.1×** | **fill 税 67,320 次/层；tap 容量墙** |
| 投影 host read+cv | 34 ms/层 | **17 ms/层** | ≤50（任务书口径 107→50） | 口径不符 | 只省 0.5 s/步 |
| 稳态步 | 14.9–15.2 s | 17.6–20 s（热漂移） | 13.75 s | — | 机器状态劣化，非代码 |

## 7. 翻默认建议表 / 差距表

**不全绿，不翻默认。**

| 旗标/形态 | 默认 | 本轮证据 | 建议 |
|---|---:|---|---|
| T22_FINAL110（投影 zero-copy） | 0 | 142.36 vs 143.98（−1.6 s，含热漂移），质量中位 0.9908 | **可作下一单底座；单独不翻** |
| attention 器件（am16f/am8） | 未接线 | 单头 62 ms 绿、层 462.8 ms 红 | **保持 host f32** |
| 交付 chunked（现默认） | 1 | 177.76 s（旧） | **保持默认** |
| 512² L1 门 | — | 142.36 s（同场）/121.5（冷态） | **FAIL（1.29× / 1.10×）** |

**到 110 s 的差距表（按冷态 121.5 底座）**：

| 项 | 现状 | 110 s 需要 | 差 |
|---|---:|---:|---:|
| 稳态步 | 15.0 s | 13.75 s | 1.25 s/步 |
| attention | 2.7 s/步（host） | <2.2 s/步 | 器件 462.8 ms/层 = 更慢，无解 |
| 投影 host | 3.2 s/步（口径含 run wait） | ~2.7 | zero-copy 只 −0.5 |
| 剩余 | — | — | **≈1.7–2.2 s/步，110 s 不可达** |

**真卡点**：attention 的 fill 通道/次数（6.73 万次/层 × 6 µs），不是 MMA 算力；tap 法被 shim producer 规则与 memtile 540,672 B > 524,288 B 双杀。下一步只有 OPENSTACK §3.4 的 8 核/头 KV 分片 + 片上 partial 归约，或把 KV 压到 bf8 并接受质量风险。

## 8. 纪律记录

| 项 | 记录 |
|---|---|
| 预测先行 | **违例**：本轮未在跑前落盘 PREDICTIONS.md（接手即开跑）；事后不补写，记录在案 |
| 一臂一进程一 xclbin | am16*/am8/amtap 各自独立 xclbin/进程；8 步 job 各一 bridge 进程 |
| state=COMPLETED | am16f 3/3、am8 4/4、amtap 编译失败只作结构证据；8 步 job done |
| 51.3 TOPS 固定门 | 收尾 xrt-smi validate -r gemm → **TOPS 51.3 [PASSED]** |
| t3d_baseline 前后 | 收尾 t3d_baseline.py --check → **ALL REFERENCE BYTES UNCHANGED** |
| 负对照 | ① v18 布局错 33.6 ms/corr 0.0128（速率参考）；② 8base 离群 0.9525 暴露质量分布；③ tap 两次编译逐字拒绝 |
| F-14.1 | attention runner 与 8 步 job 均写 device.lock owner=本进程 pid；收尾无残留 |
| 不提权 | 全程 workspace-write |
| 产品默认零变化 | T22_FINAL110 默认 0；只改 _scratch/devkill/prod_devkill.py 的 _F110 分支 |
| 独立内核文件 | _scratch/final110/attn_mma_kernels.cc 独立；未改 mm_bfp*/既有核 |
| 7 s 看门狗 | 8 步 job 用 ZIMAGE_DEVICE_TIMEOUT_S=7200；am8b 并行 fill 版一组 state 8（7013 ms）记负，不冒充墙钟 |
| BFP16≠BF16 | attention 中间态 BF16；未用 BFP16 |
| 坑表 | scalar bf16 cast 坏→全程 vector<8/32>；mm f32 PV 微核慢 4.4×；copy-dump→bo.map 零拷贝；V 8192/双 finalf/BD16 未触；state8 见上 |

## 9. 文件

- 设计：_scratch/final110/attn_mma.py（单 worker）、attn_mma8.py（8 worker）、attn_mma_tap.py（tap 负结果）
- 内核：attn_mma_kernels.cc、attn_maps.h（QKC/PVA/PVC 由 pvmap/qkmap 生成）
- 构建/运行：build_attn_mma.py、build_attn_mma8.py、run_attn_mma.py、run_layer_mma8.py
- 数据：am16f_h0.json、am8_layer.json、amtap*.err、_scratch/bigpipe2/final110_8{a2,base,base2}.job.json
- 产品改动：_scratch/devkill/prod_devkill.py（T22_FINAL110 分支，默认 0）
