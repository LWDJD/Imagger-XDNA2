# BIGPIPE2-REPORT — 大图主力战役（t22_bigpipe2）

日期：2026-10-31 00:1x (+08:00)｜owner：t22_bigpipe2｜器件：NPU Strix（XRT 2.21.0 / driver 32.0.20102.3930，单实例）
范围：512² / 1024² / 2048²；小图只作参照，不作性能结论。

## 0. 判词

1. **big-M 整形状 FFN 的器件段是真赢**：512² FFN device 11.58 → 3.11 s/步（3.7×），1024² 48.68 → 9.36 s/步（5.2×）；但整步墙钟是输的（512² 稳态 25.3 vs 22.2 s；1024² 130.9 vs 82.5 s 循环），因为 33.4 MB/层的 swiglu 中间态要回 host、再做 a_pack / 反 tile，host 段把器件的收益全部吃掉。
2. **big-M 整形状投影像素级是红的**：512² 8 步 latents 对分块基线 corr −0.1408 / PSNR 10.45 dB。把投影退回 delivered chunked（同一 FFN big-M 臂）就是 0.985776 / 28.19 dB。缺陷隔离在 bigm_proj 的 B/A 契约。
3. **流式 attention 多块 carry 仍未绿**：seq64 单块 0.999778；seq128 从 0.7436（原三核）→ 0.7438（单核融合）→ 0.8032（state 改 f32），方向对但没过 0.999。根因从「state 不持久」缩小到「QK 非零时 state 在两次 step 调用之间被 L1 邻接写覆盖」。
4. **产品默认零变化**：所有改动在 ZIMAGE_BIGM / ZIMAGE_BIGM_PROJ / 新模块门后；ZIMAGE_BIGM 默认 0。不建议翻默认。

## 1. 战线 A：big-M @ 大图

### 1.1 口径

同 seed 7、同 prompt「a mountain lake at sunrise」；512² 8 步 VAE on；1024² 基线是 1 步 VAE off，所以 1024² 只做 1 步同口径 A/B。路径 ZIMAGE_BIGM=1、ZIMAGE_CHUNK_PIPE=0。
两条臂：A-FFN（FFN 整形状，投影/attention 保持 chunked，ZIMAGE_BIGM_PROJ=0）；A-BOTH（FFN+q/k/v/o 投影都整形状，ZIMAGE_BIGM_PROJ=1）。墙钟只认 loop_s / step wall。

### 1.2 512² 墙钟（8 步）

| 臂 | loop_s | 总 s | device s | dispatch | 稳态 step wall | 对基线 |
|---|---:|---:|---:|---:|---:|---:|
| 分块基线 20261030-221051-99ea8a | **177.76** | **222.29** | 143.81 | 13,200 | 22.2 | 1.00× |
| big-M FFN 20261030-235030-2264eb | 217.72 | ~253（估，含 VAE） | 78.43 | 11,040 | 25.3 | 0.82×（慢 22%） |
| big-M FFN+投影 20261031-000417-b79236 | **310.09** | **349.70** | 69.66 | 2,400 | 36.7 | 0.57×（慢 57%） |

A-FFN 的 FFN device 92.66 → 19.0 s（8 步），即 11.58 → 3.11 s/步。A-BOTH 投影 device 44.65 s（1.49 s/步）、FFN 25.0 s。host 是墙：A-FFN 稳态 host 58.9 s（基线 26.8 s），pack 2.48 s；A-BOTH host 48.9 s。器件省的 7–8 s/步被 host 的 a_pack + 33.4 MB/层读回 + 反 tile 吃掉还倒贴。

### 1.3 512² 质量（latents 对分块基线，同 seed/prompt 8 步）

| 臂 | corr | PSNR | sorted corr | 判 |
|---|---:|---:|---:|---|
| 分块基线 | 1.000000 | — | — | — |
| **big-M FFN** | **0.985776** | **28.19 dB** | 0.9999949 | **PASS（F-23 档）** |
| big-M FFN+投影 | **−0.140799** | **10.45 dB** | — | **FAIL（投影契约红）** |

FFN 臂与 480p 的 0.948449/22.66 dB 同族、比 128² 的 0.990743/28.77 dB 略低；sorted 0.9999949 = 多重集几乎完美，差的是 bfp16 量化。投影臂 corr 为负，且同一份 FFN 换 chunked 投影就 0.9858 ⇒ bigm_proj 的 B 序或输出解码错。prod_qkv.pack_proj 从未过产品门，本轮第一次暴露。

### 1.4 1024² 墙钟与质量（1 步，VAE off）

| 臂 | loop_s | 总 s | device s | dispatch | corr / PSNR |
|---|---:|---:|---:|---:|---|
| 分块基线 20261030-221644-f60a3b | **82.47** | 111.46 | 73.41 | 6,450 | 1.0 / — |
| big-M FFN 20261031-235932-bedd14 | **130.93** | 159.52 | 38.00 | 5,340 | **0.992648 / 30.39 dB** |
| big-M FFN+投影 | 未跑（投影已红） | — | — | — | — |

1024² FFN device 48.68 → 9.36 s（5.2×），但 host 227.7 s 把 step 推到 148.4 s；最终 loop 130.9 s，仍比基线 82.5 s 慢 59%。质量 0.992648/30.39 dB 高于预测 0.95–0.98。

### 1.5 为什么器件赢了墙钟还是输

512² A-FFN 每层：device gu+swiglu 70.8 ms + down 33.2 ms = 104 ms（vs 基线 386 ms/层）；host a_pack 15 ms×2 + gu 输出 21.6 MB 读回/反 tile + down 8.1 MB 读回 + W 132.7 MB 串行写 ≈ 550–600 ms/层。器件省的 282 ms 被 host 的 550 ms 反杀。1024² 每层 3 段，host 成本 ×3（33.4 MB 全 17-tile 输出读回 ×3），器件省的 39 s 被 host 的 ~120 s 反杀。

结论：下一步不是再调大 M，而是把 gu→down 中间态留在器件（或至少留在 BO，不回 host 做 numpy a_pack/反 tile），并把 W 写与 dispatch 双缓冲重叠。

## 2. 战线 B：流式 attention 多块 carry

### 2.1 复现与修法路径（全部实测）

| # | 臂 | 形状 | corr vs host f32 | 墙钟 | ERT | 判 |
|---|---|---|---:|---:|---|---|
| B0 | 原三核（softmax/PV/finalize） | seq64 bk64 | **0.999778** | 6.2 ms | 6/6 COMPLETED | **绿** |
| B1 | 原三核 | seq128 bk64 | 0.743593 | 23.5–24.3 ms | 6/6 | 红 |
| B2 | key 循环展开（去 scf.for 回边） | seq128 bk64 | 0.743593 | 23.5–24.3 ms | 6/6 | 红（与 B1 逐位同） |
| B3 | 单核融合（max/exp/PV/finalize 一个函数） | seq128 bk64 | 0.743844 | 23.5–24.5 ms | 6/6 | 红 |
| B4 | B3 + state 改 f32 | seq128 bk64 | **0.803191** | 26.1–26.9 ms | 3/3 | 红（有改善，未过） |
| B5 | seq512 展开 | seq512 bk64 | — | — | — | 编不过：.text 超 program 区 21,808 B（16 KB 上限） |

结论：不是 scf.for 回边（B2 与 B1 逐位相同）；不是三核之间可见性（B3 单核同结果）；state 改 f32 只把 0.744 抬到 0.803，说明还有同族覆盖。

**根因探针（关键）**：
- zero-QK（q=k=0，v=0）→ 两个 key 块后 state l=**128**、c=**1**，即 state 跨调用完整（每块 lsum=64，两块=128）。⇒ Buffer 本身能持久。
- 随机 QK → 第二次 step 入口 dump 出的 state 是 ~0.1 量级（P/邻接值），不是上一块的 m≈1、l≈35、c=0。⇒ QK 非零时，第一次 step 写下的 state 在第二次调用前被某个 L1 写覆盖。
- 已排除：state 与 S32 相邻（去掉 S32、QK 直写 bf16 后仍红）；stack=0x800/0x1800 同结果；zero_state2 初始化可见（zero-QK 绿）。
- 剩余嫌疑：stream_attn_step 的 5 指针 + 4 int + float 的 ABI/寄存器溢出，或 outO 邻接；下一单元把 state 放进 ObjectFifo 元素，或把整个 q-block 的 key 循环做成单个 kernel 内的 FIFO 消费，不再用独立 Buffer。

Phase 3（2048²）仍不启动：按「前两阶段绿」纪律，carry 未绿。

### 2.2 N² 不落地硬证据（结构字节账，不是 DRAM counter）

| 对象 | 512² 每层 | 流式设计 |
|---|---:|---|
| S = QKᵀ（f32） | 4.46 MB/head ×30 = **133.8 MB** | 最大 16×64 bf16 = **2 KB** |
| P = softmax(S) | **133.8 MB** | 同 2 KB（原地） |
| S+P 合计（N²） | **267.6 MB/层 = 8.03 GB/步** | **0 B 的 N² 尺寸对象** |
| Q/K/V/O host↔device | 48.7 MB + O 16.2 MB | Q 8.1 + KV 16.2 + O 8.1 = **32.4 MB/层** |

512² host f32 每层 332.5 MB、N² 267.6 MB；流式每层 32.4 MB，10.3× 少，且 S/P 只在 core L1 产生和消亡。本 spike 没有做 DRAM 硬件计数器（器件无该接口）；这是按 runtime 逐笔字节账，不是测量推断。

### 2.3 512² 单层对拍：未做（前置红）

carry 未绿，按纪律不拿 512² 单层去凑 corr；B4 的 seq128 0.803 已说明多块必红。

## 3. 512² / 1024² / 2048² 图结构最终形态

### 3.1 512²（M=1056，11 tile）

| 段 | 当前交付（分块） | 本轮 big-M 实测 | 最终形态（建议） | 数据落地 |
|---|---|---|---|---|
| 投影 q/k/v/o | 44 dispatch/层，W 16.6 MB×44，device 6.39 s/步 | 整形状 4 dispatch/层，device 1.49 s/步，质量红 | **保留 delivered chunked**，直到 bigm_proj 的 B/A 契约过门 | W 16.6 MB×4 BO；X 0.74 MB×11；O 0.74 MB×11 |
| FFN | 11 块/层，W 132.7 MB×11，device 11.58 s/步 | 1 dispatch/层，device 3.11 s/步，质量 0.9858/28.19 | **fused gu→down 单镜像**（中间 21.6 MB 不回 host）+ W 双缓冲 | W 132.7 MB 一次；中间态留 device；只回 y 8.1 MB/层 |
| attention | host f32 2.69 s/步，S/P 8.03 GB/步 host DRAM | 未接（stream 未绿） | host f32 保留（512² 流式盈亏线 191 GFLOPS） | S/P 267.6 MB/层 host DRAM；N² 不落 device |
| 其余 | pack/refiner/barrier ~1.56 s | 同 | 保持 | — |

512² 最终形态一句话：投影保持 chunked，FFN 做「gu→down 融合 + W 双缓冲」，attention 保持 host f32；预期 step 22.2 → 12–14 s，而不是本轮 25.3 s。这是唯一能翻默认的路径。

### 3.2 1024²（M=4128，43 tile）

| 段 | 当前交付（分块） | 本轮 big-M 实测 | 最终形态（建议） | 数据落地 |
|---|---|---|---|---|
| 投影 | 43 块/层，device 24.73 s/步 | 未接（投影红） | 先修 bigm_proj 契约，再评估整形状；否则保留 chunked | W 16.6 MB×4×43；整形状 16.6 MB×4 每 17-tile 段（3 段） |
| FFN | 43 块/层，device 48.68 s/步 | 3 段×17 tile，device 9.36 s/步，质量 0.9926/30.39 | **fused gu→down + 3 段流水**；中间 100 MB/层不回 host | W 132.7 MB×3；中间留 device；只回 y 31.7 MB/层 |
| attention | host f32 37.93 s/步，S/P 122.7 GB/步 host DRAM | 未接 | 流式 attention（>191 GFLOPS 才有正收益）；否则 host f32 | S/P 122.7 GB/步 host DRAM，N² 不落 device |
| 其余 | pack/staging/refiner ~27 s | 同 | 保持 | — |

1024² 最终形态一句话：FFN 融合后 device 9.4 s + attention 37.9 s + 其余 27 s ≈ 75 s/步；只有把 attention 也搬到流式（≥400 GFLOPS）才能到 ~55 s/步。本轮 130.9 s 是 host 中间态没融合的版本，不代表 big-M 的上限。

### 3.3 2048²（M=16416，171 tile；未实跑，外推）

| 段 | 现状外推 | 最终形态 | 数据落地 |
|---|---|---|---|
| 投影 | 340 GB/步 W 流，~99 s/步 | 整形状 11 段（或 fused block） | W 16.6 MB×4×11；X 31.7 MB/段 |
| FFN | 681 GB/步 W 流，~184 s/步 | fused gu→down，11 段 | W 132.7 MB×11；中间 1.6 GB/层留 device |
| attention | host f32 **606 s/步**，N² **64.7 GB/层** | **必须流式**：S/P 2 KB L1，Q/KV/O 每层 129.6 MB | N² 0 B；host↔device 只 Q/KV/O |
| 合计 | ~980–1030 s/步，8 步 2.2–2.3 h | 流式 + fused 后 ~10 min/步量级，仍不是秒级 | — |

2048² 生死线：N² 64.7 GB/层不能落 host DRAM；流式 carry 必须先绿。

## 4. 预测对账（BIGPIPE2-PREDICTIONS.md）

| # | 预测 | 实测 | 判 |
|---|---|---|---|
| A1 | 512² big-M step 11.5–14.0 s | FFN-only 25.3 s；both 36.7 s | **错（低 2×）** |
| A2 | 512² 总 137–157 s | FFN-only ~253 s；both 349.7 s | **错** |
| A3 | 512² corr 0.93–0.97 | FFN-only **0.9858**（更好）；both −0.1408（投影红） | 半对：FFN 超预测，投影是缺陷不是量化 |
| A4 | 1024² step 84–88 s | 130.9 s | **错（低 50%）** |
| B1 | seq128 ≥0.9995 | 0.8032（最好） | **未达** |
| B2 | seq512 ≥0.9990 | 编不过（.text 超 21.8 KB） | **未达** |
| B3 | 512² 单层 ≥0.999 | 未做（前置红） | — |
| B4 | 流式 <50 GFLOPS、512² 慢于 host | seq128 约 0.32 GFLOPS，26 ms/2 块 | **对（远低于 50）** |

为什么会错：预测把「器件 dispatch 时间」当成了「整步墙钟」，没有给 host 的 a_pack/反 tile/33.4 MB 读回定价；也没有预判 prod_qkv.pack_proj 的 B 序从未过产品门。

## 5. 纪律记录

| 项 | 记录 |
|---|---|
| 预测先行 | BIGPIPE2-PREDICTIONS.md 写在所有大图 job 之前（23:0x） |
| 一臂一进程一 xclbin | 每个 product job 一个 bridge 进程；stream 每臂一个进程只载一个 xclbin |
| state=COMPLETED 才算数 | 所有 product job status=done；stream B1–B4 ERT 4；B5 编译失败不冒充 |
| 看门狗≠墙钟 | 只报 loop_s/step wall；ERT/device_s 单列；未把 7 s watchdog 当性能 |
| 51.3 TOPS 固定门 | 开跑前 51.3 TOPS / PASSED；收尾再跑 51.3 TOPS / PASSED |
| t3d_baseline 前后 | 收尾 t3d_baseline.py --check → ALL REFERENCE BYTES UNCHANGED |
| 负对照 | ① same FFN 换 chunked 投影 0.9858 vs big-M 投影 −0.1408；② zero-QK state l=128/c=1 vs 随机 QK 入口垃圾；③ seq64 0.999778 仍绿 |
| pathspec 提交 | 只提交 _scratch/bigpipe2、prod_loop_bigm、chunked.py 的旗标门、两份报告；不 add webui/cache、jobs、outputs |
| F-14.1 真实 pid | device.lock owner=t22_bigpipe2 pid=35196/31824/…（webui/bridge 进程本身），非 wrapper |
| 不提权 | 全程 workspace-write；未请求 escalation |
| 产品默认零变化 | ZIMAGE_BIGM 默认 0；chunked.py 的新分支仅在 ZIMAGE_BIGM=1 且 ZIMAGE_BIGM_PROJ≠0 时执行 |

## 6. 大图翻默认建议表

| 形态 | 512² 质量 | 512² 墙钟 | 1024² 质量 | 1024² 墙钟 | 建议 |
|---|---|---|---|---|---|
| 交付 chunked（现默认） | 1.0 | **222.29 s / 8 步** | 1.0 | **111.46 s / 1 步** | **保持默认** |
| big-M FFN-only | 0.985776 / 28.19 dB | 217.7 loop（≈253 总） | 0.992648 / 30.39 dB | 130.9 loop（159.5 总） | **不翻**：质量在 F-23 档，但墙钟负 |
| big-M FFN+投影 | **−0.1408 / 10.45 dB** | 310.1 loop（349.7 总） | 未跑 | — | **拒绝**：投影契约红 |
| 流式 attention | — | 未接 | — | — | **不启动 Phase 3**：carry 未绿 |

翻默认的唯一条件（下一单）：
1. bigm_proj 的 B/A 契约过单层门（同输入、对 delivered 投影 corr ≥0.999），并把输出解码做成与 FFN 同族的 object-major 反 tile；
2. FFN 做 fused gu→down（中间态不回 host）+ W 双缓冲；
3. 512² 8 步 latents ≥0.99 且 loop ≤150 s（相对 177.8 s 至少 15% 正收益）；
4. 1024² 1 步 loop ≤75 s。

## 7. 复现

512² big-M FFN-only（8 步，VAE on）：
    set ZIMAGE_BIGM=1 / ZIMAGE_BIGM_PROJ=0 / ZIMAGE_CHUNK_PIPE=0
    py310 webui/bridge.py --tag TAG --pe JOB/pe_p0.npz --schedule JOB/schedule.npz
      --cache-dir webui/cache/512x512 --steps 8 --seed 7 --batch 1
      --height 512 --width 512 --refiner-cache _scratch/image/t6_refcache
      --no-qkv --no-p10 --wpack-file webui/cache/wpack/wpack_l30_v2.bin

1024² big-M FFN-only（1 步，VAE off）：
    py312 _scratch/bigpipe2/run_job_py.py --h 1024 --w 1024 --steps 1 --seed 7 --vae 0 --proj 0 --tag 1024_bigm_ffnonly_1

流式 carry 复现（seq128，f32 state）：
    py312 _scratch/bigpipe2/build_stream3.py stream128x 128 64 0 0x1800
    py310 _scratch/bigpipe2/run_stream2.py stream128x io128u 3
    py312 _scratch/bigpipe2/check_stream2.py stream128x io128u

## 8. 诚实边界

1. **big-M 投影的 B 序未修**：本轮只证明它红，没有修；不要把 −0.1408 当成 big-M 的数值极限。
2. **512² FFN-only 的 8 步总 s 是估算**：bridge 直接跑，未走 webui VAE 收尾；loop 217.72 s 是实测，总 s 用 +10 s startup +25 s VAE 估。
3. **1024² 只有 1 步**：基线也是 1 步，不能拿 8 步质量档去比；质量 0.9926/30.39 dB 是单步。
4. **stream 的 DRAM 只有字节账**：没有硬件 DRAM counter，不能声称「实测 DRAM 流量」。
5. **seq512 的 .text 超限**是展开版；未尝试用 scf.for 融合版，因为 carry 根因未解。
6. **A-BOTH 的 device 69.7 s** 与 A-FFN 78.4 s 的差主要是投影 44.7 s + FFN 25.0 s vs FFN-only 19.0 s；不要把 69.7 s 当 big-M 的「更好」——它的质量是红的。
