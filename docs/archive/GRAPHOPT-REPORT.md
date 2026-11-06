# GRAPHOPT-REPORT — 出图基线复原 + 图结构优化（t22_graphopt）

> 2026-10-30 · 工作区 `<checkout>` · HEAD `725b5666`
> **状态：BLOCKED（本会话无法创建任何进程）。本报告没有本轮器件墙钟、没有本轮出图。**
> 所有器件数字都标了出处（上一轮实测）；所有带"本轮"字样的数字都是我在磁盘上重新算的哈希/字节。

---

## 0. 结论（先给答案）

1. **执行被环境挡住**：`spawn` 对所有可执行文件返回 `EPERM`（含 `C:\Windows\System32\where.exe`、
   以及复制进工作区的 exe）。本会话只有 `run_code`（Node），没有 bash/terminal；提权被禁用且被要求不提权。
   ⇒ py312 / py310 / webui / xrt-smi / aiecc / git 全部无法启动，**没有 state=COMPLETED，就没有墙钟**。
2. **第一步的容器不用重跑**：任务书说 `t3real_prep` 已被清，但本快照里它在，**60 个文件、
   3,981,312,000 B，60/60 的 payload sha16 与 `t3real_prep.json` 完全一致**（§2 表 6）。
   重跑 `--stage prep` 只会把 4.5 GB 原样重写一遍。
3. **基线字节全绿**（本轮离线重算）：`t3d_baseline` 三类 25 个文件 **0 变 0 缺**；`weights.bfp16`
   506 张量、零缝零叠、整文件 sha256 `4ff5b6a6…cb8042`；`wpack_l30_v2.bin` sha256
   `da2f45a6…67c6`（与 prewarm 日志一致）。
4. **图优化的账（核心结论，纯离线可算）**：一个 FFN 块步的 DRAM 字节 = **189.0 MB**，其中中间态
   **51.9 MB（27.4 %）**——大头不是"写一次读一次"，而是 **A_norm 被 40 个 pass 重读（29.5 MB）** 和
   **sw 被 10 个 round 重读（19.7 MB）**。全免的上限 = **4.99 ms（24.6 %）**。
5. **但墙钟大概率不动**：GU 段已经跑到 kgu=64 常驻微基准的 **95.5 %**（U4-DOSE §3），只剩 4.5 % 余量
   ⇒ A_norm 驻留的收益上限 **0.45 ms（2.2 %）**；down 段的 47 MAC/cyc 是 **k=8 的调用开销**形状线，
   不是带宽。**预期合计 0 ～ −0.5 ms，点估计 −0.2 ms（−1 %）。**
6. **而且放不下**：D1 实际用了 **7/8 个 memtile**（文档 §1.2 写"6 个"但列了 7 项，本报告 §4.4 更正）。
   只剩 1 个空位；A_norm 整块驻留要 2 个（释放拐角 (0,1) 后净 +1 → 8/8），sw 直连还要 1 个 → **9/8，撞墙**。
   "两段都不落地"在 8 memtile 预算内**不成立**，必须二选一或半驻留。
7. **产品默认行为零变化是字面事实**：本轮没有改任何产品文件（`git status` 级别的证据见 §9 文件清单）。

**一句话给负责人**：这单现在缺的不是设计，是**一台能起进程的机器**；而在那台机器上，
"中间态不落地"按内核天花板推算的**现实上限只有 2–3 %**（字节账的 24.6 % 只有带宽受限时才成立），
建议先花 0.5 天做 §4.7 的判定实验，再决定要不要投 3–5 天的内核重写。

---

## 1. 环境阻塞（逐字证据）

| 探测 | 结果 |
|---|---|
| `spawnSync("C:\\Windows\\System32\\where.exe")` | `status null, error spawnSync ... EPERM` |
| `spawnSync(process.execPath, ["-e","..."])` | 同上 `EPERM` |
| 把 `where.exe` 复制到工作区再 spawn | 同上 `EPERM` |
| `execSync("where py312")` | `spawnSync cmd.exe EPERM` |
| 本会话工具 | 只有 `run_code`；`globalThis.tools` 不存在；无 bash/terminal |
| 审批 | `approval prompts are disabled`；子代理权限启动时固定，**不得提权** |
| 端口 7870 / 7874 | 均 `ECONNREFUSED`（没有遗留服务在跑）；2849 是 DSH GUI |

结论：**不是路径、不是 ACL、不是缺依赖**，是进程创建被策略拒绝。任何"跑一下"的方案都不成立。
写文件是通的（本报告与 `_scratch/t22_graphopt/*` 都是本轮写出来的）。

---

## 2. 基线复原账（本轮离线核对，全部通过）

| # | 件 | 字节 | 本轮检查 | 结果 |
|---|---|---:|---|---|
| 1 | `zimage-port/weights.bfp16` | 6,913,965,970 | 506 张量；header 87,946 B；payload 6,913,878,016 B | **0 缝 / 0 叠 / 无越界**；sha256 `4ff5b6a66ea3b3ae5a29ecc601097d50eade61979c222f8e0c58d3a272cb8042` |
| 2 | `te_npu_weights.bin` | 4,088,254,464 | 文件存在、尺寸对（3.807 GiB） | OK |
| 3 | `_models/Z-Image-Turbo/text_encoder` | 8.045 GB | 3 分片 + `model.safetensors.index.json` + config | OK |
| 4 | `_models/Z-Image-Turbo/vae` | 0.168 GB | config + safetensors | OK |
| 5 | `_models/Z-Image-Turbo/tokenizer` | 0.016 GB | 4 文件 | OK |
| 5b | `_models/Z-Image-Turbo/transformer` | 0（无权重） | `config.json` + `index.json`（48,969 B，521 张量） | **与任务书一致：权重未下**；config 复核 dim 3840 / heads 30 / layers 30（+2+2 refiner = 34 个 block 实例） |
| 6 | **`_scratch/image/t3real_prep`** | **3,981,312,000** | 30 层 × 2 文件 = 60；**payload** sha16 对 `t3real_prep.json` | **60/60 全同**（建 2026-10-22 06:13–06:17） |
| 7 | `webui/cache/wpack/wpack_l30_v2.bin` | 1,990,656,000 | 120 slots；sha256 | `da2f45a62b0e7e48…50167c6`（== `prewarm_boot.log`） |
| 8 | `webui/cache/wpack/wpack_l30_v2.bin.json` | 5,978 | 30×4 slot 表 | OK |
| 9 | `t3d_baseline` 三类引用 | 25 文件 | 逐文件 sha256 重算 | **0 changed / 0 missing** |
| 10 | FFN 交付镜像 `step3_M96_…_L10.xclbin` | 174,251 | sha256 对 manifest | **MATCH** |
| 11 | D1 单镜像 `out/d1_ffn.xclbin` / `.bin` | 82,176 / 2,908 | 尺寸 + 17 个 `opted_main_core_*.ll` | 与 D1 文档一致 |

> 第 6 条是**与任务书冲突的**：任务书写"t3real_prep 30 层容器已被清（4.5 GB）"。
> 我这份快照里它**在且全对**。如果你那边确实清过，以你为准；但在我这里跑 `--stage prep` 是纯浪费。

### 2.1 D1 存档产物的独立复核（本轮，非重跑）

| 检查 | 结果 |
|---|---|
| 同进程 `Y_r1.bin` vs `Y_r2.bin` | 都是 `0dbc6884…`（**第二次 launch 的假结果**，F-01/U4 §7 的已知现象） |
| 跨进程 fused vs 三 dispatch 链 | `Y_c0/c1/c2` = `Y_chain_def` = `Y_u0/u1/u2` = `final_y` = `Y_d1a` = `e2c5b24e37ce4907`（**逐字节 0/737,280**） |
| `final_sw` vs `sw_alin` vs `sw_turn` vs `SW_d1a` | 全是 `77c7de04f91b7b9f`（1,966,080 B） |
| `Y_u3` / `Y_u4`（UNROLL≥3 剂量） | `c2cd1bef…`，diff **733,039/737,280**，state 8 —— 与 `U4-DOSE.md` §4 一致（**负结果：sequence 放大会死**） |

---

## 3. 出图基线口径（本轮未复跑）

**配方**（产品默认路径，与 `webui/jobs/20261021-081535-a3db76` 同参）：

    prompt  = "a mountain lake at sunrise"
    height/width = 128 / 128        steps = 8      seed = 7
    vae = true (CPU 解码)            mode = trial
    qkv / p10 = 默认开               FFN 镜像 = fs_ffn_j5（默认）

**期望字节**（F-23 口径：128² 不做换参考，仍是逐字节）：

| 产物 | sha256 | 字节 |
|---|---|---:|
| latents `.npy` | `9f86aee91de3dad09bf888b15fe560e16f9c3079920118d714697f25421dbfe2` | 16,512 |
| PNG（CPU VAE） | `104ff564709c017848f2b433fe37e602558178da32729035c27552820500219a` | 21,318 |
| PNG（NPU VAE 链，**不可与上者互比**） | `e7aef10c…` | 24,382 |

> **本轮离线核对**：这两串不是只活在文档里——工作区里能直接找到对应产物：
> `_scratch/image/w14465543340cp0_latents.npy`（16,512 B，sha256 全串 = `9f86aee9…dbfe2`）、
> `webui/evidence/t22_closurefix/out/cf_fallback_cold_2126442228c7_p0_s7.png`
> （21,318 B，sha256 = `104ff564…219a`；IHDR 128×128，8 bit，RGB）。

**参考墙钟（引自 `_scratch/t22_e2e_measure/SUMMARY.md`，不是本轮）**：steady job total **40.82 s**
（device 12.34 / loop 21.24 / 960 dispatch / 0 retry）；cold **57.16 s**。

| 臂 | 本轮墙钟 | 本轮质量门 | 状态 |
|---|---|---|---|
| 基线 128²/8步/seed7 | 未测 | 未测 | **阻塞** |
| D1 单镜像（现形态） | 未测（上轮 20.287 ms） | 上轮 Y 逐字节 0/737,280 | 上轮实测 |
| graphopt memtile 版 | 不存在 | — | 仅预算（§4） |

---

## 4. 图结构优化（核心）：预算、墙、预测

### 4.1 现状（D1 已证，本轮只复核产物）

* `d1_ffn` = 17 核、一条 runtime sequence、三相位（norm → gu+swiglu → down+add），**一次 dispatch**；
  xclbin 82,176 B / insts 2,908 B。
* 相位间**已经不上 host**：phase1 的 drain 写 A BO、phase2 的 fill 从同一 A BO 读；sw 同理。
  单镜像 20.287 ms vs 三 dispatch 链 21.329 ms（省 1.042 ms，−4.9 %），其中 host 缝合 1.423 → 0.788 ms。
* **所以"host 缝合"这块钱已经赚过了**；本轮要动的是**片内那两段 BO 往返**（A_norm / sw）。

### 4.2 中间态字节账（`py312 _scratch/t22_graphopt/byte_budget.py` 可复算）

| 项 | 字节/块步 | 说明 |
|---|---:|---|
| A in / SBG / X / Y | 737,280 / 2,211,840 / 737,280 / 737,280 | 激活 I/O |
| BG（w1,w3）+ B3（w2） | 88,473,600 + 44,236,800 | bfp16 权重，每块步都要从 DRAM 流一次 |
| **A_norm 写** | 737,280 | 相位 1 → A BO |
| **A_norm 读** | **29,491,200** | **40 个 pass × 737,280**（`a_slice_tap…repeat(passes)`） |
| **sw 写** | 1,966,080 | 相位 2 → sw BO |
| **sw 读** | **19,660,800** | **10 个 round × 1,966,080**（每个 round 都要整行 k=10240；fill 一次、6 核靠 fifo 广播） |
| **合计** | **188,989,440** | ≈189.0 MB → 20.287 ms 上平均 **9.32 GB/s** |

* 中间态 51.9 MB = **27.4 %** 的 DRAM 字节；权重 70.2 %；其余 2.4 %。
* **全免的算术上限 = 20.287 × 27.4 % = 4.99 ms**（A 3.09 + sw 1.90）。

### 4.3 为什么"不落地"大概率动不了墙钟

* **GU 段已经贴着内核天花板**：183 GMAC/s/核 = 114 MAC/cyc = kgu=64 常驻微基准（382 GFLOP/s/核）
  的 **95.5 %**（`U4-DOSE.md` §3）。也就是说，A 的 29.5 MB 重读**最多只值它剩下的 4.5 %** ≈ **0.45 ms**。
* **down 段 47 MAC/cyc** 的死因是 **k=8 的每调用开销**（1,280 次/round），不是带宽；这条线被负责人锁死
  （k→64 会换 C 的舍入链 = 换参考）。
* 器件平均带宽只有 9.32 GB/s（按整机 120 GB/s 量级算 ≈8 %），**离饱和很远 ⇒ 删流量不等于删时间**。
* 结论：这是**延迟/内核形状受限**，不是带宽受限。中间态优化的合理预期是 **0 ～ −0.5 ms（−2.5 %）**。

### 4.4 memtile 预算：文档说 6/8，代码是 7/8（本轮更正）

`d1_ffn.py` 里实际引用的 memtile：

| memtile | 用途 | in/out |
|---|---|---|
| (0,1) | A 拐角（row-major → blocked） | 1/1 |
| (1,1) | sw join（4 核 → 1 流） | 4/1 |
| (2,1) | X split（1 → 6） | 1/6 |
| (3,1) | BG hop ×4 | 4/4 |
| (5,1)(6,1)(7,1) | C3 forward ×6 | 2/2 each |

**7 个**，不是 `D1-PROTOTYPE.md` §1.2 写的"用 6 个"（那行列了 7 项，是笔误）。**空位只有 1 个。**

| 方案 | 片上驻留 | memtile 净增 | 总占用 | DRAM 省 | 上限 |
|---|---|---:|---:|---:|---:|
| S0 现状 | — | 0 | 7/8 | — | — |
| S1 A_norm 全驻留（60 slice 预 blocked，释放拐角） | 737,280 B（2 tile） | +1 | **8/8** | 28.75 MB（15.2 %） | 3.09 ms |
| S2 sw 直连（join → 6 核 fan-out） | 49,152 B ×2（1 tile） | +1 | **8/8** | 17.69 MB（9.4 %） | 1.90 ms |
| S3 S1+S2 | — | +2 | **9/8 ✗** | 46.45 MB（24.6 %） | 4.99 ms |
| S3' A 半驻留 + S2 | 368,640 + 98,304 | +2 | **9/8 ✗** | ~32 MB | 3.5 ms |

**"两段都不落地"在 8 个 memtile 里放不下**——这是第四道墙（容量+数量），而且是**编译前就能算出来的**。

### 4.5 三墙数据点（D1 实测）+ 新设计预算

| 墙 | D1 实测占用 | 新设计（S1） | 判 |
|---|---|---|---|
| shim 16 通道 | 15 MM2S / 8 S2MM | 14–15 / 8（驻留只是把 fill 的目标从核换成 memtile） | 基本不变 |
| memtile 8×(6in/6out) | **7/8**，最紧 (1,1)=4in/1out、(2,1)=1in/6out | **8/8**，S1 占满 | **贴墙**；S1+S2 直接超 |
| BD ≤16/tile | phase2 6 task、phase3 14 task | phase2 6→7、phase3 14→13 | 仍 ≤16 |
| **第四墙：memtile 容量 512 KB** | 单对象最大 49,152 B（join） | **A_norm 737,280 B = 1.41× 单个 memtile** | **必须拆 2 片** |

补充（API 风险，R1）：IRON 的 `ObjectFifo` 是**单趟**语义——一个对象 acquire 一次就没了。
"A 载入一次、40 个 pass 重读"**不能**用同一个 fifo 表达；要么 40 次 memtile→memtile 重放（序列要 2,400 个
DMA task），要么退到 `aie.dma_bd` 手写。**这是开始写代码前必须先 spike 的第一件事**（F-19 方法卡）。

### 4.6 预测（已写死在 `_scratch/t22_graphopt/GRAPHOPT-PREDICTIONS.md`）

| # | 预测 | 置信 |
|---|---|---|
| P2 | A_norm 整块当一个 memtile 对象**编不过**（737,280 > 524,288） | 0.95 |
| P3 | A_norm 驻留收益 **≤0.45 ms（≤2.2 %）** | 0.80 |
| P4 | sw 直连收益 **≤1.9 ms 字节上限**；因 down 是 k=8 调用开销受限，**预期 ≤0.3 ms** | 0.60 |
| P5 | 合计 **0 ～ −0.5 ms**，点估计 **−0.2 ms** | 0.60 |
| P6 | 全设计 8/8 memtile、shim 14–15/16、BD ≤16 | 0.55 |
| P7 | 改对象数/顺序的 sequence 重排会破 Y 逐字节（u3/u4 已证） | 0.80 |

### 4.7 先花 0.5 天做的判定实验（不写新内核）

**目的**：在投 3–5 天重写之前，用**已有镜像**把"中间态 DRAM 时间"称出来。

1. **pass 斜率**：`d1_compile.py --design d1_ffn --tag d1_p{1,10,20,40} --kw passes={1,10,20,40}`，
   每档一进程一发，量 wall。斜率 = 每 pass 的 (GU 计算 + A 重读 + BG 流)。
2. **A 重读的隔离臂**（计时用、数值不作数）：把 `a_slice_tap` 的 `.repeat(passes)` 去掉、
   GU 循环只 acquire 一轮（其余 pass 复用同一份 L1 数据）→ 若 wall 掉 ≤0.45 ms，P3 成立、**不值得做**；
   若掉 >1 ms，才值得进 S1。
3. **sw 侧**：`rounds` 固定，比较 `inA3` depth 1 vs 2 vs 3（只改 fifo 深度，不改布局）→ 看 fill 是否已经
   被 mm 调用遮住。
4. 全程 `t3d_baseline --check` 前后各一次；51.3 TOPS 门开工/收尾各一次。

**判据**：第 2 步 ≤0.45 ms ⇒ 本优化**不投**，把 3–5 天转去 down 的 k 形状（已锁）或 attention；
>1 ms ⇒ 再投 S1（单段，不是两段）。

---

## 5. 优化后出图（未执行）

未执行，原因见 §1。**没有"优化后"的 latents/PNG/墙钟，我不编。**
一旦有可执行环境，顺序是：S1 编译 → 单发 gate 0（state 4 + Y 逐字节）→ 成对 A/B（D1 vs S1，同 session）
→ 接产品旗标（默认关）→ 128²/seed7 复跑 → 对 `9f86aee9`/`104ff564` → 出对比图 → `t3d_baseline --check`。

---

## 6. 产品默认行为零变化声明

**本轮没有修改任何产品文件。** 新增的只有：

    zimage-bfp16-npu/GRAPHOPT-REPORT.md
    zimage-bfp16-npu/_scratch/t22_graphopt/GRAPHOPT-PREDICTIONS.md
    zimage-bfp16-npu/_scratch/t22_graphopt/byte_budget.py

`webui/`、`webui.py`、`_scratch/block/d1proto/`、`_scratch/image/` 一个字节都没动；
因此"产品默认行为零变化"是**字面事实**，不是承诺。默认路径仍然是 `fs_ffn_j5` + qkv/p10 开 + CPU VAE。

---

## 7. 下一步建议

1. **换执行环境**（或让有 bash 的会话接手）。命令见 §8；第一步不是跑 prep（已验），而是起 7874 复跑基线。
2. **先做 §4.7 的 0.5 天判定实验**，再决定投不投 S1。按现有证据我更倾向"不投"：
   现实上限 2–3 %，而 D1 的 dispatch 融合已经拿走了 4.9 %。
3. **真正的墙钟杠杆**（按证据强度）：
   a. down 的 **k=8 → 64**（47 → ~114 MAC/cyc，理论 −4 ms 量级）——但换 C 舍入链 = 换参考，负责人锁死，
      除非 F-23 的"换参考"裁决能覆盖到这一层；
   b. attention 段（24.9 s/图，占器件 24 %）；
   c. 多列/多 context 并发（8 列里我们只用了一列的历史问题，见 HANDOFF §2）。
4. **文档更正**：`D1-PROTOTYPE.md` §1.2 的 "memtile 用 6 个" 应改为 **7 个**（否则下一班会以为还有 2 个空位）。

---

## 8. 复跑命令（给能起进程的环境）

    cd <checkout>

    # 0) 字节账 + 基线门（只读）
    py312 _scratch/t22_graphopt/byte_budget.py
    py312 _scratch/image/t3d_baseline.py --check          # 期望 ALL REFERENCE BYTES UNCHANGED

    # 1) 基线出图（不要碰 7870）
    py312 webui.py --host 127.0.0.1 --port 7874
    # 另开一个 PowerShell 窗口：
    $body = '{"prompt":"a mountain lake at sunrise","height":128,"width":128,"steps":8,"seed":7,"vae":true,"mode":"trial"}'
    Invoke-RestMethod -Uri http://127.0.0.1:7874/api/jobs -Method Post -Body $body -ContentType 'application/json'
    # 轮询 /api/jobs/<id>，status=done 后核对：
    #   job.latents_sha256 == 9f86aee9…dbfe2
    #   images[0].sha256   == 104ff564…219a   (21,318 B)
    # 墙钟取 job.timing.total_s（同 session 连跑 2 发取第二发）

    # 2) 51.3 TOPS 固定门（开工/收尾各一次）
    xrt-smi validate -r gemm

    # 3) D1 现状复测（一进程一 xclbin）
    powershell -NoProfile -ExecutionPolicy Bypass -File _scratch/block/d1proto/d1_run1.ps1

**注意**：`t3_real.py --stage prep` 在本快照**不必跑**（§2 表 6）。

---

## 9. 文件清单 / 提交建议（本轮无法执行 git）

新增（未提交，建议 pathspec 提交）：

    git add -- zimage-bfp16-npu/GRAPHOPT-REPORT.md
    git add -- zimage-bfp16-npu/_scratch/t22_graphopt/GRAPHOPT-PREDICTIONS.md
    git add -- zimage-bfp16-npu/_scratch/t22_graphopt/byte_budget.py

本报告引用的既有证据（未改动）：`_scratch/block/d1proto/D1-PROTOTYPE.md`、`U4-DOSE.md`、
`_scratch/image/t3d_baseline.json`、`_scratch/t22_e2e_measure/SUMMARY.md`、
`webui/cache/wpack/prewarm_boot.log`。
