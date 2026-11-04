# BOTTLENECK-MAP — 512²/1024²/2048² 卡点地图

日期：2026-10-30 22:20 (+08:00)｜owner：t22_bigpipe｜器件窗：webui 7874（单实例，seed 7）

## 0. 判词（先给结论）

1. **512² 的墙不是 attention**：attention 只占 step wall 的 **12.1 %**（2.69 s/步）；**FFN 权重流 52.1 %（11.58 s/步）+ 投影权重流 28.8 %（6.39 s/步）**才是墙。
2. **attention 归属是 host f32 torch，不是 NPU attention_core**：512²/768²/1024² 全部走 webui/chunked.py 的 host-f32-torch 路径；NPU t7_ip 的 L1 在 M=1056 需要 196,096 B > 65,536 B，物理上不可用。该路径下 attention 的 device dispatch 数是 **0**。
3. **attention 的份额随尺寸平方上升**：512² 12 % → 1024² 38 % → 2048² 约 60 %（外推）。流式 attention 只在 1024²/2048² 成为主战场；在 512² 它最多省 2.7 s/步。
4. **512² 一步的 N² 硬账**：host S+P = **267.6 MB/层 = 8.03 GB/步 = 64.2 GB/8 步**；host DRAM 总流量 9.98 GB/步。它没有落在 NPU 上，也没有落在任何 64 KB L1 里。

## 1. 口径与取证

| 项 | 值 |
|---|---|
| 产品路径 | webui/chunked.py（exec_mode=chunked；ZIMAGE_CHUNK_PIPE 默认 1） |
| attention | host-f32-torch（torch f32 QKᵀ+softmax+PV；qblock=512 for M≤8192，否则 256） |
| 器件 | AMD XDNA2 NPU（driver376 / npu_mcdm_stack_prod），webui 7874 单实例 |
| 512² 任务 | 20261030-221051-99ea8a，seed 7，steps 8，VAE on，state=done |
| 768² 任务 | 20261030-221509-c28f1d，seed 7，steps 1，VAE off（缩样点） |
| 1024² 任务 | 20261030-221644-f60a3b，seed 7，steps 1，VAE off（缩样点） |
| 2048² | **未跑**；按 512/768/1024 三点拟合外推（口径见 §5） |
| 计时来源 | [t3d] dev_ms_by_image、[t3d] dev by dispatch tag、[t22g] in-loop ledger、job.json.timing |
| 墙钟纪律 | 只认 loop_s/step wall；ERT state、看门狗、device_s 单列，不混算 |

## 2. 三尺寸 × 分段墙钟表

### 2.1 每步总量（512² 为 8 步均值；768²/1024² 为 1 步）

| 尺寸 | M | 96 行块 | dispatch/步 | step wall s | device s | staging s | host attention s | proj device s | FFN device s | pack s | NPU share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **512²** | 1056 | 11 | **1,650** | **22.22** | 17.98 | 1.34 | **2.69** | 6.39 | **11.58** | 14.9 | 79.4 % |
| **768²**（缩样） | 2336 | 25 | 3,750 | 56.41 | 40.98 | 4.65 | 13.44 | 14.68 | 26.30 | 25.6 | 72.6 % |
| **1024²**（缩样） | 4128 | 43 | 6,450 | 100.78 | 73.41 | 5.41 | 37.93 | 24.73 | 48.68 | 32.6 | 72.8 % |
| **2048²**（外推） | 16416 | 171 | 25,650 | ~980–1,030 | ~890 | ~20 | **~606** | ~99 | ~184 | ~130 | ~70 % |

注：512² 的 8 步逐 step wall 为 26.85 / 22.14 / 22.00 / 22.22 / 22.10 / 21.96 / 21.84 / 21.94 s；首步含冷启动，稳态 21.8–22.2 s。512² 整任务 222.29 s（含 startup 17.58 s、VAE load 19.28 s、VAE decode 5.56 s）。

### 2.2 512² 分段占比（分母 = 22.22 s/步）

| 段 | s/步 | 占 step wall | 性质 |
|---|---:|---:|---|
| FFN device（11 个 96 行块 ×30 层） | **11.58** | **52.1 %** | 权重流：132.7 MB/块 ×330 块 = 44.3 GB/步，~3.8 GB/s |
| 投影 device（q/k/v/o，各 11 块×30 层） | **6.39** | **28.8 %** | 权重流：4×16.59 MB/块 ×330 = 23.1 GB/步，~3.6 GB/s |
| host attention（f32 torch） | **2.69** | **12.1 %** | 5.14e11 FLOP/步 ÷ 2.69 s = **191 GFLOPS**；host DRAM 9.98 GB/步 |
| 其余（pack/refiner/barrier/收尾） | ~1.56 | 7.0 % | pack 14.9 s/步被 device 重叠掉大部分；pre.refiner 2.08 s/步 |

### 2.3 512² 按 dispatch tag 的 device 明细

| tag | 总 s（8 步） | s/步 | 每 dispatch ms | dispatch/步 |
|---|---:|---:|---:|---:|
| att/q | 18.914 | 2.364 | 7.16 | 330 |
| att/k | 7.141 | 0.893 | 2.71 | 330 |
| att/v | 12.002 | 1.500 | 4.55 | 330 |
| att/o | 13.096 | 1.637 | 4.96 | 330 |
| ffn[0..10] | 92.655 | **11.582** | 35.1（均值） | 330 |
| **合计** | 143.808 | **17.976** | — | **1,650** |

att 在这里是 **q/k/v/o 四个投影**，不是 attention；attention 没有 dispatch。

## 3. attention 实现归属（决定卡点性质）

| 尺寸 | 产品路径 | attention 在哪 | device attention dispatch | 证据 |
|---|---|---|---:|---|
| ≤256²（交付六尺寸） | t18_run native | **NPU t7_ip**：qk→sm→pv 三核链，整条 key 轴在 L1 | 30/层（每 head 1 次） | t3d 交付件；T22-PROTOTYPE |
| **512²** | chunked.py | **host f32 torch**（attention_host → _attention） | **0** | [t18] chunk_path installed: host-f32-torch；dev by tag 无 attip |
| **768²** | chunked.py | **host f32 torch** | **0** | 同上（job 20261030-221509） |
| **1024²** | chunked.py | **host f32 torch** | **0** | 同上（job 20261030-221644） |
| **2048²**（未跑） | chunked.py（gate 允许） | **host f32 torch** | **0** | chunk_estimate 同路径；attention=host f32 torch |

**为什么不是 NPU attention_core**：交付 t7_ip 的 L1 不等式 `L1 = Mq*128*2 + 128*SEQ/8*9 + Mq*SEQ*2 + 6144`；512² SEQ=1056，即使用最小 Mq=16，`4096 + 152,064 + 33,792 + 6,144 = 196,096 B > 65,536 B`。所以 512² 以上在现有 t7_ip 上**不是“没优化”，是编不出来**；产品选择了 chunked host fallback，而不是 chunked device kernel。

## 4. “数据在哪落地”清单（512²，每层 / 每步）

| 段 | 数据对象 | 大小/层 | 落地位置 | 每步流量 | 有 N² 项？ |
|---|---|---:|---|---:|---|
| 投影 q/k/v/o | W（4 个权重镜像，每块重写） | 4×16.59 MB ×11 = **730 MB** | host→BO→device DRAM，每块重读 | 21.9 GB（host→dev）+ 21.9 GB（dev 读） | 否 |
| 投影 | X 输入 | 0.74 MB×11 = 8.1 MB | host RAM→BO | 0.24 GB | 否 |
| 投影 | Q/K/V/O 输出 | 4×0.74 MB×11 = 32.4 MB | device→host RAM | 0.97 GB | 否 |
| FFN | BG+B3 | 132.7 MB/块 ×11 = **1.46 GB**（BO 里同一份，每块重读） | device DRAM→core，每块重读 | 43.8 GB | 否 |
| FFN | A / Y | 0.74 MB×11 ×2 = 16.2 MB | host↔BO | 0.49 GB | 否 |
| **attention** | **S = QKᵀ（f32）** | **133.8 MB**（qblock=512：512×1056×30×4 ×3 块） | **host DRAM（torch 临时）** | **4.01 GB/步** | **是，M²×HEADS×4** |
| **attention** | **P = softmax(S)（f32）** | **133.8 MB** | **host DRAM（torch 临时）** | **4.01 GB/步** | **是，M²×HEADS×4** |
| attention | Q/K/V（f32） | 48.7 MB | host DRAM（常驻） | 1.46 GB/步 | 否（O(M·D)） |
| attention | O（f32→bf16） | 16.2 MB | host DRAM | 0.49 GB/步 | 否 |
| **attention 小计** | — | **332.5 MB/层**（其中 N² 267.6 MB） | **全部 host DRAM** | **9.98 GB/步（N² 8.03 GB）** | **是** |
| 全链 BO staging | fill/drain 对象 | ~0.91 GB/层 | host↔device | **27.3 GB/步**（实测 staging 1.34 s/步 → ~20 GB/s） | 否 |

**N² 的精确算术（512²）**：

    S+P = 2 × M² × HEADS × 4 B = 2 × 1056² × 30 × 4 = 267,632,640 B/层
        = 267.6 MB/层 = 8.03 GB/步 = 64.2 GB/8 步

**2048² 的 N² 外推**：`2 × 16416² × 30 × 4 = 64.7 GB/层 = 1.94 TB/步`（按 qblock=256 分块，任一时刻只驻留 504 MB 的 S 块 + 504 MB 的 P 块，但整层要扫 65 个 q 块，流量仍是 64.7 GB/层）。这就是“host f32 路径在 2048² 不可产品化”的硬账：不是算不动，是**内存流量与 606 s/步的平方项**。

## 5. 外推口径（2048²）

三点实测拟合（512/768/1024，同路径、同 seed、同实例）：

| 项 | 拟合 | 2048² 外推 | 说明 |
|---|---|---:|---|
| attention（host f32） | t = 2.249e-6 × M² s | **606 s/步** | 三点残差 −7 %/+9 %/−1 %（M² 拟合） |
| 投影 device | t = 0.581 s × 块数 | **99 s/步** | 512/768/1024 的每块 0.581/0.587/0.575 s，线性 |
| FFN device | t = 1.079 s × 块数 | **184 s/步** | 每块 1.053/1.052/1.132 s，线性（1024 点偏高 5 %） |
| device 合计 | 投影+FFN+attention | **~890 s/步** | 不含 host pack；attention 与 device 部分重叠 |
| step wall | device + 0.6 s/块 残差 | **~980–1,030 s/步** | 512/768/1024 的残差 4.2/15.4/27.4 s |
| 8 步 | ×8 | **~2.2–2.3 h** | 与 T22-BUDGET 的 2.0 h（7174 s）同量级；本表是实测拟合，不是预算表 |

**外推边界**：2048² 没有实跑；chunk_gate 的 6 h/8 步上限允许它跑。上述数字只用于排序卡点，不作为速度声明。若 2048² 的 host 内存带宽先饱和（S/P 64.7 GB/层），attention 项可能高于 606 s/步。

## 6. 卡点性质（一句话版）

- **512²**：权重流卡点（FFN 52 % + 投影 29 %）；attention 是 host f32 的 12 %。
- **1024²**：attention 升到 38 %，FFN 48 %，投影 25 %；三者同量级。
- **2048²**：attention 约 60 %，FFN 约 18 %，投影约 10 %；此时“流式 attention”才是主收益。
- 所有尺寸的共同底层：**device 权重流只有 3.3–3.8 GB/s**，而 host staging 能到 ~20 GB/s；所以 FFN/投影的每一块都在等权重，不是等算力。51.3 TOPS 固定门在这里没有意义——512² 的 device 利用率被权重流锁死，不是 TOPS 不够。

## 7. 复现

    # 512² 全跑（8 步）
    POST http://127.0.0.1:7874/api/jobs {"height":512,"width":512,"steps":8,"seed":7,"vae":true,"mode":"trial"}
    # 768² / 1024² 缩样（各 1 步）
    POST ... {"height":768,"width":768,"steps":1,"seed":7,"vae":false}
    POST ... {"height":1024,"width":1024,"steps":1,"seed":7,"vae":false}

证据文件：imaggers/bigpipe/512_job.json、512_bridge_*.log、768_*、1024_*。

## 4b. 三尺寸 DRAM 流量对照（按同一对象算术；外推处已标注）

| 尺寸 | 投影 W 读/步 | FFN BG+B3 读/步 | attention N²（S+P）/步 | attention 非 N²（Q/K/V/O）/步 | 合计对象流量/步 |
|---|---:|---:|---:|---:|---:|
| **512²**（实测） | 21.9 GB | 43.8 GB | **8.03 GB** | 1.95 GB | ~75.7 GB |
| **1024²**（实测 1 步） | 85.6 GB | 171.0 GB | **122.7 GB** | 7.6 GB | ~386.9 GB |
| **2048²**（外推） | 340.0 GB | 681.0 GB | **1,941 GB（1.94 TB）** | 30.1 GB | ~2,992 GB（2.99 TB） |

口径：投影 W = 4×16.59 MB × 块数 ×30 层；FFN = 132.7 MB × 块数 ×30 层；attention N² = 2×M²×30×4 B×30 层；非 N² = 3×M×3840×4×30 + M×3840×4×30。512²/1024² 的 N² 由实测 pipe.attn 时间与 qblock 算术交叉核对；2048² 为 M² 外推。
