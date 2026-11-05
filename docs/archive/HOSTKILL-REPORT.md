# HOSTKILL-REPORT — 砍非 attention 地板（t22_hostkill）

日期：2026-10-31 17:05 (+08:00)｜owner：t22_hostkill｜器件：NPU Strix（XRT 2.21.0 / driver 32.0.20102.3930，单实例）
承接：OPENSTACK-REPORT.md（512² 8 步 342.7 s 分解：attention 112.5 / 非 attn 191.4）、BIGPIPE2/3。
口径：512² / 8 步 / seed 7 / VAE off；ZIMAGE_BIGM=1、ZIMAGE_BIGM_PROJ=1、T22_BIGM_DEVHANDOFF=1、host f32 attention；ZIMAGE_DEVICE_TIMEOUT_S=7200。

## 0. 判词

**Host 重排已被砍掉，L1 110 s 仍未过。**

| 项 | 基线（512_base_8b） | 本轮 hostkill | 判 |
|---|---:|---:|---|
| 512² 8 步 loop | **212.93 s** | **169.17 s** | **−20.5%**（−43.8 s） |
| 稳态 step | 25.1 s | **17.6–18.0 s** | −28% |
| 投影/层 | 0.39 s（8 dispatch，每次 a_pack+反 tile） | **0.187 s（4 dispatch，host 重排 0）** | **−52%** |
| 非 attention 地板 | 191.4 s | **≈147.7 s** | 仍 >110 |
| MFU（1.92 s=50%） | 0.45% | **0.57%** | 对 50% 线 88× |
| 质量（对同配置基线） | 1.0 | **corr 0.995740 / PSNR 33.60 dB** | **PASS（≥0.99/28 dB）** |
| 对产品默认（继承 0.985479/26.14） | — | 0.985/26 族 | **FAIL（继承）** |

一句话：**a_pack / 反 tile / transpose 已全部下器件（shim DMA 的 micro-tile 走位），投影 dispatch 8→4，投影 host 段从 0.39 s/层降到 0.187 s/层；512² 8 步 169.17 s，质量绿。但 110 s 仍未过：剩下的 169 s 里器件 GEMM（投影 86 ms + FFN 118 ms/层）和 attention/misc 已经是地板，host 残余再归零也只能到 ~123 s。L1 的真卡点已从 host 转到器件吞吐（投影 21 ms/31 GFLOP ≈ 1.5 TFLOPS ≈ 3% 峰值）。**

## 1. 预测对账（HOSTKILL-PREDICTIONS.md）

| # | 预测 | 实测 | 判 |
|---|---|---|---|
| P3 | 投影 0.39→0.19 s/层、dispatch 8→4 | **0.187 s、4 dispatch** | **对** |
| P4 | 8 步 165–175 s，稳态 17–18 s/步 | **169.17 s，17.6–18.0 s/步** | **对** |
| P5 | 110 不过；器件地板 120–130 s | 实测 host 全归零地板 ≈123 s | **对** |
| P6 | 单 W 流融合 QKV 更慢（7 s watchdog） | **实测 state 8 / 7013 ms**（负结果，未接线） | **对** |
| P7 | plain 与 rt17+pack+decode 逐字节相等 | **equal True，corr 1.0，maxabs 0.0** | **对** |

## 2. Host 分段归零表（512²，每层 / 8 步）

口径说明：交付 chunked 投影是 44 dispatch/层（11 tile×4 投影）；本轮基线是 big-M 整形状（8 dispatch/层，其中 4 个是 o 调用里多算的 q/k/v）。本表按实测 big-M 基线列账。

| 段 | 基线 ms/层 | 本轮 ms/层 | Δ 8 步 | 做法 |
|---|---:|---:|---:|---|
| 投影 A a_pack（1056×3840，fancy index） | 26.8（2×13.4） | **0** | **−6.4 s** | X 以 plain row-major 写 BO；bigm_proj_plain 的 X_tap 用 [rt][kg][m/8][k/8][8][8] 走位，shim DMA 直接产出 mm_bfp 的 A micro-tile |
| 投影 C _proj_decode（11 tile 反 tile+转置） | 66.4（8×8.3） | **0** | **−15.9 s** | OUT_tap 用 [rt][q][m/8][n/8][8][8] 逆走位，直接写 (1056,3840) 行主序；q=h*G+i 落在列块 128q |
| 投影重复 dispatch（o 调用里又算 q/k/v） | 3×~33 = 99 | **0** | **−23.8 s** | o 调用只算 o；q/k/v 一次 |
| 残留 bf16→f32 转换（不是重排） | 0 | 30（4×7.5） | +7.2 s | 行主序输出后的 type lift；下一步可并进 rope/attention 入口 |
| 投影 dispatch 数 | 8 | **4** | — | 单投影 1 dispatch |
| FFN a_pack + Y decode（未改） | 31.2 | 31.2 | 0 | 见 §5 差距表；devhand 的 A==C 契约卡住 |
| 合计被砍 host | **192.2** | 30（type lift） | **−43.8 s** | 与 loop 实测 −43.8 s 同笔 |

**验证**：_scratch/hostkill/cmp_plain.py 对同一 X、同一 L0 W：
- plain 行主序输出 vs rt17 raw + _proj_decode：**np.array_equal True，maxabs 0.0，corr 1.0**（逐字节）。
- 负对照：rt17 raw 不反 tile 直接当行主序 vs plain：**corr −0.000255**（应红，证明 C tap 真生效）。
- 融合 QKV 负结果：qkv3_rt11（N=11520 单 W 流）数值 3/3 段逐字节对，但 **state 8 / 7013 ms**（49.8 MB 单流超 L2 驻留，10 个 row-tile 重读 W 掉进 7 s watchdog）；故不接线，保留 4 dispatch。
- 三 W 流 15-worker 融合也不采用：delivered qkv 镜像是 9.69 ms/96-row block；big-M 11 tile 外推 107 ms/层，慢于 4×21.5=86 ms 的 big-M 整形状（且 15 worker 里 5/投影，单 worker 6 heads 串行，W 重用不如整形状）。X 一次写、q/k/v 复用已在本轮 plain 路径里成立。

## 3. 512² 8 步 A/B

| 量 | 基线 512_base_8b | 本轮 hostkill_8a |
|---|---:|---:|
| job | 20261031-144813-c1343c | 20261031-164727-f4aaed |
| loop_s | **212.93** | **169.17** |
| total_s | 225.16 | 183.11 |
| step 0（含首调/JIT） | 39.06 | 48.61 |
| step 1–7 | 25.19/25.04/25.06/25.08/25.23/25.41/25.54 | 17.89/17.59/17.65/17.71/17.61/17.64/17.62 |
| 稳态中位 | 25.1 | **17.65** |
| 对 110 s 门 | 1.94× | **1.54×** |
| MFU | 0.45% | **0.57%** |
| latents corr / PSNR（对基线） | 1.0 | **0.995740 / 33.60 dB** |
| 产品默认档 | — | 0.985/26 族（继承，FAIL） |

稳态逐层（hostkill_dbg2，layer 22–29 中位）：
- proj_qkv **0.124 s**（4×21.5 ms run + 4×2.5 ms read + 4×7 ms cv）
- proj_o **0.063 s**（1×21.5 + 2.5 + 8.8）
- FFN devhand **0.206 s**（pack 20.4 + wgu 3.4 + gu 70.4 + wdn 47.9 + dn 47.5 + read 2.7 + dec 10.8）
- attention **0.085 s**，rope 0.019 s，其余（norm/残差/barrier/rec）≈0.10 s
- 合计 **0.60 s/层** → 18.0 s/步

## 4. 为什么 110 s 仍不过（差距表）

| 项 | 现状 | 110 s 需要 | 差 | 性质 |
|---|---:|---:|---:|---|
| 投影器件段 | 4×21.5 = **86 ms/层** | ~55 ms | 31 ms | 10 worker / 1.5 TFLOPS；需 G=15（HALF=2）或 32 核 |
| FFN 器件段 | gu 70.4 + dn 47.5 = **118 ms/层** | ~90 ms | 28 ms | swiglu N=20480 G=10 固定；dn 可 G=15 |
| attention host | 85 ms/层 | 85 ms | 0 | 本轮不动 attention（任务边界） |
| host 残余 | 投影 101 + FFN 85 + misc ~103 = **289 ms/层** | ~170 ms | 119 ms | FFN pack/dec 31、W 写 48、cv 30、Python/BO ~56、misc 103 |
| **host 全归零地板** | **≈123 s** | 110 s | 13 s | 即使 host 残余=0 也差 13 s；必须同时提器件 |

结论：**110 s 不是 host 重排单变量能到的**。本轮已把任务指定的 host 重排（a_pack/反 tile/transpose）全部归零，并把投影 dispatch 8→4；下一单要过 110 必须动器件（G=15/32 核、FFN 双缓冲重叠 W 写），或把 attention 也换成器件路径（但那是另一条战线）。

## 5. FFN host 段差距（未归零，诚实列账）

| 段 | ms/层 | 目标 | 卡点 |
|---|---:|---:|---|
| A a_pack（1056×3840） | 20.4 | 0 | 同一 DMA 走位在 swiglu 镜像上每 h 要 60 个 X fill，shim tile 16 BD 上限拒绝（aiecc Too many simultaneously active BDs on tile (3,0)）；需 memtile 分级或专用 pack 镜像 |
| Y 反 tile/decode | 10.8 | 0 | devhand 依赖 gu C-layout == dn32 A-layout（k=32）；把 gu 改行主序就要同时把 dn 的 A 改成 plain，需重编 dn32 + 单 BO 契约重验 |
| dn32 W 写（44.2 MB） | 47.9 | 隐藏 | 与 gu dispatch 串行；需双 W BO + 后台线程/下一次 dispatch 重叠 |

## 6. 1024²

**未跑。** 按纪律“前一级绿才启动下一级”：512² L1 169.17 s vs 110 s（1.54×）未绿，1024² 不启动。
可引用旧账：1024² 1 步 best **128.20 s**（chunked 投影 + devhand，BIGPIPE3），MFU **0.75%**（1.92 s=50% 口径），对 75 s 旧门 **1.71×**；本轮的 plain 投影只编了 RT=11（512² 形状），1024² 需 RT=17 + 3 段，未编未测。

## 7. 纪律记录

| 项 | 记录 |
|---|---|
| 预测先行 | HOSTKILL-PREDICTIONS.md 写于所有 job 前（16:20） |
| 一臂一进程一 xclbin | 每个 gate/arm 一个 py310 进程；run_one.py 单 xclbin；product job 一个 bridge 进程 |
| state=COMPLETED | 所有接线 dispatch state 4；融合 QKV 负结果 state 8 明确记负，不冒充 |
| 51.3 TOPS 固定门 | 收尾 xrt-smi validate -r gemm → **TOPS 51.3 [PASSED]** |
| t3d_baseline 前后 | 收尾 t3d_baseline.py --check → **ALL REFERENCE BYTES UNCHANGED** |
| 负对照 | ① 不反 tile corr −0.000255；② 单 W 流融合 QKV state 8/7013 ms；③ 基线 512_base_8b 同配置 |
| pathspec | 只改 _scratch/image/t18_run.py 两处 T22_HOSTKILL 旗标门、webui/chunked.py 一个旗标分支；新增 _scratch/hostkill/；未 add cache/jobs/outputs |
| F-14.1 | device.lock owner = 实际 dispatch 的 py310/bridge 进程（bench_host/run_one/product job） |
| 不提权 | 全程 workspace-write，无 escalation 请求 |
| 产品默认零变化 | T22_HOSTKILL 默认 0；t18_run 的 plain 镜像只在 flag=1 注册；chunked.install 只在 flag=1 走 prod_hostkill。flag-off 1 步 sanity job 20261031-170721-44de2c：loop **52.11 s** vs 改前同口径 **52.08 s**（同 JIT 首步，零变化） |
| 独立内核文件 | 未改 mm_bfp*.cc/kernels；新设计 _scratch/hostkill/bigm_proj_plain.py 独立文件，C++ 内核零改动 |
| 7 s 看门狗 | 融合 QKV 的 7013 ms 是 state 8 看门狗，明确不算墙钟 |
| timeout 7200 | 所有 job ZIMAGE_DEVICE_TIMEOUT_S=7200 |

## 8. 翻默认建议表

| 形态 | 512² 质量 | 512² loop | 1024² | 建议 |
|---|---|---|---|---|
| 交付 chunked（现默认） | 1.0 | 177.76 s（旧基线） | 111.46 s | **保持默认** |
| hostkill plain 投影（本轮） | 0.995740/33.60 dB | **169.17 s** | 未跑 | **不翻**：质量绿、墙钟仍 1.54× 于 110 s；只作下一单的底座 |
| 单 W 流融合 QKV | 数值逐字节对 | state 8 / 7013 ms | — | **拒绝**：L2 驻留失败 |
| 512² L1 门 | — | 110 s | — | **FAIL（1.54×）** |

## 9. 文件

- 设计：_scratch/hostkill/bigm_proj_plain.py（X plain / C row-major 的独立设计，未改内核源）
- 接线：_scratch/hostkill/prod_hostkill.py + _scratch/image/t18_run.py（T22_HOSTKILL 旗标）+ webui/chunked.py（旗标分支）
- 门：_scratch/hostkill/gate_qkv3.py、cmp_plain.py、negctl.py、run_one.py
- 数据：_scratch/hostkill/io/、_scratch/bigpipe2/hostkill_8a.job.json、hostkill_dbg2.job.json
- 编译产物：_scratch/block/bigm/out/plain_rt11.xclbin/.bin（qkv3 为负结果，不接线）
