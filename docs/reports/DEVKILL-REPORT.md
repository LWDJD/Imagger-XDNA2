# DEVKILL-REPORT — 器件段三件（t22_devkill）

日期：2026-10-31 19:45 (+08:00)｜owner：t22_devkill｜器件：NPU Strix（XRT 2.21.0 / driver 32.0.20102.3930，单实例）
承接：HOSTKILL-REPORT.md（512² 8 步 169.17 s）｜口径：512²/8 步/seed 7/VAE off；ZIMAGE_BIGM=1、ZIMAGE_BIGM_PROJ=1、T22_HOSTKILL=1、T22_DEVKILL=1、T22_BIGM_DEVHANDOFF=1、host f32 attention；ZIMAGE_DEVICE_TIMEOUT_S=7200。

## 0. 判词

**512² 8 步：146.90 s → 121.50 s（最好一臂）/ 124.95 s（同配置重跑），中位 123.2 s。对 110 s 门 1.12–1.14×，仍未过。质量两跑 0.9871/28.71 与 0.9944/32.33（同配置，中位 ~0.991）——是分布不是点，见 §3「质量可复现性」。**

这一轮真正打下来的不是任务书那三件里的任何一件，而是**把 25 s 的一次性权重预处理从 step 0 挪到 startup**（B 重排 20.8 s + G=15 W 打包 3.3 s），外加两个既有旗标。逐件表见 §2——三件里只有投影 G=15 是正收益且未达目标；W 双缓冲是负结果；A pack/Y decode 的器件化镜像已编成但撞 7 s 看门狗（负结果）。

## 1. 本会话做了什么（全部为实测，job 逐个列在 §3）

| 动作 | 结果 |
|---|---|
| ① G=15 投影扩核（前任已编 `plain_g15_rt11`） | 器件 86 → **61.6 ms/层**（4×15.4）；P2 gate 本会话重跑 **逐字节相等** |
| ② FFN RT=11 镜像（前任已编） | gu 70.4→46.5、dn 47.5→31.1 ⇒ **−40.3 ms/层**，与 RT=17 逐字节相等 |
| ② W 双缓冲 | 实现已接线，收益 **0.0**（真 wdn 1.7–2.0 ms，原 47.9 是仪器误读） |
| ② A pack / Y decode 器件化 | **已编成、gate 负结果**：sw_plain11（4.22 MB 指令流；编译 1550 s，当时 5 个重复编译并发，时间被污染，单跑约 1–3 min）→ 首跑 **state 8 / 7018 ms**（7 s 看门狗）；dn_plain11（230 s，同样并发），单独 55.7 ms vs 参考 31.6。**不接线**；host 侧仍是 8.5 / 10.7 |
| ③ **B 重排挪 startup**（本会话新发现+新做） | step 0 **44.5 → 25.6 s**（同态 A/B：8d→8e **−20.7 s**） |
| ③ **G=15 W 预热**（本会话新做） | step 0 再 **−6 s**（8f→8g −7.2，8f→8h −3.8） |
| ③ 三个既有旗标 `FAST_GLUE/GLUE2/GLUE3` | 8e→8f **−3.1 s**（稳态 −0.4 s/步） |

## 2. 器件段逐件归零表（512²，每层，稳态）

| 段 | HOSTKILL | devkill 8a | **devkill 8g/8h** | 做法 / 卡点 |
|---|---:|---:|---:|---|
| 投影 A a_pack | 26.8 | 0 | **0** | 上轮已归零（plain X + X_tap） |
| 投影 C decode | 66.4 | 0 | **0** | 上轮已归零（OUT_tap 逆走位） |
| 投影器件 GEMM | 86 | 65.6 | **61.6**（4×15.4） | G=15/HALF=2；P2 逐字节 PASS |
| 投影 host 余量（read+cv+write） | 101 | 92 | **91** | cv（bf16→f32）4×5.5 ms 仍在 host；**最大单项 host 成本** |
| FFN A a_pack | 20.4 | 8.5 | **8.5** | `a_pack_fast` 块转置；器件化未完成 |
| FFN gu / dn | 70.4 / 47.5 | 45.9 / 31.2 | **46.5 / 31.1** | RT=11 镜像 |
| FFN W 写 | 47.9* | 1.7–2.0（藏） | 1.7–2.0（藏） | *仪器误读，见 §4 |
| FFN Y decode | 10.8 | 10.7 | **10.7** | 器件化未完成 |
| FFN 小计 | 206 | 102.4 | **102.9** | |
| attention（host f32 torch） | 85 | 86 | **90** | 本单边界外 |
| rope | 19 | 19 | **19** | |
| 逐层合计 | 497 | 365 | **358** | ×30 = 10.75 s/步 |
| refiner（pre.*） | — | — | **3.3 s/步** | 每步必算（noise_refiner 2.3 + context 0.9）；**第二大 host 成本** |
| 稳态步 | 17.65 | 15.31 | **14.8–15.2** | |
| step 0（一次性） | 48.6 | 43.2 | **20.0** | B 重排+W 打包已在 startup |

## 3. 512² 8 步 A/B（同一台机、同一 xclbin 集，按时间顺序）

| arm | job | 变的是什么 | loop_s | total_s | step0 | 稳态中位 | corr / PSNR（对 512_base_8b） |
|---|---|---|---:|---:|---:|---:|---|
| 8a | 20261031-174330-75fae8 | 前任基线 | 146.90 | 160.96 | 43.23 | 15.31 | 0.993995 / 32.02 |
| 8d | 20261031-192038-d6247a | 同 8a（同态负对照） | 152.49 | 169.24 | 44.54 | 15.40 | 0.991936 / 30.75 |
| 8e | 20261031-192344-4a4f48 | **+B 重排挪 startup** | **131.82** | 166.62 | 25.61 | 15.60 | 0.992544 / 31.10 |
| 8f | 20261031-192717-8e1c0e | +三旗标 | **128.73** | 163.60 | 26.21 | 15.20 | 0.990965 / 30.24 |
| 8g | 20261031-193227-8a64e4 | +W 预热（all-on） | **121.50** | 159.49 | 19.96 | 14.90 | 0.987118 / 28.71 |
| 8h | 20261031-193547-6a7ba0 | all-on 重跑 | **124.95** | 162.11 | 20.18 | 15.10 | **0.994421 / 32.33** |

**同态 A/B（相邻、机器状态一致）**
- 8d→8e：**−20.7 s**（step0 44.5→25.6）。B 重排实测 **20.79 s**（startup 打印），与账一致。
- 8e→8f：**−3.1 s**（三旗标；稳态 −0.4 s/步）。
- 8f→8g：−7.2 s（W 预热 3.33 s + 首 dispatch/首触，跨状态有噪声）；8f→8h：−3.8 s。
- **8a→8g 总计 −25.4 s（−17.3%）**；total 只 −1.5 s（160.96→159.49）——因为 −25 s 里约 24 s 是从 loop 挪到 startup 的**确定性收益**，不是算力变快。**报 L1 时必须同时报 total。**

**质量可复现性（重要，写进纪律）**
同配置两次跑：8a vs 8d = corr 0.995086；8g vs 8h = corr 0.991103。**全流程 latents 不是逐字节可复现的**（逐层 1-ulp 偏移经 30 层放大；只有单 dispatch 级 gate 才可能字节等）。
⇒ 单跑 corr 在 **0.987–0.995** 区间波动（8g 是尾部样本 0.9871，8h 同配置 0.9944）。**质量门必须 3 跑取中位**，单跑 0.99 是抛硬币。

## 4. 预测对账

| # | 预测 | 实测 | 判 |
|---|---|---|---|
| P1 | G=15 单 dispatch 21.5→14.2±1.5；4 投影 86→57±6 | 15.4（带外 0.3）/ 61.6 | **部分** |
| P2 | G=15 输出与 G=10 plain **逐字节相等** | **本会话重跑：4,055,040 元素 u16 diff 0、maxabs 0.0、corr 1.0** | **对（PASS）** |
| P2 负对照 | null 镜像应红 | **4,055,035/4,055,040 元素不同，maxabs 7.22** | **红（有效）** |
| P3 | W 双缓冲 −47.9 ms/层 | 0.0 | **负结果** |
| P3' | RT=11 −41 + 快 a_pack −8.7 | −40.3 / −11.9 | **对** |
| P4 | plain A/C 镜像能编译 state 4 | 两者都编成，但 **gu state 8 / 7018 ms**，dn 单独 55.7 ms（参考 31.6） | **负结果（7 s 看门狗 + 更慢）** |
| P5 | RT=11 gu+dn 118→~76 | 77.6 | **对** |
| P6 | 三件全上 120–135 s | **121.5–125.0** | **中（区间内）** |
| P7 | 1024² ≤75 s 不过 | 未跑 | — |
| **新** | B 重排挪 startup −20 s、total 不变 | **−20.7 s / total 不变** | **对** |
| **新** | 全流程 latents 可逐字节复现 | **不可**（同配置 corr 0.9911–0.9951） | **错，已更正纪律** |

**两处实测更正（保留）**
1. 「dn32 W 写 47.9 ms」是仪器误读（`_td-_tbw` 里 `_tbw` 已在写之后）；真值 1.7–2.0 ms。
2. FFN 118 ms 里 42 ms 是 RT=17 给 512² 多算的 6 个死 row-tile；RT=11 镜像与 RT=17 **逐字节相等**（4,055,040 元素 diff 0）。

## 5. 差距表（到 110 s）

| 项 | 现状（8g/8h） | 110 s 需要 | 差 | 性质 |
|---|---:|---:|---:|---|
| step 0 | 20.0 s | ≈稳态 | **~5 s** | 剩 W 预热 3.3 s 已挪；余 ~2 s 首触 |
| 稳态步 | 14.9 s | 13.75 s | **1.15 s/步** | |
| ├ layers | 10.75 s | ~9.6 | **1.15** | 器件 4.6 + 投影 host 2.73 + attention 2.58 + FFN host 0.75 + rope 0.54 |
| │ ├ attention | 2.58 s（86 ms/层） | — | — | **host f32 torch，最大单项**；下器件才可能 −2.3 |
| │ ├ 投影 host | 2.73 s（91 ms/层） | — | — | cv/read/write；可压 0.6–1.0 |
| │ └ FFN host | 0.75 s（25 ms/层） | — | — | pack 8.4 + dec 10.1 + read；plain 镜像已证不可行（state 8） |
| ├ refiner（pre.*） | 3.3 s | 3.3 | 0 | 每步真算，不在本单 |
| └ misc | 0.85 s | 0.85 | 0 | |
| **合计** | **121.5–125.0** | 110 | **11.5–15** | |

**结论**：三件做完 + 本会话的两笔（重排搬家、预热）到 **121.5–125.0 s**；要过 110 必须再拿 1.3–1.9 s/步，唯一够量的单项是 **attention 下器件（2.7 s/步）**，其次是投影 host（3.2 s/步）与 FFN plain 镜像（0.6 s/步）。**建议下一单立 attention，而不是继续抠投影/FFN**（FFN plain 镜像这条路本会话已经证死：swiglu 的 60 fill/h 摊成 8 组后仍有 4.22 MB 指令流 + 3,520 个 drain 对象，直接撞 7 s 看门狗）。

## 6. 1024²

**未跑**（L1 未绿）。迁移性：plain15 只编了 RT=11（M=1056）；1024² 是 M=4128，需重编 RT=43/分 4 段；FFN RT=11 同理。即便全绿，1024² 也到不了 75 s（attention O(n²)：4128²/1056² = 15.3×）。

## 7. 纪律记录

| 项 | 记录 |
|---|---|
| 预测先行 | `imaggers/devkill/PREDICTIONS.md`（17:20，所有 job 前） |
| 一臂一进程一 xclbin | 每 arm 一个 py310 bridge 进程（job id 见 §3），单 xclbin 集 |
| state=COMPLETED | 8a/8d/8e/8f/8g/8h 全部 state 4；单 dispatch gate state 4 |
| 51.3 TOPS 固定门 | 收尾 `xrt-smi validate -r gemm` → **TOPS 51.3 [PASSED]** |
| t3d_baseline 前后 | 收尾 `t3d_baseline.py --check` → **ALL REFERENCE BYTES UNCHANGED** |
| 负对照 | ①P2 null 镜像红（4,055,035/4,055,040 不同）②同态 8d 对 8e ③RT=17 vs RT=11 逐字节 |
| F-14.1 | lock owner = launch_webui_bigm.py 真实 pid（每个 arm 的 .out.log 首行） |
| 不提权 | 全程 workspace-write |
| 产品默认零变化 | 改动都在 `T22_DEVKILL=1` 分支内：chunked.py 3 行预热 hook、t18_run.py 一段 B 重排 startup（均默认开、有 =0 关断） |
| 独立内核文件 | 未改 mm_bfp*.cc；新设计在 `imaggers/devkill/` |
| 7 s 看门狗 | 8a–8h 全部无 state 8；FFN plain 镜像 `sw_plain11` **明确 state 8 / 7018 ms**（记负结果，不算墙钟） |
| pathspec | `git commit 7049833`：`imaggers/devkill/`、`imaggers/image/t18_run.py`、`webui/chunked.py`、`DEVKILL-REPORT.md`（19 files，+2747/−2）；xclbin/bin/npy/log 均被 .gitignore 排除 |

## 8. 翻默认建议表

| 形态 | 512² 质量 | 512² loop | 1024² | 建议 |
|---|---|---|---|---|
| 交付 chunked（现默认） | 1.0 | 177.76 s | 111.46 s | **保持默认** |
| hostkill plain 投影 | 0.995740/33.60 | 169.17 s | 未跑 | 不翻 |
| devkill 8a（G=15+RT=11） | 0.993995/32.02 | 146.90 s | 未跑 | 不翻 |
| **devkill all-on（8g/8h）** | **0.9871–0.9944（中位 ~0.991）** | **121.5–125.0 s** | 未跑 | **不翻**：仍 1.12× 于 110 s；且质量是分布不是点（见 §3） |
| 512² L1 门 | — | 110 s | — | **FAIL（1.12–1.14×）** |

## 9. 文件

- 设计：`imaggers/devkill/proj_plain_g15.py`（`out/plain_g15_rt11.xclbin`，P2 已 gate）、`swiglu_plain11.py`/`dn_plain11.py`（**未编完**）
- 接线：`imaggers/devkill/prod_devkill.py`、`devkill_ffn.py`、`qkv_image_g.py`、`devkill_prewarm.py`；`imaggers/image/t18_run.py`（T22_DEVKILL B 重排 startup）、`webui/chunked.py`（旗标分支 + 预热 hook）
- 门：`devkill_run.py`（单 dispatch）+ `io/out_g15_p2.bin` / `io/out_g15_null.bin`（P2 正/负）
- 数据：`imaggers/bigpipe2/devkill_8{a,d,e,f,g,h}.job.json`、各 `webui/jobs/<id>/job.log`
- 运行脚本：`imaggers/devkill/run_arm.ps1`（env 封装）、`RUNBOOK-DEVKILL2.md`（含本轮已执行标记）
