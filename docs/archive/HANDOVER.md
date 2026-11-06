# HANDOVER — Z-Image-Turbo NPU 战役交接文档

> 用途：新会话按此文档无缝接管。生成于 2026-10-31 晚（原会话因"子代理路由允许列表随会话冻结"而交接）。
> 一句话现状：**开放栈 512² 8 步已从 222.3 s 优化到 121.5 s（1.83×），质量 3 跑中位 0.9908/30.27 dB PASS**；翻默认决策与 1024² 计时在途。

## 0. 负责人的目标与已定战略
- **大图优先**：512²/1024²/2048²，小图速度数据无价值。
- **硬标杆**：时间 ≤2× 理论值（6B×tokens×8步/50 TFLOPs），512² ≈ 2-4 s = **MFU ≥50%**（当前 0.675%，差 30-60×）。
- **核心思想**：数据在流水线上从头飞到尾、中间态不落 DRAM（"不落地"）。
- **战略已定**：走**开放技术栈**（上限高、可推广其他模型）；闭源路线已全面调研但弃用（会话加载=权重逐元素转码 12 MB/s、多形状会话倍增、DD 路径卡 op_pdi_map）。
- **术语规范（负责人定）**：Block-FP16 一律写 **BFP16**；**BF16** 专指 bfloat16，永不混写。
- **质量档待拍板**（最后一次呈报未答）：提案 C=形状门翻默认（128²/256² 翻、大图待复测）/ A'=全翻 / B=不翻。**目前一律不翻**（产品默认零变化是铁律）。

## 1. 已落袋战果（全部实测、门全过）
| 项 | 数 |
|---|---|
| 512² 8 步 | 222.29 → **121.50 s（最好）/123.2 中位**；质量 3 跑中位 **0.9908/30.27 dB PASS** |
| host 重排 | a_pack 26.8 ms/层→0、C decode 66.4 ms/层→0（tap 法）；B 重排 20.8 s 挪 startup |
| 投影 | G=15 plain 形态 86→61.6 ms/层；zero-copy −16-20 ms/层（T22_FINAL110） |
| attention 质量 | 流式 30 头 **0.999995**；MMA 表驱动 62 ms/头 0.999945 |
| big-M 契约 | 8 处契约级修复全过（BG/B3 序、A 微瓦片、对象主序×2、残差覆盖、NH=5 缓存、n2 宽度） |
| 布局表 | pvmap.json / qkmap.json（A/B/C 全表 one-hot 读出+负对照，MMA 收编路已铺） |
| 480p/128² 质量 | big-M 128² 0.990743/29.38dB、480p 0.948449（历史线，供参考） |
| 硬件事实 | 51.3 TOPS=104% 实测；shim 16 通道/memtile 8×6/BD≤16 三墙逐字；memtile 524,288 B（整头 K+V 540,672 差 16 KB） |

## 2. 判死清单（别再投人，全部有逐字证据）
- **attention 器件化**：fill 税 6.7 万次/层×6 µs≈404 ms；tap 法双杀（shim 不许 walk/memtile 差 16 KB）；比 host f32 慢 5.1×——**保持 host f32**（OPENSTACK §3.4 的 KV 分片+片上归约是唯一后路，未投）。
- **FFN 侧**（含 plain A/C 镜像、W 双缓冲、A pack/Y decode 器件化）：sw_plain11 state 8 撞 7s 看门狗、dn_plain11 慢 76%——已证死。
- **32 核喂满**：shim 16 通道/memtile 8×6/BD 16 三墙（D1 17 核可行）；相位复用只解决通道不解决容量。
- **指令流放大**：2.9→20.6 KB 平、49 KB state 8 悬崖。
- **率形态 377**（BFP16-A 装载减半）：隔离真、产品段 0.90×（数据流吃掉）。
- **批处理**：旧形态负收益（S0）。
- **闭源路线**：会话加载=O(参数) 逐元素转码（12 MB/s）；多形状倍增；DD 低级路径卡 op_pdi_map/DPU_1；缓存=图+权重一体 10 GB/形状。
- **v28e 流式链**：1056 真数据 corr 0.4957（seq32 的 0.985 不迁移）；scalar bf16 cast（static_cast<float>(bfloat16)）坏；kernels.mm output_dtype=f32 出 NaN；copy/dump 系自定义核绑定易错。

## 3. 在途/待办
1. **1024² 计时**（负责人点名）：1 步+8 步+默认对照、分段桶、MFU——任务书见下"§5 立即执行"。
2. **翻默认决策**（质量档 A/B/C）等负责人；当前全旗标默认关（T22_HOSTKILL/T22_DEVKILL/T22_FINAL110/T22_AF9M/T22_ATTNWIRE/ZIMAGE_BIGM 均 0）。
3. 投影 zero-copy（T22_FINAL110）建议并入（默认关、零风险、−0.5 s/步）。
4. 可选后路：OPENSTACK §3.4 KV 分片+片上 partial 归约（1-2 天、质量风险）。

## 4. 仓库与关键文件（都在 <checkout>，git 管理）
- 主报告链：OPENSTACK-REPORT.md（三级阶梯全账）、HOSTKILL-REPORT.md、DEVKILL-REPORT.md、FINAL110-REPORT.md、BIGPIPE{,2,3}-REPORT.md、BOTTLENECK-MAP.md、GRAPHOPT-47-REPORT.md、TIME1024-REPORT.md（待产）
- 方法论/逆向：_scratch/block/stackhw/CLOSED-RT-DIR{,S2,S3}.md（官方逆向三卷）、S3-A-PATH-SPEC.md
- 历史战役：_scratch/block/{bigm,kgu64,swiglu,d1proto}/（PRODASM-STEP345 123 节、KG64-REPORT、T22-BREADOUT-REPORT 8 契约修复全账）
- 执行序列：_scratch/devkill/RUNBOOK-DEVKILL2.md
- 布局表：_scratch/openstack/{pvmap,qkmap}.json
- 产品接线点（全旗标门后）：webui/chunked.py、_scratch/image/t18_run.py、_scratch/block/bigm/prod_loop_bigm.py、_scratch/hostkill/{bigm_proj_plain,prod_hostkill}.py、_scratch/devkill/devkill_prewarm.py

## 5. 立即执行（新会话第一件事）
跑 1024² 计时（t22_time1024 任务书）：
- 配置：T22_HOSTKILL=1 + T22_DEVKILL=1 + T22_FINAL110=1；big-M proj=1/devhand；attention=host f32。
- 坑：ZIMAGE_DEVICE_TIMEOUT_S=7200、1056 行分块、每臂一进程一 xclbin、loop+total 双报。
- 产出 TIME1024-REPORT.md（1 步+8 步 seed 7+默认对照 1 步+分段桶+MFU）。

## 6. 纪律（全套，必须继承）
1. **子代理干活、父级掌大局**；连续做完再报、真卡点才停；**崩溃单唤醒原单续跑**（连续崩 4 次以上才换新单交接）。
2. 子代理路由：**新渠道（opencode-go 等低价渠道）**；linksub 已死（中转停供）；deepseek-account 可用但贵（勿用）。
3. **不提权**（workspace-write 默认；越界写先停报；历史上仅临时目录恢复与代跑获批发权）。
4. git：**pathspec 提交、禁 -a、禁 amend**。
5. **产品默认零变化**：一切改动旗标门后、默认 0；t3d_baseline --check 前后必查。
6. 门：51.3 TOPS 固定门（xrt-smi validate -r gemm）、state=COMPLETED 才算数、看门狗≠墙钟、负对照必配、预测先行、**质量门 3 跑取中位**（latents 非逐字节可复现、单跑 0.99 是抛硬币）、**loop+total 双报**（挪 startup≠提速）。
7. F-14.1 器件锁真实 pid；一进程一 xclbin；一进程一剂量；每实验独立内核文件（禁共改）。
8. DSH 临时目录（Temp\dsh-*）被清是常事：两级提权重建即恢复（既批流程）。
9. 方法论铁律：'读出来不要推出来'（one-hot 读表法多次定案）、'自洽尺测不出自洽假设'、'合成探针必须与产品同构输入'、'比较量必须同口径'、'absmax 比值不可代相关'、'BFP16≠BF16'。

## 7. 当前差距地图（2-4 s 线的剩余路径）
已证：host 缝合归零（tap）、dispatch 合并、预处理挪启动、8 核拆头（6.17×）。
未解（按量级）：①attention host 2.7 s/步（器件线判死、KV 分片未投）②投影 host 3.2 s/步（zero-copy 已拿 0.5）③FFN host 1.0 s/步④refiner 3.3 s/步（未动过！可能有量）⑤器件核率（单核 3-3.7% 峰值，MMA 57 GFLOPS/核小瓦片）。
**数学结论**：2-4 s 需'官方形态'=编译器级预调度+32 核满配+高率内核——开放栈上是多日/周内核工程；当前 121.5 s 是有界工程的诚实终点。
