# BIGPIPE-REPORT — t22_bigpipe 收束报告

日期：2026-10-30｜owner：t22_bigpipe｜范围：512²/1024²/2048² 卡点地图 + 流式 attention 首建 + 全链路线图

## 0. 三行判词

1. **512² 的墙是权重流，不是 attention**：FFN 52.1 % + 投影 28.8 %，attention（host f32）只有 12.1 %。
2. **512² 以上 attention 全部走 host f32 torch**：`webui/chunked.py` 的 `host-f32-torch`；NPU `t7_ip` 在 M=1056 需要 196 KB L1 > 64 KB，物理不可用。device attention dispatch = **0**。
3. **流式 attention 单块首建绿、多块 carry 未绿**：单块 seq64 corr 0.999778、6.2 ms、逐字节可复现；seq128/512 多块只折进第一块。**Phase 3 不启动**（按“前两阶段绿”纪律）。

## 1. 交付物

| 文件 | 内容 |
|---|---|
| `BOTTLENECK-MAP.md` | 三尺寸 × 分段墙钟表、attention 归属、数据落地/N² 清单、2048² 外推口径 |
| `STREAM-ATTN.md` | 在线 softmax 设计、512² 片上预算、复用件、单块 spike 结果、多块未绿证据、512² 收益预测 |
| `BIGPIPE-REPORT.md` | 本文件：总判词 + 全链路线图 + 纪律记录 |
| `BIGPIPE-PREDICTIONS.md` | 跑之前写的预测（512² 全跑 / 768² / 1024²） |
| `t22_stream_attn.py`、`stream_attn_kernels.cc`、`build_stream*.py`、`run_stream*.py`、`prep_stream*.py` | 流式 attention 设计、C++ 核、构建/驱动/对照 |
| `512_*`、`768_*`、`1024_*` | 三次产品路径任务的 job.json + bridge log（原始证据） |

## 2. 卡点地图（摘要；全表见 BOTTLENECK-MAP.md）

| 尺寸 | M | 块 | dispatch/步 | step wall | device | host attention | proj device | FFN device | attention 占墙 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **512²** | 1056 | 11 | 1,650 | **22.22 s** | 17.98 s | **2.69 s** | 6.39 s | **11.58 s** | **12.1 %** |
| **768²**（缩样 1 步） | 2336 | 25 | 3,750 | 56.41 s | 40.98 s | 13.44 s | 14.68 s | 26.30 s | 23.8 % |
| **1024²**（缩样 1 步） | 4128 | 43 | 6,450 | 100.78 s | 73.41 s | 37.93 s | 24.73 s | 48.68 s | 37.6 % |
| **2048²**（外推） | 16416 | 171 | 25,650 | ~980–1,030 s | ~890 s | **~606 s** | ~99 s | ~184 s | ~60 % |

- 512² 全跑：job 20261030-221051-99ea8a，seed 7，steps 8，**13,200 dispatch，0 retry**，loop 177.76 s，total 222.29 s，step 稳态 21.8–22.2 s。
- 768²/1024² 各 1 步，VAE off；2048² 未跑，按三点拟合外推（attention ∝ M²，proj/FFN ∝ 块数），口径写在 BOTTLENECK-MAP §5。
- **attention 归属**：512²/768²/1024²/2048² 全部 host f32 torch；dev by dispatch tag 无 attip，device attention dispatch = 0。NPU attention_core 只在 M≤288（SEQ≤304）的交付尺寸可用。
- **512² N² 账**：S+P = 267.6 MB/层 = 8.03 GB/步 = 64.2 GB/8 步；host DRAM 总 9.98 GB/步。S/P 在 host DRAM，不在 NPU、不在 64 KB L1。

## 3. 流式 attention（Phase 2）

### 3.1 设计

在线 softmax：m_j=max(m_{j-1},rowmax(S_j))、c_j=exp2(m_{j-1}-m_j)、P_j=exp2(S_j-m_j)、l_j=l_{j-1}c_j+ΣP_j、O_j=O_{j-1}c_j+P_jV_j、Y=O/l。任何时刻 S/P 只有 BQ×BK（16×64），K/V 只有 BK×HD（64×128），O 是 BQ×HD f32。512² 片上预算 ~50 KB < 64 KB。

### 3.2 首建结果

| 项 | 值 |
|---|---|
| 镜像 | 单 core、Q/KV 两输入、O 输出；xclbin 16,152 B / insts 1,632 B |
| 形状 | heads=1, seq=64, hd=128, bq=16, bk=64（单 key 块） |
| 墙钟 | median **6.2 ms**（6.06–7.50） |
| 对照 | host f32 **corr 0.999778**、rel 3.58 %；host torch 同形状 0.0339 ms（device 慢 183×） |
| 可复现 | 6/6 run 逐字节相同；6/6 ERT_CMD_STATE_COMPLETED |
| N² | 最大 S/P 2 KB；host↔device 只 Q4+KV32+O4=40 KB；S/P 0 B |

### 3.3 未绿（必须写）

多 key 块 carry 未成立：seq128 bk64 corr 0.744、bk32 corr 0.110、seq512 bk64 corr 0.458；final l 停在单块量级、c=0，块 1..n 没有折进。单块 QK/PV 算术已排除（seq64 绿）、V 偏移已排除（zero-Q 对 mean(V) corr 0.999999）、S 未清零已排除（run-to-run 已变 True）。剩余嫌疑：两个自定义核之间 L1 Buffer 跨 scf.for 迭代的写可见性/别名；下一单元改 ObjectFifo state 或直接用 mha.cc 的 init_scale_buffer/partial_softmax 标准件复测。

## 4. 512²/1024²/2048² 口径的收益预测（流式 attention）

host f32 在 512² 实测 2.69 s/步（5.14e11 FLOP/步 ÷ 2.69 s = **191 GFLOPS**）。因此流式器件的盈亏线是 **191 GFLOPS**；低于它，host 更快。

| 器件有效吞吐 | 512² attention s/步 | 512² 省墙钟 | 1024² attention s/步 | 1024² 省墙钟 | 2048² attention s/步 | 2048² 省墙钟 |
|---:|---:|---:|---:|---:|---:|---:|
| 50 GFLOPS（现 ip 量级） | 10.3 | −7.6 s（负） | 157 | −119 s（负） | 2,480 | −1,870 s（负） |
| 191 GFLOPS（盈亏线） | 2.69 | 0 | 41 | 0 | 649 | 0 |
| **400 GFLOPS** | **1.29** | **1.40 s（6.3 %）** | **19.6** | **18.3 s（18 %）** | **310** | **296 s（30 %）** |
| 1 TFLOPS | 0.51 | 2.18 s（9.8 %） | 7.9 | 30 s（30 %） | 124 | 482 s（48 %） |

读法：
- **512²**：即使 400 GFLOPS，流式只值 6 % 总墙钟；它救不了 FFN/投影权重流（80 %）。
- **1024²**：attention 占 38 %；400 GFLOPS 时约省 18 %，与 FFN 同量级。
- **2048²**：attention 占 ~60 %；400 GFLOPS 时约省 30 %，是唯一“必须做流式”的尺寸。
- 若器件只有现 ip 的 50 GFLOPS，**三个尺寸全部负收益**；所以“流式 attention”的收益预测必须带吞吐前提，不能只报 corr。

## 5. 全链路线图（Phase 3，不启动）

**前置**：Phase 2 多块 carry 未绿 → 按“前两阶段绿”纪律，Phase 3 **不启动**；以下仅路线图，不是已建。

| 步 | 单元 | 内容 | 门 | 当前状态 |
|---|---|---|---|---|
| P3-1 | 每层单镜像 | norm→qkv→attn→o→ffn 全片上流水；中间态不落 BO | 单层 latents 对 F-23 参考；dispatch 数 ≤ 2/层 | 未建（前置未绿） |
| P3-2 | 跨层常驻 | 权重/激活双缓冲，层间只换权重；state-8 barrier 消失 | 512² 8 步 latents 逐字节；t3d_baseline 前后 | 未建 |
| P3-3 | 产品接入 | 旗标门后（默认零变化），512² 先开 | 六尺寸零回归 + 512² 出图对拍 | 未建 |

**预期（只在 P3-1/P3-2 成立时）**：512² dispatch 从 1,650/步降到 ~60/步（30 层 ×2），host pack 从 15 s/步降到 <2 s/步，但 FFN/投影权重流仍占 ~18 s/步 → 512² 最好 ~18–20 s/步，相对今天 22.2 s 只有 ~10–15 %。真正的量级变化在 2048²：dispatch 从 25,650/步降到 ~60/步（30 层×2）、host S/P 64.7 GB/层消失，但器件权重流 ~283 s/步 + attention ~310 s/步 → ~10 min/步，仍不是“秒级”。

## 6. 纪律记录

| 项 | 记录 |
|---|---|
| 预测先行 | BIGPIPE-PREDICTIONS.md 写在 512² 提交前（22:08）；512² 预测 35–40 s/步，实测 22.2 s/步（预测偏高，因 10-21 后 chunk pipe 已优化；记录在案） |
| 一臂一进程一 xclbin | 三次产品任务串行（webui 队列一次一个）；流式每个 py310 进程只加载 stream_attn*.xclbin |
| 一进程一剂量 | 512² 一个 job 8 步；768²/1024² 各一个 job 1 步 |
| state=COMPLETED 才算数 | 三次 job 均 done；流式 6/6 ERT_CMD_STATE_COMPLETED (4)，写入 STREAM-ATTN §2.2 |
| 看门狗 ≠ 墙钟 | 只报 loop_s/step wall；ERT state 单列；未把 7 s 看门狗当性能 |
| 51.3 TOPS 固定门 | 本报告不引用 51.3 TOPS 作速度声明；512² 被 3.3–3.8 GB/s 权重流锁死，TOPS 不是约束（BOTTLENECK-MAP §6） |
| t3d_baseline 前后 | 收尾 t3d_baseline.py --check → **ALL REFERENCE BYTES UNCHANGED**；产品文件零改动（改动只在 _scratch/bigpipe） |
| pathspec 提交 | 只 git add _scratch/bigpipe + 三份报告；不 add webui/cache、jobs、outputs |
| 负对照 | ① zero-Q → mean(V) corr 0.999999；② host torch 同形状 0.0339 ms；③ 多块 carry 失败本身是负结果，未粉饰 |
| F-14.1 真实 pid | device.lock：owner=t22_bigpipe pid=35072；Get-Process -Id 35072 为 python 进程（HTTP server 本身），非 wrapper pid |
| 不提权 | 全程 workspace-write；未请求 sandbox escalation；唯一 ACL 失败发生在编译器自建 0700 临时目录，已用 mkdir shim 规避，未改系统 ACL |
| 产品默认零变化 | 本任务未写 webui/、t18_run、交付镜像、六尺寸缓存；512²/768²/1024² 走既有 chunked 路径，默认旗标不变。工作树中 webui/evidence 的两处既存改动不属于本任务 pathspec（未提交） |

## 7. 未了

1. 流式 attention 多块 carry 的 state 可见性（P2-1，见 STREAM-ATTN §2.4）——这是 Phase 2 唯一红项。
2. 512² 的 FFN/投影权重流：44.3 + 23.1 GB/步 @3.6–3.8 GB/s，是 512² 的真卡点；下一战役应优先“大对象 + 多流”而不是 attention。
3. 2048² 实跑（约 2.2–2.3 h/8 步）未做；本报告只给外推，不外推当实测。
