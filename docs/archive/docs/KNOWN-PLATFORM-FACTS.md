# KNOWN-PLATFORM-FACTS — XDNA2 平台事实卡(权威单一来源)

> **状态**:v1(2026-10-20,t22_factcards)· 行号锚定 commit `167aff1`
> **适用**:本机 driver376 / `npu_mcdm_stack_prod` + pyxrt + open IRON / aiecc 的 AMD XDNA2 NPU(Strix,AIE2P)
> **本文不是报告**:报告是一次实验的完整叙事;本文是**可被引用、可被追责的事实条目**。
> 每条卡必须有可查出处(报告名:行号 / commit / 实测数据);无出处者一律进 §4「传闻待证」,不得当设计依据。

**路径约定**:本文相对路径以仓库根 `zimage-bfp16-npu/` 为基准;工作区根 = `<workspace>`,以 `<工作区根>/` 开头的除外。
**行号会漂移**:行号锚定 commit `167aff1`;报告被追加后请按「报告名 + 关键词」重定位,不要盲信行号。

---

## 0. 四条硬规则

1. **没有出处的数字不许进设计**。要进,先标 `传闻待证` 并附一条一小时内能做完的判据(§4)。
2. **本文不裁决矛盾**。两处事实冲突 → 进 §3 矛盾表并列,交负责人裁决(**止损线**)。
3. **不改史**。订正已发布的事实卡 = 追加「订正 YYYY-MM-DD:…」行 + 保留原行,不删旧文。
4. **共享仓库禁改历史**:禁用 `git commit --amend` / `rebase` / `filter-branch`(多代理并发会串提交,见 F-20)。

---

## 1. 事实卡索引

| ID | 事实(一行) | 主要出处 | 状态 |
|---|---|---|---|
| F-01 | **state-8 定律**:同 xclbin 背靠背第 2 次 launch 无完成信号;插异 xclbin 即愈;第二 handle 是假完成 | `webui/evidence/T20-STATE8.md:3-6,42-47` | 定案 |
| F-02 | **BO 可见性(输出侧)**:wait 后必须 `sync(FROM_DEVICE)`;v1 的"输入侧窗口/双缓冲"归因已撤回 | `_scratch/attn_impl/T22-PLATFORM-FACTS.md:6-35` | 定案 v2(**已裁决** 2026-10-20) |
| F-03 | **pyxrt `bo.write` 全程持 GIL**:线程预填在本栈是伪命题 | `_scratch/perfc/T22-UTIL-REPORT.md:154-164` | 定案(负结果) |
| F-04 | **ERT 看门狗 ~7.0 s/dispatch**:超时判 state 8,调大 wait 无用 | `webui/evidence/T20-STATE8.md:145-147` | 定案 |
| F-05 | **memtile 每 tile 仅 6 S2MM**(6 MM2S / 48 BD / 512 KB) | `_scratch/block/T5.md:117-120`, `_scratch/perfc/T22-UTIL2-MEM-REPORT.md:97` | 定案(双证) |
| F-06 | **shim 同时活跃 BD ≤ 16**(BD 是"在飞"计数,不是总量) | `_scratch/vae/T21-3B2VF-REPORT.md:94-97`, `HANDOFF.md:184-185` | 定案 |
| F-07 | **L1 65,536 B/核;程序区 16,384 B 且是硬件** | `HANDOFF.md:158-163`, `_scratch/block/STEP2.md:96-106` | 定案 |
| F-08 | **声明 data 参数 ≤ 7**(8/9/10 全部 ABORT;7 完成) | `HANDOFF.md:168-170` | 定案 |
| F-09 | **teardown 崩溃是常态,退出码不可信**:判据是 `calls.json` + 落盘产物 | `_scratch/block/STEP5.md:592-601` | 定案 |
| F-10 | **aiecc 单线程**(CPU/wall 1.00),多核按核并行是唯一提速路 | `_scratch/vae/T21-3B2-PLAN.md:618-619` | 定案 |
| F-11 | **BFP16 vs BF16 术语规范**:权重/KB/VB = BFP16(1.125 B/参数);QP/P/S/O/SBG/X/A = BF16(2 B/元素) | `_scratch/attn_impl/T22-PROTOTYPE.md:118-122`, commit `b5a08a4` | 定案 |
| F-12 | **(K=2336 ∧ N=256) stall → 必须 N=128 半宽** | `_scratch/vae/T21-3B2VF-REPORT.md:101-109` | 经验定案(机制未定) |
| F-13 | **模型与形状算术**:6.14 B / 34 blocks / hidden 3840 / 步数锁 8 / M=32+⌈…⌉×32 | `docs/HANDOFF-2026-10-19.md:11`, `README-WEBUI.md:39,159` | 定案 |
| F-14 | **器件为共享资源**:第二个使用者 → `0xc01e0009`;争用致时长漂移;**F-14.1 器件互斥锁协议** | `_scratch/vae/T22-VAE-REPORT.md:218-225` | 定案 + 协议(2026-10-20 起强制) |
| F-15 | **宿主侧并发 launch 不并行**(x1.007) | `_scratch/perfc/T22-UTIL-REPORT.md:166-173` | 定案(实测) |
| F-16 | **环境块**:py312 编译/服务、py310 器件、PEANO、driver376、每次 fresh 缓存 | `docs/HANDOFF-2026-10-19.md:99-112` | 定案(路径已复核) |
| F-17 | **m/Kc 轴闭合**:可用 L1 34,816 B ⇒ m·Kc ≤ 17,408;交付 (m=16, Kc=960) 即 norm core 架构最优 | `_scratch/perfc/T22-UTIL3-U1P-DEATH-TABLE.md:20-36` | 定案(可复算,**kgu 无关**,§10.1) |
| F-18 | **aiecc 分配器 bank 打包受限**:`C4_4` 放不下,10→8 不解;修法 = bank 感知放置/改广播拓扑 | `_scratch/perfc/T22-UTIL3-U1P-DEATH-TABLE.md:57-66`, commit `167aff1` | 实测(挂起;**W6/C4_4 = kgu=64 专属**) |
| F-19 | **方法卡:设计前可行性 spike**——三条构造的逐字死因(delegate / 非对称 fifo / join) | `_scratch/perfc/T22-UTIL3-U1P-DEATH-TABLE.md:12-18` | 方法条目 |
| F-20 | **纪律卡:共享仓库禁用 `git commit --amend`**——amend 改的是 HEAD,不是"你的上一个提交" | §3 事故链 + `git reflog`(`cafc3b9`/`9da5e6e`/`2dddef3`) | 纪律条目(事故实证) |
| F-21 | **AIE2P 标量 f32 加法与 x86 SSE 的 tie-rounding 不一致**:1e6 里 1-2 个差 1 f32 ulp;逐位门只能按幅值界 | `_scratch/vae/T22-VAE-REPORT.md:284-310` | 定案(含门禁含义) |
| F-22 | **语义陷阱**:`finish()` 取代逐次 `wait=True` → 换序列、假快、门 0 FAIL(两次差 290,791 B) | `_scratch/attn_impl/t2/T2-PREDICTIONS.md:76-82` | 定案(测量臂作废) |
| F-23 | **换参考裁决**:512²+ 的数值门对 BFP16 精确仿真参考,f32 corr 降为信息项;128–256² 不变 | `8e5fdc7` / `fbe6b24` + 负责人裁决 | 定案(裁决,含三条纪律承诺) |
| F-24 | **conv stall 触发面**:`F = rr²N²/K`;≤3.07 M 通过 / ≥5.69 M stall(21/21 分离,与 M 无关);四条单标量律全否证 | `_scratch/vae/stallprobe/README.md` §0.3/§4(commit `a078ba9`) | 强证据 + 根因未知 |
| F-25 | **闭源基线 8.x s**:硬件利用率上限远高于本栈实测;"不可达"一律限定为"本架构下";器件扩散 15.655 s → 锚点 8.x s(≈2×,**128² 同口径**) | 负责人 2026-10-20 口述(分辨率已确认)+ `_scratch/block/T13.md:270-283` | 目标/锚点(128² 已确认;大分辨率口径未知) |
| F-26 | **compute tile 2 进/2 出 DMA** ⇒ 独立 fifo 形态 NK=1 封顶;与 F-05(6)/F-06(16)构成通道预算族 | `_scratch/attn_impl/t2/T2-ARCH.md:154-188`(commit `f57187f`) | 定案(限独立 fifo 形态) |
| F-27 | **objectfifo 元素尺寸 = 一次 fill 字节数**;写大 ⇒ 静默 state 8(块 0 正常 / 块 1 MEMORY_STALL) | `_scratch/attn_impl/sring/SRING-REPORT.md` §0-§1/§10(commit `9783b8d`) | 定案(三重) |
| F-28 | **core L1 共驻大对象 ≤ 2 块/元素**;3 块是**放置失败**,不是总字节 | `_scratch/attn_impl/sring/SRING-REPORT.md` §10(commit `bea502d`) | 实测 |
| F-29 | **runtime sequence 线性、不是墙**:nk=86 一次 dispatch,28,552 B / 174 push / 174 wait / ERT [4] | `_scratch/attn_impl/sring/SRING-REPORT.md` §10/§10.1(commit `bea502d`) | 实测(含吞吐边界) |
| F-30 | **流式常驻集与 nk 无关**;"246 上限"只限共驻 ⇒ B 的 dispatch 收益不需 memtile/A1 | `_scratch/attn_impl/sring/SRING-REPORT.md` §10.2 | 定案(限流式形态) |
| F-31 | **器件 exp 不是正确舍入**:`aie::exp2<bfloat16>` 在 softmax 区间仅 1.51 % 逐位、中位 10 ulp(~4 %);"接近 f32"必带此限定 | `_scratch/attn_impl/t2/T2-PILLAR3.md` §18(commit `ce0a0b9`) | 定案(机制待证) |
| F-32 | **BD 每维 size ≤ 1023**;2-D `[4,9216]` 被拒、3-D `4×576×16` 可编;**Kc=960 是它的产物** | `_scratch/attn_impl/proj/PROJ-REPORT.md` §11(commit `09a1d2f`) | 定案(编译期对照) |

---

## 2. 逐条事实卡

### F-01 · state-8 定律(同 xclbin 背靠背第 2 次无完成信号)

**事实**:同一 xclbin 连续 launch 时,**第二次不产生完成信号**;ERT 自己的看门狗在 ~7.0 s 把它判成 `ERT_CMD_STATE_TIMEOUT`(state 8),命令自身状态**恒为 8**。中间插一次**别的 xclbin** 的 launch(哪怕 1.6 ms 的 ip)就完全消失;主机侧等待(0 / 10 / 30 / 100 / 500 / 2000 ms)**全部无效**。同一 xclbin 开**第二个 kernel handle** 会得到"4 ms 完成"的**假完成**(132.7 MB 权重流不可能 4 ms)。

**出处**
- `webui/evidence/T20-STATE8.md:3-6` — 定律陈述。
- `webui/evidence/T20-STATE8.md:42-47` — 决定性实验 A–F 六臂(standalone,零 build):A `4:40 / 8:7017 / 4:1071 / 8:7017`;C 插 ip 后 **0/4 挂,每次 39 ms**;E 第二句柄 "COMPLETED 2–4 ms"(假完成);F 超时后 `run.state()` 轮询 6×200 ms **一直是 8**。
- `webui/evidence/T20-STATE8.md:56-61` — 根因表述(可直接引用的一段)。
- `webui/evidence/T20-STATE8.md:143-147` — R1/R2 逐字:第二 handle 是陷阱;`run.state()` 超时后恒 8 ⇒ 是 ERT 的看门狗,不是我们的 wait。
- `_scratch/vae/T21-3B1-U1-PAPER.md:38` — 独立战役复现(gn512 背靠背 ×3:第 1 次 OK、第 2 次 state 8、第 3 次又 OK,输出字节与第 1 次相同)。
- `docs/HANDOFF-2026-10-19.md:51` — 一页速读版(产品侧复述)。
- commit `6e03bae`(t22_large)与 `_scratch/attn_impl/T22-PLATFORM-FACTS.md:37-41`(与可见性的关系)。

**对设计的影响**
1. 每进程**一个** xclbin;单 dispatch 测量一律 `--runs 1` + 独立进程(多跑 = 多进程)。
2. 产品路径里"同镜像连续 launch"的形态(如 3 个 FFN 块)必须在块间插**异镜像 barrier**(`t18_run.Dev.barrier`,实测 2.6 ms/次,占 8 步 loop 1.4%)。
3. **禁止**用"第二个 kernel handle"绕行 —— 它会给出"完成但没做事"的结果。
4. 一次 state-8 超时会**污染整条测量**(`_scratch/block/T10.md:303`),该轮数据作废重跑。

**状态**:定案。两个独立战役(T20 图像、T21/3B1 VAE)同签名复现。

---

### F-02 · BO 可见性:wait 之后必须 `sync(FROM_DEVICE)`(输出侧;含一次同源更正)

**事实(v2,更正版)**:`run.wait()` 返回 `ERT_CMD_STATE_COMPLETED` 只表示**内核/运行时序列完成**,不表示 shim→host 的输出 drain 已落到 host 可见内存。对**短 dispatch**(本机 qk/pv 单次 0.6–0.9 ms),host 若紧接着读输出 BO 而**不做 `bo.sync(FROM_DEVICE)`**,会读到旧/半新数据——表现为同输入两次结果不同,且差异随并发/时序漂移。

**更正记录(同源、同日,2026-10-20)**
- **v1(已撤回)**:曾把主因归到"host 填 BO → 内核读"的**输入侧窗口**,证据是"3 ms 延迟把 rel 从 0.220 压到 0.0087(25×)",并据此给出"双缓冲 + fill(i+1)→wait(i)→launch(i+1)"的发射纪律(commit `6e03bae`)。
- **v2(现行)**:后续单变量实验推翻 v1——**延迟只降低"读到旧数据"的概率**,不是修复:

| 实验(512²,同 u / 同 xclbin / 同代码) | 读前 sync(FROM_DEVICE) | layer 0 a 数组 | rel |
|---|---|---|---|
| det1 vs det2(无延迟) | 否 | 不逐字节相同 | 0.220 |
| det3 vs det4(fill 后 3 ms) | 否 | 不逐字节相同 | 0.0087 |
| det9 vs det10(10 ms) | 否 | 不逐字节相同 | 0.0093 |
| det5 vs det6(重叠 fill) | 否 | 不逐字节相同 | 0.463 |
| **detC vs detD** | **是** | **逐字节相同(sha 相同,max\|d\| = 0)** | **0** |

⇒ 真正的缺陷是**输出侧少了一次 `sync(FROM_DEVICE)`**(`t18_run.Dev` 本来就有这一步;t22 的 qk/pv 路径漏了)。

**出处**
- `_scratch/attn_impl/T22-PLATFORM-FACTS.md:6-15`(v2 事实 + 更正)、`:19-27`(证据表)、`:29-35`(纪律 + 门 0)、`:37-41`(与 state-8 的关系)。
- `_scratch/attn_impl/T22-PROTOTYPE.md:106-116` — v1 的判读与修复顺序(**历史陈述,已被更正**,不要照抄)。
- commit `6e03bae` — v1 归因定案的那次提交(保留为历史)。
- 该文件是**未提交草稿(untracked)、仍在被改**:行号锚定 2026-10-20 10:25 版,漂移时按关键词重定位。

**纪律(v2,新 dispatch 形态默认遵守)**
1. **wait 之后、读之前**:对每个输出 BO 做 `sync(XCL_BO_SYNC_BO_FROM_DEVICE)`。
2. **填侧保持最简**:`bo.write` → `sync(TO_DEVICE)` → launch;**不需要 sleep、不需要重叠、不需要双缓冲**。
3. 不要用线程预填(F-03:GIL);不要把吞吐寄望于多 xclbin 并发(F-15:1.007×)。
4. **门 0**:同输入、同 xclbin,两次 dispatch 的**完整输出数组逐字节相同**(不是 absmax、不是 corr)。所有新形态先过此门。
5. **禁止用 `finish()` / `wait=False` 取代逐次 `wait=True`**:那会静默改变执行语义(两次跑差 290,791 B、门 0 FAIL),见 **F-22**。

**depth-1 fifo 的死锁陷阱(2026-10-20,t22_t2)**:"提前发下一 head 的 fill"(PIPE=1)叠在 **depth-1** fifo 上会**死锁**(>4.5 min 无返回,连看门狗都没返回)。⇒ **重叠只能靠增加并发组 G,不能靠重排单个 head 的 fill**;这也说明 v1 的 `fill(i+1)→wait(i)` 纪律**不是 depth-1 上的普适解**(v2 已把根因改成读前 sync)。出处:`_scratch/attn_impl/t2/T2-PREDICTIONS.md:79,83`。

**对设计的影响**:512² 的 qk/pv 单 dispatch 只有 0.6–0.9 ms,是本机最短 dispatch,**最先命中**;修好前任何 corr 数字只作参考(层间误差被放大,30 层后 pred 与 host 几乎不相关是预期后果)。v1 的"双缓冲 + fill→wait→launch"**不再是必需项**(它只改概率、不改正确性);已按 v1 写下的代码可保留,但**不得**作为正确性依赖。

**状态**:定案 v2(读前 sync);v1 归因**已撤回**,但 0.220→0.0087 的观测本身仍成立(概率现象)。**C-3 已裁决(2026-10-20):采用本卡 v2,依据源头 commit `f1f656c`。**

---

### F-03 · pyxrt `bo.write` 全程持 GIL(线程预填是伪命题)

**事实**:`bo.write` 全程持 GIL;工作线程一旦拿到 GIL 就饿死主线程的 Python 胶水。定案实验:纯 numpy 胶水(200×rms_into)**0.043 s** → 同一份胶水 + 线程写 88.5 MB BO **12.360 s**(287×)。产品侧 A/B:线程预填让 fill 桶 2.28→1.06 s(staging 确实掉了 1.22 s),但 wall **30.23→30.16(−0.08 s)**,间隙桶 +1.4 s —— **时间只是搬家**。

**出处**
- `_scratch/perfc/T22-UTIL-REPORT.md:154-164` — §6.1 逐字(含两行实测)。
- `_scratch/perfc/T22-UTIL-REPORT.md:31-32,103` — 假设检验表:U2 预测 −1.0~−1.2 s,实测 −0.08,结论 ❌ GIL。
- `_scratch/attn_impl/T22-PROTOTYPE.md:111-112` — "bo.write 持 GIL,不能靠线程预填"。
- `docs/TIMING-RECONCILIATION.md:277` — 同族旁证:VAIP 原生代码持 GIL,4 线程 16.19 s ≈ 串行 wall(no gain)。

**对设计的影响**:重叠只能发生在 `run.wait()` 窗口里(且实测抢不到切片);**不要**设计"后台线程搬 staging"的架构;fill/use 顺序必须在主线程显式排序(F-02 纪律 2)。

**状态**:定案(诚实负结果,代码保留默认关)。

---

### F-04 · ERT 看门狗 ~7.0 s / dispatch

**事实**:每个 dispatch 必须在 **~7.0 s** 内完成,否则 ERT 判 `ERT_CMD_STATE_TIMEOUT`(state 8);这是**ERT 自己的固定看门狗**,不是我们的 `run.wait(TIMEOUT_MS)`,把 `T3_TIMEOUT_MS` 调大无效。超时后下一次 launch 变慢到 1.0–1.5 s(复位成本由下一次付)。

**出处**
- `webui/evidence/T20-STATE8.md:44` — A 臂逐次读数 `8:7017`(ms)。
- `webui/evidence/T20-STATE8.md:145-147` — R2/R3 逐字。
- `_scratch/block/STEP3.md:137,218` — 独立战役(block/FFN)同值:"~7 s is the ERT watchdog; it fired identically at M=16 and M=96"。
- `_scratch/vae/T21-3B2VF-REPORT.md:89-91` — VAE conv stall 的读数 `state 8 @ 7005.93 ms`。
- `_scratch/vae/T22-VAE-REPORT.md:94` — "一次 dispatch = `--runs 1`(ERT 看门狗 ~7 s);等待上限 600 s 只是 wait 的上限"。
- `_scratch/attn_impl/T22-BUDGET.md:19-22` — 平台事实表(L1 / memtile / program / watchdog 四行)。

**对设计的影响**:dispatch 时长是**硬预算**;长内核必须切块(切块后每块都要重付 staging);尺寸/批次预算以 7 s 为界;`--runs 1` 成为默认(与 F-01 叠加)。7 s 是**上限**,不是性能目标。

**状态**:定案(三处独立读数 7017 / 7005.93 / "~7 s")。

---

### F-05 · memtile 每 tile 只有 6 S2MM

**事实**:memtile 每 tile **6 MM2S / 6 S2MM / 48 BD / 512 KB**。把 per-core 流从 4 扩到 8 时 aiecc **逐字拒绝**:`tile (1, 1) requires 8 input/1 output DMA channels, but only 6 input/6 output available`。推论(已采纳为设计约束):G_GU 硬上限 **6**;而 n_tiles1=160 要求 G_GU 整除 160,≤6 且 >4 的只有 **5**。

**出处**
- `_scratch/block/T5.md:117-120` — 四行 target-model 读数(compute tile / memtile / shim / 阵列规模),由 `_scratch/attn_impl/t5_devres2.py` 从工具链 target model 直接读出(py3.12,器件无关)。
- `_scratch/perfc/T22-UTIL2-MEM-REPORT.md:97` — W5 墙表行(逐字拒绝 + "G_GU 的硬上限是 6")。
- `_scratch/perfc/T22-UTIL2-MEM-REPORT.md:59,173` — R4/8 流路线"实测不可行"、封顶 G_GU≤6。
- `docs/HANDOFF-2026-10-19.md:152` — 墙表复述(`memtile 512 KB/6 MM2S/6 S2MM/48 BD`)。
- commit `050a7cd`(t22_util2:R4 被硬件封顶)。

**对设计的影响**:任何"每核多流"的加宽先查 memtile 通道账;**tile 数不是墙,memtile 是**;G_GU=5 是被 160 整除性逼出来的值(形状变了要重算,别当常数)。

**口径注**(C-1 裁决):这里的 6 是**编译期逐字判定的 DMA 通道数**(`tile (1, 1) requires 8 input/1 output DMA channels, but only 6 input/6 output available`),不是 switchbox 连接数;shim 的 "16" 属于另一口径(见 C-1)。

**状态**:定案(工具链读数 + 编译期逐字拒绝,双证)。

---

### F-06 · shim 同时活跃 BD ≤ 16

**事实**:shim tile 有 **16 个 BD id**;超出时 aiecc 逐字拒绝:`Too many simultaneously active buffer descriptors on tile (0,0), which supports up to 16`。BD 是"**同时活跃**"计数,不是总量:交付件里每个 id 被回收 ~1,900 次(`KNOWN-ISSUES.md:1282`)。计算 tile 也是 16 BD;memtile 是 48。

**出处**
- `_scratch/vae/T21-3B2VF-REPORT.md:94-97` — 逐字拒绝 + 具体诱因(2 个 TaskGroup 在飞,拆 3 份 fill 后每组 9×2=18 > 16;584×2 = 7 个/组 编译通过)。
- `_scratch/block/T5.md:117-119` — compute 16 BD / memtile 48 BD / shim 16 BD。
- `HANDOFF.md:184-185` — "A non-outermost zero-stride fill dim expands to **one BD per repetition**; the shim BD budget is **16**"。
- `_scratch/block/STEP6.md:423,452` 与 `_scratch/attn_impl/T3c-REVIEW.md:106` — 回收次数与"16 同时活跃"的同一条事实。

**对设计的影响**:每 dispatch 的**传输组数**要显式列(交付 FFN/attention 是 3 组 = 3 BD);零步长 fill 维度会按重复数**展开成多个 BD**,是常见爆炸源;多 TaskGroup 在飞时"每 fill 一个 BD"的写法视为越界风险。

**口径注**(C-1 裁决):shim tile = **2 MM2S / 2 S2MM**,**每通道 16 BD**(`HANDOFF.md:171-174`);`_scratch/block/T5.md:119` 的 "16 MM2S / 16 S2MM" 是 switchbox 行级连接计数,既不是通道数也不是 BD 数。

**通道账(2026-10-20,t22_qkv 实测)**:那 "16" 的另一个正确读法是**每设计的全部输入通道 = 8 列 × 2 in/列 = 16**。aiecc 逐字:`16/16 input channels used … this is the device's total ShimNOCTile DMA budget, not a placement choice`。⇒ 与上句"每 tile 2 in / 2 out"**互洽**(8 列合计 16);**>8 个 worker 的设计必须让 X 走多播或 memtile**,"多铺 tile 提带宽"的方案**先过这条账**。出处:`_scratch/block/qkv/QKV-DESIGN.md:47,77-80`。

**状态**:定案。

---

### F-07 · L1 65,536 B / 核 vs 程序区 16,384 B

**事实**:compute tile 的 L1 数据内存 = **65,536 B(64 KB)**;per-core **程序区 = 16,384 B(0x4000)且是硬件常量**(生成的 ld script `LENGTH=0x4000`、aiecc 自己的 "covers ALL of the program memory"、硬件寄存器库 regdb)。`-O2` 溢出 20,992 B、`-O1` 溢出 17,040 B,**只有 aiecc `-O0` 能进** —— 这就是 `llc -O1` 注入法的由来(直接 `llc -O1` 核心 .text 3,024 B vs `-O0` 4,816 B)。

**出处**
- `HANDOFF.md:158-163` — 16,384 B 是硬件 + 三个独立佐证 + 溢出数字。
- `_scratch/block/STEP2.md:96-106` — "The 16 KB program region is the binding constraint of a one-core chain"(含 ld script 片段与无 `--gc-sections`)。
- `_scratch/block/STEP2C.md:183` — 实测 `.text` = 16384 B = 正好用满。
- `README.md:316` — "L1 is 64 KB per tile — K=1792 uses 64,768 of 65,536 B"。
- L1 用满的实例:`README.md:124-125`("L1 at the norm core measures 65,280 of 65,536 B")、`_scratch/perfc/T22-UTIL2-MEM-REPORT.md:95`(W3:30,720+7,680+23,040 = 61,440 / 65,536 = 94 %)、`_scratch/attn_impl/T22-BUDGET.md:19`。

**对设计的影响**:①内核代码预算是 16 KB(含每个被链接的 kernel 对象),单核链"能不能塞下"先算这个;②L1 缓冲区布局决定 m/Kc 上限(norm core 94 % 即天花板,再加一块就 "buffer could not be placed");③"L1 64 KB" 只对**数据内存**成立,程序区是**另一个** 16 KB,两者不可互换。

**状态**:定案。

---

### F-08 · 声明 data 参数 ≤ 7

**事实**:内核 ABI 的**声明 data 参数上限是 7**。实测:10 / 9 / 8 个全部 `ERT_CMD_STATE_ABORT`,7 个完成;此前最多只证到 6。规避 = **把同形状流 concat 成一个 buffer**(或打包成 packed element),而不是加参数。另一种失败形态:**多传一个 BO 即 `0xC0000005`**。

**出处**
- `HANDOFF.md:168-170` — 上限 + 三种失败 + 规避建议(逐字)。
- `_scratch/attn_impl/T3d-M1d.md:89-90` — 交付 attention 镜像正是 "7 个参数(ERT 上限),全部 64 B 对齐"。
- `_scratch/vae/T21-3B2VF-PLAN.md:148` — "平台的 7 个 data 参数(x,y,gamma,beta,stats,n,spatial),加第 8 个会被拒"。
- `_scratch/vae/T21-3B1-U1-PAPER.md:25` — 另一形态:多传一个 BO → 0xC0000005。
- `docs/HANDOFF-2026-10-19.md:153-154` — "ERT 只能传声明参数" + 墙表"数据参数 ≤7"。

**对设计的影响**:新算子加参数前先想 **concat / packed element / memref 偏移** 三条路(3B2VF 的 GN 就是靠 memref 偏移避开第 8 个参数);7 是硬顶。

**裁决注**(C-2,2026-10-20):**≤7 是硬限**;失败形态随路径不同——声明参数数超 → `ERT_CMD_STATE_ABORT`,绑定 BO 超 → `0xC0000005`;两者**都致命、都不可捕获绕过**,设计侧只能预防,不要设计"捕获后重试"。

**状态**:定案。

---

### F-09 · teardown 崩溃是常态;退出码不可信

**事实**:`0xC0000005`(−1073741819,access violation)在进程 teardown **常态出现**,且发生在产物落盘**之后**;switch-only 进程 4/4 都在写完 JSON 后崩(**即使工作完全成功**)。首 dispatch 也偶发 `0xC0000005`。因此:**退出码不是判据**,判据是 `*_calls.json`(存在 + dispatches 数对 + retries)+ 落盘产物。

**出处**
- `_scratch/block/STEP5.md:592-601` — 决策原文:"for these runs the exit code is not a verdict and must not be used as one — the JSON artifact written before teardown is the measurement";并给出"registered, never dispatched ⇒ crash at teardown"的相关性(未再隔离)。
- `_scratch/block/STEP5.md:607-608` — 4/4 逐字:`[run] 3.10 runner exit code: 3221225477`。
- `_scratch/block/T10.md:300-302` — 首 dispatch 0xC0000005 命中 3 次 + "判据是 *_calls.json,不是退出码(teardown 常态 -1073741819)"。
- `_scratch/block/T7.md:278` — R3 "teardown 崩溃已经是常态"(5 个全量臂 + 3 次 smoke)。
- `_scratch/attn_impl/T3d-M1d.md:57,200` — 既定规则:"teardown 崩溃不算失败,产物即测量"。
- `docs/HANDOFF-2026-10-19.md:115` — 环境事实第 1 条(自动换新进程重试 3 次)。
- `README-WEBUI.md:369` — 产品侧对策:`os._exit` 避开 teardown。

**对设计的影响**:harness **不得**用退出码判失败(否则会否掉 3/4 条好测量);每次器件测量后先查 `calls.json`;长期僵尸 pyxrt 进程会吃设备上下文并制造"假内存墙"(18 MiB 分配失败),测量前先确认无残留(F-14)。

**状态**:定案。

---

### F-10 · aiecc 单线程;并行只能按核

**事实**:`aiecc` 编译 **CPU/wall = 1.00**(单线程)。20 核机器上按核并行可行:同一单的 id 51 与另外 6 个探针并发编译,全部正常完成、互不抢核 —— 这是把 6.1 h 压到 ~1 h 的唯一办法。单镜像 aiecc ≈ 18 s(干净窗口),整尺寸 20–40 s。

**出处**
- `_scratch/vae/T21-3B2-PLAN.md:618-619` — 实测 + 建议(逐字含 CPU/wall=1.00)。
- `_scratch/vae/T21-3B2-BATCH.md:102` — 同一读数用于批量排程表。
- `docs/HANDOFF-2026-10-19.md:162` — 编译账(单镜像 ~18 s / 整尺寸 20–40 s / 产物 <1 MB)。

**对设计的影响**:大扫描不要串行等单次构建内的并行;按核起多个独立 aiecc,但**磁盘与 .prj 要逐行清**(单行峰值 15.8 MB,最低 free 曾到 867 MiB);与**器件**测量错峰(F-14),编译并行不占器件。

**状态**:定案。

---
### F-11 · BFP16 vs BF16:术语与格式口径

**事实**:本项目的"低精度"是**两种**格式,任何字节账/带宽账/参考模型都必须先说是哪一个:

| 记号 | 含义 | 带宽 | 出现在哪 |
|---|---|---|---|
| **BFP16**(`v8bfp16ebs8`) | 块浮点:8 元素共享 8 位指数 + 每元素 8 位尾数 | 9 B / 8 元素 ≈ **1.125 B/参数** | **权重 / KB / VB** 流;`vae_weights.bfp16` |
| **BF16**(`bfloat16`) | 16 位浮点 | **2 B/元素** | **QP / P / S / O / SBG / X / A** 流(激活/载荷) |

配套口径:**"重舍入"= BFP16 分块量化**(pack 用 floor/向 −inf 截断);`mm_bfp(mixed=True)` 的 **C 输出每 GEMM 后回 BF16 RNE**。

**出处**
- `_scratch/attn_impl/T22-PROTOTYPE.md:118-122` — 术语更正原文(权重/KB/VB = BFP16;QP/P/S/O/SBG/X/A = BF16;"重舍入"应称 BFP16 分块量化)。
- `_scratch/vae/T22-VAE-REPORT.md:59-70` — 负责人 2026-10-20 指令下的同一张口径表 + 逐流字节账规则 + 新算子 ABI 声明。
- `README.md:45-58` — 编码器定义(9 B/8 元素;e = floor(log2 amax) − 6;half away from zero;exponent byte (e+133)&0xFF)。
- `README.md:221` — "BFP16 (`bfp16ebs8`, 1.125 B/param) — explicitly *not* bf16"。
- `README.md:329-332` — C 侧 BF16 RNE 的后果:参考模型必须逐次回 bf16,否则在 K=1792 漂 13 ulp、"看起来像 bug"。
- `HANDOFF.md:19` — 6,143,056,384 参数、3.554× 压缩、1.125 B/参数。
- commit `b5a08a4`(t22_util2:terminology 定案)。
- 内核侧确认:`COVERAGE.md:16`(`mm_bfp` / `mm_bfp_mixed`)。

**对设计的影响**:①凡写"13 GB 权重"必须注明是 BFP16 口径;②参考模型要按"bf16-Q × f32-K"建,若没把 K 的 BFP16 分块量化放进参考,0.6–7.6 % 的残差是**预期**而非布局错误;③"BF16 重舍入"和"BFP16 分块量化"不能互称 —— 前者是 C 输出的 RNE,后者是权重打包的块量化。

**状态**:定案。

---

### F-12 · (K=2336 ∧ N=256) stall → 必须 N=128 半宽

**事实**:conv dispatch 在 **K=2336 且 N=256** 时**不产出**(输出全 0,ERT 7 s 判 state 8,**不是"慢"**)。与 M 无关、与 A fill 重复计数无关、与 Kc 无关;把 **N 切两半(N=128)** 后每块 250 ms 跑完、`ulp 0.0 / bit_exact 1.0`。触发面 9/9 命中无例外;同 K、N=128 的 3 行通过;同 N=256、K=1184/4640 的多行通过。**根因未定案**(没做 kernel 级 trace)。

**出处**
- `_scratch/vae/T21-3B2VF-REPORT.md:86-92` — 五步单变量发现路径(步 5 = N 切两半 PASS)。
- `_scratch/vae/T21-3B2VF-REPORT.md:101-112` — "知道什么/不知道什么":触发面 + 无关项 + 诚实的"根因未知"。
- `_scratch/vae/T21-3B2VF-REPORT.md:327` — 门表行:"**必须** N=128 半宽 dispatch;整宽会 stall(不是慢)"。
- `webui/evidence/T21-3B3-VAE.md:270-271` — 已写进执行器:`select_conv_image` 的强制分支 + 对 stall 镜像的断言(不是注释)。
- `_scratch/vae/T21-3B2VF-PLAN.md:78` — K=2336 / N=128 的 3 行全部 PASS 的对账。

**对设计的影响**:新 conv/GEMM 几何先查这条触发面(K=2336 **且** N=256);执行器保持"强制半宽 + 断言";`N=128` 镜像是可复用的(N 半宽不要求重编译);时间预算按"整行 4.4 s = 4 块 × 2 半 × 250 ms"算,别按整宽算。

**状态**:经验定案(逐位等价 + 9/9 命中);机制未定案,不得写成"已知根因"。

**量化收口(2026-10-20)**:这个"K=2336 ∧ N=256"的经验触发面已被 **F-24** 收成一个单式 `F = rr²·N²/K`(与 M 无关,现 23/23 分离);新几何进门**先按 F 打分**。

**订正 2026-10-20(t22_stallprobe,append-only)**:触发面**不是 (2336,256) 这一个点**,而是一个可量化的曲面。23 个不同 (K,N,Kc) 点(16 个 M=64 新臂 + 3b2 存量 7 个)上唯一能 **23/23** 分开的单式是 `F = rr²·N²/K`(rr = row_repeats = (N/16)(K/Kc)):F ≤ 3.07 M 通过、F ≥ 5.69 M stall,窗口 ×1.85,leave-one-out 稳定。已点名否证:A fill 字节体积律、rr 单标量律、N 带 / K 带律。**M 与 rep_cap 都不搬动 F、也都不搬动判定**(M=32/64/1024、rep_cap=584 四臂同几何全 stall)。**K=1184 在 N≥184 也不安全**((1184,224)/(1184,448) 双实测 stall)。机制(哪个计数器/握手)**仍未定案**。出处:`_scratch/vae/stallprobe/README.md` §4 / §5 / §9 + `out/arms.jsonl`(每臂 wall/state/verdict)。

---

### F-13 · 模型与形状算术(项目常量)

**事实**:目标模型 Z-Image-Turbo:**6,143,056,384 参数、34 blocks、hidden 3840**;**步数锁 8**(turbo 训练目标,不能多不能少);token 数 `M = 32 + ceil(((H/16)(W/16))/32) × 32`(32 = caption 行,image token 按 32 向上取整)。

**出处**
- `docs/HANDOFF-2026-10-19.md:11` — 项目一行速读(6.14B / 34 blocks / hidden 3840 / 步数锁 8)。
- `README.md:17,94-96` — 1.125 B/参数(存储/常驻);`README.md:94-96` = 6,143,056,384 参数、3.554× 压缩。
- `README-WEBUI.md:39,159` — M 公式(API 参数表 + 形状算术说明)。
- `webui/evidence/T19-256x256.md:45` — 实例:256×256 → 16×16 patch → 256 token → M = 288。
- `webui/evidence/PRODUCT-CHECKLIST.md:56` — 产品自查项:"尺寸算术 … 不是拍脑袋的 H*W"。

**对设计的影响**:所有预算/门/缓存目录名都按这条算术;H/W 必须是 16 的倍数;padding 分支(144×128 类)走单独路径;改 M 公式等于换参考,必须重跑全部门。

**状态**:定案。

---

### F-14 · 器件是共享资源:第二个使用者与争用漂移

**事实**:同一时段存在**第二个器件使用者**时,会出现 `0xc01e0009`(Failed to create context virtual);环境干净(测量前确认 0 个 python 进程)时不复现。并发争用会让**同一条 dispatch 的时长漂移**(本战役记录到 304 ms → 44 ms 量级的同类漂移)。kill 只杀外层进程时,**孤儿链进程会继续活着并各自再养一个服务进程**、继续占器件;僵尸 pyxrt 进程还会制造**假内存墙**(18 MiB 分配失败)。

**出处**
- `_scratch/vae/T22-VAE-REPORT.md:218-225` — 三条环境事实(0xc01e0009 三次全在"第二个使用者"时段;孤儿进程;僵尸进程吃上下文/假内存墙)。
- `_scratch/attn_impl/T22-PROTOTYPE.md:51` — "机器上有 ~8 个 pyxrt teardown 卡死的僵尸 python"。
- `_scratch/perfc/T22-UTIL-REPORT.md:150` — 归属判定:"这属于环境,不属于改动"(两次 b0 停在 2.47 GiB 分配)。
- 纪律侧:`docs/HANDOFF-2026-10-19.md:117` — 机器快/慢双态,性能数必须**同 session A/B**、双状态都报。

**对设计的影响**:测量前跑"0 个残留 python"检查;遇争用**标注后重测**,不得把漂移写进结论;性能主张必须同 session A/B;跨会话比较速度 = 无效证据。

**状态**:定案(定性;漂移幅度环境相关,不写死数值)。协议见 **F-14.1**(2026-10-20 起强制)。

#### F-14.1 器件互斥锁协议(2026-10-20 起强制)

> 并发使用器件正在互毁数据(state-8 风暴、计时漂移),"错峰"由此升级为**硬锁**。

1. **锁文件**:`zimage-bfp16-npu/_scratch/device.lock`,内容一行:`owner=<单名> pid=<进程号> acquired=<ISO 时间> purpose=<用途>`。
2. **取锁 = 原子创建**(Python `os.open(..., O_CREAT|O_EXCL)` 或 PowerShell `New-Item`;**创建失败即视为被占**)。**任何器件 dispatch 运行前必须持锁**;运行期间可续期。**`pid=` 必须写"真正持有器件、正在跑"的那个进程**(不是取锁脚本/helper 的进程):用一次性 helper 取锁时它写下的是 helper 自己的 pid,helper 一退出就成了死 pid,45 min 后被判"孤儿"而遭接管 —— t22_large 已因此发生**一次违规接管**。出处:`_scratch/vae/t22_devlock.py:98-101`(写入 `os.getpid()`)、`:85-86`(按 pid 判活);负责人 2026-10-20 通报。
3. **放锁 = 运行结束立即删文件**;建议按"批次窗口"用锁(一次取锁跑完一组实验再放),**不要**每次 dispatch 取放。
4. **接管**:锁龄 > **45 min** 且 owner 进程已死 → 可接管;覆盖锁文件时在内容里注明 `stale-takeover`。
5. **编译(aiecc)不占锁、可并行**;只锁**器件 dispatch**。
6. **拿到锁之前只做 CPU 侧工作**(设计 / 编译 / 文档)。

**出处**:2026-10-20 负责人指令(器件互斥锁协议);事故背景见 `_scratch/vae/T22-VAE-REPORT.md:218-225`(第二个使用者 → `0xc01e0009`)与本卡正文的计时漂移。

**对设计的影响**:测量脚本流程 = **持锁 → 确认 0 个残留 python → 跑批 → 放锁**;无锁时的器件数一律作废重测;编译/设计/文档不占锁,但**不得**在无锁状态下 dispatch。

---

### F-15 · 宿主侧并发 launch 不并行(x1.007)

**事实**:同进程、两个 kernel handle、两个不同 xclbin(att 5 tile / ffn 17 tile),`launch(att); launch(ffn); wait(att); wait(ffn)` 与顺序等待对比:**44.0 vs 44.3 ms/pair,x1.007**,state 全 4(COMPLETED)。并发 launch **不带来任何吞吐**;tile 占用 38 % 也不能靠宿主层并发补。

**出处**
- `_scratch/perfc/T22-UTIL-REPORT.md:166-173` — §6.2 逐字(`seq 44.3 / conc 44.0 / x1.007`,附带"若真能重叠应有明显收益"的推理)。

**对设计的影响**:不要设计"多 xclbin 并发流水"来填 tile 空洞;barrier(F-01)的成本就是**整段串行时间**;吞吐只能从单 dispatch 内部(分块/kernel 效率)拿。

**状态**:定案(实测;与 F-01 的"第二 handle 假完成"不冲突:那是**同一 xclbin** 的第二 handle)。

---

### F-16 · 环境块(照抄可用)

**事实**
- 两套解释器**不可互换**:`py312` = 编译/构建/服务(`<python312>\python.exe`);`py310` = 器件循环(numba、pyxrt;`<python310>\python.exe`)。
- `PEANO_INSTALL_DIR` = `<工作区根>/_pylibs/llvm-aie/llvm-aie`;`PATH` += `<xrt>`。
- 每次器件运行:**fresh** `NPU_CACHE_DIR` + `NPU_CACHE_HOME`(建目录后 0o700 → 0o777,shim 需要);手跑 bridge 需 **fresh** `NUMBA_CACHE_DIR`。
- 设备选择用 `set_current_device(NPU2())`,**不要** `from_name("npu2")`(那是 `npu2_1col`,只暴露一个 shim tile,会以"像硬件限制"的方式拒绝多 fill 设计)。
- aie 导入顺序:设计模块(改 `sys.path`)必须先于 `from aie.iron import`。
- **沙箱 mkdir 坑(会伪装成设计失败)**:`os.mkdir(p, 0o700)` 在本机沙箱建出的目录**连创建者都不可写/不可列**;而 `tempfile.mkdtemp` 硬编码 0o700,且 aie 的 `prefix_symbols_in_object()` 会把 TemporaryDirectory 建在 `.prj` **里面** ⇒ 编译死在 `open(<prj>/aie-symbol-map-*/symbols.map,"w")` 的 `PermissionError`,并被清理路径**再吞一层**(报错看起来像别的问题)。修法:编译脚本把 `os.mkdir` 强制成 **0o777**(只影响构建)。
- **`xclbinutil` 必须在 PATH**:aiecc 最后一步要它;编译进程的 PATH 要含 driver376 目录,否则 `tool 'xclbinutil' not found in search paths or PATH`(末步才炸,前面全绿)。

**出处**
- `docs/HANDOFF-2026-10-19.md:99-112` — 环境手册(逐字,含 0o700→0o777 与 NUMBA_CACHE_DIR)。
- `README.md:127-132` — `set_current_device(NPU2())` 陷阱(逐字"reads like a hardware limit but is not")。
- `docs/HANDOFF-2026-10-19.md:120` — aie 导入顺序。
- **本卡 2026-10-20 复核**:上述 4 条绝对路径 `Test-Path` 全部 `True`(py312 / py310 / PEANO / driver376);同次复核中 `fsutil / Get-CimInstance` 查询磁盘被沙箱拒绝,故磁盘余量见 §3.1(时点读数,非事实卡)。

**对设计的影响**:任何"在新 shell 里跑一下"的脚本必须先设这四件事,否则失败原因会被误判为器件故障;`from_name` 陷阱会造成"设计看起来不可行"的错误结论(已发生过)。

**状态**:定案(路径 2026-10-20 复核存在;缓存/导入顺序为历史实测)。

---

### F-17 · m/Kc 轴闭合(可用 L1 34,816 B ⇒ m·Kc ≤ 17,408)

**事实**:norm core 的 L1 常驻 = chunk(`m·Kc·2`) + inA 一行(7,680 B) + inPAR(23,040 B) ⇒
**可用 chunk 上限 = 65,536 − 7,680 − 23,040 = 34,816 B ⇒ m·Kc ≤ 17,408**。
在"Kc 整除 3840 且是 64 的倍数"的约束下逐个解:`(m=16, Kc=960)` n_chunks=4、norm 28.8 ms、dispatch **39.7 ms**(被 GU 墙顶住)= **交付点即最优**;m=16/Kc=1920 的 chunk 是 61,440 B > 34,816 **建不出来**;m=32 只能 Kc≤544 → n_chunks≥10 → norm 72 ms → dispatch ≥75.9 ms(2× 变差);m=96 → Kc=128 → ≥220 ms。**任何 m>16 都以压死 Kc 为代价,m/Kc 这条轴在交付架构上闭合。**

**出处**
- `_scratch/perfc/T22-UTIL3-U1P-DEATH-TABLE.md:20-36` — §2 全表(公式行在 :22,结论在 :33-34)。
- `_scratch/perfc/T22-UTIL3-U1P-DEATH-TABLE.md:38-41` — §3 自我订正:"我上一轮 −4.8 s 的账错在哪"(两个假设都不成立 ⇒ U2/U3 与 −4.8 s 一并作废)。
- T12 校验点(表的线性外推锚):`_scratch/block/T12.md:58`(base norm **28.78 ms**,构成 rms 20.60 + adaln 4.57 + copy 0.35 + 搬运 3.26)与 `_scratch/block/T12.md:134`(交付 `t11_nh5` dispatch 39.86 = max(39.70, 32.68))。
- 同一个 norm 时间在汇总表里也被引用:`_scratch/perfc/T22-UTIL2-MEM-REPORT.md:48`。
- commit `2413f89`(t22_util3 U1-prime:"the norm-core L1 caps m*Kc<=17408 so m/Kc axis is CLOSED … −4.8 s target withdrawn")。

**对设计的影响**
1. **不要再提"m 翻倍"或"Kc 翻倍"类优化** —— 两者同时被 34,816 B 钉死,已闭合;要动必须换 norm core 的存储架构(报告 §5 明确列为独立立项级)。
2. 34.816 KB 是**该常驻集**下的可用上界,**不是**"L1 可用总量":墙表里另有 "L1 61,696 B 可用"(`docs/HANDOFF-2026-10-19.md:152`),那是另一组常驻(总量减栈),两者口径不同,引用时必须带常驻集。
3. 反查纪律:凡设计声称"L1 还够"先列常驻集,再加总到 65,536。

**口径更正(2026-10-20,t22_normcore §10.1;commit `c059150`)**:**本卡与 kgu 无关,结论继续成立。** m·Kc ≤ 17,408 是**只由 norm core 的 L1 账**推出来的 —— `65,536 − inA 7,680 − inPAR 23,040 = 34,816 B`,chunk = m·Kc·2,**式子不含 kgu**(norm core 不碰 B)。kgu=8 与 kgu=64 逐行相同:(m=16, Kc=1920) 的整 chunk 61,440 B + inA/inPAR = 92,160 > 65,536 ⇒ **整 chunk 确实建不出来**;m=32 ⇒ Kc≤544、m=96 ⇒ Kc≤181 也不因 kgu 而变。

**本轮新增边界**:把 chunk 拆成**半块对象**(N1)后同一 m 下 Kc 可翻倍(半块 = m·Kc/2·2),但**半块要重算 rms**,norm visits 不降(Kc 减半 × 半块 ×2 相抵)⇒ "L1 可建" ≠ "norm 变快";m=96 用 6 个半块要把 rms 重算 6 次(norm ≥150 ms),**半块法救不活 m=96**,必须走 A-plane。出处:`_scratch/block/normcore/NORMCORE-REPORT.md` §10.1(commit `c059150`)。

**同族代价:编译墙 ∝ Kc=32(2026-10-20,t22_stallprobe)**:Kc=32 的臂 aiecc 墙钟 **332.1 / 335.5 / 335.7 / 340.7 / 347.0 s**,而 Kc≥64 只要 **13.6–19.4 s** —— **慢 23×**,且与 M、rr、N **全部无关**((4672,128,Kc=64) rr=584 只要 14.3 s,而 (2336,128,Kc=32) 同样 rr=584 要 332.1 s)。慢在 `aie.mlir` 之后、`input_with_addresses.mlir` 之前的 **AIE lowering**,不是 Peano 核心编译。⇒ 选 Kc 时要把这笔编译账一起算;**批量重编 K=2336 族时按 5.5 min/镜像排程**。出处:`_scratch/vae/stallprobe/README.md` §3.3 与 §9(该文件仍在追加,按 § 号定位)。

**限定语(2026-10-20,见 F-25)**:本卡的"闭合 / 最优"是 **本架构下** 的结论 —— 它建立在"整 chunk 常驻 norm core L1"这一存储架构上;换存储架构(流式 norm、chunk 由 DMA 直接拼出、A-plane 单次归一化)会改变这条轴,**不能用本卡去否定那些路线**。

**状态**:定案(可复算:kgu 无关,§10.1 已独立重推;常数取 L1 65,536 = F-07,锚点取 T12 实测 28.78 ms)—— **限本架构**。

---

### F-18 · aiecc 分配器 bank 打包限制(C 缓冲放不下)

**事实**:R4′(G_GU=5)的 C 缓冲从 **10 → 8** 后**仍然失败,且失败点没变**:
`buffer "C4_4" needs 2048 bytes and this tile has no room left for it`(aie.mlir:150,`input_with_addresses.mlir` 边)。
读法:**第 5 个 GU 核在第 5 个 C 瓦片就放不下**,说明问题不是"10 个太多",而是该核 L1 已被 outS/midB/BG/A 这些广播缓冲占满(**bank 3 = 16 KB 已满**)——这是**分配器的 bank 打包问题**,不是硬件通道上限。要修需要按 bank 做放置(或改广播拓扑),超出 0.5 h 预算。

**出处**
- `_scratch/perfc/T22-UTIL3-U1P-DEATH-TABLE.md:57-66` — §6 全过程(改动、逐字错误、读法、判)。
- `_scratch/perfc/T22-UTIL3-U1P-DEATH-TABLE.md:43-50` — §4 路线表:该报告的"便宜修法"(10→8)就是从这里来的预测,已被 §6 否证。
- commit `167aff1`(R4-prime parked)+ `_scratch/perfc/T22-UTIL3-U1P-DEATH-TABLE.md:64`("R4′ 暂时挂起(不是死,是 0.5h 修不动)")。
- 更早的 C 放置失败(另一个空 tile (1,3)/(5,2) 同样失败)见 `_scratch/perfc/T22-UTIL2-MEM-REPORT.md:98`(W6)。

**对设计的影响**:①"减少缓冲个数"不是解法,**bank 级放置**才是;②遇到 "needs N bytes and this tile has no room left" 先看**广播缓冲占了哪个 bank**,不要先砍你自己的缓冲;③这类失败**只花 0.5 h 就能判"修不动"**,应作为可行性 spike 的标准时长(见 F-19)。

**口径更正(2026-10-20,t22_normcore §10.2;commit `c059150`)**:**"C4_4 needs 2048 bytes / no room" 是 kgu=64 实验镜像专属,交付口径不成立。** 账面:B pair 对象 2·(64·64/8)·9 = **9,216 B**(kgu=64) vs 2·(8·64/8)·9 = **1,152 B**(kgu=8);inBG 5 深 46,080 B vs 5,760 B;GU core L1 **~73 KB(C4_4 放不下)vs ~33 KB(放得下)**。交付 kgu=8 下 **R4′ 已经编译过**(18 core,xclbin 121,259 B,inst 11,864 B,25.5 s);**真正的墙换成 join 5 生产者 BD**:5×2×3 = **30 > 24**(**kgu 无关**),depth 1 可解。⇒ 本卡"分配器 bank 打包"仍是通用平台事实,但**不能当作交付口径的阻塞**。

**状态**:实测(失败逐字 + 判据);**挂起**与否由负责人裁决。**W6 / C4_4 = kgu=64 实验镜像专属**,见上段与 F-18.1。

#### F-18.1 · R4′ 的现状与计划更正(2026-10-20)

- **已编译**:交付口径 kgu=8 下 **R4′(G_GU=5, NH=4)通过编译** —— 18 core,xclbin **121,259 B**,inst 11,864 B(25.5 s)。原判"放置死"**作废**。
- **计划更正**:R4′ + Kc=1920 的预测收益是 **−1.7 s/图**,**不是 −1.9 s**。−1.9 s 那个数属于 **N4 + 单次归一化(A-plane)** 组合(31.8 ms,−7.9 ms/层);N1(半块)与 G_GU=5 叠加仍是 32.7 ms,**两者不可混用**。
- **要再进一步**:必须让 norm **只归一化一次**(A-plane),约 **1.5 个设计单元**。
- **判定性实验(已建,待锁)**:实验 A = G_GU=5 计时(39.7 ms ⇒ "总字节/全局资源"是墙;~32.7 ms ⇒ "每 tile 算力"是墙);实验 B = B-cache 变体(把 BG 的 host tap 改成从 5,760 B 前缀反复读,使 B DRAM 字节→0)区分字节墙与算力墙。
- 出处:`_scratch/block/normcore/NORMCORE-REPORT.md` §10.2–§10.3 与 §6(commit `c059150`)。

---

### F-19 · 方法卡:设计前可行性 spike(三条构造的逐字死因)

**事实**:任何"看起来能省 L1/能合流"的构造,**先做 0.5 h 编译 spike**,不要直接进设计。三个已经被逐字判死的方向(2026-10-20,U1′):

| 构造 | 逐字死因 | 结论 |
|---|---|---|
| (a) `delegate_tile=Tile(0,1)`(把 chunk 池委托给 MemTile) | `'aie.objectfifo.allocate' op objectfifo has no shared memory access to delegate tile's memory module` | 委托不可用 |
| (b) 非对称 fifo(producer 30,720 → consumer 61,440) | `producer element size (30720) must be an integer multiple of consumer element size (61440)` | 该 API 只支持 **producer 大 → consumer 小**(拆分),不支持合并 |
| (c) memtile join 两个 30,720 生产者 | 能建(现有 outS 就是 join),但两个 producer 各自要在 core 里 staging 30,720 → 合计 61,440 | **不解决 L1** |

**出处**:`_scratch/perfc/T22-UTIL3-U1P-DEATH-TABLE.md:12-18`(§1 表);commits `d760061`(delegate 被 verifier 逐字拒绝)、`66397a4`(把该问题记成 2 h spike)、`2413f89`(三构造全死 + 轴闭合)。

**对设计的影响**:①"能建"≠"有收益"(c 就是典型:join 能建但 L1 一分不省);②失败要留**逐字**错误,否则下一轮会重试同一个构造;③spike 时长上限 0.5–2 h,超了就判"修不动"并挂起(F-18 的先例);④可行性 spike 的结论要进本文档(否则下一代会再试一遍)。

**同族检查单(2026-10-20,t22_sring;**已升格为 F-27,以那张卡为准**)**:**`objectfifo` 的**元素尺寸必须与 fill 实际搬运的字节数相等** —— BD 只在"元素被填满"时才完成;元素写大(NK 块)而 fill 只搬 1 块,会**静默超时**(state 8 + 拆除 6),看起来像"这个 ring 复用不被支持"。p3 的 7 个配置全栽在这两行上(`kb_ty = np.ndarray[(NK*KB_BYTES,)]` vs `KB_tap[b]`)。出处:`_scratch/attn_impl/sring/SRING-REPORT.md:70,79`。

**交付证据的优先级(2026-10-20,t22_ffnsem3/4)**:仓库里的 `_scratch/block/t11_ffn_nh5.py:683` 写的是 `sloop = range_(sweeps)`、`sweeps=40`;但**按产出/消费恒等式,交付 xclbin 必须是 sloop = 8** —— GU 每 rt 消费 **32** 个 chunk(`row_repeats = passes × n_chunks = 32`,结构常数),而 `sloop × n_chunks` 必须 = 32 ⇒ sloop = 8。若真是 40,norm 每 rt 要产 160 个行块而 A tap 只喂 32 个 ⇒ **第 33 个 chunk 就断粮、挂死在 `of_a.acquire`**;交付镜像是 **1440 dispatch / 0 重试**跑通的 ⇒ `.prj` 里的 `%9 < 40` 属**会挂死的构建 / 陈旧重建**。三重对账:产出/消费恒等式 + A tap 的 32 块结构常数 + **23.6 MB/dispatch**(3,072 × 7,680 B,与 T13 一致)。

⇒ **检查单**:①**交付二进制为准**,源码文本与 `.prj` 只当线索;②引用某算子语义前,先把它与 **xclbin 的指令流/BD 账**对账;③"源码算出的量"与"实测字节/调用数"矛盾时,**先怀疑源码漂移**,不要改模型去凑。出处:`_scratch/block/FFN-SEMANTICS4.md` §0、`_scratch/block/FFN-SEMANTICS2.md:25`。

**平台陷阱两条(2026-10-21,t22_swiglu 单实测;append-only)**

1. **`acquire(n)` 的契约是"补足到 num_elem",不是"取 n 个新元素"** —— 逐字:`If some elements are already acquired, only the additional elements needed to reach a total of num_elem are acquired` ⇒ 同一 fifo 上两次 `acquire(1)` **返回同一个元素**,于是"每 k-group 两个对象、两次 acquire"的写法会让**两个 matmul 都读 gate 瓦片**。判据:EPI=4(`mul(og,ou)`)的输出值集合 = **gate² 的排列**(sorted-corr 0.999998),EPI=2 为 0.999924。**修法**:一个对象装两半(`[gate|up]`)+ **一次 acquire** + 两个**常量偏移的 `memref.view`**。
2. **设备循环体里的第一个 kernel 调用只在第一轮生效** —— 两个 `zero` 只有第二个有效 ⇒ 第一块瓦片跨 pass 累积。判据:换序后缺陷**跟着调用走**(变成 up 累积);换不同符号的 zero kernel **无效**;去掉 `--reclaim-runtime-bds` **无效**;最前面加一个"牺牲"调用 `zero(tmp)` ⇒ **PASS**(corr 0.999986)。机制是"循环体第一个调用被吞",**与符号、与 tile 都无关**;牺牲品在被覆盖前从不被读,所以零成本。
   ⚠ **对交付镜像 `fs_ffn_j5` 的直接风险**:它的 `zero(Cs[2*j]); zero(Cs[2*j+1])` 正是这个形状(NH=1 也是两个);若机制成立,gate C 瓦片会在 `gu_iters>1` 时累积,而它**从来没有过数值门**(只有逐字节自比),不会自己暴露 ⇒ **建议 owner 复验**(该报告 §3(b) 已提)。

出处:`_scratch/block/swiglu/BIGM-SWIGLU.md` §3(a)/§3(b)(实验表 + 修法);同报告 §2 另给反汇编纪律(`vmac` 28 / `vlda.pop.576` 5,控制臂 `gu64c` 20/5)。

**工装纪律两条(2026-10-21,t22_stackhw;append-only)**

3. **bfp16 块操作数必须走 `vlda.pop.576` 路径** —— `block_vector<bfp16ebs8,64>` 是 9 B/8 元素的块类型,**没有对齐保证**,朴素指针解引用被编译成 **72 条 `lda.s8` + `vpush`**(逐字节拷贝):实测 **157 周期/MMA**,而交付库 `mm_bfp` 是 **1.68 周期/MMA**(**93× 慢**);库用的是 `vlda.pop.576`(一次 576 位装载 + 队列寻址)。⇒ **任何不照抄这条路径的探针都量不到硬件发射率**。
4. **MMA 速率探针的两道门(缺一不可)**:①**反汇编验 vmac 条数 = 循环体应有的条数,且相邻 vmac 之间无 `nopv` 填充**;②**装载路径与库内核同源**(否则量的是装载,不是 MMA)。反例:v1 四个累加器算**同一个乘积** ⇒ 编译器合成一条依赖链 ⇒ 量到的是**延迟**(1.34 ms / 4 vmac per iter);v2 换成四组操作数但三个累加器**末端不可见** ⇒ **被当死代码删掉**(反汇编只剩 **9** 条 vmac);v3 四路 `to_vector` 相加后才写回 ⇒ 15 条连续 vmac、无填充。

出处:`_scratch/block/stackhw/SPIKE-ISSUE-RATE.md`(速率表与判读:157 vs 1.68、72 条 `lda.s8`)、`_scratch/block/stackhw/ROUND2-MAC-WIDTH.md` §4(v1/v2/v3 与判据)、`_scratch/block/stackhw/STRUCTURE.md` §5(同两条)。

**状态**:方法条目(由三条实测死因淬出来的纪律,非平台测量)。

---

### F-20 · 纪律卡:共享仓库禁用 `git commit --amend`

**事实**:本仓库是**多代理共享**工作区,多个 agent 并发 commit。`git commit --amend` 改的是**当前 HEAD**,不是"你的上一个提交"。2026-10-20 10:30 实测事故:`t22_factcards` 改完 1 行表格转义后执行 amend,恰好撞上另一代理刚落下的 `9da5e6e`(t22_prodlarge),结果**把 factcards 的改动并进了 prodlarge 的提交**(HEAD 变成 3 files / 40 insertions / 13 deletions),违反"路径限定提交"。

**修复(已执行,记录在案)**:`git reset --mixed 9da5e6e` 复原他人提交 → 另立 `cafc3b9`。事故链见 `git reflog`(`cafc3b9` → `9da5e6e` → `2dddef3`)。

**纪律**
1. **禁止** `git commit --amend` / `rebase` / `filter-branch` / `reset --hard`(会改他人已落地的历史)。
2. 提交前 `git log --oneline -1` 记下**预期父提交**;提交后核对 `git show --stat --oneline HEAD` 的**文件集**是否只含自己的路径。
3. 路径限定:`git add -- <paths>`;必要时 `git commit -- <paths>`。
4. HEAD 已被他人推进 → **新建提交**,不改历史;发现误改 → `git reflog` 定位 + `git reset --mixed <sha>` 复原 + 单独提交。

**违例计数(截至 2026-10-20):2 起** —— t22_ffnlarge(`fc8da0c` 前后)与 t22_quickfix(`bb1d021` 前后)各发生一次 amend 撞车,均自纠复原。两次独立事故 ⇒ 本卡不是假想纪律。

**出处**:本卡执行时的事故(reflog 三段,2026-10-20)+ 负责人裁决"共享仓库禁止 git commit --amend"。

**对设计的影响**:提交纪律是可审计链的一部分 —— 串历史会让别人无法用 `git show` 审自己的模块。复原依赖 reflog,**不是永久可靠**(gc 后丢弃),所以预防 > 修复。

**状态**:纪律条目(事故实证,非平台测量)。

---

### F-21 · AIE2P 标量 f32 加法与 x86 SSE 的 tie-rounding 不一致

**事实**:同一表达式 `(float)bf16 + f32` 在 **AIE2P 标量 FPU** 与 **x86 SSE(RNE)** 上**不总是同一位模式**:每 1e6 元素里有 **1-2 个**元素差**恰好 1 个 f32 ulp(1.19e-7)**,形状像 exact-tie 的舍入方向不同。**单 launch 与分块 launch 都复现** ⇒ 不是调度、不是块边界、也不是额外舍入(内核算术就是 `y[i] = (float)a[i] + b[i]`)。

**证据**(参考 = numpy float32 加法,即 x86 SSE 的 round-to-nearest-even):

| 镜像 | N | bit_exact | max abs d |
|---|---|---|---|
| vaddf 单 launch | 131,072 | 0.999992 | 1.19e-7 |
| vaddf 单 launch | **1,048,576** | **1.000000** | **0** |
| vaddf 单 launch | 786,432 | 0.999997 | 1.19e-7 |
| vaddf 单 launch | 1,179,648 | 0.999998 | 1.19e-7 |
| vaddf 分块 2 × 2,097,152 | 4,194,304 | 0.9999979 | 1.19e-7 |
| vaddf 分块 4 × 2,097,152 | 8,388,608 | 0.9999989 | 1.19e-7 |

三条判读(源报告已由控制器验收):①**不是分块引入的**——单 launch 镜像的门同样是 0.999992–1.0;②差是**恰好 1 个 f32 ulp**,发生在 1–2 个/百万元素上,内核算术没有可指的额外舍入;③它比下游 **BF16 存储**的分辨率细 **16 bit**(2^-24 vs 2^-8),只在 **~2^-16** 的概率上改变下游 BF16 值。

**门禁含义(写进任何门的设计)**
1. 任何"**与宿主 f32 逐位相同**"的算子门,**不能**用 `bit_exact == 1.0` 判;**只能按幅值界** `ulp ≤ 1`(或等价绝对界)判。bit_exact 是诊断量,不是判据。
2. **端到端逐字节是概率性的,不能当门(更正 2026-10-20)**:差异比 BF16 存储分辨率细 16 bit,按 ~2^-16 的概率翻转下游 BF16 值。t22_vae 实测:**128² / 64×256 的逐字节是"概率未触发",不是保证**;**144×128 破了** —— add 上板后 PSNR 38.0508 → 37.9919(**−0.059 dB**),max|d| **0.138256**(该尺寸最大张量 2,359,296 元素,走 2×1,179,648 分块)。⇒ **端到端逐字节不可作为依赖 tie-敏感算子的门保证;此类门须改为幅值界,或做成真逐位实现。**
3. **必须逐位**时:把宿主参考换成**同一 FPU 实现**,或把结果**留在 BF16/整数域**比较。

**对设计的影响**:新算子(尤其标量 f32 输出)的门要写成"幅值界 + 记录 bit_exact 作诊断";不要因 bit_exact 0.999992 就判失败,也不要因某次 1.000000 就断言"与 SSE 同序"(N=1,048,576 恰为 1.0 就是反例:同族其它 N 不是)。**同样不要**把"端到端逐字节"当成 tie-敏感算子的稳定门 —— 144×128 已经破了。

**出处**:`_scratch/vae/T22-VAE-REPORT.md:284-310`(§3.2c 全节,含可直接引用的素材段)、`:260-262`(逐几何门表:silu/vadd 1.000000、vaddf 0.999992–1.000000)、`:279-280`(分块直测)。

**更正注(2026-10-20,append-only)**:本卡初稿写的"端到端**通常仍逐字节**"是**错的**。实测:add(silu / 残差)上板在 128²、64×256 逐字节,但 **144×128 不逐字节**(PSNR −0.059 dB、max|d| 0.138256;2.3 M 元素张量分块后,1-ulp tie 差翻转下游 BF16 值)。⇒ 逐字节在这类算子上是**概率现象,不是契约**;上面的门禁含义②已同步改写。出处:`_scratch/vae/T22-VAE-REPORT.md:274`(144×128 那一行)、`:279`(gate ② 红),§3.2a / §3.2c / §11;commits `a7f45e0` / `b5cdf8c`。

**同族注:swiglu 的 LUT 口径(2026-10-20,t22_ffnsem3/4)**:swiglu(`use_lut=True`)走的是 **32 段线性插值表** —— 每段宽 **0.5**、覆盖 **[−8, 8)**(x 先 clamp),索引 `e = clip(floor(x·2), −16, 15) + 16`,**每段线性** `y = slope_e·x + offset_e`,结果**只有一次 bf16 舍入(store)**。链上:`xw1 = bf16(x·w1)`、`xw2 = bf16(x·w2)`、`tanh_half` 由表得、`sigmoid = bf16(0.5·tanh_half + 0.5)`、`silu = bf16(max(xw2, −8)·sigmoid)`、`out = bf16(xw1·silu)`(`silu == 0` → 0)。

**口径差(引用时必须说清)**:构建开关 `ACTIVATIONS_TANH_LUT` 默认 **0(vtanh)**,而 `swiglu(use_lut=True)` 走**表路径** ⇒ 参考模型要对应 `swiglu_table_ref`;若某次构建实际落到 vtanh,则对应 `swiglu_lut_ref` —— **结构相同**(哪几步舍到 bf16、−8 的 clamp、`silu==0` 的门),**差别只在 tanh 的近似方式**。比较时若不声明是哪一版,**6.25e-02 量级的 max abs 会被误判成器件缺陷**(`_scratch/block/STEP3.md:65` 记录:swiglu LUT vs 精确 sigmoid 的 6.25e-02 **不计入 NPU**)。出处:`_scratch/block/FFN-SEMANTICS4.md` §2(表与链均在 §2.1–§2.3)。

**状态**:定案(平台事实;控制器 2026-10-20 验收,§3.2c 的三条判读即验收口径)—— **端到端结论按上面的更正注读**。

**同族但大三个数量级(见 F-31)**:器件 `aie::exp2<bfloat16>` 在 softmax 区间与正确舍入差**中位 10 ulp(≈3.9 %)**,而本卡的 tie 差只有 **1 ulp**。⇒ 引用 softmax / attention 的"接近 f32"之前,**必须先读 F-31**。

---

### F-22 · 语义陷阱:用 `finish()` 代替逐次 `wait=True` 会静默改变执行语义

**事实**:把"每次 fill 都 `wait=True`"换成"先发、最后 `finish()`"**不是优化,是换了一条 runtime 序列**。实测:同一设计的两种写法在 aiecc 里生成**不同的 .bin**(`17,328 B` vs 交付纪律版 `19,696 B`),设备上表现为**"变快 + 不可重复"** —— `0.991 ms` 的臂 **gate 0 FAIL**,两次跑输出差 **290,791 B**。

**判读**:`finish()` 只保证 TaskGroup 收尾,不保证每个传输按序完成;序列变短 = 语义变了。所以"更快"是假象,先崩的是正确性。

**纪律**
1. 所有器件路径**一律逐次 `wait=True`**;**禁止**用 `finish()` / `wait=False` 抄近路提速。
2. 任何新形态首跑**必须同时做 gate 0**(同输入两次逐字节),否则"变快"无法与"语义变了"区分。
3. 看到"某臂突然快一个量级"时,先查它生成的 .bin 与交付纪律版是否同一序列。

**出处**:`_scratch/attn_impl/t2/T2-PREDICTIONS.md:76-82`(g1seq 0.991 ms / gate0 FAIL / 290,791 B / .bin 17,328 vs 19,696;判读写在 :82)、`_scratch/attn_impl/t2/T2-MFU.md:284`(10:56 记录,该臂已作废);同一纪律的旧记录:`HANDOFF.md:175`、`KNOWN-ISSUES.md:1292`。

**状态**:定案(2026-10-20,t22_t2 实测;测量臂作废,纪律保留)。

**同族补充(2026-10-21,bigm §5.3)**:**`TaskGroup.finish()` 只 await 标了 `wait=True` 的传输,其余只是 free** ⇒ **drain 写 `wait=False` 会让内核报 COMPLETED 而 C 缓冲从未被写**。实测两条 `state 4`(1.433 / 1.349 ms)但输出**全零**,sha16 `c2cd1bef…` / `b6e6d0ef…`,后者正是 `BATCHM-DEATH-PROJ.md` 记的 **stale sha**。**修法:drain 一律 `wait=True`**(fill 仍可 fire-and-forget,由 fifo 锁限流);修后 `dma_await_task` 从 **0 变 510**。⇒ 这是本卡"假 COMPLETED"的**机制版本**:不是"序列变了",而是**完成信号根本没等到数据落地**。出处:`_scratch/block/bigm/BIGM-PLACEMENT.md` §5.3。

---

### F-23 · 换参考裁决:512²+ 的数值门对 **BFP16 精确仿真参考**,不对 host f32

**裁决(2026-10-20,负责人)**:512² 及更大尺寸的"一步数值保真 corr"**改判为对 BFP16 精确仿真参考**(逐位 / corr);**对 host f32 的 corr 降为信息项**。**128–256² 口径不变**(仍对 host f32:128² 0.999959、192×128 0.99422、128×256 0.999975、256² 0.999964)。

**依据 —— 是"地板实证",不是"放宽"**:512² 的两个上游模块**已经贴在 BFP16 量化地板**上:
- **投影**:`to_q` 的 CPU BFP16 地板 **0.886 %**(库自带 `mm_bfp_tile_ref`),设备实测 **0.877 %** ⇒ 无投影缺陷(commit `8e5fdc7`);
- **attention**:S 的地板 **0.542 %**,attention floor rms **0.852 %** / max-absmax **1.926 %**(vs f32);设备实测 **1.09 % max-absmax**,**在地板之下** ⇒ 无 attention 缺陷(commit `fbe6b24`)。

⇒ 拿 f32 的 0.999 当阈值,等于**把量化地板当缺陷**;换成 BFP16 参考,判据才继续对**实现缺陷**敏感。

**纪律承诺(三条,写进交付口径)**
1. **新参考自身 100 % 门**:BFP16 精确仿真参考必须先自证(逐位 / corr 100 %,并与库的 `mm_bfp_tile_ref` 口径一致),否则不得拿来当判据。
2. **独立门保持 100 %**:逐字节(run-to-run)、audit 全 core、计时闭合、批语义等**不因换参考而放宽**。
3. **全程声明口径**:报告、`webui/evidence/PRODUCT-CHECKLIST.md`、UI 信息项都必须写明是"对 BFP16 参考"还是"对 host f32";f32 corr 只作信息项展示(**不判定**)。

**对设计的影响**:512²+ 的门脚本要**同时输出两个数**(f32 corr 作信息项、BFP16 参考作判据),报告注明口径;任何人不得把"换参考"读成"降标准"——门是从"对错量纲的严格"变成"对正确参考的严格"。128–256² 的既有数字**照旧有效**,不需要重跑。

**出处**:`8e5fdc7`(投影地板)、`fbe6b24`(attention 地板)、负责人 2026-10-20 裁决;落地文件:`webui/evidence/PRODUCT-CHECKLIST.md` §5 的【更正 2026-10-20】条、`README-WEBUI.md` 的口径注。

**状态**:定案(裁决)。

---

### F-24 · conv stall 的触发面:F = rr²·N²/K(强证据;根因未知)

**事实**:conv_gemm 的 **stall**(判据 = ERT state 8 且 **C BO 全零**,不是"慢")由一个三参数单式**分离**:

    F = rr² · N² / K        rr = row_repeats = (N/16)·(K/Kc)
    通过区 F ≤ 3.07 M ；stall 区 F ≥ 5.69 M ；窗口 ×1.85（21/21 点全分离）
    等价写法：rr·N/√K ≤ 1,752 通过 ；≥ 2,384 stall

F 只依赖 (K, N, Kc),**与 M 无关**。

**M 无关是这条卡的核心**:M=64(4 个 tile row、2 个 TaskGroup)即复现 stall(**零输出 + state 8 @ 7,028.63 ms**)⇒ 触发面在 **TaskGroup / tile-row 尺度**,不是 dispatch 长度。这同时解释了 T21-3B2 的"M 切块无效"(切 M 只改块数,不改每个 TaskGroup 的形状),以及 M=16 臂的编译死锁(aiecc 的 runtime-sequence 检查:pingpong 退化)。

**四条单标量律全部被否证(点名反例,不是推理)**

| 律 | 反例 |
|---|---|
| 字节体积(A fill 字节) | (4672,128,Kc=64) **通过**,而 A/fill 83.3 MiB、B 74,752 块与 stall 的 (2336,256) **完全相同** |
| 重复数 rr | (2368,224,Kc=64) rr=518 **stall** vs (4672,160,Kc=64) rr=730 **通过** |
| N 带 | (1184,448,Kc=32) N=448 **stall** vs (4640,512) N=512 **通过** |
| K 带 | (4672,192,Kc=64) **stall** vs (4640,256,Kc=928) **通过** |

**外推验证(out-of-sample,2026-10-20)**:两条预测**先落盘**(`predictions5.json`)再测,**2/2 双向命中** —— (4672,96) F=0.38 M → **PASS 24.28 ms**;(1184,224) F=11.37 M → **STALL 7016.66 ms**。加上这 2 点共 **23 点全分离**,窗口仍 **×1.85** 不变 ⇒ F 从"描述性拟合"升级为**可外推的设计筛查**(仍然不是机制;根因分档不变)。

**同几何复制臂 4/4 全 STALL(M=32 / M=64 / M'=1024 / rep_cap=584,F=38.27 M)** ⇒ **"缩 M"与"拆 fill"都不是修法** —— 这是实测,不是推理。

**分档(引用时必须保留)**
- **强证据**:stall 是 TaskGroup 尺度现象(M=32/64/1024、rep_cap=584 四臂同几何全 stall);四条单标量律全部否证;触发面 F **23/23 分离**,且 **out-of-sample 2/2 双向命中** ⇒ 可用于**外推筛查**。
- **仍未知**:到底是哪个**硬件计数器 / 固件握手**。没有 kernel/固件级 trace ⇒ **不得写成"已知根因"**,也不得据此说"已修好"。

**设计规则(新几何进门先做)**
1. **先用 F 打分**:F ≤ 3.07 M 安全区;F ≥ 5.69 M **必 stall,别建**;落在 3.07–5.69 M **必须实测**(边界只细过 (2368,192)→(2368,224))。
2. **N=128 半宽 = 16× 余量**:F ∝ N⁴,N 减半把 F 降到 1/16 ⇒ (2336,128) 的 F=2.39 M 在安全区 —— 既有绕法的量化解释。
3. **K 允许时选大 Kc**:F ∝ 1/Kc²,Kc 翻倍 = 4× 余量;(4640,*) 用 Kc=928 在任何 N 上都安全就是例证。注意 K=2336 的因数只允许 Kc=32 或 2336,**提高 chunk_cols 的 cap 没有用**。
4. **K=1184 族在 N ≳ 184 不安全(实测结论)**:(1184,448,Kc=32) 与 **(1184,224)** **双双实测 stall**;此前只有预测,现已验证。
5. **不要**指望 M 分块 / TaskGroup 变小能修(F 与 M 无关,实测也无效)。

**出处**:`_scratch/vae/stallprobe/README.md`(commit `a078ba9` 版)§0.3(触发面 + 窗口)、§0.2 与 §2(四条单标量律 + 14 个判定臂)、§4(21 个器件点按 F 排序:pass max 3.07 M / stall min 5.69 M、窗口 ×1.85、stall wall 7,008.76–7,028.63 ms)、§6(§6.1 分档表、§6.2 修复方向)、§3.3(编译墙 ∝ Kc=32)、§7(精确剩余量)。**该文件仍在被追加,行号会漂 ⇒ 一律按 § 号 + 关键词定位,不要引用行号。**

**沿革(本卡的自我记账)**:v1.7 首版只有 **21** 个点、K=1184 处写的是"F 预测 (1184,224) 也会 stall"、并明记两条 falsify 预测**未做**;v1.8 补上 **out-of-sample 2/2**(预测先落盘)与 **4/4 同几何复制臂**,并把 K=1184 从**预测**升级为**实测**(N ≳ 184 不安全)。旧措辞可从 `git log -p -- docs/KNOWN-PLATFORM-FACTS.md` 的 `c6013e2 → c425bfc` diff 逐字复原。

**状态**:**强证据(可外推的筛查面)+ 根因未知**。外推预测已做完并 **2/2 命中**(`README.md` §7 表第 3 行;预测先落盘在 `predictions5.json`)。**F 是筛查工具,不是已定案的机制** —— 拿它当"已修好"或"已知根因"都是误读。

**签名读法(2026-10-20,t22_sring)**:**`[6, 8]` 是**一对**,不是两个独立故障 —— **8 = TIMEOUT** 是超时本身,**6 = ABORT** 是它被拆除时的状态。所以 `[6,8]` 只说明"runtime sequence 没在 ~7 s 内走完",**不含任何定位信息**;拿它当"死因签名"会高估信息量,定位要来自 BD 账/时间轴。出处:`_scratch/attn_impl/sring/SRING-REPORT.md:83-85`(含 pyxrt 状态枚举逐字)。

**限定语(2026-10-20,见 F-25)**:F 是**本栈 / 本内核结构**下的触发面,**不是硅的极限** —— 换 kernel 结构(流式、不同 fifo 形态、不同 DMA 描述符形态)可能落在完全不同的曲面上。对外一律写"本架构下"。

---

### F-25 · 闭源基线 8.x s(追赶锚点;硬件利用率上限远高于本栈实测)

**事实(负责人 2026-10-20 情报,重定战役目标)**:闭源路线此前做到**纯扩散计算 ~8.x s**(大头开销在**会话创建**,这正是转开源路线的原因)。这证明 **NPU 硬件可以达到很高利用率**。因此本项目此前测得的"两堵墙 / 400 GFLOPS 不可达"一类结论,必须重定性为 **"本栈 / 本架构的墙,不是硅的墙"**。

**新目标(追赶锚点)**:器件扩散段向 **8.x s** 对齐。当前 128² 器件实测 **15.655 s**(FFN 9.580 + attention 6.075),差距 **1.96×**(≈2×)。

**口径提示(2026-10-20)**:下面的 **15.655 s 是 T13 的交付基线(实测)**;T2 线的**纸面凑分**另计(见 `_scratch/attn_impl/t2/T2-ARCH.md` §6),两者**不可混引** —— 前者实测、后者是工程外推。

**A1 终判(2026-10-20,已关闭)**:「A1 / streamed memtile fifo」也判负 —— per-BD **4.63 µs ≥ 1.8 µs** 触发**预注册证伪**;**fill 数砍 65 %(2,130 → 750),时间反 +12.4 %**,把"完成链摊薄"假设**反向证伪**;memtile 那一跳只加延迟(+0.66 ms ≈ pmt2 的 +0.5 ms)。⇒ **A1 对 128² 投影段彻底关闭**(此前记的"8.17 s 含 A1"随之作废)。剩的只有"更大 BD",而 **BD 长度 16 KB 上限 × 1.8–2.9 µs/BD = 5.5–8.7 GB/s 的硬顶**(见 F-32 的形状预算)。
**凑分表定格 9.27 s**(15.655 − qkv 1.9 − N3 4.0 − p10 0.44);**8.x s 只剩"换核形状 + 换参考(改 C 舍入链)"一条,新立项、未批**。出处:`_scratch/attn_impl/proj/PROJ-REPORT.md` §11;commit `09a1d2f`。

**时间构成的口径(2026-10-20,commit `f9a6c8c`)**:15.655 s 的分解是 FFN **9.580** + attention 桶 **6.075**,但**桶里 88 % 是投影(5.36 s),ip 只有 0.71 s** —— 引用时不要读成"attention 花了 6 s"。按已入账的 qkv(**−1.9,实测**)+ N3(**−4.0,估计**)推,落点约 **9.76 s**,离 8.x s 还差 **~1.5 s**,且这段缺口**只能来自投影段**(本线从未单独测过)。

**口径定稿(2026-10-20 负责人两次确认)**:闭源 **~8.x s 的分辨率 = 128²**;其他分辨率负责人原话"**不确定其他分辨率是什么情况**"。⇒ 在 **128² 同口径**下,**"2× 差距"成立** —— **15.655 s → 8.x s 是一个同分辨率的工程压缩目标**,不是结构性重写目标。**512² / 2048² 的目标仍是模型外推**,引用时必须标注"**闭源大分辨率口径未知**"。出处:`_scratch/block/T13.md:270-283` + 负责人 2026-10-20 确认。

> **历史注(append-only,两版解读已被裁定)**:本卡 v1.9 首版按"分辨率不明"并列了两种读法 —— "若闭源跑 512²,其 FLOPs 约 16×,8 s 对应 ~16 TFLOP/s(33 % 峰值),则差距是**结构性**的、不是 2×"。**该 512² 假设段裁定为不成立**(实际口径是 128²),原文保留于此仅作沿革;**现行结论是 128² 版("2× 成立")**。

**三条重定性(适用于所有"不可达"表述)**
1. **"不可达"必须写成"本架构下不可达"**:硬件峰值 49.152 TOPS,本栈大形状实测 2,133 GFLOP/s(4.35 %),而闭源在同类硅上做到过 8.x s ⇒ **上限在软件/架构,不在硅**。
2. **锚点 = 闭源 8.x s(128² 口径)**;当前 15.655 s。量级参照(不是目标):理想 @2,133 GFLOP/s = **3.839 s**、@49 TOPS = **0.167 s**(同文件 `:282-283`)。
3. **优先级:架构级突破口 > 微优化** —— **S 环、kernel/firmware trace、融合(kernel/DMA)、会话常驻**优先;几 % 级的微优化降级,不再作为主线。

**对设计的影响**:①评审任何"不可达/上限"结论,先问"这是硅的上限还是本栈的上限";②路线排序改按"能否改变架构"打分(会话常驻/融合/S 环 > 参数微调);③对外文档与 UI 禁止裸写"不可达",必须带"本架构下"。**注意这条不改变任何已测数字**,只改它们的**解释与天花板归属**。

**出处**:负责人 2026-10-20 口述(组织事实);8 x s 的对照与口径警告已有记录:`_scratch/block/T13.md:35`、`:270-283`(闭源 ~8 s、本方案 15.655 s、1.96×)。

**状态**:目标/锚点(负责人裁决;**分辨率 = 128² 已确认**,2× 为同口径工程压缩目标;512²/2048² 仍属外推)。**8.x s 没有本仓库的测量产物**(闭源、不可复现)⇒ 是**锚点**而非本栈实测事实;引用必须带"128² 同口径 / 大分辨率口径未知"的限定。

**S 环判决更正(2026-10-20,t22_sring,append-only)**:本卡把"S 环"列为架构级突破口时,依据的是 T2-PILLAR3 §8 的"S 环判死"。**该判决已撤回** —— 死因是 p3 构造里**两行的尺寸账 bug**(objectfifo 元素写成 **NK 块**,fill 只搬 **1 块**),不是"多块复用 ring 不被支持";`rg_f2`/`rg_f3` 实测**改正后 ring 可复用**。⇒ **A1/A2 的前置条件恢复**,memtile 常驻 KB 这条路**重新开放**(修好构造后重测再裁决)。T2-PILLAR3 §8 / §10.1 已各加更正注。**后续(同日)**:修好那两行后 **p3b(NK=2)跑通 —— ERT [4]、gate0 干净、0.458 ms**(commit `761ed8e`)⇒ **S 环从"突破口候选"变为"已验证可建"**,A1/A2 恢复为**可排期**。

**产品形状已上板(2026-10-20 16:08)**:`rg_n86` = **86 个 key 块、一次 dispatch**(8,256 keys,payload 同块平铺 86 次)→ **ERT [4] COMPLETED,2.095 ms**(离 7 s 看门狗 3,350×);O 非零 8,055/8,192、finite、absmax 128(≈93× 单块的 1.375,与"86 块都进了累加器"相容)。**没有模型门**(payload 同值),这是机制证据、不是数值门。⇒ F-25 里"2× 工程压缩"的 **dispatch 那一段,产品形状在器件上已成立**。出处:`_scratch/attn_impl/sring/SRING-REPORT.md` §10.1 + 本页 F-29/F-30。出处:`_scratch/attn_impl/sring/SRING-REPORT.md:68-79` + commit `761ed8e`。

---

### F-26 · compute tile 每核只有 **2 进 / 2 出** DMA 通道

**事实**:compute tile 每核 **2 个输入 / 2 个输出** DMA 通道。超配时 aiecc 逐字拒绝:

    aie.mlir:5:23: error: tile (0, 5) requires 4 input/1 output DMA channels,
                    but only 2 input/2 output available

对既有 **4 个交付镜像**反查:每核输入数**全部 ≤ 2**(ip 的 qk = QP+KB、pv = P+VB、sm = 1;FFN 的 GU = BG+B3)—— **不是设计者的偏好,是通道硬限**。

**通道预算族(与 F-05 / F-06 合看)**

| 层 | 通道数 | 卡片 |
|---|---|---|
| shim | **16 输入/设计** = 8 列 × 2 in(2 in / 2 out 每 tile) | F-06 / C-1 |
| memtile | **6 MM2S / 6 S2MM**(48 BD) | F-05 |
| **compute tile** | **2 in / 2 out** | 本卡 |

⚠ **G_GU 4→8 被拒是 memtile 的 6**,不是 compute tile 的 2 —— 两个口径别混。

**对设计的影响**
1. **独立 fifo 形态每块就吃满 2 个输入**(qk 要 KB_b + S_b,pv 要 P_b + VB_b)⇒ **NK = 1 封顶**;"一次 dispatch 多块"在**独立 fifo 形态**下不可建。
2. 想 NK ≥ 2 只有两条:**把块分到不同核**(= G 组;G=2 实测 1.25×),或**复用同一 fifo**(ring 形态,撞数据饥饿)。
3. **口径限定(2026-10-20)**:"B / NK≥2 **不可建**"只在**独立 fifo 形态**成立;**复用形态已实测可建** —— 修好元素尺寸两行后 **p3b(NK=2)ERT [4]、gate0 干净、0.458 ms**(commit `761ed8e`);按该 commit 的记账,2048² 的 dispatch 行由 **2647 s/步 回到 15 s/步**(报告待落地,引用时注明是 commit 记账)。"S 环判死"因**尺寸账 bug**撤回(见 F-25 的更正注与 F-19 的检查单)。

**出处**:`_scratch/attn_impl/t2/T2-ARCH.md:154-188`(§7:逐字拒绝 + 四条反查 + 结论);commit `f57187f`。**独立第二证**:`sring_pv.py --fifos 2`(两块各一条通道)编译失败;commit `8523223`(`_scratch/attn_impl/sring/SRING-REPORT.md` §10 卡稿)。

**状态**:定案(硬件通道数;逐字拒绝 + 交付镜像反查双证)。**"不可建"限于独立 fifo 形态。**

---

### F-27 · objectfifo 元素尺寸必须等于一次 fill 搬运的字节数

**事实**:shim→core objectfifo 的**声明元素尺寸大于一次 `fill` 实际搬运的字节数**时,core 侧 S2MM 的 BD **要等满一个元素才完成** ⇒ full-lock 每 N 次 fill 才释放一次,core 拿到 1 个元素后第二次 `acquire` 永远等不到数据。表现为 **ERT state 8(~7.0 s)、输出 BO 全零、core trace 上块 0 正常 / 块 1 MEMORY_STALL**。**与 ring、hop、锁、队列深度、runtime sequence 的 push/await 顺序全部无关。**

**最小复现**:`pv_z2`(单核,7.018 s)vs `pv_y2`(**同一 shim 事件表**,只改元素大小,0.513 ms);BD 账 `out/pv_z2.prj/input_with_addresses.mlir`(发送 3,072 elem / 接收 12,288 B)。

**出处**:`_scratch/attn_impl/sring/SRING-REPORT.md` §0-§1、§10(卡稿);commit `9783b8d`。

**对设计的影响**:①新算子先核对「fifo 元素 == 一次 fill 的字节数」;②多块循环要么把元素改成 **1 块**,要么**一次 fill 搬满整个元素**(后者要求目标缓冲装得下,见 F-28);③看到"第二次 acquire 卡死 + 块 1 MEMORY_STALL"先查这条,不要先怀疑 ring/锁。

**状态**:定案(器件 + 静态 BD 账 + 最小复现三重)。

---

### F-28 · core L1 共驻大对象上限 = 2 块 / 元素(是"放置失败",不是总字节)

**事实**:元素 = **3 块**(3 × 13,824 B)即**建不出来**,逐字 `buffer "pvA_cons_buff_1" needs 6144 bytes and this tile has no room left`;**2 块(27,648 B)可以**。失败形态**不是总字节不够,是 buffer 放不下**(与 F-18 同族)。报告自记:预测"3 块能过"被推翻,这是本轮第三次"我算的 L1 账比实际宽松"。

**出处**:`_scratch/attn_impl/sring/SRING-REPORT.md` §10(预测表 + 实测表:`rg_e3`/`rg_e4`/`rg_e16` 失败,`rg_f2` 元素 1 块通过);commit `bea502d`。

**对设计的影响**:①**"一次搬完 nk 块"不能在 core L1 里做**(≤2 块 / 27,648 B per element),只能走 **memtile 常驻**(A1)——这条把 A1 从"建议"变成**唯一路径**(就"共驻"形态而言);②看到 `needs N bytes and this tile has no room left` **先查放置**,不要只算总字节;③L1 账要留经验余量:本轮三次预测都被实际放置推翻。

**状态**:实测(逐字失败 + 2 块可过的对照)。

---

### F-29 · runtime sequence 长度是线性的,**不是墙**(nk=86 实测)

**事实**:指令流 `.bin` 按 **~328 B/块**线性增长(836 / 1,492 / 2,804 B @ nk=2/4/8)。**nk=86 的一次 dispatch 编译通过**:28.6 s、**.bin 28,552 B**、**348 事件(174 push / 174 wait)**、每通道未 await 高水位**恒 1**;外推 28.4 KB 与实测差 **0.4 %**。**指令流在 host/DDR 的 BO 里,不占 16 KB 的 core 程序区**。
器件侧:**nk=86 一次 dispatch → ERT [4] COMPLETED,2.095 ms**(见 F-25 的 86× 附注)。

**出处**:`_scratch/attn_impl/sring/SRING-REPORT.md` §10(预测先写 + 实测表)、§10.1;臂 `rg_n16` / `rg_n86`;commit `bea502d`。

**对设计的影响**:T2-PILLAR3 §5 的"**风险 #1:runtime sequence 长度上限未测**"**结案 —— 不是墙**。86 块的 dispatch 在序列侧无障碍,**障碍只剩 L1/放置(F-28)与数据流**。
**边界**:本卡只证"序列与累加在 nk=86 成立",**没有测吞吐**(86 × 27,648 B = 2.38 MB/dispatch,按 3.45 GB/s ≈ 690 µs;搬运仍受 GB/s 墙约束)。

**状态**:实测(编译 + 事件表 + 一次器件)。

---

### F-30 · 流式形态的常驻集与 nk 无关("246 上限"只限**共驻**)

**事实**:T2-PILLAR3 §10 的「keys/dispatch ≤ **246**」算法,前提是**把 NK 块同时放进 core L1**(每块 208·LK B ⇒ NK·LK ≤ 246)。而**流式**形态只留 **1 个槽(13,824 B)**,86 块依次灌进同一个槽、core 每块算完即释放 ⇒ **常驻集与 nk 无关**,246 这个上限对"流式 + 每块 await"的形态**不适用**。

**推论(同报告订正)**:T2-PILLAR3 §10.1 的"B 要真拿到 4×,必须把 KB 放进 memtile"需要加限定 —— 那句针对的是"**同时**持有 4 块以减少 dispatch";若一次 dispatch 已经**流**过 86 块,**B/支柱三的 dispatch 收益不需要 memtile,也不需要 A1**。

**边界(必须并读)**:**吞吐墙没被这条结果动过** —— 86 × 27,648 B = 2.38 MB/次 dispatch,按 3.45 GB/s ≈ 690 µs,搬运时间仍是 GB/s 墙那一项。A1(大对象、memtile 常驻)仍独立有价值,但它解决的是**对象流率**(3.45 vs FFN 的 21 GB/s),**不是**"能否一次 dispatch 走完 nk 块"。**两件事不要混。**

**出处**:`_scratch/attn_impl/sring/SRING-REPORT.md` §10.2(含边界段落);被订正处:`_scratch/attn_impl/t2/T2-PILLAR3.md` §10.1。

**状态**:定案(编译 + 一次器件 + 报告自我订正)。**只对"流式 + 每块 await"形态成立**;共驻形态仍受 F-28 约束。

---

### F-31 · 器件 softmax 的 exp **不是正确舍入**:`aie::exp2<bfloat16>` 与 `bf16(2^x)` 有 ~4 % 元素级偏差

**事实**:AIE2P 上 `aie::exp2<bfloat16>` 是**硬件指令**(原生路径);`exp2_bf16.h` 里那个 bf16 三次多项式**只走 AIE2 分支**,在 AIE2P 上根本不参与编译 —— 所以我们的 softmax 走的是**源码里查不到舍入行为**的硬件 vexp2。
探针(`t2_exp2_probe.cc`,10 行直调该指令;1 核、64 瓦片 × 1024 = **65,536 点**)实测:

| 区间 | n | exact % | p50 ulp | p90 ulp | max ulp |
|---|---|---|---|---|---|
| 全部 | 65,536 | 15.11 | 10 | 32 | 74 |
| x ∈ [−8, 8] | 26,070 | 3.49 | 8 | 11 | 14 |
| **x ∈ [−30, 0](softmax 实际区间)** | 25,518 | **1.51** | **10** | **18** | **26** |
| x ∈ [−200, −60] | 18,296 | 48.51 | 2 | 56 | 74 |
| x > 0 | 13,815 | 4.36 | 7 | 10 | 11 |

⇒ softmax 区间里只有 **1.51 %** 的输出与正确舍入的 `bf16(2^x)` 逐位相同,**中位偏差 10 ulp(≈3.9 % 相对)**,p90 **18**。

**三条排除(机制仍未定)**
1. **不是错位**:对 x **严格单调**(65,535 步无一处回退),按瓦片的 exact % 随 x 平滑变化(≈−8 处 0.78 %、≈0 处 4.5 %)⇒ 数据顺序正确、探针有效。
2. **不是单纯的舍入模式差**:truncation / RNE 差应 ≤ **1 ulp**;实测中位 **10 ulp**。
3. **不是常数比例**:锚点 x = −1 是 0.996×、x = 0 是 1.000×,而全样本中位 ratio **1.057** ⇒ 误差随 x 及其小数部分变化,**像 LUT / 分段近似**。下一步(0.2 单元、无器件):把 `(dev/ref − 1)` 按 x 的小数部分/整数部分分组,验证是否"每 1/16 或 1/32 一个节点 + 线性插值"。

**对判据与产品的影响**
1. **判据**:softmax 链上「**0.9999 / 8 ulp**」这一档**在 vexp2 表建出来之前不可达**(p90 30–35 与此吻合);在此之前 **C 判据是唯一可用的那一档**。
2. **产品**:**任何"softmax / attention 输出接近 f32"的表述都必须带本条限定** —— 交付门能过,是因为**门是 corr 型 + softmax 的归一化吸收了大部分共模**,不是因为逐元素接近。
3. 顺带第二条证据:该探针构建时撞 **F-06**(shim 活跃 BD ≤16),逐字 `Too many simultaneously active buffer descriptors on tile (0,0), which supports up to 16`。

**出处**:`_scratch/attn_impl/t2/T2-PILLAR3.md` §18(§18.1 探针 / §18.2 表 / §18.3 三条排除 / §18.4 两条含义);commit `ce0a0b9`;探针 `t2_exp2_probe.cc` / `t2_exp2_run.py`(py310)。

**收尾补充(2026-10-20,§18.5–18.7;commit `88279da`)—— "建表"这条也判死**

| 检查 | 结果 | 判 |
|---|---|---|
| 误差是否是 **f(x 的小数部分)** 的函数 | 16 个 f 分箱的中位 ratio 全在 **1.02–1.10**(f≈0 也有 **+4.7 %**) | **不是分段 LUT**(若是 1/16 或 1/32 节点的 LUT,f→误差应有清晰锯齿、且在节点处为 0) |
| 误差是否是 **k(x 的整数部分)** 的函数 | 各 k 的 p50 从 1.082(k=−20)缓变到 1.060(k=−13) | 有缓慢趋势,但**无分段结构** |
| 能否用 **log2(dev) = a·x + b** 描述 | 拟合 a=0.99753、b=0.05452(x=0 增益 1.0385);**残差 p50 = 7.3 ulp、p90 = 32 ulp** | **不成立**:残差与信号同量级 |
| `bf16(2^(a·x+b))` vs 器件 | exact **6.60 %**、p50 4 ulp(优于直接用 2^x 的 2.28 % / 10 ulp,但**远不到逐位**) | 只能当近似模型,**不能当参考** |

⇒ **定性(终局)**:`aie::exp2<bfloat16>` 在 AIE2P 上是**精度约 4–5 %(≈4 bit 尾数级)的近似函数**,且**系统性偏大 4–7 %**;既不是正确舍入,也不是"正确舍入 + 舍入模式差",**也没有可用的一维 LUT / 线性结构**。

**判据终局(§18.6)**
1. **含 softmax 的链:C 判据长期有效**(gate0 + 跨镜像逐字节 + 剂量单调性),本轮实测全绿。
2. **(b) 的 0.9999 / 8 ulp 档只适用于不含 softmax 的链**(pv2b:corr 0.99997 / p90 4)。
3. **"逐位"宣称收窄**:只有"不含 softmax"的那一半可逐位;含 softmax 的一半**本质不逐位**(除非改用器件实测的 P 表)。

**产品读法(§18.7)**:attention 输出的"相对 f32 误差 0.107 %@T3d 看起来很小"**并不代表 exp 本身准** —— 是 softmax 的归一化吸收了大部分共模。

**状态**:定案(器件 65,536 点 + 严格单调 + 三重排除;且**建表/线性结构两条已被 §18.5 否证**)。**仍未知的是指令内部结构**(逆向不在 0.2 单元内、且未必可逆),但**对判据与产品口径的结论已经终局**,不再等待机制。

---

### F-32 · `aie.dma_bd` 的**每一维** size ≤ 1023

**事实**:BD 的**单维**尺寸上限是 **1023** —— 2-D walk `[4, 9216]` 被拒,逐字 `Size may not exceed 1023`;**3-D 绕法**(`4 × 576 × 16`,每维 ≤1023、**字节序不变**)编译通过(10.7 s)。⇒ 这是**单维**限制(不是总长度):超了就把那一维**拆成几维**,字节序可以保持。

**对设计的影响**
1. **FFN 的 Kc = 960 是这个上限的产物**:Kc 要同时满足"整除 K=3840"、"是 64 的倍数"、"作为 BD 的一维 ≤1023" —— 1,000 以下唯一解就是 **960**(1,920 > 1023 被排除)。所以 960 不是调参结果,而是**硬限的交集**(再与 F-17 的 L1 账一起把它钉死)。
2. 任何"一行 walk 走完 K"的设计,在 **K > 1023** 时必须改 3-D;改法只要保住字节序,就是等价重排。
3. 与 **F-07** 的「≤3 维 / ≤16,383 words」合看:维数 ≤3、每维 ≤1023、总 words ≤16,383,三者共同构成 BD 的**形状预算**,任何一条都可能先撞。

**出处**:`_scratch/attn_impl/proj/PROJ-REPORT.md` §11(逐字拒绝 + 3-D 绕法编译通过 + 生成物核对);commit `09a1d2f`。旁证:F-07 引的 `HANDOFF.md` §5(≤3 维、≤16,383 words、Kc 必须整除 K)。

**状态**:定案(编译期逐字拒绝 + 3-D 绕法通过,对照)。

---

## 3. 矛盾表(并列记录,不裁决)

> 规则:两处出处互相冲突 → 原样并列 + 各自证据 + 写清"对设计的影响",**由负责人裁决**。

### C-1 · shim 的 DMA 通道数:16 入/16 出,还是 2 入/2 出?—— **已裁决 2026-10-20**

| | 说法 | 出处 |
|---|---|---|
| A | "shim DMA **16 MM2S / 16 S2MM**,16 BD" | `_scratch/block/T5.md:119`(同表里 compute tile 是 "2 in / 2 out,16 BD") |
| B | "A ShimNOCTile has **4 DMA channels (2 in / 2 out) and that is the DEVICE TOTAL**" | `HANDOFF.md:171-174` |
| B′ | "shim **16 BD / 2-in-2-out**" | `docs/HANDOFF-2026-10-19.md:152` |

**裁决(2026-10-20,负责人)**:**设计口径取「每 shim tile 2 MM2S / 2 S2MM,每通道 16 BD」**(`HANDOFF.md:171-174`,与 `_scratch/util/UTIL_T1.md:65` 一致)。`_scratch/block/T5.md:119` 的 "16 MM2S / 16 S2MM" 记的是 **switchbox 行级连接计数**——那张表由 `_scratch/attn_impl/t5_devres2.py` 打印,取数用的是 `get_num_source_switchbox_connections` / `get_num_dest_switchbox_connections`;而工具链的 `shim_dma_channels_in/out`(`<工作区根>/_pylibs/mlir_aie/python/aie/iron/device/device.py:97-116`,按 shim mux 连接求和)才是"DMA 通道"的定义。**两者不矛盾,口径不同。**

**对设计的影响**:按 2-in/2-out 设计 fill 数(交付的 `[x|gamma]` packed element 就是这么被逼出来的,`README.md:124`);BD 按 **16/通道**记账;引用 T5 的 "16" 必须写成"switchbox 连接数",**不得当通道数用**。
**换算(堵误读)**:*每 shim tile 2 in / 2 out* × **8 列** = **16 条输入通道/设计**(aiecc 的 `16/16 input channels used`,已由 t22_qkv 实测,见 F-06 的"通道账")。要更多 worker 只能走多播/memtile。

### C-2 · data 参数上限 7 的失败形态:ABORT 还是 0xC0000005?—— **已裁决 2026-10-20**

| | 说法 | 出处 |
|---|---|---|
| A | "10/9/8 全部 `ERT_CMD_STATE_ABORT`,7 个完成" | `HANDOFF.md:157-158` |
| B | "data 参数 ≤ 7;**多传一个 BO 即 0xC0000005**" | `_scratch/vae/T21-3B1-U1-PAPER.md:25` |

**裁决(2026-10-20,负责人)**:**data 参数 ≤ 7 是硬限**;失败形态**随路径不同**——声明参数数超 → `ERT_CMD_STATE_ABORT`(A 路径),绑定 BO 超 → `0xC0000005`(B 路径)。两者**都致命,都不可捕获绕过**(后者是 native 崩溃,Python 的 try/except 接不住)。

**对设计的影响**:设计侧只做**预防**(concat / packed element / memref 偏移),不要设计"捕获后重试"或"多绑一个 BO 试试";一旦越界,该轮进程直接作废,按 F-09 用产物判据判定。

---

### C-3 · BO 可见性:输入侧窗口(v1)vs 输出侧缺 sync(v2)—— **已裁决 2026-10-20**

| | 说法 | 出处 |
|---|---|---|
| A | "BO 可见性**窗口**:host 填 BO 与 kernel 读竞态;纪律 **fill(i+1)→wait(i)→launch(i+1)**" | 2026-10-20 任务书「背景定案(勿重查)」;v1 报告(commit `6e03bae` 的叙述);`_scratch/attn_impl/T22-PROTOTYPE.md:106-116` |
| B | "真正的缺陷是**输出侧少了一次 `sync(FROM_DEVICE)`**;填侧不需要 sleep/重叠/双缓冲" | `_scratch/attn_impl/T22-PLATFORM-FACTS.md:6-15,19-27`(同源更正版,含 detC/detD 逐字节证据) |

**裁决(2026-10-20,负责人)**:**采用 B(v2:读前 `sync(FROM_DEVICE)`)**。依据:①源头 commit `f1f656c`(t22_large 的 run-to-run 根因提交,题注逐字 "the qk/pv output BOs were read without sync(FROM_DEVICE) after run.wait() … Platform fact card corrected: output-side visibility, not the input fill window");②detC/detD 读前 sync 后两次**逐字节相同**(max|d| 0),而延迟 3 / 10 ms 与重叠 fill 都只是改概率(0.0087 / 0.0093 / 0.463)。**v1 归因保留为历史,不再作为设计依据。**

**对设计的影响**:按 A 做(双缓冲/延迟/重叠)会**看不出错**却也不保证正确;按 B 做(读前 sync)detC/detD 逐字节相同。**F-02 的纪律以 v2 为准**;今后若有人引用 v1 的 "fill(i+1)→wait(i)→launch(i+1)",必须同时引用本条裁决。

---

### 3.1 时点读数(不是事实卡,开工必须当场自查)

| 项 | 读数 | 出处/时间 |
|---|---|---|
| 磁盘余量 | "~0.9 GB"(紧张) | `docs/HANDOFF-2026-10-19.md:118`,2026-10-19 |
| 磁盘余量 | "~25 GB" | 2026-10-20 任务书背景块 |
| 本次(2026-10-20)复核 | **无法实测**:`fsutil volume diskfree` → Error 5 Access denied;`Get-CimInstance Win32_LogicalDisk` → 拒绝访问(沙箱) | 本卡执行时的实测 |

**处置**:磁盘是两个时点的读数,不做矛盾裁决(清理战役 `docs/CLEANUP-EXECUTION.md` 动过 17.8 GB);开工先自查,大件构建前先清 `.prj`。

---

## 4. 传闻待证(不得作为设计依据)

| ID | 说法 | 现状 | 出处/为什么不可信 | 一小时内能做完的判据 |
|---|---|---|---|---|
| U-1 | "BO 可见性窗口宽度与 dispatch 时长**同量级**" | 推论,非测量 | `_scratch/attn_impl/T22-PLATFORM-FACTS.md:9-10` 是解释句;实测只有"0 ms vs 3 ms"两点 | 延迟扫描 0 / 0.5 / 1 / 2 / 3 / 5 ms × 3 次,看 rel 差何时进入 1e-3 |
| U-2 | state-8 的物理机制 = 交付镜像 gate/up 核带 **ACQ CLOBBER**,锁未释放 | **假设**,作者明写"本轮没有做去掉 clobber 重建镜像的实验" | `webui/evidence/T20-STATE8.md:71-75` | 重建一版无 clobber 的 `t11_nh5`,背靠背 ×4 看是否还 state 8 |
| U-3 | 首 dispatch `0xC0000005` 是 "~50 % 冷启动彩票" | 概率数字无统计出处(实测只有"命中 3 次""3 次") | `docs/HANDOFF-2026-10-19.md:115`(50 %)、`_scratch/block/T10.md:300`(3 次)、`_scratch/vae/T22-VAE-REPORT.md:217`(3 次) | 20 次全新进程首 dispatch,记命中率(±1 二项区间) |
| U-4 | "器件为全军共享,测量/门类工作错峰" | 组织约定,**不是测量** | 任务书背景块;F-14 只证"第二个使用者会报 0xc01e0009" | 不需要:按约定执行,遇 `0xc01e0009` 即视为争用 |
| U-5 | "`C4_4` 放不下是 bank 3 被广播缓冲占满" | 读法/假设(报告自己写"这是分配器的 bank 打包问题") | `_scratch/perfc/T22-UTIL3-U1P-DEATH-TABLE.md:63-65` | 打印该核的 bank 分配表(编译期,零器件)对照 16 KB/bank |

---

## 5. 与其它文档的互链

**本文的出边**(引用它们作为出处,细节读原文)
- 交接与总览:`HANDOFF.md`(§5 HARD CONSTRAINTS)、`docs/HANDOFF-2026-10-19.md`(§2 状态-8 定律 / §5 环境手册 / §8 墙表)、`README.md`(格式定义 + 硬限制)、`README-WEBUI.md`(产品口径)。
- 定案报告:`webui/evidence/T20-STATE8.md`(F-01/F-04)、`_scratch/attn_impl/T22-PLATFORM-FACTS.md`(F-02)、`_scratch/perfc/T22-UTIL-REPORT.md`(F-03/F-15)、`_scratch/perfc/T22-UTIL2-MEM-REPORT.md`(F-05)、`_scratch/perfc/T22-UTIL3-U1P-DEATH-TABLE.md`(F-17/F-18/F-19)、`_scratch/vae/T21-3B2VF-REPORT.md`(F-06/F-12)、`_scratch/block/STEP5.md`(F-09)、`_scratch/vae/T22-VAE-REPORT.md`(F-11/F-14)、`_scratch/block/T12.md`(F-17 的锚点)。
- 历史与台账:`KNOWN-ISSUES.md`(逐条 refusal 归档)、`docs/TIMING-RECONCILIATION.md`、`docs/WORKLOG.md`、`docs/CLEANUP-EXECUTION.md`。

**指向本文的入边**(在这些文档里加了引用行,原文不动)
- `docs/HANDOFF-2026-10-19.md` §3 索引 — "平台事实卡总表"。
- `HANDOFF.md` §5 开头 — 一行注记。
- `README.md` "Hard limits found on the way" 段末 — 一行注记。

**数据卫生提醒(原提醒,2026-10-20 已修复)**:`_scratch/perfc/T22-UTIL3-U1P-DEATH-TABLE.md` 曾是混合编码(头部 UTF-8、§6 段为 GBK),直接 read 会报 invalid UTF-8,需分段解码(前 3,872 B 按 UTF-8、其后按 CP936);**现已重存为 strict UTF-8**(内容逐字节等价,commit `0f37cc9`)。历史提醒保留:新文件请勿再写混合编码。

---

## 6. 本文自身的门(新增/修改一张卡时必须过)

| 门 | 判据 | 怎么做 |
|---|---|---|
| G1 出处可查 | 每条事实至少 1 个 `报告名:行号` / commit / 实测数据;行号指向的那几行**确实包含**该事实 | 逐条点开原文核对(v1 做过一次全量脚本核对,见 §7) |
| G2 不改史 | 原有文档只**追加**引用行,不删改既有句子 | `git diff` 只允许出现新增行 |
| G3 矛盾不裁决 | 冲突进 §3 表,两列都有出处 | 有冲突就建行,不写"正确说法是…" |
| G4 传闻分区 | 无出处者只能在 §4,且必须带判据 | §1 索引表里不得出现 §4 的条目 |

**维护流程**:新事实先写进对应报告(带逐字证据)→ 过 G1 → 才允许进本文;**每次改本文必须在同一次提交里更新 §1 索引表**。

---

## 7. 版本

| 版本 | 日期 | 变更 |
|---|---|---|
| v1 | 2026-10-20 | 建卡 F-01…F-19(含 F-17 m/Kc 轴闭合、F-18 分配器 bank 打包、F-19 设计前 spike 方法卡)、矛盾 C-1/C-2、传闻 U-1…U-5。锚定 commit `167aff1`。F-02 初版即按同源更正写为 v2(读前 sync),v1 归因保留为历史。 |
| v1.1 | 2026-10-20 | 裁决落地:C-3 采用 v2(依据 commit `f1f656c`);C-1 取"每 shim tile 2 MM2S/2 S2MM、每通道 16 BD";C-2 取"≤7 硬限、失败形态随路径";新增 **F-14.1 器件互斥锁协议** 与 **F-20 共享仓库禁 amend**;把 `_scratch/perfc/T22-UTIL3-U1P-DEATH-TABLE.md` 重存为 strict UTF-8(内容不变,commit `0f37cc9`)。 |
| v1.2 | 2026-10-20 | 新增 **F-21**(标量 f32 tie-rounding,含门禁含义);F-17/F-18 加 **kgu=64 口径限定**(交付是 kgu=8);F-14.1 补 pid 纪律(误写 helper pid 致违规接管);F-20 记违例计数 2 起;死亡表顶部加口径注(行号 +2 已重锚)。 |
| v1.3 | 2026-10-20 | 按 t22_normcore §10(commit `c059150`)定稿口径:F-17 改为 **kgu 无关**(补半块/A-plane 边界),F-18 的 W6/C4_4 标为 **kgu=64 实验镜像专属**,新增 **F-18.1**(R4′ 已编译 18 core/121,259 B;计划更正 −1.7 s 非 −1.9 s);新增 **F-22**(finish() 语义陷阱)并在 F-02 加指针。 |
| v1.4 | 2026-10-20 | 新增 **F-23**(换参考裁决:512²+ 对 BFP16 精确仿真参考,f32 corr 降为信息项;三条纪律承诺),并落地到 `PRODUCT-CHECKLIST.md` §5 与 `README-WEBUI.md` 口径注。 |
| v1.5 | 2026-10-20 | F-06/C-1 补"2 in/列 × 8 列 = 16 通道/设计"的换算(t22_qkv 实测,aiecc 逐字);F-16 补沙箱 `mkdir 0o700` 与 `xclbinutil` PATH 两个坑;F-02 补 depth-1 fifo 的 PIPE=1 死锁陷阱。 |
| v1.6 | 2026-10-20 | **F-21 更正**:"端到端通常仍逐字节"→"逐字节是概率性的"(144×128 实测 −0.059 dB / max\|d\| 0.138256);门禁含义②改为"端到端逐字节不可作为 tie-敏感算子的门保证,须改幅值界或真逐位实现"。 |
| v1.7 | 2026-10-20 | 新增 **F-24**(conv stall 触发面 F = rr²N²/K:21/21 分离、M 无关、四条单标量律否证、分档"强证据/根因未知"、五条设计规则);F-12 加量化收口指针;F-17 补"编译墙 ∝ Kc=32(23×)"同族代价。 |
| v1.9 | 2026-10-20 | 新增 **F-25**(闭源 8.x s 锚点:硬件上限远高于本栈;"不可达"改口径为"本架构下";架构级突破口优先);F-17/F-24 加限定语(结论限本架构 / F 不是硅的极限)。 |
| v1.18 | 2026-10-21 | F-19 加**两条工装纪律**(bfp16 块操作数必须走 `vlda.pop.576`:朴素指针 = 72 条 `lda.s8`、157 vs 1.68 周期/MMA = 93×;MMA 速率探针两道门:反汇编验 vmac 条数/无 `nopv` + 装载路径同源);F-22 加 **`finish()` 只 await `wait=True`** 的机制版本(drain `wait=False` ⇒ 假 COMPLETED + 全零,修后 `dma_await_task` 0→510)。 |
| v1.17 | 2026-10-21 | F-19 加**两条平台陷阱**:①`acquire(n)` 契约="补足到 num_elem"(两次 acquire(1) 返回同一元素 ⇒ 一对象两半 + 常量偏移 view);②**设备循环体首个 kernel 调用只在首轮生效**(双 zero 只第二个有效;换序/换符号/去 reclaim 都无效,加牺牲调用可解)—— 并对交付镜像 `fs_ffn_j5` 标了"建议 owner 复验"的风险。 |
| v1.16 | 2026-10-20 | 新增 **F-32**(BD 每维 ≤1023 ⇒ Kc=960 是硬限交集;与 F-07 的 ≤3 维/≤16,383 words 合成形状预算);F-25 记 **A1 终判关闭**(streamed-fifo per-BD 4.63 µs、fill −65 % 时间 +12.4 % ⇒ 完成链摊薄反向证伪),**凑分表定格 9.27 s**,8.x s 只剩换核形状+换参考(未批)。 |
| v1.15 | 2026-10-20 | **F-31 收尾**:建表/线性结构两条被否证(§18.5,f≈0 仍 +4.7 %、log2 拟合残差 7.3/32 ulp);定性为 **~4–5 % 粗近似 + 系统性偏大 4–7 %**;判据终局(含 softmax 用 C 判据、(b) 只适用无 softmax 链、"逐位"收窄)。 |
| v1.14 | 2026-10-20 | 新增 **F-31**(器件 softmax 的 `aie::exp2<bfloat16>` 与正确舍入差 1.51 % 逐位 / 中位 10 ulp / p90 18;三条排除;"接近 f32"表述的强制限定);F-21 加同族指针。 |
| v1.13 | 2026-10-20 | F-19 加**交付证据优先级**检查单(源码 sweeps=40 vs 交付 sloop=8:产出/消费恒等式 + 32 块结构常数 + 23.6 MB/dispatch 三重对账);F-21 加 **swiglu LUT 口径注**(32 段 × 0.5 表、vtanh vs 表路径的 6.25e-02 口径差)。 |
| v1.12 | 2026-10-20 | 新增 **F-27**(objectfifo 元素尺寸)、**F-28**(core L1 共驻 ≤2 块/放置失败)、**F-29**(runtime sequence 线性非墙,nk=86)、**F-30**(流式常驻集与 nk 无关,246 只限共驻);F-25 加**86× 已上板**附注(ERT [4]、2.095 ms);F-19 检查单升格指针;F-26 补第二证(`sring_pv.py --fifos 2`)。 |
| v1.11 | 2026-10-20 | 新增 **F-26**(compute tile 2 in/2 out ⇒ 独立 fifo 形态 NK=1 封顶;通道预算族 shim16/memtile6/core2);F-19 加 objectfifo 元素尺寸检查单;F-24 加 `[6,8]` 一对的签名读法;F-25 加 **S 环判决撤回**(p3 两行尺寸账 bug,A1/A2 前置恢复)。 |
| v1.10 | 2026-10-20 | **F-25 口径定稿**:闭源 8.x s 的分辨率 = **128²**(负责人确认)⇒ 2× 差距在同口径下**成立**,属"同分辨率工程压缩目标";512²/2048² 仍为外推(标注"闭源大分辨率口径未知");512² 假设段降为**历史注**(保留原文,裁定为不成立)。 |
| v1.8 | 2026-10-20 | F-24 升级:out-of-sample **2/2 双向命中**((4672,96) PASS 24.28 ms / (1184,224) STALL 7016.66 ms,预测先落盘 predictions5.json)⇒ 由"描述性拟合"升为**可外推设计筛查**(根因未知分档保留);同几何复制臂 **4/4 全 STALL**(M=32/64/1024、rep_cap=584)⇒ "缩 M/拆 fill 非修法"入卡;K=1184 族 **N ≳ 184 不安全**升级为实测。F-12 追加 §9 订正文;F-17 补 5.5 min/镜像排程。 |


---