# T22 GLUE — 扩散 loop 宿主胶水的细分计时、两条宿主快路径与止损线账

> 2026-10-21 09:0x–09:4x +08:00 · 128² / 8 步 / 30 层 / batch 1 / seed 7 · 产品 argv（qkv+p10 默认开）
> **状态：NOT SHIPPABLE。** 本文件交付的是①loop 内细分计时（代码 + 表定义）、②U1 快路径的产品透传与四件套执行方案、③已有档案的间接细分账与预测。
> **本轮首个器件窗口由父会话代跑**：`t22g_L`（128² 8 步，qkv+p10+fast_codec，U1 未开）WALL 30.993 s，账本 residual −9.556 s 已定位并修正（§10）。
> 子代理自身沙箱禁止创建进程（`spawn EPERM`），因此代码/账本/解读由本会话给，跑测量由有进程权限的会话执行。

## 0. 一句话

19 桶只到 loop 粒度：128² 的 loop 28.06 s 里 **器件 wait 13.4–15.7 s，其余 12.4–14.7 s 全记在"胶水"**；
用现有唯一一份 19 相位 + 1440 dispatch 时间线的档案（t22a1）做间接分解，**其中 4.48 s（15.7 % 的 wall）连 pmark 都没落在任何桶里**。
本轮把"必须能闭合到 loop wall"的**独占账本**写进驱动（`T22_GLUE_PROF=1` / `--prof`），并把 U1 快路径做成 bridge 可透传的显式开关；
**"胶水砍半"在只动宿主代码的前提下不可达**——两个最大项是结构性的（staging 的 3.7 s 需要器件 BO 常驻；refiner 的 2.6 s 需要换参考），本文件给出可复算的账。

## 1. 口径与环境

* 仓库 `<checkout>`；py310=器件，py312=服务/工具。
* 本文件引用的"现有档案"：
  * `_scratch/t22_e2e_measure/SUMMARY.md` + `webui/jobs/20261021-081535-a3db76/job.json`（128² 稳态基线，**latents `9f86aee91de3dad0…`**）
  * `webui/jobs/20261021-084429-192522/job.json`（128²，同 latents，VAE=cpu 的旁路作业）
  * `_scratch/perfc/out/t22a1_events.json`（2026-10-20 **U1 臂**：19 相位 pmark + 每 dispatch 时间线；**无 qkv/p10**，是唯一一份全相位档案）
  * `_scratch/perfc/T22-UTIL-REPORT.md`（U1 的 −1.68 s 同窗口 A/B、GIL/并发两条负结果）
* 任务书口径：`loop_s = 28.056 s`，`device = 15.655 s` → 胶水 `12.401 s`。下文的分解一律**同时给**这个口径与本机档案口径，不混用。

> **口径陷阱（必须写进任何后续 A/B）**：runner 的 `loop_s` 是 **span**（startup_done → 最后一条 step 行），
> 而驱动的 `sum(steps_rec.seconds)` 是 **step wall 之和**；两者差 `loop_overlap_s` = **1.56–1.59 s**（step 0 的计时起点早于 startup_done，
> 见 runner.py:1188-1203）。所以：**细分表闭合到 step wall 之和（29.65 s），不是 loop_s（28.06 s）**；换算时减去 1.59 s。

## 2. 预测先行（写在新表开跑之前，供下一窗口判卷）

对当前产品（qkv+p10，128²，8 步）跑 `T22_GLUE_PROF=1` 的预测：

| 桶 | 预测 s | 依据 |
|---|---:|---|
| `host.wait`（器件 launch→wait） | **13.1–15.7** | 08:44 作业 `device_s`=13.115；08:15 作业=13.396；任务书=15.655 |
| `host.fill`（bo.write+sync） | **2.7–3.7** | 08:44 `staging_s`=2.691；08:15=3.745；t22a1 pmark=2.250（无 qkv+p10 时更少） |
| `host.read` + `host.osync` | 0.25–0.35 | t22a1 = 0.262 + 0.037 |
| `att.qkv.img`+`att.o.img`（X/X2 打包） | 0.55–0.75 | t22a1 `att.ximg`=0.614（两次/层） |
| `att.unpack3`（Q/K/V 三片） | 0.55–0.70 | t22a1 = 0.634 |
| `att.unpack_o`（ip 输出，**不含 dispatch**） | 0.20–0.30 | t22a1 的 1.028 里 0.773 是 ip dispatch（见 §3.2 脚注） |
| `att.rms_rope` | 0.32–0.40 | t22a1 = 0.356 |
| `att.packkb`+`att.packvb`+`att.packq` | 0.55–0.70 | t22a1 = 0.631 |
| `att.unpack_a` | 0.20–0.28 | t22a1 = 0.245 |
| `att.other`（Qm/o_flat/a_flat 分配等） | **0.2–0.8（未知）** | 新桶 |
| `pre.refiner`（4 个 refiner 块） | **2.2–2.6** | t22a1 `T3.full_block_host` 2.566/32 次 |
| `pre.temb/xemb/capemb` | 0.03 | t22a1 = 0.029 |
| `blk.norm1`+`blk.norm2`+`blk.sbg`+`blk.ffnpost` | 0.7–1.0 | t22a1 = 0.146+0.168+0.006+0.570 |
| **`blk.other`（每层未归属）** | **3.5–5.0（未知，本轮要回答的就是它）** | t22a1 反推：wall − 已知 = 4.48 s |
| `step.ckpt`（np.save + json） | 0.25–0.45 | 每步序列化 `calls`（8 步累积，step7 最大） |
| `step.lat` | 0.03–0.08 | 8 × (96×1280 f32) |
| `step.other`（每步尾 + print） | 0.4–1.0 | 首层 [tim] 打印、absmax 汇总等 |

**判卷规则**：`[t22g] residual`（step-wall − Σ桶）必须 ≤ **±0.05 s**（账本按构造闭合，残差只可能来自 gmark 自身的开销量级）。
若 `blk.other` ≥ 3 s，则"胶水砍半"有救（那是纯 Python/分配开销，可砍）；若 `blk.other` < 1.5 s 而 `host.fill`+`pre.refiner` ≥ 5 s，则宿主侧已到结构地板（§5）。

## 3. 现状：已有档案的间接细分（不是新测，全部可复算）

### 3.1 128² 产品作业（qkv+p10）的 5 段账

| 作业 | loop_s(span) | step 之和 | device_s | staging_s | 非器件非 staging |
|---|---:|---:|---:|---:|---:|
| 20261021-081535（SUMMARY 稳态基线） | 28.056 | 29.648 | 13.396 | 3.745 | **12.51** |
| 20261021-084429（同 latents，快态） | 25.034 | 26.599 | 13.115 | 2.691 | **10.79** |

读法：`step 之和 − device_s − staging_s` 就是**全部宿主 CPU 胶水 + 步边界**；它 10.8–12.5 s，与任务书的 12.4 s 同量级。

### 3.2 19 相位档案（t22a1，U1 臂）的独占分解 — 唯一一份能落到函数的账

| 段 | s | %step-wall(28.55) | 来源 |
|---|---:|---:|---|
| 器件 wait | 15.186 | 53.2 % | `host.wait`（1440 dispatch） |
| staging fill | 2.250 | 7.9 % | `host.fill` |
| drain read + BO sync | 0.299 | 1.0 % | `host.read` 0.262 + `host.osync` 0.037 |
| 层内 pack/unpack（**已剥离 ip dispatch**） | 2.735 | 9.6 % | unpack3 .634 / unpack_o .255* / ximg .614 / rms_rope .356 / packvb .243 / packkb .235 / unpack_a .245 / packq .153 |
| norm + SBG + modulation | 0.395 | 1.4 % | post.norm1 .146 / SBG(=.174−.168 的 norm2 重叠) .006 / loop.sbg .057 / loop.mod .186 |
| ffnpost 残差 add | 0.570 | 2.0 % | post.ffnpost 11.415 − ffn 的 dev 9.463 − fill 1.308 − read .074 |
| step 前导（refiner 2.566） | 2.633 | 9.2 % | phases: full_block_host 2.566 + t_emb .012 + x_emb .013 + cap .004 + final .038 |
| **未归属（任何 pmark 之外）** | **4.481** | **15.7 %** | wall − 上列 |
| `sum` | 28.549 | 100 % | = `sum(steps_rec.seconds)` |

\* **两处必须知道的相位重叠（旧档案的坑，新账本已经拆开）**：
`att.unpack_o` 的 pmark 把整次 ip dispatch 算进去了（t0 在 packq 之前），减掉 ip 的 dev .693 + fill .047 + read .033 后真正的 unpack 只有 **0.255 s**；
`post.sbg` 与 `post.norm2` 共用同一个 t0，SBG 本身只有 0.006 s。
**所以"unpack_o 是单项最大宿主函数"是错的——最大的是 unpack3（0.634 s）。** 这两条正是把旧 19 相位表当结论用会自欺的地方。

**这就是本轮最重要的现状结论**：不是"12.4 s 都在某几个已知函数里"，而是**有 4.5 s（15.7 %）连相位都没进**。
它们只可能落在：每层 3 次 `np.abs().max()`、每层 1 行 `print(flush=True)`、Qm/o_flat/a_flat 的分配、
step 的 `np.save`+`json.dumps`、以及 dispatch 之外的解释器开销。**新账本（§4）就是为这 4 s 造的。**

### 3.3 已证伪/已闭合的方向（不要重测）

| 方向 | 结论 | 证据 |
|---|---|---|
| 线程预填载荷（U2/T22_PFILL） | **0**：fill 2.28→1.06 但 wall −0.08，间隙 +1.4 | `T22-UTIL-REPORT.md` §6.1（GIL：纯 numpy 0.043 s vs 并发写 12.36 s） |
| 多 xclbin 并发 / 双 handle | **0**：44.3→44.0 ms/对（x1.007） | 同上 §6.2；同 xclbin 第二 handle = 假完成 |
| FFN 形状 m→32/96（去 6× 重复读） | 锁死；且 m/Kc 轴已闭合 | `T22-UTIL3-M96-FEASIBILITY.md`、`T22-UTIL3-U1P-DEATH-TABLE.md` |
| refiner 4 块搬 NPU | 拒绝：f32→bf16 换参考 | `T22-UTIL-REPORT.md` §4 第 3 项 |

## 4. 本轮代码：独占账本 + 两条宿主快路径（全部落在允许的两个文件）

### 4.1 `_scratch/image/t18_run.py`

**(a) 独占账本 `GLUE`（`T22_GLUE_PROF=1` / `--prof` / `--glue-prof`）**

* 每个 call site 记**一个叶子段**：`gmark(name, t0)`；嵌套层各自贡献 `*other` 残差：
  `blk.other` = block_step 墙 − 其全部子桶，`att.other` / `ffn.other` / `step.other` 同理。
  **因此 Σ桶 ≡ step wall（按构造成立），残差只暴露账本自身的开销**。
* 叶子清单（30+ 个）：`host.wait/fill/read/osync`、`pre.temb/xemb/refiner/capemb/refiner2`、`step.us/final/lat/ckpt/other`、
  `blk.norm1/norm2/sbg/ffn/ffnpost/other`、`att.qkv.img/unpack3/rms_rope/packkb/packvb/packq/unpack_o/o.img/unpack_a/other`、`ffn.aimg/ymap/other`。
* 关闭时 = 每 call site 一次 `if GLUE is None`（实测开销量级：每层 28 次 × 240 层 ≈ 1 ms，不进算术）。
* 落盘：`<tag>_calls.json` 新增 `glue`（每桶 s/n）、`glue_wall_s`、`glue_residual_s`；stdout 打 `[t22g]` 表。
* **顺带修正**：`att.unpack_o` 原先的 pmark 把整次 ip dispatch 也算进去了（t=0 在 packq 前）——新账本把 dispatch 与 unpack 分开。

**(b) glue-2 快路径（`T22_GLUE2=1` / `--t22-glue2`，默认关）**

| 改动 | 省什么 | 预测 |
|---|---|---:|
| `unpack_bytes_t_into`：ip/o 输出直接写进 `o_flat`/`a_flat` 行块 | 每层 2 次 491 KB scratch→目的拷贝 | 0.02–0.05 s |
| Q/K/V 三片用 `memoryview` + `unpack_bytes_mv` | 每层 3 次 245 KB `bytes()` 切片拷贝 | 0.01–0.03 s |
| `_xblock`：96 行 X 块复用（`np.zeros` 480 次/图） | 每层 2 次 491 KB 分配+清零 | 0.02–0.05 s |
| `inv64`/`inv_q64` 预转 int64 | 每层 4 次 `astype`（t3d_codec 包装里） | <0.01 s |
| **合计** | | **0.05–0.15 s（保守；不是"大头"）** |

*逐位论证*：四条都只改**目的地地址/缓冲复用**，不改核函数、不改元素顺序、不改 inv。
`unpack_blocked_t` 的 `rows = u16.size // (m*hd)` 依赖 `out.shape`，所以**只在整块（rows==PROJ_M / SEQ-q0>=IP_MQ）启用**，短块自动回退老路（128²/256² 全是整块；144×128、128×256 会走回退）。

### 4.2 `webui/bridge.py`

* 新增两个显式转发：`--t22-fast-glue`（默认 = env `T22_FAST_GLUE`，**OFF**）、`--t22-glue2`（默认 = env `T22_GLUE2`，**OFF**）；
  `driver_argv()` 只在开启时追加（`is_t18` 守卫：`--driver t13_run` 不会收到这两个 flag），meta 里记 `t22_fast_glue/t22_glue2`。
* **常驻 device worker 也能开**：`dev_worker.py` 自建的 namespace 没有这两个字段，所以 `driver_argv()` 的缺省取**服务级开关**
  （`FAST_GLUE_DEFAULT/GLUE2_DEFAULT`），而不是 `False`——服务环境设 `T22_FAST_GLUE=1` 时 bridge 与 worker 两条路都生效。
  逐 job 的字段本模块没加（worker 侧要改 dev_worker.py，不在本单范围）；A/B 用服务级 env 即可。
* **默认翻法是一行**：`FAST_GLUE_DEFAULT = True`（或服务环境 `T22_FAST_GLUE=1`）。**本轮不翻**：四件套没跑（§7）。
* 不传时行为与交付完全一致（不会因为"忘了传"而静默启用）。

## 5. 为什么"胶水砍半"在宿主内不可达（可复算）

把任务书的 12.4 s 按 §3.2 的占比摊开，再问"每一项能被宿主代码砍多少"：

| 项 | 量级 s | 宿主能砍吗 | 上限 |
|---|---:|---|---|
| staging fill（49.4 GB @ ~22 GB/s） | 2.7–3.7 | **不能**：字节数由 BG/B3 132.7 MB×30 层×8 步 + W 66 MB×30×8 决定；减字节要 `--bo-resident`（+3.98 GB BO） | −0.9~1.4（且内存墙） |
| 层内 pack/unpack（剥离 dispatch 后） | 2.7 | 能，但都是 nJIT 已经加速过的 gather；glue-2 只拿拷贝 | −0.05~0.15 |
| refiner 4 块（host f32） | 2.2–2.6 | **不能**：搬 NPU 换参考（拒绝）；numpy 已是 f32 参考本身 | 0 |
| norm/SBG/mod | 0.7 | U1 已砍 2.8 s 相位；余量为 0.7 | −0.1 |
| 未归属 | 4.5 | **未知**，本轮账本要回答；若是 print/absmax/分配，可砍 1–2 s | **−0~2（待测）** |
| **合计** | **12.4–14.7** | | **现实 −1.7（U1 已证）~ −3.5（含未知全砍）** |

U1 的 −1.68 s 是**同窗口实测**（30.23→28.55，latents 逐字节同，单元门 204/204）。
即：**U1 已把胶水砍掉 13.5 %；再加 glue-2 与未知项，天花板约 −3.5 s ≈ 28 %，仍低于止损线的 30 %。**
要过 30 % 必须动`--bo-resident`（器件内存）或 refiner（换参考）——两条都超出"宿主胶水"范围。

## 6. 下一窗口的执行顺序（一条命令一条链，全部可复制）

```powershell
# 0) 健康门（不占器件）
py310 _scratch/image/t3d_baseline.py --check

# 1) U1 单元门（不占器件，204/204 + 100/100 + 240/240）
py310 _scratch/perfc/t22_gate.py

# 2) 细分账本（占器件 ~2 min；t22_prof 会自己打开 PROF -> 账本同时开）
py310 _scratch/perfc/t22_prof.py --tag t22g_L --height 128 --width 128 --steps 8 --seed 7 ^
  --pe webui/jobs/20261018-170459-ed2a32/pe_p0.npz ^
  --schedule webui/jobs/20261018-170459-ed2a32/schedule.npz --cache-dir webui/cache/128x128

# 3) 产品四件套 A/B（不设 env = A 臂；设 T22_FAST_GLUE=1 = B 臂；两次都要新 NPU cache）
py312 _scratch/t22_e2e_measure/run_e2e.py --port 7872 --prefix t22gA
$env:T22_FAST_GLUE="1"; py312 _scratch/t22_e2e_measure/run_e2e.py --port 7872 --prefix t22gB
Remove-Item Env:T22_FAST_GLUE

# 4) glue-2 单变量（B 臂加 T22_GLUE2=1；先只在驱动级跑 t22_prof 对比 latents）
```

**门禁（缺一不可）**：
1. latents 逐字节：128² `9f86aee91de3dad0…`、256² `8ad7453293f07378…`（`run_e2e` 直接落 sha）；
2. run-to-run：同臂两次同 sha；
3. closure ±1 s：`job.json.timing.closure.pass == true`；
4. `t3d_baseline --check` 前后；
5. 双态 A/B（快/慢窗各一对；本轮档案全是快态 21.6–24.5 GB/s，**不能把单态当双态**）；
6. 四件套：并排图（U1 若逐字节同，PNG sha 必同，并排图直接复用 `T21-3B3-128x128-side-by-side.png` 作对照）、闭合面板、实现声明（"宿主快路径，不改镜像/算术"）、批语义（batch=2 逐字节）；
7. `[t22g] residual` ≤ ±0.05 s。

## 7. 能力缺口（为什么本文件没有新数字）

本会话的沙箱**禁止创建任何进程**：`child_process.spawnSync` 对 `python.exe`/`cmd.exe`/`git` 一律 `EPERM`，
提升到 `workspace-write` 亦同。因此：不能起 webui 服务、不能跑 py310 器件、不能 `git commit`、不能跑 `t3d_baseline --check`。
**本轮交付 = 代码 + 可复算的间接账 + 预测 + 执行单**；上述 §6 的 2/3/4 步必须在有进程权限的会话里跑，跑完把
`[t22g]` 表贴回本文件 §3.2 旁边，才算"细分计时先行"闭环。

## 8. 精确剩余量

| 项 | 状态 | 剩余 |
|---|---|---|
| loop 内细分账本 | **代码完成**（未跑） | 1 个器件窗口（~2 min）+ 贴表 |
| U1 产品透传 | **完成**（bridge 显式开关 + meta） | 四件套 2 个窗口（A/B，各 ~2 min） |
| U1 翻默认 | **未翻**（等四件套） | 一行 + 复跑一次默认作业 |
| glue-2 | **代码完成**（默认关，未测） | 1 对 A/B（预计 −0.05~0.15 s，可能不值得单独立项） |
| `blk.other` 的 4.5 s | **未知**（本轮账本的目标） | 账本跑完才能定方案 |
| bo-resident（BG-only 变体） | 未做 | 先量 commit 余量；+2.65 GB BO，预计 staging −0.9~1.0 s，风险中 |
| 胶水砍 30 % | **未达**（U1 13.5 %；天花板 ~28 %） | 按止损线：带本表数据回负责人重议 |

## 9. 复现/交接

* 开关：`T22_GLUE_PROF=1`（账本）、`T22_FAST_GLUE=1`（U1）、`T22_GLUE2=1`（glue-2）；CLI 同名（`--glue-prof/--t22-fast-glue/--t22-glue2`）。
* 账本读法：先看 `residual`，再看 `blk.other`、`host.fill`、`pre.refiner` 三项的排序；`step.ckpt/step.lat` 在 step wall 之外，不进闭合。
* 本轮改动文件：`_scratch/image/t18_run.py`、`webui/bridge.py`、本文件。**未改** `webui/runner.py`、`vae_*/config.py`、任何镜像。

## 10. t22g_L 实测（2026-10-21 09:41，第一个真账本）与两处修正

> 128² / 8 步 / 30 层 / seed 7 / qkv+p10 开 / fast_codec 开 / **U1(T22_FAST_GLUE) 未开**（t22_prof 不设它，见 §10.4）。
> 产物：`_scratch/image/t22g_L_glue.json`、`t22g_L_calls.json`、`t22g_L_latents.npy`。WALL（step 之和）**30.993 s**，dispatches 1440 / retries 0，device 15.236 s、staging 2.432 s。

### 10.1 先回答"residual = −9.556 是怎么来的"：账本有一个父桶 + 一处 clamp

v1 的 `blk.ffn` 用 `gmark` 包住整个 `L.ffn()`，而 `L.ffn()` 内部会调 `dev.dispatch()`，后者已经把 `host.wait/fill/read` 记成叶子桶。
于是 `blk.ffn` 成了**父桶**，被重复计入一次。可逐位复算：

    blk.ffn 11.2226 = ffn 的子桶 11.1120 + ffn.other 0.1106
    子桶 11.1120 = dev 9.4678 + fill 1.3490 + read 0.116 + aimg 0.0382 + ymap 0.1413

而 `step.other = max(0, wall − 子桶)` 在父桶超算时被**钳到 0**，把真正的步级开销 **1.5559 s** 藏掉了：

    Σ(全部桶) 40.5492 − 重复 11.1120 + 被藏 1.5559 = 30.9931 = WALL   ✔

所以 −9.556 = −11.112（重复）+ 1.556（被 clamp）。**修法（已提交）**：`blk.ffn` 改成"减去内部子桶后的独占残差"，
并删除重复的 `ffn.other`；clamp 自然消失，`step.other` 会把那 1.556 s 报出来。

### 10.2 ① blk.ffn 11.22 s 里"host 真活"只有 0.29 s（2.6 %）

| 组成 | s | 占比 | 性质 |
|---|---:|---:|---|
| 器件 wait（ffn dispatch） | 9.468 | 84.4 % | 器件 |
| BO 写（A+SBG+BG+B3 = 133 MB/层） | 1.349 | 12.0 % | 器件 IO |
| BO 读（Y 245 KB/层） | 0.116 | 1.0 % | 器件 IO |
| ffn.aimg（A 的 bf16） | 0.038 | 0.3 % | **宿主** |
| ffn.ymap（Y 的 bf16→f32） | 0.141 | 1.3 % | **宿主** |
| ffn.other（Python/列表/字典） | 0.111 | 1.0 % | **宿主** |
| **合计** | **11.223** | 100 % | **宿主真活 0.29 s** |

结论：FFN 这条 dispatch 没有宿主肥肉，46.8 ms/层 = 39.5 ms 器件 + 6.1 ms BO 写 + 1.2 ms 宿主。
**不要再往 ffn 里找。**

### 10.3 ③ pre.refiner 5.22 s：旧账 2.2–2.6 s 是"只算 f32 forward"，新账把 step-0 的 2.83 GB 权重载入也算进来了

| 口径 | 数值 | 说明 |
|---|---:|---|
| 旧 `T3.full_block_host`（t22a1 phases） | 2.566 s / 32 次 | **只包 forward**，不含 `L.refblock()` 的权重读盘；且 t22a1 的 step0 额外 2.8 s 被记在"首步"桶 |
| 新 `pre.refiner`+`pre.refiner2` | 5.223 s / 32 次 | 包 `L.refblock()` + `T3.modulation` + `full_block_host` |
| 差 | ≈ 2.6 s | step 0 的 refiner cache（2.83 GB .npy）读入 |

旁证：8 步的 wall 是 `6.687 / 3.477 / 3.403 / 3.454 / 3.434 / 3.398 / 3.443 / 3.697`，step0 比稳态多 **3.25 s**。
**下一步已加 `pre.refblock` / `pre.refblock2` 两个桶**，下次直接分开报"读盘 vs forward"。
判读：稳态 refiner ≈ 2.6 s/图（占 wall 8.4 %，纯 f32 参考本身，不能砍）；step-0 的 ~2.6 s 只能移出 loop（总量不变，见 §5 第 6 行）。

### 10.4 修正后的独占账（t22g_L，WALL 30.993）

| 桶 | s | % | 备注 |
|---|---:|---:|---|
| host.wait（器件） | 15.236 | 49.2 | ffn 9.468 / att 5.768 |
| host.fill | 2.441 | 7.9 | ffn 1.349 / att 1.083 |
| host.read | 0.281 | 0.9 | |
| host.osync | 0.047 | 0.2 | |
| pre.refiner+refiner2 | 5.223 | 16.9 | 稳态 2.6 + step0 2.6 |
| blk.norm1+norm2+ffnpost | 2.168 | 7.0 | **U1 未开**时的价格；U1 后约 0.7–0.9 |
| att.other | 1.767 | 5.7 | 分配/杂项，见 §10.5 |
| att.packkb+vb+q | 0.647 | 2.1 | |
| att.unpack_o+a | 0.447 | 1.4 | |
| att.rms_rope | 0.356 | 1.1 | |
| att.o.img + qkv.img + xblk | ~0.44 | 1.4 | |
| blk.sbg | 0.107 | 0.3 | |
| step.final | 0.070 | 0.2 | |
| blk.ffn（独占，= ffn.other） | 0.111 | 0.4 | 修好后 |
| **blk.other + step.other（原被 clamp 藏掉）** | **1.556** | **5.0** | 每层 absmax/print/字典 |
| **合计** | **30.993** | 100 | |

**胶水 = 30.993 − 15.236 = 15.757 s**（本窗口；任务书 12.4 s 是另一窗口且 U1 开）。U1 未开使 norm1+norm2+ffnpost 多付 ≈1.2–1.5 s。

### 10.5 ② 下一刀（按"可砍 × 风险"排序，全部有本次实测支撑）

| # | 刀 | 实测依据 | 预测 | 风险 | 判 |
|---|---|---|---:|---|---|
| 1 | **U1 翻默认**（四件套） | norm1 .654+norm2 .747+ffnpost .766 = 2.17 s，U1 后同类相位 ≈0.7–0.9 | **−1.2~1.5 s** | 低（已有 −1.68 同窗口 A/B、latents 逐字节、单元门 204/204） | **先做** |
| 2 | **att 缓冲复用（GLUE2）** | v2 实测：真正的分配桶 `att.alloc` 只有 0.006 s（v1 的 1.77 s 里 0.97 s 是 xblk+qkv.img）；见 §11 实测 **−0.55 s** | **−0.5~0.6 s** | 低（只改目的地/缓冲复用；GLUE2 已实现，待 A/B） | **本轮已写代码** |
| 3 | **blk.other+step.other 1.56 s**：每层 3×`np.abs().max()` + 1 行 print(flush) + rec 字典 | 本次首次被看见；已加 `blk.rec` 与 `step.print` 两个桶直接拆 | **−0.5~1.0 s** | 低（absmax 合并/延后；print 批量） | 下一轮账本确认后做 |
| 4 | **host.fill 2.44 s**：BG-only BO 常驻 | 132.7 MB/层×240 = 31.8 GB 写；BG 占 2/3 | **−1.2~1.4 s** | 中（+2.65 GB BO、30 个 BO 崩溃面、内存余量未知） | 需裁决 |
| 5 | pre.refblock step-0 2.6 s 移出 loop | 旁证 step0 +3.25 s | loop −2.6，**总量不变** | 低 | 只在对 loop_s 记账时有意义 |
| 6 | refiner 稳态 2.6 s | 纯 f32 参考 | 0 | 换参考=拒绝 | 不做 |

1+2+3 = **−2.4~3.6 s（胶水的 15–23 %）**；加 4 为 **−3.6~5.0 s（23–32 %）**。这就是止损线 30 % 的边界：**能不能过线取决于第 4 刀（器件内存）**。

### 10.6 口径：`[prof]` 与 `[t22g]` 为什么并存且数字不同

`[prof]` 的 `post.ffnpost` 与 `[t22g]` 的 `blk.ffn` **是同一段、不同边界**：`post.ffnpost` 从 `y = L.ffn(...)` 之前量到 ffnpost 之后，
`blk.ffn` 只到 `L.ffn()` 返回。本次实测 `post.ffnpost 11.988 − blk.ffn 11.223 = 0.765 ≈ blk.ffnpost 0.766`（1 ms 内），
**两套时钟互相验证**。以后对外只引 `[t22g]`（独占、带 residual），`[prof]` 保留作历史对照。

## 11. v2 双窗（t22g_L2 / L2b）复核：GLUE2 不是 −4.2 s，是 **−0.55 s**

> 两窗产物都在：`t22g_L2_glue.json`、`t22g_L2b_glue.json`、两份 `_calls.json`、两份 `_latents.npy`。
> **三份 latents（L / L2 / L2b）sha256 全部 = `9f86aee91de3dad0…`（产品 128² 门）** —— 逐字节，零算术改动，这条是硬的。

### 11.1 先把 WALL 核清

| 窗 | wall（step 之和） | device_s | staging_s | 备注 |
|---|---:|---:|---:|---|
| t22g_L2（A，GLUE2 off） | **31.518** | 16.374 | 2.670 | 父会话引的 "31.0" 是上一窗 t22g_L 的 30.993，**本窗是 31.518** |
| t22g_L2b（B，GLUE2 on） | **26.827** | 15.494 | 2.433 | 快态 |

两窗差 **−4.69 s**，但其中 **−1.12 s 是 device+staging**（16.374→15.494 / 2.670→2.433）——**GLUE2 不碰器件、不碰 BO 写**，这部分只能是机器状态。

### 11.2 逐桶差 + 对照归一

对照桶（GLUE2 完全不碰、且同为访存型 numpy）：`blk.norm1` 0.600→0.418（**0.70**）、`blk.norm2` 0.709→0.515（0.73）、`blk.ffnpost` 0.729→0.510（0.70）、`att.rms_rope` 0.336→0.242（0.72）、`blk.rec` 0.295→0.212（0.72）。
→ 本窗"机器快了约 30 %"，**一切差值先除以 0.71 才算 GLUE2 的账**。

| 桶 | L2 | L2b | 原始 Δ | 归一后归属 GLUE2 |
|---|---:|---:|---:|---:|
| att.xblk（X 块 np.zeros→复用） | 0.375 | 0.070 | −0.305 | **−0.19** |
| att.other（Qm/o_flat/a_flat 分配+杂项） | 0.722 | 0.386 | −0.336 | **−0.12** |
| att.unpack_o + unpack_a（直写 dest） | 0.429 | 0.188 | −0.241 | **−0.11** |
| att.qkv.img（memoryview 切片） | 0.592 | 0.371 | −0.222 | −0.04 |
| packkb+vb+q | 0.615 | 0.383 | −0.232 | −0.05 |
| ffn.ymap（y 复用） | 0.127 | 0.062 | −0.065 | −0.03 |
| att.alloc | 0.006 | 0.000 | −0.006 | −0.00 |
| 其余（含误差项） | | | | +0.00 |
| **合计** | | | | **≈ −0.55 s（区间 0.45–0.70）** |

**结论：GLUE2 = −0.55 s，不是 −4.2 s。** 原始 −4.69 里 ≈1.12 s 是器件/IO、≈3.0 s 是宿主侧整体变快（同一份代码在对照桶上也掉了 30 %）。
按纪律：**报告里只能写 −0.55 s**，并注明"同窗原始差 4.69 s 不可用作 GLUE2 收益"。

### 11.3 刀 3 的实测天花板：0.31 s，不是 1.0 s

`blk.rec` 0.295 s + `step.print` 0.016 s = **0.311 s/图**（240 层 × 3 次 absmax + 240 行 print）。
→ print 批量化只值 **16 ms**（不值得改日志形态）；absmax 去临时量（`T22_GLUE3`，已实现）预计 **−0.1~0.2 s**。
**刀 3 从"−0.5~1.0"下调为"−0.1~0.2"。**

### 11.4 刀序与止损线（刀 4 被否后重算）

| 刀 | 实测/预测 | 状态 |
|---|---:|---|
| 1 U1 翻默认 | −1.0~1.4（本窗 norm1+norm2+ffnpost = 2.04 s，U1 后 ≈0.7） | GO（四件套） |
| 2 GLUE2 | **−0.55**（§11.2） | GO（latents 已逐字节 ×3） |
| 3 absmax 去临时量 | −0.1~0.2 | GO（已实现 T22_GLUE3） |
| 4 BG-only BO 常驻 | −1.2~1.4 | **否决**（avail_commit 10.32 GB，+2.65 GB 后破 9 GB 内存门） |
| **合计** | **−1.65~2.15 s** | 胶水 11.3–15.2 s 的 **11~19 %** |

**止损线判定：宿主侧天花板 ≈ −2.1 s（11–19 %），30 % 不可达。** 除非重开"器件 BO 常驻 / 内存门"或"refiner 换参考"，
否则本单应在 U1+GLUE2+GLUE3 落地后收线，按止损条款带本表回负责人。

### 11.5 下一步（建议）

1. **同窗配对 A/B 复核 GLUE2**（A/B/B/A 四跑，或至少连续 A/B 两跑）：目标是把"机器态"从 4.69 s 里剥掉，确认 −0.55 s；
2. 四件套：U1+GLUE2(+GLUE3) 一起过，别单开 GLUE3（它只有 0.1–0.2 s，单独一次四件套不划算）；
3. 下一窗顺便看 `step.other`：L2 0.992 / L2b 0.647 s（8 步 = 81–124 ms/步）——本轮已把 `loop.mod`/`loop.sbg`/`loop.payload` 也记进账本，下次它会被拆开；
   若 `step.other` 仍 >0.3 s，那是最后一块可砍的宿主时间。

## 12. 同窗 A/B/A（两对）+ C1：GLUE2+3 = −1.2~−1.7 s（点 −1.5）

| 窗 | 开关 | WALL |
|---|---|---:|
| A1 | 0 | 30.684 |
| B1 | GLUE2+3 | 27.908 |
| A2 | 0 | 28.546 |
| C1 | GLUE2 only | 27.178 |
| A3 | 0 | 27.338 |
| B2 | GLUE2+3 | 25.924 |
| A4 | 0 | 26.970 |

- 第一对：反事实 (A1+A2)/2 = 29.615 → **−1.707 s（−5.8 %）**
- 第二对：反事实 (A3+A4)/2 = 27.154 → **−1.230 s（−4.5 %）**
- 合并 **−1.47 s，区间 −1.2~−1.7**；两对同向。整段漂移单调（30.7→28.5→27.3→27.0），第二对在更快基底上测得，**−1.23 是下界**；第一对是慢基底，**−1.71 是上界**。
- C1（GLUE2-only，27.178）夹在 A3/A4 之间——**单跑无法与漂移分离，放弃单跑分离**，按组合效应过门（GLUE3 可并；GLUE3 单项 ≤0.27 s，占比 <15 %）。
- 每条 run 的 latents 均 = `9f86aee91de3dad0…`（产品 128² 门），残差全 +0.000，1440 dispatch / 0 retry。

### 12.1 与 §11 的 −0.55 冲突（更正）

§11 把 blk.norm1/norm2/ffnpost/rms 当"不受处理影响的对照"按 0.70 归一得 GLUE2=−0.55。
但 A/B/A 实测墙漂移只有 4.5–7 %：**对照桶本身被处理影响**（GLUE2 去掉 588 MB/图新页分配 → 缺页压力下降 → 无关 numpy 访存也变快），拿它做分母把效应减了两次。
用 −1.71 回解 L2/L2b：host Δ 3.574 − 处理 1.71 = 1.86 s 漂移（12.3 % 宿主 ≈ 7 % 墙），两对窗口自洽。
→ **§11 的 −0.55 作废，以本节 −1.2~−1.7 为准。**

### 12.2 门 1 产品级 E2E（2026-10-21，四跑八 job）

A/B 两臂各 2 次（cold+steady）：latents 全部 `9f86aee9…`、closure 全 True（+0.03~0.09）、960 dispatch/0 retry、PNG 全 `104ff564`（CPU VAE 口径）。B 臂稳态 43.59/43.56 s，A 臂稳态 44.59/52.18 s（A2 为慢态离群）。
