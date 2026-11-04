# T22-CLOSUREFIX — 三段进程成本进桶表（预测，写在任何实现/测量之前）

**任务书**：t22_closurefix。把三段已知未归属时间纳入 runner 的计时桶表
（bridge import / py312 VAE spawn / prewarm 冷建等待），让**重建闭合**在真实路径
与回退路径上都 ≤ 1 s，且对产品默认路径零回归。

**本文件写于** 2026-10-20T21:0x，**先于** `webui/runner.py` 的任何改动、先于任何器件运行。
判据先写死，测量后不修改（如预测错，如实记录并保留原判据）。

---

## 0. 起点事实（只读，不重查）

| 项 | 值 | 来源 |
|---|---|---|
| 真实 job（flip_job，产品默认路径） | total 144.783 s，unaccounted **+0.0934 s PASS** | job 20261020-201656-7ff25f |
| 16:06 真实 job | total 155.656 s，unaccounted **+1.1236 s FAIL** | job 20261020-160603-795d69 |
| 重建口径（§5 rig，bridge+vae_worker 直跑） | unaccounted **+4.58/+5.94/+4.98/+5.81 s** | PROD-REPORT §3 G9、§5 |
| 分解 | 桥 import 2.06/3.78/2.83/3.47 s + py312 VAE spawn 2.6/2.1/2.1/2.36 s | PROD-REPORT §3 |
| 冷建（T22-PREWARM E 臂） | startup 37.83 = 子进程 10.80 + **等待 27.04**，closure +0.049 PASS | T22-PREWARM.md §5 |
| 桶表 | 15 桶，|unaccounted| ≤ 1.0 s | webui/runner.py TIMING_BUCKETS |

**16:06 的 +1.12 s 不是 pack 的墙钟**（该 job 的 `prewarm_wait_s=0`、`prewarm_src=reused`，
pack 36.363 s 是产物 meta 里的历史值，本次打包进程只跑了 2.571 s 的校验）。本单的测量要
回答它到底是什么（见 P5）。

---

## 1. 方案（先定结构，再改代码）

按 `webui/runner.py` 现有结构，"谁在等这笔时间"决定桶：

| 新增桶 | 归属 | 测量锚点（runner 内） | 为什么不是别的桶 |
|---|---|---|---|
| `bridge_import_s` | 桥进程 py310 解释器 + import（**交付桥路径**） | 父进程 `dev_perf0` → 子进程第一个 `@@WEBUI@@` 事件 `tp` | 旧 `startup_s = startup_done.tp − dev_perf0` 把它吞了；重建口径（`startup_done.tp − 第一个事件.tp`）看不见它 |
| `vae_spawn_s` | 一次性 `vae_worker.py`（py312）进程启动 | 父侧 `stream_process` 墙钟 − 子进程自报 `total_s` | 子进程的 `total_s` 从 `decode_batch` 内开始，进程启动不在里面 |
| `prewarm_pack_s` | 冷建时 job **阻塞等待**打包的秒数 | `_wpack_join` 的 `thread.join()` 时长（原 `prewarm_wait_s`） | 打包进程自身墙钟与 TE/VAE 重叠，记进去就是重复记账（T22-PREWARM §6 定案）；等待是从 `startup_s` 里**搬家**出来的，不是新增时间 |

**恒等式（要守住的两条）**：

1. 桥路径：`bridge_import_s + startup_s == 旧 startup_s`；
   `device_tail_s = device_wall − (bridge_import_s + startup_child_s) − loop_s`
   （tail 必须仍按"spawn→startup_done"整段扣，否则 import 会被记两次）。
   ⇒ **同一 job 的桶和、unaccounted 逐位不变**，只是归属从 startup 挪到 import。
2. 冷建：`startup_s` 只留子进程启动，等待搬进 `prewarm_pack_s`；桶和不变。

**新增的原始锚点**（只供离线重建，不进桶）：`device_launch_tp`、`first_event_tp`、
`startup_done_tp`、`vae_child_wall_s`、`prewarm_pack_wall_s`（打包进程自身墙钟，
原 `prewarm_pack_s` 改此名）。

---

## 2. 预测（判据先写死）

| # | 预测 | 判据（测量后不改） |
|---|---|---|
| **P1** 产品默认路径（worker + NPU link）两 job | `bridge_import_s = 0`、`vae_spawn_s = 0`；unaccounted **+0.05…+0.20 s** | 两 job 都 PASS，且 unaccounted 与 flip_job 的 +0.0934 同一个带（≤ 0.25 s） |
| **P2** 回退路径（交付桥 + `vae_worker.py` py312） | `bridge_import_s` **1.7–2.5 s**、`vae_spawn_s` **2.1–2.8 s**；闭合 **≤ 1 s** | 两桶都在预测带内；|unaccounted| ≤ 1.0 s |
| **P3** 冷建（盘上无 artifact，job 模式） | `prewarm_pack_s`（等待）**25–32 s**；`startup_s` 相应下降同样的量；闭合 ≤ 1 s | 等待 > 20 s 且与 startup 的下降量之差 ≤ 0.05 s；closure PASS |
| **P4** 恒等式 | 改动前后同一 job 的桶和差 **0.000 ± 0.02 s**（P1/P3 路径）；回退路径的桶和**增加** `vae_spawn_s`（这是修好的那笔） | 重算 `sum` 与 sealed `sum_s` 差 ≤ 0.02 s |
| **P5** 16:06 的 +1.1236 s | **不是 pack**：`prewarm_wait_s=0`；预测是"器件段前停用常驻 worker"（kill 一个持 4 个上下文的 4 GB 进程）**0.9–1.1 s** + 段间开销 ~0.05 s | J3（前面有活 worker 的 low 模式 job）测 `worker_drop_s`：若 ≥ 0.5 s 则解释成立；若 < 0.2 s 则**本预测被否**，16:06 归因回到未知，如实写进报告 |
| **P6** 字节门 | latents `9f86aee9`；worker/NPU-VAE 路径 PNG `e7aef10c`；`vae_worker` 回退路径 PNG `104ff564`（两套不可互比，按路径断言） | 逐 job 断言，任一不等则 FAIL |
| **P7** 门禁 | `t3d_baseline --check` 前后 "ALL REFERENCE BYTES UNCHANGED" | 改前一次、改后一次 |

**止损线**：0.5 设计日内不闭合 → 带数据回来（哪条路径、差多少、残余在哪一段）。

**不做**：不改 `bridge.py` / `dev_worker.py` / `vae_*`（禁动），不改桶门限 1.0 s，
不把任何未测时间倒进桶里（P0-4 的"每秒只入一桶"不变）。

---

## 3. 运行矩阵（同 session）

| 会话 | 服务 env | job | 覆盖路径 |
|---|---|---|---|
| S1 | `ZIMAGE_DEV_WORKER=1`（默认 residency） | J1 p10 默认（body 不带 p10）、J2 p10=off | 产品默认（worker + NPU link），A/B 双态 |
| S1 | 同上 | J3 `memory_mode=low` | worker 被停用 + 交付桥（16:06 形状），量 `worker_drop_s`/`bridge_import_s` |
| S2 | `ZIMAGE_DEV_WORKER=0` + `ZIMAGE_VAE_ONESHOT=1` + 删 artifact | J4 | 回退 `vae_worker.py` + 冷建 prewarm（两段同 job） |

**J4 的测试门**：`ZIMAGE_VAE_ONESHOT=1` 是新增的**测试门**（默认关，产品路径逐行不变），
让已经存在的 `vae_worker.py` 回退分支可被确定性触发——否则那条分支只能靠
`vae_local` 抛异常才会走到，A/B 不可复现。
