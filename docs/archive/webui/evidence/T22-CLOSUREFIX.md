# T22-CLOSUREFIX — 四段进程成本进桶表（bridge import / py312 VAE spawn / 冷建等待 / worker 停用）

**一句话**：三段"已知未归属"（桥进程 pre-event、py312 VAE 子进程启动、冷建打包等待）
进了 runner 的桶表，测量后**还发现第四段**——低内存 job 在器件段前**杀掉常驻 worker**
（0.60–0.67 s，16:06 那次 ~1.0 s），它才是 16:06 的 +1.1236 s FAIL 的原因，
**不是 prewarm pack**（那个 job 的 pack 等待是 0 s，产物是复用的）。
同 session 七个真实 job：产品默认 **+0.0444 / +0.0335**（同进程 A/B 对）与
**+0.096 / +0.062**（重复）、交付桥（worker 停用）**+0.042**、回退 `vae_worker` + 冷建
**+0.037**，全部 PASS；重建口径（从原始锚点重算）**7/7 PASS**；
latents 全 `9f86aee9`、PNG 按路径 `e7aef10c`（NPU 链）/ `104ff564`（py312 回退）、
960 dispatch / 0 重试；`t3d_baseline --check` 前后全绿。

**NOT SHIPPABLE 的两条**（先写在前面）：本单不主张任何速度收益（桶只是归属，不产生秒数）；
`ZIMAGE_VAE_ONESHOT=1` 是**测试门**（默认关），只用于让回退分支可复现，不是产品开关。

---

## 0. 结论表

| # | 门 | 判据 | 结果 |
|---|---|---|---|
| ① | 产品默认路径（常驻 worker + NPU VAE 链） | \|unaccounted\| ≤ 1 s，且 ~0.09 s 量级 | **+0.0444 / +0.0335 s**（同进程 A/B 对）+ **+0.0960 / +0.0624 s**（重复）✅ |
| ② | 回退路径（交付桥 + py312 `vae_worker.py`） | ≤ 1 s | **+0.0372 s** ✅ |
| ③ | 含 `prewarm_pack` 的冷建 | ≤ 1 s，且等待被记账 | **+0.0372 s**（`prewarm_pack_s=17.028`）✅ |
| ④ | 附加：交付桥 + worker 停用（16:06 形状） | ≤ 1 s | **+0.0416 s**（`worker_drop_s=0.595`）✅；旧表下同一 job 是 **+0.6366** |
| ⑤ | 重建闭合（从原始锚点重算，不读 runner 的桶字典） | 每个 job ≤ 1 s | **7/7 PASS**，与 runner 的 sealed sum 差 **0.000 s** |
| ⑥ | latents 逐字节 | `9f86aee91de3dad0` | **7/7** ✅ |
| ⑦ | PNG 逐字节（按路径） | `e7aef10c`（NPU 链）/ `104ff564`（py312 回退） | **6/6 + 1/1** ✅ |
| ⑧ | dispatch / 重试 | 960 / 0 | **7/7 960 / 0** ✅ |
| ⑨ | `t3d_baseline --check` | ALL REFERENCE BYTES UNCHANGED | 改前（21:07）+ 改后（21:31）✅ |

---

## 1. 定案：16:06 的 +1.1236 s 是什么（读源 + 数据）

**不是 prewarm pack**。job `20261020-160603-795d69`：

| 证据 | 值 |
|---|---|
| `prewarm_wait_s` | **0.000**（job 没等打包） |
| `prewarm_src` | `reused`（产物 12:46 就建好了，meta `pack_s=36.363` 是**历史值**） |
| 该 job 自己的打包进程 wall | 2.571 s（只做了"校验已有产物"） |
| `prewarm_pack_s`（旧语义 = 打包进程自身墙钟） | 36.363 s —— 它**本来就不该进桶**（与 TE/VAE 重叠，T22-PREWARM §6） |

真正的时间在**器件段之前**：这个 job 落 low 模式、不能用常驻 worker，runner 在
器件段前 `_drop_worker()` 杀掉了 16:02 那个 worker（4 个 device context + 3.98 GB 驻留）。
旧代码只写一行日志、**不计时**。同形状 job 今天实测这段 **0.671 / 0.595 s**
（s1b / s1d），16:06 那次更大（worker 更大 + 当时负载），量级与 1.12 s 的缺口一致。
**旧的 18 桶表下**：s1d 的 +0.042 会变成 **+0.637**；s1b（在旧表下封账）实测 **+0.712**。
16:06 的 1.124 无法再拆精确（旧代码没给 kill 计时），只给这一段是同一形状这个事实。

---

## 2. 改动（`webui/runner.py`，四个桶 + 一个测试门 + 三个锚点）

| 桶 | 量什么 | 锚点 | 出现在哪条路径 |
|---|---|---|---|
| `bridge_import_s` | py310 解释器 + 桥 import | 父进程 launch 标记 → 子进程**第一个** `@@WEBUI@@` 事件的 `tp` | 交付桥（worker 路径恒 0） |
| `vae_spawn_s` | py312 `vae_worker.py` 进程启动 | 父侧 `stream_process` 墙钟 − **子进程自报** `total_s` | 仅一次性回退（常驻/链接路径恒 0） |
| `prewarm_pack_s` | 冷建时 job **阻塞等待**打包 | `_wpack_join` 里 `thread.join()` 的墙钟 | 仅冷建 |
| `worker_drop_s` | 杀掉**活着的**常驻 worker | `Worker.kill()` 前后 `perf_counter` | 只有"这个 job 用不上 worker 但 worker 还活着" |

**关键结构决定（读源得出）**：

1. `startup_s` 收窄为 `first_event → startup_done`（原来是 `launch → startup_done`）。
   ⇒ **`bridge_import_s + startup_s == 旧 startup_s`**，桶和逐位不变，只是归属搬家。
2. `device_tail_s` 仍按 **`device_wall − (import + startup) − loop`** 扣；
   `device_wall` 从 `stream_process` 内部开始（在 launch 标记之后），
   所以 tail 必须扣整段 `launch → startup_done`。**否则 import 会被记两次**
   （identity_check 里算了这条反例：只扣新 startup 会让 tail 多 1.710 s）。
3. `prewarm_pack_s` 是从 `startup_s` **搬家**出来的等待，不是新增时间；
   打包进程自身墙钟改名 `prewarm_pack_wall_s`，**仍然不进任何桶**。
4. `ZIMAGE_VAE_ONESHOT=1`（默认关）：让已有的 `vae_worker.py` 回退分支可确定性触发。
   不开时产品路径逐行不变（`oneshot=False` 走原 `vae_local`）。
5. 原始锚点 `device_launch_tp / first_event_tp / startup_done_tp / vae_child_wall_s`
   进 `timing`（**只作信息位，不进桶**），供离线重建核对。

**恒等式自检**：`_scratch/t22_closurefix/identity_check.py` → `IDENTITY PASS`
（桥拆分 0.000 差、冷建搬家 0.000 差、19 桶不重不漏）。

---

## 3. 门表：七个真实 job（同 session，产品 HTTP API）

| job | 路径 | p10 | mem | VAE | total | sum | **unacc** | drop | import | startup | prewarm | vspawn | PNG |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `211058-ee20f1` (s1a) | 产品默认 worker | **默认 on** | fast | npu | 131.060 | 130.964 | **+0.0960** | 0 | 0 | 10.399 | 0 | 0 | `e7aef10c` |
| `211854-4f2e77` (s1c) | 产品默认 worker | off | fast | npu | 133.476 | 133.414 | **+0.0624** | 0 | 0 | 8.663 | 0 | 0 | `e7aef10c` |
| `213405-5e13a8` (s1a2) | 产品默认 worker（**同进程 A**，冷 worker） | **默认 on** | fast | npu | 125.436 | 125.391 | **+0.0444** | 0 | 0 | 9.710 | 0 | 0 | `e7aef10c` |
| `213611-fedabe` (s1c2) | 产品默认 worker（**同进程 B**，热 worker startup 1.221 s） | off | fast | npu | 104.610 | 104.577 | **+0.0335** | 0 | 0 | 1.221 | 0 | 0 | `e7aef10c` |
| `211313-89ff11` (s1b) | 交付桥（worker 停用） | off | low | npu | 122.772 | 122.061 | +0.7117 → **+0.0402** | **0.671** | 0.595 | 4.466 | 0 | 0 | `e7aef10c` |
| `212109-8091c1` (s1d) | 交付桥（worker 停用） | **默认 on** | low | npu | 122.749 | 122.708 | **+0.0416** | **0.595** | 0.503 | 3.500 | 0 | 0 | `e7aef10c` |
| `212644-2228c7` (s2a) | 交付桥 + **py312 回退** + **冷建** | **默认 on** | fast | cpu | 74.813 | 74.776 | **+0.0372** | 0 | 0.456 | 8.458 | **17.028** | **1.147** | `104ff564` |
| `160603-795d69`（对照，旧代码） | 交付桥（worker 停用） | — | low | npu | 155.656 | 154.532 | **+1.1236 FAIL** | 未测 | 未测 | 7.329(含import) | 0 | 0 | `e7aef10c` |

* s1b 是**旧表封账**的 job（当时 `worker_drop_s` 还不存在）：sealed +0.7117；
  把已记录的 `worker_drop_s=0.671` 并入后 **+0.0402** —— 同一 job 的前后对照。
* s1d 是**新表封账**的同形状 job：**+0.0416**；把它换回旧表（去掉该桶）就是 **+0.6366**。
* s1a/s1c 的 `device_tail_s = 0`：worker 冷启动在 `startup_s` 里，比 run 段还长，
  tail 被 clamp 到 0 —— 这是 P0-4 的既有语义（冷启动是真实时间），不是漏账（闭合 +0.096/+0.062 已证）。
* 冷建：s2a 的打包进程 wall **23.502 s**（本次），被 TE 8.479 s 遮住一部分，
  job 实等 **17.028 s**；重建产物 **sha `da2f45a6`（与 12:46 那份逐字节相同）**。
* 另有一条 low 模式 job `211519-58440c`（同一批，前一个 job 已把 worker 杀掉，所以
  **没有 drop 可记**）：total 113.66、**unacc +0.0247 PASS**、`bridge_import_s=0.628`、
  startup 3.965、tail 0.435、PNG `e7aef10c` —— 它的 PNG 也在证据目录里（第 8 张），
  这里补一句说明它的来历。

---

## 4. A/B 双态（p10 默认 on / off，同 session；唯一变量 = o 投影镜像）

**主对：同一个 server 进程里的连续两个 job**（先 A 后 B；两个都强制 `fast`，否则第二个 job
会被内存门翻成 low —— T22-WORKERVAE §6 已记录，见 §9 第 4 条）。

| 臂 | job | 路径 | p10 | `p10_image` | 器件 o 桶 | startup | dispatch | unacc |
|---|---|---|---:|---|---:|---:|---:|---:|
| **A** | 213405 | worker | **默认 on**（body 不带 p10） | `p10.xclbin` | **attout 966.5 ms** | 9.710（冷） | 960 | **+0.0444** |
| **B** | 213611 | worker | off | （交付） | **att 1361.6 ms** | **1.221（热）** | 960 | **+0.0335** |

前一对（另一次 server，同窗口）作重复：A 211058 `attout 945.7` / +0.0960，
B 211854 `att 1335.5` / +0.0624；低内存重复两条：211313（桥，off）att 1331.8 + 0.712→+0.040、
212109（桥，默认 on）attout 869.4 +0.0416。B 的 startup 1.221 s 是**热 worker 的稳态**
（载荷已驻留、artifact 已映射），那才是产品 steady state；冷 worker 那条是 9.710 s。

器件账（`dev_ms_by_image`，128² 8 步 30 层，六条臂）：qkv 2.22–2.77 s、ffn 9.48–9.58 s
（±0.5 %）同带；`attip` 0.69–0.81 s 是本次窗口里漂移最大的非 o 桶；只有 o 那一桶
整块搬家（`attout` 869–967 ms ↔ `att` 1332–1362 ms）—— 方向与 PROD-REPORT §4 一致。
**本单不重复主张 −0.4 s 的速度数字**（桶账不是速度证据）。

---

## 5. 桶归属矩阵（19 桶，谁在等这笔时间）

| # | 桶 | 归属 | 测量方式 | 备注 |
|---:|---|---|---|---|
| 1 | `queue_s` | 提交→开跑 | `created_ts → started_ts` | 一次性 |
| 2 | `prepare_s` | 构建缓存校验 | `t_start → ensure_shape 返回` | 一次性 |
| 3 | `prewarm_pack_s` | **冷建等待**（兄弟打包进程） | `thread.join()` 墙钟 | **不是**打包进程墙钟（后者进 `prewarm_pack_wall_s`，不进桶） |
| 4 | `textenc_s` | 文本编码全墙 | `textenc.run` 自身 | 含 TE 进程 boot/load |
| 5 | `schedule_s` | 调度表 | 写 npz 前后 | |
| 6 | `retry_s` | 失败重试/快→慢回退 | 整段失败尝试 | |
| 7 | `worker_drop_s` | **停用常驻 worker** | `kill()` 前后 | 新增；四个 drop 站点共用 |
| 8 | `bridge_import_s` | **py310 桥解释器+import** | launch → 第一个事件 | 新增；worker 路径 0 |
| 9 | `startup_s` | 载荷/注册/warm → `startup_done` | first_event → startup_done；worker 路径 launch → startup_done | 语义收窄（桥路径） |
| 10 | `loop_s` | 扩散跨度 | startup_done → 最后一步 | T20 口径 |
| 11 | `device_tail_s` | 器件收尾/落盘/退出 | `device_wall −(import+startup)− loop` | 必须扣整段 |
| 12 | `vae_spawn_s` | **py312 VAE 子进程启动** | 父墙 − 子 `total_s` | 新增；常驻/链接路径 0 |
| 13–17 | `vae_load/decode/npu/save/other` | VAE 阶段 | 子进程 result json | 不变 |
| 18 | `gate_second_s` | 复现门第二遍 | 整遍 | 不变 |
| 19 | `post_s` | 任务收尾 | 最后一遍结束→封账 | 不变 |

**不进桶（信息位）**：`prewarm_pack_wall_s` / `prewarm_proc_s`（兄弟进程墙钟，与 TE/VAE 重叠）、
`device_s`、`staging_s`、`loop_overlap_s`、`job_save_s`、`log_io_s`、
以及锚点 `device_launch_tp` / `first_event_tp` / `startup_done_tp` / `vae_child_wall_s`。

---

## 6. 重建口径（不读 runner 的桶字典）

`_scratch/t22_closurefix/closure_gate.py`：从 `job.json` 的**原始锚点**重算每个桶
（桥日志里的 `@@WEBUI@@` 事件作为独立副本核对 `first_event_tp`/`startup_done_tp`，
VAE 从 `vae_p0.json` 重算，tail 用 `device_wall −(import+startup)− loop`），
再与 sealed `sum_s` 对账：

| label | 重建 unacc | sealed unacc | 差 |
|---|---:|---:|---:|
| s1a | +0.0960 | +0.0960 | 0.000 |
| s1b | +0.0402 | +0.7117（旧表） | +0.671 = 新桶 |
| s1c | +0.0624 | +0.0624 | 0.000 |
| s1d | +0.0416 | +0.0416 | 0.000 |
| s2a | +0.0372 | +0.0372 | 0.000 |
| s1a2 | +0.0444 | +0.0444 | 0.000 |
| s1c2 | +0.0335 | +0.0335 | 0.000 |
| old1606（对照） | +1.124 | +1.124 | 0.000 |

日志副本与锚点逐位一致（`first_event_tp` / `startup_done_tp` 差 0.000），
桥的 `startup_done → driver_done` 跨度 = `loop_s` + 0.007–0.065 s（落盘段，在 tail 里）。

---

## 7. §5 直跑 rig 的 +4.58…+5.94 s 分解（离线，不占器件）

把 `prod_job_prod.ps1` 四个 run 的 wall.txt 与桥日志对齐（`_scratch/t22_closurefix/rig_decompose.py`）：

| tag | 桥 wall | 进程内 span | **pre+post** | VAE wall | 子进程 total | **spawn** | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| p10jobA | 69.06 | 67.04 | **2.03** | 46.16 | 43.60 | **2.56** | 4.59 |
| p10jobB | 58.81 | 55.03 | **3.78** | 29.92 | 27.76 | **2.16** | 5.94 |
| p10jobB2 | 55.62 | 52.77 | **2.85** | 38.00 | 35.87 | **2.13** | 4.98 |
| p10jobA2 | 64.15 | 60.63 | **3.52** | 35.32 | 33.03 | **2.29** | 5.81 |

与 PROD-REPORT §3 记录的 4.58 / 5.94 / 4.98 / 5.81 **逐位相同** ——
即"桥 import 2.06–3.78 + VAE spawn 2.1–2.36"就是这两列。

**但这两列是上界**：桥的 "pre+post" 里既有 pre-event 的 import（runner 新桶量到的
**0.456–0.595 s**），也有 **driver_done 之后的 teardown** 和 **PowerShell `Start-Process`
包装开销**。runner 的 `device_tail_s` 本来就book了 teardown；重建脚本把 tail 写死 0，
所以同一个 job 在 runner 里能过、在重建里过不了。这也是本单新增
`device_tail_s = device_wall −(import+startup)− loop` 这条口径的原因。

---

## 8. 预测 vs 实测（预测写在 `T22-CLOSUREFIX-PREDICTIONS.md`，先于代码）

| # | 预测 | 实测 | 判 |
|---|---|---|---|
| P1 | 产品默认 unacc +0.05…+0.20 | **+0.0444 / +0.0335**（同进程对）、+0.0960 / +0.0624（重复） | ✅（比预测更低） |
| P2 | `bridge_import` 1.7–2.5 s、`vae_spawn` 2.1–2.8 s | **0.456–0.595 / 1.147** | ❌ **高估**：热机 + pyc 缓存；§5 的 2.1–3.8 含 PowerShell 与 teardown（§7） |
| P3 | 冷建等待 25–32 s | **17.028 s**（pack 23.502 s，TE 遮 8.5 s） | ❌ **高估**：本轮打包比 12:46 的 36.4 s 快 |
| P4 | 桶和不变（±0.02 s） | 桥拆分 / 冷建搬家均 **0.000** | ✅ |
| P5 | 16:06 的 +1.12 是 worker 停用 0.9–1.1 s；判据 `worker_drop_s ≥ 0.5` 则解释成立 | **0.671 / 0.595**（`≥0.5` 成立；幅度小于预测） | ⚠️ **半中**（方向对、幅度低；16:06 那次 ≈1.0 s 属同一段） |
| P6 | latents/PNG 按路径 | 7/7 `9f86aee9`；6×`e7aef10c` + 1×`104ff564` | ✅ |
| P7 | `t3d_baseline` 前后 | 前（21:07）/ 后（21:31）全绿 | ✅ |

---

## 9. 诚实清单（负结果与边界）

1. **预测 P2/P3 的幅度错了**（见上），保留原判据。原因已定位：§5 的"桥 import"是
   PowerShell 包装的差值，不是子进程 import 边界；本轮冷建的打包也更快。
2. **16:06 的 1.1236 不能再拆精确**：旧代码没给 kill 计时。新代码同形状实测
   0.595–0.671 s（drop）+ 0.04 s（段间），依据是"同一段、当时 worker 更大"，
   不是同一 job 的重算。
3. **s1a/s1c 的 `device_tail_s = 0`**：worker 冷启动 > run 段，tail 被 clamp；
   闭合仍 PASS（冷启动是真实时间，记在 startup）。这是 P0-4 语义，不改。
4. **A/B 的墙钟不可比（按设计）**：同进程主对里 A 是冷 worker（startup 9.710），
   B 是热 worker（1.221），相差的就是 worker 暖机；两条的 VAE load 也差 12.7 s（冷/热链）。
   本单只把它当"两态都记账、字节相同"的证据，**不当速度证据**；器件账里非 o 桶同带
   （ffn ±0.5 %）才说明两臂可比。
5. **worker 启动途中死掉的那条路径**记不住 import 拆分：worker 在 `w.start()` 期间死、
   同一 attempt 改走桥时，`first_ev` 已被 worker 事件占用（tp 早于新的 launch 标记），
   守卫生效后 `bridge_import_s=0`、import 留在 `startup_s` 里。**总和不漏**（不重复记账），
   只是归属粗一档；本条路径本轮未跑（无现成触发手段），如实记录。
6. **`ZIMAGE_VAE_ONESHOT` 是测试门**，默认关；产品路径（`vae_local` 常驻）逐行未改。
7. **冷建用的是 job 模式**（不是 boot）：产物已按 sha 校验等于 `da2f45a6`，
   备份文件已删（盘上只剩重建后的那份）。
8. **没有动** `bridge.py` / `dev_worker.py` / `vae_*` / `config.py` / 前端；
   门限 1.0 s 未动。

---

## 10. 精确剩余量

| 项 | 值 | 说明 |
|---|---|---|
| 产品默认（同进程 A/B） | **+0.0444 / +0.0335 s** | 热 worker 那条 +0.0335 是稳态；残余是阶段边界记账（pe 载入、calls json、job.json 写盘），未单列 |
| 产品默认（重复两条） | **+0.0960 / +0.0624 s** | 同上 |
| 交付桥 + worker 停用 | **+0.0416 s** | drop 已进桶；残余 0.04 s |
| 回退 `vae_worker` | **+0.0372 s** | spawn 已进桶 |
| 冷建 | **+0.0372 s**（同上） | 等待 17.028 s 已进桶 |
| 旧表下同形状 job | **+0.6366 s** | 只差 `worker_drop_s` 这一桶 |
| 16:06 历史 job | **+1.1236 s（FAIL，对照）** | 旧代码无锚点、无该桶，不改写 |

**尚未进桶、且刻意不进桶**：打包进程自身墙钟（重叠）、`job_save_s`/`log_io_s`
（与主线程重叠或在阶段内部）、PowerShell/外壳开销（不是 job 的时间）。

---

## 11. 提交与复现

| commit | 内容 |
|---|---|
| `7a34da7` | 预测（先于代码） |
| `78d6ac7` | 三个桶 + 测试门 + 锚点 + identity_check |
| `5e7eb1c` | 第四个桶 `worker_drop_s`（16:06 的真正缺口） |
| `90802c7` | 证据 `webui/evidence/t22_closurefix/` + 本报告 + README 索引 |
| `c5ee522` | 五个 job 的 PNG 放进证据目录（`out/` 被 .gitignore） |
| `87283c9` | 同进程 A/B 对（`s1a2` / `s1c2`）与七 job 的汇总 |
| `eb2c3fc` | 第 8 张 PNG（`211519`，worker 已死、无 drop）的来历 + 小 dump 工具 |
| （本节） | 后人怎么跑：五个工具的用法/读法/退出码 |

```powershell
# 恒等式自检（无器件）
py312 _scratch\t22_closurefix\identity_check.py

# S1：产品默认（常驻 worker）两态 + 低内存（worker 停用）
$env:ZIMAGE_DEV_WORKER='1'; py312 webui.py --host 127.0.0.1 --port 18810
py312 _scratch\t22_closurefix\run_jobs.py --port 18810 --jobs s1a,s1b
py312 _scratch\t22_closurefix\run_jobs.py --port 18810 --jobs s1c,s1d

# S2：回退 vae_worker + 冷建（先把产物改名拿走）
$env:ZIMAGE_DEV_WORKER='0'; $env:ZIMAGE_VAE_ONESHOT='1'; py312 webui.py --host 127.0.0.1 --port 18811
py312 _scratch\t22_closurefix\run_jobs.py --port 18811 --jobs s2a

# 重建闭合（从原始锚点）+ 证据汇总
py312 _scratch\t22_closurefix\closure_gate.py --jobs s1a:20261020-211058-ee20f1 ... --control old1606 --json webui\evidence\t22_closurefix\closure_gate.json
py312 _scratch\t22_closurefix\make_evidence.py
```

证据文件：`webui/evidence/t22_closurefix/{s1a,s1b,s1c,s1d,s1a2,s1c2,s2a}.json`（产品 API 原始返回）、
`closure_gate.json`（重建）、`summary.json`（汇总，含每个 bucket 的秒数与字节门）。

---

## 12. 后人怎么跑（五个工具的用法、读法与退出码）

全部 py312，都在 `_scratch/t22_closurefix/`；**前两个不占器件**，可以随时重跑。

### 12.1 `identity_check.py` —— 纯算术，秒级，无器件

```powershell
py312 _scratch\t22_closurefix\identity_check.py
```

验证两条恒等式（改桶表/改归属**前后各跑一次**）：桥拆分（`import + startup == 旧 startup`，
且 tail 仍按整段 `launch→startup_done` 扣 ⇒ 桶和逐位不变）与冷建搬家
（`startup + prewarm_pack == 旧 startup`）。顺带断言 19 个 key 不重不漏、
`timing_closure` 每个桶只加一次。
**读法**：最后一行 `IDENTITY PASS (0 failed)`；`exit 0` 才算过。
第 [1] 段还会打印反例（"只扣新 startup 会让 tail 多 1.710 s"）——那是故意留的，说明为什么 tail 不能改。

### 12.2 `closure_gate.py` —— 从原始锚点重建闭合，无器件

```powershell
py312 _scratch\t22_closurefix\closure_gate.py --jobs s1a:20261020-211058-ee20f1 s1b:... \
       --control old1606 --json webui\evidence\t22_closurefix\closure_gate.json
```

输入是 `label:job_id`（job 目录在 `webui/jobs/<id>`）。它**不读 runner 的桶字典**：
`bridge_import_s/startup_s` 从 `device_launch_tp/first_event_tp/startup_done_tp` 重算，
并用桥日志里的 `@@WEBUI@@` 事件作独立副本核对；`device_tail_s = device_wall −(import+startup)− loop`；
`vae_*` 从 `vae_p0.json` 重算；`vae_spawn_s = vae_child_wall_s − 子进程 total_s`；
`worker_drop_s` 是 kill 处直接测的量（只能读）。
**读法**：首行 `19 buckets`（证明用的是 runner 的表）；每 job 列出 `read`（单阶段测量：
queue/prepare/textenc/schedule/retry/gate_second/post/worker_drop）与 `derived`（重算）；
`[runner sum] ... d +0.000` 是与 runner 自己封账的差；`--control` 里的 job **只报告不判分**
（历史 job 用，如 16:06）。
**退出码**：`0` = 所有 graded job ≤1 s；`2` = 有 job 超 1 s。判 FAIL 时先看是哪条路径、
再看 `d` 是否非零（非零 = 重建与封账不一致，才是工具问题）。

### 12.3 `run_jobs.py` —— **占器件**，起服务后提交真实 job

```powershell
# S1 产品默认（常驻 worker + NPU VAE 链）
$env:ZIMAGE_DEV_WORKER='1'; py312 webui.py --host 127.0.0.1 --port 18810
py312 _scratch\t22_closurefix\run_jobs.py --port 18810 --jobs s1a2,s1c2

# S2 回退 vae_worker + 冷建：先把 1.99 GB 产物改名拿走，再起服务
Rename-Item webui\cache\wpack\wpack_l30_v2.bin      wpack_l30_v2.bin.bak
Rename-Item webui\cache\wpack\wpack_l30_v2.bin.json wpack_l30_v2.bin.json.bak
$env:ZIMAGE_DEV_WORKER='0'; $env:ZIMAGE_VAE_ONESHOT='1'; py312 webui.py --host 127.0.0.1 --port 18811
py312 _scratch\t22_closurefix\run_jobs.py --port 18811 --jobs s2a
```

预设见文件顶部 `SPECS`（`s1a` p10 默认、`s1b` p10=off、`s1c/s1c2` off、`s1d` low 模式、
`s1a2` 默认 on、`s2a` 回退+冷建）。结果 JSON 落 `webui/evidence/t22_closurefix/<label>.json`。
**两个坑（都踩过）**：① 产品默认的 p10 A/B 必须**同一个 server 进程**里连跑，且两条都
`memory_mode="fast"` —— 否则第二个 job 会被内存门翻成 low（T22-WORKERVAE §6），A/B 就不同路径了；
② 冷建会重写 2 GB 产物，跑完核对 `sha256 == da2f45a6…`，相等再删 `.bak`（本轮就是这么做的）。
**退出码**：`0` = 全部 job `done` 且 `closure.pass`；`2` = 有 job 不 done 或闭合 FAIL（不影响已落盘的 JSON）。

### 12.4 `make_evidence.py` —— 无器件，把 harness 记录 + 重建结果合成 `summary.json`

```powershell
py312 _scratch\t22_closurefix\make_evidence.py
```

改 job 列表在文件顶部 `JOBS` / `PATH_OF`。它逐 job 输出 total/sum/unaccounted、每个桶、
latents/PNG sha、dispatch/retry、p10 溯源（`p10_image`），以及
**`unaccounted_without_worker_drop_s`**（把该 job 换回 18 桶表的数字——s1d 就是这么得到 +0.6366 的）。
是否"封账时就带 worker_drop 桶"由 `closure.buckets` 的 key 集合判断，不看日历。

### 12.5 `rig_decompose.py` —— 无器件，复核 §7 的 §5-rig 分解

```powershell
py312 _scratch\t22_closurefix\rig_decompose.py
```

把 `_scratch/attn_impl/proj/prod/*.wall.txt`（PowerShell 墙钟）与桥日志的
`first`/`driver_done` 事件对齐，输出 `bridge wall − 进程内 span`（= import + teardown + 外壳）
与 `vae wall − 子进程 total_s`（= py312 spawn）。两列相加应等于 PROD-REPORT §3 的
4.58/5.94/4.98/5.81 —— 用来证明"残余就是这两段"，也用来提醒那列是**上界**。

### 12.6 一条新窗口的完整验收顺序

1. `identity_check`（改代码前）→ 2. `t3d_baseline --check` →
3. 取锁 `_scratch/device.lock`（owner=t22_closurefix，pid 写真进程）→
4. S1 起服务、`run_jobs s1a2,s1c2`（要 16:06 形状再加 `s1d`）→
5. 停 S1、改名产物、起 S2、`run_jobs s2a` → 6. 停服务、核产物 sha、放锁 →
7. `closure_gate` + `make_evidence` → 8. `t3d_baseline --check`（改后）→ 9. `identity_check`。

**收线判据**：`closure_gate` exit 0、`summary.json` 里 7 个 job 的 latents/PNG 与 §0 表一致、
`t3d_baseline` 前后全绿。任一条不过就不收线，把 FAIL 的 job id 与 `d` 写进报告。

