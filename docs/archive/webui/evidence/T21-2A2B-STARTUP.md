# T21 2a+2b 第一梯队：TE 常驻化 + /api/status 记忆化

**一句话结论**：2b-a（/api/status 记忆化）**成立**，/api/status 中位 77.0 ms → 29.9 ms（−61 %），
语义经 touch/删除/恢复/目录增删四类失效测试验证。2a-a（TE 常驻）**逐字节门通过，但在本机是净亏**
（省 10.4 s 文本编码，却因 6.74 GB Vulkan 常驻内存把 auto 内存模式挤到 --no-cache-all，器件循环
25.8 s → 59.6 s），按止损线②**停下**，代码保留为 opt-in 且**默认关闭**，不计入交付速度。
**本单元不引速度：NOT SHIPPABLE 的门禁里有一项（gate 模式计时闭合）是改动前就存在的缺陷**。

锚点：2026-10-19T08:44 ~ 09:20 +08:00；git HEAD 起点 272a5d2；隔离端口 7870；py312 = ryzen-ai-1.8.0；
器件：AMD XDNA2 / open IRON+XRT，128×128 (M=96)，8 步；机器双态（见 §6）。

---

## 1. 预测 vs 实测

预测写在实现之前（08:52 前的报告草稿，见 git 历史里本文件的首版）。

| 指标 | 预测 | 实测 | 结论 |
|---|---|---|---|
| /api/status 中位延迟 | 20-30 ms | **29.9 ms**（09:2x，5 连打；同树对照：未记忆化工作量 49 ms） | ✅ 命中 |
| 热态 textenc 桶 | 2.0-3.0 s | **0.47-0.75 s**（09:0x-09:1x） | ✅ 但比预测好 4 倍：新 ctx 初始化只要 ~0.1 s，我按 1-2 s 估的 |
| 热态单图 job 总时长 | 60-62 s | **97.6 s**（09:00:24 job，低内存模式） | ❌ 预测错，且错在主因 |
| 常驻 TE 后 commit | 12.5-13.5 GB | **7.9-8.2 GB** | ❌ 预测错：我以为 GGUF 是 mmap 文件页，实际 Vulkan 后端把 8.05 GB 拷进设备缓冲，吃 6.74 GB commit |
| 器件循环不受影响 | 25.6-25.8 s | **59.6 s**（触发 --no-cache-all） | ❌ 这是本单元最大的一笔账，预测里完全没出现 |

**预测失败点（诚实记录）**：我把"TE 常驻"当成纯时间变量，没有先算内存账；实际它同时是一个内存变量，
而内存变量在本机直接决定器件路径（fast/low）。这也正是止损线②存在的理由。

---

## 2. U1 基线（改动前，同一 session）

| 项 | 值 | 读数时间 |
|---|---|---|
| ① /api/status 5 连打 | 97.9 / 77.0 / 79.2 / 57.5 / 69.4 ms，**中位 77.0 ms**，15,427 B | 08:47 |
| ② 128×128 单图 job（首跑，VAE 冷） | total 83.98 s；textenc 11.02（load 7.3 / decode 1.96）；startup 30.74；loop 25.59；device 14.91；staging 2.03；vae_load 15.42；closure **+0.048 s PASS** | 08:48:44 |
| ② 128×128 单图 job（第二次，VAE 热） | total **68.89 s**；textenc **11.16**（load 7.3 / decode 1.96）；startup 30.78；loop 25.77；closure **+0.072 s PASS**；latents 9f86aee9…，pe 90514d08…，PNG 104ff564… | 08:50:18 |
| ③ te_worker 重载 GGUF 的可观测时间 | 驱动自报 `model loaded in 7.3s`（跨 6 次运行 3.4-11.6 s）；textenc 桶 11.02-11.16 s；减去 load 与 decode(1.96) 后，进程/DLL 冷启 ≈ **1.9 s** | 08:48-08:50 |
| 内存（job 开始时） | avail 14.34 / commit 13.74 / load 54 % → auto=fast | 08:48 |

基线三个数就是本单元要打的靶：**11.16 s 的文本编码里 9.2 s 是重复载入，30.78 s 的 startup 是 2a-b 的靶子。**

---

## 3. U2（2b-a）状态记忆化 —— 成立，已交付

### 改动（commit 3057668）
1. `webui/config.py`：`find_te_gguf()` 记忆化。失效条件 = **每个候选文件的 (mtime_ns,size)** +
   **每个搜索目录自身的 mtime**（目录里新建/删除文件会改目录 mtime）+ `ZIMAGE_TE_GGUF` 每次重读 +
   30 s TTL 兜底（TTL 只兜"已存在的 dsh-* 目录里新建 zimage-te/ 子目录"这一种目录戳看不到的情况）。
   新增 `find_te_gguf(refresh=True)` 与 `find_te_gguf_stats()`。
2. `webui.py`：`cache.status()` / `cache.shapes_status()` 按 **stat stamp** 记忆化，stamp =
   每个 profile 目录的 `manifest.json` (mtime_ns,size) + manifest 里**每个文件**的 (mtime_ns,size) +
   profile 目录集合。**cache.py 一行未改**（stamp 与 memo 都在 webui.py，避免动只读引用文件）。
3. `/api/status` 语义不变，仅**新增** `status_cache`（hits/misses/stamp_entries/te_gguf 计数）与
   `te.resident` / `te.resident_enabled` 两个诊断字段。

### 为什么(实测，不是猜)
未记忆化时每次 /api/status 的纯工作：`cache.status()` 12-18 ms + `cache.shapes_status()` 37-50 ms
= **60 个缓存文件 / 6.79 MB 全量 sha256**；`find_te_gguf()` 在本机只要 0.2 ms（webui/models 命中早退），
**任务书里"全盘 glob"的假设在本机不成立**——如实记录，不虚报。
记忆化后同一棵树上的对照：cache 两个函数仍是 49-69 ms，HTTP 端到端 29.9 ms（差额是 stat 与 HTTP 开销）。

### 验证（4 类失效，全部 PASS；每次改动后都还原）
| 测试 | 期望 | 实测 |
|---|---|---|
| 5 连打 | 首调 miss、后续 hit | hits +2/次，misses 不变 |
| `os.utime` touch 一个缓存文件 | 下一次调用必须失效，值不变（manifest 只看 size+hash） | misses 4→6（两个 memo 条目各 +1），值仍 ready；再下一次 hit +2 |
| 删除缓存文件 `64x256/t7_p5.xclbin` | 下一次调用必须反映 | `cached=False, "cached file missing: t7_p5.xclbin"` |
| 从源码恢复该文件 | 必须翻回 | `cached=True, "ready"` |
| webui/models 里新建/删除任意文件 | GGUF 搜索结果失效 | te misses 2→3→4，结果仍是同一个 GGUF |

---

## 4. U3（2a-a）TE 常驻 —— pe 门通过，但止损线②触发

### 改动（commit 797982a，**默认关闭**）
- `webui/te_worker.py`：新增 `--serve` 常驻模式（协议行前缀 `@@TE@@ `，非协议行原样当日志）。
  模型只载入一次；**每个请求新建一个 llama context**（用完即 free），因此请求序列 = 一次性路径在
  "共享模型载入"之后的**同一串调用**——这是 pe 逐字节相同的构造性理由（不是事后比对出来的）。
- `webui/textenc.py`：常驻客户端（`ZIMAGE_TE_RESIDENT`，默认 **0**）。保留 Vulkan→CPU 降级与
  终身记忆 `_GPU_OK`；worker 死亡/超时 → 丢弃并在同一后端重建一次（崩溃自愈）；取消 → 杀 worker。
  `timing["textenc_s"]` 仍是本步**全墙钟**（含载入），另加两个**信息性**键 `textenc_boot_s` /
  `textenc_reused`（不进 Σ桶，不重复计入）。
- `webui/runner.py`：只加了上述信息性键与一行日志；**门禁与桶逻辑一行未动**。
- `webui.py`：/api/status 加 `te.resident`。

### 门禁 U4-① pe 逐字节：**PASS**（两次跑，共 6 组）
| 路径 | prompt 1 pe | prompt 2 pe | 时间 |
|---|---|---|---|
| 一次性（改动前 job 084844/085018） | 90514d08… | — | 08:48 |
| 一次性（改动后同树） | 90514d089ec4b356 | 620cca8f0ba87fa6 | 09:14 |
| 常驻-首调（spawn+载入） | 90514d089ec4b356 | — | 09:14 |
| 常驻-热调 ×2 | 90514d089ec4b356 | 620cca8f0ba87fa6 / 同 | 09:14 |

热调 0.47-0.75 s（decode 0.32-0.52 s），冷调 boot 3.7-14.1 s。**逐字节一致，且与改动前交付物一致。**

### 止损线②：内存
| 读数 | avail | commit | load |
|---|---|---|---|
| 无 TE worker（09:06 前重启后） | 17.90 GB | 14.12 GB | 42 % |
| Vulkan TE 常驻 | 9.13-8.91 GB | **8.16-7.93 GB** | 70-97 % |
| 隔离探针（独立进程，专属测量） | −8.32 GB | **−6.74 GB** | 70→97 % |
| 该 worker 自身 | ws 8.00 GB | private **9.02 GB** | — |

**后果（实测，不是推断）**：下一个 job 的 auto 选择变成
`low: commit 7.9 GB < 9.0 GB; using --no-cache-all`（job 090024，09:00:24）：
device_s 15.53 s 与基线 14.91 s 同量级、dispatches 1440 完全相同，但 **staging 2.4 s、loop 59.64 s**
（基线 25.59/25.77 s）→ total **97.59 s**，比基线 68.89 s **慢 28.7 s**。省 10.4 s、亏 34 s。
对照证据：job 085832（它的模式在 worker 出现前就已选定为 fast；起点 commit 14.45）在 TE 常驻下
loop 28.53 s、device_s 15.47 s、staging 2.27 s；而 low 模式 job 090024 的 device_s 15.53 s、staging 2.41 s
**器件侧一模一样**，多出来的 32 s 全在每步"宿主机重读权重"里（不进 device_s/staging_s，只进 loop 墙钟）—— 说明
**fast 路径与常驻 TE 共存是可行的，卡的是 job 开始时的 9.0 GB 策略阈值，不是硬件墙**。

### 处理
按止损线②**停下**：不改阈值、不改内存策略、不动 config.py:202。代码保留为 opt-in（默认关闭），
默认交付路径 = 一次性（逐字节与 272a5d2 相同）。默认路径的 9 个 job 计时证明它没变（§6）。

---

## 5. 门禁五项（全过才准引速度 —— **没有全过**）

| # | 门 | 结果 | 证据 |
|---|---|---|---|
| ① | 同 prompt+seed 常驻 vs 冷载 pe 逐字节 | **PASS** | §4 表，两个 prompt × 3 路径，sha256 全等 |
| ② | seed 复现门（gate 模式第二遍 latents 逐字节） | **PASS**（复现部分） | job 20261019-090737-30d768：latents 9f86aee9… == 9f86aee9…，pe_equal True，gate.status PASS |
| ③ | 批语义：batch 2 == 各自单图 latents | **PASS** | job b2 20261019-090952-fb2b20 (batch 2, seed 7,8) vs s7 091139 / s8 091323：`np.squeeze` 后 **np.array_equal True**，展平 max|d| = 0（batch-2 落盘形状 (2,1,16,16,16)，单图 (1,16,16,16)，只差一个单例轴） |
| ④ | 计时闭合 Σ桶 == total ±1 s | **trial PASS / gate FAIL** | trial 三个 job：+0.07 / +0.08 / +0.04 s；gate job：**−3.54 s** |
| ⑤ | t3d_baseline --check | **PASS** | 改动前 08:46 `ALL REFERENCE BYTES UNCHANGED`；改动后 09:2x 同 |

**④ 的 gate 模式失败是改动前就存在的缺陷，不是本单元引入**（本单元对 runner.py 的 diff 只有信息性键）：
runner 先 `job["timing"].update(pass1["timing"])`（用第二遍覆盖第一遍的桶），**然后**又
`gate_second_s = pass1["pass_s"]`（整遍再记一次）→ **第二遍被计两次、第一遍一次没计**。
算术复核：sum − 一次性的桶 = 137.73 ≈ 2 × 第二遍 68.88 = 137.75；残差 = 第一遍 − 第二遍 − 0.05
= 65.30 − 68.88 − 0.05 = **−3.63**，实测 −3.54。272a5d2 的 runner.py:424 就是这行，逐字未改。
**建议单列一个 10 行修复单元**（pass0 桶 + gate_second_s + queue/prepare/post），不在本单夹带。

**因此：NOT SHIPPABLE，本报告不引任何速度数字。**（2b-a 的 61 % 是 CPU 侧、与器件无关的实测，已注明。）

---

## 6. A/B（同 session，机器双态如实）

### 双态
| 时段 | startup 桶 | 循环 | llama 载入 | 说明 |
|---|---|---|---|---|
| 08:48-08:50 | 30.7-30.8 s | 25.6-25.8 s | 7.3 s | 快态 |
| 08:58-09:13 | 29.5-42.0 s | 28.1-28.4 s（batch1）/ 53.8 s（batch2） | 6.4-10.5 s | 慢态 |
| 更早 07:55-08:10（改动前） | 40-46 s | 27.0 / 28.6 / 55.3 / **168.5** s | 6.6-10.1 s | 慢态，含一次 6.6× 抖动 |

### /api/status（本单元唯一被引用的数字）
| | 5 连打（ms） | 中位 | 稳态（2-5 次） |
|---|---|---|---|
| U1 基线（08:47，改动前树） | 97.9 / 77.0 / 79.2 / 57.5 / 69.4 | **77.0** | 57.5-79.2 |
| 定稿树（09:2x） | 41.2 / 29.9 / 23.4 / 28.9 / 32.5 | **29.9** | 23.4-32.5 |
| 同树对照：未记忆化的两个函数之和 | — | 49-69 ms | — |

→ **−61 %**，且不受机器双态影响（纯 CPU/IO 侧）。

### 单图 job（trial, 128×128, seed 7, 8 步）
| | total | textenc | startup | loop | 模式 |
|---|---|---|---|---|---|
| U1 基线（快态） | 68.89 s | 11.16 | 30.78 | 25.77 | fast |
| 定稿树（慢态，09:11/09:13） | 81.66 / 104.21 s | 10.09 / 11.23 | 41.99 / 40.83 | 28.41 / 28.12 | fast |
| 定稿树 + 常驻 TE（慢态，09:00） | 97.59 s | **0.62** | 36.37 | **59.64** | **low** |

**跨态不可比，故不引速度**：textenc 桶在默认路径上保持 10.1-11.2 s（与基线同区间，本就该不变）；
定稿树多出来的 13-35 s 全部落在 startup/loop 的机器慢态上（08:58 之后的 job 全部 startup 36-42 s）。

---

## 7. 精确剩余量

| 单元 | 状态 | 改动量 |
|---|---|---|
| 2b-a /api/status 记忆化 | **完成并交付** | config.py + webui.py，1 commit |
| 2a-a TE 常驻 | **建成、门禁①过、默认关闭、净亏证伪** | te_worker.py/textenc.py/runner.py/webui.py，1 commit |
| **2b-b 载荷 RAM 预置换** | **未动（0 行）** | — |
| **2a-b 器件 worker 保温** | **未动（0 行）** | — |
| gate 模式计时闭合（改动前缺陷） | **未修，已定位到 272a5d2:runner.py:424** | 建议 10 行修复单元 |

### 第二梯队纸面建议（只分析，未实施）——6.74 GB 与 --cache-all/BO 常驻的内存冲突怎么破

内存账（本机实测，2026-10-19T09:06-09:16）：
- 空闲 commit 14.1-14.6 GB（无 TE）；器件 fast 路径需要 **~6 GB private commit**（W 1.85 + BG/B3 3.71）；
  config 的策略阈值 **9.0 GB**（留 3 GB 余量）。
- 常驻 bf16 **Vulkan** TE：**−6.74 GB commit / −8.32 GB 物理**（因为权重被拷进 iGPU 设备缓冲，
  不是 mmap 文件页）→ job 起点 7.9-8.2 GB < 9.0 → low。
- **缺口 = 只需 +1.0-1.1 GB commit 就能回到 fast。** 6.74 + 3.71 = 10.45 GB 正是冲突的全部来源。

| 候选 | commit 账 | 物理 RAM | textenc（冷/热） | 器件路径 | 前提/代价 |
|---|---|---|---|---|---|
| **(a) TE 常驻换 CPU 后端**（已实测） | **−0.66 GB ≈ 0** | −7.58 GB（文件页，可回收） | **0.98 / 0.55 s** | 保持 fast | **pe 变 e6642367…**（文档：cosine 1.000000 / max abs 0.37）→ 需产品决策 + 图像并排 A/B；两条 pass 仍互相逐字节一致 |
| (b) bf16 换 Q8_0 常驻 Vulkan | ~3.6 GB（纸面，按 4.3/8.05 缩放） | ~4.5 GB | ~1.0 / ~0.6 s | fast（commit≈10.7） | 新权重产物 + 重新基线；量化噪声未测 |
| (c) bf16 Vulkan 每 job 载入/卸载 | 0 | 0 | 3.4-13.7 / — | fast | 净收益只剩进程冷启 ~1.9 s（job −2~4 s） |
| (d) 器件侧 3.71-3.98 GB 转 BO 常驻（2a-b） | 6.74+1.85+0.3 ≈ **8.9 GB** | 同（UMA） | 0.47 | fast（若 BO 不吃 pagefile） | **只有这条路能保持交付的 bf16 pe 逐字节不变**；但本机是 UMA，BO 仍占物理 RAM，赌的是 commit 记账；属 2a-b 的测量项 |
| (e) 调低 9.0 GB 阈值到 ~7.5 | — | — | 0.47 | fast（有 MemoryFailure→low 兜底） | 已有实证：job 085832 在 TE 常驻下 fast 跑完 28.53 s；但兜底路径在 TE 常驻下未压测过，不建议先做 |

**推荐顺序**：(a) 先做图像级 A/B（成本最低、收益最大：job 68.9→~58.5 s，−15 %，且彻底去掉 GPU 依赖与
Vulkan→CPU 降级分支）；(d) 若要保住交付的 bf16 pe，就靠 2a-b 的 BO 常驻把 3.71 GB 移出 commit 账；
(c) 是零风险的兜底小胜。预期收益区间：**(a)/(d) −10.4 s/job（−15 %），(c) −2~4 s/job（−5 %）**。
注意 (a) 是**换编码器**而不是启动优化，不能混在本单元里做。

---

## 8. 改动文件与 commit

| commit | 内容 | 文件 |
|---|---|---|
| `3057668` | t21_2b: /api/status 记忆化 | `webui/config.py`, `webui.py` |
| `797982a` | t21_2a: 常驻 TE（opt-in，默认关） | `webui/te_worker.py`, `webui/textenc.py`, `webui/runner.py`, `webui.py` |
| 本文件 | 证据 | `webui/evidence/T21-2A2B-STARTUP.md` |

未改：`webui/cache.py`（一行未动）、`webui/bridge.py`、`webui/vae_*.py`、`_scratch/` 下任何文件。
复现脚本（gitignore 内，不进树）：`webui/_selftest/t21_http.py`（状态/作业驱动）、`t21_memo_check.py`
（失效四测）、`t21_te_gate.py`（pe 门）、`t21_gates.py`（复现/批语义/闭合）、`t21_te_mem.py`（常驻内存）、
`t21_te_cpu.py` / `t21_te_cpu_resident.py`（第二梯队纸面数据）、`t21_audit_gate.py`（gate 闭合残差）。

## 9. 复现（隔离端口 7870，收尾杀进程）

```
py312 webui.py --port 7870                       # 定稿树（常驻 TE 默认关）
py312 webui/_selftest/t21_http.py 7870 status 5  # → §6 表
py312 webui/_selftest/t21_memo_check.py 7870     # → §3 表
py312 webui/_selftest/t21_te_gate.py             # → §4 pe 门（resident=True 显式 opt-in）
py312 webui/_selftest/t21_gates.py 7870          # → §5 ②③④
py312 _scratch/image/t3d_baseline.py --check     # → §5 ⑤
ZIMAGE_TE_RESIDENT=1 后才走常驻路径（默认一次性）
```
