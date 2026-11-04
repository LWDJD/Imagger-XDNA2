# T22-PRODLARGE — 产品面为 128²–2048² 全谱做好准备（无器件、无新内核）

**范围**：产品侧 6 个文件 `webui/config.py shapes.py build_shape.py cache.py
prebuild.py static/index.html`。`runner.py / bridge.py / vae_*.py` 未动，
`_scratch` 未动，未建内核，**未占器件**（唯一一次"器件"动作是 WebUI 服务本身
起在 127.0.0.1:7873 上做 API 取证）。

**时间**：2026-10-20 10:15–10:50 (+08:00)。**HEAD 起点**：`fa67fef`。

## 0. 结论

| 目标 | 状态 | 证据 |
|---|---|---|
| ① H/W 限值与 M 算术扩到 2048，未建尺寸 `ready=false`+原因 | **达成** | MAX_HW 2048 / MAX_M 16416；2048²→M=16416、1536²→9248、1024²→4128；`/api/shapes` 10 行，4 个未建尺寸各带 512 规范原因 |
| ② 尺寸预算面板（M/预估 dispatch/预估机时，T22-BUDGET 口径） | **达成** | 面板 4 个数字 + 9 行明细；18 个代表尺寸 × 10 列与 T22-BUDGET.md §2 **逐格一致**；dispatch 与 5 次已交付运行 + 2 次 T22 门取证一致 |
| ③ prebuild/未建尺寸文案与 512² 同规范 | **达成** | 1024²/2048² 由 400 泛化句 → 200 + 一句 512 规范句（含精确 M/图 token/L1）；512² 句**逐字节冻结** |
| ④ 缓存/manifest 对全谱的可扩展性（LRU/磁盘预算） | **达成（并修了 2 个真问题）** | LRU 改为"磁盘清单驱动"；死条目被剪；非 profile 目录永不计数/淘汰；`cache.usage()` 实测 6 profiles / 5,863,889 B / 0.55% |

**NOT SHIPPABLE**：本单元**没有**任何新尺寸的器件构建、内核或门禁。面板与文案
都在说"本树未构建、未过门、不可引用为速度"。真正可跑的最大的仍是交付的
256×256（M=288）；512² 以上只有纸面预算。

## 1. 预测 vs 实测 vs 结论

预测写在**任何代码改动之前**（10:17），下表是收尾时的实测。

| # | 预测 | 实测 | 结论 |
|---|---|---|---|
| P1 | `MAX_HW`→2048，`MAX_M`→16416 | `/api/shapes.limits` = h/w 2048、m_max 16416 | ✅ |
| P2 | 2048²→16416、1536²→9248、1024²→4128、1024×512→2080、512²→1056、256×384→416 | 全部命中（`t22_prod_budget.py` 18 行表；2048²/1024²/1536² 见 §2 表） | ✅ |
| P3 | `/api/shapes` 8→10 行、presets 8→10 | 10 / 10（新增 1024×1024、2048×2048） | ✅ |
| P4 | 预算与 T22-BUDGET.md §2 的 18 行逐位一致 | 180 格全部通过（108 个数值格在打印精度内，72 个整数格精确） | ✅ |
| P5 | 2048² dispatch：proj 684 / qk+pv 88236 / FFN 171（每层） | 684 / 88236 / 171；合计 89,091/层 = **2,672,730/步** = 21,381,840/8 步 | ✅（并更正了"每步"口径，见 §5.2） |
| P6 | 8 步机时：2048² 14.93 h、1024² 1.09 h、512² 0.11 h | 14.93 / 1.09 / 0.11 h；2048² 每步 6720.4 s | ✅ |
| P7 | 六尺寸 ready/M/images 不变 | 6/6 ready=true、M=96/96/128/128/160/288、镜像名逐个不变（API 门 a 段 18 条断言） | ✅ |
| P8 | 512² 理由逐字节不变 | `/api/status.resolutions[512].reason` 与改前**完全相同**（b 段断言） | ✅ |
| P9 | 1024² L1=736,768 B；2048² L1=2,899,456 B | 736,768 / 2,899,456（`t7_ip` 不等式，Mq=16） | ✅ |
| P10 | `/api/jobs` 512²→400；prebuild 1024²/2048²→200+`ok=false` | 400 / 200+ok=false / 200+ok=false（改前：1024² 是 200+英文泛化句；**2048² 是 400 "each side must be 64..1024"**） | ✅ |
| P11 | 缓存 6 profiles / 5,863,889 B / 0.55% / 0 死条目；本轮不删目录 | 实测完全相同；无目录被删 | ✅ |
| P12 | 6 个非 profile 目录不计数不被淘汰 | `non_profile_dirs` 列出 `128, build_logs, lru.json, npu_t21, numba, numba_fresh_t16, schedules`；沙盒里 12 profiles 淘汰到 10 时 5 个设施目录原样 | ✅ |
| P13 | 128² job 复算：m=96、shape=128x128、cache 8 文件、created 2026-10-18T19:16:19 | 全部命中，另加一条：**dispatch 模型复算 = 1440 = 该 job 的 calls.json 实测值** | ✅ |
| P14 | `node --check` 通过；面板数字全来自服务端 | 通过；前端不再有任何 M/dispatch/ms 计算（收尾把占位 M 也删了） | ✅ |
| P15 | `t3d_baseline --check` 前后全绿 | 改动前 10:14、改动后 10:41 均 `ALL REFERENCE BYTES UNCHANGED` | ✅ |

**16/16 预测全部命中，0 处需要回退。**

## 2. 目标①：全谱算术与"未建就说不"

`shapes.py`：`MIN_HW=64`、`MAX_HW=2048`、`MAX_M=32+16384=16416`、`FFN_ROWS=96`。
M 公式不变（`M = 32 + ceil(((H/16)(W/16))/32)×32`）：

| H×W | 图 token | 补齐 | M=SEQ | 今天的 ip | 现算 L1 (Mq=16) |
|---|---|---|---|---|---|
| 128×128 | 64 | 64 | 96 | delivered (Mq=96) | 62,976（Mq=96） |
| 256×256 | 256 | 256 | 288 | delivered (Mq=16) | 60,928 |
| 512×512 | 1024 | 1024 | 1056 | **none** | 196,096（冻结文案里写的是 83,456，见 §7.1） |
| 1024×512 | 2048 | 2048 | 2080 | none | 376,320 |
| 1024×1024 | 4096 | 4096 | 4128 | none | **736,768** |
| 1536×1536 | 9216 | 9216 | 9248 | none | 1,637,888 |
| 2048×2048 | 16384 | 16384 | 16416 | none | **2,899,456** |

`shapes.reason_text()` 分三个真实分支，绝不一句"缺 6 镜像"打天下：

1. **SEQ 已交付**（M∈96/128/160/288，如 96×128）：可复用该 M 的镜像，缺的是
   patch grid 的 perm 表 + host 常量（RoPE/SBG/image-token padding）+ 四门禁。
2. **今天的 ip 还能编**（如 128×64 → SEQ=64, Mq=64, L1 21,504 B ≤ 64 KB）：
   缺 1 次 `t18_attn_compile` + perm 表 + 四门禁。
3. **什么都编不了**：512 规范句（"…需按 key 分块 + 在线 softmax 的新内核
   （M=… 的 L1 预算 … B > 64 KB），本树未构建，不可用。"）。

`plan["missing"]` 同步按这三支重写（分支 1 不再谎称"6 device images"），
`plan["budget_t22"]` 出现在**每一行**（含已建尺寸），供面板对照。

`/api/shapes` 实测（2048×2048 行）：
`{"key":"2048x2048","m":16416,"ready":false,"reason":"2048² 需新内核设计…2,899,456 B > 64 KB…","images":["FFN","4 projections","IP"],"cached":false,…}`

## 3. 目标②：尺寸预算面板

新面板（左栏 H/W 之下）4 个大数字 + 最多 11 行明细，全部来自服务端
`budget_t22`：M(=SEQ)、预估 dispatch/步、预估机时/8 步、今天的 ip 能否承载；
明细含 attn 分项（max(计算,KV)+S/P+dispatch）、投影/FFN、dispatch 三分解、
state-8 屏障数、L1 vs 65,536 B、最长单次 dispatch vs 7.0 s 看门狗、缓存占用、
以及交付口径的线性 loop（带"256² 以上严重低估"的 caveat）。

**数据来源只有一个**：preset 行来自 `/api/shapes`；用户手输的尺寸走
`POST /api/shapes/prebuild`（去抖 350 ms；未建尺寸**不产生任何编译/写盘**，
已建尺寸是 manifest HIT）。前端不做算术，因此面板与 T22-BUDGET.md 不可能各自
漂移——这一点由 `t22_prod_budget.py` 的 180 格比对守住。

2048×2048 的面板实测值：M=16,416；dispatch **2,672,730**/步（8 步
21,381,840，屏障 40,800）；机时 **14.93 h**；L1 2,899,456 B；最长 dispatch
40.30 ms。

## 4. 目标③：文案与 512² 同规范

| 入口 | 改前 | 改后 |
|---|---|---|
| `/api/shapes` 512×512 `reason` | 英文泛化 "this shape has no gated device build…" | 512 规范句（与 `/api/status.resolutions` 同源同字节） |
| `/api/shapes` 1024²/2048² | 同上一句泛化 | 512 规范句 + 精确 L1 |
| `POST /api/shapes/prebuild` 1024² | 200，`error="1024x1024 (M=4128) 未构建: this shape has no gated device build in this tree; the delivered images are M=96 only"` | 200，`error="1024² 需新内核设计，暂不可用：M=4128（4096 image tokens + 32 caption）…（M=4128 的 L1 预算 736,768 B > 64 KB），本树未构建，不可用。 本树没有 M=4128 的编译配方 / artifact spec：build=true 也不会构建（未触发任何编译）。"` + `budget_t22` + `build_note` |
| `POST /api/shapes/prebuild` 2048² | **400** `{"ok":false,"error":"2048x2048: each side must be 64..1024"}` | 200 + 512 规范句（2,899,456 B） |
| `build_shape.py --height 1024 --width 1024` | `BuildError: 1024x1024 (M=4128): no compile recipe (only M=128/288 built)` | `1024x1024 (M=4128) 未构建：1024² 需新内核设计…本树的编译配方只有 M=128/160/288。` |
| prebuild 的 `error` | `"1024x1024 (M=4128) 未构建: <reason>"`（"未构建"重复、且不说 build=true 也没用） | 直接就是那句规范话，并追加"本树没有 M=… 的编译配方：build=true 也不会构建（未触发任何编译）" |

## 5. 目标④：缓存/manifest 的全谱可扩展性

### 5.1 改了什么（两个真问题）

1. **LRU 只认 lru.json**：手工拷贝 / 中断的构建 / 另一份 checkout 产生的
   profile 目录既不计入预算也永不被淘汰 → 磁盘预算可被绕过。现在
   `profile_dirs()` 扫 `webui/cache` 的 `{H}x{W}` 目录作为唯一清单，
   未知时间戳按"最旧"处理。
2. **死条目无界增长**：目录被删后 lru.json 的条目永远留着。现在 evict/clean
   都会剪枝，且 `lru.json` 原子写。

另外：非 profile 目录（`build_logs`、`numba*`、`schedules`、`npu_t21`、
T16 之前遗留的 `128`——README §13 的手跑命令还在用它）**永不计数、永不淘汰**；
`cache.usage()` 把 profiles/bytes/budget/死条目/非 profile 目录清单挂到每一行，
`/api/status`、`/api/cache`、`/api/shapes` 都能显示预算（1 s memo + 显式失效）。

### 5.2 实测

* 本树：6 profiles、5,863,889 B（5.59 MiB）= 1 GiB 预算的 **0.55%**、0 死条目。
  → 绑定约束是**数量上限 10**，不是字节预算；一个 profile ≈ 1 MB（镜像是
  xclbin/bin/perm，不是权重）。
* 沙盒（12 profiles、lru.json 只记 9 个、5 个设施目录）：淘汰 2 个 → 10 个；
  设施目录与保留项原样；lru 剪枝到 10 条。字节预算臂（3 MiB / 100 上限）：
  淘汰 11 个后 2,097,156 B ≤ 3 MiB。
* manifest 完整性：翻一个字节 → `cached hash changed: img.bin`（仍然拦得住）。

### 5.3 大尺寸下的成本（诚实记账）

`ensure_shape` 的 HIT 路径会对每个文件做 sha256；本树 6 个 profile ≈ 49 ms。
若将来一个 2048² profile 带 2 MB 的 perm 表，单次校验约 +8 ms；而
`/api/status` 的 stamp memo 把它摊成"每次 stamp 变化一次"。本单元**没有**
改完整性契约（不做 stat-only 快路径），因为那是用安全性换毫秒。

## 6. 门表

| # | 门 | 结果 | 证据 |
|---|---|---|---|
| 1 | `t22_prod_api.py gate before/after`（API 级，无器件） | **PASS 0 失败**（a 六尺寸 18 条、b 512 冻结、c schema 超集、d 全谱暴露、e 未建拒绝、f resolutions） | `evidence/t22_prodlarge/before|after/` |
| 2 | `t22_prod_ui.py`（node --check + 桩 DOM 真跑内联 JS） | **PASS**：3 条静态 + 18 条运行时断言 | 本文 §7 |
| 3 | `t22_prod_budget.py`（面板 vs T22-BUDGET.md） | **PASS**：18 行 × 10 列 = 180 格；§6 dispatch 3 行；5 次已交付运行 + 2 次 T22 取证 | 同上 |
| 4 | `t22_prod_cache.py`（LRU/预算/manifest） | **PASS 21 条** | 同上 |
| 5 | `t22_prod_job.py`（既有 job.json 复算） | **PASS 15 条**，含 dispatch 1440=实测 | `webui/jobs/20261019-231006-57af05/job.json` |
| 6 | `t3d_baseline.py --check` | **ALL REFERENCE BYTES UNCHANGED**（改前 10:14 / 改后 10:41） | 终端 |
| 7 | 改动范围 | 只有 6 个允许文件（`git log --name-only fa67fef..HEAD -- webui/`） | 7 个 commit |

测试脚本在 `webui/_selftest/`（与 T21 惯例一致，该目录被 .gitignore 忽略；
不提交编译产物）。复跑：

    py312 webui/_selftest/t22_prod_api.py  capture webui/evidence/t22_prodlarge/after
    py312 webui/_selftest/t22_prod_api.py  gate before after
    py312 webui/_selftest/t22_prod_ui.py   webui/evidence/t22_prodlarge/after
    py312 webui/_selftest/t22_prod_budget.py
    py312 webui/_selftest/t22_prod_cache.py
    py312 webui/_selftest/t22_prod_job.py
    py312 _scratch/image/t3d_baseline.py --check

（API/UI 门需要一个起在 7873 的 `py312 webui.py --port 7873`；其余纯离线。）

## 7. 诚实负结果 / 已知偏差

1. **config.RESOLUTIONS[512] 的 "83,456 B" 是错的（但它被冻结）**。按 t7_ip 的
   L1 不等式，83,456 B 是 SEQ=416（256×384）的值；M=1056 在 Mq=16 下是
   **196,096 B**。结论（>64 KB、不可用）不变，只有那个数出自另一行。该句被前端
   `SHAPE_512_REASON` 与 README 引用，本单元按零回归**逐字节冻结**；正确值以
   `budget_t22.t22_cur_ip_l1_bytes` 呈现（1024²=736,768、2048²=2,899,456 都是
   现算的）。改文案是一次独立的产品文案变更。**同样地，config.RESOLUTIONS[256]
   的理由是历史文本**（M=288 早已交付），只在 256×256 被反建时才会出现。
2. **T22-BUDGET.md §6 的 "total/step" 列其实是"每层"**。该列等于同表各"per
   layer"列之和，且 §7 的 dispatch 行要乘 30 才对得上。产品字段按**每步**给出
   （2048²：89,091/层 → 2,672,730/步），并用 5 次已交付运行 + 512² 一步门
   (13,530/步) 交叉验证。纸面表本身**未改**（不属于本单元范围），但引用时必须
   注意口径。
3. **交付口径的 loop 外推（27.7 s × M/96）线性放大**，在 256² 以上严重低估
   （2048² 给 4,736.7 s，T22 模型给 53,760 s）。面板只在**已建尺寸**的信息行显示
   它，明细行带 caveat；真正的机时列是 T22 模型。
4. **`/api/status` 变重**：15,793 B → 43,938 B（每行多 `budget_t22` 与
   `cache_usage`），本机实测 3 连打 57.3 / 44.3 / 30.2 ms（中位 44.3 ms，
   T21 记录的同口径中位 29.9 ms）。仍在 5 s 轮询预算内（<1% 占空），stamp memo
   命中正常（hits 8→10→12，misses 恒为 2）。若日后要削，优先削每行重复的
   `cache_usage`（≈3.5 KB）。
5. **共享机器内存压力**：收尾测延迟时本会话的 Node 进程遇到一次
   `VirtualAlloc failed`（整机 commit 紧绷）；未影响任何一次门禁（门禁都是短命
   子进程）。
6. **没有做的事**（见 §8）。

## 8. 精确剩余量

| 项 | 还差什么 | 量级 |
|---|---|---|
| 512² 真跑通 | T22 分块 qk/pv 镜像 + 35 个 perm 表 + host 常量 + 四门禁（模型门/逐核审计/run-to-run 字节/首图对比） | 设计已纸面（T22-LARGE-DESIGN.md），未编译；T22 一步门只验到 latents 不一致（max\|d\| 0.0309），**未过门** |
| 1024²–2048² | 同上 + ip MFU 40→400 GFLOPS 或 KV-outer 调度（T22 §7） | 2048² 今日速率 14.93 h/8 步 |
| 未建但可编的 SEQ（257..304 图 token 等） | 每个 SEQ 1 次 ip 编译 + perm 表 + 四门禁 | 单尺寸 |
| 缓存数量上限 | 10 个 profile 对全谱偏紧；按今天 1 MB/profile，可安全放到 ~100（100 MB） | 一次配置变更 + 一次实测 |
| 512 文案的 83,456 | 改成 196,096（或改成现算） | 一行文案 + 一处测试期望 |
| /api/status 瘦身 | 把 `budget_t22`/`cache_usage` 挪到 `/api/shapes` 专属 | 需改 `webui.py`（**本单元禁动**） |
