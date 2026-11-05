# T22 文本编码缓存（t22_te_cache）：同 prompt 复跑 textenc 8.57 s → 0.007 s

**一句话结论**：TE 结果按 **prompt 文本 + 实际后端 + 编码器指纹** 缓存（磁盘
webui/cache/te/ + 内存层，64 条 / 256 MiB 双上限、LRU 淘汰、逐字节校验），命中时把
首跑 pe_p0.npz 的**原始字节**复制进 job；schedule.npz 也从一份字节稳定的缓存复制。
7873 四态门禁 **33 checks / 0 fail，全 PASS**：冷 job textenc 8.57 s，同 prompt 第二 job **0.0068 s
（内存）/ 0.0104 s（磁盘）**，pe 文件 437634b6…、schedule f80e3699… 逐字节相同，
latents 9f86aee9…；换 prompt 7.83 s 正常重算且 pe 不同；3 条上限灌 5 条淘汰最旧 2 条、
LRU 命中项幸存；closure +0.029…+0.076 s；t3d_baseline --check 前后 ALL REFERENCE
BYTES UNCHANGED。

门禁端口/缓存目录：**7873 + webui/cache/te_gate**（产品默认是 webui/cache/te，同一
代码路径；独立目录只为不与 7872 上的 seedres 单共享缓存）。一键脚本：
py312 webui/_selftest/t22_te_cache_gate.py --port 7873 --cache-dir webui\cache\te_gate
末行 GATE ALL_PASS，退出码 0；原始输出见 webui/evidence/t22_te_cache/gate.log。

---

## 1. 预测 vs 实测（预测先于实现，单变量）

| # | 预测 | 实测 | 判 |
|---|---|---|---|
| P1 | 冷 job（首跑）textenc 8-12 s（本机 load 5-7.6 s） | **8.57 s**（load 5.5 + decode 1.7 + 进程 ~1.3） | ✅ |
| P2 | 同 prompt 第二 job ≤0.5 s | **0.0068 s**（内存）/ **0.0104 s**（磁盘） | ✅ 超预期 ~10-15x |
| P3 | pe_p0.npz 文件逐字节相同 | 冷/热内存/热磁盘/回切四态均 **437634b6…** | ✅ |
| P4 | schedule.npz 逐字节相同 | 五态均 **f80e3699…**（numpy savez 时间戳问题已消除） | ✅ |
| P5 | 换 prompt 正常重算、pe 不同 | **7.83 s**，pe **b4b5e814…**（与冷 job 90514d08… 不同） | ✅ |
| P6 | 容量上限 + 最旧淘汰 | 3 条上限灌 5 条：**剩 3 条，最旧 2 条被删**；LRU 命中项在再存一条后幸存 | ✅ |
| P7 | closure ±1 s | 五态 **+0.062 / +0.031 / +0.029 / +0.031 / +0.039 s**；gate 模式 +0.076 s | ✅ |
| P8 | t3d_baseline --check 前后不变 | 前后均 **ALL REFERENCE BYTES UNCHANGED** | ✅ |
| P9 | 模型常驻（强制 ON）首跑 ~7-8 s、后续 ~0.6 s | **7.13 s → 0.59 s**（不同 prompt，pe b4b5e814… 与缓存条目互证） | ✅ 机制 |

---

## 2. 键：prompt + 影响 pe 的全部参数；尺寸无关性自证

缓存键 = sha256(规范化 JSON)，字段只有：
prompt（**编码器真正读到的 chat 模板串**）、backend（gpu/cpu，绝不含 auto）、
lid=35（hidden_states[-2] 的层 taps）、gguf（path/size/mtime_ns）、
te_worker.py（size/mtime_ns）、encode_api.py（size/mtime_ns）、v=1。
H/W/steps/seed/batch **刻意不入键**——它们不可能影响 pe，下面是**现有 job 历史**的实证
（不是本次自证循环）：

| prompt | shape | job | pe sha256（前 16） |
|---|---|---|---|
| a mountain lake at sunrise | 128x128 | 20261021-081314-302c18 | 90514d089ec4b356 |
| 同上 | 144x128 | 20261021-112442-4fc4ca | 90514d089ec4b356 |
| 同上 | 128x256 | 20261021-112544-2afdfa | 90514d089ec4b356 |
| 同上 | 256x256 | 20261021-112230-491a08 | 90514d089ec4b356 |
| 同上 | 64x256 | 20261018-200806-a8ac5c | 90514d089ec4b356 |
| 一个漂亮的女孩在海边 | 128x128 / 256x256 | 20261021-120058 / 120321 | 389bec61f0fe532e（两形状同） |

**后端必须入键**：同一 prompt 的 GPU 结果 90514d08…（20261019-230845-5f4eac, backend=gpu），
CPU 回退结果 e6642367…（20261019-231006-57af05, backend=cpu）。runner 在查缓存前把
auto 解析成**与 textenc.run 完全相同的具体后端**（_GPU_OK 为 False 时用 cpu），所以
"auto 命中 GPU 条目、随后 CPU 重算"这种混用不会发生。

失效：GGUF/te_worker/encode_api 任一被重建（size 或 mtime_ns 变）→ 键变 → 旧条目
不可达，并在下一次 store 时被 prune；TE_CACHE_VERSION 变更同理。

---

## 3. 实现（只改 webui/cache.py 与 webui/runner.py）

**webui/cache.py（+526 行，新 T22 TE cache 段）**
- 目录：webui/cache/te/（可用 ZIMAGE_TE_CACHE_DIR 覆盖），每条一个 <key>.npz
  （首跑原始字节，含 pe/tokens/prompt/layer 四个数组）+ <key>.json（字节 sha256、
  pe 数组 sha256、tokens、backend、prompt、编码器戳、last_used）+ index.json（LRU 顺序）。
- 内存层：OrderedDict，默认 8 条；连续命中不碰磁盘。
- 容量：ZIMAGE_TE_CACHE_ENTRIES=64、ZIMAGE_TE_CACHE_MB=256，双上限，按 last_used
  淘汰最旧；keep_key 永不淘汰；index 缺失/中断写留下的孤儿 npz 会按 manifest 重建行，
  无 manifest 的孤儿字节删除。
- 完整性：命中时校验 size + sha256(文件) + pe 形状 (1,L,2560) + sha256(pe 数组)；
  任一不符 → 删除该条并记 miss（绝不返回坏字节）。
- schedule_bytes(steps, sched)：把 steps_N.npz 固定成一份，数组逐值比对后按原字节复制
  （numpy savez 会把本地时间写进 zip 成员，这是旧实现两次运行文件不同字节的根因）。
- te_stats() 并入 cache.usage()，/api/cache 的 cache_usage.te 可直接看到
  entries/bytes/hits/misses/evicted（实测 OK，见 §7）。

**webui/runner.py（+179/-11）**
- TE 阶段：render_chat(prompt) → 解析具体 backend → cache.te_lookup → 命中则
  te_materialize 到 job 的 pe_p0.npz、写回 pe_p0.prompt.txt、追加 te_p0.log 命中记录；
  未命中则原样 textenc.run，CAP_TOKENS 校验通过后 cache.te_store。
  timing 新增信息键 textenc_cache / textenc_cache_source / textenc_cache_key，
  job 新增 te_cache（hit/key/source/backend/tokens/seconds/pe_sha256）。
  textenc_s 仍是本步**全墙钟**（含查缓存/物化），命中 0.007 s、未命中 8.6 s。
- schedule 阶段：cache.schedule_bytes(...) 复制字节；timing.schedule_cache=hit/miss。
- 模型常驻策略 _te_model_policy()：ZIMAGE_TE_MODEL_RESIDENT=off|on|auto（默认 auto）。
  auto = 仅当 commit >= 9.0 GB（快路径门槛）+ 7.0 GB（TE 常驻实测 commit）时保持常驻；
  否则一次性进程（并在发现旧常驻 worker 时 textenc.shutdown() 先释放）。
  本机 commit 9.0-9.9 GB < 16.0 GB → auto 关闭，这正是 T21 2a-a 的 3x 净损失被
  构造性避免（不是没实现，而是这台机器的内存不允许）。

**未改**：webui/textenc.py、webui/te_worker.py、bridge.py、vae*.py、shapes.py、
static/index.html、webui.py。te_p0_gpu.log / te_p0_cpu.log 的命名与内容口径不变
（真实编码才有；命中只写 te_p0.log 一行 [te-cache] HIT …）。

---

## 4. 缓存命中表（四态 + 回切；数字来自 gate.log 的 5 个 job）

| 态 | job | textenc_s | te_cache | pe 数组 sha16 | pe 文件 sha16 | schedule sha16 | latents sha16 |
|---|---|---|---|---|---|---|---|
| 冷（miss→store） | 20261021-125054-f26404 | **8.570** | miss → store | 90514d089ec4b356 | 437634b665835e3f | f80e3699d22175d5 | 9f86aee91de3dad0 |
| 热/内存（同 server 第二 job） | 20261021-125134-ce608c | **0.0068** | mem hit | 90514d089ec4b356 | 437634b665835e3f | f80e3699d22175d5 | 9f86aee91de3dad0 |
| 热/磁盘（重启 server 后） | 20261021-125205-5d1751 | **0.0104** | disk hit | 90514d089ec4b356 | 437634b665835e3f | f80e3699d22175d5 | 9f86aee91de3dad0 |
| 换 prompt | 20261021-125233-872753 | **7.825** | miss → store | b4b5e8144e3b6357 | 308d839b0084cb40 | f80e3699d22175d5 | 270dd4b58aff2c6d |
| 回切标准 prompt | 20261021-125312-b8c70e | **0.0069** | mem hit | 90514d089ec4b356 | 437634b665835e3f | f80e3699d22175d5 | 9f86aee91de3dad0 |

补充：**gate 模式**（两遍同 prompt，20261021-124438-df7ce6）：pass0 7.878 s（miss，
store），pass1 **0.004 s**（mem hit），gate PASS、pe_equal=true、latents 两遍均
9f86aee9…，gate_second_s 27.008 s，closure +0.076 s，total 65.23 s（旧口径会再付
~8 s 的 TE）。日志见 webui/evidence/t22_te_cache/gate_mode.log。

**淘汰实测**：TE_CACHE_MAX_ENTRIES=3，连续存 p0..p4 五条 → entries 3/3，p0、p1 的
npz/meta 均被删除，p2/p3/p4 保留；清内存层后 lookup(p2) 命中（last_used 被刷新），
再存 p5 → 被淘汰的是 p3 而不是 p2。LRU 语义成立（脚本 phase 3，PASS）。

---

## 5. 门表

| 门 | 判据 | 实测 | 判 |
|---|---|---|---|
| ① 同 prompt 两连 job | 第 2 个 textenc ≤0.5 s；pe/schedule 字节逐位同第 1 个 | 8.570 → **0.0068 s**（mem）/**0.0104 s**（disk）；pe 437634b6… 相同；schedule f80e3699… 相同；latents 均 9f86aee9… | **PASS** |
| ② 换 prompt | 正常重算；标准 prompt latents 仍 9f86aee9 | 7.825 s、pe b4b5e814…（不同）；回切 0.0069 s、latents 9f86aee9… | **PASS** |
| ③ run-to-run / closure / t3d | 同参数重复 latents 相同；closure ±1 s；t3d_baseline --check 前后一致 | 五态 latents 9f86aee9…（标准 prompt）逐字节相同；closure +0.029…+0.076 s 全 PASS；baseline 前后 ALL REFERENCE BYTES UNCHANGED | **PASS** |
| ④ 淘汰与上限 | 灌满 N 条验证淘汰 | 条数上限：3 条上限 5 条数据 → 最旧 2 条删除、LRU 命中项幸存；字节上限：2.5 条预算存 3 条 → 剩 2 条 165,904 B ≤ 207,380 B，最旧删除；entries/files/index 一致 | **PASS** |

门禁脚本输出：**33 checks, 0 fail — GATE ALL_PASS**（退出码 0）。原始输出
webui/evidence/t22_te_cache/gate.log。

---

## 6. 模型级常驻（目标②）：机制已交付，本机 auto=off（诚实负结果）

- 机制：textenc 的 resident worker（LRU=1，懒加载，进程内单例，崩溃自愈）由
  runner 的 _te_model_policy 控制；auto 只在 commit 余量足够时启用，否则一次进程。
- 强制 ON 实测（_scratch/te_resident_probe，见 resident.log）：第一次 encode
  **7.133 s**（boot 5.198 s，load 4.908 s，decode 1.61 s），第二次**不同 prompt**
  **0.590 s**（reused=true，decode 0.41 s，无 load）。即"首 job 付费、后续免付"成立。
- 为什么本机 auto 关闭：TE 常驻占 6.74 GB commit（T21 2a-a 实测），会把下一个 job 的
  device loop 从 fast 推到 --no-cache-all（25.8 → 59.6 s，3x 净损失）。本机 commit
  9.0-9.9 GB，加 7.0 GB 后仍低于 9.0 GB 快路径门槛，所以 auto 的判据
  commit >= 9.0 + 7.0 = 16.0 GB 不成立。这是**按测量设的门槛**，不是拍脑袋；在
  ≥16 GB commit 的机器上同一代码会自动常驻。
- 对本任务目标的效果：**同 prompt 复跑已经完全不碰模型**（结果缓存），所以 ② 的
  "后续免付"在重复 prompt 上由 ① 实现；"换 prompt 的第二 job 免加载"只在大内存机器
  上由 auto 实现，本机仍需付 ~8 s（见 §8 精确剩余量）。

---

## 7. 现有产物口径

- 真实编码：pe_p0.npz（含 pe/tokens/prompt/layer 四数组）、te_p0_gpu.log（或 _cpu）、
  以及 textenc 的全部日志行，均与改动前同路径同格式（冷 job 20261021-125054-f26404
  同时存在 te_p0_gpu.log 与 pe_p0.npz，已核对）。
- 命中：pe_p0.npz 是首跑**原始字节**（不是重新 savez），额外产出 te_p0.log 一行
  [te-cache] HIT key=… source=mem|disk backend=gpu tokens=13 pe=… in 0.00x s。
- job.json：新增 te_cache 字段；timing 新增 textenc_cache/textenc_cache_source/
  textenc_cache_key/schedule_cache（信息性，不改任何既有桶含义）。
- /api/cache：cache_usage.te 出现 entries/files/bytes/max_entries/max_bytes/hits/
  misses/stores/evicted/invalid/errors（实测 7873 返回 ok=true，每次请求实时）。
  注：/api/status 的 resolutions 走 webui.py 既有的 _status_memo（本单禁改），其
  cache_usage.te 会滞后到下一次构建缓存变化；job.json 的 te_cache 与 job 日志始终实时。
- 产品默认目录 webui/cache/te 单独复验：7873 上一个正常 job（20261021-124905-80da62，
  未设 cache-dir）冷跑 textenc 5.50 s、store 1 条 134,468 B，latents 9f86aee9…，
  entries after=1 / bytes=134,468 / dir=webui/cache/te —— 默认落盘路径工作。
  （此前 12:27:52 的 7872 job 也写过同一目录；条目被 t22_prod_cache.py 的 full
  clean() 测试按产品语义清掉后，用上面这个 job 重新建立。）

---

## 8. 精确剩余量

| 项 | 数值 | 说明 |
|---|---|---|
| 同 prompt 热路径 | **0.0068 s（mem）/ 0.0104 s（disk）** | 目标 ~0.1 s，实际好 10-15x |
| 冷路径（每个新 prompt 第一次） | **8.57 s**（load 5.5 + decode 1.7 + 进程 ~1.3） | 结果缓存不改变首跑成本；模型常驻 auto 在本机 off |
| 换 prompt 的第二 job（本机） | 仍 **~7.8-8.9 s** | 模型级"后续免付"受 16 GB commit 门槛限制；≥16 GB 机器上为 ~0.6 s |
| gate 模式（两遍） | TE 只付一次：7.878 s + 0.004 s | 旧口径两遍各 ~8 s |
| 缓存占用 | **134,468 B/条**（13 tokens；14 tokens 的换 prompt 条目 144,692 B） | 64 条上限 ≈ 8.6 MiB，256 MiB 字节上限不先触发 |
| 磁盘/内存上限 | entries 64 / 256 MiB；mem 8 条 | 均可 env 覆盖（ZIMAGE_TE_CACHE_ENTRIES / _MB / _MEM_ENTRIES） |
| 门禁未覆盖 | 无 | ①②③④ 全 PASS；② 的模型常驻是目标项，已给强制实测 + 本机 auto 关闭理由 |

---

## 9. 复现

    py312 webui/_selftest/t22_te_cache_gate.py --port 7873 --cache-dir webui\cache\te_gate

（自起 127.0.0.1:7873 服务、清空该 cache-dir、跑冷/热内存/重启后热磁盘/换 prompt/
回切五态 + 3 条上限淘汰 + closure + t3d 前后检查；每态打印 textenc 桶、pe/schedule sha、
PASS/FAIL；末行 GATE ALL_PASS/FAIL，退出码 0/2。--keep 可留服务调试。）

模型常驻机制复现：py312 _scratch/_te_resident_probe.py（强制 resident=True 两次不同
prompt，打印 7.133 s → 0.590 s，结束 shutdown）。

## 10. 变更文件

| 文件 | 变更 |
|---|---|
| webui/cache.py | +526 行：T22 TE cache（磁盘+内存、LRU、双上限、完整性、schedule 字节稳定）、usage() 带 te 统计、clean() 清 TE |
| webui/runner.py | +179/-11：TE 阶段查/存缓存、schedule 字节复制、模型常驻策略、te_cache 字段 |
| webui/_selftest/t22_te_cache_gate.py | 新增：一键四态门禁（自起 7873、独立 cache-dir）。该目录被 .gitignore 整体忽略（与既有 selftest 一致），可提交副本见下 |
| webui/evidence/t22_te_cache/gate.py | 门禁脚本的入库副本（与 _selftest 版本同一文件） |
| webui/evidence/T22-TE-CACHE.md + t22_te_cache/*.log | 本报告与原始输出 |

未动：bridge.py、vae_local.py、vae_worker.py、shapes.py、static/index.html、webui.py、
textenc.py、te_worker.py。
