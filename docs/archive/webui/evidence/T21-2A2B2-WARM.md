# T21 2a+2b 第二梯队：载荷 RAM 预置换 + 器件 worker 保温

**一句话结论**：2b-b 的"入 RAM 即置换"（融合读+置换，去掉两趟全量拷贝）成立但只省 0.49 s；
把它改成"与 W 打包并发"是**净亏**（同 session：startup +3.56 s；一次 run 的 loop +16.8 s，内存压力换页），
按止损线回退为顺序。2a-b 的 py310 器件 worker **成立**：同 session 热 job startup **35.2 → 4.3-6.9 s**
（−82 %），1440 dispatch / 0 重试，latents/pe 与交付逐字节相同；但**常驻器件上下文与默认的 VAE-NPU
阶段不能共存**（2/2 次 VAE 期间 worker 0xC0000005；同样 2 GB 的纯宿主进程存活，故是器件上下文不是内存），
且在本进程内释放上下文本身也会崩（xclbin/BO 析构的已知崩溃类）。因此 **worker 默认关闭**（opt-in），
产品默认路径仍是交付的 one-process bridge。**NOT SHIPPABLE**（默认路径无速度变化，且 VAE 共存是硬阻断）。

锚点 2026-10-19T22:24-23:12 +08:00；HEAD 起点 05bd0c9；隔离端口 7870；py312 服务 / py310 器件；
本 session 全程慢态；所有 A/B 均在同一个 session、同一窗口内。

---

## 1. 预测 vs 实测（预测写在 T21-2A2B2-PREDICTIONS.md，先于任何测量/实现）

| 项 | 预测 | 实测 | 判 |
|---|---|---|---|
| U2 载荷融合（读+置换，去 2 趟拷贝） | 5.4→2.2 s，startup −4.5~5.3 s | **6.1→4.9 s，startup −0.49 s**（35.67→35.18） | ❌ 高估：job 里是磁盘/缓存冷读，拷贝只占一小段 |
| U2 与 W 打包并发 | 藏进 22 s 窗口，−4.5~5.3 s | **startup +3.56 s**（35.67→39.23）；一次 run loop **+16.8 s**（28.9→45.7，step 0 21 s 换页） | ❌ **净亏，已回退** |
| U2 "载荷先于 W 打包" | 避开 cache 驱逐，−2~4 s | **fast pass MemoryError**，回退 low，total 134.8 s | ❌ 内存不允许 |
| U3 worker startup | 6-8 s | **4.27 / 5.27 / 6.46 s**（三次热 job）；E2E 冷 job 7.57 s（另加 prewarm 27.4，见 §6） | ✅ 略好于预测 |
| U3 worker loop | 23.1-26.1 s（refiner 热） | **28.4-29.3 s**（refiner 未常驻，自适应规则丢弃） | ⚠️ 差 2.4 s：内存不允许保温 refiner |
| U3 worker 常驻内存 | 4.9-5.1 GB | **1.85 GB**（只保温 W；payload/refiner 被 fast 阈值规则丢弃） | ⚠️ 比预测少，收益也少 |
| U3 state-8 / 重试 | 0 次 | **0 次**（每个 worker job 1440 dispatch；batch-2 2880 也是 0） | ✅ |
| U3 pe/latents 逐字节 | 构造性一致 | **latents 9f86aee9…、pe 90514d08…** 与交付一致 | ✅ |

**预测失败点（诚实记录）**：我把"并发"当成免费的时间重叠，没有先算**内存带宽/工作集**这笔账：
本机 commit 余量只有 7.3-8.4 GB，W 打包本身要吃 8 GB 的权重读 + 2 GB 数组，
再叠一个 3.71 GB 的载荷线程就把工作集推过余量，换页把收益全部吃掉（还倒亏）。
这和第一梯队"TE 常驻"的失败形状是同一个：**时间变量同时是内存变量**。

---

## 2. U1 基线（同 session，改动前）

128×128 单图 job（TE 正常、8 步、seed 7）。startup 的子相位由 bridge 日志拆出：

| 子相位 | 实测 | 说明 |
|---|---|---|
| py310 进程 + import | ~0.7 s | startup 与 bridge_start 的差 |
| device open + 3 xclbin 注册 | ~1.5 s | registered 3 images 0.01s each + open |
| **--warm：30 层 × 4 投影打包** | **27.1-30.8 s** | 读 ~7 GB bf16 权重 + 重编码 2.0 GB；startup 的 76-83 % |
| **--cache-all：3.71 GB 载入 + NH=5 置换** | **6.1-7.3 s** | 旧实现：np.load→array→.tobytes→再拷进 permuted 目标（3 趟） |
| refiner 前 2 块（在 startup 内） | 1.4 s | 后 2 块在 step 0 内（1.3 s） |
| **startup_s 合计** | **35.7 s（本次 session 基线）** | 第一梯队记录的 30.8-42.0 s 区间内 |

基线 job（20261019-223730，vae off，fast）：total 78.19，textenc 12.86，startup 35.67，
loop 28.91，device 15.61，staging 2.07，closure **+0.088 PASS**，latents 9f86aee9…。
带 VAE 的基线（20261019-223055）：total 198.81，startup 40.41，loop 28.53，
VAE load 18.58 + NPU 94.79（VAE-NPU 是当前默认，128² 下这两项 >110 s，与本单元无关但影响 total 口径）。

**内存账（本 session，job 起点快照）**：available 13.0-14.2 GB，**commit 7.3-8.4 GB**，
load 54-60 %。commit 已低于 config 的 9.0 GB fast 阈值 —— 本 session 的 auto 判定**本来就是 low**；
所有 A/B 都是显式 memory_mode=fast 跑的（与第一梯队 §2/§6 同口径），如实记录，不把环境状态记成改动效果。

---

## 3. U2（2b-b）载荷"入 RAM 即置换"

### 3.1 改动
_scratch/image/t18_run.py：
* _preload_payload() 把"读 30 层 BG/B3 → NH=5 置换 → 写 CACHE_BG/B3"变成一个函数，
  **不再 .tobytes()**：置换直接读 np.load 返回的 uint8 数组（同一批字节、同一层序、同一目标数组），
  全量拷贝从 3 趟降到 2 趟。产物字节与旧路径逐字节相同（构造性：写进 CACHE_BG 的表达式没变）。
* 三种落点由 T18_PAYLOAD_ORDER 控制：last（默认，T13 位置）、first（W 打包前）、
  overlap（W 打包期间，A/B 用）。默认 last。
* 新增"已驻留则跳过"分支（供常驻 worker 复用，见 §4）。

### 3.2 同 session A/B（vae off，单变量）

| 臂 | startup | warm | payload | loop | 备注 |
|---|---|---|---|---|---|
| A 旧（bytes，inline） | 35.67 | 27.1 | 6.1 | 28.91 | 基线 |
| **B 融合（last）** | **35.18** | 27.9 | **4.9** | 28.44 | ✅ 采用；−0.49 s |
| C 并发（overlap，bytes） | 39.23 | **37.0** | 0.0（隐藏） | 28.55 | ❌ startup +3.56 |
| D 并发（overlap，融合） | 37.80 | 35.5 | 0.0（隐藏） | **45.74** | ❌ step 0 21 s（换页） |
| E 先载荷（first，融合） | (fast 崩) | 34.9 | 0.0 | — | ❌ fast MemoryError → low，total 134.85 |

**结论**：并发不是免费的。W 打包是内存带宽密集（读 7 GB + 写 2 GB），载荷线程再压 7.4 GB 流量，
在 commit 余量 8 GB 的机器上把 W 数组页换出，step 0 再读回来。融合本身只省 0.5 s（磁盘读占大头）。
按止损线保留 last，overlap/first 作为有据可查的负结果留在代码里（默认不走）。

### 3.3 门
latents 9f86aee9…（B 臂）与交付一致；pe 90514d08…；1440 dispatch / 0 重试；closure +0.048 PASS。

---

## 4. U3（2a-b）器件 worker 保温

### 4.1 设计
webui/dev_worker.py（新，py310 常驻）+ bridge.driver_argv()（把 argv 构造从 main 提出来，
两条路径不再有第二份 argv）+ runner 的懒启动/回退。worker：
* 启动一次：device + 3 xclbin 注册 → **W 打包 120 张投影图（2.0 GB）** → 可选载荷/refiner；
* 每个 job：复用同一个 t18_run.run() 函数体、同一份 driver_argv（除 --wpack-array），
  T.Dev 工厂返回常驻 Dev，Loop.w_images 命中预热缓存，dev.stats 每 job 重置；
* 协议：stdin 一行一个 JSON（run/release/ping/shutdown），stdout @@WEBUI@@ 事件流；
* 回退：worker 起不来/中途死 → 本 job 剩余 attempt 走交付 bridge；任何非 128² / low / BO 常驻
  都不走 worker；worker 死掉后 runner 丢弃它；VAE 前杀掉 worker（§4.3）。

### 4.2 实测（同 session，单变量）

| 臂 | startup | loop | wall | dispatch/重试 | latents |
|---|---|---|---|---|---|
| 交付 bridge（A 旧） | **35.67** | 28.91 | 65.2（设备进程） | 1440 / 0 | 9f86aee9 |
| worker job 1（首个热 job） | **4.27** | 28.60 | 32.9 | 1440 / 0 | 9f86aee9 |
| worker job 2（同一 worker） | **5.27** | 28.78 | 34.1 | 1440 / 0 | 9f86aee9 |
| runner E2E 冷 job（含 prewarm） | **34.94**（prewarm 27.37 + 7.57） | 35.57 | total 79.75 | 1440 / 0 | 9f86aee9 |
| worker 冷启动 prewarm | **25.3-27.4 s**（W 25.4-27.4 + device 0.04） | — | — | — | — |

worker 热 job 的 startup 构成：载荷 2.7 s（融合后）+ refiner 前 2 块 1.4 s + pre()/握手 ~0.5-2 s。
**没有把 refiner 常驻**（2.67 GB）：自适应规则要求"加上它之后 commit 仍 ≥ 9.0"，
本 session commit 4.3-8.4 GB，不满足 → 丢弃，loop 保持 28.6-29.3（step 0 仍有 1.3 s 解码）。
在 14.5 GB commit 的空闲机器上，W 1.85 + refiner 2.67 = 4.5 GB 仍过阈值，则 loop 可再少 2.4 s（未在本 session 复现）。

### 4.3 阻断：VAE-NPU 共存（止损线③）
| 实验 | 结果 |
|---|---|
| worker 常驻（空转）→ py312 VAE-NPU decode | VAE **成功**（96.2 s，npu，PNG sha e7aef10c…），**worker 死 0xC0000005**（2/2 次） |
| 纯宿主 2 GB py310 进程（无 pyxrt）→ 同一 VAE | 进程 **存活**（DUMMY SURVIVED），VAE 成功 → **不是内存** |
| 在 worker 内 release（丢 device、保留 W） | **进程直接崩**（无 traceback；xclbin/BO 析构的已知崩溃类） |

结论：本驱动栈上**一个常驻 py310 器件上下文不能与 py312 VAE 链共存**；在进程内释放上下文也不安全。
因此 runner 在 VAE 前**杀掉** worker（不是 release），worker 默认关闭。
这条同时解释了为什么不能把 worker 做成"服务器启动即预热"：VAE 是默认路径，预热出来的上下文活不过第一个 VAE。

### 4.4 第二个 E2E job 的内存墙（如实记录）
runner 连续第二个 job（20261019-231006）在 worker 内 **refiner 解码时 MemoryError**
（"Unable to allocate 14.1 MiB"，worker ready 时 commit 仅 5.05 GB）：fast pass 8.37 s 失败 →
runner 按既有 MemoryFailure 路径回退 low（retry_s 58.57），且 TE 在同一次尝试里 Vulkan→CPU 降级
（pe 变成 e6642367…，latents 457212ee…），low pass total 142.37，closure **+0.270 PASS**。
这不是 worker 逻辑错误，而是本机内存态：worker 的 1.85 GB + TE 的 6.74 GB（失败前尝试）+ 载荷/refiner
越过了 commit 上限。它同时说明 **worker 不能在这个环境里做默认**：默认 VAE + TE-GPU + worker 三者相加就超了。
（E2E 冷 job 的 closure 仍 PASS，说明计时归属正确。）

### 4.5 可用的收益面（opt-in）
* ZIMAGE_DEV_WORKER=1 + vae=false（trial 或 gate 第二遍）：startup 35.2 → 4.3-7.6 s；
* ZIMAGE_VAE_NPU=0（CPU VAE）时同样可用，且没有共存问题；
* gate job（两遍）：第一遍有 VAE→worker 被杀，第二遍重新预热 25 s；两遍合计仍比两遍 bridge 少 ~40 s
  （按分项数字推算，未在本轮跑完整 gate，不计入交付速度）。

---

## 5. 门禁五项

| # | 门 | 结果 | 证据 |
|---|---|---|---|
| ① | pe 逐字节 | **PASS** | 90514d08… 在 A/B/C/D/E、worker、E2E 冷 job、基线 223055 全部相同 |
| ② | seed 复现（同参数两次 latents 逐字节） | **PASS** | 同 seed 7 的 7 个 job 全部 9f86aee9…；worker 连续两 job 亦是 |
| ③ | 批语义（batch 2 == 两张各自单图） | **PASS** | batch-2 t22b2（2880 dispatch/0 重试）squeeze 后 img0 == seed-7 单图、img1 == seed-8 单图 t22s8，两次 np.array_equal True、max|d| **0.0**（img1 sha be558003f1820e88 两路径相同） |
| ④ | 计时闭合 Σ桶 == total ±1 s | **PASS** | worker E2E 冷 job +0.072；融合 B +0.048；基线 +0.088；low 回退 job +0.270 |
| ⑤ | t3d_baseline --check | **PASS** | ALL REFERENCE BYTES UNCHANGED |

**④ 说明（已修复的缺陷）**：第一版 worker 集成把 prewarm 放在 dev_perf0 之后 → 首个 job 的 closure 差
−26.5 s（FAIL）；修复 = 时钟在 prewarm **之前**起（冷 prewarm 是这单 job 真实花掉的时间，进 startup_s）。
修复后的 E2E 冷 job：startup 34.94 = prewarm 27.37 + 7.57，closure **+0.072 PASS**。

---

## 6. A/B 与 E2E（同 session）

### E2E（runner 产品入口，128²，8 步，seed 7，vae=false，forced fast，ZIMAGE_DEV_WORKER=1）
| job | 臂 | startup | loop | total | closure | 备注 |
|---|---|---|---|---|---|---|
| 223730 | bridge（基线） | 35.67 | 28.91 | 78.19 | +0.088 | 交付路径 |
| 224550 | bridge + U2 融合 | 35.18 | 28.44 | 75.39 | +0.048 | 默认路径 |
| 230845 | worker，冷（含 prewarm） | **34.94** | 35.57 | 79.75 | **+0.072** | prewarm 27.37 + 7.57；1440/0；9f86aee9 |
| 231006 | worker，第二个 job（内存墙） | 32.36（low 回退） | 43.87 | 142.37 | +0.270 | fast 8.37 s 失败→low；TE 退 CPU（pe e6642367…） |

**读法**：worker 的收益在**热 job**（4.3-6.9 s，见 §4.2）；冷 job 的 prewarm 25-27 s 记进 startup 后与 bridge 基线持平（−0.7 s）。
本机内存态不允许连续跑第二个 worker job，因此 E2E 层没有拿到一个干净的"第二个热 job"数字；
热 job 的数字来自独立 client 的 3 次实测（4.27/5.27/6.46），latents/pe 均与交付逐字节相同。

### batch-2 门（t22b2，seed 7/8，8 步，与单图同 pe/同镜像）
* 2880 dispatch、0 重试、wall 49.14 s、device 30.29 s、staging 4.07 s；
* img0 squeeze 后与 seed-7 单图 np.array_equal True、max|d| **0.0**；
* seed-8 单图 t22s8（同 session、同 pe、batch 1）跑通后，img1 与之 np.array_equal True、
  max|d| **0.0**（sha16 be558003f1820e88 两路径相同）；
* 结论：batch 语义逐字节 PASS（2/2 图），2880 dispatch / 0 重试。

---

## 7. 与第一梯队 §7 预测的对账

| 第一梯队 §7 预测 | 本轮实测 | 对账 |
|---|---|---|
| (d) BO 常驻把 3.71 GB 移出 commit，fast 保住 | 未走 BO 路线：BO 与 RAM 在 UMA 上同样吃 commit；改为"worker 常驻 W + 载荷每 job 融合载入" | 路线改判：BO 常驻不解决 commit 账，且 60 BO 的首 dispatch 崩溃面更大 |
| 2a-b 是"唯一保 bf16 pe 逐字节"的路 | **成立**：worker/融合路径 pe 与 latents 全逐字节 | ✅ |
| 预期 −10.4 s/job | VAE-off 热 job startup −29~31 s；VAE-on 路径 0（worker 被 VAE 杀） | 部分兑现 |
| "内存缺口 1.0-1.1 GB" | 本轮 commit 余量 7.3-8.4 GB，缺口比第一梯队记录的更大（环境态不同） | 环境态如实记录 |

---

## 8. 改动文件与 commit

| commit | 内容 | 文件 |
|---|---|---|
| 9315c92 | t21_2a2b2: U2 载荷融合 | _scratch/image/t18_run.py |
| 1047f6c | t21_2a2b2: U3 器件 worker（opt-in） | webui/dev_worker.py, webui/bridge.py, webui/runner.py, webui/config.py |
| 见本条 commit | 证据 | webui/evidence/T21-2A2B2-WARM.md, T21-2A2B2-PREDICTIONS.md |

未改：webui/cache.py、webui/textenc.py、webui/te_worker.py、_scratch/image/t13_run.py、
交付 xclbin/bin、任何 reference 字节。

## 9. 精确剩余量

| 项 | 状态 |
|---|---|
| 2b-b 融合 | 完成，−0.5 s，默认路径 |
| 2b-b 并发 | 证伪（+3.6 s / +16.8 s loop），代码保留为 A/B 开关，默认关 |
| 2a-b worker（VAE-free） | 完成，热 job startup −29~31 s，opt-in |
| 2a-b worker 跨 VAE | **阻断**：器件上下文与 VAE-NPU 不能共存；需要"VAE 独立子进程 + 每 job 释放上下文"的架构，约 0.5-1 设计日 |
| worker + refiner 常驻 | 未兑现（内存阈值规则丢弃）；在 14.5 GB commit 机器上可再 −2.4 s loop |
| 第二个 E2E 热 job | 未测到干净数字：本机内存墙（worker 1.85 + TE-GPU 6.74 + 载荷/refiner），非逻辑错误 |
| 首个 job 的 prewarm 25 s | 可移到 VAE 期间（纯 CPU 打包，不带 device）→ 需要 worker 的"纯打包预热"拆分，约 0.5 设计日 |
