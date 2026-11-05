# MEM-PROFILE —— 内存与耗时归因（含对早先结论的更正）

**日期：2026-10-12**
**探针：`zimage-port/probe_mem_profile.py`**
**数据：`zimage-port/_probe_out/mem_profile/{timing_layers_00_02,mem_stock,mem_lean}.json`**

---

## ⚠️ 0. 本文第一版有两处错误，已在下文更正

第一版据「17 个无权重骨架会话只占 2.42 GiB」推翻了 `MEMORY-AND-TIMING.md` /
`ROOT-CAUSE-SILU-ORDER.md` 的实测。**这是错的**：

| 第一版说法 | 更正 | 依据 |
|---|---|---|
| 「BFP16 从不以 BFP16 形态驻留 RAM」 | ❌ **错**。**initializer 路径下 VAIP 打包完就释放源**，权重确实以 BFP16 驻留 | `ROOT-CAUSE-SILU-ORDER.md:210-217` |
| 「0.88/0.522 GiB 每会话开销不复现」 | ❌ **错**。那是 initializer 路径的实测；我测的是**无权重骨架 + 15 片共享编译产物**的另一工况 | `MEMORY-AND-TIMING.md:57-59` |

**根因**：我把 `SkeletonRunner` 这份新代码的缺陷（解包成 fp32 缓存）
误当成了平台事实。**平台行为是好的，是我们的 runner 把它做坏了。**

---

## 1. 平台侧的既有实测（正确，勿再质疑）

### 1.1 initializer 路径：权重以 BFP16 驻留

`probe_session_memory2.py` 对照（`ROOT-CAUSE-SILU-ORDER.md:207-217`）：

| 分片集 | 权重存成 | S | 每会话边际 RSS | 隐含 B/参数 |
|---|---|---|---|---|
| `shards_i1024_c32` | fp16（0.674 GiB/片） | 1056 | 1.137 GiB | 3.37 |
| `shards` | **fp32（1.348 GiB/片）** | 96 | **0.891 GiB** | **2.64** |

> **推论**：fp32 那组若保留源，每会话至少 1.348 GiB；**实测 0.891 GiB，
> 比它自己的 fp32 权重还小** ⇒ **VAIP 打包完就释放了源**。

### 1.2 每会话内存拆解（`MEMORY-AND-TIMING.md:53-74`）

| 几何 | S | 每会话 RSS | 权重 | 无法归属 | 17 会话合计 |
|---|---|---|---|---|---|
| 128 | 96 | **0.902 GiB** | 0.379 | 0.522 GiB (58%) | **15.33 GiB**（实测建满 17 个）|
| 512 | 1056 | **1.154 GiB** | 0.379 | 0.775 GiB (67%) | 19.61 GiB（实测 15 个即耗尽）|

```
BFP16 权重        0.379 GiB (33%)   ← 真实大小，不可压
每会话固定开销     0.522 GiB (45%)   ← 与 S 无关
随 S 增长         0.255 GiB (22%)
```

### 1.3 各格式边界（`CPU-FALLBACK-FORMAT.md:129-139`）

| 格式 | B/参数 | 总计 | 5-block shard | 能加载 |
|---|---|---|---|---|
| fp32 | 4.0 | 22.93 GiB | 3.54 GB ❌ **超 protobuf 2 GB** | ✅ |
| **fp16** | 2.0 | **11.46 GiB** | **1.69 GB** ✅ | ✅ |
| int8 | 1.0 | 5.73 GiB | 0.84 GB ✅ | ❌ 四种形式全失败 |
| **BFP16** | 1.125 | **6.45 GiB** | — | ❌ **ONNX `TensorProto` 无块浮点类型** |

> **fp32 不只是占地方，它还把分片卡死在 2 block —— 这才是 17 个会话的根源。**

---

## 2. 本次新增：耗时归因（此前完全没测过）

单片 `layers_00_02`（骨架路径），30 个权重输入，重复 2 次：

| 项 | rep 0（冷） | rep 1（热） | 均值 | 占比 |
|---|---|---|---|---|
| `ort.InferenceSession()` | 0.72 s | 0.31 s | **0.52 s** | **5.5%** |
| **`_load_weights()` 解包** | 12.41 s | 5.30 s | **8.86 s** | **94.5%** |

解包后 fp32 字节：**1.348 GiB/片**（与 `ROOT-CAUSE-SILU-ORDER.md:213` 的
「fp32（1.348 GiB/片）」精确吻合，交叉验证通过）。
packed BFP16 只有 **0.337 GiB**，**4.0×**。

### 这条是新的，而且解释了 BFP16 为什么反而慢

- `skeleton_runner.build_times` **只包住 `InferenceSession`**（第 283–286 行），
  `self.weights[name] = self._load_weights(name)`（第 287 行）**在计时器之外**。
  所以打印的「session X: xx s」**漏掉了 94.5%**。
  **与最初问题的「计时盲区」是同一类错误，只是发生在另一个字段上。**
- **这就是 BFP16 那次 1098 s 输给 fp32 468 s 的全部原因**：
  272 次重载 × 5–12 s 解包。磁盘省 4 倍，RAM 一分没省，还白付解包。
- 与既有结论并不矛盾：`MEMORY-AND-TIMING.md` 的「建会话 10–26 s」是
  **initializer 路径**（含 VAIP 打包 ~17 s，见 `SESSION-COST-ATTRIBUTION.md:46-50`
  「0.23 s 磁盘 1% / 2.4 s 解析 12% / ~17 s EP 87%」）。
  **骨架路径下建会话只要 0.52 s**（没有 initializer 要打包），
  于是大头换成了权重解包 —— 而它没被计时。

---

## 3. 本次新增：骨架路径的会话侧已经实现了「共享 overlay」

`mem_stock.json`：17 个无权重骨架会话逐个建，跟踪 `psutil` 可用内存：

| | stock | lean |
|---|---|---|
| 建 3 个后 | 2.20 GiB | 2.22 GiB |
| **17 个全驻留** | **2.417 GiB** | **2.454 GiB** |
| 边际/片（第 4 片起） | ≈0.01 GiB | ≈0.01 GiB |
| 建完剩余可用 | 12.06 GiB | 12.02 GiB |

**前 3 个（noise_refiner 588 / context_refiner 522 / layers_00_02 588 节点）各占 ~0.72 GiB，
第 4 片起几乎不增加** —— 因为 15 个 `layers_*` 结构完全相同，
VAIP 缓存以图为键，**共享同一份编译产物**。

这正是 `ROOT-CAUSE-SILU-ORDER.md:230-238` 构想的正解：

> ```
> 现在        : 6.4 GiB 权重 + 12.9 GiB 开销 = 19.3 GiB  ❌
> 共享 overlay: 6.4 GiB 权重 + ~0.7 GiB 开销 =  ~7.1 GiB  ✅
> ```

**骨架路径已经把会话侧做到了。剩下的是权重侧。**

**lean 会话选项（单线程 / 关 mem arena / 关 mem pattern）完全无效**（2.42 vs 2.45 GiB），
`probe_lean_sessions.py` 那条路线可关闭。

> ⚠️ **口径与风险**：
> (a) 这是**只建会话不跑**的内存，首次 `run()` 另有激活/工作区缓冲。
> (b) 边际≈0 依赖「15 片结构相同 → 共享产物」。若某天这些骨架被改成
> 结构互异（或 VAIP 缓存键改成含权重），这个收益会消失，
> 届时会退回 `MEMORY-AND-TIMING.md` 的 0.522 GiB/片。
> **(b) 需要单独验证一次：15 片是否真的共用一份 `.rai`/编译产物。**

---

## 4. 预算表（可用 14.48 GiB 实测）

| 方案 | 会话 | 权重 | 合计 | 判定 |
|---|---|---|---|---|
| **现状**（骨架 + fp32 解包缓存 + cache=2） | ~1.4 | 2×1.348 | ~4.1 GiB | 装得下，但 272 次重载 |
| 骨架 + fp32 解包缓存 + cache=17 | 2.42 | 22.92 | **25.3 GiB** | ✗ 超 10.8 GiB |
| **骨架 + packed BFP16 直喂 + cache=17** | 2.42 | **5.73** | **8.15 GiB** | ✅ **< 10 GB，余 6.3 GiB** |
| 共享 overlay 构想（文档） | ~0.7 | 6.4 | **~7.1 GiB** | ✅ |
| 旧 initializer 路径 17 会话（实测） | 8.9 | 6.4 | 15.33 GiB | 装得下但紧 |

**目标配置是 7–8.15 GiB —— 就是「不到 10 GB」。**

---

## 5. 对下一步的影响（方向不变，理由要改）

**路线 ①②③（让 EP 直接吃 packed BFP16）仍然对**，但理由是：

> ~~因为 BFP16 根本进不了 RAM~~（错）
> → **因为 `SkeletonRunner` 现在把权重解包成 fp32 缓存（22.9 GiB）。
> 只要改成保留 packed 形态（5.73 GiB）就是 8.15 GiB。
> 而「保留 packed 并直喂」需要 EP 接受 packed BFP16。**

**同时多了一条更省事的候选**（`CPU-FALLBACK-FORMAT.md` §4.3 已论证）：

> **fp16 存储 + 5-block 分片** → 11.46 GiB，会话 17→7，两个 EP 都能加载，
> 能回落 CPU，**不需要破任何 EP 接口**。离 10 GB 差 1.5 GiB。

| # | 动作 | 性质 |
|---|---|---|
| **0** | **验证 15 片是否真共用一份编译产物**（决定 §3 的收益是否稳固） | 先做，便宜 |
| **1** | **修 `SkeletonRunner` 的 fp32 解包缓存**（改成保留 packed，解包延后/免去） | 修 bug，独立 |
| **2** | **修计时盲区**：`build_times` 加上 `_load_weights` | 修 bug，独立 |
| **3** | 路线 ①②③：Cast → AMD 自定义算子（`register_castavx.cpp` 日志已确认注册）→ `.rai` 权重区注入 | 主线 |
| 4 | fp16 + 5-block 分片（11.46 GiB，不需要新接口） | 备选，需重导出 |
| 5 | 真实分片阶梯测试，回答「能不能不分片」 | 见下 |

### 「能不能不分片」仍未回答

`_probe_out/ceiling/ceiling.json`：MatMul 链 **100 OK / 200 崩溃 0xC0000005**。
**不可外推** —— 那是不融合的 NPU op，真实分片 588 节点融成 1 个。
正确探针是**真实分片的层阶梯**（2/4/8 层）。全模型 9930 节点 / 506 权重输入。

---

## 6. 复现

```powershell
$py = 'D:\Users\15081\miniforge3\envs\ryzen-ai-1.8.0\python.exe'
& $py zimage-port\probe_mem_profile.py --info
& $py zimage-port\probe_mem_profile.py --timing --shard layers_00_02 --reps 2
& $py zimage-port\probe_mem_profile.py --mem --mode stock --max 17 --floor-gb 3.0
& $py zimage-port\probe_mem_profile.py --mem --mode lean  --max 17 --floor-gb 3.0
```
