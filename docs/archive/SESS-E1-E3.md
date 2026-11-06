# SESS-E1-E3 — FusionRuntime 状态复用 / txn 命中率 / atom 直连

> t22_sessA 续（2026-10-24）。承接 `SESS-OVERHEAD-ANATOMY.md` 的四段解剖，
> 按 sessB 的 API 指南（`_scratch/block/stackhw/CLOSED-RT-DIRS3.md` §3）做三个实验。
> 全部只写 `_scratch/sessA/`，产品零变化。

---

## 0. 三实验判定

| # | 问题 | 判定 | 关键数字 |
|---|---|---|---|
| **E1** | `save_state` → 新进程 `load_state` 能否替代重建会话？ | **时间上成立，语义上未证** | **load_state 2.7–11.5 ms** vs 重建 **257–425 s**（≈2–9 万倍）；`execute` 打不到故字节一致**未验**；缺失文件会**硬崩** |
| **E3** | 我们的 shape 在预编译 txn 库里命中吗？ | **0 命中** | 3,218 keys；M 集合 21 个值**无 96**；`_96_1024_10240_` **0 条** |
| **E2** | atom 直连可用吗？ | **API 面可用，执行面未通** | `import atom` 需先 `import torch`；构造 28–48 ms；`execute` 40 个 (M,K) 全 `No valid tiling found` |

**一句话**：会话状态**确实可以跨进程复用，而且只要毫秒级**——但要用它，必须先解决"xclbin 与图的 kernel/PDI 配对"，
而这一半在磁盘上不存在。这是 E1 没能走到 `execute` 的唯一原因，也是官方路线组装方案的第一块缺口。

---

## 1. E1 — FusionRuntime 状态复用（核心）

### 1.1 API 面（实测确认，非文档推断）

| 调用 | 签名（实测 `__doc__`） | 备注 |
|---|---|---|
| `dd.load_meta_json` | `(arg: str) -> Metadata` | 参数是**文件路径** |
| `FusionRuntime()` / `(str)` | `__init__(self)` / `(self, arg: str)` | 无参构造**不碰设备**（0.0 ms） |
| `compile` | `(meta, xclbin_name: str[, config: DDConfig]) -> None` | 第二参数是 **xclbin 文件** |
| `init` | `(meta[, DDConfig]) -> None` | 建 hw_context / kernel |
| `execute` | `(Sequence[ndarray], Sequence[ndarray]) -> None` | 输入/输出列表 |
| `save_state` / `load_state` | `(arg: str) -> None` | MetaState protobuf |

`DDConfig` 26 字段实测可枚举：`cache_dir/model_name/use_elf_flow/enable_preemption/en_lazy_constbo/
dealloc_elfs/dealloc_input_bo/dealloc_output_bo/dealloc_scratch_bo/optimize_scratch/optimize_memory_noop/
instr_xrt_bo_stack_size_mb/profile/reload_const/txn_opt/...`。

### 1.2 已有实物：一份真实的 MetaState

项目早先的 DD 实验留下了完整的一套（`zimage-port/_probe_out/dd_live/out/`）：

| 件 | 大小 | 内容 |
|---|---:|---|
| `dd_metastate_Sd30_Mmdit_ln0.state` | **59,902 B** | MetaState protobuf |
| `dd_metastate_Sd30_Mmdit_ln0.ddbundle` | 645,522 B | bundle 容器（**load_state 不收**，见 §1.5） |
| `cache/ln0_meta.json` | 9,217 B | DD meta v1.4，4 op |
| `cache/ln0_*.const` | 21.5 MB（5 件） | const/权重镜像（外置） |

图：`ln0`（SDLayerNorm_bfbfp）→ `mm1` → `mm2` → `cast`；
IO 由 meta 的 `tensor_map` 给出：in `[1,1024,1536] bf16`（3,145,728 B）、out 同形、scratch 9,437,184 B、const 21,534,848 B。

state 内部（字符串扫描，189 条可打印串）确认它是**真状态**而非桩：
`pm_load_ln0`、`PM_LOAD2`、`op_type=SDLayerNorm_bfbfp`、`op_dtype=bfloat16`、
张量名、`ctrl_packet`、ELF 引用 `SDGGMARNBfp_elf.bin`、`elf_flow`。

### 1.3 实测：load_state 是毫秒级

四次独立进程测量（每次新建 `FusionRuntime`）：

| 调用 | 耗时 |
|---|---:|
| `FusionRuntime(xclbin)` | 0.0009 s |
| `load_state(.state)` #1 | **0.0115 s** |
| `load_state(.state)` #2（bare 构造） | **0.0070 s** |
| `load_state(.state)` #3（ladder 内） | **0.0049 s** |

对照：官方路线重建 17 个会话 **257.1 s（安静）/ 425.5 s（负载）**，单会话 15.1 / 25.0 s。

**⇒ 单个会话的状态恢复比重建快 ≈ 1,300–5,100 倍；整条 17 分片流水线快 ≈ 2.2 万–8.6 万倍。**

### 1.4 这条为什么重要：状态里到底存了什么

MetaState 存的是 `fused_tensors / tensor_map / super_instr_map / const_map / control_pkt_map / const_offset_map / scratch_op_set`。
注意 `const` 在 meta 里是**外置文件 + offset**：

```
ln_w_16_0  const  offset 0        file cache/ln0_0.const
W_up_53_1_0 const  offset 6144     file cache/ln0_2.const   (10,764,288 B, bfp16ebs8)
W_dn_53_1_1 const  offset 10770432 file cache/ln0_3.const   (10,764,288 B, bfp16ebs8)
```

**所以 load_state 恢复的是"说明书"（谁在哪个 offset、用哪个 ctrl packet、调哪个 ELF），不是权重本身。**
权重仍在磁盘上，需要时按 offset 读。

这正好接上四段解剖的 ③：如果运行时按 `const_offset_map` 直接 DMA 416 MB 镜像，
那么 **O(参数) 的逐元素 CPU 处理就可以被跳过**——416 MB @ 5.9 GB/s ≈ **0.07 s**，而不是 25 s。
**这是 E1 最大的潜在收益，也是本次唯一没能证明的一步。**

### 1.5 两个诚实负结果

**N1 — 负对照失败：load_state 遇到不存在的文件，进程直接硬崩。**

```
FusionRuntime()                          -> ok
load_state('<...>/does_not_exist.state') -> 进程死亡 exit=-1073740791 (0xC0000409)
                                            无 Python 异常、无 stderr、无返回值
```

这比"静默成功"更糟，也直接违反 sessB 的判据（"不存在的 state 文件必须报错，不得静默回退编译"）：
**它既不报错也不回退，而是把宿主进程打死。**

而且校验行为**不一致**：

| 输入 | 行为 |
|---|---|
| 格式错的 `.ddbundle` | **正常抛** `RuntimeError: Error parsing metastate file` |
| **不存在的路径** | **0xC0000409 硬崩** |

工程含义（对"跨进程状态缓存"是硬约束）：
1. 服务进程**不能**把 `load_state` 放在可恢复路径上——一次缓存淘汰/文件被清理就会崩掉整个器件进程；
2. 必须在调用前自己做 `os.path.exists` + sha256 校验（B2）；
3. **更正**：我最初把这一条读成"静默成功"，是因为调用之后的打印没有出现——那是进程已经死了，不是函数返回了。

**N2 — execute 打不到：xclbin 与图的 kernel/PDI 对不上。**

| 尝试 | 结果 |
|---|---|
| `compile(meta, rai_xclbin.bin)` | `RuntimeError: Op type SDLayerNorm_bfbfp not registered in op_pdi_map` |
| `compile` 前（错误 CWD） | `Couldn't open file for reading : cache/ln0_3.const` → const 路径按 **CWD** 解析 |
| `init(meta)`（load_state 之后） | `RuntimeError: No such kernel 'DPU_1': invalid argument` |

原因查清：**唯一拿得到的 xclbin 是从厂商 `.rai` 尾部抽出来的** `unified-2x4x4.xclbin`（112,020 B，magic `xclbin2\0`）。
六个存活 `.rai` **全部内嵌同一个 112,020 B 的 xclbin**；它带的 PDI/kernel 是 Z-Image 那条图的，
没有 `SDLayerNorm_bfbfp`、也没有 `DPU_1`。DD 自己用的那份 xclbin **不在磁盘上**
（`_DynamicDispatch.pyd` / `dyn_bins.dll` 里搜到的 `xclbin2` 只是错误串 `"xclbin magic (%s) mismatched"`；
`dyn_bins.dll` 内嵌 xclbin 数 = **0**）。

**⇒ E1 的结论边界："状态可跨进程复用且极快"成立；"复用后能算出正确结果"未证。**
因此 **"权重装载不可缓存" 这一条既没有被推翻、也没有被证实——它现在是"未决"，而不是我上一份报告里的"不可缓存"。**
这个措辞修正很重要：上一份报告说"不可缓存"，指的是**官方 EP 路径每次会话都要重做**；
E1 表明**状态层是可以缓存的**，问题只在 PDI 配对。

---

## 2. E3 — 预编译 txn 命中率

### 2.1 实测

```python
a = dd.Attributes(); a.set('op_version','v2'); a.set('dtype','bfp16ebs8')
keys = dd.llm_txn_query.get_all_txn_keys('MladfMatMul', a)   # 3,218 条，0.024 s
```

| 查询 | 命中 |
|---|---:|
| 总 key 数（MladfMatMul / v2 / bfp16ebs8） | **3,218** |
| 含 `_96_` | **0** |
| 含 `_1024_10240` | 2（但都是 `1024_10240_2560_*`，M=1024 不是 96） |
| 精确 `_96_1024_10240_` | **0** |
| 含 `10240` | 59 |
| 含 `_2560_` / `_512_` | 457 / 409 |

**M 的全部取值（21 个）**：`1, 64, 128, 192, 256, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3584, 4096, 7168`。
**没有 96。**

### 2.2 判定

我们的 shape（M=96, K=1024, N=10240）**在预编译库里 0 命中** ⇒
**S2（missing-txn 代码生成）不可绕**，第一次必须现场生成。

首编成本量化：历史实测 **350–380 s/片**（`HANDOFF.md:319,347`；17 片 ≈95 min，`STATIC-SHAPE-ENGINEERING.md:341`）。
**本次没有重新测量首编**，因为 `compile` 卡在与 E1-N2 同一个 xclbin/PDI 缺口上——这是同一个根因，不是两件事。

注意 M 集合的语义：DD 的 matmul txn 按 **M** 分档（1/64/128/…/7168），我们的 M=96 落在 64 与 128 之间。
两个可能：①按 128 档做 padding（需 M 可 pad，代价 33% 空算）；②现场生成 96 档。
**这是个可以直接测的岔路**（用 M=128 的 key 跑 M=96 的输入看是否报错/是否结果正确），列为后续。

---

## 3. E2 — atom 直连

### 3.1 实测

| 步骤 | 结果 |
|---|---|
| `import atom` 直接 | **失败** `ImportError: DLL load failed while importing _atom` |
| `import torch` 后再 `import atom` | **成功**（torch 2.4.1+cpu） |
| `atom_npu_rmsnorm()` 构造 | **0.028–0.048 s**（不碰设备） |
| `atom_npu_silu/elemw_mul/mha` | 同样无参构造 |
| `atom_npu_gemm(...)` | 需 6 参 `(str,str,str,str,bool,bool)`，语义未文档化 |
| `rmsnorm.execute(x3d, g, F, F)` | `madlfRMSNorm expects ONLY rank 2 tensors [M,K]` |
| `rmsnorm.execute(x2d, g, F, F)` | `No valid tiling found`（`txn_tiling_util.cpp:208`） |
| 40 个 (M,K) 组合扫描（K∈{128,256,512,768,1024,1536,2560,3840} × M∈{1,8,64,96,1024}） | **全部 `No valid tiling found`** |

**依赖发现（可复用）**：`_atom.cp312-win_amd64.pyd`（323.9 MB）静态链 `torch_cpu.dll / c10.dll / torch_python.dll / _cuda.dll / xrt_coreutil.dll`，
所以 **必须先 `import torch` 再 `import atom`**，否则报"找不到指定的模块"。这不是缺 DLL 路径，是加载顺序。

### 3.2 判定

**API 面可用，执行面未通。** 与 sessB 的定位一致（E2 只作路径验证）：
atom 是 **bf16 × int4**，不是我们要的 **bf16 × bfp16**，本来就不能直接替代产品权重格式；
本次进一步证明：即使只做路径验证，`execute` 也过不了 tiling 关（大概率同样缺 DD 的 txn/PDI 资源）。

**净收益**：拿到两条确定事实——①atom 的加载顺序依赖；②无参 op 构造是 **28–48 ms 量级**（vs ORT 会话 28.6 s），
说明"不建会话"确实是毫秒级入口，值得继续投入打通。

---

## 4. 官方路线组装方案（定稿）

### 4.1 目标

- **warm 8.3 s/图**（`batch.log` 图 2 已实测）
- **会话状态可跨进程缓存**（E1 证明 ms 级恢复可行）
- 首次编译付一次 95 min，之后永不再付

### 4.2 三段式组装

```
[离线一次性]                        [服务启动]                [每张图]
export ONNX (17 片)                 读 17 个 .state          load 输入 latent
  → cold compile 350-380s/片         每个 load_state ~5-11ms   execute 8 步
  → save_state ×17                   const 按 offset mmap     VAE 0.57s
  → 落盘 .state + .const             建 1 个 hw_context       合计 ≈ 8.3s
     ≈ 95 min + 6.6 GiB              ≈ 0.1-0.2 s
```

对比现状（`_prof128.log`）：服务启动 425.5 s → **0.1–0.2 s**，每图仍是 8.67 s 计算 + 0.57 s VAE。

### 4.3 必须补的三块（按阻塞程度）

| # | 缺口 | 为什么阻塞 | 怎么补 |
|---|---|---|---|
| **B1** | **xclbin ↔ 图的 kernel/PDI 配对** | E1 的 `compile`/`init` 都死在这；`execute` 打不到 | 从一次**成功**的 DD 运行里把 xclbin 落盘（`XclbinContainer` 的 `/xclbin/stx/...` 名字表在 pyd 里，文件不在）；或让 EP 在编译成功时导出 |
| **B2** | **状态有效性校验** | `load_state` 对缺失文件**硬崩进程**（E1-N1），格式错却正常抛 | 调用前 `os.path.exists` + sidecar（state sha256 + golden 输入的输出 sha256）；不匹配即拒绝，绝不把裸 `load_state` 放进服务路径 |
| **B3** | **M=96 无预编译 txn** | E3：0 命中 | 二选一：M=128 档 padding（+33% 空算）或接受首编 350–380 s/片 |

### 4.4 与开放栈的分工（不变）

官方路线优势区间 = 大批量 / 单会话 / 固定几何；开放栈 = 小批量 / 任意几何 / 可改图。
**E1 若打通（B1+B2），官方路线的启动开销会从 425 s 掉到 0.2 s，届时它对"单张也要快"的场景也成立**——
这是唯一能把两条曲线合并的改动，值得优先投入。

---

## 5. 复现

```powershell
$py = '<python312>\python.exe'
$d  = '<checkout>'
& $py -u "$d\_scratch\sessA_e1c.py"      # load_state 计时 + 缺失文件负对照
& $py -u "$d\_scratch\sessA_e1ladder.py" # load_state -> init -> execute 阶梯（停在 init）
& $py -u "$d\_scratch\sessA_e3.py"       # txn key 命中率
& $py -u "$d\_scratch\sessA_e2c.py"      # atom 加载顺序
& $py -u "$d\_scratch\sessA_e2g.py"      # rmsnorm 40 个 (M,K) 扫描
& $py -u "$d\_scratch\sessA_raix.py"     # 从 .rai 抽 xclbin
```

---

## 6. 纪律与边界

- **预测先行**：E1 预判"load_state ≪ compile"——**成立**；预判"负对照会报错"——**否证**（不是报错，是 0xC0000409 硬崩）。
- **负对照必配**：E1 缺失 state（失败：静默成功）；E2 不支持 shape（失败：不报错而是 `No valid tiling`）；E3 错 op 名返回 0（沿用 sessB）。
- **诚实负结果**：E1 的 execute 未通、E2 的执行面未通、E3 的首编未重测——三条都写在正文，未用"预计"掩盖。
- **产品零变化**：只新增 `_scratch/sessA*` 与本文。
- **未做的**：B1 的 xclbin 落盘；M=128 padding 的实测；atom GEMM 的 6 参语义；51.3 TOPS 门（本轮无吞吐测量，未触碰）。
- **看门狗 ≠ 墙钟**：所有秒均为 `time.perf_counter()`。
- **F-14.1**：本轮无器件动作（execute 未通），未取设备锁，故不适用。
