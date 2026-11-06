# t22_quickfix — 三件小挂账的处置记录（2026-10-20）

> 范围：`_scratch/block/step4_run.py`、`_scratch/block/step2d_audit.py`、`docs/` 下本文件与
> `OBS-id79-amplitude-ulp-2.000.md`。其余文件一字未动（`step4_accept.py` / `step4_compose.py` /
> `step4_oracle.py` 的 git diff 为空，可自证）。

## 0. 一句话结论

| # | 挂账 | 处置 | 证据 |
|---|---|---|---|
| ① | `step4_run.py --verify-only` 的 ulp 门（max ulp 43072、exit 3） | **修掉**：判定量换成项目自己的 `rel_to_absmax <= 5.4e-2`，ulp 降级为标注过的诊断；实测 exit 3 → exit 0（7.9365e-03 / 2.7902e-02） | §1 |
| ② | `check_order` 的 ORDER 门（IR 提取器返回空表、对任何设计都 FAIL） | **修掉 5 个提取器缺陷 + 把门移出验收腿**：修后 IR/ASM 各 36 事件、多重集完全相同，线性序仍差 8 处 —— 证明比较本身不成立，不是 codegen 重排 | §2 |
| ③ | id79 幅值界 ulp 2.000 零余量 | **归档**，门限不动（≤2.0） | `docs/OBS-id79-amplitude-ulp-2.000.md` |

**没有改任何门限。** 5.4e-2、2.0、5.0760e-02 口径全部原样；改的只有 (a) 用哪个量去判，(b) 工具能不能正确读出事件。

---

## 1. ① step4_run.py `--verify-only`：ulp 门不是“门坏了”，是这个量在这里不成立

### 1.1 实测分解（为什么 max ulp 会是 1.66e35）

`refs.check_within_ulps` 是**逐元素相对**量：`err / max(bf16_ulp(ref), 2**-126)`，
`bf16_ulp(v) = 2**(floor(log2|v|) - 7)`（`_scratch/kernels/refs.py:180-199`）。
参考元素一旦是 0 或远小于 drain 的 absmax，分母就塌了。记录下来的 M=96/K=96 产物上实测：

| drain | max abs err | max ulp（改前） | 其中零参考元素 | >2 ulp 的元素 | absmax | rel_to_absmax |
|---|---|---|---|---|---|---|
| x_out | 3.125000e-02 | **1.661535e+35** | 1 / 9216（dev=1.953125e-03） | 1780 / 9216 | 3.9375 | **7.9365e-03** |
| Z | 2.441406e-02 | **2.199840e+05** | 0 / 9216 | 6021 / 9216 | 0.8750 | **2.7902e-02** |

最坏 x_out 元素是**一个**参考为 0 的点：err = 1.953125e-03，`1.953125e-03 / 2**-126 = 1.66e35`。
去掉它之后第二名也有 1024 ulp：ref = 9.765625e-04（= absmax 的 2.48e-04），err = 7.8125e-03。
Z 没有任何零参考元素，仍有 6021/9216 超过 2 ulp —— 说明这不是“偶发零元素”的边界问题，
而是**逐元素相对量 + 绝对误差有界**的链根本不该用 ulp 判。
该链的误差是绝对有界的（≤ 0.79 % absmax），这正是项目把判定量定成 `rel_to_absmax` 的原因。

### 1.2 改了什么（`_scratch/block/step4_run.py`）

* 新增 `ACCEPT_REL_BUDGET = 5.4e-2`（注释写明：**不是新门限**，就是 `step4_accept.py` 自己的预算）。
* `verify()` 内的 `report()`：判定量改为 `rel_to_absmax = max_abs / max|ref|`，`<= 5.4e-2` 判 PASS/OVER；
  ulp 仍计算、仍打印，但整行标 `[DIAGNOSTIC ONLY, not a verdict]`，并带出零参考元素数与超差元素数。
* 两条 bf16-floor 的“诊断对照”行改为 `judge=False`（打印 `informational`，不参与判定）。
* 结尾判定行改为 `CROSS-CHECK PASS (rel_to_absmax <= 5.4e-2 ...)`，并打印一行
  `acceptance authority: step4_accept.py (hidden 3840 ... rel_to_absmax 5.0760e-02 PASS)`。
* 文件头加 VERDICT 段：写明 ulp 为什么不是判定量、`step4_accept.py` 是唯一权威、
  以及本脚本的数（M=96/K=96，对 composed model）**与 5.0760e-02 不可比**。

### 1.3 预测 vs 实测 vs 结论

| 项 | 预测（改前写下） | 实测 | 结论 |
|---|---|---|---|
| x_out `rel_to_absmax` | 3.125e-2 / 3.9375 = 7.94e-3 | **7.9365e-03** | ✓ |
| Z `rel_to_absmax` | 2.441406e-2 / 0.875 = 2.79e-2 | **2.7902e-02** | ✓ |
| 两 drain 均 ≤ 5.4e-2 | 是 | 是 | ✓ exit 0 |
| exit 语义 | 3 → 0 | **3 → 0** | ✓ |
| ulp 数字保持不变（仅标注） | 是 | 1.662e+35 / 2.200e+05 原样打印 | ✓ |

### 1.4 改后的 exit 语义（**以本节为准**）

| code | 含义 |
|---|---|
| 0 | 两条判定 drain 的 `rel_to_absmax <= 5.4e-2`（对 composed device-arithmetic model，M=96 K=96） |
| 3 | 同上但超预算 —— **只是本几何的 cross-check 失败**，不是验收结论；验收看 `step4_accept.py` |
| 1 / 2 | 编译/器件侧与 usage（未改动） |

### 1.5 如实标注：这是本次唯一一处“判定量”改动

按止损线要求单独声明：①要“修掉”就必然要换判定量，因为 ulp 在这里**按定义**不适用；
不换判定量就只能“明确废弃”（exit 3 永远为真）。这里选的是“换判定量 + 写明唯一权威”，
理由是：仓库文化里“永远绿”比“永远红”更危险（HANDOFF §5：“看起来绿”是最坏的失败），
而且换过去的量**就是项目自己的量、自己的阈值**（`step4_oracle.divergence` 的 `rel_to_absmax`，
`step4_accept.py` 的 5.4e-2），没有新造任何判据。
若控制器更想要“纯废弃、脚本不再判定”，把 `report()` 里的 `ok` 改成常量 False、
`return 0 if ok else 3` 改成 `return 0` 即可（两行），本次没有这么做。

---

## 2. ② check_order 的 ORDER 门：五个提取器缺陷 + 一个不成立的比较

### 2.1 五个提取器缺陷（全部实测，全部只产生**假**分歧）

被测对象：`_scratch/block/step4_out/s4a_M96_K96_h3_d32_m8_d111.prj` 的
`elfs_main_core_0_2.elf`（24760 B）与同目录 `peano-linked_main_core_0_2.ll`（11459 B, 287 行）。

| # | 缺陷 | 实测 | 修法 |
|---|---|---|---|
| 1 | IR 内核调用正则 `call\s+void\s+@("?[\w.$]+")\(` **强制要求结尾引号** | Peano 只在名字以数字开头时加引号：`@"4a0d8d91_matmul..."` 命中、`@d52c1240_rms_norm_weighted(` **不命中**。36 个内核调用里丢 4 个（rms_norm_weighted 1、rope 2、eltwise_add 1） | 改 `@"?([\w.$]+)"?\(`，并去掉 strip 引号 |
| 2 | 函数头正则只认 `@(main\|core_0_2)` | `peano-linked_main_core_0_3.ll`（step1 的 core）上 `event_order_ir` 返回 **[]**；修后 **7** 个事件 | `@"?(main\|core_0_\d+)"?\s*\(` |
| 3 | 找不到函数头时**静默返回 []**：body 扫描从第 0 行开始，那里 depth 已是 0，循环第一行就 break | 正是 KNOWN-ISSUES L1410-1415 记的 `IR[0:0]=[]` | 只有当“已见过 `{` 且 depth 回到 0”才 break；`strict=True` 时找不到函数头直接抛 `OrderExtractError` |
| 4 | ASM 的 `JL_RE.match` **锚定行首** | AIE2P 一行一个 6 槽 bundle：`nopa ; nopb ; jl #0x1eb0 ; nopv`。全 ELF 含 `jl #0x` 的行 **43** 条，`JL_RE` 只命中 **21** 条，漏 **22** 条 —— 其中 `0x1f0: nopa;nopb;nops;jl #0x1eb0;nopv` 就是 `adaln_modulate` 的调用 | 改成按行 `finditer` 三种事件、按**打印顺序**（槽序）排列 |
| 5 | `_main_extent` 的名字表写死为字面量 `("main", "core_0_2")` | 其它入口名（如 `core_0_3`）会走 `return lines`，把**所有内核函数体**都算进 core | 改 `^(?:main\|core_0_\d+)$` |

缺陷 1 和 4 是“静默漏事件”，比返回空表更隐蔽：漏掉的事件让序列对不齐，于是 SequenceMatcher
把**位置错位**报成 insert/delete/replace —— 外观上很像“codegen 把调用挪过了 acquire”。

### 2.2 修完提取器之后：多重集完全一致，线性序仍差 8 处

五个缺陷全修后，同一对 ELF/.ll：

```
[audit]   IR 36 event(s) vs ASM 36 event(s); multiset identical: True; added 0 / dropped 0
[audit]   linear order differs in 8 place(s) with an IDENTICAL multiset
```

即：codegen **没有增删任何一个** acq/rel/call 事件（多重集逐项相同），但两边的**线性次序**差 8 处。
原因不是重排，是**基本块布局**：反汇编按**地址序**排，而 llc 可以任意摆放基本块。可举证的实例：

```
<.LBB0_5> 0x162:  rel #0x32, r10          <- k 循环的出口块
<.LBB0_7> 0x170:  ... mova r0,#0x8 ...
          0x1ba:  jl  #0x990                (rms_norm_weighted)
          0x1c6:  padda [p6],m0; acq #0x33, r9; mov m0,r0
```

IR 里 `acq 51`（=0x33）在 `rel 50`（=0x32）**之前**（.ll:57 vs .ll:90），地址序里却相反 ——
因为 0x162 属于 k 循环退出块、0x1c6 属于下一层 j 循环体，**两种次序都是正确执行**。
（另有一条同源理由：普通调用可能被内联，IR 有 call 而 ELF 没有。）

结论：**序列比较本身不能当门**。这不是“再修一个 bug”能解决的，需要换算法。

### 2.3 处置：门移出验收腿，保留一个**成立**的信号

`--order` 现在：

* 开头打印 `ORDER: DIAGNOSTIC ONLY -- retired as an acceptance leg (t22_quickfix); it does not change the exit code`；
* 提取失败时打印 `IR EXTRACTION FAILED: <原因>` 并明确“这是工具失败，不是 codegen 结论”；
* 保留**多重集不变量**（added / dropped = 0）作为唯一成立的门级信号；
* 线性序差异只作为证据打印（≤8 条），并说明是布局；
* **不再影响 exit code**（原来 `order_problems` 非空 → return 1）。

`step2d_audit.py` 的 exit 语义（改后，**以模块 docstring 为准**）：

| code | 含义 |
|---|---|
| 0 | ACQ 门 clean，且（若给 `--expect N`）暴露数 == N |
| 1 | 有 ACQ 暴露，或 `--expect` 不符 |
| 2 | usage / 文件不存在 |

ORDER 不再出现在任何退出路径上。

### 2.4 修复指引（durable fix，留给下一个单）

要恢复 ORDER 作为**验收**腿，必须把“线性序比较”换成**按锁的路径检查**：

1. 对每条 `acq L`，在**同一个基本块内**向后找配对的 `rel L`；跨块则用 CFG 求支配关系；
2. 判据：在**任何**从 acq L 到它的 rel L 的路径上，不得出现内核 `jl`（这才是 STEP2C 0d 的失败机制）；
3. 等价的可执行检查：IR 侧按 `peano-linked_*.ll` 的支配树求“acq 之后、rel 之前可达的 call 集合”，
   与 ASM 侧同锁的集合比对**集合**，而不是**序列**；
4. 现有 `_scratch/block/step2d_order_check.py` 是“每站点 ±14 行窗口”的近似探针，可以当骨架，
   但它同样按地址序取窗口，跨块时会误报（见 §2.6）。

在 (1)-(3) 落地前，验收三元组第一条只写 ACQ 门（6 acquires / 0 CLOBBER / 1 informational ARG）。

### 2.5 兼容性影响（逐个查过）

* **调用方**：所有 `*_build_win.py` 只传 `--ll`，**没有任何脚本传 `--order`** —— 自动化的 exit code 不变。
* **JSON**：`order` 字段保留原语义（分歧原文列表，供人看）；新增 `order_gate`
  （`not-run` / `retired-diagnostic-only`）与 `order_diag`（status / ir / asm / added / removed /
  same_multiset）。`webui/t22_gate.py:61` 用 `j.get("order")` 判 FAIL —— 因为没人传 `--order`，
  实际恒为空；但**若将来有人把 `--order` 加进构建脚本，它仍会按老语义报 FAIL**，这一点没改，
  也不该由本单悄悄改（那个脚本不在修改范围内）。
* **`step2d_order_check.py`**（import 了本模块的 `event_order_ir` / `_main_extent` / `ACQ_RE` / `JL_RE`）：
  实测仍可运行（`event_order_ir` 默认 `strict=False`，行为向后兼容）。

### 2.6 一个未验证的观察（**不是**结论）

`step2d_order_check.py` 在修好的 ELF 上对 `acq #0x36` 报 `ORDER VIOLATION`（它前面 0x7c2 有
`jl #0x2320` cast）。按 §2.2 的理由，这同样可能是跨块窗口误报；**本单没有做 CFG 分析，
因此既不能确认也不能否证**。若下一个单做 §2.4 的按锁检查，这一条是第一个应被解释的站点。

---

## 3. ③ id79 归档

单独成文：`docs/OBS-id79-amplitude-ulp-2.000.md`（从磁盘记录直接取出，无手抄）。
要点：`device_vs_model_exact_max_ulp = 271.0`，**幅值界** `max_ulp_mag = 2.000`，门 `<= 2.0`，
零余量；`bit_exact = 0.9988372325897217`；跨进程 raw sha256 两次相同（detpass.json）。
**门限不动**，本单只归档、不调参、不改 `gate_ulp_ok`。

---

## 4. 门表（全部本机实测，2026-10-20）

| 门 | 改前 | 改后 | 判 |
|---|---|---|---|
| `step4_accept.py` 判定口径 | `rel_to_absmax 5.0760e-02 PASS`（记录：STEP4.md:23 / REPRODUCE.md:191） | **文件零 diff**；本机重测未成（见 §5 第 3 条） | 构造性不变，重测列入剩余 |
| `step4_run.py --verify-only` | exit **3**；max ulp 1.661535e+35（x_out）/ 2.199840e+05（Z） | exit **0**；`rel_to_absmax` **7.9365e-03 / 2.7902e-02**（≤ 5.4e-2） | **PASS** |
| `step2d_audit.py`（ELF-only，ACQ 门） | 6 acquires / 0 ACQ CLOBBER / 1 ARG，exit 0 | 同左，逐字一致 | **不变** |
| `step2d_audit.py --ll ... --order` | 7 divergence，exit **1**（FAIL） | 8 处线性差异、**多重集相同**，exit **0** | 退役（不再是门） |
| `event_order_ir` 于 `core_0_3.ll` | `[]`（0 事件） | **7** 事件 | 修好 |
| `t3d_baseline.py --check` | `ALL REFERENCE BYTES UNCHANGED`（改动前跑一次） | `ALL REFERENCE BYTES UNCHANGED`（收尾再跑一次） | **PASS** |
| `py_compile` 两文件 | — | 0 | **PASS** |
| `step2d_order_check.py`（import 本模块） | 可运行 | 可运行（输出同前一节） | 未破坏 |

复现命令（py312 = `<python312>\python.exe`）：

```
cd zimage-bfp16-npu\_scratch\block
py312 step4_run.py --verify-only                      # 期望 exit 0，两行 rel_to_absmax PASS
py312 step2d_audit.py step4_out\s4a_M96_K96_h3_d32_m8_d111.prj\elfs_main_core_0_2\elfs_main_core_0_2.elf \
      --ll step4_out\s4a_M96_K96_h3_d32_m8_d111.prj\peano-linked_main_core_0_2.ll --order
py312 step2d_audit.py <同上 ELF>                        # 期望 exit 0，ACQ 门逐字不变
py312 ..\image\t3d_baseline.py --check                 # 期望 ALL REFERENCE BYTES UNCHANGED
```

---

## 5. 诚实负结果 / 未验证面

1. **ulp 门不是被修好，是被替换。** 按定义它在这个链上不可判定（§1.1），任何“修”都是换量。
   本单换了项目自己的量与阈值，没有新造判据；但这一点必须被记录，不能被读成“ulp 现在能过了”。
2. **ORDER 门不是修好，是退役。** 五个提取器缺陷是真 bug、已修；但“序列比较”这个算法本身不成立
   （§2.2 用同一多重集 + 8 处布局差异证明）。真正的修复是 §2.4 的 CFG 版本，本单没做。
3. **`step4_accept.py` 本机重测失败**，失败发生在任何本单代码之前（它不 import 本单两个文件）：
   第一次 `[step4_oracle] CHECKPOINT QK-NORM UNAVAILABLE (... os error 1455，页面文件太小)` →
   回退 w=1 → 在 `step4_compose.gemm_device` 的 `(96,10240) uint32`（3.75 MiB）分配上
   `numpy ArrayMemoryError`；第二次被系统直接杀掉（无输出，rc=-1）。这是共享机的内存/页面文件压力，
   不是代码回归。5.0760e-02 仍是**记录值**，不是本机复测值。
4. **记录里的 “max ulp 43072” 在今天产物上复现不出来**：实测 1.661535e+35（x_out，1 个零参考元素）
   与 2.199840e+05（Z）。43072 应理解为**历史几何/历史产物**的数（记录在第四轮抽验），
   不应再被当作“当前值”引用。这也从另一面说明 ulp 数在这种链上是产物的函数，不是一个稳定指纹。
5. **`--verify-only` 的 7.9365e-03 / 2.7902e-02 不是验收数**：几何 M=96/K=96、参考是 composed
   device-arithmetic model、3 heads。与 hidden 3840 的 5.0760e-02 不可比，脚本里已写明。
6. `webui/t22_gate.py:61` 仍把 `order` 当门字段（范围外，未动）。没人传 `--order`，实际不影响；
   但这是下一个改 `--order` 的人必须知道的坑（§2.5）。
7. **本单没有任何器件测量**，也没有新增性能/整链结论；结论只覆盖两个工具与一份观测归档。

---

## 6. 精确剩余量

| # | 事项 | 量 | 位置 / 命令 | 范围 |
|---|---|---|---|---|
| 1 | 文档措辞改为“已修 + 新 exit 语义” | 1 段 | `REPRODUCE.md:194-203`（“should be fixed or dropped”那段） | 范围外（仓库根） |
| 2 | KNOWN-ISSUES 两处：提取器已修 / ORDER 已退役 | 2 处 | `KNOWN-ISSUES.md` L1410-1415、L1436-1440 | 范围外（仓库根） |
| 3 | 验收三元组第一条去掉“ORDER 门 FAIL”表述 | 3 处 | `HANDOFF.md` §5；`HANDOFF-SUPPLEMENT-v2.md` §T1；`docs/zimage-bfp16-npu-questions.md` T1 | 前二范围外，第三条在 docs/ 但属上一轮记录，未改 |
| 4 | `step4_accept.py` 本机重测（需足够页面文件/内存） | 1 次 ≈ 2-4 min | `py312 _scratch/block/step4_accept.py`，期望 `5.0760e-02 PASS` | 待环境 |
| 5 | CFG 按锁 ORDER 检查（真正的修复） | 0.5–1 人日 | 见 §2.4；骨架 `_scratch/block/step2d_order_check.py` | 新单 |
| 6 | 门限改动 | **0** | — | — |
