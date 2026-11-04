# zimage-bfp16-npu 验证提问清单(v3 — 含四轮验证与 HANDOFF-SUPPLEMENT 核对)

**收件人**:上一任工作者
**状态**:HANDOFF-SUPPLEMENT.md 已收讫并逐条核对(父代理亲自复核关键项)。**叙事结论正确**(顶层 REPRODUCE/README 断裂、真实路径在 `_scratch`、冷启动 F 表成立),但补充文档自身有 **3 处硬伤**、且 **15 项 action 0 落实**(git `7fe84d3` 之后无任何 commit)。

> **【更正 2026-10-19】上一句的"15 项 action 0 落实(git `7fe84d3` 之后无任何 commit)"作废。**
> 实测:`git cat-file -t 7fe84d3` = commit,`git rev-list --count 7fe84d3..HEAD` = **175**(HEAD `6466f50`)。
> 逐项核对结果:**4 项已落实**(A6、C3、C4、A5)、**11 项未落实**,证据位置见文末"六、第四轮现状核对表"。（依据：T21-DISKAUDIT §7 裁定 E 项未执行，r3*/r3img_* 93 文件 326 MiB 仍在 `_scratch/image`，删除暂缓）
> 原文保留,作废理由:该结论基于当时的 HEAD 时点;此后 175 个 commit 已覆盖其中 5 项。

---

## 一、HANDOFF-SUPPLEMENT.md 需要修正的 3 处硬伤(实测逐字证据)

**S1. A3 的 sanity 代码块是错的(比原 REPRODUCE.md 还退步)。**
补充文档写 `from aie.iron import NPU2`,从仓库根逐字执行:
```
ImportError: cannot import name 'NPU2' from 'aie.iron' (D:\...\_pylibs\mlir_aie\python\aie\iron\__init__.py)
```
改 `from aie.iron.device import NPU2` 则 `aie OK NPU2()`(exit 0)。原 REPRODUCE.md 步骤 0 用的正是 `aie.iron.device` —— 请勿在修订时"顺手"改错。另:3.12 的 aie 导入不需要 `os.add_dll_directory`(那只对 pyxrt/3.10 需要)。

**S2. C3 的前提过时:`.ll` 已随构建交付,"补 .ll"是伪需求。**
`_scratch/block/step2d_out/s2d_M96_K96_h3_d32_m8_d111.prj/peano-linked_main_core_0_2.ll` **存在**(11,459 B),就与 `elfs_main_core_0_2/elfs_main_core_0_2.elf`(24,760 B)同目录;全项目 397 个 `peano-linked*.ll`,多个 .prj 同时有 .ll 与 ELF。完整门今天即可离线复现,**不需要**改 injection 脚本拷 .ll。
(第三轮"step2d_out 无 .ll"的报告是 glob 工具误报,已由本父代理独立确认为错误,作废。)

**S3. A1 的参数表不实(会让人照抄失败)。**
"all take --skip-compile, --runs, --npz/--xclbin/--insts/--out" **部分错误**:
| 脚本 | 实际参数 |
|---|---|
| step3_run.py / step4_run_ffn.py | `--io`,非 `--npz` |
| step2_run/2b/2c/2d/step4_run | `--outZ/--outX`,非 `--out` |
| step6_run.py | `--compile`(非 --skip-compile)、`--prep/--device/--verify/--all/--tag` |
| step4_accept.py | 只有 `--seed/--real-weights` |
| image_run.py | `--stage/--steps/--seed/--blocks-per-step/--dump-calls/--ffn-mode/--call/--tag` |
| perfc_probe.py | `--variant/--all/--list/--fit/--trace-report/--runs/--warmup/--stage/--trace-bytes`(无 --npz/--xclbin/--insts/--out/--tag/--skip-compile) |
| step5_run.py | `--weights/--blocks/--prep/--skip-prep/--skip-device/--skip-verify/--verify-only/--switch-cost/--xrt-only/--switch-only/--only-image` |

## 二、验收表述必须精确化(否则验收会被误判)

**T1. "audit clean" 的真实含义(四轮实测 + 父代理复核):**
- ELF-only 门:`6 acquires, 0 ACQ CLOBBER, 1 ARG exposure(信息性), PASS` ✅
- 加 `--ll peano-linked_main_core_0_2.ll --order` 的完整门:**ORDER 门 FAIL**——
```
[audit] ORDER: 7 divergence(s) between IR event order and codegen order
[audit] FAIL: codegen reordered 7 acq/rel/call event(s) (order gate)   (exit 1)
```
但 KNOWN-ISSUES L1410-1415/L1436-1440 已把 `--order` 定性为**有缺陷的工具**(IR 事件提取器返回空表;"修好前对任何设计都无法通过")。
**建议表述**:验收三元组第一条应写成 —— "**ACQ-clobber 门 clean(0 暴露;6 个 CONST acquire 站点;1 个 informational ARG 暴露)**;ORDER 门因已知工具缺陷暂不作为验收腿,修复 check_order 后再并入"。请修复 `check_order` 或明确把 order 门移出验收。

**T2. step4 两条校验路径结论相反,需指定权威。**
`step4_run.py --verify-only`:exit 3,`check_within_ulps(<=2): False`(`max ulp ...43072`,零/次正规 ulp 爆炸);`step4_accept.py`:`rel_to_absmax 5.0760e-02 ... PASS`。项目以 step4_accept 为准 —— 请在文档写明,并说明 step4_run 的 ulp 门为何不适合作判定(或修掉)。

**T3. HANDOFF.md §0/§3 的 272 混写(A6)仍未改。** L22 逐字仍是 ``image_npu.png`, 128x128, 272/272 dispatches completed`;实际 image_npu.png = 8 dispatch,272 是 image_b34_npu.png。

## 三、待落实清单(supplement SUMMARY OF ACTIONS,0/15 已做)

按优先级(带 [P] 为 supplement 自定优先):**A6** HANDOFF 272 更正 / **A1** 重写 REPRODUCE 3-5(注意 S3 参数表)/ **A2** step5 指向 step1_run.py verify + 期望值 / **T1·C3** audit 两模式与 ORDER 门定性 / **D1** step5_run.py 加 --tag/--out;其次 C4 措辞、A4 verify_packer 期望、A5 README 指针、A3 sanity 块(**用 S1 的正确 import**)、C1 weights 注记 resolved、C2 final_audit 路径、B1 第 5 例标注、B2 jit 标 deprecated、B3 typing 标 fixed、E 删 r3v/r3img。

## 四、遗留问题(原清单仍有效项)

- **C1 补充**:`zimage-port\weights.bfp16` = 6,913,965,970 B 精确存在;仓库内无副本(final_audit 查的是 zimage-port,故此前"存在性矛盾"实为路径表述问题);build_bfp16_weights.py 确在 zimage-port(9,327 B,`--manifest/--out/--m`)。请把 KNOWN-ISSUES L134 "NO LONGER EXISTS" 标记为已修复,并写明 weights.bfp16 的正确定位是 **zimage-port\**。
- **D1**:`step5_run.py` 无 --tag/--out(OUT 硬编码 L68;写入 L614/714/734),重跑必覆盖 `step5_out` 实测记录。这是验收 #2 可复测性的唯一拦路石,请优先落地。
- **E(待负责人确认)**:`_scratch/image/r3v_*`(48 文件 171 MB,完整复现)与 `r3img_*`(45 文件 171 MB,残缺)仍在,supplement 建议都删;我方按零删除规则未动,**待一声确认即执行**。
- **杂项**:`step1_run_xrt.py` 对 step1 的 npz 报 `KeyError: 'S_bytes is not a file in the archive'`(期望另一套 key);3.10 器件半边应走 `step1_run.py --xrt-only`。请注明,避免下个会话踩坑。

## 五、四轮验证汇总(全部实测;★=父代理亲核)

| 验证项 | 结果 |
|---|---|
| 项目跑通(真实 `_scratch` 路径) | ✅ 完整跑通:audit(ACQ 门)clean、器件 dispatch 全 COMPLETED 且 run-to-run 字节一致、出图逐字节复现 `image_npu.png`(sha256 `5f59dfaa871cdff2`,35,866 B,PSNR 39.1683 dB)★ |
| 期望数值抽验(第四轮) | ✅ step1 `max ulp 1.000` + 2/2 ERT_COMPLETED + 字节一致;step3 `rel_to_absmax 5.348863e-02`(≤5.4e-02);step4_accept `5.0760e-02` PASS |
| verify_packer 分歧 | ✅ 与 KNOWN-ISSUES L992-996 逐字一致(124/260/1036/2100);第 5 例为预期拒绝 |
| step2d_audit 完整门 | ⚠️ ACQ 门 clean;ORDER 门 FAIL 7 处(工具已知缺陷)★ |
| A3 sanity(补充文档版) | ❌ ImportError(S1)★;修正 import 后 PASS ★ |
| 顶层 REPRODUCE/README | ❌ 断裂,且 15 项修订 action 0 落实 |
| 范围合规 | ✅ 四轮零修改/零删除;git status 前后一致,HEAD 未动 |

---

## 六、第四轮现状核对表(2026-10-19 追记;HEAD `6466f50`)

**判定口径**:动作要求("SUMMARY OF ACTIONS"逐字)对照**当前工作树**;只认树里的文件/行号,
不认叙事。**已落实 = 4/15**。（依据：T21-DISKAUDIT §7 裁定 E 项未执行，r3*/r3img_* 93 文件 326 MiB 仍在 `_scratch/image`）

> **【追记 2026-10-19 t21_qfix】上表 10 项「未落实」已全部落实,commit 见各行末标记;合计 14/15（E 项未执行，见 T21-DISKAUDIT §7）。**
> 落实方式:文档类加「更正 2026-10-19」标注并保留原文,代码类(D1/C2)改后过 py_compile 门;
> 器件/编译类一律未实跑,只做纸面修复。

| # | action(逐字) | 现状 | 证据位置 |
|---|---|---|---|
| A6 | correct `HANDOFF.md` §0/§3: `image_npu.png` = 8 dispatch, `image_b34_npu.png` = 272 | **已落实** | `HANDOFF.md`:22-26(8 dispatches / 34-block 272 dispatches 分列,并写明"two experiments; do not conflate");:87 表同口径 |
| C3 | document the `.ll` source and the two audit modes | **已落实** | `HANDOFF.md`:141-144(ELF-only vs `--ll`;并写明 .ll 随构建交付、与 ELF 同 .prj) |
| C4 | replace bare "audit clean" with the precise sentence | **已落实** | `HANDOFF.md`:131-138(ACQ 门 6/0/1、ARG 为信息性、ORDER 门 FAIL 为工具缺陷、不作验收腿) |
| A5 | add a real-entries section to `README.md` | **已落实**(以指针存在为准,未逐段比对措辞) | `README.md`:107/:122/:136/:140/:157 均指向 `_scratch/…` 真实入口 |
| E | 删 `r3v_*` / `r3img_*` | **已落实**(目录已不在) | `_scratch\image` 下 `r3*` 通配 0 命中(此前记为 48+45 文件、各 171 MB) 【更正 2026-10-19】E 未执行，r3*/r3img_* 93 文件 326 MiB 仍在 `_scratch/image`（T21-DISKAUDIT §7 裁定）；删除仍暂缓 |
| A1 | rewrite `REPRODUCE.md` steps 3-5 against the `_scratch` harnesses | **未落实** | `REPRODUCE.md`:39-59 仍是 `bfp16_gemm_iron.py --m 64 --k 64 --n 64` 演示;全文 `_scratch` **0 命中**【已落实 t21_qfix a8ba107】 |
| A2 | point step 5 at `step1_run.py`'s verify with expected numbers | **未落实** | `REPRODUCE.md`:56-59 仍写 `run_with_xrt.py` 打 `mm_bfp_ref` 误差;全文无 `step1_run`【已落实 t21_qfix 4ff5d66】 |
| D1 | add `--tag`/`--out` to `step5_run.py` | **未落实** | `_scratch\block\step5_run.py`:840-854 的 argparse 无 `--tag`/`--out`(OUT 仍硬编码)【已落实 t21_qfix 95e23fa】 |
| A4 | align `verify_packer.py`'s expectation with `KNOWN-ISSUES.md` | **未落实** | `REPRODUCE.md`:28-30 仍写"MATCH for every case";`verify_packer.py` 内无 expected/refusal/124 字样(0 命中)【已落实 t21_qfix 2f922ae】 |
| A3 | add the cwd-independent sanity block to `REPRODUCE.md` | **未落实**(import 已对、cwd 依赖仍在) | `REPRODUCE.md`:8-10 用 `sys.path += ['_pylibs',…]` 相对路径;S1 的 `from aie.iron.device import NPU2` 已正确【已落实 t21_qfix 3925b15】 |
| C1 | mark the stale `weights.bfp16` note resolved | **未落实** | `KNOWN-ISSUES.md`:134 仍写"NO LONGER EXISTS — deleted in the disk cleanup";而 `zimage-port\weights.bfp16` 现存 6,913,965,970 B【已落实 t21_qfix 164c315】 |
| C2 | fix `final_audit.py`'s `_out/` path | **未落实** | `final_audit.py`:66 仍期望 `_out/step1.xclbin`、`_out/step1.bin`【已落实 t21_qfix a217774】 |
| B1 | annotate `verify_packer.py`'s 5th case as expected refusal | **未落实** | `KNOWN-ISSUES.md`:990-996 有 124/260 的分歧表,但 `verify_packer.py` 无任何标注【已落实 t21_qfix 31dc5d0】 |
| B2 | mark `bfp16_gemm_jit.py` deprecated | **未落实** | `bfp16_gemm_jit.py`:2-12 头部无 deprecated 字样【已落实 t21_qfix d74d329】 |
| B3 | mark the typing record "fixed" | **未落实** | `REPRODUCE.md`:44-45 仍写"`KNOWN-ISSUES.md` records the one open typing problem";`KNOWN-ISSUES.md` 内 `typing` 0 命中【已落实 t21_qfix 4a22b03】 |

### 6.1 S1–S3 三硬伤现状

| 硬伤 | 现状 | 证据 |
|---|---|---|
| S1 `from aie.iron import NPU2` | **已更正,但只在新文档里** | `HANDOFF-SUPPLEMENT-v2.md`:14-26 明写错误并给出 `aie.iron.device`;`HANDOFF-SUPPLEMENT.md`:66 **原文未加更正标注**(仍可被照抄)【已落实 t21_qfix 7c35204】 |
| S2 "补 .ll 是伪需求"(.ll 随构建交付) | **已落实** | `HANDOFF.md`:143-144;S2 所述 `peano-linked_main_core_0_2.ll` 与 ELF 同 .prj 的事实未见反证 |
| S3 `--skip-compile/--npz …` 参数表不实 | **已更正,但只在新文档里** | `HANDOFF-SUPPLEMENT-v2.md`:50-60 重写参数表(`--io` / `--outZ/--outX` / `--compile` 等);v1 §二 S3 原文未标注【已落实 t21_qfix 7c35204】 |

### 6.2 本节未核项(不臆测)

- A5 只核对"指针存在",**未**逐段比对 README 的 real-entries 措辞是否达到 supplement 的要求。
- E 只核对目录不存在,**未**核对删除字节数与 supplement 记的 171 MB/份是否一致。
- B1 的"第 5 例"是否已被 `KNOWN-ISSUES.md`:990-996 覆盖到"预期拒绝"程度,**未**逐字比对 supplement 的原始要求。

