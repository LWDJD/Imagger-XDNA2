# SESS-TXNBUILD2 — 编译推进到"只剩 3 个调度器宏"

> t22_sessA 第十七轮（2026-10-25）。骨架已由父会话建好，本轮实跑编译。

---

## 0. 进度（每步都实测）

| 步 | 结果 |
|---|---|
| 骨架 junction 验证 | ✅ 四个 include 全部命中 |
| `std::string⟧ include 顺序 | ✅ 预包含后消失 |
| `K_GEMM_SUBV_A16W8⟧ 等 | ✅ 从节点 JSON 的 `COMPILE_FLAGS⟧ 注入后满足 |
| `RESHAPED_*⟧ 三个宏 | ❌ **未解决**（JSON 里没有，全盘 prebuilt 也没有） |

---

## 1. 骨架验证（父会话建的 junction 是对的）

```
OK  ..\..\kernels\conv\gemm_host_runtime.hpp
OK  ..\..\kernels\pwl_nlf\lut_host_runtime.hpp
OK  ..common\test_matrix_utils.hpp
OK  ..\..\kernels\pwl_nlf\gelu_lut_bf16_512.h
```

三条 junction：`kernels\conv⟧→SP`kernels\conv⟧、`kernels\pwl_nlf⟧→SP`kernels\pwl_nlf⟧、`WAIC_Outputs\common⟧→SP`test\common⟧。**全部生效。**

---

## 2. COMPILE_FLAGS：从节点 JSON 直接注入（新发现）

`MatMul_0.json → testbench_args.COMPILE_FLAGS⟧ 有 **26 个**宏，包括：

```json
"M_GEMM_A16W8": 96,        "K_GEMM_A16W8": 3840,   "N_GEMM_A16W8": 10240,
"M_GEMM_SUBV_A16W8": 24,   "K_GEMM_SUBV_A16W8": 64, "N_GEMM_SUBV_A16W8": 128,
"DRAM_IFM_SIZE": 737280,   "DRAM_WGT_SIZE": 44236800, "DRAM_OFM_SIZE": 1966080,
"PERMA": "0,1,2", ... (共 26 个)
```

**注意 `M_GEMM_A16W8 = 96⟧** —— 这是**独立证据**：tiler 当年就是按我们的 **M=96** 几何跑的。
（对 B3/padding 议题是旁证：M=96 在这条链上是原生可表达的。）

注入方式：数值型 `-DKEY=VAL⟧、字符串型 `-DKEY="VAL"⟧。注入后所有 `*_GEMM_*⟧ 宏都满足。

---

## 3. 剩下的缺口：3 个调度器宏

```
main_gemm.cpp:66: error: 'RESHAPED_A_OUTSHAPE' was not declared in this scope
main_gemm.cpp:67: error: 'RESHAPED_B_OUTSHAPE' was not declared in this scope
main_gemm.cpp:68: error: 'RESHAPED_Y_INSHAPE' was not declared in this scope
```

查证：
- `OGOAT\Collaterals\MatMul_kernel_metadata.yaml:125⟧ 的 CFLAGS 列表里**有**这三个名字；
- 但 `MatMul_0.json⟧ 的 `COMPILE_FLAGS⟧ **没有**；
- 全盘 4,000+ 个 prebuilt json **一个都没有**带这三个值。

**⇒ 这三个宏由调度器（scheduler）在编译前算出，不是 tiler 输出的一部分。**
我们手上的 `MatMul_0.json⟧ 缺了这一层。

---

## 4. 下一轮的三个选项（按代价）

| # | 做法 | 代价 | 风险 |
|---|---|---|---|
1 | 从 `OGOAT⟧ 源码里找 `RESHAPED_*⟧ 的计算式（`scheduling_engine⟧ / `build_gemm_m4_n4X.py⟧） | 小 | 需读懂 reshape 语义 |
2 | 用 `in_act_shape_orig⟧ 等已知量手工推（`[96,3840]⟧ → reshape 四元组） | 小 | 猜错则静默算错 |
3 | 跑一次完整 tiler+scheduler 重新生成 `MatMul_0.json⟧（带 scheduler 段） | 中 | 需 OGOAT 全链可用 |

**推荐 1**：`build_gemm_m4_n4X.py⟧ 里已经有 `--Xpreproc="-DK_GEMM_SUBV_A16W8=..."⟧ 的构造，同一处大概率也有 `RESHAPED_*⟧。

---

## 5. 纪律与边界

- **预测先行**：预判"补完骨架即可编译"——**部分成立**（include 全过、宏过了一大半），剩 3 个调度器宏。
- **负对照必配**：三处报错逐层不同（std::string → GEMM_SUBV → RESHAPED），每层都被单独修掉，证明不是同一原因反复。
- **诚实负结果**：`txn_dma_gemm.exe⟧ **未产出**，四件套未产出，补丁表仍为空。
- **产品零变化**：只新增 `_scratch/sessA_build2.py⟧、`sessA_cflags.py⟧、`sessA_reshape.py⟧、`sessA_resh2.py⟧、`_scratch/build_gemm.ps1⟧ 与本文。
- **51.3 TOPS / F-14.1**：本轮无吞吐测量、无器件独占，未触碰。
