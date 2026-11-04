# SESS-G2d — 盘上救件路线走到尽头：txn 二进制自身不自洽

> t22_sessA 第十四轮（2026-10-24）。按代价序查全树编译中间产物、配对、救 cast。
> **结论：盘上凑不出可用三件套，cast 也要进"需真编译"清单。**

---

## 0. 逐层推进（每层都真实前进了一步）

| 轮 | 报错 | 含义 | 修法 |
|---|---|---|---|
| 上一轮 | `unexpected negative offset⟧ | meta 描述的 ctrl 与包里的 ctrl 不是同代 | 三件套整取 `gl16_pack⟧（✅ 已修） |
| 本轮 | `curr_argidx(4) >= args_map size(4)⟧ | **txn 二进制引用第 5 个 arg，但自身只声明 4 个** | ❌ **不可修** |

**关键点**：`args_map⟧ 来自 **txn 的 `.bin⟧ 本身**，不是来自 meta。我给 meta 加第 5 个 buffer（`super_instr⟧）**没有改变任何东西**——
证明问题在 **txn 二进制**，不在我产出的 meta。

---

## 1. 全树配对结果（①）

扫描 `zimage-port/_probe_out⟧ + `zimage-bfp16-npu/_scratch⟧ 全部 `*_ctrl_meta.json⟧ 与 `*_ctrl.bin⟧：

- 有 ctrl_meta 的 base：**9**；有 ctrl.bin 的 base：**11**；两者都有：**9**。
- 按"max_offset < ctrl_size"判据，**14 对通过**。

| base | 来源目录 | maxoff | ctrl | 判定 |
|---|---|---:|---:|---|
| `cast_a16bfacc16bfp_1056_3840⟧ | gl16_pack / gl16_pack | 8,388 | 8,496 | **OK** |
| 同上 | gl16_pack / gx_pack1 | 8,388 | 8,784 | OK |
| 同上 | gl16_pack / pack·cast·cast_real | 8,388 | 9,024 | OK |
| 同上 | gx_pack1 / gl16_pack | 8,676 | 8,496 | **OVERFLOW** |
| `cast_gl12fixB⟧ | gl16_pack / gl16_pack | 8,388 | 8,496 | OK |
| `cast_regen_orig / cast_restore / restore2 / restore3⟧ | gl18–gl21 | 8,676 | 8,784 | OK |

**⇒ 偏移判据能选出 14 对"看起来配套"的件，但偏移判据不足以证明可用**（§0 的 argidx 错误在偏移通过之后才暴露）。

---

## 2. 救 cast 的结果（②）

整取 `gl16_pack⟧ 的 `.bin⟧ + `_ctrl.bin⟧ + `_ctrl_meta.json⟧（同目录、同代次），转换后装进 fix5：

```
same-generation triple available for cast_a16bfacc16bfp_1056_3840 : True
packed: {'empty': 7, 'real(20)': 1}
```

加载结果：**`curr_argidx(4) >= args_map size(4)⟧** —— txn 自身声明 4 个 arg 却引用第 5 个。

**⇒ 这份 txn 二进制是坏的**（或至少与这个节点不匹配）。盘上没有任何一件能救回来。

---

## 3. 需真编译清单（④）

| key | 状态 | 依据 |
|---|---|---|
| `cast_a16bfacc16bfp_1056_3840⟧ | **需真编译** | 盘上三件套自洽但 txn 二进制 argidx 越界 |
| `cast_a16bf16..._1056_3840⟧ | 需真编译 | 无任何真补丁表 |
| `gemm_..._1_1056_3840_10240⟧ | 需真编译 | 无真补丁表 |
| `gemm_..._1_1056_10240_3840⟧ | 需真编译 | 同上 |
| `gemm_..._1_1056_3840_3840⟧ | 需真编译 | 同上 |
| `gemm_a16bfw16bfpacc16bf_1_1056_3840_3840⟧ | 需真编译 | 同上 |
| `layernorm_..._1_1056_3840⟧ | 需真编译 | 同上 |
| `elwmul_..._1_1056_3840___1_1056_3840⟧ | 需真编译 | 同上 |

**8 个 key 全部进清单**，无一旁路。真编译成本：**350–380 s/件**（历史实测），且缓存已证可落盘（`SESS-PACK3⟧）。

---

## 4. 对 C 路线的净影响

| 项 | 状态 |
|---|---|
| schema 转换（G2 前半） | ✅ 完成，工具可用 |
| 盘上救件（G2 后半） | ❌ **失败**，8 个 key 全需真编译 |
| 80 子图加载 | ❌ 未达成 |
| 26+ 规模点 | ❌ 未取 |
| C 路线 ≈4–5 s | 仍是**机制已证、工程未通** |

**⇒ C 路线的下一步是且只是：真编译这 8 个 key（约 8 × 350–380 s ≈ 47–51 min，一次性，可缓存）。**
没有更便宜的路了——盘上已经穷尽。

---

## 5. 纪律与边界

- **预测先行**：预判"盘上能找到配套三件套救回 cast"——**否证**（偏移判据通过，但 txn 二进制 argidx 越界）。
- **负对照必配**：偏移判据在官方件上通过且能跑；在我们这份上通过却仍失败 ⇒ **证明该判据不充分**，已写进 §1。
- **诚实负结果**：80 子图未加载、26+ 未取；cast 也进真编译清单。
- **产品零变化**：只新增 `_scratch/sessA_pairing.py⟧、`sessA_fixpack5.py⟧、`sessA_load80c.py⟧ 与本文；
  所有重建在 `plugins_fix5/⟧。
- **51.3 TOPS / F-14.1**：本轮无吞吐测量、无器件独占，未触碰。
