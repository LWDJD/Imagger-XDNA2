# SESS-TXNBUILD — 编译入口找到了，而且比 350–380 s 便宜得多

> t22_sessA 第十五轮（2026-10-24）。按①查编译入口。
> **结论：DD 的 txn 生成不走 aiecompiler，走 g++；350–380 s 是 VAIP 路径的数字，被误用到 DD 上了。**

---

## 0. 入口在哪（逐字证据）

`probe_txn_build.py:5-24⟧ 记录了 `dataflow_common.py:752-773⟧ 的分派：

```
BackEnd.Adf           -> build_sim_aiecompiler()   (aiecompiler + aiesimulator)
BackEnd.TxnHostPatch  -> build_txn_aiert()         (g++ only)
```

且 `dmacompiler/host_runtime/txn.hpp:64-67⟧ 逐字写着：

```
"we are not linking ADF or AIECOMPILER libraries during bin generation."
```

`build_txn_aiert⟧（`dataflow_common.py:831-896⟧）做的事：用 g++ 编译 host .cpp，**然后运行那个二进制**，
二进制自己写出 `txn.bin / param.bin / ctrl.bin / patch.json⟧（`dma.hpp:13-34 DmaBins::save()⟧）。

**⇒ 生成一个 key 的四件套是"编译一个 .cpp 并运行它"，不是 350–380 s 的 AIE 编译。**
那 350–380 s 是 **VitisAI/VAIP** 路径（`.rai⟧ 那次），我前几轮把它错套到 DD 的 txn 生成上了。

---

## 1. 这条路当年卡在哪（探针的真实结果）

`_probe_out/txn_build/last_gpp_link.log⟧：

```
main_common.cpp:1:10: fatal error: dma.hpp: No such file or directory
```

`_probe_out/txn_build/last_txn_dma_run.log⟧：

```
[AIE ERROR] XAie_SetIOBackend():456: Invalid backend request
```

### 1.1 第一个卡点已可修（本轮查证）

`dma.hpp⟧ **就在机器上**，两处：

| 路径 | 大小 |
|---|---:|
| `site-packages/dmacompiler/host_runtime/dma.hpp⟧ | 10,065 B |
| `site-packages/dataflow/xclbin/dma.hpp⟧ | 330,081 B |

探针失败只是**没把这两个目录加进 `-I⟧**。

### 1.2 第二个卡点是真的

全盘搜 `libxaiengine*⟧ 与 `xaiengine⟧ 目录：**0 命中**。
而 vendor 的链接行带 `-lxaiengine -lLibCdo⟧。所以：

- 若 host .cpp 真的引用 `XAie_*⟧ 符号 ⇒ 链接必失败；
- 若只是头文件里声明、实际不调用 ⇒ 可过（这正是探针想验的）。

`last_txn_dma_run.log⟧ 的 `XAie_SetIOBackend⟧ 错误说明**它当年至少跑到了运行阶段**——
即链接**曾经成功过**，只是运行时 backend 没配好。**这是好消息。**

---

## 2. 成本重估

| 路径 | 每 key 成本 | 依据 |
|---|---|---|
| 我上一轮说的"真编译 350–380 s × 8 ≈ 50 min" | 350–380 s | **错用了 VAIP 的数字** |
| **实际：g++ 编译 host .cpp + 运行** | **秒级–十几秒** | `build_txn_aiert⟧ 只是 g++ |

**⇒ 8 个 key 可能是一两分钟的事，不是 50 分钟。**

---

## 3. 交给父会话的脚本（RUN-TXNBUILD.ps1）

脚本做三件事，每步都有明确判据：

| 步 | 动作 | 成功判据 |
|---|---|---|
| 1 | 用 vendor flag + **补上两个 dma.hpp 目录** 编 `main_common.cpp⟧ | 产出 `txn_dma.exe⟧ |
| 2 | 运行它 | 写出 `txn.bin/ctrl.bin/param.bin/patch.json⟧ 四件套 |
| 3 | 校验 `patch.json⟧ 的 `ctrl_pkt_patch_info⟧ 非空 | 非空 ⇒ 真补丁表到手 |

**为什么仍要提权**：g++ 要写临时文件、且 vendor 的 `-L⟧ 路径可能落在 `C:\temp⟧⟧；同 `RUN-ELEVATED.ps1⟧ 协议。

---

## 4. 纪律与边界

- **预测先行**：预判"编译入口是 aiecompiler、成本 350–380 s/件"——**否证**（是 g++，成本秒级）。这条更正很重要。
- **负对照必配**：`last_txn_dma_run.log⟧ 有 `XAie_SetIOBackend⟧ 运行期错误 ⇒ 反证链接曾成功（否则到不了运行）。
- **诚实负结果**：脚本**未执行**；`xaiengine⟧ 库缺失仍是风险；8 个 key 的四件套未产出。
- **产品零变化**：只新增 `RUN-TXNBUILD.ps1⟧ 与本文。
- **51.3 TOPS / F-14.1**：本轮无吞吐测量、无器件独占，未触碰。
