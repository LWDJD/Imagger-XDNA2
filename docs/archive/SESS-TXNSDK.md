# SESS-TXNSDK — txn 生成被 vendor SDK 缺件挡住（不是我们的代码问题）

> t22_sessA 第二十轮（2026-10-25）。按 B 案找 `main_common.cpp⟧ 的生成点，结果找到了**真正的宿主定义**，
> 但也撞上了这台机器缺 vendor SDK 的硬墙。

---

## 0. 结论

**`DmaBins run_dma_layer_config()⟧ 的定义找到了**（`dataflow/xclbin/dma.hpp:847⟧，330 KB 的 TXN_HPP），
**但它所在的头在 `__TXNRT__⟧ 下强制 include `xaiengine.h⟧，而后者在 Windows 上需要 `<elf.h>⟧ —— 不存在。**

```
dma.hpp:32   #include "xaiengine.h"          <- __TXNRT__ 分支，无守卫
xaiengine.h:11  #include <xaiengine/xaie_clock.h>
xaie_elfloader.h:32  #include <elf.h>       <- Linux ELF 头，Windows 没有
```

---

## 1. 这轮查清的三件事

### 1.1 谁写 `main_common.cpp⟧？——没人写，它是 vendor 的静态 release 宿主

`grep -r "main_common"⟧ 在 `dataflow/⟧ 只有两处：
- `dataflow_common.py:769⟧（替换宿主名）
- `conv/conv_build.py:404⟧（conv 的同款替换）

磁盘上两份 `main_common.cpp⟧（`dataflow/⟧ 与 `dataflow/conv/⟧）**都是 373 B 的同一份**。
它不是生成的，是**静态模板**，靠 `dma.hpp⟧ 提供 `DmaBins⟧ 与 `run_dma_layer_config()⟧。

### 1.2 无参 `run_dma_layer_config()⟧ 在哪？——在 330 KB 的 TXN_HPP 里

| 文件 | 有无参定义 |
|---|---|
| `dmacompiler/host_runtime/dma.hpp⟧（10 KB） | ❌ |
| **`dataflow/xclbin/dma.hpp⟧（330 KB，TXN_HPP）** | ✅ **:847 `DmaBins run_dma_layer_config()⟧** |

**⇒ 我前几轮 include 顺序把 10 KB 那份排在前面，命中的是错的 dma.hpp。**
正确顺序：`dataflow/xclbin⟧ 必须第一。

### 1.3 撞墙：xaiengine.h → <elf.h>

| 尝试 | 结果 |
|---|---|
| `-I dataflow/xclbin⟧ 优先 + `main_common.cpp⟧ | ❌ `xaiengine.h: No such file⟧ |
| 加 `-I aie-rt/driver/src⟧（工作区有 aie-rt 检出） | ❌ 需要 `<xaiengine/xaie_clock.h>⟧ |
| 把 85 个 aie-rt 头拼成 `xaiengine/⟧ 平铺目录 | ❌ **需要 `<elf.h>⟧（Linux ELF 头）** |

**`<elf.h>⟧ 是 Linux 的东西，Windows 上不存在。**
工作区那份 `aie-rt⟧ 是 **Linux 版**；Windows 版 xaiengine SDK 不在机器上（全盘搜 `xaiengine⟧ **0 命中**）。

---

## 2. 这对 C 路线意味着什么

| 项 | 状态 |
|---|---|
| 编译链（main_gemm.cpp，dev 路径） | ✅ **通**（rc=0，M=96/K=3840/N=10240 实测跑通，写 ifm/wgt/ofm） |
| txn 生成（main_common.cpp，release 路径） | ❌ **缺 Windows xaiengine SDK** |
| 四件套 / 补丁表 | ❌ |

**⇒ "g++ 就能生成 txn"这句话要修正**：工具链确实只是 g++，但 **TXN_HPP 依赖 xaiengine 头**，
而 Windows 版 xaiengine SDK 不在这台机器上。**这需要安装/获取 vendor SDK，不是改代码能解决的。**

---

## 3. 三条出路

| # | 做法 | 可行性 |
|---|---|---|
| 1 | 从 RyzenAI 安装包/驱动里找 Windows 版 `xaiengine.h⟧ 与其依赖（`xaie_*.h⟧ 的 Windows 变体） | 若 SDK 随驱动分发则可行 |
| 2 | 用 Linux 侧（WSL/容器）跑 txn 生成——aie-rt 是 Linux 版，`<elf.h>⟧ 天然存在 | **可能最省**，只需 g++ + 这棵 aie-rt |
| 3 | 放弃自产补丁表，回到真编译（AIE 编译器，350–380 s/件） | 有工具链才行 |

**建议 2**：WSL 里 g++ + 工作区这份 aie-rt 头 + TXN_HPP，`<elf.h>⟧ 与 `xaie_*⟧ 都齐，是最短路径。

---

## 4. 纪律与边界

- **预测先行**：预判"tiler 会生成带层实现的 main_common.cpp"——**否证**（它是静态 373 B 模板）。
- **负对照必配**：10 KB dma.hpp 无该函数 vs 330 KB 有（证明前几轮是 include 顺序问题）；Linux aie-rt 缺 Windows 头（证明是平台问题不是版本问题）。
- **诚实负结果**：四件套仍未产出；本轮把"缺什么"从"不知道"收敛到"**缺 Windows 版 xaiengine SDK**"。
- **产品零变化**：只新增 `_scratch/sessA_build6/7/8.py⟧、`_scratch/xaiengine_inc/⟧（85 个头副本）与本文。
- **51.3 TOPS / F-14.1**：本轮无吞吐测量、无器件独占，未触碰。
