# SESS-TXNHOST — txn 变体开关找到了（release 模式换 main_common.cpp）

> t22_sessA 第十九轮（2026-10-25）。查 ②：`build_txn_aiert⟧ 的 host_filename 从哪来。

---

## 0. 开关在这（逐字）

`dataflow_common.py:752-773⟧：

```python
def build_sim_overlay(backend, host_filename, compile_flags, ...):
    if backend == BackEnd.Adf:
        build_sim_aiecompiler(host_filename, ...)
    elif backend == BackEnd.TxnHostPatch:
        should_use_common_cpp = False
        if (os.name == "nt" and not gcc_available) or (
            gcc_available and waic_config.mode == "release"
        ):
            should_use_common_cpp = True
        if should_use_common_cpp:
            host_filename = "main_common.cpp"      # <-- 换宿主
        else:
            host_filename = host_filename
        build_txn_aiert(host_filename, compile_flags)
```

**⇒ `main_common.cpp⟧ 不是探针桩，是 vendor 的 release 模式 txn 宿主。**
两条触发条件：①Windows 且 gcc 不可用；②**gcc 可用且 `waic_config.mode == "release"⟧**。

而 `main_common.cpp⟧ 的内容正是：

```cpp
DmaBins bins = run_dma_layer_config();
bins.save();
```

——**这就是四件套的写出点**。

---

## 1. 但这台机器上缺一环

| 检查 | 结果 |
|---|---|
| `main_gemm.cpp⟧ 里的无参 `run_dma_layer_config()⟧ | **被注释（:580-581）** |
| `kernels/⟧ 全树定义 | **0 命中** |
| `kernels/conv/⟧ 定义 | **0 命中** |
| 有参版 `run_dma_layer_config(graph, out, act, const_ptr)⟧ | :635 调用，**编译通过**（说明定义在某个可达头里） |

**⇒ 无参版在我们这棵树上不存在**；release 宿主 `main_common.cpp⟧ 拿不到它。

我们手上的 `main_common.cpp⟧ 只有 373 B，只 include 了 `dma.hpp⟧ + iostream/fstream/sstream，**没有任何层定义**。
vendor 的 release 流程里，这份文件应当是**生成出来的**（带层实现），我们这份是 tiler 输出的空壳。

---

## 2. 两条可行路径

### A. 改 `main_gemm.cpp⟧ 自己写 bins（推荐，可控）

`main_gemm.cpp⟧ 已经能编译能跑，且在 `__TXNRT__⟧ 分支里已经把 `aie_act⟧ / `const_bo_ptr⟧ / `aie_out⟧ 都准备好了（:584-632 写 ifm/wgt/ofm 的就是它们）。
只要把那三份数据按 `DmaBins⟧ 的字段塞进去再 `save()⟧：

```cpp
DmaBins bins;
bins.layer_params = ...;   // param.bin
bins.ctrl_pkts    = ...;   // ctrl.bin
bins.txn_bin      = ...;   // txn.bin
bins.patch_json   = ...;   // patch.json
bins.save();
```

但**这四份数据本身要靠 `run_dma_layer_config()⟧ 里的 txn 生成逻辑产出**，不是我们能凭空填的。所以 A 只解决"写"，不解决"生成"。

### B. 找到 vendor 生成 `main_common.cpp⟧ 的那一步（推荐，正解）

既然 release 模式会换宿主，那 **tiler 必然也能生成带层实现的 `main_common.cpp⟧**。
要找的是 `dataflow⟧ 包里写 `main_common.cpp⟧ 的那段（模板/字符串拼接）。
下一步：`grep -r "main_common.cpp" site-packages/dataflow⟧，看谁写它。

---

## 3. 纪律与边界

- **预测先行**：预判"txn 宿主是另一份文件"——**成立**（`main_common.cpp⟧），且是 release 模式替换而非另存。
- **负对照必配**：有参版能编译（证明头文件链路通）vs 无参版不存在（证明是缺定义不是缺 include）。
- **诚实负结果**：四件套仍未产出；本轮只把"变体开关"定位到 `dataflow_common.py:762-771⟧ 这一处。
- **产品零变化**：只新增本文，未改任何文件。
- **51.3 TOPS / F-14.1**：本轮无吞吐测量、无器件独占，未触碰。
