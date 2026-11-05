# STREAM-ATTN — 在线 softmax 流式 attention：设计与首建

日期：2026-10-30 22:55 (+08:00)｜owner：t22_bigpipe｜状态：**单块绿 / 多块 carry 未绿**

## 0. 判词

| 项 | 结果 |
|---|---|
| 设计（在线 softmax，S/P 永不超过 BQ×BK） | **完成**（§1–§2） |
| 单镜像 Q/K/V→Y（单 key 块，seq=64，hd=128） | **绿**：corr 0.999778、rel 3.58 %、run-to-run 逐字节相同、ERT 6/6 COMPLETED |
| 多 key 块 carry（seq=128/512） | **未绿**：块 1..n 的 m/l/c 没有折进最终 O；seq128 corr 0.744（bk64）/0.110（bk32），seq512 corr 0.458 |
| N² 不落地 | **结构性成立**：最大 S/P 缓冲 16×64（2 KB bf16），整图无 seq² 尺寸对象；host↔device 只有 Q/KV/O |
| 512² 收益 | **条件性**：host f32 已 191 GFLOPS；器件 <191 GFLOPS 时流式必输。400 GFLOPS 时约 2×，只省 512² 总墙钟的 ~6 % |

## 1. 设计：在线 softmax 的流式形态

对每个 query 块（BQ=16 行）和每个 key 块（BK=64 列）：

    S_j   = (Q * scale) @ K_j^T              # BQ x BK, f32->bf16
    m_j   = max(m_{j-1}, rowmax(S_j))        # BQ
    c_j   = exp2(m_{j-1} - m_j)              # BQ, m 是已乘 scale 的域
    P_j   = exp2(S_j - m_j)                  # BQ x BK, bf16
    l_j   = l_{j-1} * c_j + rowsum(P_j)      # BQ
    O_j   = O_{j-1} * c_j + P_j @ V_j        # BQ x HD, f32
    Y     = O_last / l_last                  # bf16

**关键不变量**：任何时刻 S/P 只有 BQ×BK；K/V 只有 BK×HD；O 是 BQ×HD f32。
没有 `BQ×SEQ`、`SEQ×SEQ` 或 `BK×SEQ` 的对象。

### 1.1 512² 的片上预算（BQ=16, BK=64, HD=128）

| 对象 | 大小 | 说明 |
|---|---:|---|
| Q 块 | 16×128×2 = **4 KB** | 常驻 |
| KV 块（K 半 + V 半） | 2×64×128×2 = **32 KB** | 一个 object fifo |
| S32 | 16×64×4 = **4 KB** | QK 输出 |
| S16/P | 16×64×2 = **2 KB** | softmax 原地 |
| O32 | 16×128×4 = **8 KB** | 在线累加 |
| state（m,l,c） | 4×16×2 = **128 B** | m_prev/m_new/l/c |
| **合计** | **~50 KB** | < 64 KB L1；memtile 不用 |

### 1.2 可复用件（库里的，不新写）

| 件 | 位置 | 用法 |
|---|---|---|
| `partial_softmax_alias_bf16` | `_pylibs/mlir_aie/include/aie_kernels/activation/softmax_aie2p.h` | 现成的块内 running-max + exp2 + block-sum；本 spike 的 `stream_softmax_step` 是它的“显式 state 版” |
| `partial_softmax` / `matmul_PV` / `rescale_O` / `init_scale_buffer` | `aie_kernels/linalg/mha.cc` | 完整的 64×64 在线 softmax 工具箱（含 O 的 c 缩放与 1/l 收尾） |
| `flash_attn_prefill.h` 五步 API | `aie_kernels/linalg/flash_attn_prefill.h` | `round_begin/qk_step/block_mid/fv_step/epilogue`，head_dim 512/256 几何，可直接作为 T2 在线形态的参考实现 |
| `mm_bfp` / `mm_aie2p.h` | `aie_kernels/linalg/` | QK/PV 的 bf16→f32 累加；本 spike 先用 scalar 版（正确性优先），向量化版是下一单元 |
| `kgu64` 的 B 读形态 | `_scratch/block/kgu64/` | 大对象 B 的 per-core 读法；用于把 BK 从 64 拉到 192/256 |

## 2. 首建 spike（已跑）

### 2.1 形态

- 单镜像 `stream_attn.xclbin` + `stream_attn.bin`：1 个 compute core（Tile 0,2），2 个输入 fifo（Q、KV）+ 1 个输出 fifo（O）。
- 形状：`heads=1, seq=64, hd=128, bq=16, bk=64`（一个 key 块；用于验证 kernel 与数据通路）。
- 自定义核：`stream_softmax_step`（在线 softmax 的 state 更新）、`stream_pv_accum`（O=O*c+P@V）、`stream_finalize_cast`（O/l→bf16）、`stream_zero_state`。
- 编译：py312 IRON + Peano，**24–28 s**；xclbin **16,152 B**，insts **1,632 B**。

### 2.2 结果（py310 pyxrt，单进程，6 次 dispatch）

| 量 | 值 |
|---|---|
| 墙钟 | 6.06–7.50 ms，median **6.2 ms** |
| ERT state | 6/6 **ERT_CMD_STATE_COMPLETED (4)** |
| run-to-run | **逐字节相同**（6/6） |
| 对 host f32 参考 | **corr 0.999778**，rel **3.58 %**（bf16 S/P/O 的 F-23 档） |
| host torch 同形状 | 0.0339 ms（device 慢 **183×**，此尺寸无速度意义） |
| host↔device 字节 | Q 4 KB + KV 32 KB + O 4 KB = **40 KB**；**S/P 0 B** |

### 2.3 N² 不落地的硬证据

- 整图最大 S/P 对象 = `16×64 bf16 = 2 KB`；完整 seq=64 的 N² f32 = `64×64×4 = 16 KB`（8× 小）。
- 对 512²：设计最大 S = `16×64×4 = 4 KB`，P = 2 KB；完整 1056² f32 = **4.46 MB/head**（约 **1,115×** 小）。
- host↔device 只有 Q/KV/O 三个对象（40 KB/块）；S/P 在 core L1 内产生并消亡，没有任何 top-level BO 承载它们。
- 口径：这是**按 runtime sequence 的逐笔字节账**，不是硬件 DRAM counter；但“没有 N² 尺寸对象”是设计结构，不是测量推断。

### 2.4 多块 carry 的未绿证据（必须记录）

| 臂 | 形状 | corr vs host f32 | 现象 |
|---|---|---:|---|
| seq64 单块 | 1×64×128, bk64 | **0.999778** | 绿 |
| seq128 两块 | 1×128×128, bk64 | 0.744 | 块 0 对，块 1 未折进 |
| seq128 两块 | 1×128×128, bk32 | 0.110 | 同上；final `l`=23（单块量级），`c`=0 |
| seq512 八块 | 1×512×128, bk64 | 0.458 | 同上 |

**根因定位（已缩到一层）**：`scf.for` 的 key 块循环确实生成 4/8 次（MLIR 可见），但 `stream_softmax_step` 的 `state[0]/[2]/[3]` 在第二次迭代仍读到块 0 前的值（final `l` 等于单块和、`c`=0）。已排除：QK/PV 单块算术（seq64 绿）、V 偏移（改为字节后 zero-Q 与 mean(V) corr 0.999999）、S 未清零（加 `zsf` 后 run-to-run 从 False 变 True）。**未排除**：两个自定义核之间的 L1 Buffer 跨 `scf.for` 迭代的写可见性/别名。下一单元应把 state 改成 `ObjectFifo`（producer/consumer 同核）或用 `mha.cc` 的 `init_scale_buffer/partial_softmax` 标准 state 件复测。

## 3. 512² 收益预测（按实测 host 基线）

host f32 在 512² 实测 **2.69 s/步**（5.14e11 FLOP/步 ÷ 2.69 s = **191 GFLOPS**）。所以流式器件的盈亏线是 **191 GFLOPS**：

| 器件有效吞吐 | 512² attention s/步 | 相对 host | 省下的 step wall（分母 22.22 s） |
|---:|---:|---:|---:|
| 50 GFLOPS（现 ip 量级） | 10.3 | 0.26×（更慢） | −7.6 s（负收益） |
| 191 GFLOPS（盈亏线） | 2.69 | 1.0× | 0 |
| 400 GFLOPS | 1.29 | 2.1× | **1.40 s（6.3 %）** |
| 1 TFLOPS | 0.51 | 5.2× | 2.18 s（9.8 %） |

结论：**512² 的流式 attention 即使做到 400 GFLOPS，也只值 6 % 总墙钟**；它不能解决 512² 的 FFN/投影权重流卡点。1024²/2048² 的 attention 占比 38 %/60 %，同样吞吐下收益分别为 ~18 %/~30 %，才是这条线的正确战场。
