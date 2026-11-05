# T21 2a+2b 第二梯队 —— 预测先行 (写于任何测量/实现之前)

锚点: 2026-10-19T22:3x +08:00, HEAD 05bd0c9, 隔离端口 7870。
127x127 交付形状: 128x128 (M=96), 8 步, seed 7, batch 1。

## U1 基线预测 (来自第一梯队 §2 + T13 账本, 未重测)

| 桶 | 预测 |
|---|---|
| py310 进程冷启 (解释器 + numpy/numba import + DLL) | 2.5-4.5 s |
| 1.85 GB W 数组分配 + 30 层 x 4 投影打包 ("warm") | 20-23 s (实测日志 22.2 / 23.5 s) |
| 器件 open + 3 xclbin 注册 | 0.3-0.8 s |
| cache-all 载荷: 30 层 np.load (3.71 GB) + NH=5 置换 | 4.8-5.8 s (T13: 5.4 s) |
| refiner 4 块 f32 缓存载入 (2.83 GB) — 发生在 step 0 内 (loop) | 2.0-2.6 s (4 x 0.6 s) |
| **startup_s 合计** | **30-34 s (快态) / 36-42 s (慢态)** |
| loop_s | 24-26 s (快态), 28-29 s (慢态) |

## U2 预测: 载荷"入 RAM 即置换" (在 cache-all 读的同时置换, 与 W 打包重叠)

做法: run() 一开始就起一个后台线程, 按层 np.load + NH 置换进 CACHE_BG/B3;
主线程继续 W 打包 (22 s); cache-all 汇合点只等剩余部分。

预测:
* 载荷装载+置换 5.4 s **完全藏进** W 打包的 22 s 窗口 -> startup 减少 **4.5-5.3 s**。
* loop 不变 (同样的数组、同样字节), staging 不变, latents/pe 逐字节不变 (构造性: 写进
  CACHE_BG 的字节与旧 cache-all 完全一致, 顺序也一致)。
* 内存峰值不变 (3.71 GB, 只是提前分配; 峰值仍在 W 打包期内, +3.71 GB concurrent 与旧
  路径的 W 2.0 + payload 3.71 peak 相同)。
* 冷路径 (首个 job / worker 未启用) 收益 4.5-5.3 s; 热路径若 worker 已持有 W, 收益 0
  (此时用 U3 的"TE 期间预取载荷"补上)。

## U3 预测: 器件 worker 跨 job 保温 (py310 常驻: device + 3 xclbin + W 打包 + refiner 缓存)

预测 (相对同一 session 的 bridge 冷启):
* startup 33-42 s -> **6-8 s** (只剩载荷 5.4 s + pre() ~0.8 s + 握手 0.2 s);
  W 打包 22 s、进程冷启 3 s、open/注册 0.5 s、refiner 2.4 s 全部移出 job。
* loop 25.5-28.5 s -> **23.1-26.1 s** (refiner 已热, step 0 少 2.4 s)。若 refiner 不保温,
  loop 不变。
* 内存: worker 常驻 W 2.0 + refiner 2.67 + 进程 0.3 = **4.9-5.1 GB commit**;
  idle commit 14.1-14.6 -> **9.0-9.7 GB**, 仍在 config 9.0 阈值边缘 -> 需要实测;
  若 <9.0 则只保温 W (2.3 GB, 11.8-12.3) 或让 worker 自适应释放 refiner。
* 状态: 同一 xclbin 一 handle 不变; FFN 跨 job 不背靠背 (job 末 FFN -> 下一 job 首 proj);
  state-8 0 次; 首 dispatch 0xC0000005 语义 = worker 死则 runner 用新 bridge 进程重试。
* pe/latents 逐字节: worker 复用同一份 W 字节与同一 packed payload (与 bridge 相同函数),
  构造性一致; 门用 sha256 对比。

## 止损线 (任务书)
1. pe/latents 不一致 -> 回退该单元;
2. 内存判定掉出 fast (loop 退化 >10%) -> 停查;
3. 破坏 state-8 防护语义 -> 停;
4. 预算 >1.5 设计日 -> 带数据回来。
