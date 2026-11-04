# HOSTKILL-PREDICTIONS — 砍非 attention 地板（t22_hostkill）

写于 2026-10-31 16:20 (+08:00)，所有本轮 job 之前。口径：512² / 8 步 / seed 7 / VAE off /
ZIMAGE_BIGM=1 / ZIMAGE_BIGM_PROJ=1 / T22_BIGM_DEVHANDOFF=1 / host f32 attention。

| # | 预测 | 置信 | 依据 |
|---|---|---:|---|
| P1 | 基线 512² 8 步 = 212.9 s（非 attn 191.4 s），稳态 25.1 s/步 | 0.95 | 512_base_8b 既有实测 |
| P2 | 投影 host 段 = a_pack 13.4 ms×2 + C decode 8.3 ms×8 ≈ 93 ms/层；加上 o 调用里多算的 q/k/v 三个 dispatch（~99 ms/层），共 ~192 ms/层可砍 | 0.8 | 1-step T22_BIGM_DEBUG 逐 dispatch |
| P3 | 把 a_pack/反 tile 交给 shim DMA（X 平铺 + C 行主序）后，这两段 host 归零；投影 dispatch 8→4；投影 0.39→0.19 s/层 | 0.7 | mm_bfp 的 A 布局就是 [m/8][k/8][8][8]，TAP 可表达 |
| P4 | 512² 8 步 loop 落到 165–175 s（−18~22%），稳态 17–18 s/步 | 0.6 | P2/P3 |
| P5 | 110 s L1 门仍不过：即使 host 残余全归零，器件段（投影 86 ms + FFN 118 ms + attention 85 ms + misc）地板 ≈ 120–130 s | 0.65 | 每层实测分解 |
| P6 | 融合 QKV（单 W 流 49.8 MB）在 big-M 上会更慢：49.8 MB 超出 L2 驻留，11 个 row-tile 会把 W 从 DDR 重读，实测会掉进 7 s watchdog | 0.5 | W 16.6 MB/投影时 21 ms；49.8 MB 单流风险 |
| P7 | 质量：plain 投影与 rt17+a_pack+_proj_decode 逐字节相等，端到端 corr 只差跑间方差，≥0.99/28 dB | 0.9 | 同字节、同算术 |

负对照预登记：把 rt17 的 raw C 直接当行主序读（不做反 tile）→ corr 应 ≈0；若 plain 输出与它高相关，说明 C tap 没生效。
