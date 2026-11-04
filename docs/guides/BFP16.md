# BlockFP16 (BFP16) — the weight format

**BFP16 is an 8-bit format.** It is not bf16, it is not fp16, and calling the scale
"an fp16 scale" produces a codec that diverges from the vendor `.rai` on about
0.4 % of groups. This document exists because that mistake was made more than once.

## 1. The layout

```
one group  =  9 bytes  =  1 scale byte  +  8 payload bytes
```

| byte | meaning |
|---|---|
| 0 | biased exponent, unsigned |
| 1..8 | eight int8-range payload values `m` |

decode:

```
value_i = m_i * 2 ** (byte0 - 133)
```

The dequantiser multiplies the group by `2**(byte0-133)`; there is no exponent per
value and no per-value sign bit beyond the payload's own signed range.

**Bytes per parameter: 9/8 = 1.125.** That is the number to use for every capacity
and bandwidth calculation.

## 2. Peak arithmetic on this hardware

```
49.152 TOPS = 32 cores x 512 MAC/cycle x 2 (MAC = 2 ops) x 1.5 GHz
```

32 tiles is the XDNA2 "Strix" arrangement: **8 columns x 4 chained cores**. Each core
has **65,536 B** of L1 — the constraint that decides what fits (a 1056-row attention
stage needs 196,096 B and therefore does not fit, see `docs/reports/ATTN-SEG-REPORT.md`).

The measured ceiling is lower than the arithmetic peak: **0.73–0.76 MMA/cycle**,
i.e. **35.9–37.5 TFLOP/s**, which is what any realistic plan should be built on.

## 3. Why it must not be re-encoded

Re-encoding the payload changes nothing visible — it decodes to nearly the same
floats — but **0.297 % of the stored bytes differ from any re-encode**. Those bytes
are the ones the device multiplies.

Three guards live in `imaggers/block/bigm/brepack_cache.py`:

| guard | line | what it catches |
|---|---|---|
| 64-bit word multiset | :326-343 | any byte-level re-encode |
| layer-0 256-bin histogram | :378 | a value-level re-encode that preserves words |
| Merkle-1 over 16 MiB pieces | :348 | a partial/torn rewrite |

If `--verify` fails, the file is not the file — rebuild it from the source `.npy`
set, do not "fix" it.

## 4. Where BFP16 sits in the precision budget

The project allows **9.73e-2 rel_rms** end to end (`docs/reports/PRECISION-BUDGET.md`).
Spending is tracked per stage, and the two measured spenders are:

| change | cost |
|---|---|
| skip one diffusion step | 7.5 % of the budget |
| fast attention (`T22_ATTN_REUSE` + A2) | 71.3 % of the budget |

They are additive enough that turning both on is not obviously safe — measure the
pair, do not assume.

## 5. Reading order

1. this file — the format
2. `docs/WEIGHTS.md` — the two chains that produce and consume it
3. `docs/reports/BFP16-FORMAT.md` — the long-form original with the vendor comparison
4. `docs/reports/PRECISION-BUDGET.md` — what each stage may spend
