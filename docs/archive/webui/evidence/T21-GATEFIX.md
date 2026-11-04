# T21-GATEFIX — gate 模式计时闭合修复

**结论**:gate 模式闭合残差从 **+11.547 s(HEAD 复现)/ −3.542 s(存档)** 修到 **+0.090 s**,
同 session 实测落入门限 ±1 s;13 桶、前端计时表、门限均未改;latents/pe 逐字节不变。
**NOT SHIPPABLE**:本单不引速度数字,下文所有墙钟仅用于闭合算术。

---

## 0. 环境、夹具与提交

| 项 | 值 |
|---|---|
| repo | `zimage-bfp16-npu` |
| 修复前 HEAD | `8158568` |
| 修复 commit | `f69337c` "t21_gatefix: U3 - stop double-counting the gate second pass in timing closure" |
| 报告 commit | 见 `git log --grep t21_gatefix`(本文件提交) |
| 服务 | py312 `D:\Users\15081\miniforge3\envs\ryzen-ai-1.8.0\python.exe`,隔离端口 **7870**,验证后已杀 |
| 器件 | py310 `D:\Users\15081\AppData\Local\GAIA\python\python.exe`,经 `webui/bridge.py`(未改动) |
| 夹具 | prompt "a mountain lake at sunrise",128×128(M=96),batch 1,steps 8,seed 7,vae **off**,memory_mode auto |
| 门禁 | `py312 _scratch/image/t3d_baseline.py --check`(前后各一次) |

job 记录(均可复核):

| 用途 | job id | 模式 | memory_mode | 闭合 |
|---|---|---|---|---|
| 修复前(before) | `20261019-142440-f5b92b` | gate | low | **+11.547 s FAIL** |
| 修复后(after) | `20261019-142951-bb8dbe` | gate | low | **+0.090 s PASS** |
| trial 对照 | `20261019-143330-c2cc21` | trial | fast | +0.038 s PASS |
| 存档(历史证据) | `20261019-090737-30d768` | gate | fast | **−3.542 s FAIL** |

---

## 1. 根因复核(定案)

代码路径(修复前 HEAD 行号):

* `webui/runner.py:780` `_execute_pass()` 末尾无条件执行 `job["timing"].update(timing)` —— **第二遍(pass1)也执行**,它的 textenc/schedule/startup/loop/device_tail 覆盖了 pass0 的同名桶;
* `webui/runner.py:424` `_execute()` 随后 `job["timing"]["gate_second_s"] = pass1["timing"]["pass_s"]` —— 又把 **pass1 整遍**加进第 12 桶。

于是 13 桶之和 = 一次性桶 + **pass1 各阶段桶 + pass1 整遍**,第二遍被计两次;第一遍在 `gate_second_s` 里没有任何份额,只剩被覆盖后残留的表象。恒等式:

```
sum − 一次性桶  ≈  2 × pass1          (修复前,两遍同一签名)
残差 = total − sum  =  pass0_wall − Σpass1_buckets + 遍间隙 = (pass0 − pass1) + O(0.02 s)
```

**证据 1 —— 存档 job `20261019-090737-30d768`(fast 模式)**:

| 量 | 值 |
|---|---|
| pass0 / pass1 | 65.297 / 68.877 s |
| total / sum | 134.238 / 137.780 s |
| unaccounted | **−3.542 s** |
| pass1 各桶 | textenc 10.269 + schedule 0.003 + startup 29.493 + loop 28.445 + tail 0.644 = **68.854** |
| sum − 一次性(0.049) | **137.731 ≈ 2 × 68.877** |

**证据 2 —— HEAD 修复前同参实跑(before,`20261019-142440-f5b92b`,low 模式)**:

| 量 | 值 |
|---|---|
| pass0 / pass1 | 108.041 / 96.545 s |
| total / sum | 204.633 / 193.086 s |
| unaccounted | **+11.547 s** |
| pass1 各桶 | 11.253 + 0.003 + 37.654 + 47.259 + 0.345 = **96.514** |
| sum − 一次性(0.026) | **193.060 ≈ 2 × 96.545** |

两次实跑残差符号相反,只因 pass0 与 pass1 的相对快慢随内存模式变化(fast: pass1 慢 3.6 s → 负;low: pass1 快 11.5 s → 正);**“sum − 一次性 ≈ 2×pass1”的重复计数签名完全一致**。−3.54 s 不是常数,只是 fast 模式下的一个实例。

---

## 2. 13 桶语义与方案选择

13 桶定义(`webui/runner.py:39-53`)保持不变,前端计时表(`webui/static/index.html:447-461`)由 `closure.buckets` 的 label 动态渲染,无需改表:

| # | 桶 | 标签 | 归属(修复后) |
|---|---|---|---|
| 1 | queue_s | 排队 | 一次性(submit→worker) |
| 2 | prepare_s | 构建缓存 | 一次性 |
| 3 | textenc_s | 文本编码 | pass0 |
| 4 | schedule_s | 调度 | pass0 |
| 5 | retry_s | 冷启动重试 | pass0 |
| 6 | startup_s | 启动/注册/载荷驻留 | pass0 |
| 7 | loop_s | 扩散 | pass0 |
| 8 | device_tail_s | 器件收尾/落盘 | pass0 |
| 9 | vae_load_s | VAE 加载 | pass0(vae 只跑第一遍) |
| 10 | vae_decode_s | VAE 解码 | pass0 |
| 11 | vae_save_s | 保存 PNG | pass0 |
| 12 | gate_second_s | 复现门第二遍 | **pass1 整遍(唯一归属)** |
| 13 | post_s | 任务收尾 | 一次性 |

**候选方案**:

* **A. `gate_second_s` 记“第二遍增量”**(pass1 整遍 − pass1 各桶):能闭合,但第 12 桶标签“复现门第二遍”会名不符实(只含收尾/落盘),需要改前端语义 —— 否决。
* **B. `gate_second_s` 保持“第二遍整遍”,pass1 的各阶段桶不并入 13 桶(标记为非重复)** —— **采用**。标签、桶定义、门限一律不动,唯一改动是一行守卫。
* C. 封账前把 pass0 桶恢复回去:结果同 B,但多一份暂存状态、多一处可漂移点 —— 不取。

实际改动(`webui/runner.py:779-789`,`f69337c`):

```python
        timing["pass_s"] = time.perf_counter() - t_pass
        # T21 gatefix: the 13 closure buckets describe ONE pass. ...
        if pass_index == 0:
            job["timing"].update(timing)
```

**语义不变量**:13 桶 = 一次性桶 + pass0 全部桶 + pass1 整遍(`gate_second_s`);pass1 的分阶段值不再进入 13 桶,整遍值仍由 `job["gate"]["timing"]["second_total_s"]` 记录,无重复、无丢失。

---

## 3. 前后闭合表(同 session 实跑)

夹具同上;before/after 均为 low 模式,机器状态相近。

| 桶 | before 142440 (s) | after 142951 (s) |
|---|---:|---:|
| queue_s | 0.004 | 0.004 |
| prepare_s | 0.019 | 0.019 |
| textenc_s | 11.253 | 10.788 |
| schedule_s | 0.003 | 0.003 |
| retry_s | 0.000 | 0.000 |
| startup_s | 37.654 | 37.383 |
| loop_s | 47.259 | 47.282 |
| device_tail_s | 0.345 | 0.380 |
| vae_load_s | 0.000 | 0.000 |
| vae_decode_s | 0.000 | 0.000 |
| vae_save_s | 0.000 | 0.000 |
| gate_second_s | 96.545 | 95.522 |
| post_s | 0.003 | 0.003 |
| **分项之和 sum** | **193.086** | **191.385** |
| **总计 total** | **204.633** | **191.475** |
| **未归账** | **+11.547** | **+0.090** |
| 闭合门(±1 s) | **FAIL** | **PASS** |

after 的闭合分解(逐秒可复算):

```
一次性桶 0.0263  + pass0 各桶 95.8362  + pass1 整遍 95.5220 = 191.3846 (sum)
total 191.4747 = 一次性 0.0263 + pass0 整遍 95.9098 + 遍间隙 0.0165 + pass1 整遍 95.5220
unaccounted 0.0901 = pass0 内部开销 0.0736 + 遍间隙 0.0165   ✅ |0.090| << 1 s
```

**预测 vs 实测**:

| 项 | 预测(U1,先行) | 实测 |
|---|---|---|
| gate 闭合残差 | +0.04 ~ +0.08 s(对照 trial 口径) | **+0.0901 s**(略高 0.01,来自本次 pass0 内部开销 0.074) |
| 是否落 ±1 s | 是 | **是** |
| gate 判决 | PASS | **PASS**(latents/pe 逐字节相同) |
| 13 桶 | 每桶非负,gate_second 有归属无重复 | **是**(见上表) |

**trial 对照(同参数单遍,`20261019-143330-c2cc21`)**:

| total | sum | unaccounted | 门 |
|---:|---:|---:|---|
| 83.627 | 83.589 | **+0.038** | PASS |

修复前最近 12 个完成的 trial job(`20261019-023928` → `20261019-091323`,fast/low,含 128×128 与 256×256)unaccounted 为 **+0.043 ~ +0.087 s**;本单 trial 对照 +0.038 s 同属该口径,与修复后 gate 的 +0.090 s 同量级,证明 gate 与 trial 的闭合口径已一致。

> 历史样本注记:更早的 20261018~20261019-01xx trial job 有 −0.28 ~ −1.05 s,还有一个 error job(`20261018-200655-fdbded`,sum=0)异常 +48.1 s;这些是修复前旧口径/异常态,不作为本次基线,也不在统计内。

---

## 4. 数值不变性证据

计时改动不得影响数值路径。before/after 两个 gate job 的 **两遍** 产物逐字节比对:

| 文件 | 大小 | sha256 | 比对 |
|---|---:|---|---|
| before 输出 latents(`t21g_gate_before_142440f5b92b_p0_latents.npy`) | 16,512 B | `9f86aee91de3dad09bf888b15fe560e16f9c3079920118d714697f25421dbfe2` | — |
| before 器件 pass0 latents(`w142440f5b92bp0_latents.npy`) | 16,512 B | `9f86aee9…dbfe2` | 同上 |
| before 器件 pass1 latents(`w142440f5b92bp1g_latents.npy`) | 16,512 B | `9f86aee9…dbfe2` | 同上 |
| after 输出 latents(`t21g_gate_after_142951bb8dbe_p0_latents.npy`) | 16,512 B | `9f86aee9…dbfe2` | **完全一致** |
| after 器件 pass0 latents(`w142951bb8dbep0_latents.npy`) | 16,512 B | `9f86aee9…dbfe2` | **完全一致** |
| after 器件 pass1 latents(`w142951bb8dbep1g_latents.npy`) | 16,512 B | `9f86aee9…dbfe2` | **完全一致** |
| before pe_p0.npz | 134,468 B | `437634b665835e3f268cc4cefe29efa884061d9b643ed137e58bf8ff72dd1bdf` | — |
| after pe_p0.npz | 134,468 B | `437634b6…1bdf` | **完全一致** |

补充:两 job 的 `pe` 数组 sha256 均为 `90514d089ec4b35662d81a8ae000ea909dacb39347c4d25d389cd804bbe9a04c`(13 tokens),且与存档 job `20261019-090737-30d768` 相同;两个 gate job 的 `gate.latents_equal=true`、`gate.pe_equal=true`、`gate.status=PASS`。trial 对照 job 的 latents/pe 也与上表一致。**计时语义修改未触动任何数值路径。**

---

## 5. 门禁与纪律记录

| 项 | 结果 |
|---|---|
| `t3d_baseline.py --check`(修复前) | `ALL REFERENCE BYTES UNCHANGED`,rc 0 |
| `t3d_baseline.py --check`(修复后) | `ALL REFERENCE BYTES UNCHANGED`,rc 0 |
| `py_compile webui/runner.py` | rc 0 |
| 写权限 | 仅 `webui/runner.py`、`webui/evidence/T21-GATEFIX.md` |
| 未改 | 13 桶定义、前端计时表、`CLOSURE_TOL_S=1.0`、器件/数值路径 |
| 端口 | 7870 验证后已杀,连接确认关闭 |
| 并发代理 | `_scratch/vae` 未触碰 |

---

## 6. 精确剩余量(交接)

1. **gate FAIL 路径仍未闭合**:pass1 抛错时 `_execute_pass` 不会走到封账,已耗掉的第二遍时间只在 `total_s` 里,不在任何桶 → 失败 job 的闭合仍会 FAIL。建议单独一单:在 gate 的 `except config.WebUIError` 分支记录 `gate_second_s = elapsed`(需在调用前取 `t_gate0`),再做一次“故意失败 gate”的闭合验证。**本单未夹带。**
2. **fast→low 回退路径**:pass0 触发 MemoryFailure 时,失败那次的墙钟未进任何桶(`retry_s` 只记器件重试),回退成功后闭合会偏大。修法同上是“失败尝试计时桶”,属另一单。
3. **gate 第二遍期间的实时 UI**:13 桶现在保持 pass0 值(语义正确),头部 `t-startup/t-loop` 在 gate 阶段不再实时跳 pass1 的数;权威值见 `job.gate.timing.second_total_s`。纯显示口径,未改前端。
4. **VAE-on 的 gate 闭合未实跑**:本轮夹具 `vae=false`(保持单变量;VAE 桶只属于 pass0,逻辑上不受本修复影响)。补验命令:同夹具 `vae=true, mode=gate`,预期 unaccounted ≈ +0.4~+1.0 s(VAE 内部开销),仍需以 ±1 s 门复核。
5. **256×256(M=288)未实跑**:镜像未建,与本修复无关;修复是形状无关的。
6. 磁盘:作业产物约 2 MB;验证结束时 D: 余量仍约 0.9 GiB,无大件残留。

---

## 7. NOT SHIPPABLE 声明

本报告只论证**计时语义正确性与数值不变性**,不提供、不引用任何速度/性能结论。before 与 after 的墙钟差异来自内存模式与机器状态(并发代理在跑编译探针),不可作 A/B 性能比较。

---

## 8. 尾巴收口(t21_gatefix2,2026-10-19)

**结论**:§6 剩余量 **1 / 2 / 4** 已收口。① 故意失败的 gate job 闭合 **+0.047 s PASS**
(`gate_second_s`=11.329 s,失败原因如实);② fast→low 回退的失败墙钟归入 `retry_s`,
单元级实测闭合 **+0.050 s PASS**(本机未触发真实内存墙,见 §8.3);④ VAE-on 的 gate
闭合 **+0.061 s PASS**,VAE 三桶只属 pass0,latents/pe 与 vae=false 同 seed **逐字节相同**。
13 桶定义、标签、前端、`CLOSURE_TOL_S=1.0` 均未改。**NOT SHIPPABLE**:本节不引任何
速度/性能结论,墙钟只用于闭合算术。

### 8.1 提交、环境与夹具

| 项 | 值 |
|---|---|
| 前置 HEAD | `2e25fb9`(f69337c 的 pass0-喂桶口径) |
| 本单代码 commit | `774dde9` "t21_gatefix2: U2 - book failed gate pass and fast->low fallback wall in timing closure" |
| 本文件 commit | `git log --grep t21_gatefix2`(报告提交) |
| 服务 | py312 `D:\Users\15081\miniforge3\envs\ryzen-ai-1.8.0\python.exe`,隔离端口 **7870**;测试启动器见 §8.2,测后已杀 |
| 器件 | py310 `D:\Users\15081\AppData\Local\GAIA\python\python.exe`,经 `webui/bridge.py`(未改动) |
| NPU 缓存 | fresh `NPU_CACHE_DIR`/`NPU_CACHE_HOME` = `_scratch/t21_gatefix2/npu_cache_144833`(0o777) |
| 夹具 | prompt "a mountain lake at sunrise",128×128(M=96),batch 1,steps 8,seed 7,mode gate;memory_mode 见下表 |
| 门禁 | `t3d_baseline.py --check`(改前/改后各一次,§8.7) |

> 并发注记:验证期间另一单提交了 `c4fa8dd`(只改 `_scratch/vae/T21-3B2-PROBE-REPORT.md`,
> 未动 `webui/runner.py`);`774dde9` 的父提交即它,与本单无关。本单对 runner.py 的改动
> 相对 `2e25fb9` 仍是 1 file / +34 −3。

本单四个 job(均可复核):

| 用途 | job id | memory_mode | vae | gate | 闭合 residual |
|---|---|---|---|---:|---:|
| ① pass1 故障注入 | `20261019-144924-d1d6b5` | low | off | **FAIL** | **+0.0474 s PASS** |
| ④ VAE-on gate | `20261019-145141-789528` | low | **on** | **PASS** | **+0.0611 s PASS** |
| 回归(fast) | `20261019-145644-7033ff` | fast | off | PASS | +0.0396 s PASS |
| 回归(low) | `20261019-150017-8f5495` | low | off | PASS | +0.0512 s PASS |

### 8.2 ① gate FAIL 路径(pass1 抛错)

**改动**(`webui/runner.py`,commit `774dde9`):

* gate 调用前取 `t_gate0 = time.perf_counter()`;
* `except config.WebUIError` 分支用 `second_elapsed = perf_counter() - t_gate0` 写入
  `job["timing"]["gate_second_s"]`,并在 `job["gate"]["timing"]["second_elapsed_s"]`
  留权威副本;补一行 `[gate] FAIL after ...` 日志;设置 `_passes_end` 后才封账,
  使 `post_s` 覆盖失败后的保存/收尾;
* 13 桶的 `pass_index == 0` 守卫未动:失败 pass1 的分阶段值仍不并入 `job["timing"]`,
  失败尝试只在 `gate_second_s` 出现一次。

**故障注入(生产代码零钩子)**:`_scratch/t21_gatefix2/serve_inject.py`(untracked,
不提交)在本进程内 monkeypatch `Runner._execute_pass`:仅当环境变量
`T21_FAIL_PASS1_PREFIX` 命中 job prefix 且 `pass_index==1` 时,临时把
`runner.schedule.get_schedule` 换成抛 `config.WebUIError` 的函数。pass1 因此**真实
跑完 textenc**(11.33 s)后在 schedule 处失败。`git grep "T21_FAIL_PASS1|injected pass1"
-- webui/runner.py` 无命中(`grep_rc=1`),钩子不在生产路径;服务以 `runpy` 走
`webui.py` 的 `__main__` 语义,验证后 7870 已杀、连接确认关闭。

**实测 bucket(job `20261019-144924-d1d6b5`)**:

| 桶 | s | 桶 | s |
|---|---:|---|---:|
| queue_s | 0.0020 | device_tail_s | 0.3974 |
| prepare_s | 0.0123 | vae_load_s | 0 |
| textenc_s | 12.0484 | vae_decode_s | 0 |
| schedule_s | 0.0030 | vae_save_s | 0 |
| retry_s | 0 | **gate_second_s** | **11.3288** |
| startup_s | 39.8504 | post_s | 0.0053 |
| loop_s | 47.0526 | **sum** | **110.7001** |
| | | **total** | **110.7475** |
| | | **unaccounted** | **+0.0474(PASS)** |

判读:
* `gate_second_s` = `gate.timing.second_elapsed_s` = 11.3288 s,> 0,正是失败 pass1 的墙钟;
* `textenc_s` 仍是 pass0 的 12.0484(未被 pass1 的 ~11.3 覆盖),sum 也没有出现
  "×2 pass1" 的重复计数签名;每桶非负,sum 逐项复算差 0.000000;
* `gate.status=FAIL`、`detail="second pass failed: injected pass1 failure at schedule ..."`、
  镜像 `gate=FAIL`、`job.error=null`,失败原因如实;job 状态 done(不是 error);
* **预测 +0.05 ~ +0.20 s → 实测 +0.0474 s**,略低于下限(失败后的 post_s 仅 0.005、
  遍间隙小)。失败 job 不只是"保证桶无重复归属",已实际落 ±1 s 门。

### 8.3 ② fast→low 回退的计时归属

**改动**(`774dde9`):

* `_execute` 在第一次 pass0 前取 `t_pass0`;`except MemoryFailure` 里
  `fallback_s = perf_counter() - t_pass0` 并记一行日志;
* 低内存重跑 `_execute_pass(..., tag_suffix="low", carry_retry_s=fallback_s)`;
  `_execute_pass` 以 `timing["retry_s"] = float(carry_retry_s)` 起账,另记信息键
  `fallback_s`。13 桶定义不变;失败 fast pass 从不封账,它的整段墙钟只在
  `retry_s` 出现一次。

**单元级实测**(`_scratch/t21_gatefix2/test_fallback_unit.py`,untracked;真实
`Runner._run/_execute`,只把两次 pass 调用换成假的):第一次 sleep 1.2503 s 后抛
`MemoryFailure`,第二次 sleep 0.55 s 并返回 timing。结果:

| 量 | 值 |
|---|---:|
| 调用次数 | 2(fast 失败 + low 重跑) |
| carry(fallback_s) | 1.2503 s |
| retry_s | 1.2503 s |
| sum / total | 1.7704 / 1.8207 s |
| **unaccounted** | **+0.0503 s(PASS)** |
| 断言 | 8/8 true(含 retry_s 记失败墙钟、fallback_s==retry_s、闭合 PASS、status done、mode low) |

**预测 +0.05 ~ +0.20 s → 实测 +0.0503 s**。**边界(如实)**:本机没有触发真实内存墙
——服务自动模式先选 low(commit 8.2 GB),VAE 之后 commit ≥ 9 GB、auto 选 fast,而该
fast gate job(`20261019-145644-7033ff`)正常完成;强制 low 的回归也正常。因此 ② 是
"真实 `_execute` 路径 + 假 pass 墙钟"的单元级验证,真实回退的绝对墙钟未测;代码路径
与生产共用,没有第二套分支。失败 fast pass 的分阶段值不单独保留(未封账),只保留
整段墙钟与 `attempts` 记录——这是"桶无重复归属"的边界。

### 8.4 ④ VAE-on 的 gate 闭合

job `20261019-145141-789528`:vae=true,memory low,`vae_resident=true`(常驻解码器,
load 为每服务一次性;同服务器后续 job 为 0)。

| 桶 | s | 桶 | s |
|---|---:|---|---:|
| queue_s | 0.0040 | vae_load_s | **24.3710** |
| prepare_s | 0.0233 | vae_decode_s | **0.4556** |
| textenc_s | 9.0102 | vae_save_s | **0.0507** |
| schedule_s | 0.0030 | **gate_second_s** | **83.1336** |
| retry_s | 0 | post_s | 0.0034 |
| startup_s | 36.3874 | **sum** | **200.5402** |
| loop_s | 46.7192 | **total** | **200.6013** |
| device_tail_s | 0.3787 | **unaccounted** | **+0.0611(PASS)** |

* VAE 三桶之和 24.3710 + 0.4556 + 0.0507 = **24.8812 s**,与 `vae_p0.json` 的
  `total_s=24.8812` 完全相等,VAE 内部无漏账;
* 日志顺序:`[vae] resident decoder loaded ...` 与 PNG 行在 `[gate] second pass` 之前;
  pass1 以 `vae=False` 调用(runner 既有代码),其整遍 = `gate_second_s` =
  `gate.timing.second_total_s` = 83.1336 s,pass1 不含任何 VAE 桶,无重复归属;
* 数值:两个 pass 的 latents sha 均 `9f86aee9…dbfe2`,pe sha 均 `90514d08…9a04c`;
  `gate.latents_equal=pe_equal=true`,status PASS;
* 逐字节比对(与 vae=false 的 `20261019-142951-bb8dbe`):

| 文件 | 大小 | sha256 | 比对 |
|---|---:|---|---|
| VAE job p0 latents(`w145141789528p0_latents.npy`) | 16,512 B | `9f86aee9…dbfe2` | 与 ref p0 **bytes equal** |
| VAE job p1g latents(`w145141789528p1g_latents.npy`) | 16,512 B | `9f86aee9…dbfe2` | 与 ref p1g **bytes equal** |
| VAE job pe_p0.npz | 134,468 B | `437634b6…1bdf` | `np.array_equal` 全数组 true |
| VAE job pe_p1g.npz | 134,468 B | `437634b6…1bdf` | `np.array_equal` 全数组 true |
| 输出 PNG(`t21gf2vae_145141789528_p0_s7.png`) | 21,318 B | `104ff564…219a` | 合法 PNG(`\x89PNG\r\n\x1a\n`) |

**预测 +0.1 ~ +1.0 s → 实测 +0.0611 s**,落在门内且接近 vae=false 的 +0.05 口径。

### 8.5 常规 gate 回归

| job | mode | pass_s | gate_second_s | sum / total | unaccounted | gate |
|---|---|---:|---:|---:|---:|---|
| `20261019-145644-7033ff` | fast | 76.8018 | 66.1004 | 142.9013 / 142.9409 | **+0.0396** | PASS |
| `20261019-150017-8f5495` | low | 89.4436 | 91.3828 | 180.8399 / 180.8911 | **+0.0512** | PASS |

两个 job 的 latents/pe 均为 `9f86…` / `90514…`(与 §4 参考一致),gate PASS。
low 回归里 pass1(91.38 s)比 pass0(89.44 s)慢,闭合仍 +0.05:这正好复验
"gate_second_s = 第二遍整遍"的归属与两遍相对快慢无关。与 142951 的 **+0.0901**
同属 +0.0x 口径(机器状态不同:startup 32.94 vs 37.38、loop 45.51 vs 47.28,
不作性能比较)。

**预测 +0.05 ~ +0.15 s → 实测 +0.0396(fast)/ +0.0512(low)**。

### 8.6 三项预测 vs 实测汇总

| 项 | U1 预测(先行) | 实测 | 判定 |
|---|---|---|---|
| ① gate FAIL 闭合 | +0.05 ~ +0.20 s,gate_second>0,原因如实 | +0.0474 s,gate_second 11.3288 s,FAIL detail 如实 | **PASS** |
| ② fast→low 回退闭合 | +0.05 ~ +0.20 s(回退后),无重复 | +0.0503 s(单元级;真实墙钟未触发) | **PASS(单元级)** |
| ④ VAE-on gate 闭合 | +0.1 ~ +1.0 s,VAE 桶属 pass0 | +0.0611 s,三桶和=vae total,pass1 无 VAE | **PASS** |
| 常规回归 | +0.05 ~ +0.15 s | +0.0396 / +0.0512 s | **PASS** |

### 8.7 门禁与纪律

| 项 | 结果 |
|---|---|
| `t3d_baseline.py --check`(改前) | `ALL REFERENCE BYTES UNCHANGED`,rc 0 |
| `t3d_baseline.py --check`(改后) | `ALL REFERENCE BYTES UNCHANGED`,rc 0 |
| `py_compile webui/runner.py` | rc 0 |
| 写权限 | 仅 `webui/runner.py`(774dde9)、`webui/evidence/T21-GATEFIX.md`(本小节) |
| 未改 | 13 桶定义/标签、前端计时表、`CLOSURE_TOL_S=1.0`、器件/数值路径 |
| 注入钩子 | 只在 untracked scratch 启动器;runner 无命中;测后进程已杀 |
| 端口 | 7870 已杀,`Invoke-WebRequest` 连接失败确认关闭 |
| 并发代理 | `_scratch/vae` 未触碰 |
| 磁盘 | D: 余 0.898 GiB;四个 job 产物合计 2.67 MiB |

### 8.8 精确剩余量(交接)

1. **真实 fast→low 内存墙未实跑**:② 为单元级验证(真实 `_execute`,假 pass 墙钟)。
   要实测需真触发 MemoryError,本机当前 commit ≥ 9 GB,强行压测不在本单止损线内。
2. **pass1 的非 WebUIError 失败仍不记 `gate_second_s`**:`Cancelled` 与未预期异常走
   cancelled/error 分支,闭合仍可能 FAIL;本项只覆盖 `config.WebUIError`(含
   MemoryFailure)的 gate FAIL 路径,属既有边界。
3. §6 第 3 项(第二遍期间 UI 实时显示)未动,纯显示口径。
4. §6 第 5 项(256×256 / M=288)未实跑,与本修复无关。
5. 失败 fast pass 只有整段墙钟进 `retry_s`,没有分阶段拆分(未封账);若将来需要
   fast 失败的分项,需另立"失败尝试 partial timing"设计,不在本单范围。

