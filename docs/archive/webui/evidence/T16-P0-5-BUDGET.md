# T16-P0-5 — 任意宽高:尺寸算术、预算、边界与构建单元

产品需求:"为什么不能自定义设置图像的长宽,我需要随时切换"。
本文件回答三件事:(1) 尺寸怎么算;(2) 为什么"任意尺寸"是一次构建而不是一个
下拉框;(3) 哪些代码已落地、哪些还差、边界在哪。

## 0. 本轮落地状态

| 项 | 状态 | 证据 |
|---|---|---|
| UI 宽/高两个数值输入 + 预设 | **已落地** | webui/static/index.html(`#height` / `#width` / `#presets`) |
| M 算术 + 校验(16 倍数、64..1024、M 64..4128) | **已落地** | webui/shapes.py |
| 任意尺寸按 (H,W) 解析 | **已落地** | `GET /api/shapes`;`POST /api/jobs` 返回 400 + 构建计划 |
| 按尺寸缓存 `webui/cache/{H}x{W}/` + sha256 manifest | **已落地(128x128 实测)** | webui/cache.py;cache/128x128/manifest.json |
| LRU 10 尺寸 / 1 GiB 预算 | **已落地** | cache._lru_evict;API limits |
| 交付 manifest 入库 | **已落地** | webui/shape_manifests/128x128.json |
| 新尺寸的构建器(6 镜像 + perm + host 常量) | **未落地** | 见 §4;它是设计单元,不是 --M 参数 |
| 每尺寸模型门/逐字节门/audit/首图并排 | **未落地** | 同上 |
| 设备验证 | **本轮后半段设备栈卡死** | xrt_coreutil.dll+0x1228ce;见 T16-REPORT §3.3 |

**重要:没有把"任意尺寸"做成一个会失败的输入框。** 输入任意合法 (H,W) 都会得到
该尺寸的 M、对象预算和构建计划;只有已构建的尺寸(当前 128x128)能提交。这是
"首次构建提示"的诚实版本:提示里是 6 镜像 + perm + host 常量的清单,而不是
静默降级/放大。

## 1. 尺寸算术(以 checkpoint 的实际口径)

    latent        = (H/8, W/8)              # Z-Image VAE 8x down
    image tokens  = (H/16) * (W/16)         # patch 2 on the latent
    padded tokens = ceil(tokens/32) * 32    # SEQ_MULTI_OF=32;x_pad_token, pos (0,0,0)
    M             = 32 + padded             # 32 行 caption + image 行

| H×W | latent | patch grid | image tokens | padded | M | 相对 128 的 M |
|---|---|---|---|---|---|---|
| 64×64 | 8×8 | 4×4 | 16 | 32 | 64 | 0.67 |
| 64×256 | 8×32 | 4×16 | 64 | 64 | 96 | 1.00 |
| 128×128 | 16×16 | 8×8 | 64 | 64 | 96 | 1.00 |
| 192×128 | 24×16 | 12×8 | 96 | 96 | 128 | 1.33 |
| 128×256 | 16×32 | 8×16 | 128 | 128 | 160 | 1.67 |
| 256×256 | 32×32 | 16×16 | 256 | 256 | 288 | 3.00 |
| 256×384 | 32×48 | 16×24 | 384 | 384 | 416 | 4.33 |
| 512×512 | 64×64 | 32×32 | 1024 | 1024 | 1056 | 11.00 |
| 1024×1024 | 128×128 | 64×64 | 4096 | 4096 | 4128 | 43.00 |

约束(webui/shapes.py):

* H、W 必须是 16 的倍数且 64..1024;
* M 必须是 16 的倍数(FFN 的 m=16 row tile),区间 64..4128;
* **当前交付的 host 代码没有 image-token padding 分支**(M=96 的 tokens=64 恰好
  是 32 的倍数),任意尺寸要先补这个分支(x_pad_token 行、pos (0,0,0)、pad mask)。

## 2. 对象预算(每图/每 dispatch)

D=3840,FFN=10240,bf16/BFP16 流 2 B/元素;权重按图重读,与 M 无关。

| 对象 | 公式 | M=96 | M=288 | M=1056 | M=4128 |
|---|---|---:|---:|---:|---:|
| A 一行 | 7680 B | 7.5 KB | 7.5 KB | 7.5 KB | 7.5 KB |
| A 合计 | M×7680 | 0.74 MB | 2.21 MB | 8.11 MB | 31.7 MB |
| SBG | M×3×3840×2 | 2.21 MB | 6.64 MB | 24.3 MB | 95.1 MB |
| Y | M×7680 | 0.74 MB | 2.21 MB | 8.11 MB | 31.7 MB |
| 每层权重(BG+B3) | 常数 | 132.7 MB | 132.7 MB | 132.7 MB | 132.7 MB |
| loop 估计 | 27.7×(M/96) s/图 | 27.7 s | 83 s | 305 s | 1191 s |

BD / L1 / RX 的口径来自 T13/T10 的实测:per-row A 对象、memtile BD cap
30,720 words、C 走 memtile 4 维 walk、compute tile L1 64 KB、m=16/k=8/n=64 的
mm_bfp tile。M 只放大宿主 BO 与 C walk 的重复次数,不改变单次 tile 形状;因此
"任意 M" 的第一道墙不是 L1,而是**每图的宿主 pack/传输时间**和 **SBG 的 BO**。
上表的 loop 估计是线性外推,不是实测;新 M 的第一张图必须先跑单图 bench。

## 3. 极端比例:64×256 与 1024×1024

* **64×256**:面积与 128² 相同 → **M=96**,patch grid 4×16。这是最便宜的新形状:
  镜像可能完全复用(M 相同),只差 host 的 RoPE/index map 与 4×16 的 walk 常量。
  仍需要 geometry 门:同一 M 不代表同一位置编码,必须用 host f32 并排验证。
* **192×128 / 128×256 / 256×384**:非正方形、M 变化,走完整构建单元。
* **1024×1024 (M=4128)**:SBG 95.1 MB、A/Y 各 31.7 MB,宿主 BO 仍在 Windows
  进程可分配范围;但 loop 线性外推 ~20 min/图,且 cache-all 需要 ~6 GB +
  refiner 2.67 GB,需要单独内存预算。**不宣称可用**,只作为上限。
* **M=64(64×64)**:最小可用形状,只有 16 个 image token + 32 caption;语义上
  模型没见过这么小的图,图像质量不承诺,只作为下界。

## 4. 每个新 M 的构建单元(为什么不是 --M 一个参数)

1. FFN 镜像(t11_nh5 等价物)按 M 编译:`t4_compile.py --M` 已经有这个旋钮;
2. attention 镜像 t7_p5 按 M 编译(当前 96);
3. attention IP 镜像 t7_ip 按 M 编译;
4. 4 个投影(按任务书口径,和 FFN/IP 一起构成 6 镜像);
5. perm/packer/perm64 的 M 版本(交付的是 96x128、96x64);
6. host 常量:patch grid、RoPE index map、SBG/walk、image-token padding、
   层循环里的 96/64/32 硬编码(t13_run.py / t3_real.py,交付只读,需要新驱动);
7. 每尺寸门:blockcheck(该形状 vs diffusers)、模型门(逐位)、audit 全 core、
   run-to-run 逐字节、首图与 host f32 并排(PRODUCT-CHECKLIST 四件套)。

这就是 T13 §13.3 说的"完整设计单元"。本轮只把 **1-6 的接口、缓存、UI 和门框架**
做出来;镜像与 host 驱动本体没有半建。

## 5. 缓存 / LRU / 磁盘

* 目录:`webui/cache/{H}x{W}/`;manifest v2 记录 shape/h/w/m + 每个文件的
  size/sha256/source;命中只校验 sha256,不重拷。
* LRU:`cache/lru.json` 记录 last-used;超过 10 个尺寸或 1 GiB 构建产物预算时,
  从最旧的开始删(当前只在 ensure 成功后触发;不会删 refiner cache)。
* 交付 manifest:`webui/shape_manifests/128x128.json`(入库),记录交付镜像的
  sha256;cache.py 的 copy-vs-source 校验仍是运行时硬门。
* `.gitignore`:cache 目录不入库;`shape_manifests/` 入库。

## 6. API

    GET  /api/shapes                    预设 + M + ready + 缓存状态 + 构建计划
    GET  /api/cache                     按尺寸的缓存状态
    POST /api/cache/prepare {height,width}   准备/校验一个已构建尺寸
    POST /api/cache/clean  {height,width}    删一个尺寸(无 body = 全删)
    POST /api/jobs {height,width,...}   已构建 → 202;未构建 → 400 + missing 列表

## 7. 结论

* 任意宽高在 **接口/算术/缓存/门框架** 层面已就绪;
* 真正生成任意尺寸需要 §4 的构建单元 + 一台未卡死的设备;
* 本文的 M/对象/loop 是纸面预算,任何新尺寸的第一张图必须重新 bench 并过四件套,
  不得引用本文的估计值当实测。
---

## 8. 切片 1 实测:64×256(M=96,host geometry)

**结论:PASS。** 同一批交付镜像(M=96),只改 host 几何,64×256 出图,四件套过。

### 8.1 实现(只 patch 接缝,不改交付文件)

bridge 新增 `_install_shape_geometry(height,width,...)`:

* `t3_real.IMG_GRID = (1, 4, 16)` → `real_ids()` 的 RoPE 图像坐标(z=33,y=0..3,x=0..15);
* `t3_real.unpatchify` 包装 → 强制 (gh,gw)=(4,16),覆盖 t13_run 里写死的 `unpatchify(...,8,8)`;
* 初始 latent 经交付 driver 的 `--resume` 接缝注入:t13_run 硬编码 `(1,16,16,16)`,
  bridge 写 `<tag>_ckpt_lat.npy`((B,1,16,8,32))+ `<tag>_ckpt.json`(next_step=0),
  driver 自己加载,交付文件不动。

**为什么 perm/packer/SBG/walk 不用按形状生成**:它们是 **M 与行序**(64 image +
32 caption)的函数,不是 2D 网格的函数。M 不变、行序不变 → 交付的 96x128/96x64
perm 与 SBG/walk 原样复用。只有 M 变化(切片 2)才需要按 M 生成。

### 8.2 缓存

`webui/cache/64x256/`(manifest v2,shape=64x256,m=96),复用交付的 8 个 M=96 文件;
命中只校验 sha256。`/api/shapes` 该尺寸 `ready=true`。

### 8.3 四件套门

| 门 | 结果 | 证据 |
|---|---|---|
| run-to-run 逐字节 | **PASS**:两次独立 job latents `array_equal=True`,sha256 `0df40ca3c18baa0a`,shape (1,16,8,32) | `_scratch/image/t16_bytegate64.py` |
| 模型门(对 host f32) | **PASS**:同 seed/prompt/latent0,host f32 8 步 vs 器件;latent corr **0.96598**;图像 PSNR **20.70 dB**,mean abs 0.0362;host final rms 0.9828 vs 器件 0.9776 | `T16-P0-5-slice1-64x256-host-vs-npu.png` |
| audit 全 core | **N/A(复用交付镜像)**:没有新 xclbin,交付时的 core audit 不变 | 交付 t7_p5/t7_ip/t11_nh5 |
| 首图 host f32 并排 | **PASS** | `T16-P0-5-slice1-64x256-host-vs-npu.png` |

20.70 dB 与 128² 的 28 dB 同属 BFP16 权重/算术 vs host f32 的差异,不是形状回归:
latent corr 0.966、构图一致(山脊/日出/倒影位置相同)。

### 8.4 计时与产物

* 器件 loop 26.91 s(单图,N=1;128² 是 27.7 s,同 M 同量级);
* 计时闭合 PASS:Σ 140.69 + 未归账 **-0.16** = 140.53;
* PNG 256×64,22,481 B,sha256 `6ab2218205774112...`;
* 一次首 dispatch 冷启动重试;另一次 job 的首 attempt 挂了 573.7 s 后重试成功
  (环境挂起,记档,不算失败)。

### 8.5 复现

    py312 webui.py --port 7860
    POST /api/jobs {"prompt":"a mountain lake at sunrise","height":64,"width":256,
                    "steps":8,"seed":0,"vae":true}
    # 第二次同参数 -> latents 逐字节相同
    py312 _scratch\image\t16_hostref.py --pe webui\jobs\<job>\pe_p0.npz ^
          --tag t16host64 --height 64 --width 256 --steps 8 --seed 0
    py312 _scratch\image\t16_cmp64.py
---

## 9. 切片 2/3 的实测修正(2026-10-18)

M=128 的第一个动作(FFN 镜像)在 aiecc 放置阶段失败:

    error: 'aie.buffer' op could not be placed: buffer "midB_3_cons_buff_0"
    needs 2048 bytes and this tile has no room left for it

Kc=384 同样失败(Kc=480 先被 `Kc % kgu == 0` 拒绝)。结论:交付 FFN 设计的
L1 预算只装得下 6 个 row tile(M=96);M=128 的 8 个 row tile 需要**设计变体**,
不是 --M 重编译。这把"6 镜像 = 按 M 重编译"的预算假设修正为"6 镜像 = 设计变体 +
重新 audit"。完整剩余量、顺序与复现命令见
`T16-P0-5-SLICE2-REMAINING.md`。

切片 1(64×256)不受影响:它复用 M=96 镜像,已 PASS。
