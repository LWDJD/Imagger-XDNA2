# T22 factcards2 — 512 文案更正 + 零回归验证留证(2026-10-20)

> 单元:t22_factcards2(文档收尾单第 ③ 项)
> 变更:把 `config.RESOLUTIONS[512].reason` 里的 L1 数字 **83,456 B → 196,096 B**;
> 同步 `webui/shapes.py` 的 `t22_fmt_l1` docstring、`webui/static/index.html` 的
> `SHAPE_512_REASON` 兜底句、`README-WEBUI.md` §9 限制清单。
> 依据:`webui/evidence/T22-PRODLARGE.md` §7 第 1 条 —— 83,456 B 是 **SEQ=416(256×384)**
> 的值;M=1056 在 Mq=16 下是 **196,096 B**。结论("> 64 KB、不可用")不变。
> 口径:这是**逐字文案变更**,不改任何逻辑/门限。

## 1. 自洽 + 冻结门兼容(纯 Python,无器件)

命令(py312):

    py312 webui/_selftest/t22_factcards2_copy.py

(该脚本放在 `webui/_selftest/`,按 `.gitignore:46` **不入库**,与同目录的
`t22_prod_ui.py` / `t22_prod_api.py` 一致;本文件记录它的命令与输出即为留证。)

结果(2026-10-20,exit 0):

    PASS  config 512 reason carries 196,096
    PASS  config 512 reason no longer carries 83,456
    PASS  shapes 512 ready=false
    PASS  shapes 512 reason carries 196,096 (on-demand path agrees)
    PASS  256x384 (SEQ=416) still computes 83,456 -- the old number belongs to that row
    PASS  corrected 512 sentence still matches _selftest SPEC_RE
          parsed: hw=512² m=1056 tok=1024 l1=196,096
    PASS  SPEC_RE parses l1 as 196,096
    PASS  128 stays ready=true
    PASS  256 keeps its historical reason
    RESULT: OK

要点:**config 与按需路径现在给同一个数**(改前 config 说 83,456、shapes 说 196,096);
256×384 仍然算出 83,456 —— 这正好证明旧数字属于那一行,不是 512 的。

## 2. 前端零回归(离线,无浏览器)

命令:

    py312 webui/_selftest/t22_prod_ui.py webui/evidence/t22_prodlarge/after

结果(2026-10-20,exit 0):

    [ui] ids in markup: 68, referenced by $(): 62
      PASS no duplicate element ids
      PASS every $() id exists in the markup
      PASS budget panel elements present
    [ui] inline script: 27344 B
    [ui] node --check: rc=0
    PASS A1..A6 (128x128 交付尺寸信息 + budget 面板 96 / 180 / 已交付 Mq=96 / 1035 B 表 / 缓存行)
    PASS B1..B9 (2048x2048 探测 + 512-spec 句 + M 16,416 / 2,672,730 / 14.93 h / 不可用 / 1087 B / NOT SHIPPABLE)
    PASS C1..C2 (非法尺寸不臆造数字)
    [ui] UI GATE PASS

## 3. 没有做的事(如实)

* **没有**重跑 `t22_prod_api.py gate BEFORE AFTER` —— 该 gate 的 (b) 项要求两个 capture
  的 512 reason **逐字节相同**,而本单元的全部目的就是改这句话;拿旧 after/ 目录比会**必然 FAIL**,
  那是判据错用,不是回归。上面 §1 用产品的 `SPEC_RE` 直接验证"改后仍匹配冻结的 512 规约",
  §2 用离线 UI 门验证"前端仍能渲染",等价覆盖。
* **没有**起 7873 的 HTTP 服务做一次新 capture(可另立 1 次 ~2 min 的小单);
  因此本文件不含"线上 /api/status 文本"的直接证据,只有 import 级证据。
