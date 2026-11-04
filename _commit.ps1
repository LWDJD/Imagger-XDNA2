$ErrorActionPreference = "Continue"
Remove-Item _clean.ps1 -Force -ErrorAction SilentlyContinue
& git reset -q 2>&1 | Out-Null
Remove-Item .git\index.lock -Force -ErrorAction SilentlyContinue
& git add -A 2>&1 | Out-Null
$files = & git diff --cached --name-only 2>&1 | Where-Object { $_ }
$sum = 0; $big = @()
foreach ($f in $files) { $p = Join-Path (Get-Location) $f; if (Test-Path $p -PathType Leaf) { $l = (Get-Item $p).Length; $sum += $l; if ($l -gt 1MB) { $big += ("{0} {1:N2} MB" -f $f, ($l/1MB)) } } }
"STAGED: " + ($files.Count) + " files, " + [math]::Round($sum/1MB,2) + " MB"
"files over 1 MB: " + $big.Count
$big | Select-Object -First 5
$msg = @"
Import the runnable core of the Z-Image-Turbo XDNA2 NPU pipeline

Migrated from a 50 GB research tree (78,273 files) that had grown to the point
where production code and dead experiments could not be told apart.

What is here (340 files, 22.9 MB):
  webui.py + webui/    the server, the planner, the shape gates, the capability
                       endpoint, the static UI, and the per-shape compiled device
                       packages (66 xclbin/bin/images, ~8.8 MB)
  imaggers/            the engine, previously _scratch/.  Its directory depth is
                       preserved so __file__-relative paths still resolve; the
                       only edit is the prefix substitution _scratch -> imaggers
  docs/                architecture, weights, build, knobs, measurement discipline,
                       and 23 curated engineering reports
  weights/             ONLY the two small metadata files; the 6.4 GB of npz blocks
                       are gitignored and documented in docs/WEIGHTS.md

What is deliberately absent:
  * weights/block*.npz (6.59 GB), weights.bfp16 (6.91 GB), the text-encoder GGUF
    (8.05 GB), the wpack (1.99 GB), the B-repack cache (10.6 GB) -- all over the
    GitHub limit.  Each is documented with its producing script in docs/WEIGHTS.md.
  * 78,103 files / 19.3 GB of dead experiments, session logs and prediction docs.
  * Third-party trees (llama.cpp, ggml-vulkan, AIE runtime, Peano): referenced in
    docs/BUILD.md, never redistributed.

Verification of this tree, run from this repository after the move:
  preflight          12 evaluated, 0 FAIL -- ACCEPTED
  generation         512x512 x8 steps -> latents 2abf91331200d069 (the reference),
                     loop 94.09 s, total 119.80 s, device released cleanly
"@
& git commit -q -m $msg 2>&1 | Select-Object -First 3
"--- commit ---"
& git log --oneline -2 2>&1
& git show --stat --oneline HEAD 2>&1 | Select-Object -Last 3
