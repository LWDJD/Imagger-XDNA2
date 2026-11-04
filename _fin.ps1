$ErrorActionPreference = "Continue"
foreach ($f in @("_chk.ps1","_commit.ps1","_final.ps1","_dist.ps1","_clean.ps1","_c2.ps1","_stage.ps1")) { Remove-Item $f -Force -ErrorAction SilentlyContinue }
& git add -A 2>&1 | Out-Null
& git reset -q _msg.txt 2>&1 | Out-Null
$files = & git diff --cached --name-only 2>&1 | Where-Object { $_ }
$sum = 0; foreach ($f in $files) { $p = Join-Path (Get-Location) $f; if (Test-Path $p -PathType Leaf) { $sum += (Get-Item $p).Length } }
"staged: " + $files.Count + " files, " + [math]::Round($sum/1MB,2) + " MB"
$files
& git commit -q -F _msg.txt 2>&1 | Select-Object -First 2
"--- log ---"
& git log --oneline -3 2>&1