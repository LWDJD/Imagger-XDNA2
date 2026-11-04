$ErrorActionPreference = "Continue"
Remove-Item _y.ps1 -Force -ErrorAction SilentlyContinue
& git rm -q --cached _y.ps1 -ErrorAction SilentlyContinue 2>&1 | Out-Null
& git add -A 2>&1 | Out-Null
& git commit -q -m "Drop the last staging helper" 2>&1 | Select-Object -First 2
"--- final state ---"
& git log --oneline -4 2>&1
"status: " + ((& git status --porcelain 2>&1 | Measure-Object).Count) + " uncommitted"
"tracked files: " + ((& git ls-files 2>&1 | Measure-Object).Count)
"tracked helpers: " + ((& git ls-files 2>&1 | Where-Object { $_ -match "^_" } | Measure-Object).Count)
$f = Get-ChildItem . -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch "\\.git\\" }
"working tree: " + $f.Count + " files, " + [math]::Round((($f | Measure-Object Length -Sum).Sum/1MB),2) + " MB"