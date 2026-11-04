$ErrorActionPreference = "Continue"
"--- uncommitted ---"
& git status --porcelain 2>&1
"--- tracked helpers? ---"
& git ls-files 2>&1 | Where-Object { $_ -like "_*" }
Remove-Item _z.ps1 -Force -ErrorAction SilentlyContinue
& git add -A 2>&1 | Out-Null
& git commit -q -m "Remove staging helpers" 2>&1 | Select-Object -First 2
"--- final ---"
& git log --oneline -3 2>&1
"status: " + ((& git status --porcelain 2>&1 | Measure-Object).Count)
& git ls-files 2>&1 | Where-Object { $_ -like "_*" }