<#
  SPDX-License-Identifier: MIT
  t22_run.ps1 -- the ONLY entry point a fresh agent needs.

      imaggers\repro\t22_run.ps1 -Size 512 -Steps 8 -Tag mytag
      imaggers\repro\t22_run.ps1 -Size 128 -Steps 1 -Tag small

  Three arguments, and they are the whole interface.  This wrapper deliberately
  sets NO environment variable: every switch lives in the RECIPE block at the
  top of t22_repro.py, which also generates RUNBOOK.md from it.  Nothing here
  can drift from the engine because nothing here decides anything.

  -Action doc         regenerate RUNBOOK.md from the engine's constants
  -Action preflight   run the preflight only (no job)
  -Action postflight  re-prove the path of the job already run under -Tag
  -Action status      what the device looks like right now
  -Action cache       resolve the B-repack cache (add -Build to build it once)

  -Extra NAME=VALUE   A/B arm only: one variable set AFTER the recipe and the
                      scrub.  It is printed and stored, and the run is then an
                      ARM, not a reference run (e.g. -Extra T22_BREPACK_CACHE=0
                      measures the un-mapped startup arm).
  -Build              with -Action cache: derive the 5.31 GB file once and
                      report the seconds per stage.

  Exit codes: 0 all checks passed, 2 a check FAILED (read the FAIL lines).
#>
param(
  [Parameter(Mandatory=$false)][string]$Tag = "",
  [Parameter(Mandatory=$false)][string]$Size = "512",
  [Parameter(Mandatory=$false)][int]$Steps = 8,
  [Parameter(Mandatory=$false)]
  [ValidateSet("run","preflight","postflight","doc","status","cache")]
  [string]$Action = "run",
  [Parameter(Mandatory=$false)][string[]]$Extra = @(),
  [Parameter(Mandatory=$false)][string[]]$Skip = @(),
  [Parameter(Mandatory=$false)][switch]$Build
)
$ErrorActionPreference = "Stop"

$here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo    = Split-Path -Parent (Split-Path -Parent $here)
$engine  = Join-Path $here "t22_repro.py"
if (-not (Test-Path $engine)) { Write-Host "[t22_run] engine missing: $engine"; exit 2 }

# The two interpreters the product itself hard-codes (webui/config.py:22-23).
# The engine re-checks which one actually ran (preflight P01).
$cands = @(
  "D:\Users\15081\miniforge3\envs\ryzen-ai-1.8.0\python.exe",
  "D:\Users\15081\AppData\Local\GAIA\python\python.exe",
  "python"
)
$py = $null
foreach ($c in $cands) {
  if ($c -eq "python") { $py = "python"; break }
  if (Test-Path $c) { $py = $c; break }
}
if (-not $py) { Write-Host "[t22_run] no interpreter found (tried: $($cands -join ', '))"; exit 2 }

Set-Location $repo
$argv = @($engine, $Action, "--size", $Size, "--steps", "$Steps")
if ($Tag -ne "") { $argv += @("--tag", $Tag) }
foreach ($e in $Extra) { if ($e -ne "") { $argv += @("--extra", $e) } }
foreach ($s in $Skip) { if ($s -ne "") { $argv += @("--skip", $s) } }
if ($Build) { $argv += @("--build") }
Write-Host ("[t22_run] " + $Action + " size=" + $Size + " steps=" + $Steps +
            " tag=" + $Tag + "  ->  " + $py + " " + ($argv -join " "))

& $py @argv
exit $LASTEXITCODE
