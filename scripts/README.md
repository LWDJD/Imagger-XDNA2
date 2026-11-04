# Scripts

The entry points live next to the engine, not here, because their path arithmetic
(`Split-Path -Parent` twice, `parents[N]`) assumes their original depth:

| script | purpose |
|---|---|
| `imaggers/repro/t22_run.ps1` | the one entry point: `-Action run|preflight|postflight|doc|status|cache` |
| `imaggers/repro/t22_repro.py` | the verification engine (preflight P01-P12, postflight Q01-Q18) |
| `imaggers/repro/RUNBOOK.md` | generated runbook, 65 KB |
| `webui.py` | start the WebUI |

## The three commands you will actually use

```powershell
# 1. prove the path without generating anything (host-only, no device)
.\imaggers\repro\t22_run.ps1 -Action preflight -Size 512 -Steps 8

# 2. generate
.\imaggers\repro\t22_run.ps1 -Size 512 -Steps 8 -Tag mytag

# 3. run an A/B arm (the name must already be in the recipe, or it is filtered out)
.\imaggers\repro\t22_run.ps1 -Size 512 -Steps 8 -Tag arm1 -Extra 'ZIMAGE_SKIP_STEPS=7'
```

## Which check means what

**Preflight P01-P12** runs before the job and decides whether the run may start. The
ones that fail for real reasons:

| check | what a failure means |
|---|---|
| P01 | the repo tree or one of the two interpreters is missing |
| P03 | a stale `device.lock` -- usually a wedged earlier run |
| P05 | the shape is not renderable; the reason names the limit |
| P08 | a device image the path registers is absent |
| P09 | the wpack or the B-repack cache is missing or stale |
| P11 | the last run of this recipe+shape did not prove its path (a WARNING, not a gate) |

**Postflight Q01-Q18** runs after the job and decides whether its number is usable.
`Q12` is the latents sha256 against the reference table; `Q06` checks that the FFN's
devkill rows partition `M` per layer, which is what catches a missing block after
the projection was split.

## Arm flags and the trap

An arm's flags are recorded in `imaggers/repro/last/<tag>.preflight.json` under
`extras`, and echoed in the job log as `[extra] ARM OVERRIDE ...`.

**`job.json`'s `env_flags` is a different, fixed list (`LAUNCH_FLAGS`) and will NOT
show an arm flag.** Checking it and concluding the flag did not land is a mistake
this project has made; checking it and concluding the flag DID land is worse. See
`docs/MEASUREMENTS.md` section 5.