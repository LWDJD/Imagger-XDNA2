# Report index

These are the engineering records kept from the port. They are **measurements and
post-mortems, not tutorials** — read them when you need to know *why* a thing is
the way it is, or before you re-open a question that was already settled.

Each file states its own date and whether a number is [M] measured, [A] arithmetic,
[S] structural or [X] not measured. That convention is load-bearing: this project
has a long history of confident numbers that were measured on the wrong array.

## Design and structure

| file | what it answers |
|---|---|
| `ASSEMBLY.md` | how the pieces fit: the block, the plan, the images, the host |
| `PIPELINE.md` | the end-to-end sequence for one image |
| `STAGE3.md` | the stage-3 FFN path in detail |
| `COVERAGE.md` | what is on the NPU and what is not, per stage |
| `BOTTLENECK-MAP.md` | where the wall goes, by segment |
| `ADAPTIVE-NPU-REPORT.md` | the resolution-independent planner |

## Format and precision

| file | what it answers |
|---|---|
| `BFP16-FORMAT.md` | **the 9-bytes-per-8-values format. Read this before touching weights** |
| `PRECISION-BUDGET.md` | how much drift each stage may spend, and the 9.73e-2 total |

## Performance records

| file | what it answers |
|---|---|
| `UTIL-CAMPAIGN-REPORT.md` | the utilisation campaign and its ceiling |
| `FLOOR-VERDICT.md` | why a 10-second image is arithmetically unreachable here |
| `HOSTPROF-REPORT.md` | the host-side profile, segment by segment |
| `SESS-OVERHEAD-ANATOMY.md` | per-step idle, decomposed |
| `OVL-REPORT.md` | host/device overlap: the two segments that can move, and why the others cannot |
| `CONTENTION-REPORT.md` | what happens when two things want the device |
| `ATTN-SEG-REPORT.md` | the four attention variants; **variant D is rejected on quality** |
| `BREPACK-CACHE-REPORT.md` | the content-addressed B-repack cache and its startup win |

## Build and environment

| file | what it answers |
|---|---|
| `BUILD.md` | the device toolchain |
| `ENVIRONMENT.md` | exact versions observed on the reference box |
| `REPRODUCE.md` | how to reproduce a number |

## Operations

| file | what it answers |
|---|---|
| `KNOWN-ISSUES.md` | **read before debugging anything** |
| `README-WEBUI.md` | the WebUI |
| `WEBUI-CAPABILITY-REPORT.md` | what the capability endpoint decides and why |
| `DEVKILL-REPORT.md` | the devkill image pair path |

## What was deliberately NOT kept

The original tree held 76 root reports. Left behind: fifteen `SESS-*.md` session
logs, five `*-PREDICTIONS.md` documents whose predictions are superseded by the
reports that followed them, and the transient `HANDOFF*` / `RESUME.md` /
`WORKLOG.md` chain. They are history, not documentation, and keeping them would
recreate exactly the "which of these is current?" problem this repository exists
to fix.
