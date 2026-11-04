# Documentation

There are **472 Markdown documents** in the original research tree. All of them are
here, sorted by how much you should trust them as *current*.

| folder | files | what it is | when to read it |
|---|---:|---|---|
| **[guides/](guides/)** | 6 | written for this repository | start here |
| **[reference/](reference/)** | 2 | generated reference tables | when you need a flag or an endpoint |
| **[reports/](reports/)** | 24 | curated engineering records + an index | when you need to know *why* |
| **[archive/](archive/)** | 450 | everything else, paths mirroring the original tree | when you need the history |

The point of the split: the original tree had no way to tell which of 472 documents
was current. `guides/` is current by construction; `reports/` is dated and states its
own evidence class; `archive/` is kept for completeness and should be read as
history.

## guides/ — read these first

| file | answers |
|---|---|
| [ARCHITECTURE.md](guides/ARCHITECTURE.md) | what runs where, the 20-segment chain, the device model, resolution independence |
| [BUILD.md](guides/BUILD.md) | the two-interpreter split, the NPU toolchain, how the images were built, the GPU options |
| [WEIGHTS.md](guides/WEIGHTS.md) | the two weights chains, the two hard constraints, **and the blocker** |
| [BFP16.md](guides/BFP16.md) | the 9-byte/8-value format and why it is not bf16 |
| [MEASUREMENTS.md](guides/MEASUREMENTS.md) | how to measure without fooling yourself |
| [GPU.md](guides/GPU.md) | the iGPU situation, including what is broken and why its numbers are void |

## reference/ — look things up

| file | contents |
|---|---|
| [KNOBS.md](reference/KNOBS.md) | all 177 environment variables, classified by whether they can move a byte |
| [API.md](reference/API.md) | the WebUI's HTTP surface |

## reports/ — the engineering record

[INDEX.md](reports/INDEX.md) explains each of the 24 and, importantly, **what was
deliberately dropped** and why. Reports state their own date and mark every number
`[M]` measured, `[A]` arithmetic, `[S]` structural or `[X]` not measured. That
convention is load-bearing in a project with this history.

Worth reading early:

* [KNOWN-ISSUES.md](reports/KNOWN-ISSUES.md) — before debugging anything
* [BFP16-FORMAT.md](reports/BFP16-FORMAT.md) — the long-form format derivation
* [README-original-tree.md](reports/README-original-tree.md) — the original root README,
  which carries the BFP16 tiling derivation (`slot = (r//8)*32 + (c//8)*8 + (r%8)`) that
  the new format note only summarises

## archive/ — history, not guidance

[archive/INDEX.md](archive/INDEX.md) lists all 450 with sizes. The folder structure
mirrors the source tree, so a reference to `_scratch/par/gate/REPORT.md` is at
`docs/archive/_scratch/par/gate/REPORT.md`.

What is in there, by kind:

| kind | where | why it is kept |
|---|---|---|
| session logs (`SESS-*.md`, 15) | `archive/` | the raw record of how conclusions were reached |
| predictions (`*-PREDICTIONS.md`, 5) | `archive/` | paired with the report that tested them |
| handoffs and worklog | `archive/` | continuity across sessions |
| per-experiment reports (328) | `archive/_scratch/…` | each `_scratch/par/<topic>/REPORT.md` is the evidence for one decision |
| WebUI and GPU notes | `archive/webui/`, `archive/_gpuwork/` | subsystem detail |

## What is NOT backed up, and why

The repository holds code, compiled device images and documentation. It does **not**
hold the large artifacts, because GitHub cannot:

| artifact | size | reproducible? |
|---|---:|---|
| `weights/block*.npz` | 6.38 GB | **NO — see WEIGHTS.md §4** |
| `weights.bfp16` | 6.91 GB | **NO — see WEIGHTS.md §4** |
| text-encoder GGUF | 8.05 GB | yes (shards present) |
| `wpack_l30_v2.bin` | 1.90 GB | yes, ~33 s from `weights.bfp16` |
| B-repack cache | 10.6 GB (2 files) | yes, but only with the same payload order |
| `t3real_prep`, `t6_refcache` | 3.98 + 2.86 GB | yes |

**The first two cannot be rebuilt on the reference box** because the upstream
Z-Image-Turbo transformer shards were deleted. They are the only copy. Nothing in
this repository is a substitute for archiving them.