# SPDX-License-Identifier: MIT
"""devkill_prewarm.py -- pack the 120 G=15 projection W images BEFORE the loop.

WHY.  prod_devkill._w() converts work/qkv/L<layer>_to_<name>.npy into the G=15
image the first time that layer runs.  Measured on job 20261031-174330-75fae8
(512_devkill_8a): the first proj_qkv of a layer costs 0.27-0.35 s against
0.103 s steady (+0.17 s x 30 layers ~ 5.1 s), and every one of those first uses
lands inside step 0.  The delivered path already ships this prewarm
("[t3d] warm: 30 layers' projection images packed in 1.4 s (startup, outside
the loop's wall)"); this is the same thing for the devkill image.

WHAT IT IS NOT.  No device, no xclbin, no arithmetic: it only fills the
in-process _PROJ_W byte cache that _w() would otherwise fill lazily.  The bytes
handed to the device are identical -- this is a scheduling change, not a
numeric one, and the latents must be byte-equal to the lazy arm.

FLAG.  T22_DEVKILL_PREWARM=0 restores lazy packing (the A/B control arm).
       T22_DEVKILL_PREWARM_LAYERS=N limits the sweep (default 30).
"""
from __future__ import annotations
import os
import time

LAYERS = int(os.environ.get("T22_DEVKILL_PREWARM_LAYERS", "30"))
NAMES = ("q", "k", "v", "o")


def prewarm_all(layers=None, names=NAMES, verbose=True):
    """Fill prod_devkill._PROJ_W for every (layer, projection).  Returns (n, s)."""
    import prod_devkill as PD          # same module object the loop will use
    n_layers = LAYERS if layers is None else int(layers)
    t0 = time.perf_counter()
    n = 0
    for li in range(n_layers):
        key = "layers.%d" % li
        for nm in names:
            PD._w(key, nm)
            n += 1
    dt = time.perf_counter() - t0
    if verbose:
        print("[devkill] prewarm: %d G=15 projection images packed in %.2f s "
              "(startup, outside the loop's wall)" % (n, dt), flush=True)
    return n, dt


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    n, dt = prewarm_all()
    print("[devkill] selftest: %d images, %.2f s, %.1f ms/image"
          % (n, dt, 1e3 * dt / max(n, 1)))
