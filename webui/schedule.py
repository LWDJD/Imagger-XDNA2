# SPDX-License-Identifier: MIT
"""The Euler flow-match schedule handed to the device loop.

steps == 8 uses the delivered t13b/t3real trajectory byte-for-byte.  Any other
step count is derived from the same FlowMatchEulerDiscreteScheduler and is
labelled out-of-distribution in the UI.
"""
from __future__ import annotations

import numpy as np

from . import config


def _load_ref():
    for name in ("t13b_traj.npz", "t3real_traj.npz"):
        p = config.IMAGE_DIR / name
        if p.is_file():
            z = np.load(p)
            return (np.asarray(z["timesteps"], np.float64),
                    np.asarray(z["sigmas"], np.float64), p)
    return None


def _compute(steps):
    from diffusers import FlowMatchEulerDiscreteScheduler
    s = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(config.MODELS), subfolder="scheduler")
    s.set_timesteps(int(steps))
    ts = np.asarray([float(t) for t in s.timesteps], np.float64)
    sg = np.asarray(s.sigmas.cpu().numpy(), np.float64)
    return ts, sg


def get_schedule(steps):
    steps = int(steps)
    if steps < 1 or steps > config.MAX_STEPS:
        raise ValueError(f"steps must be 1..{config.MAX_STEPS}")
    if steps == 8:
        ref = _load_ref()
        if ref:
            ts, sg, src = ref
            if len(ts) >= 8 and len(sg) >= 9:
                return {"steps": 8, "timesteps": ts[:8], "sigmas": sg[:9],
                        "source": f"delivered:{src.name}",
                        "out_of_distribution": False}
    cache = config.CACHE_DIR / "schedules" / f"steps_{steps}.npz"
    if cache.is_file():
        z = np.load(cache)
        ts, sg = z["timesteps"], z["sigmas"]
        if len(ts) == steps and len(sg) == steps + 1:
            return {"steps": steps, "timesteps": ts, "sigmas": sg,
                    "source": f"cached:{cache.name}",
                    "out_of_distribution": steps != 8}
    ts, sg = _compute(steps)
    if len(ts) != steps or len(sg) != steps + 1:
        raise ValueError(
            f"scheduler returned {len(ts)} timesteps / {len(sg)} sigmas for {steps} steps")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, timesteps=ts, sigmas=sg)
    return {"steps": steps, "timesteps": ts, "sigmas": sg,
            "source": f"derived:{cache.name}",
            "out_of_distribution": steps != 8}
