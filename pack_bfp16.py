# SPDX-License-Identifier: MIT
"""pack_bfp16 -- the AMD XDNA2 / VitisAI `bfp16ebs8` weight packer.

This is the format reverse-engineered from a vendor-produced `.rai` and verified
byte for byte over 4,915,200 / 4,915,200 groups (`probe_bfp16_rulecheck.py`:
"whole region identical: True").  It replaces `wgt_formatting`, which AMD ships
but which refuses BFP16:

    Unsupported OP type for weight formatting: MatMul_bfp16xbfp16xbfp16

THE FORMAT
----------
A weight matrix W of shape (N, K) is packed as follows.

1. Stream order.  W is walked row-major with K contiguous, and every GROUP = 8
   consecutive elements of that stream forms one quantisation group.

2. Per-group encoding, 9 bytes = [exponent_byte][8 x int8 mantissa]:

       amax   = max |v| over the 8 values
       E'     = unbiased IEEE exponent of amax, i.e. amax = 2**E' * (1+f)
       e      = E' - 6   UNCONDITIONALLY.

       (An earlier version used E' - 7 when f == 0 -- amax an exact power of
       two -- reasoning that it keeps more headroom.  That is wrong.  On a real
       16-tensor shard ~0.4% of weight groups have an exact power-of-two amax,
       and on every one of the 7,198 mismatching groups of layers.10.to_q the
       vendor's exponent byte is exactly 1 higher with the mantissas exactly
       halved.  Synthetic fixtures never exposed it because none of them
       happened to have a power-of-two amax.)
       q      = v / 2**e
       m      = sign(q) * floor(|q| + 0.5)     # ROUND HALF AWAY FROM ZERO
       if m.max() > 127:                        # +128 overflows int8
           e += 1                               # -128 is legal, +128 is not
           recompute m
       m      = clip(m, -128, 127).astype(int8)
       exponent_byte = (e + 133) & 0xFF         # == IEEE bias 127

   Every one of these details was pinned by measurement, not guessed:
     * the exponent byte comes BEFORE the mantissas (phase scan: one byte in 9
       has entropy 1.16 while the other eight have 7.74)
     * rounding is half-away-from-zero (an exact -50.5 tie stores -51, not -50)
     * the overflow rule is asymmetric (a group peaking at -128 was NOT bumped)
     * `ceil(log2(amax))` is numerically unsafe near powers of two; use the bit
       pattern instead

3. Permutation.  Packed groups are laid out in 32-row x 32-K-element tiles:

       tile_index    = (row // 32) * (K // 32) + (kelem // 32)
       slot_in_tile  = (r // 8) * 32 + (c // 8) * 8 + (r % 8)
       slot          = tile_index * 128 + slot_in_tile

   with r = row % 32, c = kelem % 32.  K must be a multiple of 32 and N a
   multiple of 32 for the rule to apply unchanged.

USAGE
-----
    from pack_bfp16 import pack, unpack, selftest

    blob = pack(W)                 # W float32 (N, K) -> bytes, ready for the
                                   # kernel's `coalesed_weights` buffer
    W2   = unpack(blob, N, K)      # approximate round trip (BFP16 precision)

    selftest()                     # re-verify the quantiser on random data

Sizes: len(pack(W)) == N * K * 9 // 8 exactly, i.e. 1.125 bytes per parameter.
"""

from __future__ import annotations

import numpy as np

GROUP = 8
TILE_ROWS = 32
TILE_KELEMS = 32
GROUPS_PER_TILE = (TILE_ROWS // 8) * (TILE_KELEMS // GROUP) * 8   # = 128


def _quantise(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One flat stream of groups -> (mantissa int8 [ng,8], exponent byte [ng]).

    `v` is (ng, 8) float64.  Implements the measured rule exactly.
    """
    amax = np.abs(v).max(axis=1).astype(np.float32)

    # e from the IEEE bit pattern, not np.log2: for amax just below a power of
    # two np.log2 rounds up and `ceil` then lands on the wrong side.
    bits = np.ascontiguousarray(amax).view(np.uint32)
    efield = ((bits >> 23) & 0xFF).astype(np.int64)
    frac = (bits & 0x7FFFFF).astype(np.int64)
    e_unb = efield - 127
    e = e_unb - 6
    # ^^^ UNCONDITIONAL.  Was np.where(frac == 0, e_unb - 7, e_unb - 6) -- the
    #     power-of-two branch is wrong and was the sole reason a real 16-tensor
    #     shard failed to match: ~0.4% of real-weight groups have an exact
    #     power-of-two amax, and on all 7,198 mismatching groups of
    #     layers.10.to_q the vendor exponent byte is exactly 1 higher with the
    #     mantissas exactly halved.  See route3_coalesce.py / COALESCED-LAYOUT.md.

    def mant(scale_e: np.ndarray) -> np.ndarray:
        q = v / np.exp2(scale_e)[:, None]
        return np.sign(q) * np.floor(np.abs(q) + 0.5)

    m = mant(e)
    over = m.max(axis=1) > 127
    if over.any():
        e = np.where(over, e + 1, e)
        m = mant(e)
    return np.clip(m, -128, 127).astype(np.int8), \
        ((e + 133) & 0xFF).astype(np.uint8)


def _slots(N: int, K: int, ng: int, tile_rows: int = 32) -> np.ndarray:
    """group index -> byte slot inside the packed region.

    `tile_rows` is the height, in N-rows, of one packing tile.  It is the ONE
    quantity that varies between shapes; everything else is fixed.  Measured
    against vendor .rai files:

        (N=3840,  K=15360) -> 32      (N=15360, K=3840) -> 48
        (N=10240, K=3840)  -> 32

    The value comes from the kernel's subvolume tiling (OGOAT
    `MatMul_kernel_metadata.yaml`, `step_Ky: 9 * M_SUBV * K_SUBV // 8`), so it
    must be established per shape -- see `detect_tile_rows`.
    """
    if N % tile_rows or K % TILE_KELEMS:
        raise ValueError(
            f"pack_bfp16's tiling rule needs N%{tile_rows}==0 and "
            f"K%{TILE_KELEMS}==0, got N={N} K={K} tile_rows={tile_rows}")
    if tile_rows % 8:
        raise ValueError(f"tile_rows must be a multiple of 8, got {tile_rows}")
    idx = np.arange(ng, dtype=np.int64)
    kg = idx % (K // GROUP)          # group index along K
    row = idx // (K // GROUP)
    ke = kg * GROUP                  # element index along K
    r, c = row % tile_rows, ke % TILE_KELEMS
    tile = ((row // tile_rows) * (K // TILE_KELEMS) + (ke // TILE_KELEMS))
    inside = ((r // 8) * (TILE_KELEMS // GROUP) * 8
              + (c // 8) * 8 + (r % 8))
    per_tile = (tile_rows // 8) * (TILE_KELEMS // GROUP) * 8
    return tile * per_tile + inside


# tile_rows known to reproduce a vendor .rai byte for byte, keyed by (N, K).
KNOWN_TILE_ROWS = {
    (3840, 15360): 32,
    (10240, 3840): 32,
    (15360, 3840): 48,
    (15360, 256): 64,
    (3840, 256): 48,
    (64, 3840): 16,
    (3840, 64): 80,
    (3840, 2560): 32,
    (1024, 256): 64,
    (256, 1024): 64,
    (3840, 3840): 48,
    (3840, 10240): 32,
}


def detect_tile_rows(W: np.ndarray, blob: bytes) -> int:
    """Find the `tile_rows` that makes `pack(W)` match a vendor `.rai` region.

    `blob` is the content of the `.rai` compiled from a graph carrying exactly
    these weights.

    The match is decided by comparing the FULL packed region, never a short
    probe: a 96-byte probe produced a false positive on (15360, 256), where the
    middle of the stream happened to agree while the rest did not.  Short
    96-byte probes only CANDIDATE a region base (several offsets, every
    occurrence of each probe); acceptance requires all `len(mine)` bytes to
    agree.  Candidate `tile_rows` values are tried measured-value-first, then
    exhaustively over every multiple of 8 that divides N, so a shape outside the
    measured set is still detected rather than guessed.
    """
    W = np.ascontiguousarray(W, dtype=np.float32)
    N, K = W.shape
    ng = N * K // GROUP
    if N * K % GROUP:
        raise ValueError(f"N*K must be a multiple of {GROUP}, got {N}*{K}")
    # quantise once; only the slot permutation differs between candidates
    v = W.reshape(-1).reshape(ng, GROUP).astype(np.float64)
    m, expb = _quantise(v)
    units = np.empty((ng, 9), dtype=np.uint8)
    units[:, 0] = expb
    units[:, 1:] = np.ascontiguousarray(m).view(np.uint8).reshape(ng, 8)

    seen: set[int] = set()
    cands: list[int] = []
    for tr in (16, 32, 48, 64, 96, 128, 24, 40, 80, 8, 256):
        if tr % 8 == 0 and N % tr == 0 and tr not in seen:
            seen.add(tr)
            cands.append(tr)
    for tr in range(8, N + 1, 8):
        if N % tr == 0 and tr not in seen:
            seen.add(tr)
            cands.append(tr)

    want = ng * 9
    for tr in cands:
        out = np.zeros((ng, 9), dtype=np.uint8)
        out[_slots(N, K, ng, tr)] = units
        mine = out.tobytes()
        for frac in (0.5, 0.125, 0.875, 0.25, 0.75):
            off = int(want * frac)
            off -= off % 9
            chunk = mine[off:off + 96]
            if len(chunk) < 96:
                continue
            start = 0
            while True:
                i = blob.find(chunk, start)
                if i < 0:
                    break
                base = i - off
                if base >= 0 and base + want <= len(blob) \
                        and bytes(blob[base:base + want]) == mine:
                    return tr
                start = i + 1
    raise ValueError(f"no tile_rows reproduced the .rai for shape {W.shape}")


def pack(W: np.ndarray, tile_rows: int | None = None) -> bytes:
    """float32 (N, K) -> the kernel-ready `coalesed_weights` byte stream."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"expected a 2-D (N,K) matrix, got shape {W.shape}")
    N, K = W.shape
    if tile_rows is None:
        tile_rows = KNOWN_TILE_ROWS.get((N, K), 32)
    ng = N * K // GROUP
    if N * K % GROUP:
        raise ValueError(f"N*K must be a multiple of {GROUP}, got {N}*{K}")

    v = W.reshape(-1).reshape(ng, GROUP).astype(np.float64)
    m, expb = _quantise(v)

    units = np.empty((ng, 9), dtype=np.uint8)
    units[:, 0] = expb
    units[:, 1:] = np.ascontiguousarray(m).view(np.uint8).reshape(ng, 8)

    out = np.zeros((ng, 9), dtype=np.uint8)
    out[_slots(N, K, ng, tile_rows)] = units
    return out.tobytes()


def unpack(blob: bytes, N: int, K: int,
           tile_rows: int | None = None) -> np.ndarray:
    """Inverse of `pack`, up to BFP16 precision.  Returns float32 (N, K)."""
    if tile_rows is None:
        tile_rows = KNOWN_TILE_ROWS.get((N, K), 32)
    ng = N * K // GROUP
    if len(blob) != ng * 9:
        raise ValueError(f"expected {ng * 9} bytes for ({N},{K}), "
                         f"got {len(blob)}")
    reg = np.frombuffer(blob, dtype=np.uint8).reshape(ng, 9)
    # `pack` scatters: reg[slot[i]] = units[i].  The inverse is a GATHER,
    # units[i] = reg[slot[i]] -- NOT another scatter, which is what an earlier
    # draft did and which silently permuted the wrong way.
    units = reg[_slots(N, K, ng, tile_rows)]

    m = np.ascontiguousarray(units[:, 1:]).view(np.int8).astype(np.float32)
    e = (units[:, 0].astype(np.int32) - 133)
    v = m * np.exp2(e.astype(np.float32))[:, None]
    return v.reshape(N, K)


def selftest(seed: int = 0, N: int = 64, K: int = 64) -> int:
    """Round-trip + quantiser invariants on random data.  Returns 0 on pass."""
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    blob = pack(W)
    assert len(blob) == N * K * 9 // GROUP, (len(blob), N * K * 9 // GROUP)

    W2 = unpack(blob, N, K)
    scale = float(np.abs(W).max())
    err = float(np.abs(W2 - W).max()) / (scale + 1e-30)

    # every mantissa must be inside int8 and the exponent byte is a pure
    # function of the group's amax
    ng = N * K // GROUP
    v = W.reshape(-1).reshape(ng, GROUP).astype(np.float64)
    m, expb = _quantise(v)
    assert np.abs(m).max() <= 127
    # no group may leave a representable value on the table: after the (possibly
    # bumped) exponent, the peak mantissa must be <= 127
    peak = np.abs(m).max(axis=1)
    assert peak.max() <= 127

    print(f"pack_bfp16 selftest: {N}x{K}, {ng:,} groups, "
          f"{len(blob):,} B = {len(blob) / (N * K):.4f} B/param")
    print(f"  round-trip max rel err {err:.4e}  "
          f"(BFP16 floor is ~{2 ** -7:.4f})")
    print(f"  mantissa peak range {peak.min()}..{peak.max()}  "
          f"exponent bytes {int(expb.min())}..{int(expb.max())}")
    ok = err <= 2 ** -6
    print(f"  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(selftest())
