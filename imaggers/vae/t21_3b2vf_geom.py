# t21_3b2vf_geom.py -*- Python -*-
"""t21_3b2_vf: the pure-math core of the three design variants.

No aie, no torch, no device: this module is importable from the plan printer,
from vaeg_design.py's factories and from the runner, so the numbers the plan
promises are the numbers the factories and the gates use.

THE THREE VARIANTS (see T21-3B2VF-PLAN.md)

GAP2  M-block dispatch   -- split a conv row's M into <=GMAC_CAP blocks and
                            dispatch one block per process.  M is CompileTime,
                            so a block xclbin (M') is a NEW artifact; the
                            numerical path is untouched because C = A@B is
                            row-independent (a row of C sees only its own row
                            of A and the whole B, and the k-loop order, the
                            per-k-call bf16 re-rounding of C and Kc are all
                            M-independent).

GAP1  UP row strips      -- the upsample kernel is per-plane; a plane of S
                            elements needs 10*S bytes of L1 (2S in + 8S out)
                            and S=6144..16384 does not fit 64 KB.  A strip of
                            R rows is a contiguous run of BOTH the input
                            (R*W) and the output (4*R*W) -- nearest-2x maps
                            input rows [r0,r1) to output rows [2r0,2r1), which
                            are contiguous in the (2H,2W) plane -- so the
                            design just processes C*H/R strip objects in the
                            same order and the gate compares the full plane.

GAP1  GN sub-plane chunks -- a chunk is always the contiguous run
                            [c*Cs, (c+1)*Cs) of a group segment; what changes
                            is only whether that run is whole channel planes
                            (Cs % S == 0, gamma span Cs/S) or a sub-plane run
                            of one channel (Cs < S, Cs | S, gamma span 1,
                            spatial = Cs).  Both keep the 7-parameter kernel
                            ABI: the channel base is a memref OFFSET, not an
                            argument (group_norm_apply already spends all 7).
"""
from __future__ import annotations

import math

GROUPS = 32
L1_BYTES = 64 * 1024
WORKER_STACK = 0x1000          # 4 KB, every design here

# ---------------------------------------------------------------------------
# GAP2: M-block dispatch
# ---------------------------------------------------------------------------
# The block size is ONE number for every family: it has to be a multiple of
# the m tile (16), and the worst block must stay under the 3.0 GMAC budget
# (BATCH section 4.4: -> 0.6 GMAC/s conservative x 5 s).  1024 rows costs
# 1024*4640*512 = 2.43 GMAC in the worst family, which is 81% of the budget
# and half of the largest geometry that is known to complete a dispatch
# (4.86 GMAC, row 37).  Larger blocks buy fewer dispatches but eat that
# margin; the margin is what the 18 rows are missing in the first place.
M_BLOCK = 1024
GMAC_CAP = 3.0
M_TILE = 16

# Tap-split for the stall family: the IN-A fill walks the m-row tape
# row_repeats = (N/n) * (K/Kc) times.  Every dispatch that completed has
# row_repeats <= 1024, and the nine K=2336 N=256 rows (1168) are the only
# rows in the table above 1024 -- they are also exactly the nine rows that
# produce no output at all.  rep_cap splits that one fill into ceil(rep/cap)
# fills over the SAME rows in the SAME order, so the delivered stream (and
# therefore the arithmetic) is identical; it is a dispatch-side knob.
REP_CAP_DEFAULT = 0            # 0 = one fill, the 3b2 behaviour


def m_blocks(M: int, m_block: int = M_BLOCK) -> int:
    return -(-int(M) // int(m_block))


def m_plan(M: int, K: int, N: int, m_block: int = M_BLOCK) -> dict:
    """The block plan of one conv row (paper numbers, no device)."""
    if m_block % M_TILE:
        raise ValueError(f"block M={m_block} is not a multiple of {M_TILE}")
    blocks = m_blocks(M, m_block)
    per = m_block * K * N
    tail = M - (blocks - 1) * m_block
    return dict(m_block=int(m_block), blocks=blocks, block_mac=int(per),
                block_gmac=round(per / 1e9, 3),
                tail_rows=int(tail), tail_pad=int(m_block - tail),
                padded_mac=int(blocks * per),
                over_frac=round(blocks * per / (M * K * N) - 1.0, 4),
                under_cap=bool(per / 1e9 <= GMAC_CAP),
                worst_block_s=round(per / 1e9 / 0.6, 2))


def rep_cap_for(K: int, N: int, Kc: int, rep_cap: int = REP_CAP_DEFAULT) -> int:
    return int(rep_cap) if rep_cap else 0


def row_repeats(K: int, N: int, Kc: int, n: int = 16) -> int:
    """(N/n) * (K/Kc): the IN-A fill's repeat count (the stall suspect)."""
    return (N // n) * (K // Kc)


# ---------------------------------------------------------------------------
# GAP1: upsample row strips
# ---------------------------------------------------------------------------
# The strip must leave room for the Worker stack and the FIFO bookkeeping:
# 10*R*W + 8192 <= 64 KB (the same rule up_plane() uses for a whole plane).
UP_L1_BUDGET = L1_BYTES - 2 * WORKER_STACK - 4096      # 57344 B


def up_strip_rows(H: int, W: int, budget: int = UP_L1_BUDGET) -> int:
    """Largest R dividing H with 10*R*W <= budget (>= 1)."""
    best = 1
    for r in range(1, H + 1):
        if H % r:
            continue
        if 10 * r * W <= budget:
            best = r
    return best


def up_strip_plan(C: int, H: int, W: int, batch_tiles: int = 512) -> dict:
    """Strip geometry + the fill/drain batching of one upsample row."""
    R = up_strip_rows(H, W)
    n_in = R * W
    tiles = C * (H // R)
    return dict(rows=R, strips_per_channel=H // R, plane_in=n_in,
                plane_out=4 * n_in, tiles=int(tiles),
                batches=int(-(-tiles // batch_tiles)),
                batch_tiles=int(min(batch_tiles, tiles)),
                l1_bytes=10 * n_in, fits=bool(10 * n_in + 8192 <= L1_BYTES),
                out_bytes=4 * C * H * W * 2)


# ---------------------------------------------------------------------------
# GAP1: GroupNorm sub-plane chunks
# ---------------------------------------------------------------------------

def _divisors(n: int) -> list:
    out = []
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            out.append(d)
            out.append(n // d)
    return sorted(set(out))


def _gn_candidates(C: int, S: int, cs_max: int) -> list:
    """Legal (Cs, nch, spatial) for one group segment, largest Cs first.

    Whole-plane first: Cs = k*S (k >= 1) with L % Cs == 0, searched over the
    MULTIPLES of S -- not by subtracting S from cs_max, which misses valid
    sizes: (512,1152) has Cs = 2304 = 2*S with L%Cs == 0, yet pick_chunk steps
    4096, 2944, 1792 and gives up (that is WHY rows 60/66 are in the gap list
    with the self-contradictory note "S=1152 > cs_max=4096").  Then sub-plane:
    a divisor of S, one channel per chunk.
    """
    L = (C // GROUPS) * S
    out = []
    for k in range(cs_max // S, 0, -1):
        cs = S * k
        if L % cs == 0:
            out.append((int(cs), int(k), int(S)))
    if out:
        return out
    return [(int(d), 1, int(d)) for d in _divisors(S)
            if d <= cs_max and L % d == 0][::-1]


def gn_chunk_sub(C: int, S: int, cs_max: int = 4096, align: int = 64,
                 max_chunks: int = 64, l1_budget: int = 57344) -> tuple:
    """(Cs, nch, spatial) for one group segment of cg*S elements.

    Two constraints beyond L1 (4*Cs + GB + stack + slack <= 64 KB):

      * cs_max: the 3b2 chunk cap (4096), kept so that geometries which 3b2
        already gated chunk IDENTICALLY here;
      * max_chunks: the WORKER's chunk loop is unrolled (constant call sites),
        so n_chunks is code size.  Measured: (256,65536) at Cs=4096 -> 128
        chunks overflowed .text by 4640 B, while every row at <= 64 chunks
        compiles.  When the default pick exceeds it, take the LARGEST legal Cs
        instead (8 KB per chunk at (256,65536) -> 64 chunks).

    A sub-plane chunk is one channel run (nch=1, spatial=Cs); a whole-plane
    chunk is k channels (nch=k, spatial=S).
    """
    L = (C // GROUPS) * S
    cands = _gn_candidates(C, S, cs_max)
    if not cands:
        raise ValueError(f"no chunk for C={C} S={S} cs_max={cs_max}")
    cs, nch, sp = cands[0]
    if L // cs <= max_chunks:
        return cs, nch, sp
    for cs, nch, sp in _gn_candidates(C, S, min(l1_budget // 4, S)):
        if 4 * cs <= l1_budget and L // cs <= max_chunks:
            return int(cs), int(nch), int(sp)
    raise ValueError(f"cannot get C={C} S={S} under {max_chunks} chunks")


def gn_sub_plan(C: int, S: int, cs_max: int = 4096) -> dict:
    cg = C // GROUPS
    L = cg * S
    cs, nch, spatial = gn_chunk_sub(C, S, cs_max)
    n_chunks = L // cs
    return dict(cs=cs, chpc=nch, spatial=spatial, group_len=L, n_chunks=int(n_chunks),
                whole_plane=bool(cs % S == 0),
                l1_bytes=4 * cs, gb_bytes=((4 * C + 63) // 64) * 64 + 16,
                fills_per_group=2 * n_chunks,
                total_fills=2 * GROUPS * n_chunks,
                drain_tiles=GROUPS * n_chunks,
                x_bytes=L * 2 * GROUPS, y_bytes=L * 2 * GROUPS,
                note="" if cs % S == 0 else "sub-plane (one channel per chunk)")


def gn_chan_offset(C: int, S: int, cs: int, c: int) -> int:
    """First channel of chunk c inside its group (gamma/beta memref offset)."""
    return (c * cs) // S if cs < S else c * (cs // S)
