# SPDX-License-Identifier: MIT
"""brepack_cache.py -- the big-M B repack as a content-addressed, mmapped file.

WHAT THIS REPLACES
------------------
prod_loop_bigm.repack_cache_inplace_bigm() turns the product's resident
BG/B3 payload rows into the big-M image order.  It is a PURE PERMUTATION of
bytes that already live on disk (t3real_prep/layers.N_{bg,b3}.npy), yet it is
re-derived from scratch on EVERY run:

    per layer   _bg_unpermute(BG, nh=5)   88,473,600 B -> 88,473,600 B
                bg_to_image(BG)           88,473,600 B -> 88,473,600 B
                b3_to_image(B3)           44,236,800 B -> 44,236,800 B
                b3_to_image_k32(B3)       44,236,800 B -> 44,236,800 B  (python loop!)
    x 30 layers = 7.98 GB of gathered bytes, 5.31 GB of them NEW

Measured on this host: 5.35 s of pure CPU (bench_brepack2.py, 0.18 s/layer),
and 19.91 s of WALL inside the product (HOSTPROF-REPORT.md section 8: the first
glue.cached_payloads call), the difference being the allocation and the
first-touch page faults of 5.3 GB while the device, the 8 GB of weights and
the refiner cache are all resident.

WHAT THIS DOES
--------------
Exactly what wpack does for the projection images (webui/bridge.py:591
_load_wpack_file): write the derived bytes ONCE to a file and mmap it
afterwards.  The file is

    +0            64 KiB   header: UTF-8 JSON -- sha256 of the content, plus
                           the identity of the sources it was derived from
    +64 KiB       BG   : layers x 88,473,600   image order (gu / swiglu1)
                  B3   : layers x 44,236,800   image order (ffn_dn_rt1)
                  ALT  : layers x 44,236,800   the second down generation
                                              (k32 when T22_BIGM_DEVHANDOFF=1,
                                               n64 otherwise; absent when
                                               T22_BIGM_B3_DUAL=0)

Every section offset is 64 KiB aligned (88,473,600 = 1350 x 65536 and
44,236,800 = 675 x 65536), so each one is a legal np.memmap offset on Windows.

A hit costs: 60 stat() calls + one JSON parse + one threaded read of the file
(0.65 s for 5.31 GB on 16 threads) and nothing else.  A miss costs exactly what
the old path cost, plus an asynchronous 5.3 GB write.

INTEGRITY / WHY THIS CANNOT SILENTLY RE-ENCODE
---------------------------------------------
The bytes are BFP16 and the stored blocks are NOT canonical (0.297 % of bytes
differ from any re-encode), so the cache must be a byte-for-byte PERMUTATION.
Three independent guards, in increasing strength:

 1. IDENTITY.  The file is named after a sha256 over (a) the sha256 of all 60
    source .npy files, (b) the source digests of the four permutation functions
    of prod_bigborder plus _bg_unpermute, (c) the geometry actually in force
    (layers, sizes, nh_bg, dual, alt kind).  Change any of them and the key
    changes, so a stale file is never even opened.  The same key is repeated in
    the header and re-derived on every load.
 2. MULTISET.  At build the repacked bytes are asserted, per layer, to have the
    SAME 64-bit word multiset as the source (count, sum, and sum of w*K mod
    2^64 -- order-invariant statistics over 8-byte words, which a permutation
    of 72-byte units preserves exactly), and for layer 0 additionally the
    EXACT 256-bin byte histogram.  On load the same statistics are recomputed
    from the mapped bytes and compared with the values recorded for the SOURCE.
    A re-encode fails this; a permutation cannot.
 3. CONTENT.  The header carries a Merkle-1 sha256 over 16 MiB pieces of the
    content region and the load recomputes it (0.38 s on 16 threads; sha256
    runs at 14 GB/s here because of SHA-NI).  Any torn or edited byte is caught.

The build also asserts that the order is a BIJECTION: an arange probe is pushed
through the same index expression and must come back as a permutation of
0..n-1 (selftest), which is the structural half of guard 2.

ENV (everything defaults to OFF / harmless)
-------------------------------------------
  T22_BREPACK_CACHE=1       enable the cache          (default 0)
  T22_BREPACK_DIR=<path>    where the file lives     (default imaggers/brepack_cache)
  T22_BREPACK_VERIFY=0      skip guards 2+3 on load  (default 1: verify)
  T22_BREPACK_THREADS=N     verification threads     (default min(16, cpu))
  T22_BREPACK_BUILD=0       never build on a miss    (default 1)
  T22_BREPACK_FASTK32=0     use the vectorised b3_to_image_k32 at build
                                                     (default 1, self-checked)
  T22_BREPACK_PIECE=<bytes> verification piece size  (default 16 MiB)

CLI (CPU only -- no device, no lock)
------------------------------------
  python brepack_cache.py --build     build the file from t3real_prep
  python brepack_cache.py --stat      print the header of the matching file
  python brepack_cache.py --verify    re-verify it
  python brepack_cache.py --selftest  geometry + fast-k32 equality gates
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import pathlib
import sys
import threading
import time

import numpy as np

MAGIC = "T22-BREPACK"
VERSION = 1
HEADER_BYTES = 64 * 1024          # 64 KiB, keeps every section 64 KiB aligned
PIECE_DEFAULT = 16 * 1024 * 1024
K_MOMENT = np.uint64(0x9E3779B97F4A7C15)

HERE = pathlib.Path(__file__).resolve().parent
NPU = HERE.parents[2] if len(HERE.parents) > 2 else HERE.parent


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------
def _env_int(name, default):
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def enabled() -> bool:
    return os.environ.get("T22_BREPACK_CACHE", "0") == "1"


def cache_dir() -> pathlib.Path:
    d = os.environ.get("T22_BREPACK_DIR", "").strip()
    return pathlib.Path(d) if d else (NPU / "imaggers" / "brepack_cache")


def _verify_on() -> bool:
    return os.environ.get("T22_BREPACK_VERIFY", "1") != "0"


def _threads() -> int:
    n = _env_int("T22_BREPACK_THREADS", 0)
    if n > 0:
        return n
    return max(1, min(16, (os.cpu_count() or 4)))


def _piece() -> int:
    n = _env_int("T22_BREPACK_PIECE", PIECE_DEFAULT)
    return max(1 << 20, (n // 8) * 8)


# ---------------------------------------------------------------------------
# the product modules
# ---------------------------------------------------------------------------
def _bigm_dir() -> pathlib.Path:
    return NPU / "imaggers" / "block" / "bigm"


def _ensure_path():
    for p in (_bigm_dir(), NPU / "imaggers" / "image"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _perm_modules():
    _ensure_path()
    import prod_bigborder as BB
    import prod_loop_bigm as PB
    return BB, PB


# ---------------------------------------------------------------------------
# source identity: sha256 of the 60 .npy payload sources, memoised by
# (name, size, mtime_ns) so a warm load pays 60 stat() calls and no hashing.
# ---------------------------------------------------------------------------
def _idmap_path() -> pathlib.Path:
    return cache_dir() / "src_idmap.json"


def _sha256_file(p, threads=1):
    """sha256 of a file, optionally over N threads (Merkle-1 by piece)."""
    size = os.path.getsize(p)
    piece = _piece()
    if threads <= 1 or size <= piece:
        h = hashlib.sha256()
        with open(p, "rb", buffering=0) as f:
            while True:
                b = f.read(piece)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    offs = list(range(0, size, piece))

    def one(o):
        h = hashlib.sha256()
        with open(p, "rb", buffering=0) as f:
            f.seek(o)
            left = min(piece, size - o)
            while left > 0:
                b = f.read(min(1 << 20, left))
                if not b:
                    break
                h.update(b)
                left -= len(b)
        return h.digest()

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(threads) as ex:
        parts = list(ex.map(one, offs))
    return hashlib.sha256(b"".join(parts)).hexdigest()


def _load_idmap():
    try:
        with open(_idmap_path(), "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_idmap(d):
    try:
        cache_dir().mkdir(parents=True, exist_ok=True)
        tmp = _idmap_path().with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=0, sort_keys=True)
        os.replace(tmp, _idmap_path())
    except OSError:
        pass


def source_ids(prep, layers) -> dict:
    """sha256 of every layers.N_bg.npy / layers.N_b3.npy payload source."""
    prep = pathlib.Path(prep)
    memo = _load_idmap()
    out, fresh = {}, {}
    for li in range(int(layers)):
        for kind in ("bg", "b3"):
            name = "layers.%d_%s.npy" % (li, kind)
            p = prep / name
            st = p.stat()
            tok = "%s|%d|%d" % (name, st.st_size, st.st_mtime_ns)
            got = memo.get(tok)
            if got is None:
                got = _sha256_file(p, _threads())
                fresh[tok] = got
            out[name] = got
    if fresh:
        _save_idmap(fresh)
    return out


def _norm_src(fn) -> str:
    """inspect.getsource with the def line normalised.

    The digest must cover the TRANSFORM, not its name: prod_bigborder.
    b3_to_image_k32 is now a dispatcher (T22_BB_FASTK32) over the reference
    _b3_to_image_k32_ref, and renaming a function must not invalidate a
    4.94 GiB artifact.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return repr(fn)
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith("def ") and "(" in s:
            lines[i] = ln[:len(ln) - len(s)] + "def perm(" + s.split("(", 1)[1]
            break
    return "\n".join(lines)


def _k32_ref(BB):
    """The reference b3_to_image_k32 of this tree, under whatever name."""
    return getattr(BB, "_b3_to_image_k32_ref", None) or BB.b3_to_image_k32


def _k32_fast(BB):
    """The vectorised b3_to_image_k32, or None on a tree that has none."""
    return getattr(BB, "b3_to_image_k32_fast", None)


def geometry() -> dict:
    """The permutation code actually in force, as source digests."""
    BB, PB = _perm_modules()
    g = {}
    for nm, fn in (("bg_to_image", BB.bg_to_image),
                   ("b3_to_image", BB.b3_to_image),
                   ("b3_to_image_n64", BB.b3_to_image_n64),
                   ("b3_to_image_k32", _k32_ref(BB)),
                   ("_bg_unpermute", PB._bg_unpermute)):
        g[nm] = hashlib.sha256(
            _norm_src(fn).encode("utf-8")).hexdigest()[:16]
    g["GU"] = "%d/%d/%d/%d" % (BB.GU_G, BB.GU_HALF, BB.GU_NG, BB.GU_CH)
    g["DN"] = "%d/%d/%d/%d" % (BB.DN_G, BB.DN_HALF, BB.DN_NG, BB.DN_CH)
    g["DN64"] = "%d/%d/%d" % (BB.DN64_HALF, BB.DN64_OBJ, BB.DN64_CH)
    return g


def identity(prep, layers, bg_bytes, b3_bytes, nh_bg, dual, alt_kind) -> dict:
    return {"v": VERSION, "layers": int(layers),
            "bg_bytes": int(bg_bytes), "b3_bytes": int(b3_bytes),
            "nh_bg": int(nh_bg), "dual": bool(dual), "alt": str(alt_kind),
            "geom": geometry(), "src": source_ids(prep, layers)}


def identity_key(ident) -> tuple:
    blob = json.dumps(ident, sort_keys=True, separators=(",", ":")).encode()
    full = hashlib.sha256(blob).hexdigest()
    return full, full[:16]


def prep_default() -> pathlib.Path:
    """The product's payload sources (t18_run.PREP)."""
    p = os.environ.get("T22_BREPACK_PREP", "").strip()
    return pathlib.Path(p) if p else (NPU / "imaggers" / "image" / "t3real_prep")


def file_for(key16, layers, alt_kind) -> pathlib.Path:
    return cache_dir() / ("brepack_%d_%s_%s.bin"
                          % (int(layers), alt_kind or "none", key16))


# ---------------------------------------------------------------------------
# the order-invariant statistics (guard 2) and the content hash (guard 3)
# ---------------------------------------------------------------------------
def _stats(buf):
    """(n_words, sum(w), sum(w*K)) over the 8-byte words of buf.

    Order-invariant, and every permutation this cache stores is a permutation of
    72-byte units (72 == 9 words), so the WORD multiset is preserved exactly by
    a correct repack and destroyed by a re-encode.
    """
    a = buf if isinstance(buf, np.ndarray) else np.frombuffer(buf, np.uint8)
    a = np.ascontiguousarray(a, dtype=np.uint8).reshape(-1)
    n = (a.size // 8) * 8
    w = a[:n].view("<u8")
    s1 = int(w.sum(dtype=np.uint64))
    s2 = int(np.multiply(w, K_MOMENT).sum(dtype=np.uint64))
    tail = int(a[n:].sum(dtype=np.uint64))
    return (int(n // 8), (s1 + tail) & 0xFFFFFFFFFFFFFFFF, s2)


def _hist256(a):
    """Exact 256-bin byte histogram -- the rigorous build-time anchor."""
    return np.bincount(np.asarray(a, np.uint8).reshape(-1), minlength=256)


def content_digest(view, piece=None, threads=None, do_hash=True) -> dict:
    """sha256 (Merkle-1 by piece) + word statistics of a flat uint8 view."""
    piece = piece or _piece()
    threads = threads or _threads()
    n = int(view.size)
    offs = list(range(0, n, piece))

    def one(o):
        seg = view[o:o + piece]
        h = hashlib.sha256(seg).digest() if do_hash else b""
        return h, _stats(seg)

    if threads <= 1:
        parts = [one(o) for o in offs]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(threads) as ex:
            parts = list(ex.map(one, offs))
    s = [0, 0, 0]
    for _, st in parts:
        s[0] += st[0]
        s[1] = (s[1] + st[1]) & 0xFFFFFFFFFFFFFFFF
        s[2] = (s[2] + st[2]) & 0xFFFFFFFFFFFFFFFF
    out = {"n_words": s[0], "s1": s[1], "s2": s[2], "piece": piece,
           "pieces": len(parts)}
    if do_hash:
        out["sha256"] = hashlib.sha256(b"".join(h for h, _ in parts)).hexdigest()
    return out


def _same_stats(a, b) -> bool:
    return (int(a["n_words"]) == int(b["n_words"])
            and int(a["s1"]) == int(b["s1"])
            and int(a["s2"]) == int(b["s2"]))


_M64 = 0xFFFFFFFFFFFFFFFF


def _acc(acc, arr):
    """acc += the word multiset of one array (in place)."""
    st = _stats(arr)
    acc[0] += st[0]
    acc[1] = (acc[1] + st[1]) & _M64
    acc[2] = (acc[2] + st[2]) & _M64
    return acc


def _add(a, b):
    return [a[0] + b[0], (a[1] + b[1]) & _M64, (a[2] + b[2]) & _M64]


def _mul(a, k):
    return [a[0] * k, (a[1] * k) & _M64, (a[2] * k) & _M64]


# ---------------------------------------------------------------------------
# the repack itself -- a byte-exact mirror of repack_cache_inplace_bigm
# ---------------------------------------------------------------------------
_FASTK = {"ok": None}


def _probe_b3():
    """A 44,236,800 B payload whose 614,400 72-byte blocks are all DISTINCT.

    The block index is written into each block's first 8 bytes, so a wrong
    permutation cannot hide behind repeated bytes the way a periodic pattern
    would let it.
    """
    n = 44_236_800 // 72
    ids = np.arange(n, dtype=np.int64)
    p = np.zeros((n, 72), np.uint8)
    p[:, :8] = ids.view(np.uint8).reshape(n, 8)
    return p.reshape(-1)


def _k32(b3_row):
    """The fastest correct b3_to_image_k32 for this process.

    prod_bigborder owns the vectorised transpose now (b3_to_image_k32_fast,
    env-gated by T22_BB_FASTK32 for the PRODUCT path).  The cache always wants
    it, whatever that flag says, so it is selected here directly -- but only
    after a probe on 614,400 distinct 72-byte blocks proves it byte-identical
    to the reference.  _k32 must never consult the dispatcher, or the probe
    would be comparing the fast path with itself.
    """
    BB, _ = _perm_modules()
    if _FASTK["ok"] is None:
        fast = _k32_fast(BB)
        ok = False
        if os.environ.get("T22_BREPACK_FASTK32", "1") != "0" and fast is not None:
            try:
                probe = _probe_b3()
                ok = bool(np.array_equal(fast(probe), _k32_ref(BB)(probe)))
            except Exception:                                  # noqa: BLE001
                ok = False
        _FASTK["ok"] = ok
        _FASTK["fn"] = fast if ok else None
        print("[brepack] b3_to_image_k32: %s"
              % ("prod_bigborder.b3_to_image_k32_fast "
                 "(probe-identical on 614,400 distinct blocks)" if ok
                 else "the reference python loop"), flush=True)
    fn = _FASTK.get("fn")
    return fn(b3_row) if fn is not None else _k32_ref(BB)(b3_row)


def repack_rows(bg_row, b3_row, nh_bg, dual, devhand) -> tuple:
    """(bg_image, b3_image, alt_image) for one layer -- the same bytes
    repack_cache_inplace_bigm writes into the resident cache."""
    BB, PB = _perm_modules()
    src = bg_row if (not nh_bg or nh_bg <= 1) else PB._bg_unpermute(bg_row, nh_bg)
    bg = BB.bg_to_image(src)
    if b3_row is None or not dual:
        return bg, None, None
    raw = np.array(b3_row, np.uint8, copy=True)
    b3 = BB.b3_to_image(raw)
    alt = _k32(raw) if devhand else BB.b3_to_image_n64(raw)
    return bg, np.asarray(b3, np.uint8), np.asarray(alt, np.uint8)


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------
def _sections(layers, bg_bytes, b3_bytes, alt_kind):
    out, off = {}, HEADER_BYTES
    out["bg"] = [off, layers * bg_bytes]
    off += layers * bg_bytes
    out["b3"] = [off, layers * b3_bytes]
    off += layers * b3_bytes
    alt_bytes = 0 if alt_kind == "none" else b3_bytes
    out["alt"] = [off, layers * alt_bytes]
    off += layers * alt_bytes
    return out, off


def write_cache(path, bg, b3, alt, ident, key, expect_fp, repack_s=None,
                chunk=32 << 20):
    """Serialise (bg, b3, alt) -- each (layers, n) uint8 -- to path atomically."""
    layers = int(bg.shape[0])
    bg_bytes = int(bg.shape[1])
    b3_bytes = int(b3.shape[1]) if b3 is not None else int(alt.shape[1]) if alt is not None else 0
    alt_kind = ident["alt"]
    sec, total = _sections(layers, bg_bytes, b3_bytes, alt_kind)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    t0 = time.perf_counter()
    with open(tmp, "wb", buffering=0) as fh:
        fh.write(b"\x00" * HEADER_BYTES)
        for arr in (bg, b3, alt):
            if arr is None:
                continue
            for li in range(layers):
                row = np.ascontiguousarray(arr[li]).reshape(-1)
                for o in range(0, row.size, chunk):
                    fh.write(memoryview(row[o:o + chunk]))
        fh.flush()
        os.fsync(fh.fileno())
    write_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    mm = np.memmap(str(tmp), dtype=np.uint8, mode="r+", shape=(total,))
    try:
        dig = content_digest(mm[HEADER_BYTES:])
    finally:
        del mm
    t_ver = time.perf_counter() - t1
    hdr = {"magic": MAGIC, "version": VERSION,
           "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "built_ts": time.time(),
           "layers": layers, "bg_bytes": bg_bytes, "b3_bytes": b3_bytes,
           "alt_bytes": 0 if alt is None else b3_bytes, "alt_kind": alt_kind,
           "nh_bg": int(ident["nh_bg"]), "dual": bool(ident["dual"]),
           "key": key[0], "key16": key[1],
           "geom": ident["geom"], "src": ident["src"],
           "sections": sec, "file_bytes": total,
           "content": dig, "multiset": expect_fp,
           "repack_s": repack_s,
           "write_s": round(write_s, 3), "verify_s": round(t_ver, 3),
           "tool": "brepack_cache.py v%d" % VERSION}
    blob = json.dumps(hdr, sort_keys=True, separators=(",", ":")).encode()
    if len(blob) > HEADER_BYTES - 16:
        raise RuntimeError("header is %d B, the region is %d B"
                           % (len(blob), HEADER_BYTES - 16))
    with open(tmp, "r+b", buffering=0) as fh:
        fh.seek(0)
        fh.write(blob)
        fh.write(b"\n" + b"\x00" * (HEADER_BYTES - len(blob) - 1))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path, hdr


def read_header(path):
    p = pathlib.Path(path)
    with open(p, "rb", buffering=0) as fh:
        blob = fh.read(HEADER_BYTES)
    end = blob.find(b"\x00")
    if end < 0:
        end = len(blob)
    txt = blob[:end].strip()
    if not txt:
        raise ValueError("empty header")
    return json.loads(txt.decode("utf-8"))


def restamp(path=None, prep=None, log=print, dry=False):
    """Re-stamp a cache file's identity after a PURE RENAME of a transform.

    Why this can exist at all: the geometry guard hashes the permutation
    functions, and prod_bigborder.b3_to_image_k32 became a dispatcher over
    _b3_to_image_k32_ref (T22_BB_FASTK32).  The transform is identical, but its
    source text moved, so the derived key moves and a 4.94 GiB artifact would be
    orphaned.  Re-stamping rewrites the 64 KiB header and renames the file; the
    4.94 GiB content is never touched.

    Refuses unless ALL of these hold, and it verifies the content before it
    writes anything:
      * the 60 source .npy digests are unchanged           (guard 1)
      * layers / sizes / nh_bg / dual / alt kind unchanged  (guard 1)
      * the content region still matches the header's sha256 and its word
        multiset                                            (guards 2+3)
    """
    prep = prep or prep_default()
    path = pathlib.Path(path) if path else None
    if path is None:
        cand = sorted(cache_dir().glob("brepack_*.bin"))
        if len(cand) != 1:
            log("[restamp] need exactly one *.bin in %s, found %d"
                % (cache_dir(), len(cand)))
            return None
        path = cand[0]
    if not path.is_file():
        log("[restamp] %s is not a file" % path)
        return None
    hdr = read_header(path)
    ident = identity(prep, hdr["layers"], hdr["bg_bytes"], hdr["b3_bytes"],
                     hdr["nh_bg"], hdr["dual"], hdr["alt_kind"])
    if ident["src"] != hdr["src"]:
        diff = [k for k in ident["src"] if ident["src"].get(k) != hdr["src"].get(k)]
        log("[restamp] REFUSED: the payload sources changed (%d of 60 differ, "
            "e.g. %s). That is not a rename -- rebuild." % (len(diff), diff[:3]))
        return None
    if ident["alt"] != hdr["alt_kind"]:
        log("[restamp] REFUSED: alt kind is %r in the header, %r now"
            % (hdr["alt_kind"], ident["alt"]))
        return None
    for k in ("layers", "bg_bytes", "b3_bytes", "nh_bg", "dual"):
        if ident[k] != hdr[k]:
            log("[restamp] REFUSED: %s is %r in the header, %r now"
                % (k, hdr[k], ident[k]))
            return None
    key = identity_key(ident)
    if key[1] == hdr["key16"]:
        log("[restamp] %s is already stamped with the current identity"
            % path.name)
        return path
    total = int(hdr["file_bytes"])
    if path.stat().st_size != total:
        log("[restamp] REFUSED: %s is %d B, the header says %d"
            % (path.name, path.stat().st_size, total))
        return None
    t0 = time.perf_counter()
    mm = np.memmap(str(path), dtype=np.uint8, mode="r+", shape=(total,))
    try:
        got = content_digest(mm[HEADER_BYTES:],
                             piece=int(hdr["content"]["piece"]))
    finally:
        del mm
    if got["sha256"] != hdr["content"]["sha256"]:
        log("[restamp] REFUSED: content hash does not match the header")
        return None
    ref = hdr.get("multiset") or hdr["content"]
    if not _same_stats(got, ref):
        log("[restamp] REFUSED: content multiset does not match the header")
        return None
    log("[restamp] %s: content verified (sha256 %s..., multiset ok) in %.2f s"
        % (path.name, got["sha256"][:12], time.perf_counter() - t0))
    new = {k: v for k, v in hdr.items()}
    new["key"], new["key16"], new["geom"] = key[0], key[1], ident["geom"]
    new["restamped_from"] = hdr.get("restamped_from") or hdr["key16"]
    new["restamped_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    new["restamp_reason"] = ("geometry digest moved with a pure rename "
                             "(prod_bigborder.b3_to_image_k32 -> dispatcher); "
                             "sources, sizes and content bytes unchanged")
    blob = json.dumps(new, sort_keys=True, separators=(",", ":")).encode()
    if len(blob) > HEADER_BYTES - 16:
        log("[restamp] REFUSED: the restamped header would not fit")
        return None
    dst = file_for(key[1], int(hdr["layers"]), hdr["alt_kind"])
    if dry:
        log("[restamp] DRY: %s -> %s" % (path.name, dst.name))
        return dst
    with open(path, "r+b", buffering=0) as fh:
        fh.seek(0)
        fh.write(blob)
        fh.write(b"\n" + b"\x00" * (HEADER_BYTES - len(blob) - 1))
        fh.flush()
        os.fsync(fh.fileno())
    if dst != path:
        os.replace(path, dst)
    log("[restamp] %s -> %s (header rewritten, 4.94 GiB of content untouched)"
        % (path.name, dst.name))
    return dst


# ---------------------------------------------------------------------------
# load / build
# ---------------------------------------------------------------------------
def _devhand() -> bool:
    return os.environ.get("T22_BIGM_DEVHANDOFF", "0") == "1"


def _dual() -> bool:
    return os.environ.get("T22_BIGM_B3_DUAL", "1") != "0"


def _order_is_map() -> bool:
    return os.environ.get("T22_BIGM_ORDER", "map") != "raw"


def alt_kind_now() -> str:
    if not _dual():
        return "none"
    return "k32" if _devhand() else "n64"


def load(prep, layers, nh_bg, dual=None, devhand=None, verify=None, log=print):
    """Try the cache.  Returns a dict of mapped views, or None.  Never raises."""
    if not _order_is_map():
        log("[brepack] T22_BIGM_ORDER=raw: the B payloads are the delivered "
            "order, the cache does not apply")
        return None
    prep = prep or prep_default()
    dual = _dual() if dual is None else bool(dual)
    devhand = _devhand() if devhand is None else bool(devhand)
    verify = _verify_on() if verify is None else bool(verify)
    t0 = time.perf_counter()
    try:
        ident = identity(prep, layers, 88_473_600, 44_236_800, nh_bg,
                         dual, alt_kind_now())
    except Exception as exc:                                  # noqa: BLE001
        log("[brepack] identity failed: %r" % (exc,))
        return None
    t_id = time.perf_counter() - t0
    key = identity_key(ident)
    path = file_for(key[1], layers, ident["alt"])
    if not path.is_file():
        log("[brepack] cache MISS %s (identity resolved in %.2f s); the "
            "in-run repack will build it" % (path.name, t_id))
        return None
    t1 = time.perf_counter()
    try:
        hdr = read_header(path)
    except Exception as exc:                                  # noqa: BLE001
        log("[brepack] header unreadable (%r); treating as a miss" % (exc,))
        return None
    t_hdr = time.perf_counter() - t1
    want = {"magic": MAGIC, "version": VERSION, "layers": int(layers),
            "nh_bg": int(nh_bg), "dual": bool(dual), "alt_kind": ident["alt"],
            "key": key[0], "key16": key[1], "geom": ident["geom"],
            "src": ident["src"]}
    bad = [k for k, v in want.items() if hdr.get(k) != v]
    if bad:
        log("[brepack] cache rejected: %s differ; treating as a miss" % (bad,))
        return None
    total = int(hdr["file_bytes"])
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size != total:
        log("[brepack] cache is %d B, the header says %d; miss"
            % (size, total))
        return None
    t2 = time.perf_counter()
    if verify:
        mm = np.memmap(str(path), dtype=np.uint8, mode="r+", shape=(total,))
        try:
            got = content_digest(mm[HEADER_BYTES:],
                                 piece=int(hdr["content"]["piece"]))
        finally:
            del mm
        if got["sha256"] != hdr["content"]["sha256"]:
            log("[brepack] CONTENT HASH MISMATCH in %s; treating as a miss"
                % path.name)
            return None
        ref = hdr.get("multiset") or hdr["content"]
        if not _same_stats(got, ref):
            log("[brepack] MULTISET MISMATCH in %s: stored %s, source %s -- "
                "the file is not the permutation it claims to be; miss"
                % (path.name,
                   (got["n_words"], "%016x" % got["s1"], "%016x" % got["s2"]),
                   (ref["n_words"], "%016x" % int(ref["s1"]),
                    "%016x" % int(ref["s2"]))))
            return None
    t_ver = time.perf_counter() - t2
    sec = hdr["sections"]
    bg_b, b3_b = int(hdr["bg_bytes"]), int(hdr["b3_bytes"])
    alt_b = int(hdr["alt_bytes"])
    try:
        # mode="r+" so numpy exposes a WRITABLE buffer: pyxrt's bo.write takes
        # the buffer protocol and the delivered path hands it a writable
        # ndarray view (the same reason webui/bridge.py:595 opens r+).
        bg = np.memmap(str(path), dtype=np.uint8, mode="r+",
                       offset=sec["bg"][0], shape=(int(layers), bg_b))
        b3 = np.memmap(str(path), dtype=np.uint8, mode="r+",
                       offset=sec["b3"][0], shape=(int(layers), b3_b))
        alt = None
        if alt_b:
            alt = np.memmap(str(path), dtype=np.uint8, mode="r+",
                            offset=sec["alt"][0], shape=(int(layers), alt_b))
    except Exception as exc:                                  # noqa: BLE001
        log("[brepack] cannot map %s: %r" % (path.name, exc))
        return None
    log("[brepack] HIT %s (%.2f GB, %s) in %.2f s = identify %.2f + header "
        "%.3f + verify %.2f + map ~0; the 5.31 GB repack is skipped"
        % (path.name, size / 2**30,
           "sha256 + word-multiset ok" if verify else "NOT verified",
           t_id + t_hdr + t_ver, t_id, t_hdr, t_ver))
    return {"path": str(path), "name": path.name, "bytes": size,
            "ident_s": round(t_id, 3), "header_s": round(t_hdr, 3),
            "verify_s": round(t_ver, 3), "verified": bool(verify),
            "key16": key[1], "alt_kind": ident["alt"], "hdr": hdr,
            "bg": bg, "b3": b3, "alt": alt}


_WRITE = {"th": None, "res": None, "err": None}


def _write_async(path, bg, b3, alt, ident, key, src_fp, repack_s, log):
    def work():
        try:
            _WRITE["res"] = write_cache(path, bg, b3, alt, ident, key, src_fp,
                                        repack_s=repack_s)
            p, hdr = _WRITE["res"]
            log("[brepack] cache written: %s (%.2f GB, %.2f s write + %.2f s "
                "re-verify)" % (p.name, hdr["file_bytes"] / 2**30,
                                hdr["write_s"], hdr["verify_s"]))
        except BaseException as exc:                          # noqa: BLE001
            _WRITE["err"] = exc
            log("[brepack] cache write failed: %r" % (exc,))

    th = threading.Thread(target=work, name="t22-brepack-writer", daemon=True)
    _WRITE["th"] = th
    th.start()
    try:
        import atexit
        atexit.register(lambda: th.join(240))
    except Exception:                                         # noqa: BLE001
        pass
    return th


def write_join(timeout=None, log=print):
    th = _WRITE["th"]
    if th is not None:
        th.join(timeout)
        if th.is_alive():
            log("[brepack] the cache writer is still running")
    return _WRITE["res"]


def build(cache_bg, cache_b3, prep, layers, nh_bg, dual=None, devhand=None,
          log=print, write=True):
    """Repack the resident rows and store the result.

    Mirrors repack_cache_inplace_bigm exactly: the resident arrays are updated
    in place, so every consumer downstream sees the same bytes it saw before.
    """
    if not _order_is_map():
        return None
    prep = prep or prep_default()
    dual = _dual() if dual is None else bool(dual)
    devhand = _devhand() if devhand is None else bool(devhand)
    _, PB = _perm_modules()
    n = int(layers) if cache_bg is not None else 0
    if n <= 0:
        return None
    bg_b = int(cache_bg.shape[1])
    b3_b = int(cache_b3.shape[1]) if cache_b3 is not None else 0
    alt_kind = "none" if (not dual or cache_b3 is None) else (
        "k32" if devhand else "n64")
    t0 = time.perf_counter()
    PB._B3_ALT = np.empty_like(cache_b3) if alt_kind != "none" else None
    fp_bg = [0, 0, 0]
    fp_b3 = [0, 0, 0]
    for li in range(n):
        src_bg = cache_bg[li]
        src_b3 = cache_b3[li] if cache_b3 is not None else None
        _acc(fp_bg, src_bg)
        if src_b3 is not None:
            _acc(fp_b3, src_b3)
        bg_i, b3_i, alt_i = repack_rows(src_bg, src_b3, nh_bg, dual, devhand)
        if li == 0:
            # guard 2, anchored rigorously on the first layer: the exact byte
            # histogram AND the word multiset must both survive the repack.
            # The destination carries BG once and B3 TWICE (the delivered down
            # generation and the second one are the same bytes re-indexed).
            fp_dst = [0, 0, 0]
            for a in (bg_i, b3_i, alt_i):
                if a is not None:
                    _acc(fp_dst, a)
            fp_want = _add(fp_bg, _mul(fp_b3, 2 if alt_i is not None else 1))
            if fp_dst != fp_want:
                raise RuntimeError(
                    "brepack: layer 0 word multiset changed (%s -> %s): the "
                    "repack re-encoded bytes" % (fp_want, fp_dst))
            for a, b in ((src_bg, bg_i), (src_b3, b3_i), (src_b3, alt_i)):
                if a is None or b is None:
                    continue
                if not np.array_equal(_hist256(a), _hist256(b)):
                    raise RuntimeError(
                        "brepack: layer 0 byte histogram changed over %d B: "
                        "not a permutation" % a.size)
        cache_bg[li] = bg_i
        if src_b3 is not None and dual:
            cache_b3[li] = b3_i
            if alt_i is not None:
                PB._B3_ALT[li] = alt_i
        del bg_i, b3_i, alt_i
    t_rp = time.perf_counter() - t0
    # guard 2 over the whole cache: the multiset of everything handed to the
    # device must equal the multiset of everything the product read.
    dst = [0, 0, 0]
    for arr in (cache_bg, cache_b3 if dual else None,
                PB._B3_ALT if alt_kind != "none" else None):
        if arr is None:
            continue
        for li in range(n):
            _acc(dst, arr[li])
    want = _add(fp_bg, _mul(fp_b3, 2 if alt_kind != "none" else 1))
    if dst != want:
        raise RuntimeError("brepack: whole-cache multiset changed (%s -> %s)"
                           % (want, dst))
    log("[brepack] repacked %d layers in %.2f s; word multiset identical "
        "(%d words, s1=%016x s2=%016x), layer-0 byte histograms exact"
        % (n, t_rp, dst[0], dst[1], dst[2]))
    out = {"bg": cache_bg, "b3": cache_b3, "alt": PB._B3_ALT,
           "alt_kind": alt_kind, "repack_s": round(t_rp, 3),
           "path": None, "name": None, "bytes": 0, "verified": False,
           "hdr": None, "key16": None}
    if not write or os.environ.get("T22_BREPACK_BUILD", "1") == "0":
        return out
    try:
        ident = identity(prep, n, bg_b, b3_b, nh_bg, dual, alt_kind)
        key = identity_key(ident)
        path = file_for(key[1], n, alt_kind)
        out.update({"path": str(path), "name": path.name, "key16": key[1]})
        stale = path.with_name(path.name + ".tmp")
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass
        fp = {"n_words": want[0], "s1": want[1], "s2": want[2]}
        sync = os.environ.get("T22_BREPACK_WRITE_SYNC", "0") == "1"
        args = (path, cache_bg, cache_b3 if dual else None,
                PB._B3_ALT if alt_kind != "none" else None,
                ident, key, fp, round(t_rp, 3), log)
        if sync:
            p, hdr = write_cache(*args)
            out.update({"bytes": int(hdr["file_bytes"]), "hdr": hdr})
            log("[brepack] cache written: %s (%.2f GB, %.2f s)"
                % (p.name, hdr["file_bytes"] / 2**30, hdr["write_s"]))
        else:
            out["bytes"] = int(_sections(n, bg_b, b3_b, alt_kind)[1])
            _write_async(*args)
    except Exception as exc:                                  # noqa: BLE001
        log("[brepack] could not schedule the cache write (%r); the run "
            "continues without it" % (exc,))
    return out


def arm(hit, log=print):
    """Make prod_loop_bigm hand the cached payloads out without repacking."""
    _ensure_path()
    import prod_loop_bigm as PB
    PB._B3_ALT = hit["alt"]
    PB._BGM_CACHE_READY = True
    return True


# ---------------------------------------------------------------------------
# background prefetch, so the 0.65 s verify hides under device open
# ---------------------------------------------------------------------------
_PRE = {"th": None, "res": None, "err": None, "args": None}


def start(prep, layers, nh_bg, dual=None, devhand=None, log=print):
    """Begin resolving + verifying the cache on a worker thread."""
    if not enabled():
        return None
    if _PRE["th"] is not None:
        return _PRE["th"]
    _PRE["args"] = (prep, layers, nh_bg, dual, devhand, log)

    def work():
        a = _PRE["args"]
        try:
            _PRE["res"] = load(*a)
        except BaseException as exc:                          # noqa: BLE001
            _PRE["err"] = exc

    th = threading.Thread(target=work, name="t22-brepack-cache", daemon=True)
    _PRE["th"] = th
    th.start()
    return th


def join(log=print):
    th = _PRE["th"]
    if th is not None:
        th.join()
    if _PRE["err"] is not None:
        log("[brepack] cache prefetch failed: %r" % (_PRE["err"],))
        return None
    return _PRE["res"]


def started() -> bool:
    return _PRE["th"] is not None


# ---------------------------------------------------------------------------
# CLI (CPU only -- no device, no lock)
# ---------------------------------------------------------------------------
def _fill_sources(prep, layers, nh_bg):
    """The product's resident cache, built the way _preload_payload builds it."""
    _ensure_path()
    import t11_permute as P11
    bg = np.empty((layers, 88_473_600), np.uint8)
    b3 = np.empty((layers, 44_236_800), np.uint8)
    for li in range(layers):
        src = np.load(pathlib.Path(prep) / ("layers.%d_bg.npy" % li))
        if nh_bg > 1:
            P11.permute_bg(src.reshape(-1), bg[li], nh_bg)
        else:
            np.copyto(bg[li], src.reshape(-1))
        np.copyto(b3[li], np.load(pathlib.Path(prep)
                                  / ("layers.%d_b3.npy" % li)).reshape(-1))
        del src
    return bg, b3


def _selftest():
    """CPU-only gates: sizes, bijectivity, multiset, fast-k32 equality."""
    ok = True
    BB, PB = _perm_modules()
    rng = np.random.default_rng(7)
    bg = rng.integers(0, 256, 88_473_600, dtype=np.uint8)
    b3 = rng.integers(0, 256, 44_236_800, dtype=np.uint8)
    _fast = _k32_fast(BB)
    cases = (("BG", bg, BB.bg_to_image(bg)),
             ("B3", b3, BB.b3_to_image(b3)),
             ("B3n64", b3, BB.b3_to_image_n64(b3)),
             ("B3k32", b3, _k32_ref(BB)(b3)),
             ("B3k32f", b3, _fast(b3) if _fast is not None else _k32_ref(BB)(b3)))
    for nm, src, dst in cases:
        print("[selftest] %-7s size %-5s word-multiset %-5s byte-hist %-5s "
              "moved %s" % (nm, src.size == dst.size,
                            _stats(src) == _stats(dst),
                            bool(np.array_equal(_hist256(src), _hist256(dst))),
                            not np.array_equal(src, dst)), flush=True)
        ok = ok and bool(src.size == dst.size)
        ok = ok and _stats(src) == _stats(dst)
        ok = ok and bool(np.array_equal(_hist256(src), _hist256(dst)))
    same = (_fast is None) or bool(np.array_equal(_k32_ref(BB)(b3), _fast(b3)))
    print("[selftest] vectorised k32 == reference: %s" % same, flush=True)
    ok = ok and same
    # the unpermute must be the exact inverse of the product's NH-5 permute
    import t11_permute as P11
    perm = np.empty_like(bg)
    P11.permute_bg(bg, perm, 5)
    back = PB._bg_unpermute(perm, 5)
    rt = bool(np.array_equal(back, bg))
    print("[selftest] _bg_unpermute(permute_bg(x,5),5) == x: %s" % rt, flush=True)
    ok = ok and rt
    print("[selftest] %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def _cli(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--stat", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--restamp", action="store_true",
                    help="re-stamp the header after a PURE RENAME of a "
                         "permutation function (verifies the content first)")
    ap.add_argument("--restamp-dry", action="store_true")
    ap.add_argument("--file", default="")
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--nh-bg", type=int, default=int(os.environ.get("NH_BG", "5")))
    ap.add_argument("--prep",
                    default=str(NPU / "imaggers" / "image" / "t3real_prep"))
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.restamp or a.restamp_dry:
        p = restamp(a.file or None, a.prep, dry=bool(a.restamp_dry))
        if p is None:
            print("restamp did not happen")
            return 1
        if a.restamp_dry:
            return 0
        info = load(a.prep, a.layers, a.nh_bg)
        if info is None:
            print("restamped, but the load does not HIT: investigate")
            return 1
        print("restamped and re-verified: HIT")
        return 0
    if a.stat or a.verify:
        info = load(a.prep, a.layers, a.nh_bg)
        if info is None:
            print("no usable cache")
            return 1
        h = info["hdr"]
        keep = ("magic", "version", "built", "layers", "bg_bytes", "b3_bytes",
                "alt_kind", "nh_bg", "dual", "key16", "content", "multiset",
                "sections", "file_bytes", "write_s", "verify_s")
        print(json.dumps({k: h[k] for k in keep}, indent=1))
        return 0
    if a.build:
        t0 = time.perf_counter()
        bg, b3 = _fill_sources(a.prep, a.layers, a.nh_bg)
        print("[build] sources resident in %.2f s" % (time.perf_counter() - t0),
              flush=True)
        out = build(bg, b3, a.prep, a.layers, a.nh_bg)
        write_join()
        print("[build] done: %s" % ((out or {}).get("path"),))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
