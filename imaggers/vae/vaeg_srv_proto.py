# vaeg_srv_proto.py -*- Python 3.8+ -*-
"""Framing for the VAE device server (py310 child <-> py312 host).

One message = MAGIC | u32 n_parts | (u32 len | bytes) * n_parts

Part 0 is always a JSON header (utf-8); later parts are raw device payloads.
No pickle, no msgpack: both sides are plain Python and the blobs are bytes
that go straight into a BO.  Reads are exact (a pipe read may be short); writes
go through a BufferedWriter so a 10 MB payload is a bounded number of syscalls.
"""
from __future__ import annotations

import io
import struct

MAGIC = b"VDS1"


def reader(f):
    """Wrap a raw binary pipe for 1 MB buffered exact reads."""
    return io.BufferedReader(f, 1 << 20)


def writer(f):
    return io.BufferedWriter(f, 1 << 20)


def write_all(f, b) -> None:
    mv = memoryview(b)
    n = 0
    while n < len(mv):
        n += f.write(mv[n:])


def send_msg(f, parts) -> None:
    write_all(f, MAGIC)
    write_all(f, struct.pack("<I", len(parts)))
    for p in parts:
        write_all(f, struct.pack("<I", len(p)))
        write_all(f, p)
    f.flush()


def _read_exact(f, n: int):
    buf = bytearray(n)
    got = 0
    while got < n:
        chunk = f.read(n - got)
        if not chunk:
            return None
        buf[got:got + len(chunk)] = chunk
        got += len(chunk)
    return bytes(buf)


def recv_msg(f):
    """None means the peer closed (crash or clean exit)."""
    m = _read_exact(f, len(MAGIC))
    if m is None:
        return None
    if m != MAGIC:
        raise IOError("bad frame magic %r" % (m,))
    raw = _read_exact(f, 4)
    if raw is None:
        return None
    n = struct.unpack("<I", raw)[0]
    if n > 64:
        raise IOError("absurd part count %d" % n)
    parts = []
    for _ in range(n):
        raw = _read_exact(f, 4)
        if raw is None:
            return None
        ln = struct.unpack("<I", raw)[0]
        body = _read_exact(f, ln)
        if body is None:
            return None
        parts.append(body)
    return parts
