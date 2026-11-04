# SPDX-License-Identifier: MIT
"""provenance.py -- what the job record must say about the environment.

INSTALLED AS webui/provenance.py by imaggers/par/provenance/install.py.  It is
record-only: it reads os.environ, never writes it, and never touches a tensor.

THE PROBLEM THIS CLOSES
    Two runs the tree believed were the SAME arm differed at corr 0.999108509136.
    They were two different numeric arms, selected by T22_BREPACK_CACHE, which
    appeared in NEITHER job record.  A resolved value without its origin cannot
    distinguish "we chose this" from "the default did it", so this module records
    three things per variable -- value, explicit, source -- for every variable the
    product reads, and it records an ARM ID derived only from the variables whose
    effect class can move bytes or pick a dispatch path.

WHAT "SOURCE" MEANS (the rule, stated so it can be argued with)
    shell    present in the process environment when this module was imported and
             unchanged since -- unless ZIMAGE_LAUNCH_SOURCE says otherwise
    recipe   present at import AND ZIMAGE_LAUNCH_SOURCE names a recipe (e.g.
             "recipe:t22_repro.py"), or ZIMAGE_ENV_SOURCE lists the name
    runtime  set or changed AFTER import -- nobody declared it, so it is flagged
    default  absent: the value is the code default, and the registry says which
             literal that is and which readers use a different one

    A recipe that starts the product with its env in the child cannot be told
    apart from a shell export by inspection; the honest fix is the one-line
    declaration (ZIMAGE_LAUNCH_SOURCE), and without it the source reads "shell".
    That is why the field exists rather than being inferred and believed.

THE GUARD, IN THE RECORD
    unknown_present lists every variable in the product's own namespace that is
    present in the environment but NOT in the registry.  A job that carries one is
    not reproducible by the record, and says so in its own file.
"""
from __future__ import annotations

import hashlib
import json
import os

try:                                                       # installed in webui/
    from . import knobs as _knobs
except (ImportError, ValueError):                          # imported flat (tools)
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import knobs as _knobs

SCHEMA = "zimage.env-provenance/1"

# the environment as it was when this module was first imported.  Taken here and
# not in record() on purpose: a variable added after the process started is a
# RUNTIME injection, and the record has to be able to say so.
_BOOT = dict(os.environ)
_BOOT_ORDER = sorted(_BOOT)

SOURCE_HELP = {
    "shell": "inherited at process start, no recipe declared",
    "recipe": "declared by the launcher (ZIMAGE_LAUNCH_SOURCE / ZIMAGE_ENV_SOURCE)",
    "runtime": "set or changed after the process started -- undeclared",
    "default": "absent; the value is the code default in the registry",
}


def launch_source(environ=None):
    env = os.environ if environ is None else environ
    raw = (env.get("ZIMAGE_LAUNCH_SOURCE") or "").strip()
    return raw or "shell"


def source_manifest(environ=None):
    env = os.environ if environ is None else environ
    raw = (env.get("ZIMAGE_ENV_SOURCE") or "").strip()
    if not raw:
        return {}
    try:
        got = json.loads(raw)
    except ValueError:
        return {}
    return got if isinstance(got, dict) else {}


def resolve(name, environ=None, boot=None, manifest=None, label=None):
    """One knob: value, raw, explicit, source.  Never guesses, never writes."""
    env = os.environ if environ is None else environ
    boot = _BOOT if boot is None else boot
    manifest = source_manifest(env) if manifest is None else manifest
    label = launch_source(env) if label is None else label
    k = _knobs.KNOBS.get(name)
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return {"value": _knobs.normalize(name, k["default"] if k else ""),
                "raw": None,
                "explicit": False,
                "source": "default"}
    explicit = True
    if name in manifest:
        source = str(manifest[name])
    elif name in boot and boot[name] == raw:
        source = "recipe" if label.startswith("recipe") else "shell"
    else:
        source = "runtime"
    out = {"value": _knobs.normalize(name, raw), "raw": raw,
           "explicit": explicit, "source": source}
    injected = getattr(_knobs, "INJECTED_BY", {}).get(name)
    if injected and source in ("shell", "runtime"):
        out["injected_by"] = injected
        # the product writes this itself; a value equal to its own default is
        # NOT a choice anybody made
        if _knobs.normalize(name, raw) == _knobs.normalize(name, k["default"] if k else ""):
            out["source"] = "default"
            out["explicit"] = False
    return out


def unknown_present(environ=None):
    """Product-namespace variables that are set but not in the registry."""
    env = os.environ if environ is None else environ
    return sorted(n for n in env
                  if _knobs.in_namespace(n) and n not in _knobs.KNOBS
                  and n not in _knobs.META_ENV)


def record(environ=None, boot=None, label=None):
    """The whole record: every registered knob + the arm identity."""
    env = os.environ if environ is None else environ
    boot = _BOOT if boot is None else boot
    label = launch_source(env) if label is None else label
    manifest = source_manifest(env)
    out = {}
    counts = {"total": 0, "explicit": 0, "implicit": 0,
              "shell": 0, "recipe": 0, "runtime": 0, "source_default": 0, "arm": 0}
    for name in _knobs.names():
        r = resolve(name, env, boot, manifest, label)
        k = _knobs.KNOBS[name]
        entry = {"value": r["value"], "explicit": r["explicit"], "source": r["source"]}
        if r["raw"] is not None:
            entry["raw"] = r["raw"]
        if k["conflict"]:
            entry["default_conflict"] = True
        out[name] = entry
        counts["total"] += 1
        counts["explicit" if r["explicit"] else "implicit"] += 1
        if r["source"] in ("shell", "recipe", "runtime"):
            counts[r["source"]] += 1
        else:
            counts["source_default"] += 1
        if k["cls"] in _knobs.ARM_CLASSES:
            counts["arm"] += 1
    arm = _knobs.arm_values(env)
    try:
        unknown = unknown_present(env)
    except Exception:                                  # env_audit not installed
        unknown = []
    rec = {
        "schema": SCHEMA,
        "registry": {"schema": _knobs.SCHEMA, "sha256": _knobs.registry_sha(),
                     "knobs": len(_knobs.KNOBS)},
        "arm_id": hashlib.sha256(json.dumps(arm, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()[:16],
        "arm_values": arm,
        "launch_source": label,
        "counts": counts,
        "env_digest": hashlib.sha256(json.dumps(out, sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest(),
        "vars": out,
        "unknown_present": unknown,
        "changed_after_boot": sorted(
            n for n in set(env) | set(boot)
            if env.get(n) != boot.get(n) and n in _knobs.KNOBS),
    }
    return rec


def flags(environ=None):
    """Superset of webui/config.py resolved_flags() with the same shape.

    config.py is patched to rebind resolved_flags to this, so a caller that
    already reads job.json:env_resolved silently starts getting the complete set.
    """
    env = os.environ if environ is None else environ
    label = launch_source(env)
    manifest = source_manifest(env)
    out = {}
    for name in _knobs.names():
        k = _knobs.KNOBS[name]
        r = resolve(name, env, _BOOT, manifest, label)
        out[name] = {"value": r["value"],
                     "source": r["source"],
                     "explicit": r["explicit"],
                     "raw": r["raw"],
                     "default_is": k["default"],
                     "effect": k["cls"]}
    return out


def arm_id(environ=None):
    return _knobs.arm_id(environ)


def summary(rec=None):
    """One line for a log: the arm identity and where each value came from."""
    rec = rec or record()
    c = rec["counts"]
    return ("[provenance] arm_id %s  %d knobs (%d arm)  explicit %d / default %d  "
            "launch %s  unknown %d"
            % (rec["arm_id"], c["total"], c["arm"], c["explicit"], c["source_default"],
               rec["launch_source"], len(rec["unknown_present"])))


def _selftest():
    """Record-only is a claim, so prove it: no env mutation, deterministic id."""
    before = dict(os.environ)
    a = record()
    b = record()
    assert dict(os.environ) == before, "provenance mutated the environment"
    assert a["arm_id"] == b["arm_id"], "arm id is not deterministic"
    assert a["env_digest"] == b["env_digest"], "record is not deterministic"
    assert set(a["vars"]) == set(_knobs.KNOBS), "record does not cover the registry"
    assert not unknown_present(), "unknown env vars present: %r" % unknown_present()
    print(summary(a))
    return True


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(json.dumps(record(), indent=1, sort_keys=True))
