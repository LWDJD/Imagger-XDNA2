# SPDX-License-Identifier: MIT
"""stepskip.py -- a SCHEDULE-DERIVED WHOLE-STEP SKIP for the Z-Image-Turbo
8-step FlowMatch-Euler loop, with the decision rule and its assertion.

Nothing here is a cache.  A skipped step performs NO DiT call, stores no
residual and consults no history of activations: the set of skipped steps is
fixed by the sigma schedule BEFORE the first latent exists, and the latent is
advanced by the product's own integrator statement with the missing model
output replaced by the last one that was actually computed.

--------------------------------------------------------------------------------
1. WHERE IT GOES IN THE PRODUCT, AND WHY THAT IS THE ONLY SEAM
--------------------------------------------------------------------------------
imaggers/image/t18_run.py:2934-2942 is the whole sampler:

    for i, t in enumerate(timesteps):
        if i < start:
            continue
        preds, rec = one_step(i, t, lat, args.host_ref and i == 0)
        _t = time.perf_counter()
        d = np.float32(sigmas[i + 1] - sigmas[i])
        for b in range(B):
            lat[b] = (lat[b] + d * (-preds[b])).astype(np.float32)

Three facts about it decide the design:

  (a) ONE EVALUATION PER STEP, FORWARD EULER.  one_step is called once and its
      output is multiplied by d and added to lat.  The ODE being integrated is
      dx/dsigma = -model(x, sigma); preds[b] is the model's output at
      (lat[b], t).  There is no multistep history, no accumulator, no
      predictor-corrector: the state between steps is lat ALONE.  So a skipped
      step is fully determined by one question -- what value does the update
      use in place of model(x_k, sigma_k)?

  (b) d IS READ AT THE CURRENT INDEX.  d = sigmas[i+1] - sigmas[i] is the
      CURRENT interval's width.  Whatever model output is substituted is
      therefore integrated over [sigma_i, sigma_i+1] -- the interval that is
      actually being crossed.  This matters because this schedule's steps are
      wildly unequal: |dsigma| runs 0.00893 ... 0.32904, a factor of 36.8.  A
      substitute must therefore be a FIELD VALUE at step k, never a
      displacement carried over from step k-1, whose width was a different
      number.

      This is exactly where sd.cpp's primitive parts company with the loop.
      condition_cache_utils.hpp:32-34/56-57 (verified in _sdref) stores
      diff = out - in and returns out = in + diff, so on a skipped step the
      model output handed to the integrator is in_k + (out_{k-1} - in_{k-1}),
      i.e. the previous output PLUS the latent's own motion since the previous
      step.  The substituted quantity therefore carries a term that the
      integrator has already applied, and it is not the previous field value.
      STEP-CACHE-PRICING.md section 3 measures the x0-flavoured restatement of
      that substitution at rel_rms 1.89e-2 for the last step, against 6.78e-3
      for the velocity hold -- 2.8x worse for the same skipped step.  Nothing
      from it is reused here.

  (c) NOTHING DOWNSTREAM RE-EVALUATES THE SKIPPED STATE.  lat is only ever
      read by the next one_step and finally by the VAE.  So the error a skip
      injects is exactly

          ||delta|| = |d_k| * || f(x_k, s_k) - <substitute> ||

      and the entire design question is which substitute minimises that.

Three candidate substitutes, and what (a)-(c) say about each:

  HOLD  (this module, policy 'hold'): substitute f(x_k,s_k) ~= f(x_j,s_j),
        the last evaluation actually made (j < k).  Error |d_k|*||f_k - f_j||.
        This is the zero-order hold of the field in sigma and it is the only
        substitute that uses information the integrator itself consumes.

  ZERO  (policy 'zero': carry the latent forward, lat <- lat): substitute 0.
        Error |d_k|*||f_k||.  HOLD beats ZERO iff ||f_k - f_j|| < ||f_k||,
        i.e. iff the measured per-step renewal rate is below 1.  Measured on
        this trajectory (STEP-CACHE-PRICING.md section 2): 1.326 at k=1 and
        0.434-0.659 at k>=2.  So HOLD is strictly better than ZERO at every
        step this rule would ever pick, and worse at exactly one step (k=1)
        that it never picks.  Both are implemented here; policy 'zero' is the
        control that measures the difference instead of asserting it.

  X0  (what sd.cpp's three caches compute under a velocity output):
        out = in + (out_prev - in_prev) = x + (v_prev - x) = v_prev, then
        denoised = x - sigma_k * v_prev, i.e. the previous CLEAN-LATENT
        estimate.  It divides by sigma_k, which is 0.00893 at k=7 -- exactly
        where this schedule has its redundancy -- and is measured (same
        report, same harness) at rel_rms 1.89e-2 against HOLD's 6.78e-3 for
        the same skipped step.  Not implemented: it is not a candidate.

--------------------------------------------------------------------------------
2. THE RULE, AND ITS ASSERTION
--------------------------------------------------------------------------------
    pattern = argmin over k in 1..n-1 of |sigma_{k+1} - sigma_k|

k = 0 is excluded because there is no previous evaluation to hold (see (c) at
k=0: 'held' is None).  No threshold, no data, no tuning, no oracle: sigma alone.

For the product's 8-step schedule (sigmas[0..8] = 1, .94754255, .88278776,
.80083734, .69379312, .54804558, .33797216, .00892857, 0) the rule returns
{7} and the module ASSERTS it: |dsigma_7| = 8.9286e-3 = 0.89 % of the sigma
range, and the runner-up (k=1, 6.4755e-2) is 7.25x larger, so the rule is
unambiguous -- there is no tie to break.

The rule is deliberately capped at ONE skip.  Extended to two, the same rule
takes k=1 second, and {1,7} is measured at corr 0.983623 / rel_rms 0.1833 --
a hard FAIL, the WORST of the seven single-skip arms.  So the second-cheapest
step by this criterion is the most expensive step in fact.  That is why
{4,7} is NOT a rule output and is only accepted here as an EXPLICIT, RECORDED
literal pattern (C1) pending the SEQ-1056 transfer test (E1).

--------------------------------------------------------------------------------
3. DEFAULT OFF  (gate C0)
--------------------------------------------------------------------------------
resolve() returns an EMPTY pattern unless ZIMAGE_SKIP_STEPS is set.  With it
unset the loop takes the else branch on every step, so the extra sources of
change are exactly zero: 'held = preds' is a reference assignment and the
computed d / lat statements are byte-for-byte the product's.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

ENV = "ZIMAGE_SKIP_STEPS"          # "" / "auto" / "7" / "4,7"
POLICY_ENV = "ZIMAGE_SKIP_POLICY"  # "hold" (default) | "zero"
POLICY_HOLD, POLICY_ZERO = "hold", "zero"
RULE = "smallest_dsigma"
RULE_VERSION = 1

# the product's 8-step FlowMatchEuler schedule (imaggers/image/t3real_traj.npz)
N_STEPS = 8
SIGMAS_8 = np.array([1.0, 0.94754255, 0.88278776, 0.80083734, 0.69379312,
                     0.54804558, 0.33797216, 0.00892857, 0.0], np.float64)
TIMESTEPS_8 = np.array([1000.0, 947.5425415, 882.78778076, 800.83734131,
                        693.79309082, 548.04559326, 337.97216797,
                        8.9285717], np.float64)
EXPECTED_PATTERN = (7,)


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------
def dsigma(sigmas):
    """The schedule's own step widths, signed (all negative here), length n."""
    s = np.asarray(sigmas, np.float64)
    return np.diff(s)


def pattern_from_schedule(sigmas, n_skip=1, eligible_from=1):
    """The decision rule.  Schedule only -- no latent, no model, no data.

    Returns (pattern, table) where table is the full ranking with every number
    the rule looked at, so the choice is auditable rather than asserted.
    """
    ds = dsigma(sigmas)
    n = len(ds)
    cand = [k for k in range(max(0, int(eligible_from)), n)]
    if not cand:
        return [], {"candidates": [], "dsigma": ds.tolist(),
                    "why": "no eligible step (n_steps=%d)" % n}
    order = sorted(cand, key=lambda k: (abs(ds[k]), k))
    rg = float(np.max(sigmas) - np.min(sigmas))
    rows = []
    for rank, k in enumerate(order):
        rows.append({"rank": rank, "step": int(k),
                     "abs_dsigma": float(abs(ds[k])),
                     "share_of_sigma_range": float(abs(ds[k]) / rg) if rg else 0.0,
                     "sigma_k": float(sigmas[k]),
                     "sigma_k1": float(sigmas[k + 1])})
    take = order[:max(0, int(n_skip))]
    margin = (abs(ds[order[1]]) / abs(ds[order[0]])
              if len(order) > 1 and abs(ds[order[0]]) > 0 else float("inf"))
    degenerate = bool(abs(ds[order[0]]) == 0.0)
    table = {"rule": RULE, "rule_version": RULE_VERSION, "n_steps": n,
             "sigma_range": rg, "candidates": [int(k) for k in cand],
             "ranking": rows, "eligible_from": int(eligible_from),
             "n_skip": int(n_skip),
             "abs_dsigma": [float(abs(v)) for v in ds],
             "dsigma": [float(v) for v in ds],
             "margin_runner_up": float(margin),
             "degenerate_zero_width_step": degenerate,
             "why": "argmin |dsigma| over steps %d..%d; min is step %d"
                    % (min(cand), max(cand), order[0])}
    return sorted(int(k) for k in take), table


def assert_product_pattern(sigmas=SIGMAS_8, pattern=None, table=None):
    """The assertion the brief asks for: the rule selects EXACTLY {7}.

    Raises AssertionError with the offending numbers if it does not.  Also
    checks the three structural properties the arm leans on.
    """
    if pattern is None:
        pattern, table = pattern_from_schedule(sigmas)
    pat = tuple(int(k) for k in pattern)
    if pat != EXPECTED_PATTERN:
        raise AssertionError(
            "the smallest-dsigma rule selected %r on this schedule, expected %r"
            % (pat, EXPECTED_PATTERN))
    ds = np.abs(dsigma(sigmas))
    rg = float(np.max(sigmas) - np.min(sigmas))
    if float(ds[pat[0]]) == 0.0:
        # a schedule whose smallest interval has zero width cannot be asserted
        # against this arm: |dsigma| = 0 means the "step" moves nothing, the
        # margin is infinite and the rule's whole premise is vacuous.
        raise AssertionError(
            "step %d has |dsigma| = 0 on this schedule (sigmas end %r): the "
            "rule is degenerate here and the arm must not be enabled"
            % (pat[0], [float(v) for v in np.asarray(sigmas)[-3:]]))
    # (i) the selected step is the global minimum of |dsigma| among i >= 1
    if int(np.argmin(ds[1:])) + 1 != pat[0]:
        raise AssertionError("step %d is not argmin|dsigma| over i>=1" % pat[0])
    # (ii) it is a rounding error's worth of the schedule's whole motion
    share = float(ds[pat[0]] / rg)
    if not share < 1e-2:
        raise AssertionError("selected step is %.4f of the sigma range, not <1%%"
                             % share)
    # (iii) no tie: the runner-up is far away, so no tie-break rule is needed
    others = sorted(float(v) for v in ds[1:] if float(v) != float(ds[pat[0]]))
    margin = others[0] / float(ds[pat[0]])
    if not margin > 5.0:
        raise AssertionError("runner-up is only %.2fx away -- a tie-break rule "
                             "would be needed" % margin)
    return {"pattern": list(pat), "share_of_sigma_range": share,
            "margin_runner_up": margin,
            "abs_dsigma_selected": float(ds[pat[0]]),
            "abs_dsigma_runner_up": float(others[0]),
            "why": "step %d spends %.3f%% of the sigma range on 1/%d of the "
                   "model evaluations; the runner-up (step %d) is %.2fx away"
                   % (pat[0], 100.0 * share, len(ds),
                      int([k for k in range(1, len(ds))
                           if float(ds[k]) == others[0]][0]), margin)}


# ---------------------------------------------------------------------------
# the plan object
# ---------------------------------------------------------------------------
class Plan:
    """A resolved skip set.  as_dict() is the job.json payload (gate C1)."""

    def __init__(self, pattern, mode, policy, table=None, assertion=None,
                 n_steps=N_STEPS, source="schedule"):
        self.pattern = tuple(sorted(int(k) for k in pattern))
        self.mode = mode                # "off" | "auto" | "explicit"
        self.policy = policy            # "hold" | "zero"
        self.table = table or {}
        self.assertion = assertion or {}
        self.n_steps = int(n_steps)
        self.source = source            # "schedule" | "literal"

    def __len__(self):
        return len(self.pattern)

    def __contains__(self, k):
        return int(k) in self.pattern

    def __repr__(self):
        return "Plan(pattern=%r, mode=%r, policy=%r)" % (
            list(self.pattern), self.mode, self.policy)

    @property
    def start(self):
        """First step whose INPUT could differ from a no-skip run."""
        return self.pattern[0] if self.pattern else self.n_steps

    def as_dict(self, label=None):
        d = {"mode": self.mode, "policy": self.policy,
             "pattern": list(self.pattern), "n_skipped": len(self.pattern),
             "n_steps": self.n_steps, "source": self.source,
             "rule": RULE if self.mode == "auto" else self.mode,
             "rule_version": RULE_VERSION,
             "env": {ENV: os.environ.get(ENV, ""),
                     POLICY_ENV: os.environ.get(POLICY_ENV, "")},
             "assertion": self.assertion, "schedule": self.table}
        if label:
            d["label"] = label
        d["pattern_sha12"] = hashlib.sha256(
            (",".join(str(k) for k in self.pattern)).encode()).hexdigest()[:12]
        return d


def resolve(spec=None, sigmas=SIGMAS_8, n_steps=N_STEPS, policy=None,
            require_assertion=True):
    """Turn the env/config surface into a Plan.  DEFAULT IS OFF.

    spec  None -> read ZIMAGE_SKIP_STEPS.  "" / "0" / "off" -> no skip.
          "auto" -> the rule.
          "7" / "4,7" -> an explicit literal pattern (recorded as such, C1).
    """
    if spec is None:
        spec = os.environ.get(ENV, "")
    spec = ("" if spec is None else str(spec)).strip().lower()
    policy = (policy or os.environ.get(POLICY_ENV, "")
              or POLICY_HOLD).strip().lower()
    if policy not in (POLICY_HOLD, POLICY_ZERO):
        raise ValueError("unknown policy %r" % (policy,))
    if spec in ("", "0", "off", "none", "false"):
        return Plan([], "off", policy, {}, {}, n_steps)

    if spec in ("auto", "rule", "schedule"):
        pat, table = pattern_from_schedule(sigmas)
        ass = assert_product_pattern(sigmas, pat, table) if require_assertion else {}
        return Plan(pat, "auto", policy, table, ass, n_steps, source="schedule")

    try:
        pat = sorted({int(v) for v in spec.replace(" ", "").split(",") if v != ""})
    except ValueError:
        raise ValueError("%s=%r is not 'auto' or a comma-separated step list"
                         % (ENV, spec))
    if not pat:
        return Plan([], "off", policy, {}, {}, n_steps)
    if pat[0] < 1:
        raise ValueError("step 0 cannot be skipped: there is no previous model "
                         "output to hold (t18_run.py:2937 -- held is None at "
                         "i == start == 0)")
    if pat[-1] > n_steps - 1:
        raise ValueError("step %d is outside a %d-step schedule"
                         % (pat[-1], n_steps))
    if any(b - a == 1 for a, b in zip(pat, pat[1:])):
        raise ValueError("pattern %r has adjacent skips; two consecutive holds "
                         "compound the same field error and are outside what "
                         "this arm priced" % (pat,))
    table, ass = {}, {}
    if require_assertion and tuple(pat) == EXPECTED_PATTERN:
        table, ass = pattern_from_schedule(sigmas)[1], assert_product_pattern(sigmas)
    return Plan(pat, "explicit", policy, table, ass, n_steps, source="literal")


# ---------------------------------------------------------------------------
# the loop -- t18_run.py:2934-2942 with the guard
# ---------------------------------------------------------------------------
def skip_rec(i, held_from, policy):
    """The per-step record a skipped step emits (C1 is read out of this)."""
    return {"step": int(i), "skipped": True, "held_from_step": int(held_from),
            "policy": policy, "pred_absmax": None, "wall_s": 0.0,
            "model_calls": 0}


def run_chain(step_fn, latents, timesteps, sigmas, plan, start=0, on_step=None,
              held0=None, held_at0=None):
    """THE MECHANISM.  step_fn(i, t, latents) -> list[np.ndarray] is one_step.

    latents is a list of B arrays, exactly as t18_run.py keeps it.  Returns the
    latents (mutated in place, as the product does) plus the record.

    The three statements that matter are copied from t18_run.py:2934-2941 with
    the same dtypes and the same operation order, so that with an empty plan
    the output is byte-identical to the product's loop -- which the gate proves
    against arm_base.npz rather than assumes.
    """
    ts = np.asarray(timesteps, np.float64)
    sg = np.asarray(sigmas, np.float64)
    pat = set(plan.pattern)
    # held0 lets a caller that starts at step k > 0 hand in the velocity the
    # product computed at k-1 (a resume, or the prefix shortcut the pricing
    # campaign used).  Without it a skip at step start would be impossible and
    # is recorded as planned_but_computed.
    held = held0           # the last COMPUTED model output, per image
    held_at = held_at0
    calls = 0
    steps = []
    computed_after_plan = []
    for i, t in enumerate(ts):
        if i < start:
            continue
        skip_this = (i in pat) and (held is not None)
        if i in pat and held is None:
            # resume / start>0 with nothing held: the only correct action is to
            # compute.  Recorded, because it makes the run's real pattern differ
            # from the planned one and C1 must show that.
            computed_after_plan.append(int(i))
        if skip_this:
            rec = skip_rec(i, held_at, plan.policy)
            preds = ([np.zeros_like(x) for x in latents]
                     if plan.policy == POLICY_ZERO else held)
        else:
            preds = step_fn(i, t, latents)
            calls += 1
            held = preds
            held_at = i
            rec = {"step": int(i), "skipped": False, "held_from_step": None,
                   "policy": None, "model_calls": 1}
        d = np.float32(sg[i + 1] - sg[i])
        for b in range(len(latents)):
            latents[b] = (latents[b] + d * (-preds[b])).astype(np.float32)
        rec["d_sigma"] = float(d)
        rec["pred_absmax"] = None if rec["skipped"] else float(
            max(np.abs(p).max() for p in preds))
        rec["lat_absmax"] = float(max(np.abs(x).max() for x in latents))
        steps.append(rec)
        if on_step:
            on_step(i, rec, latents)
    return {"latents": latents, "steps": steps, "model_calls": calls,
            "planned": list(plan.pattern),
            "actually_skipped": [s["step"] for s in steps if s["skipped"]],
            "planned_but_computed": computed_after_plan,
            "policy": plan.policy, "sigmas": sg, "timesteps": ts}


# ---------------------------------------------------------------------------
# metrics (the ones quality_gate.py reports)
# ---------------------------------------------------------------------------
def corr(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def rel_rms(ref, test):
    r = np.asarray(ref, np.float64)
    t = np.asarray(test, np.float64)
    d = t - r
    return float(np.sqrt((d * d).sum()) / np.sqrt((r * r).sum()))


def sha12(a):
    return hashlib.sha256(
        np.ascontiguousarray(a, np.float32).tobytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# the product patch, as text
# ---------------------------------------------------------------------------
PRODUCT_PATCH = r"""
--- a/imaggers/image/t18_run.py
+++ b/imaggers/image/t18_run.py
@@ run(args), before the loop
+    # T23 SKIP7: the skip set is a function of the SCHEDULE, so it is decided
+    # here, before any latent exists, and never consulted again.
+    import stepskip as SS
+    _skip = SS.resolve(os.environ.get(SS.ENV, ""), sigmas, len(timesteps))
+    _held, _held_at = None, None
+    if len(_skip):
+        print("[skip] %s -> pattern %s (policy %s): %s"
+              % (_skip.mode, list(_skip.pattern), _skip.policy,
+                 _skip.assertion.get("why", "")), flush=True)
+
@@ the loop, for i, t in enumerate(timesteps):
         if i < start:
             continue
-        preds, rec = one_step(i, t, lat, args.host_ref and i == 0)
+        # --- T23 SKIP7: one guard -------------------------------------------
+        if i in _skip and _held is not None:
+            preds, _held_from = _held, _held_at
+            rec = {"step": i, "skipped": True, "held_from_step": _held_from,
+                   "pred_absmax": None, "wall_s": 0.0, "tim": {}}
+        else:
+            preds, rec = one_step(i, t, lat, args.host_ref and i == 0)
+            _held, _held_at = preds, i
+            rec["skipped"] = False
         _t = time.perf_counter()
         d = np.float32(sigmas[i + 1] - sigmas[i])
         for b in range(B):
             lat[b] = (lat[b] + d * (-preds[b])).astype(np.float32)

@@ the job record (webui/runner.py, where the job dict is filled)
+        "skip": {
+            "mode": _skip.mode, "policy": _skip.policy,
+            "pattern": list(_skip.pattern),
+            "planned": list(_skip.pattern),
+            "actually_skipped": [s["step"] for s in steps_rec
+                                 if s.get("skipped")],
+            "held_from": {s["step"]: s["held_from_step"]
+                          for s in steps_rec if s.get("skipped")},
+            "rule": _skip.table.get("rule"),
+            "assertion": _skip.assertion,
+            "model_calls": total_calls, "n_steps": len(timesteps),
+        },
"""


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="the skip rule, printed")
    ap.add_argument("--sigmas", default="")
    ap.add_argument("--spec", default=None)
    a = ap.parse_args(argv)
    sg = (np.array([float(v) for v in a.sigmas.split(",")], np.float64)
          if a.sigmas else SIGMAS_8)
    pat, table = pattern_from_schedule(sg)
    print(json.dumps(table, indent=1))
    print("rule selects:", pat)
    print("assert:", json.dumps(assert_product_pattern(sg, pat, table), indent=1))
    p = resolve(a.spec, sg, len(sg) - 1)
    print("resolve(%r) ->" % (a.spec,), p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
