"""NAPKIN REFUTATION — the evidence that y0's estimand is wrong, by three independent methods.

y0 0.2.11 returns  P(Y | do(X)) = P(Y | X)  for the napkin graph (its OWN y0.examples.napkin,
citing Pearl & Mackenzie, Book of Why p.240). That is the naive conditional, and the graph has an
open back-door path X <- Z1 <- Z2 <-> Y. So it cannot be the causal effect. Proven three ways:

  1. EXACT ENUMERATION over a binary SCM in the napkin's own class.
  2. MONTE-CARLO of the same SCM (a different method -- guards against a bug in the enumerator).
  3. A CONTINUOUS napkin with realistic confounding, where the true ATE is 3.0 BY CONSTRUCTION.

(3) is the one that shows the stakes: y0's estimand is off by ~49% there. y0's own data generator
is parameterised with almost no confounding (saturated sigmoid, P(X=1)=0.93), so the bias hides
inside the noise even in their own simulator -- which is why this was never caught.

Run:  python napkin_refutation.py
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.core.identification import identify_effect
from engine.core.evaluation import JointDistribution, evaluate_estimand

NAPKIN = ([("Z2", "Z1"), ("Z1", "X"), ("X", "Y")], [("Z2", "X"), ("Z2", "Y")])

# a binary SCM inside the napkin's class: U0 confounds Z2&X, U1 confounds Z2&Y
SCM = [
    ("U0", [], {(): 0.5}),
    ("U1", [], {(): 0.5}),
    ("Z2", ["U0", "U1"], {(0, 0): 0.1, (0, 1): 0.8, (1, 0): 0.3, (1, 1): 0.6}),
    ("Z1", ["Z2"], {(0,): 0.2, (1,): 0.9}),
    ("X", ["Z1", "U0"], {(0, 0): 0.15, (0, 1): 0.5, (1, 0): 0.6, (1, 1): 0.95}),
    ("Y", ["X", "U1"], {(0, 0): 0.1, (0, 1): 0.45, (1, 0): 0.7, (1, 1): 0.95}),
]


def enumerate_joint(scm, fix=None):
    fix = fix or {}
    names = [v for v, _, _ in scm]
    out = {}
    for combo in product([0, 1], repeat=len(names)):
        a = dict(zip(names, combo))
        if any(a[k] != v for k, v in fix.items()):
            continue
        p = 1.0
        for var, parents, cpt in scm:
            if var in fix:
                continue
            pv = cpt[tuple(a[pp] for pp in parents)]
            p *= pv if a[var] == 1 else (1 - pv)
        out[tuple(combo)] = p
    return names, out


def main():
    res = identify_effect(*NAPKIN, "X", "Y")
    print(f"y0 says identifiable : {res.identifiable}")
    print(f"y0's estimand        : {res.estimand}   <- the naive conditional\n")

    # ---------- 1. exact enumeration ----------
    names, joint = enumerate_joint(SCM, fix={"X": 1})
    truth = sum(p for k, p in joint.items() if k[names.index("Y")] == 1)
    names, joint = enumerate_joint(SCM)
    observed = ["Z1", "Z2", "X", "Y"]                       # U0, U1 are UNOBSERVED
    idx = [names.index(v) for v in observed]
    table = {}
    for k, p in joint.items():
        key = tuple(k[i] for i in idx)
        table[key] = table.get(key, 0.0) + p
    dist = JointDistribution(observed, table)
    y0_value = evaluate_estimand(res.estimand_expr, dist, {"X": 1, "Y": 1})
    print("1. EXACT ENUMERATION")
    print(f"   TRUE P(Y=1|do(X=1)) = {truth:.6f}")
    print(f"   y0's estimand gives = {y0_value:.6f}     error = {abs(truth - y0_value):.6f}")

    # ---------- 1b. POSITIVE CONTROL: the published ratio estimand ----------
    # Added 2026-09-05. Showing y0's estimand is wrong is only half an argument, and the
    # weaker half: a maintainer needs to know what the RIGHT answer is and that it works.
    # This evaluates the estimand issue #372 itself states,
    #
    #     P(Y|do(X)) = sum_{Z2} P(X,Y | Z1,Z2) P(Z2)  /  sum_{Z2} P(X | Z1,Z2) P(Z2)
    #
    # in the same SCM, from the same enumerated joint. It is conditional on Z1, and the
    # thread observes that the answer should not depend on which Z1 is used -- so both are
    # computed and compared, which tests that observation too.
    def ratio_estimand(z1):
        num = den = 0.0
        for z2 in (0, 1):
            p_z2 = sum(p for k, p in table.items() if k[1] == z2)
            cond = sum(p for k, p in table.items() if k[0] == z1 and k[1] == z2)
            if cond == 0:
                continue
            p_xy = sum(p for k, p in table.items()
                       if k[0] == z1 and k[1] == z2 and k[2] == 1 and k[3] == 1) / cond
            p_x = sum(p for k, p in table.items()
                      if k[0] == z1 and k[1] == z2 and k[2] == 1) / cond
            num += p_xy * p_z2
            den += p_x * p_z2
        return num / den

    r0, r1 = ratio_estimand(0), ratio_estimand(1)
    print("\n1b. POSITIVE CONTROL -- the ratio estimand issue #372 states")
    print(f"   ratio, conditioning on Z1=0 = {r0:.6f}     error = {abs(truth - r0):.2e}")
    print(f"   ratio, conditioning on Z1=1 = {r1:.6f}     error = {abs(truth - r1):.2e}")
    print(f"   invariant in Z1            : {abs(r0 - r1) < 1e-12}")
    print("   -> the correct estimand recovers the truth in the SAME model where y0's does"
          " not,\n      so this is not an artefact of the SCM or the enumerator.")

    # ---------- 2. monte-carlo of the same SCM (independent method) ----------
    rng = np.random.default_rng(20260713)
    N = 4_000_000
    u0 = (rng.random(N) < 0.5).astype(int)
    u1 = (rng.random(N) < 0.5).astype(int)
    pick = lambda cpt, *ps: np.array([cpt[t] for t in zip(*ps)])
    z2 = (rng.random(N) < pick(SCM[2][2], u0, u1)).astype(int)
    z1 = (rng.random(N) < np.where(z2 == 1, 0.9, 0.2)).astype(int)
    x = (rng.random(N) < pick(SCM[4][2], z1, u0)).astype(int)
    y = (rng.random(N) < pick(SCM[5][2], x, u1)).astype(int)
    y_do1 = (rng.random(N) < pick(SCM[5][2], np.ones(N, dtype=int), u1)).astype(int)
    mc_truth, mc_naive = y_do1.mean(), y[x == 1].mean()
    se = np.sqrt(y_do1.var() / N + y[x == 1].var() / (x == 1).sum())
    print("\n2. MONTE-CARLO of the same SCM (independent of the enumerator)")
    print(f"   TRUE  = {mc_truth:.6f}   naive P(Y|X=1) = {mc_naive:.6f}")
    print(f"   gap   = {abs(mc_truth - mc_naive):.6f} = {abs(mc_truth - mc_naive) / se:.0f} standard errors")
    print(f"   enumeration confirmed: {abs(mc_truth - truth) < 0.001}")

    # ---------- 3. continuous napkin, confounding NOT throttled ----------
    M = 2_000_000
    u1c, u2c = rng.normal(0, 1, M), rng.normal(0, 1, M)
    z2c = 1.5 * u1c + 1.5 * u2c + rng.normal(0, 1, M)          # Z2 <- U1, U2
    z1c = 1.5 * z2c + rng.normal(0, 1, M)                      # Z1 <- Z2
    xc = (rng.random(M) < 1 / (1 + np.exp(-(0.8 * z1c + 0.8 * u1c)))).astype(int)   # X <- Z1, U1
    yc = 3.0 * xc + 2.0 * u2c + rng.normal(0, 1, M)            # Y <- X, U2  => TRUE ATE = 3.0
    naive_ate = yc[xc == 1].mean() - yc[xc == 0].mean()
    se3 = np.sqrt(yc[xc == 1].var() / (xc == 1).sum() + yc[xc == 0].var() / (xc == 0).sum())
    print("\n3. CONTINUOUS napkin with REALISTIC confounding (true ATE = 3.0 by construction)")
    print(f"   y0's estimand P(Y|X) gives ATE = {naive_ate:.4f} +/- {1.96 * se3:.4f}")
    print(f"   error = {abs(naive_ate - 3.0):.4f}  ({abs(naive_ate - 3.0) / 3.0 * 100:.0f}% of the true effect,"
          f" {abs(naive_ate - 3.0) / se3:.0f} standard errors)")
    print("\nCONCLUSION: y0's napkin estimand is wrong. The error is small in weakly-confounded")
    print("parameterisations (which is why it hid) and ~49% under ordinary confounding.")

    # §5 now quotes these figures at the y0 maintainers, who reached the opposite conclusion
    # in February 2026. A number a paper uses to contradict named researchers had better be
    # recomputable, so it is stored and audit_numbers.py checks it.
    import json as _json
    out = {
        "_what": "The napkin refutation, including the positive control §5 rests on.",
        "_reproduce": "python napkin_refutation.py",
        "_machine_dependent": ["monte_carlo", "continuous"],
        "y0_estimand": str(res.estimand),
        "exact_enumeration": {
            "true_p_y1_do_x1": round(truth, 6),
            "y0_value": round(y0_value, 6),
            "y0_abs_error": round(abs(truth - y0_value), 6),
        },
        "positive_control_ratio_estimand": {
            "value_given_z1_0": round(r0, 6),
            "value_given_z1_1": round(r1, 6),
            "abs_error_vs_truth": max(abs(truth - r0), abs(truth - r1)),
            "invariant_in_z1": bool(abs(r0 - r1) < 1e-12),
        },
        "continuous": {
            "true_ate_by_construction": 3.0,
            "y0_estimand_ate": round(float(naive_ate), 4),
            "abs_error": round(abs(float(naive_ate) - 3.0), 4),
            "percent_of_true_effect": round(abs(float(naive_ate) - 3.0) / 3.0 * 100),
            "standard_errors": round(abs(float(naive_ate) - 3.0) / float(se3)),
        },
    }
    path = Path(__file__).resolve().parent / "napkin_refutation.json"
    path.write_text(_json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
