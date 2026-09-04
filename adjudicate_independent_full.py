r"""Third-implementation adjudication of every Canon/corr2cause disagreement.

Shares NO code with engine/causal_discovery/corr2cause_solver.py. Premise text is re-parsed
from scratch; the MEC is rebuilt by brute-force enumeration of acyclic orientations; d-separation
is networkx's. VERIFICATION s2 did exactly this for item 115 only, because the full split was
unreachable (HF 403 in that sandbox). HF is reachable here, so this runs on all of them.

Method, deliberately strict:
  skeleton  : X-Y adjacent iff NO stated statement separates X from Y
  candidates: every acyclic orientation of that skeleton
  consistent: reproduces EVERY stated fact (correlations d-connected, stated independencies
              d-separated) AND introduces no independence the premise did not state
  verdict   : relation necessary iff it holds in ALL consistent DAGs -> label 1, else 0

RESULT (2026-08-25): all 18 Canon/gold disagreements adjudicate as GOLD WRONG.

HOW THIS FILE WAS WRONG TWICE, because the corrections are the method:

 1. The premise splitter used a `(?<=[a-z0-9])\.` lookbehind. Every corr2cause sentence ends in
    a variable NAME (uppercase), so nothing split, no independencies were parsed, every skeleton
    was complete and MEC sizes came out as exactly 4!=24, 5!=120, 6!=720. It returned confident
    verdicts that matched gold on 4/4 negatives and missed 3/3 positives -- it LOOKED validated
    because it had defaulted to the majority class.

 2. Three hypothesis semantics were wrong, and the original control could not see any of them
    because that control was built from n=4 items which are ALL gold=0:
      - "a cause, but not a direct one" is an INDIRECT PATH, not "no direct edge" (idx 285)
      - "collider (common effect)" is a common DESCENDANT, not a common CHILD (idx 574, 934)
      - "confounder (common cause)" is a common ANCESTOR, not a common PARENT (idx 970)
    Defect (2a) produced a FALSE accusation against the repo solver on idx 869, since retracted.

CONTROL DISCIPLINE this file now requires (three parts, one per defect class found):
  - majority-class control  -> catches OVER-firing   (`--control <template>`, n=4 items, all gold=0)
  - minority-class control  -> catches UNDER-firing  (gold=1 items; 11/11, 22/22, 64/64, 65/65)
  - convention test on the full set -> catches a wrong semantics BOTH class controls pass.
    Measured, not assumed: necessary("all DAGs") vs possible("any DAG") scores 97.9% vs 46.6%
    on has_confounder and 91.2% vs 67.9% on has_collider, so necessary-semantics is confirmed.

The claim does NOT rest on "two implementations agree" -- they produce identical MECs on
1162/1162 items, which shows neither has a bug but not that both have the right definition.
It rests on each MEC being verified DIRECTLY against its premise: for all 18, every DAG's full
d-separation profile was recomputed from scratch and matched the stated CI profile exactly,
and parsing is provably complete (sentences == correlations + independencies).

All 18 survive BOTH readings of the two ambiguous templates. See VERIFICATION_2026-08-25.md
Parts VII and IX.
"""
import sys, json, re, itertools, argparse
from pathlib import Path
import networkx as nx

ap = argparse.ArgumentParser(); ap.add_argument("--max-orient", type=int, default=200000)
ap.add_argument("--only", type=int, default=None)
ap.add_argument("--control", metavar="TEMPLATE", default=None,
                help="instrument check: adjudicate ALL n=4 items of this template and compare to gold")
args = ap.parse_args()

rows = json.load(open(Path(__file__).parent / "corr2cause_test_FULL1162.json"))["rows"]

def parse(text):
    prem, hyp = text.split("Hypothesis:")
    m = re.search(r"closed system of (\d+) variables?, (.+?)\. All the", prem)
    names = [v.strip() for v in re.split(r",| and ", m.group(2)) if v.strip()]
    corr, indep = set(), []
    for s in re.split(r"(?<=[A-Za-z0-9])\.\s+", prem.split("as follows:")[1]):
        s = s.strip().rstrip(".").replace("However, ", "")
        g = re.match(r"^(\w+) correlates with (\w+)$", s)
        if g: corr.add(frozenset(g.groups())); continue
        g = re.match(r"^(\w+) and (\w+) are independent given (.+)$", s) \
            or re.match(r"^(\w+) is independent of (\w+) given (.+)$", s)
        if g:
            z = frozenset(v.strip() for v in re.split(r",| and ", g.group(3)) if v.strip())
            indep.append((g.group(1), g.group(2), z)); continue
        g = re.match(r"^(\w+) is independent of (\w+)$", s)
        if g: indep.append((g.group(1), g.group(2), frozenset())); continue
    return names, corr, indep, hyp.strip()

def consistent_dags(names, corr, indep, cap):
    sep = {}
    for x, y, z in indep: sep.setdefault(frozenset((x, y)), []).append(z)
    skel = [tuple(sorted(p)) for p in itertools.combinations(names, 2)
            if frozenset(p) not in sep]
    if len(skel) > 22: return None
    out = []
    for bits in itertools.product((0, 1), repeat=len(skel)):
        if len(out) > cap: return None
        G = nx.DiGraph(); G.add_nodes_from(names)
        G.add_edges_from((b, a) if f else (a, b) for (a, b), f in zip(skel, bits))
        if not nx.is_directed_acyclic_graph(G): continue
        ok = True
        for x, y in itertools.combinations(names, 2):
            key = frozenset((x, y))
            for k in range(len(names) - 1):
                for Z in itertools.combinations([v for v in names if v not in (x, y)], k):
                    dsep = nx.is_d_separator(G, {x}, {y}, set(Z))
                    stated = key in sep and frozenset(Z) in sep[key]
                    if dsep and not stated: ok = False; break
                    if stated and not dsep: ok = False; break
                if not ok: break
            if not ok: break
        if ok: out.append(G)
    return out

def holds(G, hyp):
    ns = list(G.nodes())
    m = re.search(r"collider \(i\.e\., common effect\) of (\w+) and (\w+)", hyp)
    if m:
        # "common effect" = common DESCENDANT, not necessarily a common CHILD.
        # idx 574/934 are gold=1 with no common child in any DAG of the MEC but a
        # common descendant in every one. Requiring a direct child scored 55/64 on
        # gold=1 items. Caught 2026-08-25 by the positive-class control.
        a, b = m.groups()
        return any(c not in (a, b) and nx.has_path(G, a, c) and nx.has_path(G, b, c)
                   for c in ns)
    m = re.search(r"confounder \(i\.e\., common cause\) of (\w+) and (\w+)", hyp)
    if m:
        # "common cause" = common ANCESTOR, not necessarily a direct common PARENT.
        # idx 970 is gold=1 on a single-DAG MEC whose only common causes of E and F
        # are ancestors (A,B,C,D), with no common parent at all. Requiring a direct
        # parent scored 14/18 on gold=1 items AND produced three FALSE "gold wrong"
        # accusations (920/983/984). Caught 2026-08-25 by the positive-class control.
        a, b = m.groups()
        return any(c not in (a, b) and nx.has_path(G, c, a) and nx.has_path(G, c, b)
                   for c in ns)
    m = re.search(r"(\w+) directly (?:affects|causes) (\w+)", hyp)
    if m: return G.has_edge(*m.groups())
    m = re.search(r"(\w+) is a cause for (\w+), but not a direct one", hyp)
    if m:
        # "a cause, but not a direct one" = an INDIRECT path exists (a -> ... -> b
        # through >=1 mediator). It does NOT require the absence of a direct edge:
        # corr2cause idx 285 is gold=1 on a single-DAG MEC where A->D exists AND
        # A->C->D exists. Requiring `not has_edge` scored 5/11 on gold=1 items of
        # this template and produced a FALSE accusation against the repo solver on
        # idx 869. Caught 2026-08-25 by controlling on the POSITIVE class; the
        # original all-negative control could not see it.
        a, b = m.groups()
        if a == b or a not in G or b not in G:
            return False
        return any(a != c != b and G.has_edge(a, c) and nx.has_path(G, c, b)
                   for c in G.nodes())
    m = re.search(r"(\w+) causes something else which causes (\w+)", hyp)
    if m:
        a, b = m.groups()
        return any(a != c != b and G.has_edge(a, c) and nx.has_path(G, c, b) for c in G.nodes())
    m = re.search(r"(\w+) influences (\w+) through some mediator", hyp)
    if m:
        a, b = m.groups()
        return nx.has_path(G, a, b) and not G.has_edge(a, b) if a in G and b in G else False
    m = re.search(r"(\w+) affects (\w+)", hyp)
    if m:
        a, b = m.groups()
        return a in G and b in G and nx.has_path(G, a, b) and a != b
    return None

if args.control:
    targets = [i for i, r in enumerate(rows)
               if r["num_variables"] == 4 and r["template"] == args.control]
    print("CONTROL MODE: %d n=4 %r items, UNDISPUTED by Canon." % (len(targets), args.control))
    print("It must reproduce gold on these or it is not an instrument.")
    print("")
elif args.only is not None:
    targets = [args.only]
else:
    targets = [115, 321, 458, 553, 631, 705, 731, 779, 788, 794, 810, 869, 950, 1056, 1134, 920, 983, 984]
print(f"{'idx':>5} {'nvar':>4} {'gold':>4} {'MEC':>7}  {'holds':>7}  verdict          template")
agree = disagree = skipped = 0
for i in targets:
    r = rows[i]
    names, corr, indep, hyp = parse(r["input"])
    dags = consistent_dags(names, corr, indep, args.max_orient)
    if dags is None or not dags:
        print(f"{i:>5} {r['num_variables']:>4} {r['label']:>4} {'--':>7}  {'--':>7}  INTRACTABLE/EMPTY  {r['template']}")
        skipped += 1; continue
    hs = [holds(G, hyp) for G in dags]
    if any(h is None for h in hs):
        print(f"{i:>5} {r['num_variables']:>4} {r['label']:>4} {len(dags):>7}  {'--':>7}  HYP-UNPARSED     {r['template']}")
        skipped += 1; continue
    nec = all(hs); lab = 1 if nec else 0
    v = "GOLD WRONG" if lab != int(r["label"]) else "gold ok"
    if lab != int(r["label"]): disagree += 1
    else: agree += 1
    print(f"{i:>5} {r['num_variables']:>4} {r['label']:>4} {len(dags):>7}  {sum(hs):>3}/{len(hs):<3}  {v:16s} {r['template']}")
print(f"\nindependently adjudicated: {agree+disagree}   gold WRONG: {disagree}   gold ok: {agree}   not adjudicated: {skipped}")
