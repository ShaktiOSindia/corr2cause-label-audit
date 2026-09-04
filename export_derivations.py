r"""Export the per-item derivation for every disputed item, as a machine-readable file.

WHY THIS EXISTS. §10 says "the adjudication code, the per-item derivations for every disputed
item, and the figure generator are released". The code and the generator exist; the
derivations did not, as a file. §3.8 prints them as a table, and the reproducibility checklist
answers item 8 "Partial -- printed in §3.8, not shipped as a file", which is the weakest form
of availability and the paper says so.

This produces the file. Everything in it comes from `adjudicate_independent_full.py`'s own
`parse`, `consistent_dags` and `holds` -- imported, not restated -- so the shipped derivations
cannot drift from the instrument whose output the paper reports.

WHAT A DERIVATION IS HERE. For one disputed item: the premise as parsed, the equivalence class
rebuilt by enumerating acyclic orientations of the independence-derived skeleton, and for each
member of that class whether the hypothesis holds in it. The label follows by the necessary
reading -- entailed iff it holds in EVERY member -- so a reader can recompute the verdict from
the file without running anything, and can see exactly which graph refutes a gold 1.

Writes derivations_18_disputed.json.

Usage:
    python export_derivations.py
"""
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = os.path.join(ROOT, "derivations_18_disputed.json")

# --- the adjudicator's own semantics, imported unchanged ---------------------------------
_src = io.open("adjudicate_independent_full.py", encoding="utf-8").read()
_head = _src.split("if args.control:")[0]
_head = _head.replace("args = ap.parse_args()", "class A:\n    max_orient=400000\nargs=A()")
_head = _head.replace("Path(__file__).parent", 'Path(r"%s")' % ROOT)
_ns = {}
exec(_head, _ns)
rows = _ns["rows"]
parse, consistent_dags, holds = _ns["parse"], _ns["consistent_dags"], _ns["holds"]

# The disputed set, taken from the adjudicator's own target list rather than retyped: if that
# list ever changes, this file changes with it instead of silently describing a stale set.
#
# Anchored on the NUMERIC list. A plain split on "targets = [" finds the control branch first
# -- `targets = [i for i, r in enumerate(rows) ...]` -- and produced 'i for i' as an index.
import re as _re
_m = _re.search(r"targets\s*=\s*\[\s*(\d[\d,\s]*)\]", _src)
assert _m, "the disputed-item list is no longer a literal in adjudicate_independent_full.py"
DISPUTED = [int(t) for t in _m.group(1).split(",") if t.strip()]
assert len(DISPUTED) == len(set(DISPUTED)) and DISPUTED, "malformed disputed list"


def derive(idx):
    row = rows[idx]
    names, corr, indep, hyp = parse(row["input"])
    dags = consistent_dags(names, corr, indep, 400000)
    members = []
    for g in dags:
        members.append({
            "edges": sorted("%s->%s" % (a, b) for a, b in g.edges()),
            "hypothesis_holds": bool(holds(g, hyp)),
        })
    derived = 1 if all(m["hypothesis_holds"] for m in members) else 0
    gold = int(row["label"])
    return {
        "index": idx,
        "template": row["template"],
        "num_variables": row["num_variables"],
        "premise_variables": names,
        "stated_correlations": sorted("~".join(sorted(p)) for p in corr),
        "stated_independencies": [
            {"x": x, "y": y, "given": sorted(z)} for x, y, z in indep],
        "hypothesis": hyp,
        "equivalence_class_size": len(members),
        "members": members,
        "holds_in": sum(1 for m in members if m["hypothesis_holds"]),
        "derived_label": derived,
        "gold_label": gold,
        "gold_is_wrong": derived != gold,
        "direction": ("gold says not entailed, derivation says entailed"
                      if derived == 1 and gold == 0 else
                      "gold says entailed, derivation says not entailed"
                      if derived == 0 and gold == 1 else "agrees"),
    }


def main():
    items = [derive(i) for i in DISPUTED]
    wrong = [d for d in items if d["gold_is_wrong"]]
    doc = {
        "_what": "Per-item derivations for every disputed Corr2Cause test item (P1 §3.8).",
        "_method": ("Premise re-parsed; equivalence class rebuilt by enumerating acyclic "
                    "orientations of the independence-derived skeleton; a relation is "
                    "entailed iff it holds in EVERY member. Shares no equivalence-class code "
                    "with the solver under test."),
        "_reproduce": "python export_derivations.py",
        "_source": "benchmarks/adjudicate_independent_full.py (parse, consistent_dags, holds)",
        "_data": ("causalnlp/corr2cause test split, revision "
                  "42ba12c769e11ff6427c9f52d7db58e3f9bf3e53, 1162 rows"),
        "_licence": "MIT, matching the Corr2Cause repository the underlying data comes from.",
        "disputed_items": len(items),
        "gold_wrong": len(wrong),
        "by_direction": {
            "solver says entailed, gold says not":
                sum(1 for d in items if d["derived_label"] == 1 and d["gold_label"] == 0),
            "solver says not entailed, gold says entailed":
                sum(1 for d in items if d["derived_label"] == 0 and d["gold_label"] == 1),
        },
        "by_template": {},
        "items": items,
    }
    for d in items:
        doc["by_template"][d["template"]] = doc["by_template"].get(d["template"], 0) + 1

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(doc, indent=2) + "\n")

    print("disputed items derived : %d" % len(items))
    print("gold wrong             : %d" % len(wrong))
    print("by direction           : %s" % doc["by_direction"])
    print("by template            : %s" % doc["by_template"])
    print("class sizes            : min %d, max %d"
          % (min(d["equivalence_class_size"] for d in items),
             max(d["equivalence_class_size"] for d in items)))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
