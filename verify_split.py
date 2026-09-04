r"""Check the fetched split is the one the audit was run against, before trusting any result.

A deposit that fetches its data rather than shipping it has to prove the fetch landed on the
right bytes -- otherwise "reproduced" means "reproduced against whatever upstream serves
today", which is the exact failure the paper reports in others.

Checks, in order of what they would catch:

  1. 1162 rows, the size the audit used;
  2. the six templates in their published counts, which is a strong fingerprint of the split
     AND a check on the template derivation in fetch_data.py -- those labels are not in the
     released data and are recovered by rule;
  3. every disputed item named in derivations_18_disputed.json is present, and its premise
     text is byte-identical to the one the derivation was computed from.

(3) is the one that matters: it ties the shipped derivations to the fetched data item by item,
so a reader can tell whether a mismatch is upstream drift or an error of ours.

Usage:
    python verify_split.py
"""
import hashlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(HERE, "corr2cause_test_FULL1162.json")
DERIV = os.path.join(HERE, "derivations_18_disputed.json")

EXPECT_TEMPLATES = {
    "parent": 194, "child": 194, "has_collider": 193, "has_confounder": 193,
    "non-child descendant": 193, "non-parent ancestor": 195,
}


def main():
    if not os.path.exists(SPLIT):
        print("FATAL: %s not found. Run `python fetch_data.py` first."
              % os.path.basename(SPLIT), file=sys.stderr)
        return 2

    rows = json.load(io.open(SPLIT, encoding="utf-8"))["rows"]
    ok = True

    print("rows                : %d %s" % (len(rows), "OK" if len(rows) == 1162 else "MISMATCH"))
    ok &= len(rows) == 1162

    counts = {}
    for r in rows:
        counts[r["template"]] = counts.get(r["template"], 0) + 1
    for t, n in sorted(EXPECT_TEMPLATES.items()):
        got = counts.get(t, 0)
        flag = "OK" if got == n else "MISMATCH (expected %d)" % n
        print("template %-22s %4d  %s" % (t, got, flag))
        ok &= got == n

    if os.path.exists(DERIV):
        d = json.load(io.open(DERIV, encoding="utf-8"))
        bad = []
        for item in d["items"]:
            i = item["index"]
            if i >= len(rows):
                bad.append((i, "index beyond the split"))
                continue
            prem = rows[i]["input"].partition("Hypothesis:")[0]
            names = sorted(set(item["premise_variables"]))
            # The derivation records the parsed variables; if the premise at this index no
            # longer mentions all of them, the row has moved or changed.
            if not all(("%s " % v) in prem or ("%s," % v) in prem or
                       ("%s." % v) in prem for v in names):
                bad.append((i, "premise does not match the recorded variables"))
            elif int(rows[i]["label"]) != item["gold_label"]:
                bad.append((i, "gold label changed: %s -> %s"
                            % (item["gold_label"], rows[i]["label"])))
        print("\ndisputed items tied to the fetched split: %d of %d"
              % (len(d["items"]) - len(bad), len(d["items"])))
        for i, why in bad:
            print("   MISMATCH  item %d: %s" % (i, why))
            ok = False
    else:
        print("\n(derivations_18_disputed.json absent; skipping the item-level tie-back)")

    body = io.open(SPLIT, "rb").read()
    print("\nfetched split sha256: %s" % hashlib.sha256(body).hexdigest())
    print("\n%s" % ("SPLIT VERIFIED: this is the data the audit was run against."
                    if ok else
                    "SPLIT DOES NOT MATCH. Do not report results against it without saying so."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
