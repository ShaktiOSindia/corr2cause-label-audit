r"""Fetch the Corr2Cause data this deposit needs, from its own sources, at a pinned revision.

WHY A FETCHER AND NOT A COPY. This deposit deliberately redistributes none of Jin et al.'s
data. Their GitHub repository is MIT licensed, but the HuggingFace dataset card that people
actually download declares no licence at all -- its body is the word "TODO" -- so a
redistributor cannot determine the terms from the artefact itself. Fetching at a pinned
revision is also better provenance than a copy: it proves which bytes the audit was run
against, and it fails loudly if upstream moves.

WHAT IT FETCHES

  test split   causalnlp/corr2cause, revision 42ba12c769e11ff6427c9f52d7db58e3f9bf3e53,
               1162 test rows -> corr2cause_test_FULL1162.json
               (that revision has been unmodified since January 2024)

WHAT IT DOES NOT FETCH, AND WHY. `generate_p1_figures.py` needs the authors' released
RoBERTa-Large-MNLI prediction file. It is not in the GitHub repository -- whose `data/` holds
only a README pointing at HuggingFace -- and it is not among the twelve files in the
HuggingFace dataset either. It comes from a separate archive of the authors' released
predictions (the one carrying `data_v1_depreciated/` and `data_v2/`, which P1 §3.1 is about).
Since there is no canonical URL to pin, nothing is fetched and nothing is redistributed: the
generator is shipped, its input is not, and this note says so rather than the script failing
at an unexplained 404. The adjudication -- the part §10 is chiefly about -- needs none of it.

Usage:
    pip install -r requirements.txt
    python fetch_data.py
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REV = "42ba12c769e11ff6427c9f52d7db58e3f9bf3e53"
SPLIT = os.path.join(HERE, "corr2cause_test_FULL1162.json")


# What the audit was run against. If either of these changes, the paper's figures describe
# different bytes than the ones you just downloaded, and you should say so rather than
# quietly re-running.
EXPECT_ROWS = 1162


def fetch_split():
    try:
        from datasets import load_dataset
    except ImportError:
        print("FATAL: `pip install datasets` first (see requirements.txt).", file=sys.stderr)
        raise SystemExit(2)
    print("fetching causalnlp/corr2cause test split at revision %s ..." % REV[:8])
    ds = load_dataset("causalnlp/corr2cause", split="test", revision=REV)
    rows = [{"input": r["input"], "label": r["label"]} for r in ds]

    # The adjudicator keys on `template` and `num_variables`, which the released split does
    # not carry; both are derivable from the item itself, so they are computed here rather
    # than shipped as opinions about somebody else's data.
    import re
    for r in rows:
        prem, _, hyp = r["input"].partition("Hypothesis:")
        m = re.search(r"closed system of (\d+) variables?", prem)
        r["num_variables"] = int(m.group(1)) if m else 0
        r["template"] = classify(hyp)
    if len(rows) != EXPECT_ROWS:
        print("WARNING: got %d rows, the audit was run against %d. Upstream may have moved."
              % (len(rows), EXPECT_ROWS), file=sys.stderr)
    # newline="\n" so the file is byte-identical on every platform. Without it Windows
    # translates to CRLF and the sha256 verify_split.py prints -- offered as a provenance
    # fingerprint -- differs between machines for the same upstream bytes. Caught by running
    # this deposit in a Linux container: the verdicts matched, the hash did not.
    io.open(SPLIT, "w", encoding="utf-8", newline="\n").write(
        json.dumps({"revision": REV, "rows": rows}, indent=1))
    print("  wrote %s  (%d rows)" % (os.path.basename(SPLIT), len(rows)))


def classify(hyp):
    """The template a hypothesis belongs to.

    DERIVED FROM THE DATA, NOT GUESSED, AND CHECKED. A first version keyed on phrasing alone
    and misclassified 387 of the 1162 rows, with two templates exactly backwards. The rule
    below was recovered by bucketing every hypothesis in the released split by phrasing AND
    by whether the first-named variable sorts before the second, then reading off the mapping:
    every bucket resolved to exactly one template, with no unmatched hypothesis and no
    ambiguity. It reproduces all 1162 templates exactly -- `verify_split.py` re-checks that.

        directly causes,  A<B  -> parent        (194)
        directly causes,  A>B  -> child         (194)
        cause ... not a direct one   -> non-child descendant   (193, both orders)
        causes something else which  -> non-parent ancestor     (195, both orders)
        collider (common effect)     -> has_collider            (193)
        confounder (common cause)    -> has_confounder          (193)

    The direction split on "directly causes" is the same asymmetry P1 §3.7 reports as a
    shortcut: an item naming the alphabetically later variable as the cause is always negative.
    """
    import re as _r
    h = hyp.strip()
    m = _r.search(r"(\w+) directly causes (\w+)", h)
    if m:
        return "parent" if m.group(1) < m.group(2) else "child"
    if _r.search(r"(\w+) is a cause for (\w+), but not a direct one", h):
        return "non-child descendant"
    if "causes something else which causes" in h:
        return "non-parent ancestor"
    if "collider" in h:
        return "has_collider"
    if "confounder" in h:
        return "has_confounder"
    raise ValueError("unrecognised hypothesis template: %r" % h[:80])



if __name__ == "__main__":
    fetch_split()
    print("\nNow run:  python adjudicate_independent_full.py")
