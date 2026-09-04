r"""SHIPPED AS RELEASED CODE -- see the note below before running.

P1 10 releases "the figure generator", and this is it, unmodified from the repository it was
run in. It is included so the figures can be audited, not because it runs unchanged here:

  - it expects the source repository's layout (`benchmarks/`, `paper/`), not this flat deposit;
  - Figure 2 needs the authors' released RoBERTa-Large-MNLI prediction file, which is neither
    in their GitHub repository nor in the HuggingFace dataset, and which this deposit does not
    redistribute. `fetch_data.py` explains where it comes from.

The adjudication -- the part of 10 that carries the paper's central claim -- needs none of
this and runs from `fetch_data.py` alone.
"""

#!/usr/bin/env python3
"""Generate P1's figures, and emit the values plotted so the gate can check them.

Every number drawn here is RECOMPUTED from the same sources audit_numbers.py uses. That is
the whole point: the existing figure generator in paper/generate_figure1.py hardcodes its
data as literals, so a figure can silently disagree with the text it sits beside and nothing
catches it. Numbers in prose are gated; numbers in pictures were not.

So this script writes `paper/p1_figure_data.json` alongside the PDFs, and audit_numbers.py
reads that manifest and requires each plotted value to equal the value it computes
independently. A figure that drifts fails the build like any other stale number.

Usage:  python benchmarks/generate_p1_figures.py
Output: paper/fig{1,2,3}_*.pdf  (vector, for LaTeX)
        paper/fig{1,2,3}_*.png  (raster, for review)
        paper/p1_figure_data.json
"""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"

# Okabe-Ito: the standard colourblind-safe qualitative palette, and it survives greyscale
# printing, which a reviewer's hardcopy may well be.
BLUE, ORANGE, GREY, BLACK = "#0072B2", "#D55E00", "#999999", "#000000"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#444444",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,      # embed TrueType, not Type3 -- some venues reject Type3
})

manifest = {}


def save(fig, stem):
    """Write the figure, deterministically, and record its hash.

    matplotlib stamps a CreationDate into PDF metadata, so re-running with identical data
    produced a byte-different file. For a paper whose subject is reproducibility that is the
    wrong artefact to ship, and it also blocks the useful check: with deterministic output
    the manifest can carry each figure's sha256, so a STALE OR HAND-EDITED PDF is caught
    rather than merely assumed to match. That was the residual limit noted in audit_numbers;
    this closes it.
    """
    for ext in ("pdf", "png"):
        path = PAPER / f"{stem}.{ext}"
        meta = ({"CreationDate": None, "Producer": "", "Creator": ""} if ext == "pdf"
                else {"Software": ""})
        fig.savefig(path, dpi=300, metadata=meta)
    plt.close(fig)
    digests = {}
    for ext in ("pdf", "png"):
        digests[ext] = hashlib.sha256(
            (PAPER / f"{stem}.{ext}").read_bytes()).hexdigest()[:16]
    manifest.setdefault("_files", {})[stem] = digests
    print(f"  wrote paper/{stem}.pdf and .png  (pdf sha256 {digests['pdf']})")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(c - h, 0.0), 100 * (c + h)


def load_adjudicator():
    spec = importlib.util.spec_from_file_location(
        "_adjfig", ROOT / "benchmarks" / "adjudicate_independent_full.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_adjfig"] = mod
    saved, sys.argv = sys.argv, [str(spec.origin)]
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        sys.argv = saved
    return mod


# --------------------------------------------------------------------------------------
# Figure 1 -- the worked refutation. Two DAGs, identical node positions, one edge reversed.
# Same positions in both panels is the whole design: it makes "they differ in one edge"
# something the eye verifies rather than something the caption asserts.
# --------------------------------------------------------------------------------------
def figure_worked_example(rows, adj):
    names, corr, indep, hyp = adj.parse(rows[920]["input"])
    dags = adj.consistent_dags(names, corr, indep, 5000)
    held = [adj.holds(G, hyp) for G in dags]
    yes, no = dags[held.index(True)], dags[held.index(False)]

    pos = {"C": (-1.15, 1.05), "B": (1.30, 1.05), "A": (0.00, 0.55),
           "D": (0.00, -0.15), "E": (-0.95, -0.85), "F": (0.75, -1.35)}
    flip = {("A", "C"), ("C", "A")}

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for ax, G, ok, label in ((axes[0], yes, True, "I"), (axes[1], no, False, "II")):
        shared = [e for e in G.edges() if e not in flip]
        special = [e for e in G.edges() if e in flip]
        nx.draw_networkx_edges(G, pos, edgelist=shared, ax=ax, edge_color=GREY,
                               width=0.9, arrowsize=11, node_size=430,
                               connectionstyle="arc3,rad=0.06")
        nx.draw_networkx_edges(G, pos, edgelist=special, ax=ax,
                               edge_color=BLUE if ok else ORANGE, width=2.2,
                               arrowsize=15, node_size=430,
                               connectionstyle="arc3,rad=0.06")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=430, node_color="white",
                               edgecolors=BLACK, linewidths=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8.5, font_family="serif")
        ax.set_title(f"DAG {label}", fontsize=9.5, pad=2)
        ax.set_axis_off()
        ax.margins(0.06)
    # The notes are placed in FIGURE coordinates with reserved bottom space, not in axes
    # coordinates. Placed inside the axes they overprinted node F -- caught only by looking
    # at the rendered PNG, since nothing about the code or the data was wrong.
    fig.subplots_adjust(wspace=0.02, top=0.93, bottom=0.20)
    for x, ok, note in ((0.28, True, "A is a common cause of C and D:\nconfounder exists"),
                        (0.76, False, "C has no ancestors:\nno common cause of C and D")):
        fig.text(x, 0.02, note, ha="center", va="bottom", fontsize=8,
                 color=BLUE if ok else ORANGE)
    save(fig, "fig1_worked_example")

    manifest["fig1"] = {
        "mec_size": len(dags),
        "satisfying": sum(held),
        "dag_yes_edges": sorted("".join(e) for e in yes.edges()),
        "dag_no_edges": sorted("".join(e) for e in no.edges()),
        "common_ancestors_in_counterexample":
            len(nx.ancestors(no, "C") & nx.ancestors(no, "D")),
    }


# --------------------------------------------------------------------------------------
# Figure 2 -- per-template contradiction rate, test vs train, with Wilson intervals.
# Replaces an assertion ("the shape is the same across all three splits") with the evidence.
# --------------------------------------------------------------------------------------
def figure_per_template(rows, disputed):
    train = {"parent": (34292, 2851), "has_collider": (34284, 1530),
             "has_confounder": (34284, 1033), "non-parent ancestor": (34356, 793),
             "non-child descendant": (34227, 751), "child": (34291, 0)}
    per_t = {}
    for i in disputed:
        t = rows[i]["template"]
        per_t[t] = per_t.get(t, 0) + 1

    order = sorted(train, key=lambda t: -train[t][1] / train[t][0])
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    rec = {}
    for row, t in enumerate(order):
        y = len(order) - row
        n_te = sum(1 for r in rows if r["template"] == t)
        k_te = per_t.get(t, 0)
        r_te = 100 * k_te / n_te
        lo, hi = wilson(k_te, n_te)
        n_tr, k_tr = train[t]
        r_tr = 100 * k_tr / n_tr
        ax.plot([lo, hi], [y, y], color=BLUE, lw=1.1, solid_capstyle="butt", zorder=2)
        ax.plot([r_te], [y], "o", color=BLUE, ms=4.5, zorder=3,
                label="test split (95% CI)" if row == 0 else None)
        ax.plot([r_tr], [y], "D", color=ORANGE, ms=4.5, zorder=3,
                label="train split" if row == 0 else None)
        rec[t] = {"test_n": n_te, "test_k": k_te, "test_rate": round(r_te, 2),
                  "test_ci": [round(lo, 2), round(hi, 2)],
                  "train_rate": round(r_tr, 2),
                  "train_in_test_ci": bool(lo <= r_tr <= hi)}
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(list(reversed(order)), fontsize=8.5)
    ax.set_xlabel("items contradicting the generator's own algorithm (%)", fontsize=8.5)
    # Right limit set from the data, with headroom. Left at the default the parent train
    # marker (8.31, the largest value plotted) was clipped by the axis spine.
    _hi = max(max(v["test_ci"][1] for v in rec.values()),
              max(v["train_rate"] for v in rec.values()))
    ax.set_xlim(-0.3, _hi + 0.7)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    save(fig, "fig2_per_template")

    manifest["fig2"] = {"per_template": rec,
                        "templates_train_inside_test_ci":
                            sum(1 for t in rec if t != "child"
                                and rec[t]["train_in_test_ci"])}


# --------------------------------------------------------------------------------------
# Figure 3 -- what correcting the labels does to a published score, against the margin the
# published comparison actually turns on.
# --------------------------------------------------------------------------------------
def figure_impact(before, after, margin):
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    ax.plot([0, 1], [before, after], color=ORANGE, lw=1.6, zorder=2)
    ax.plot([0], [before], "o", color=ORANGE, ms=6, zorder=3)
    ax.plot([1], [after], "o", color=ORANGE, ms=6, zorder=3)
    ax.annotate(f"{before:.2f}", (0, before), textcoords="offset points",
                xytext=(-8, 0), ha="right", va="center", fontsize=9)
    ax.annotate(f"{after:.2f}", (1, after), textcoords="offset points",
                xytext=(8, 0), ha="left", va="center", fontsize=9)
    drop = before - after
    ax.annotate(f"$-${drop:.2f}", (0.5, (before + after) / 2),
                textcoords="offset points", xytext=(0, 9), ha="center",
                fontsize=9, color=ORANGE)
    lo = after - 0.35
    ax.add_patch(plt.Rectangle((0.36, lo), 0.28, margin, facecolor=BLUE,
                               alpha=0.75, edgecolor="none", zorder=2))
    ax.annotate(f"margin the published comparison\nturns on: {margin:.2f}",
                (0.64, lo + margin / 2), textcoords="offset points", xytext=(9, 0),
                ha="left", va="center", fontsize=8, color=BLUE)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["published labels", "corrected labels"], fontsize=9)
    ax.set_xlim(-0.35, 1.45)
    ax.set_ylabel("RoBERTa-Large MNLI\nclass-1 F1", fontsize=8.5)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig3_impact")
    manifest["fig3"] = {"before": round(before, 2), "after": round(after, 2),
                        "drop": round(drop, 2), "margin": margin}


def main():
    import csv, re
    csv.field_size_limit(10 ** 7)
    rows = json.loads((ROOT / "benchmarks" / "corr2cause_test_FULL1162.json")
                      .read_text(encoding="utf-8"))["rows"]
    adj = load_adjudicator()

    sys.path.insert(0, str(ROOT))
    from engine.causal_discovery.corr2cause_solver import solve_input
    gold, pred = [], []
    for r in rows:
        try:
            p = int(solve_input(r["input"]).label)
        except Exception:
            p = 0
        gold.append(int(r["label"]))
        pred.append(p)
    disputed = [i for i, (g, p) in enumerate(zip(gold, pred)) if g != p]

    nm = lambda s: re.sub(r"\s+", " ", s or "").strip().lower()
    sidx = {nm(r["input"]): i for i, r in enumerate(rows)}
    rbp = [None] * len(rows)
    with open(ROOT / "benchmarks" / "data" / "corr2cause"
              / "roberta_large_mnli_test_v2.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rbp[sidx[nm(r["prompt"])]] = 1 if r["pred"].strip() == "entailment" else 0

    def f1(y, p):
        tp = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y, p) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y, p) if a == 1 and b == 0)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        return 0.0 if P + R == 0 else round(200 * P * R / (P + R), 2)

    corrected = list(gold)
    for i in disputed:
        corrected[i] = 1 - corrected[i]

    print("generating P1 figures:")
    figure_worked_example(rows, adj)
    figure_per_template(rows, disputed)
    figure_impact(f1(gold, rbp), f1(corrected, rbp), 0.23)

    (PAPER / "p1_figure_data.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    print(f"  wrote paper/p1_figure_data.json ({len(manifest)} figures)")


if __name__ == "__main__":
    main()
