# Corr2Cause label audit — adjudication code and per-item derivations

Supporting artefact for *Auditing causal-inference artefacts by independent re-derivation:
label errors in Corr2Cause and unsound estimands in y0*.

This deposit contains what the paper's §10 releases: **the adjudication code, the per-item
derivations for every disputed item, and the figure generator.** It is deliberately small.
It is not the research repository, and it redistributes no third-party data.

## The claim you can check here

**18 of the 1162 gold labels in the Corr2Cause test split are wrong, in both directions** —
15 where the derivation says entailed and the gold says not, 3 the other way.

```bash
pip install -r requirements.txt
python fetch_data.py       # pulls the test split at its pinned revision
python verify_split.py     # confirms you have the data the audit was run against
python adjudicate_independent_full.py
```

The last command prints one line per disputed item and ends:

```
independently adjudicated: 18   gold WRONG: 18   gold ok: 0   not adjudicated: 0
```

`python export_derivations.py` regenerates `derivations_18_disputed.json` from the same code.
Run against a freshly fetched split it reproduces the shipped file exactly, field for field —
only the `_reproduce` string differs, because the path is shorter here than in the repository.

## What the method is

For each disputed item the answer is re-derived **from the premise alone**, sharing no
equivalence-class code with the solver under test: the premise is re-parsed, the equivalence
class is rebuilt by enumerating acyclic orientations of the independence-derived skeleton, and
d-separation comes from a general graph library. A relation is entailed **iff it holds in every
member of the class**.

The argument does not rest on two implementations agreeing. It rests on each class being
checked directly against its premise — which is why `derivations_18_disputed.json` ships the
class itself, not just the verdict. Every item carries every member graph and whether the
hypothesis holds in it, so you can recompute the label by hand and see exactly which graph
refutes a gold label.

## What is here

| file | what it is |
|---|---|
| `adjudicate_independent_full.py` | the adjudication code, unmodified |
| `derivations_18_disputed.json` | the per-item derivations — 18 items, every member graph |
| `export_derivations.py` | regenerates the above from the adjudicator |
| `fetch_data.py` | fetches the test split at its pinned revision |
| `verify_split.py` | confirms the fetched data is what the audit used |
| `generate_p1_figures.py` | the figure generator, shipped as released code — read its header |

## What is not here, and why

**No third-party data.** The Corr2Cause GitHub repository is MIT licensed, but the HuggingFace
dataset card that people actually download **declares no licence at all** — its body is the
word `TODO`. A redistributor cannot determine the terms from the artefact itself, so this
deposit redistributes none of it and fetches at a pinned revision instead. That is also better
provenance than a copy: it proves which bytes the audit ran against and fails loudly if
upstream moves.

`fetch_data.py` derives the `template` and `num_variables` fields, which the released split
does not carry. That derivation was recovered from the data and reproduces all 1162 templates
exactly; `verify_split.py` re-checks it every run. An earlier version guessed and got 387 rows
wrong, with two templates exactly backwards — hence the check.

**No y0 code.** The paper's second case study concerns a BSD-3-Clause library; none of its code
is redistributed here.

## Licence

MIT — see `LICENSE`. This matches the licence of the Corr2Cause repository the underlying data
comes from. The derived labels are released on the assumption that the repository's licence
governs the data it generated; if the authors intend different terms for the distributed
dataset, we will follow them, and we have raised the undeclared dataset-card licence with them
so that it need not stay an assumption.
