"""
engine/core/identification.py
Canon's general causal-identification interface — the thin wrapper over y0 (BSD-3-Clause)
that turns do-calculus into a Canon capability with a stable contract.

WHY a wrapper (not call y0 everywhere): Canon owns
  - the abstraction   : the rest of Canon never imports y0 directly,
  - the certificate   : the symbolic estimand string is the audit artifact,
  - the abstention    : non-identifiable -> identifiable=False, estimand=None (honest "I can't",
                        never a guessed number).

Identification itself (Shpitser-Pearl ID) is delegated to y0 — verified 2026-06-17:
front-door -> exact estimand; bow-arc -> Unidentifiable. See tests/test_identification_y0.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

from y0.graph import NxMixedGraph
from y0.dsl import Variable, P
from y0.algorithm.identify import identify, Identification, Unidentifiable

Edge = Tuple[str, str]


@dataclass(frozen=True)
class IdentificationResult:
    """Result of asking 'is P(outcome | do(treatment)) identifiable in this graph?'"""
    identifiable: bool
    estimand: Optional[str]   # symbolic certificate (str of the y0 estimand) or None if abstaining
    treatment: str
    outcome: str
    estimand_expr: object = None  # internal: the raw y0 Expression, so the SAME identification
                                  # that produced the certificate is the one that gets evaluated
                                  # (no second identify call -> certificate and number can't diverge)


def identify_effect(
    directed: List[Edge],
    bidirected: List[Edge],
    treatment: str,
    outcome: str,
) -> IdentificationResult:
    """Identify P(outcome | do(treatment)) in an ADMG (directed + bidirected edges).

    bidirected edges encode unobserved confounding. Returns a certified estimand when
    identifiable, or an honest abstention (identifiable=False, estimand=None) otherwise.

    A MALFORMED query (treatment==outcome, or a variable absent from the graph) raises
    ValueError — that is a usage error, distinct from a non-identifiable effect. We never
    silently return a confidently-wrong answer for a query about a variable not in the graph.
    """
    if treatment == outcome:
        raise ValueError(f"treatment and outcome must differ (both '{treatment}')")
    nodes = {n for edge in (*directed, *bidirected) for n in edge}
    for role, var in (("treatment", treatment), ("outcome", outcome)):
        if var not in nodes:
            raise ValueError(f"{role} '{var}' is not a node in the graph")
    graph = NxMixedGraph.from_edges(directed=directed, undirected=bidirected)
    query = P(Variable(outcome) @ Variable(treatment))
    try:
        estimand = identify(Identification.from_expression(query=query, graph=graph))
    except Unidentifiable:
        return IdentificationResult(False, None, treatment, outcome)
    return IdentificationResult(True, str(estimand), treatment, outcome, estimand)
