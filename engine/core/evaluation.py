"""
engine/core/evaluation.py
Evaluate a Canon identification certificate (a y0 estimand) to a certified NUMBER on an
OBSERVED joint distribution. For latent-confounded graphs the naive g-formula is impossible
(the confounder is unobserved); the identified estimand evaluated on the observed joint is
the correct route — and where Canon earns its keep.

Transparent and deterministic by design: the evaluation is plain arithmetic over the joint,
so the number is as auditable as the symbolic estimand that produced it.

Scope: discrete (binary-or-categorical) variables, estimands composed of the y0 AST fragment
{Probability, Product, Sum} — which is what the do-calculus ID algorithm returns.
"""
from __future__ import annotations
from itertools import product as _iproduct
from typing import Dict, List, Tuple

from y0.algorithm.identify import Unidentifiable

from engine.core.identification import identify_effect

Assignment = Dict[str, int]


class AmbiguousEstimand(Exception):
    """The estimand does not determine a single number on this data.

    Raised when an estimand leaves variables free and the value is NOT invariant across their
    assignments — the "free handle" did not drop out, so there is no one answer to give. Also
    raised when the free-variable space is too large to enumerate, where the honest report is
    that we did not check rather than a number we did not earn.

    Distinct from `UnverifiedEffect`, which means a single number exists but no independent
    route confirms it.
    """


class UnverifiedEffect(Exception):
    """The estimand evaluated, but no independent derivation reproduced the number.

    Raised by `prob_under_do`, which returns a bare float and therefore has no channel in
    which to carry a caveat. Its siblings already refuse rather than return something they
    cannot stand behind — `Unidentifiable` when the effect is not identifiable, `ValueError`
    when the data cannot evaluate the estimand — and this is the third member of that family:
    identified, evaluable, but not independently confirmable (y0 0.2.11 is unsound on part of
    the ID problem, so "identified" alone does not mean "correct").

    Callers who genuinely want the unconfirmed number can ask for it with
    `require_verified=False`; what they cannot do is receive it while believing it certified.
    """


class JointDistribution:
    """A fully-specified discrete joint distribution over `variables`.

    `table` maps a tuple of values (aligned to `variables` order) to its probability.
    """

    def __init__(self, variables: List[str], table: Dict[Tuple[int, ...], float]):
        self.variables = list(variables)
        self._index = {v: i for i, v in enumerate(self.variables)}
        self.table = dict(table)

    def marginal(self, assignment: Assignment) -> float:
        """P(assignment): sum of table entries consistent with the partial assignment."""
        items = [(self._index[v], val) for v, val in assignment.items()]
        return float(sum(p for key, p in self.table.items()
                         if all(key[i] == val for i, val in items)))

    def conditional(self, children: Assignment, given: Assignment) -> float:
        """P(children | given). With empty `given`, returns the marginal P(children)."""
        if not given:
            return self.marginal(children)
        denom = self.marginal(given)
        if denom == 0:
            return 0.0
        return self.marginal({**given, **children}) / denom

    def domain(self, var: str) -> set:
        """The set of values `var` takes in the table."""
        i = self._index[var]
        return {key[i] for key in self.table}


FREE_VAR_TOLERANCE = 1e-9      # agreement required across a free variable's assignments
MAX_FREE_ASSIGNMENTS = 16      # refuse rather than enumerate a large free-variable space


def _free_vars(expr, bound: set) -> set:
    """Names `expr` references that are bound neither by `bound` nor by an enclosing Sum."""
    node = type(expr).__name__
    if node == "Probability":
        d = expr.distribution
        return ({v.name for v in d.children} | {v.name for v in d.parents}) - bound
    if node == "Product":
        out = set()
        for factor in expr.expressions:
            out |= _free_vars(factor, bound)
        return out
    if node == "Sum":
        return _free_vars(expr.expression, bound | {v.name for v in expr.ranges})
    if node == "Fraction":
        return _free_vars(expr.numerator, bound) | _free_vars(expr.denominator, bound)
    return set()               # One / Zero reference nothing


def free_estimand_vars(estimand_expr, context: Assignment) -> set:
    """Variables the estimand references that `context` does not bind and no Sum ranges over.

    Sibling of `missing_distribution_vars`: that one asks whether the DATA has the variable,
    this one asks whether the QUERY pins it down. Both must be empty before a single number
    exists — and the two failures used to look completely different, one a clear ValueError
    and the other a bare KeyError from deep inside the recursion.
    """
    return _free_vars(estimand_expr, set(context))


def evaluate_estimand(expr, distribution: JointDistribution, context: Assignment) -> float:
    """Evaluate a y0 estimand AST (Probability / Product / Sum / Fraction) under `context`.

    `context` binds the query's treatment & outcome values. Sum nodes bind their own range
    variables, shadowing the context within their body (correct scoping).

    FREE VARIABLES. y0 returns estimands that mention variables the query does not bind and
    no Sum ranges over — `line_3_example` identifies P(Y|do(X)) as `P(Y | X, Z)`, and
    `identifiability_1` leaves Z2 and Z4 free. Such a handle is legitimate precisely when the
    value does not depend on it, which is the same signature that marks a valid identification
    elsewhere in this codebase (the napkin ratio estimand gives one answer for either z1).

    So rather than looking the variable up, missing, and dying with a bare KeyError — a crash
    where an answer was available — this evaluates the estimand at every assignment of the
    free variables and requires them to agree. Agreement returns the common value; that the
    handle dropped out is the proof it was free. Disagreement means the estimand does not
    determine a single number on this data, and we raise `AmbiguousEstimand` rather than
    silently picking one or averaging them (averaging would be a guess wearing arithmetic).

    Measured on y0's own examples (identifiability_1, identifiability_2, line_3_example; 3
    random SCMs each): spread across the free assignments <= 2.2e-16, and the common value
    equals brute-force truth to ~1e-16. All three used to be crashes.
    """
    free = sorted(_free_vars(expr, set(context)))
    if not free:
        return _eval_bound(expr, distribution, context)

    unknown = [v for v in free if v not in distribution.variables]
    if unknown:
        raise ValueError(
            f"estimand references variable(s) {sorted(unknown)} that the distribution does "
            f"not contain, and the query does not bind them either"
        )

    domains = [sorted(distribution.domain(v)) for v in free]
    total_assignments = 1
    for dom in domains:
        total_assignments *= len(dom)
    if total_assignments > MAX_FREE_ASSIGNMENTS:
        raise AmbiguousEstimand(
            f"estimand leaves {len(free)} variable(s) free ({free}) spanning "
            f"{total_assignments} assignments, above the {MAX_FREE_ASSIGNMENTS} this will "
            f"enumerate. Refusing rather than guessing or hanging; bind them in `context` "
            f"if you know their values."
        )

    values = []
    for combo in _iproduct(*domains):
        scoped = dict(context)
        scoped.update(zip(free, combo))
        values.append(_eval_bound(expr, distribution, scoped))

    spread = max(values) - min(values)
    if spread > FREE_VAR_TOLERANCE:
        raise AmbiguousEstimand(
            f"estimand value depends on free variable(s) {free}: it ranges over "
            f"[{min(values):.6f}, {max(values):.6f}] (spread {spread:.2e} > "
            f"{FREE_VAR_TOLERANCE:.0e}), so it does not identify a single number on this "
            f"data. The handle did not drop out, which means it was not free."
        )
    return values[0]


def _eval_bound(expr, distribution: JointDistribution, context: Assignment) -> float:
    """The recursion proper. Every variable it meets is bound; `evaluate_estimand` guarantees it."""
    node = type(expr).__name__
    if node == "Probability":
        dist = expr.distribution
        children = {v.name: context[v.name] for v in dist.children}
        parents = {v.name: context[v.name] for v in dist.parents}
        return distribution.conditional(children, parents)
    if node == "Product":
        result = 1.0
        for factor in expr.expressions:
            result *= _eval_bound(factor, distribution, context)
        return result
    if node == "Sum":
        range_vars = [v.name for v in expr.ranges]
        domains = [sorted(distribution.domain(v)) for v in range_vars]
        total = 0.0
        for combo in _iproduct(*domains):
            scoped = dict(context)
            scoped.update(zip(range_vars, combo))
            total += _eval_bound(expr.expression, distribution, scoped)
        return total
    if node == "Fraction":
        denom = _eval_bound(expr.denominator, distribution, context)
        if denom == 0:
            return 0.0   # honest: an undefined ratio contributes nothing rather than crashing
        return _eval_bound(expr.numerator, distribution, context) / denom
    if node == "One":
        return 1.0
    if node == "Zero":
        return 0.0
    raise TypeError(f"unsupported estimand node type: {node}")


def missing_distribution_vars(estimand_expr, distribution: JointDistribution) -> set:
    """Variables the estimand references but the observed distribution does not contain.
    Non-empty -> the effect is identified but cannot be evaluated on this data."""
    needed = {v.name for v in estimand_expr.get_variables()}
    return needed - set(distribution.variables)


def prob_under_do(directed, bidirected, treatment: str, outcome: str,
                  distribution: JointDistribution, x_value: int, y_value: int = 1,
                  require_verified: bool = True) -> float:
    """Certified P(outcome=y_value | do(treatment=x_value)) in an ADMG, evaluated on the
    observed `distribution`. Raises ValueError on a malformed query (variable absent /
    treatment==outcome) and Unidentifiable if the effect is not identifiable.

    Goes through identify_effect so the SAME (validated) identification produces the estimand
    that is evaluated — input validation and certificate/number consistency in one place.

    "Certified" is load-bearing, so it is now enforced rather than asserted. Identification is
    delegated to y0, and y0 0.2.11 is unsound on part of the ID problem: it returns a wrong
    estimand while reporting identifiable=True (napkin, identifiability_3, identifiability_7 —
    its own examples; ~0.17% of random ADMGs). This function returns a bare float and so has
    nowhere to put a caveat, which makes it the worst place in the codebase to hand back a
    number nobody checked. It therefore raises `UnverifiedEffect` when an independent
    derivation cannot reproduce the value — the same refusal it already makes when the effect
    is unidentifiable or the data cannot evaluate the estimand.

    Pass `require_verified=False` for the old behaviour when you want the unconfirmed number
    deliberately (the soundness benchmarks do, since measuring the defect requires seeing it).
    """
    # Deferred: engine.core.verify imports JointDistribution from this module, so a top-level
    # import here would be circular. Imported at call time instead of restructuring both.
    from engine.core.verify import verify_value

    result = identify_effect(directed, bidirected, treatment, outcome)  # validates + identifies
    if not result.identifiable:
        raise Unidentifiable(f"P({outcome} | do({treatment})) is not identifiable in this graph")
    missing = missing_distribution_vars(result.estimand_expr, distribution)
    if missing:
        raise ValueError(f"distribution is missing variable(s) required by the estimand: {sorted(missing)}")
    context = {treatment: x_value, outcome: y_value}
    value = evaluate_estimand(result.estimand_expr, distribution, context)

    if require_verified:
        check = verify_value(directed, bidirected, treatment, outcome, distribution,
                             value, x_value, y_value)
        if not check.agrees:
            raise UnverifiedEffect(
                f"P({outcome} | do({treatment})) evaluated to {value!r}, but no independent "
                f"derivation confirms it ({check.method}). The estimand is y0's word alone and "
                f"y0 is unsound on part of the ID problem. Pass require_verified=False to take "
                f"the number anyway."
            )
    return value
