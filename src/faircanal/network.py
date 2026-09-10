"""The canal as a tree: how the reaches are connected, and on whose word.

What is published and what is not
---------------------------------
Everything about the reaches is published: their geometry, their offtakes,
the discharge at every check gate, the flow entering the head. What is not
published is a *branched* version of them - the ASCE test canals are both
single cascades, and no source describes either of them as a tree.

So the branching is this study's, and it is declared ``assumed`` wherever
it appears. What is not left to taste is *how* it was chosen: the rule
below is written out, applied by :func:`balanced_split`, and produces one
answer from the published offtakes with nothing to tune. A reader who
disagrees with the rule can change one line and get a different network;
a reader who wants to check that this network is what the rule gives can
run the test.

The rule
--------
1. The reaches, their geometry and their offtakes are the published ones.
   Nothing is invented, duplicated, moved or rescaled.
2. Exactly one branch point is introduced. The published cascade
   ``P1 -> ... -> P8`` becomes a shared stem ``P1..Pp`` and two branches,
   ``P(p+1)..Pq`` and ``P(q+1)..P8``, both fed by ``Pp``.
3. ``(p, q)`` is the split whose two branches have the most nearly equal
   aggregate offtake, so that neither branch trivially dominates the other
   in a fairness comparison. Ties are broken towards branches of equal
   length, and then towards the smaller ``p``. Each branch must hold at
   least two reaches, so that a branch is a canal and not a single gate.
4. The published tail discharge of the cascade is divided between the two
   branch ends in proportion to their aggregate offtake.
5. Every reach's nominal discharge is recomputed from 1, 2 and 4, and
   every pool model is identified at that discharge. No coefficient is
   carried over from the cascade.

Rule 4 is what keeps the network comparable with the published one: the
discharge entering the head comes out at exactly the published value,
because the same water is being delivered to the same offtakes and the
same amount is leaving at the far end. ``test_the_tree_keeps_the_published
_heading_flow`` checks it rather than trusting it.

On the Corning canal the rule gives a stem of one reach, a branch of three
(offtakes 4.8 m3/s) and a branch of four (4.5 m3/s).

Node numbering
--------------
The pool model of :mod:`faircanal.pool` is written in the source's
indexing: pool ``n`` has an inflow ``u_n``, and its outflow is the inflow
of the pool below it, so indices *decrease* going downstream. A tree has
no single line to count along, so the numbering is fixed by a rule that
reduces to the source's on a cascade:

    walk the tree from the root, depth first, taking children in the order
    the source lists them, and hand out ``N, N-1, ..., 1`` in that order.

A parent is always visited before its children, so **every child's number
is smaller than its parent's** - which is the property the rest of the
code relies on and the one thing that makes the cascade formulas carry
over unchanged. Arrays are indexed ``node - 1``, as everywhere else in
this package.

The assumption that comes with branching
----------------------------------------
The outflow coefficients of a pool were identified for a single
downstream gate. Writing the outflow as the sum over the children says
the pool's level does not depend on *which* branch the water leaves
through - that the branch gates sit at one cross section. That is the
A-tree assumption of the model document, it is ``assumed`` and not
measured, and it is declared alongside the topology every time the
topology is reported.

Sources
-------
Geometry, offtakes, check discharges and heading discharge: Bonet E,
Yubero MT, Bascompta M, Alfonso P (2025). A Linear Model for Irrigation
Canals Operating in Real Time Applied in ASCE Test Cases. Water
17(9):1368, doi:10.3390/w17091368, Tables 5 and 6.

Target depths: Litrico X, Fromion V (2004),
doi:10.1061/(ASCE)0733-9437(2004)130:5(373), Table 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faircanal.config import DT_PLANT_S
from faircanal.geometry import (
    CORNING_CHECK_FLOWS,
    CORNING_HEADING_FLOW,
    CORNING_OFFTAKES,
    CORNING_POOLS,
    TrapezoidalPool,
    top_width,
)
from faircanal.identify import identify_pool
from faircanal.pool import PoolParams, simulate_level

__all__ = [
    "MIN_BRANCH_REACHES",
    "Reach",
    "Network",
    "NetworkError",
    "balanced_split",
    "compose_cascade",
    "compose_tree",
    "corning_cascade",
    "corning_tree",
    "identify_network",
    "simulate_network",
    "surface_area",
]

#: A branch shorter than this is a gate, not a canal. Part of the stated
#: rule, not a tuning knob: changing it changes the published rule.
MIN_BRANCH_REACHES = 2


class NetworkError(ValueError):
    """Raised when a topology is not a tree, or not a usable one."""


@dataclass(frozen=True)
class Reach:
    """One reach of the network, with its place in it.

    ``node`` is the model index: one is a branch end, ``N`` is the reach
    the headworks feeds. ``parent`` is the node the water comes from, and
    is ``None`` only for the root.
    """

    node: int
    label: str
    pool: TrapezoidalPool
    offtake_m3_s: float
    tail_m3_s: float
    parent: int | None

    def __post_init__(self) -> None:
        if self.node < 1:
            raise NetworkError(f"{self.label}: node numbers start at one")
        if self.offtake_m3_s < 0.0 or self.tail_m3_s < 0.0:
            raise NetworkError(f"{self.label}: discharges cannot be negative")
        if self.parent is not None and self.parent <= self.node:
            raise NetworkError(
                f"{self.label}: its parent is node {self.parent}, which is not "
                f"above node {self.node}. Water flows from high index to low, so "
                f"a parent's index is always the larger one"
            )


@dataclass(frozen=True)
class Network:
    """A canal network, its topology and where that topology came from.

    ``reaches`` is ordered by node, so ``reaches[n - 1].node == n``.
    ``topology_provenance`` is ``observed`` for the published cascade and
    ``assumed`` for anything branched - and it is carried on the object so
    that a report can never quietly omit it.
    """

    name: str
    reaches: tuple[Reach, ...]
    topology_provenance: str
    topology_rule: str
    source: str

    def __post_init__(self) -> None:
        if not self.reaches:
            raise NetworkError(f"{self.name}: a network needs at least one reach")
        numbers = [reach.node for reach in self.reaches]
        if numbers != list(range(1, len(self.reaches) + 1)):
            raise NetworkError(
                f"{self.name}: reaches must be given in node order 1..N, got "
                f"{numbers}"
            )
        roots = [reach.node for reach in self.reaches if reach.parent is None]
        if len(roots) != 1:
            raise NetworkError(
                f"{self.name}: a tree has exactly one root, found {len(roots)}: "
                f"{roots}"
            )
        if roots[0] != len(self.reaches):
            raise NetworkError(
                f"{self.name}: the root must carry the largest node number, "
                f"found {roots[0]} of {len(self.reaches)}"
            )
        if self.topology_provenance not in ("observed", "assumed"):
            raise NetworkError(
                f"{self.name}: topology provenance must be 'observed' or "
                f"'assumed', got {self.topology_provenance!r}"
            )
        # Every child's index is below its parent's, so following parents
        # strictly increases and has to reach the root. That is the whole
        # acyclicity argument, and __post_init__ of Reach has already
        # enforced the ordering.
        for reach in self.reaches:
            seen = reach.node
            walker = reach.parent
            while walker is not None:
                if walker <= seen:
                    raise NetworkError(f"{self.name}: node {reach.node} loops")
                seen = walker
                walker = self.reaches[walker - 1].parent

    @property
    def size(self) -> int:
        return len(self.reaches)

    @property
    def root(self) -> int:
        return self.size

    def children(self, node: int) -> tuple[int, ...]:
        """Nodes fed by *node*, in increasing order for determinism."""
        return tuple(
            reach.node for reach in self.reaches if reach.parent == node
        )

    @property
    def leaves(self) -> tuple[int, ...]:
        return tuple(reach.node for reach in self.reaches if not self.children(reach.node))

    @property
    def is_cascade(self) -> bool:
        """True when every reach has at most one child."""
        return all(len(self.children(reach.node)) <= 1 for reach in self.reaches)

    def path_to_root(self, node: int) -> tuple[int, ...]:
        """Nodes from *node* up to the root, inclusive."""
        path = [node]
        while self.reaches[path[-1] - 1].parent is not None:
            path.append(self.reaches[path[-1] - 1].parent)
        return tuple(path)

    @property
    def steady_discharges(self) -> tuple[float, ...]:
        """Nominal discharge entering each reach, indexed by ``node - 1``.

        Accumulated from the branch ends upwards: what a reach must carry
        is what its children carry, plus what is drawn at its own gate,
        plus whatever leaves the canal there.

        On the published cascade this returns the published check
        discharges exactly, which is what says the accumulation is right -
        ``test_the_cascade_reproduces_the_published_check_flows``.
        """
        discharges = [0.0] * self.size
        for reach in self.reaches:  # node order: children before parents
            discharges[reach.node - 1] = (
                reach.offtake_m3_s
                + reach.tail_m3_s
                + sum(discharges[child - 1] for child in self.children(reach.node))
            )
        return tuple(discharges)

    @property
    def heading_discharge(self) -> float:
        """What the headworks must release to hold the nominal point."""
        return self.steady_discharges[self.root - 1]

    @property
    def aggregate_demand(self) -> float:
        """Sum of the offtakes - the denominator of the scarcity scan."""
        return sum(reach.offtake_m3_s for reach in self.reaches)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def balanced_split(
    offtakes: "tuple[float, ...] | list[float]",
    min_branch: int = MIN_BRANCH_REACHES,
) -> tuple[int, int]:
    """Where to branch a published cascade, by the rule in the docstring.

    ``offtakes`` is listed most upstream first, as the sources list them.
    Returns ``(p, q)``: the stem is the first ``p`` reaches, the first
    branch is reaches ``p+1 .. q`` and the second is ``q+1 .. len``, all
    counted from one.

    The choice minimises the difference between the two branches'
    aggregate offtake; ties go to branches of more nearly equal length,
    then to the shorter stem. Searching every admissible pair and sorting
    the candidates is what makes the answer independent of the order they
    were generated in.
    """
    count = len(offtakes)
    if count < 1 + 2 * min_branch:
        raise NetworkError(
            f"{count} reaches cannot be split into a stem and two branches of "
            f"at least {min_branch} reaches each"
        )
    candidates = []
    for stem in range(1, count - 2 * min_branch + 1):
        for cut in range(stem + min_branch, count - min_branch + 1):
            first = sum(offtakes[stem:cut])
            second = sum(offtakes[cut:])
            candidates.append(
                (
                    abs(first - second),
                    abs((cut - stem) - (count - cut)),
                    stem,
                    cut,
                )
            )
    candidates.sort()
    _, _, stem, cut = candidates[0]
    return stem, cut


def _number_nodes(parents_by_source: "list[int | None]") -> list[int]:
    """Assign node numbers by the depth-first rule in the module docstring.

    ``parents_by_source[i]`` is the source position of reach ``i``'s parent,
    or ``None`` for the root. Returns the node number of each source
    position.
    """
    count = len(parents_by_source)
    children: list[list[int]] = [[] for _ in range(count)]
    root = None
    for position, parent in enumerate(parents_by_source):
        if parent is None:
            root = position
        else:
            children[parent].append(position)
    if root is None:
        raise NetworkError("no root: every reach claims a parent")

    order: list[int] = []
    stack = [root]
    while stack:
        position = stack.pop()
        order.append(position)
        # Reversed, so that the source's own order is what comes off the
        # stack first and the numbering is reproducible.
        stack.extend(reversed(children[position]))
    if len(order) != count:
        raise NetworkError("some reaches are not reachable from the root")

    numbers = [0] * count
    for rank, position in enumerate(order):
        numbers[position] = count - rank
    return numbers


def _build(
    name: str,
    pools: "tuple[TrapezoidalPool, ...]",
    offtakes: "tuple[float, ...]",
    tails: "tuple[float, ...]",
    parents_by_source: "list[int | None]",
    provenance: str,
    rule: str,
    source: str,
) -> Network:
    numbers = _number_nodes(parents_by_source)
    reaches = [None] * len(pools)
    for position, pool in enumerate(pools):
        parent = parents_by_source[position]
        reaches[numbers[position] - 1] = Reach(
            node=numbers[position],
            label=pool.name,
            pool=pool,
            offtake_m3_s=float(offtakes[position]),
            tail_m3_s=float(tails[position]),
            parent=None if parent is None else numbers[parent],
        )
    return Network(
        name=name,
        reaches=tuple(reaches),
        topology_provenance=provenance,
        topology_rule=rule,
        source=source,
    )


def compose_cascade(
    name: str,
    pools: "tuple[TrapezoidalPool, ...]",
    offtakes: "tuple[float, ...]",
    tail: float,
) -> Network:
    """The published canal, as published: one line of reaches.

    Kept because it is the thing the tree has to be checked against, and
    because a cascade is the case ``|ch(n)| = 1`` that every tree formula
    has to reduce to.
    """
    tails = tuple(
        tail if position == len(pools) - 1 else 0.0 for position in range(len(pools))
    )
    parents = [None if position == 0 else position - 1 for position in range(len(pools))]
    return _build(
        name=name,
        pools=pools,
        offtakes=offtakes,
        tails=tails,
        parents_by_source=parents,
        provenance="observed",
        rule="the published cascade, unchanged",
        source=_CORNING_SOURCE,
    )


def compose_tree(
    name: str,
    pools: "tuple[TrapezoidalPool, ...]",
    offtakes: "tuple[float, ...]",
    tail: float,
    split: "tuple[int, int] | None" = None,
) -> Network:
    """Branch a published cascade by the rule in the module docstring.

    ``split`` overrides the rule, which is how the sensitivity runs look at
    other topologies. A split given by hand is not held to the minimum
    branch length - the point of it is to try what the rule would not
    choose - but what comes out is still ``assumed`` and still carries the
    rule it was built by, so no result can be reported without saying which
    topology produced it.
    """
    stem, cut = balanced_split(offtakes) if split is None else split
    count = len(pools)
    if not 1 <= stem < cut < count:
        raise NetworkError(f"the split {(stem, cut)} does not lie inside {count} reaches")

    first = sum(offtakes[stem:cut])
    second = sum(offtakes[cut:])
    if first + second <= 0.0:
        raise NetworkError("the two branches draw nothing, so the tail cannot be split")

    tails = [0.0] * count
    tails[cut - 1] = tail * first / (first + second)
    tails[count - 1] = tail * second / (first + second)

    parents: list[int | None] = []
    for position in range(count):
        if position == 0:
            parents.append(None)
        elif position == cut:
            parents.append(stem - 1)  # the second branch hangs off the stem
        else:
            parents.append(position - 1)

    rule = (
        f"published cascade of {count} reaches, branched at reach {stem}: "
        f"reaches {stem + 1}-{cut} draw {first:g} m3/s and reaches {cut + 1}-{count} "
        f"draw {second:g} m3/s, the most nearly equal split available; the "
        f"published tail discharge of {tail:g} m3/s is divided between the two "
        f"branch ends in that same proportion"
    )
    return _build(
        name=name,
        pools=pools,
        offtakes=offtakes,
        tails=tuple(tails),
        parents_by_source=parents,
        provenance="assumed",
        rule=rule,
        source=_CORNING_SOURCE,
    )


_CORNING_SOURCE = (
    "Bonet E, Yubero MT, Bascompta M, Alfonso P (2025). A Linear Model for "
    "Irrigation Canals Operating in Real Time Applied in ASCE Test Cases. "
    "Water 17(9):1368, doi:10.3390/w17091368, Tables 5 and 6. Topology "
    "composed by the authors and declared assumed."
)


def corning_cascade() -> Network:
    """The Corning canal exactly as published."""
    return compose_cascade(
        "corning-cascade", CORNING_POOLS, CORNING_OFFTAKES, CORNING_CHECK_FLOWS[-1]
    )


def corning_tree() -> Network:
    """The Corning canal branched by the rule."""
    return compose_tree(
        "corning-tree", CORNING_POOLS, CORNING_OFFTAKES, CORNING_CHECK_FLOWS[-1]
    )


# ---------------------------------------------------------------------------
# From a network to a plant
# ---------------------------------------------------------------------------


def identify_network(
    network: Network, dt: float = DT_PLANT_S
) -> tuple[PoolParams, ...]:
    """Identify every reach at the discharge the topology gives it.

    Nothing is carried over from the published cascade: branching changes
    what each reach carries, so it changes the pool model, and the model is
    recomputed rather than reused. Returned in node order, so
    ``models[n - 1]`` belongs to node ``n``.
    """
    discharges = network.steady_discharges
    return tuple(
        identify_pool(reach.pool, discharges[reach.node - 1], dt=dt)
        for reach in network.reaches
    )


def surface_area(reach: Reach) -> float:
    """Water-surface area of a reach at its target level [m^2].

    The storage bound of the slow layer is this times the level band, so
    it is a geometric quantity and is labelled ``derived``.
    """
    depth = reach.pool.target_level_m
    if depth is None:
        raise NetworkError(f"{reach.label}: no target level to measure the surface at")
    return top_width(reach.pool, depth) * reach.pool.length_m


def simulate_network(
    network: Network,
    models: "tuple[PoolParams, ...]",
    inflow: np.ndarray,
    offtake: np.ndarray,
    initial_levels: "np.ndarray | None" = None,
) -> np.ndarray:
    """Run the whole network open loop.

    Parameters
    ----------
    network:
        The topology.
    models:
        Pool models in node order, as :func:`identify_network` returns
        them.
    inflow:
        ``inflow[k, n - 1]`` is the discharge entering reach ``n`` during
        step ``k`` - the source's ``u_n``. The root's column is what the
        headworks releases.
    offtake:
        ``offtake[k, n - 1]`` is what the users at reach ``n``'s gate draw
        during step ``k``. Positive takes water out.

    Returns
    -------
    numpy.ndarray
        Level deviations, shape ``(steps + 1, N)``, in node order.

    A reach's outflow is the sum of what its children take in, which is
    the whole of the tree extension: a cascade is the case of one child
    and gives the same answer as the cascade code
    (``test_a_cascade_network_matches_the_plain_pool_model``).
    """
    inflow = np.asarray(inflow, dtype=float)
    offtake = np.asarray(offtake, dtype=float)
    size = network.size
    if inflow.ndim != 2 or inflow.shape[1] != size:
        raise NetworkError(f"inflow must have shape (steps, {size}), got {inflow.shape}")
    if offtake.shape != inflow.shape:
        raise NetworkError(
            f"offtake must have the same shape as inflow, got {offtake.shape} "
            f"and {inflow.shape}"
        )
    if len(models) != size:
        raise NetworkError(f"{len(models)} models for {size} reaches")
    if initial_levels is None:
        initial_levels = np.zeros(size)
    initial_levels = np.asarray(initial_levels, dtype=float)
    if initial_levels.shape != (size,):
        raise NetworkError(f"initial_levels must have shape ({size},)")

    steps = inflow.shape[0]
    levels = np.zeros((steps + 1, size))
    for reach, model in zip(network.reaches, models):
        column = reach.node - 1
        children = network.children(reach.node)
        outflow = (
            np.sum([inflow[:, child - 1] for child in children], axis=0)
            if children
            else np.zeros(steps)
        )
        levels[:, column] = simulate_level(
            model,
            inflow[:, column],
            outflow,
            offtake[:, column],
            y_initial=float(initial_levels[column]),
        )
    return levels
