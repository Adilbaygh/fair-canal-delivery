"""Permanent tests for the network topology. Never deleted.

The topology is the one part of this study that no source states, so it is
the part most at risk of drifting into something nobody can reconstruct.
These tests hold three things in place: that the published cascade is
reproduced exactly, that the branching is what the written rule gives and
not what somebody preferred, and that the tree formulas collapse to the
cascade ones when there is only one child.
"""

from __future__ import annotations

import numpy as np
import pytest

from faircanal.geometry import (
    CORNING_CHECK_FLOWS,
    CORNING_HEADING_FLOW,
    CORNING_OFFTAKES,
    CORNING_POOLS,
)
from faircanal.network import (
    MIN_BRANCH_REACHES,
    Network,
    NetworkError,
    Reach,
    balanced_split,
    compose_cascade,
    compose_tree,
    corning_cascade,
    corning_tree,
    identify_network,
    simulate_network,
    surface_area,
)
from faircanal.pool import simulate_level


# ---------------------------------------------------------------------------
# The published canal, reproduced
# ---------------------------------------------------------------------------


def test_the_cascade_reproduces_the_published_check_flows():
    """Accumulating from the far end gives back every published discharge.

    The source publishes the offtake at each gate, the discharge past each
    check and the discharge entering the head - eighteen numbers that have
    to be consistent with each other. Rebuilding the middle sixteen from
    the offtakes and the tail alone is what says the balance is right, and
    it catches an off-by-one in the topology that nothing else would.
    """
    cascade = corning_cascade()
    published = (CORNING_HEADING_FLOW,) + CORNING_CHECK_FLOWS[:-1]
    # Node order runs the other way: node N is the reach at the head.
    computed = tuple(reversed(cascade.steady_discharges))
    assert computed == pytest.approx(published, abs=1e-12)
    assert cascade.heading_discharge == pytest.approx(CORNING_HEADING_FLOW, abs=1e-12)


def test_the_published_numbers_are_consistent_on_their_own():
    """Offtakes plus tail equal the heading discharge, before any code.

    If this fails, the transcription is wrong and every other test in this
    file is checking one wrong number against another.
    """
    assert sum(CORNING_OFFTAKES) + CORNING_CHECK_FLOWS[-1] == pytest.approx(
        CORNING_HEADING_FLOW, abs=1e-12
    )


def test_the_cascade_numbering_is_the_sources_numbering():
    """Node N is the reach at the head; node one is the far end.

    The pool model counts downwards - pool ``n`` takes in ``u_n`` and gives
    out ``u_{n-1}`` - while the geometry tables list reaches from the head.
    Getting that inversion wrong would run the canal backwards and every
    level would still look plausible, so it is pinned here.
    """
    cascade = corning_cascade()
    assert cascade.root == 8
    assert cascade.reaches[7].label == "corning-1"
    assert cascade.reaches[0].label == "corning-8"
    assert cascade.reaches[7].parent is None
    assert cascade.children(8) == (7,)
    assert cascade.leaves == (1,)
    assert cascade.is_cascade


# ---------------------------------------------------------------------------
# The rule, and that it is a rule
# ---------------------------------------------------------------------------


def test_the_split_is_the_most_balanced_one_available():
    """Nothing admissible balances the branches better than what was chosen.

    The rule is written as an argmin, so it can be checked by searching the
    same space again from scratch. That is the difference between a rule
    and a preference: a preference cannot be audited.
    """
    stem, cut = balanced_split(CORNING_OFFTAKES)
    chosen = abs(sum(CORNING_OFFTAKES[stem:cut]) - sum(CORNING_OFFTAKES[cut:]))
    count = len(CORNING_OFFTAKES)
    for other_stem in range(1, count):
        for other_cut in range(other_stem + 1, count):
            if other_cut - other_stem < MIN_BRANCH_REACHES:
                continue
            if count - other_cut < MIN_BRANCH_REACHES:
                continue
            gap = abs(
                sum(CORNING_OFFTAKES[other_stem:other_cut])
                - sum(CORNING_OFFTAKES[other_cut:])
            )
            assert gap >= chosen - 1e-12, (
                f"the split ({other_stem}, {other_cut}) balances the branches "
                f"better ({gap:.3f} against {chosen:.3f})"
            )


def test_the_split_is_deterministic_and_ties_are_broken_in_writing():
    """Two splits tie on balance; the written tie-break picks one of them.

    Reaches 2-3 against 4-8 and reaches 2-4 against 5-8 both leave a gap of
    0.3 m3/s. The rule says branches of more nearly equal length win, so it
    is the second. Without a written tie-break this would depend on the
    order the candidates happened to be generated in.
    """
    assert balanced_split(CORNING_OFFTAKES) == (1, 4)
    assert sum(CORNING_OFFTAKES[1:4]) == pytest.approx(4.8)
    assert sum(CORNING_OFFTAKES[4:]) == pytest.approx(4.5)
    # The tie really is a tie.
    assert abs(sum(CORNING_OFFTAKES[1:3]) - sum(CORNING_OFFTAKES[3:])) == pytest.approx(
        abs(sum(CORNING_OFFTAKES[1:4]) - sum(CORNING_OFFTAKES[4:])), abs=1e-12
    )


def test_the_tree_keeps_the_published_heading_flow():
    """Branching moves water around; it does not add or remove any.

    The same offtakes are served and the same amount leaves at the far
    ends, so the headworks releases exactly what the source says it does.
    This is what makes the composed network comparable with the published
    one rather than merely inspired by it.
    """
    tree = corning_tree()
    assert tree.heading_discharge == pytest.approx(CORNING_HEADING_FLOW, abs=1e-12)
    assert tree.aggregate_demand == pytest.approx(sum(CORNING_OFFTAKES), abs=1e-12)
    tails = sum(reach.tail_m3_s for reach in tree.reaches)
    assert tails == pytest.approx(CORNING_CHECK_FLOWS[-1], abs=1e-12)


def test_the_tail_is_split_in_proportion_to_what_each_branch_draws():
    tree = corning_tree()
    ends = [reach for reach in tree.reaches if reach.tail_m3_s > 0.0]
    assert len(ends) == 2
    first, second = sorted(ends, key=lambda reach: reach.node)
    branches = {}
    for end in (first, second):
        path = tree.path_to_root(end.node)
        branches[end.node] = sum(
            tree.reaches[node - 1].offtake_m3_s for node in path[:-1]
        )
    total_offtake = sum(branches.values())
    for end in (first, second):
        assert end.tail_m3_s == pytest.approx(
            CORNING_CHECK_FLOWS[-1] * branches[end.node] / total_offtake, rel=1e-12
        )


def test_the_tree_is_a_tree_and_says_so():
    tree = corning_tree()
    assert not tree.is_cascade
    assert tree.topology_provenance == "assumed"
    assert "branched at reach" in tree.topology_rule
    assert corning_cascade().topology_provenance == "observed"

    branch_points = [
        reach.node for reach in tree.reaches if len(tree.children(reach.node)) > 1
    ]
    assert len(branch_points) == 1
    assert tree.children(branch_points[0]) == (4, 7)
    assert set(tree.leaves) == {1, 5}
    for leaf in tree.leaves:
        assert tree.path_to_root(leaf)[-1] == tree.root


def test_every_child_is_numbered_below_its_parent():
    """The invariant the pool formulas rest on, checked on both networks."""
    for network in (corning_cascade(), corning_tree()):
        for reach in network.reaches:
            for child in network.children(reach.node):
                assert child < reach.node, network.name
            if reach.parent is not None:
                assert reach.parent > reach.node, network.name


# ---------------------------------------------------------------------------
# The plant the topology produces
# ---------------------------------------------------------------------------


def test_branching_changes_the_pool_models():
    """Every model is recomputed, because branching changes what is carried.

    Reusing the cascade's coefficients in the tree would be the quiet kind
    of error: the numbers stay plausible and the canal is a different
    canal. Only the head reach, whose discharge the branching does not
    touch, may come out the same.
    """
    cascade = dict(zip(range(1, 9), identify_network(corning_cascade())))
    tree = dict(zip(range(1, 9), identify_network(corning_tree())))
    unchanged = [node for node in cascade if cascade[node].b == tree[node].b]
    assert unchanged == [8], f"models unchanged at nodes {unchanged}"
    assert cascade[8].b == tree[8].b  # the head carries the same 13.7 m3/s


def test_every_reach_of_both_networks_can_be_identified():
    for network in (corning_cascade(), corning_tree()):
        models = identify_network(network)
        assert len(models) == network.size
        for model, reach in zip(models, network.reaches):
            assert model.order == 3, reach.label
            assert model.tau > 0, reach.label
            assert model.provenance == "derived"


def test_a_cascade_network_matches_the_plain_pool_model():
    """The tree extension reduces to the cascade, run rather than argued.

    A cascade is the case of one child, and the cascade formula is written
    somewhere else in this package as ``outflow = u[n - 1]``. The two are
    compared on the same random input so that a disagreement about which
    index means which reach shows up as a number.
    """
    cascade = corning_cascade()
    models = identify_network(cascade)
    steps = 120
    rng = np.random.default_rng(11)
    inflow = rng.normal(size=(steps, cascade.size))
    offtake = rng.normal(size=(steps, cascade.size))

    got = simulate_network(cascade, models, inflow, offtake)

    want = np.zeros((steps + 1, cascade.size))
    for index, model in enumerate(models):
        outflow = inflow[:, index - 1] if index >= 1 else np.zeros(steps)
        want[:, index] = simulate_level(model, inflow[:, index], outflow, offtake[:, index])
    assert got == pytest.approx(want, abs=1e-12)


def test_a_parent_sees_both_of_its_children():
    """Two branches drawing at once draw twice, and the parent knows it.

    Writing the outflow as a sum is the entire tree extension, so it is
    worth checking that dropping one branch really changes the parent's
    level and that the two contributions add - which they must, since the
    pool model is linear.
    """
    tree = corning_tree()
    models = identify_network(tree)
    steps = 150
    branch_point = 8
    first, second = tree.children(branch_point)

    def run(drive: "tuple[int, ...]") -> np.ndarray:
        inflow = np.zeros((steps, tree.size))
        for node in drive:
            inflow[20:, node - 1] = 1.0
        return simulate_network(tree, models, inflow, np.zeros((steps, tree.size)))

    only_first = run((first,))[:, branch_point - 1]
    only_second = run((second,))[:, branch_point - 1]
    both = run((first, second))[:, branch_point - 1]

    assert np.abs(only_first).max() > 1e-6
    assert np.abs(only_second).max() > 1e-6
    assert only_first[-1] < 0.0, "a branch drawing water must lower the parent"
    assert both == pytest.approx(only_first + only_second, abs=1e-12)
    assert np.abs(both).max() > np.abs(only_first).max()


def test_the_surface_area_is_the_reach_at_its_target_level():
    tree = corning_tree()
    head = tree.reaches[tree.root - 1]
    assert head.label == "corning-1"
    # 7 km long, 7 m wide at the bed, banks at 1.5 to 1, held at 2.1 m.
    assert surface_area(head) == pytest.approx((7.0 + 2 * 1.5 * 2.1) * 7000.0, rel=1e-12)
    for reach in tree.reaches:
        assert surface_area(reach) > 0.0


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def make_reach(node, parent, offtake=1.0, tail=0.0, position=0):
    return Reach(
        node=node,
        label=f"r{node}",
        pool=CORNING_POOLS[position],
        offtake_m3_s=offtake,
        tail_m3_s=tail,
        parent=parent,
    )


def test_a_parent_below_its_child_is_refused():
    with pytest.raises(NetworkError, match="not above"):
        make_reach(node=3, parent=2)


def test_two_roots_are_refused():
    with pytest.raises(NetworkError, match="exactly one root"):
        Network(
            name="two-roots",
            reaches=(make_reach(1, None), make_reach(2, None, position=1)),
            topology_provenance="assumed",
            topology_rule="broken",
            source="test",
        )


def test_the_root_cannot_be_anything_but_the_highest_node():
    """Why the root is always node N, argued once and pinned here.

    A reach's parent must carry a larger number, so the largest node can
    never have one - it is always a root. Add "exactly one root" and the
    root has to be node N. The guard in ``Network`` for a root that is not
    the largest is therefore unreachable, and this test records that so
    nobody spends an afternoon trying to trigger it.
    """
    with pytest.raises(NetworkError, match="not above"):
        make_reach(node=2, parent=2)
    with pytest.raises(NetworkError, match="not above"):
        make_reach(node=2, parent=1)
    for network in (corning_cascade(), corning_tree()):
        assert network.reaches[-1].parent is None
        assert network.root == network.size


def test_reaches_out_of_node_order_are_refused():
    with pytest.raises(NetworkError, match="node order"):
        Network(
            name="shuffled",
            reaches=(make_reach(2, None, position=1), make_reach(1, 2)),
            topology_provenance="assumed",
            topology_rule="broken",
            source="test",
        )


def test_an_undeclared_provenance_is_refused():
    with pytest.raises(NetworkError, match="provenance"):
        Network(
            name="unlabelled",
            reaches=(make_reach(1, None),),
            topology_provenance="probably fine",
            topology_rule="none",
            source="test",
        )


def test_an_unreachable_reach_is_refused():
    """A reach hanging off nothing is caught while the numbers are assigned."""
    from faircanal.network import _number_nodes

    assert _number_nodes([None, 0, 1, 2]) == [4, 3, 2, 1]
    with pytest.raises(NetworkError, match="not reachable"):
        _number_nodes([None, 0, 3, 2])  # 2 and 3 point at each other
    with pytest.raises(NetworkError, match="no root"):
        _number_nodes([1, 0])


def test_an_explicit_split_bypasses_the_rule_and_says_so():
    """The rule is the default; a caller may still ask for another split.

    Passing ``split`` by hand is how the sensitivity runs look at other
    topologies. It is not held to the minimum branch length, because the
    point of it is to try what the rule would not choose - so anything
    built that way is still ``assumed`` and still carries its own rule
    string.
    """
    odd = compose_tree(
        "one-reach-branch",
        CORNING_POOLS[:4],
        CORNING_OFFTAKES[:4],
        1.0,
        split=(1, 2),
    )
    assert not odd.is_cascade
    assert odd.topology_provenance == "assumed"
    assert len(odd.children(odd.root)) == 2


def test_a_canal_too_short_to_branch_is_refused():
    with pytest.raises(NetworkError, match="cannot be split"):
        balanced_split((1.0, 1.0, 1.0, 1.0))


def test_a_split_outside_the_canal_is_refused():
    with pytest.raises(NetworkError, match="does not lie inside"):
        compose_tree("bad", CORNING_POOLS, CORNING_OFFTAKES, 2.7, split=(4, 4))


def test_mismatched_simulation_inputs_are_refused():
    tree = corning_tree()
    models = identify_network(tree)
    with pytest.raises(NetworkError, match="shape"):
        simulate_network(tree, models, np.zeros((10, 3)), np.zeros((10, 3)))
    with pytest.raises(NetworkError, match="same shape"):
        simulate_network(tree, models, np.zeros((10, 8)), np.zeros((9, 8)))
    with pytest.raises(NetworkError, match="models for"):
        simulate_network(tree, models[:3], np.zeros((10, 8)), np.zeros((10, 8)))
