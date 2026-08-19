"""
Tests for Equilibrium_finder_1603.py.

The module under test is left exactly as written for the dissertation. It ends
with a usage block that instantiates n = 5 and prints several thousand lines, so
importing it normally would run that on every test session. `load_game_class`
below compiles the file with that trailing block removed, which means the class
under test is the file's own code and nothing here edits it.

Two claims are checked, and they are not the same claim:

  test_bound_is_attained            the count matches the closed form
  test_every_profile_is_nash        each profile really is an equilibrium

A generator can satisfy either while failing the other. Counting the right
number of things does not show the things are equilibria, and emitting only
genuine equilibria does not show none were missed.
"""

from __future__ import annotations

import io
import itertools
import pathlib
import sys
from contextlib import redirect_stdout
from math import comb, factorial

import numpy as np
import pytest

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "Equilibrium_finder_1603.py"
USAGE_MARKER = "#------------------------------------Usage starts here"

# Bound values for n = 1..8. n = 3 and n = 5 are the dissertation's worked
# cases: 9 equilibria in Table 4.1, and 185 for the n = 5 delta sort order.
KNOWN_BOUNDS = {1: 1, 2: 3, 3: 9, 4: 37, 5: 185, 6: 1111, 7: 7777, 8: 62217}


def load_game_class():
    """Compile the module without its trailing usage block and hand back the class."""
    src = SOURCE.read_text().split(USAGE_MARKER)[0]
    namespace: dict = {}
    exec(compile(src, str(SOURCE), "exec"), namespace)
    return namespace["ProductTwoActionGame"]


ProductTwoActionGame = load_game_class()


def quiet(fn, *args, **kwargs):
    """Run a method that prints, keeping the test output readable."""
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def subfactorial(k: int) -> int:
    return int(round(factorial(k) * sum((-1) ** i / factorial(i) for i in range(k + 1))))


def hertling_vujic_bound(n: int) -> int:
    """Equation (1.1): !n + sum_l C(n,l) 2^(l-1) !(n-l)."""
    return subfactorial(n) + sum(
        comb(n, l) * 2 ** (l - 1) * subfactorial(n - l) for l in range(1, n + 1)
    )


# --------------------------------------------------------------------- Theorem 3.3

def partial_utility(game, i: int, x: np.ndarray) -> float:
    """Q_i(x_-i) = prod_{j != i} (x_j - a^i_j), Definition 3.2."""
    product = 1.0
    for j in range(game.n):
        if j != i:
            product *= x[j] - game.A[i, j]
    return product


def is_nash(game, x: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Theorem 3.3, checked directly:
        Q_i > 0  =>  x_i = 1
        Q_i < 0  =>  x_i = 0
        Q_i = 0  =>  x_i free in [0, 1]

    Deliberately independent of the increment and parity machinery the module
    uses to build its profiles, so this can catch a disagreement between the two.
    """
    for i in range(game.n):
        q = partial_utility(game, i, x)
        if q > tol and not np.isclose(x[i], 1.0):
            return False
        if q < -tol and not np.isclose(x[i], 0.0):
            return False
        if abs(q) <= tol and not (-tol <= x[i] <= 1.0 + tol):
            return False
    return True


# ------------------------------------------------------------------- coefficients

@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_coefficients_follow_delta_sort_order(n):
    """
    Equation (4.4): for each column j,
        1 > a^{j+1}_j > ... > a^n_j > a^1_j > ... > a^{j-1}_j > 0
    which is what a^i_j = [j-i]_n / n is claimed to satisfy in 4.3.2.
    """
    game = ProductTwoActionGame(n)
    for j in range(n):
        rows = [(j + 1 + t) % n for t in range(n - 1)]
        values = [game.A[i, j] for i in rows]
        assert values == sorted(values, reverse=True), f"column {j}: {values}"
        assert 1.0 > values[0]
        assert values[-1] > 0.0


@pytest.mark.parametrize("n", [2, 3, 5, 7])
def test_diagonal_coefficients_are_zero(n):
    assert np.allclose(np.diag(ProductTwoActionGame(n).A), 0.0)


def test_coefficients_distinct_within_each_column():
    """Non-degeneracy condition from Definition 3.1: a^{i1}_j != a^{i2}_j."""
    n = 6
    game = ProductTwoActionGame(n)
    for j in range(n):
        column = [game.A[i, j] for i in range(n) if i != j]
        assert len(set(column)) == len(column)


def test_n_equals_5_matches_table_4_2_values():
    """The n = 5 utilities in 4.2.1 use 0.2, 0.4, 0.6, 0.8 as the off-diagonals."""
    game = ProductTwoActionGame(5)
    assert sorted(set(game.A.flatten())) == [0.0, 0.2, 0.4, 0.6, 0.8]


# -------------------------------------------------------------------- permutations

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_possible_equilibria_enumerates_every_permutation_once(n):
    perms = ProductTwoActionGame(n).possible_equilibria()
    assert len(perms) == factorial(n)
    assert len(set(perms)) == len(perms)
    assert set(perms) == set(itertools.permutations(range(1, n + 1)))


@pytest.mark.parametrize("n", [3, 4, 5])
def test_fixed_point_counts_are_binomial_times_derangements(n):
    """Permutations with exactly k fixed points number C(n,k) * !(n-k)."""
    game = ProductTwoActionGame(n)
    for k in range(n + 1):
        found = game.get_derangements_with_fixed_points(k)
        assert len(found) == comb(n, k) * subfactorial(n - k)
        for pi in found:
            assert sum(1 for i, v in enumerate(pi) if v == i + 1) == k


def test_upsteps_and_downsteps_partition_the_non_fixed_points():
    """Definition 4.1: D(pi) and U(pi) cover everything that is not fixed."""
    game = ProductTwoActionGame(6)
    for pi in game.possible_equilibria():
        up, down = game.count_upsteps(pi)
        fixed = sum(1 for i, v in enumerate(pi) if v == i + 1)
        assert up + down + fixed == 6


def test_permutation_parity_against_hand_computed_cases():
    game = ProductTwoActionGame(4)
    assert game.perm_parity((1, 2, 3, 4)) == 0   # identity
    assert game.perm_parity((2, 1, 3, 4)) == 1   # single transposition
    assert game.perm_parity((2, 1, 4, 3)) == 0   # two transpositions
    assert game.perm_parity((2, 3, 1, 4)) == 0   # 3-cycle, even


# ----------------------------------------------------------------------- matrices

def test_game_matrix_has_zero_diagonal():
    game = ProductTwoActionGame(4)
    M = game.game_matrix(np.array([0.1, 0.4, 0.6, 0.9]))
    assert np.allclose(np.diag(M), 0.0)


def test_parity_flags_exactly_the_negative_entries():
    game = ProductTwoActionGame(4)
    M = game.game_matrix(np.array([0.1, 0.4, 0.6, 0.9]))
    assert np.array_equal(game.parity(M), (M < 0).astype(int))


def test_extract_coefficients_picks_a_pi_of_j_by_column():
    game = ProductTwoActionGame(5)
    pi = (2, 3, 4, 5, 1)
    got = game.extract_coefficients(pi)
    expected = [game.A[pi[j] - 1, j] for j in range(5)]
    assert np.allclose(got, expected)


# ---------------------------------------------------------------------- Lemma 3.4

@pytest.mark.parametrize("n", [3, 4, 5])
def test_lemma_3_4a_all_pure_is_nash_iff_even_number_of_zeros(n):
    game = ProductTwoActionGame(n)
    for bits in itertools.product([0.0, 1.0], repeat=n):
        expected = sum(1 for v in bits if v == 0.0) % 2 == 0
        assert is_nash(game, np.array(bits)) is expected, f"{bits}"


@pytest.mark.parametrize("n", [3, 4, 5])
def test_lemma_3_4b_a_lone_mixer_among_pure_players_is_never_nash(n):
    """
    With every other player pure, Q_i is a product of (0 - a) and (1 - a) terms,
    none zero, so player i cannot be made indifferent and cannot mix.
    """
    game = ProductTwoActionGame(n)
    for i in range(n):
        for bits in itertools.product([0.0, 1.0], repeat=n - 1):
            x = np.array(list(bits[:i]) + [0.37] + list(bits[i:]))
            assert not is_nash(game, x)


def test_is_nash_rejects_a_deliberately_broken_profile():
    game = ProductTwoActionGame(4)
    all_ones = np.array([1.0, 1.0, 1.0, 1.0])
    assert is_nash(game, all_ones)
    broken = all_ones.copy()
    broken[0] = 0.0                      # now an odd number of zeros
    assert not is_nash(game, broken)


# ----------------------------------------------------------------------- headline

@pytest.mark.parametrize("n", sorted(KNOWN_BOUNDS))
def test_bound_is_attained(n):
    """
    equilibrium_count weights each representative by 2^(l-1) per Corollary 3.5
    and the total must equal equation (3.3).
    """
    game = ProductTwoActionGame(n)
    counted = quiet(game.equilibrium_count)
    assert counted == KNOWN_BOUNDS[n] == hertling_vujic_bound(n)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_every_profile_is_nash(n):
    """Every profile the module emits satisfies Theorem 3.3."""
    game = ProductTwoActionGame(n)
    profiles = quiet(game.equilibria)
    assert len(profiles) == factorial(n)
    failures = [x for x in profiles if not is_nash(game, x)]
    assert not failures, f"{len(failures)} of {len(profiles)} failed at n = {n}"


@pytest.mark.parametrize("n", [3, 4, 5])
def test_totally_mixed_equilibria_number_the_derangements(n):
    """Subsection 3.1.1: a TMNE corresponds to a derangement, so there are !n."""
    game = ProductTwoActionGame(n)
    profiles = quiet(game.equilibria)
    totally_mixed = [x for x in profiles if all(v != int(v) for v in x)]
    assert len(totally_mixed) == subfactorial(n)


@pytest.mark.xfail(
    strict=True,
    raises=NameError,
    reason=(
        "Known bug, left unfixed so the source file stays exactly as submitted. "
        "partition_equilibria calls print(game.A) instead of print(self.A), so it "
        "depends on a module-level variable named 'game' that only exists because "
        "the usage block at the bottom of the file creates one. Import the class "
        "anywhere else and it raises NameError. The dissertation appendix has "
        "print(self.A), so this is a regression in the working copy. One-character "
        "fix: game.A -> self.A on line 286."
    ),
)
@pytest.mark.parametrize("n", [3, 4, 5])
def test_partition_covers_every_profile(n):
    game = ProductTwoActionGame(n)
    parts = quiet(game.partition_equilibria)
    assert sum(len(v) for v in parts.values()) == factorial(n)


def test_two_counting_methods_agree():
    """
    equilibrium_count sums len(partitions) * 2^(l-1), so it counts what was
    enumerated. equilibrium_count_1 evaluates C(n,l) 2^(l-1) !(n-l) directly.
    They must agree, and if they ever diverge it is the enumeration that is
    wrong, since the second is just the formula.
    """
    for n in range(1, 8):
        game = ProductTwoActionGame(n)
        assert quiet(game.equilibrium_count) == quiet(game.equilibrium_count_1)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_lemma_3_4d_flipping_an_even_number_of_pure_actions_preserves_nash(n):
    """
    Lemma 3.4(d) and Corollary 3.5: starting from an equilibrium, flipping the
    actions of an even number of pure players gives another equilibrium, and
    flipping an odd number does not. This is what justifies the 2^(l-1) factor
    in the bound, so it is the load-bearing structural claim rather than a
    restatement of how the code builds its profiles.
    """
    game = ProductTwoActionGame(n)
    tested = 0

    for pi in game.possible_equilibria():
        x = game.extract_coefficients(pi)
        inc = (1 + x + np.sum(game.parity(game.game_matrix(x)), axis=1) % 2) % 2
        if 1.0 in [float(v) for v in inc if v == int(v)]:
            for i in range(len(x)):
                if x[i] == 0.0:
                    x[i] = 1.0
                    break
        if not is_nash(game, x):
            continue

        pure = [j for j in range(n) if x[j] == int(x[j])]
        for size in range(1, len(pure) + 1):
            for subset in itertools.combinations(pure, size):
                y = x.copy()
                for j in subset:
                    y[j] = 1.0 - y[j]
                assert is_nash(game, y) is (size % 2 == 0), (
                    f"n={n}, pi={pi}, flipped {subset}"
                )
                tested += 1

    assert tested > 0
