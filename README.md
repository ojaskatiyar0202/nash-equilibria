# Maximal number of equilibria in product two-action games

Computational companion to my undergraduate dissertation, which reproves the
Hertling and Vujic (2024) result that their upper bound on the number of Nash
equilibria in n-player product two-action games is attained, arguing directly
via the best-response condition rather than through their increment function.

`Equilibrium_finder_1603.py` is the dissertation code, unmodified. The tests are
new and check it from the outside.

## The game

Each of `n` players has two actions. Under mixed profile `x` in `[0,1]^n`,

    u_i(x) = x_i * prod_{j != i} (x_j - a^i_j)

with `a^i_j` in `(0,1)`. Writing `Q_i(x_-i)` for the product, `u_i = x_i * Q_i`
is linear in `x_i`, so the best response depends only on the sign of `Q_i`:

    Q_i > 0  =>  x_i = 1
    Q_i < 0  =>  x_i = 0
    Q_i = 0  =>  x_i free in [0,1]

That is Theorem 3.3, and using it in place of the increment function is the main
simplification the dissertation makes.

## The bound

    !n + sum_{l=1}^{n} C(n, l) * 2^(l-1) * !(n-l)

with `!k` the derangements of `k` elements: choose which `l` players go pure,
`2^(l-1)` admissible pure profiles by Lemma 3.4(d), and `!(n-l)` derangements of
the remaining mixers. Every strategy profile corresponds to a permutation, with
mixing players at the deranged indices and pure players at the fixed points.

Whether the bound is attained depends on the parameters, and section 4.1.2 gives
an `n = 4` counterexample where it is not. The dissertation specifies them
explicitly as `a^i_j = [j-i]_n / n`, verifies this satisfies the delta sort order
of equation (4.4), and proves in Theorem 4.4 that it attains the bound. Whether
a permutation yields an equilibrium then reduces to the parity of its downstep
count, which is the link to permutation combinatorics that Corollary 4.3
establishes.

## Running it

```bash
pip install -r requirements.txt
python Equilibrium_finder_1603.py    # n = 5, as submitted
pytest tests/ -q                     # 50 passed, 3 xfailed
```

Change `n` at the bottom of the file to run other cases.

## What the tests check

Two claims that are not the same claim:

| Test | Claim |
|---|---|
| `test_bound_is_attained` | the count matches the closed form, `n = 1..8` |
| `test_every_profile_is_nash` | each profile really is an equilibrium, `n = 2..6` |

A generator can satisfy either while failing the other. Counting the right
number of things does not show the things are equilibria, and emitting only
genuine equilibria does not show that none were missed.

The Nash check is deliberately independent of the increment and parity machinery
the module uses to build its profiles: it computes `Q_i` from the definition and
compares against Theorem 3.3. All 5,040 profiles at `n = 7` pass.

Verified counts, matching equation (1.1):

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| equilibria | 1 | 3 | 9 | 37 | 185 | 1111 | 7777 | 62217 |

`n = 3` reproduces the nine equilibria of Table 4.1. `n = 5` gives 185.

Also checked: the coefficients satisfy the delta sort order of equation (4.4)
column by column; totally mixed equilibria number exactly `!n`, per subsection
3.1.1; Lemma 3.4(a), that an all-pure profile is an equilibrium precisely when it
has an even number of zeros, brute-forced over all of `{0,1}^n`; Lemma 3.4(b),
that a single mixer among otherwise pure players is never an equilibrium; and
Lemma 3.4(d), that flipping an even number of pure actions preserves equilibrium
while an odd number does not, over 257 flips for `n = 3,4,5`. That last one is
what justifies the `2^(l-1)` factor, so it carries weight.

## Scope

Enumeration walks all `n!` permutations, so runtime is factorial. `n <= 8` runs
in seconds. Coefficients are rounded to two decimals, which keeps permutation
recovery by value comparison exact as long as the spacing `1/n` exceeds the
rounding step, so `n <= 200`; the factorial wall arrives long before that.

The result covers product two-action games specifically, where utilities carry
that multiplicative structure. Dropping the product structure while keeping two
actions per player is open beyond `n = 3`, settled there by Jahani and von
Stengel (2022) via automated analysis.

## References

Hertling, C. and Vujic, M. (2024). Maximal number of mixed Nash equilibria in
generic games where each player has two pure strategies. arXiv:2412.17890.

von Stengel, B. (2022). *Game Theory Basics*. Cambridge University Press.

Jahani, S. and von Stengel, B. (2022). Automated equilibrium analysis of
2 x 2 x 2 games. In *Algorithmic Game Theory, SAGT 2022*, 223-237.
