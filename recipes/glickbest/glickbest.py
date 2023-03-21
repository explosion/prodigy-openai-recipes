import numpy as np
import numpy.typing as npt

"""
Implement the "GlickBest" algorithm to find the best player/arm/option,
using only noisy pairwise comparisons. This is mainly intended for A/B/C testing
and eliciting human preferences.

GlickBest keeps a Glicko 1 rating per player, on each round it computes the probability that each
player has the highest rating and matches up the top two.

This is related to (but different from) a dueling bandits problem.
The bandits problem has an explore-exploit trade-off. This is a pure exploration approach.

Due to the use of a Glicko rating, this assumes skill/goodness is mostly transitive.

N.B. because this is meant for comparison of static options, the RDs are not increased
per round, that is RDs can only decrease after matchups.

using functions `next_match()` and `update()`
"""

_q = np.log(10.) / 400.

def _g(RD: npt.ArrayLike) -> npt.ArrayLike:
    return 1. / np.sqrt(1. + 3.*(_q**2)*(RD**2) / np.pi**2)

def _E(r: npt.ArrayLike, r_j: npt.ArrayLike, RD_j: npt.ArrayLike) -> npt.ArrayLike:
    return 1. / (1. + np.power(10, -_g(RD_j) * (r - r_j) / 400.))

def glicko_update(
        r: npt.ArrayLike,
        RD: npt.ArrayLike,
        s_j: npt.ArrayLike,
        r_j: npt.ArrayLike,
        RD_j: npt.ArrayLike
    ) -> npt.ArrayLike:
    """
    Glicko 1 rating update.
    http://www.glicko.net/glicko/glicko.pdf

    :param r: current player rating
    :param RD: current player rating deviation
    :param s_j: (N) outcome vector for player being updated; 0 for loss, 1 for win, 0.5 for draw
    :param r_j: (N) opponent ratings vector
    :param RD_j: (N) opponent rating deviations vector
    :returns: (rating update size, new ratings deviation)
    """
    ps = _E(r, r_j, RD_j)
    d2 = 1. / ((_q**2) * np.sum((_g(RD_j)**2) * ps * (1. - ps)))
    r_delta = (_q / (1./RD**2 + 1./d2)) * np.sum(_g(RD_j)*(s_j - ps))
    RD_prime = np.sqrt(1. / (1./RD**2 + 1./d2))
    return r_delta, RD_prime

def rank1_probs(ratings: npt.ArrayLike, num_samples=4000) -> npt.ArrayLike:
    """
    Estimate the probability of having the highest rating.

    :param ratings: (K, 2) ratings with RD, first column is ratings, second column is ratings deviation.
    :param num_samples: number of samples to draw
    :returns: (K) p[i] such that p[i] := P(rank(i) = 1 | ratings)
    """
    assert ratings.ndim == 2
    assert ratings.shape[1] == 2
    assert num_samples > 0
    num_players = ratings.shape[0]
    loc = ratings[:,0,np.newaxis]
    scale = ratings[:,1,np.newaxis]
    samples = np.random.normal(loc=loc, scale=scale, size=(num_players, num_samples))
    counts = np.bincount(np.argmax(samples, axis=0), minlength=num_players)
    p = counts / num_samples
    return p

def top_k(x: npt.ArrayLike, k=2) -> npt.ArrayLike:
    """
    Return indices of top k values.

    :param x: input values
    :param k: number of indices to return
    :returns: (k) list of indices into :x:
    """
    return np.argsort(x)[::-1][:k]

def win_prob_bradley_terry(theta_1: npt.ArrayLike, theta_2: npt.ArrayLike, base=10, scale=400):
    """
    Compute the bradley terry win probability given two known ratings

    :param theta_1: known rating 1
    :param theta_2: known rating 2
    :returns: p(s = 1 | theta_1, theta_2)
              prob. player with rating theta_1 beats player with rating theta_2
    """
    odds = base**((theta_1 - theta_2)/scale)
    p = odds / (1 + odds)
    return p

def update(ratings: npt.ArrayLike, i: int, j: int, s: float) -> None:
    """
    update ratings array after matchup

    :param ratings: (K, 2) ratings table, first column rating, second column ratings deviation (RD)
    :param i: index of first player
    :param j: index of second player
    :param s: match outcome, player :i: wins = 1, player :i: loses = 0, draw = 0.5
    """
    r_1, RD_1 = ratings[i]
    r_2, RD_2 = ratings[j]
    r_delta_1, RD_prime_1 = glicko_update(r_1, RD_1, s, r_2, RD_2)
    r_delta_2, RD_prime_2 = glicko_update(r_2, RD_2, 1. - s, r_1, RD_1)
    ratings[i, 0] += r_delta_1
    ratings[i, 1] =  RD_prime_1
    ratings[j, 0] += r_delta_2
    ratings[j, 1] =  RD_prime_2

def next_match(ratings: npt.ArrayLike) -> tuple[int, int]:
    """
    pick next matchup

    :param ratings: (K, 2) ratings table, first column rating, second column ratings deviation (RD)
    """
    p_best = rank1_probs(ratings)
    i, j = top_k(p_best, k=2)
    return i, j


def test_glicko_known_values():
    # Known values taken from Glicko paper
    assert np.round(_g(30), 4)  == 0.9955
    assert np.round(_g(100), 4) == 0.9531
    assert np.round(_g(300), 4) == 0.7242
    assert np.round(_E(1500, 1400, 30), 3)  == 0.639
    assert np.round(_E(1500, 1550, 100), 3) == 0.432
    assert np.round(_E(1500, 1700, 300), 3) == 0.303
    r_j  = np.array([1400, 1550, 1700])
    RD_j = np.array([30, 100, 300])
    s_j  = np.array([1, 0, 0])
    r_delta, RD_prime = glicko_update(r=1500, RD=200, s_j=s_j, r_j=r_j, RD_j=RD_j)
    assert np.round(1500 + r_delta) == 1464
    assert np.round(RD_prime, 1) == 151.4
