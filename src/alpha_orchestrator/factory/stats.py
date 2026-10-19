"""The statistics every selection decision uses. Pure functions, no I/O.

Noise model: the IS Sharpe of an expression with no information is roughly N(0, sigma)
with sigma ~ 1/sqrt(years of IS); 0.40 was measured on this ledger's own idea runs (a
median-based estimate, so a few real signals do not inflate it).

    exceed_prob     chance noise reaches |Sharpe| >= line (one formula, or best of k)
    Sprt            "does this dataset carry more signal than noise?", decided as soon as
                    the evidence is enough, with both error rates fixed in advance
    shrunk_mean     per-stratum signal concentration, pulled toward the dataset mean
                    when the stratum has few observations (empirical Bayes)
    thompson_pick   allocate the next batch: draw a plausible value per arm, take the best
    luck_bar        best Sharpe pure noise reaches after n tries: sigma * sqrt(2 ln n)
"""

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

_N = NormalDist()


def exceed_prob(line: float, sigma: float, k: int = 1) -> float:
    """P(max over k independent noise formulas of |Sharpe| >= line)."""
    one = 2 * (1 - _N.cdf(line / sigma))
    return 1 - (1 - one) ** k


def expected_best_abs(sigma: float, k: int, draws: int = 20000, seed: int = 7) -> float:
    """E[max over k of |N(0, sigma)|], by simulation (k is small)."""
    rng = random.Random(seed)
    total = 0.0
    for _ in range(draws):
        total += max(abs(rng.gauss(0, sigma)) for _ in range(k))
    return total / draws


def luck_bar(sigma: float, tries: int) -> float:
    """Best |Sharpe| pure noise is expected to reach after `tries` independent tries."""
    return sigma * math.sqrt(2 * math.log(max(tries, 2)))


def binomial_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


@dataclass
class Sprt:
    """Sequential probability ratio test on hit / miss observations.

    H0: hit rate = p0 (what noise gives). H1: hit rate = p1 (a dataset worth working).
    alpha = P(call noise a signal), beta = P(miss a real signal)."""

    p0: float
    p1: float
    alpha: float = 0.05
    beta: float = 0.20
    llr: float = 0.0
    n: int = 0
    hits: int = 0

    @property
    def upper(self) -> float:
        return math.log((1 - self.beta) / self.alpha)

    @property
    def lower(self) -> float:
        return math.log(self.beta / (1 - self.alpha))

    def add(self, hit: bool) -> str:
        self.n += 1
        self.hits += hit
        self.llr += math.log(self.p1 / self.p0) if hit else math.log((1 - self.p1) / (1 - self.p0))
        return self.verdict

    @property
    def verdict(self) -> str:
        if self.llr >= self.upper:
            return "signal"
        if self.llr <= self.lower:
            return "noise"
        return "undecided"


@dataclass(frozen=True)
class Arm:
    mean: float
    sd: float


def shrunk_means(
    groups: Mapping[str, Sequence[float]], prior_weight: float = 2.0, noise_sd: float = 0.3
) -> dict[str, Arm]:
    """Per-group mean of excess scores, shrunk toward the pooled mean; sd of that mean."""
    pooled = [x for xs in groups.values() for x in xs]
    grand = sum(pooled) / len(pooled) if pooled else 0.0
    out = {}
    for name, xs in groups.items():
        n = len(xs)
        mean = (sum(xs) + prior_weight * grand) / (n + prior_weight)
        out[name] = Arm(mean, noise_sd / math.sqrt(n + prior_weight))
    return out


def thompson_pick(arms: Mapping[str, Arm], rng: random.Random, k: int = 1) -> list[str]:
    """The k arms with the best plausible value this draw."""
    draws = sorted(((rng.gauss(a.mean, a.sd), name) for name, a in arms.items()), reverse=True)
    return [name for _, name in draws[:k]]


def beta_draw(successes: float, failures: float, rng: random.Random) -> float:
    return rng.betavariate(1 + successes, 1 + failures)
