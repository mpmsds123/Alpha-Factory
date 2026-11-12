"""Method cards: the fixed list an idea or a change may cite as its evidence (Design Q2, Q5).

Adapted from a public research toolkit: an 8-question bank for feature engineering
(ideas) and a step 0-6 reasoning chain for single-signal refinement (changes). The
program checks that a cited card id is on this list; nothing else counts as a card.
"""

IDEA_CARDS: dict[str, str] = {
    "Q1": "What is stable? Invariants and stability measures of a field or combination.",
    "Q2": "What is changing? Rate of change, acceleration, volatility; trend versus noise.",
    "Q3": "What is anomalous? Outliers and breaks from normal; size of the deviation.",
    "Q4": "What is combined? How fields interact, amplify or offset each other.",
    "Q5": "What is structural? Constituent parts and proportions, and how they shift.",
    "Q6": "What is cumulative? Build-up over time, decay, memory and persistence.",
    "Q7": "What is relative? Position against peers, ranking, normalization.",
    "Q8": "What is essential? The first-principles meaning of the data.",
}

CHANGE_CARDS: dict[str, str] = {
    "S0": "Clean the signal first: extremes, long tails, missing values; map it to a "
    "standard shape (backfill, winsorize, rank, zscore).",
    "S1": "Is there signal over the whole range? If yes, only control the shape; if not, go to S2.",
    "S2": "Where does the signal sit: only the high end, only the low end, or both ends with "
    "noise in the middle (bucket, keep only the tails).",
    "S3": "Comparable across the whole market, or only within groups (industry, sector, "
    "size)? Compare within the group when levels differ by group.",
    "S4": "Today's level across stocks, or the stock against its own history (time-series "
    "change, zscore over its past)?",
    "S5": "Trade only when the signal is strong (trade_when), or keep exposure but size it "
    "in steps.",
    "S6": "Tradability and risk without changing the signal: decay, smoothing, turnover, "
    "concentration, sub-universe.",
}

# Published findings serve as priors for a mechanism, with a boundary on extrapolation.
DIRECTION_REFERENCES: dict[str, dict[str, str]] = {
    "earnings_underreaction": {
        "source": "Chan, Jegadeesh and Lakonishok, Momentum Strategies",
        "url": "https://www.nber.org/papers/w5375",
        "finding": "Earnings news and past returns predicted subsequent drift in the study.",
        "boundary": "An estimate level alone does not measure new information.",
    },
    "slow_information_diffusion": {
        "source": "Hong, Lim and Stein, Bad News Travels Slowly",
        "url": "https://www.nber.org/papers/w6553",
        "finding": "Momentum was stronger at low analyst coverage after controlling for size.",
        "boundary": "Coverage can condition the reaction to news. It is not a standalone sign.",
    },
    "liquidity_premium": {
        "source": "Amihud, Illiquidity and Stock Returns",
        "url": "https://www.sciencedirect.com/science/article/pii/S1386418101000246",
        "finding": "The study documents a positive association of illiquidity and returns.",
        "boundary": "A gross premium can reflect trading costs and may disappear after costs.",
    },
}


def is_card(card_id: str, cards: dict[str, str]) -> bool:
    return card_id.strip().upper() in cards
