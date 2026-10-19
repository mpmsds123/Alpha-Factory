"""Public, independently written research-question cards.

The identifiers are stable API values persisted by the evidence checker. The descriptions
are general research prompts, not formulas or imported prompt text.
"""

IDEA_CARDS: dict[str, str] = {
    "Q1": "Level: what persistent cross-sectional information does the field contain?",
    "Q2": "Change: does a recent change carry more information than the current level?",
    "Q3": "Surprise: is the useful quantity a deviation from the field's own history?",
    "Q4": "Interaction: is there a defensible mechanism for combining two quantities?",
    "Q5": "Composition: do ratios or component shares reveal a structural shift?",
    "Q6": "Persistence: how quickly should the observation decay or be smoothed?",
    "Q7": "Relative value: should the quantity be compared within an economic peer group?",
    "Q8": "Direction: why should high values outperform or underperform low values?",
}

CHANGE_CARDS: dict[str, str] = {
    "S0": "Data quality: handle missingness and extreme values without changing the thesis.",
    "S1": "Shape: decide whether level, rank, or standardized distance represents the thesis.",
    "S2": "Range: test whether the relationship is global or concentrated in a tail.",
    "S3": "Peer context: compare within a defensible group when levels are not comparable.",
    "S4": "Time context: compare the observation with its own recent history when appropriate.",
    "S5": "Activation: condition exposure only on an independently motivated state.",
    "S6": "Implementation: control turnover and concentration without inventing a new thesis.",
}


def is_card(card_id: str, cards: dict[str, str]) -> bool:
    return card_id.strip().upper() in cards
