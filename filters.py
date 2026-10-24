"""Deterministic search filters read from the user's words.

Every filterable value in the feed is a plain phrase ("Indian", "Slow Cooker", "Gluten-Free"),
so matching them in the text is more reliable than asking a small model to pick them, which
tends to invent filters the user never asked for.
"""

from __future__ import annotations

import re

# Extra phrases that mean a value, on top of the value itself and its plural.
SYNONYMS: dict[str, dict[str, list[str]]] = {
    "course": {
        "Dinner": ["supper", "main course", "main dish", "mains"],
        "Breakfast": ["brunch"],
        "Appetizer": ["starter", "starters", "snack", "snacks", "canape", "canapes"],
        "Side": ["side dish", "side dishes"],
        "Non-Alcoholic Drink": ["drink", "drinks", "smoothie", "smoothies", "mocktail", "mocktails"],
        "Alcoholic Drink": ["cocktail", "cocktails"],
    },
    "method": {
        "No-Cook": ["no bake", "no-bake", "no oven", "without an oven", "don't need an oven",
                    "doesn't need an oven", "without cooking", "no cooking"],
        "Grill": ["grilled", "bbq", "barbecue"],
        "Air Fryer": ["air fried", "air-fried", "airfryer"],
        "Pressure Cooker": ["instant pot", "instapot"],
        "Slow Cooker": ["crockpot", "crock pot", "crock-pot"],
    },
    "difficulty": {
        "Easy": ["easiest", "simple", "simplest", "beginner"],
        "Medium": ["intermediate"],
        "Hard": ["hardest", "challenging", "advanced"],
    },
    "diets": {
        "Vegetarian": ["veggie"],
        "Vegan": ["plant-based", "plant based"],
    },
}

# Values never matched from text: everyday words ("mixed greens", "standard size"), and
# Mediterranean, which is matched as a cuisine rather than a diet.
IGNORED = {"method": {"Mixed"}, "diets": {"Standard", "Mediterranean"}}

TIME_RE = re.compile(
    r"\b(?:under|less than|within|in|max(?:imum)?|up to|no more than|at most|below)\s+"
    r"(an?|\d+)\s*(min(?:ute)?s?|hours?|hrs?)\b"
)
CALORIE_RE = re.compile(r"\b(?:under|less than|below|max(?:imum)?|up to|at most)\s+(\d+)\s*(?:cal|kcal|calories)\b")
QUICK_RE = re.compile(r"\b(quick|quickly|fast|speedy)\b")
QUICK_MINUTES = 30


def _norm(text: str) -> str:
    return " " + re.sub(r"[\s\-_/]+", " ", text.lower()).strip() + " "


def _phrases(field: str, value: str) -> list[str]:
    base = value.lower()
    out = [base]
    if not base.endswith("s"):
        out.append(base + "s")
    out += SYNONYMS.get(field, {}).get(value, [])
    return [_norm(p) for p in out]


def _match(text: str, field: str, values: list[str]) -> list[str]:
    """Values of a field mentioned in the (normalised) text, longest phrases first, so
    "Italian-American" wins over "Italian" and "Southern American" over "Southern"."""
    values = [v for v in values if v not in IGNORED.get(field, set())]
    candidates = sorted(((p, v) for v in values for p in _phrases(field, v)), key=lambda c: -len(c[0]))
    found: list[str] = []
    for phrase, value in candidates:
        if phrase in text:
            text = text.replace(phrase, " ")
            if value not in found:
                found.append(value)
    return found


def extract(text: str, facets: dict[str, list[str]], ingredients: list[str]) -> dict:
    """Search criteria for the text. `ingredients` (from the model) are kept only when
    they actually appear in the text."""
    t = _norm(text)
    criteria: dict = {}
    for field in ("cuisine", "course", "method", "difficulty"):
        if found := _match(t, field, facets[field]):
            criteria[field] = found[0]
    if found := _match(t, "diets", facets["diets"] + facets["restrictions"]):
        criteria["diets"] = found

    grounded = [i.strip().lower() for i in ingredients if isinstance(i, str) and i.strip() and _norm(i) in t]
    if grounded:
        criteria["ingredients"] = grounded[:4]

    plain = text.lower()
    if m := TIME_RE.search(plain):
        amount = 1 if m.group(1) in ("a", "an") else int(m.group(1))
        criteria["max_minutes"] = amount * 60 if m.group(2).startswith(("hour", "hr")) else amount
    elif QUICK_RE.search(plain):
        criteria["max_minutes"] = QUICK_MINUTES
    if m := CALORIE_RE.search(plain):
        criteria["max_calories"] = int(m.group(1))
    return criteria
