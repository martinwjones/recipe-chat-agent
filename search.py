"""Recipe retrieval from Qdrant: vector search narrowed by metadata filters."""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger("recipe-chat.search")

# Payload fields the router may filter on, with the values it may use (loaded from Qdrant).
FACET_FIELDS = ["cuisine", "course", "method", "difficulty", "diets", "restrictions"]

# How each criterion is described when a search has to drop it.
LABELS = {
    "cuisine": "the cuisine",
    "course": "the course",
    "method": "the cooking method",
    "difficulty": "the difficulty",
    "diets": "the diet",
    "ingredients": "the ingredient match",
    "max_minutes": "the time limit",
    "max_calories": "the calorie limit",
}


class RecipeStore:
    def __init__(self, qdrant_url: str, collection: str) -> None:
        self._http = httpx.Client(base_url=f"{qdrant_url}/collections/{collection}", timeout=30)
        self._facets: dict[str, list[str]] = {}

    def facets(self) -> dict[str, list[str]]:
        """Distinct values per filterable field, e.g. every cuisine in the collection."""
        if not self._facets:
            for field in FACET_FIELDS:
                res = self._http.post("/facet", json={"key": field, "limit": 500, "exact": True})
                res.raise_for_status()
                self._facets[field] = sorted(h["value"] for h in res.json()["result"]["hits"])
        return self._facets

    def search(self, vector: list[float], criteria: dict, limit: int = 5) -> tuple[list[dict], str | None]:
        """Returns (recipes, relaxed). When nothing matches every criterion, criteria are dropped
        in stages and `relaxed` says which, so the answer can be honest about near misses."""
        stages = [
            criteria,
            {k: v for k, v in criteria.items() if k != "ingredients"},
            {k: v for k, v in criteria.items() if k in ("cuisine", "course", "diets")},
            {},
        ]
        seen: set[str] = set()
        for crit in stages:
            key = repr(sorted(crit.items()))
            if key in seen:
                continue
            seen.add(key)
            # The ingredient match is best-effort (the feed only has descriptions), so dropping
            # it alone is not worth telling the user about.
            dropped = [LABELS[k] for k in criteria if k not in crit and k != "ingredients"]
            relaxed = None if not dropped else ", ".join(dropped[:-1]) + (" and " if len(dropped) > 1 else "") + dropped[-1]
            body: dict = {"query": vector, "limit": limit, "with_payload": True}
            flt = self._filter(crit)
            if flt:
                body["filter"] = flt
            res = self._http.post("/points/query", json=body)
            res.raise_for_status()
            points = res.json()["result"]["points"]
            if points:
                return [p["payload"] for p in points], relaxed
        return [], None

    @staticmethod
    def _filter(c: dict) -> dict | None:
        must: list[dict] = []
        for field in ("cuisine", "course", "method", "difficulty"):
            if c.get(field):
                must.append({"key": field, "match": {"value": c[field]}})
        for value in c.get("diets", []):
            # A diet like "Vegan" lives in diets, a restriction like "Gluten-Free" in restrictions.
            must.append({"should": [
                {"key": "diets", "match": {"value": value}},
                {"key": "restrictions", "match": {"value": value}},
            ]})
        for ingredient in c.get("ingredients", []):
            must.append({"key": "ingredientsText", "match": {"text": ingredient}})
        if c.get("max_minutes"):
            must.append({"key": "totalTimeMinutes", "range": {"lte": c["max_minutes"]}})
        if c.get("max_calories"):
            must.append({"key": "caloriesPerServing", "range": {"lte": c["max_calories"]}})
        return {"must": must} if must else None
