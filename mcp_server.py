"""Recipe catalog MCP server. AM run command: ``python mcp_server.py``.

Exposes the Qdrant recipe collection as MCP tools over streamable HTTP at /mcp. Search is
semantic (query embedded with nomic-embed-text through the AI Gateway, like the chat agent)
with exact metadata filters on top, using the chat agent's search.py. Agent Manager deploys
it as the custom-api component "recipe-mcp" and its gateway adds API-key authentication.
"""

from __future__ import annotations

import os
import uuid

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from config import Config
from search import RecipeStore

ID_NAMESPACE = uuid.UUID("5b3c1f0e-7d2a-4c1e-9a55-2f3e6b8d0c11")  # same as ingest.py
MAX_RESULTS = 20

CONFIG = Config.from_env()
llm = CONFIG.llm_client()
store = RecipeStore(CONFIG.qdrant_url, CONFIG.collection)
qdrant = httpx.Client(base_url=f"{CONFIG.qdrant_url}/collections/{CONFIG.collection}", timeout=30)

mcp = FastMCP(
    "recipes",
    instructions=(
        "Search a catalog of about 4,700 recipes from Textual Chef by free text and by "
        "cuisine, course, cooking method, difficulty, diet, time, calories and ingredients. "
        "Call list_filter_values first to get the exact filter values. Recipes link to their "
        "page on textualchef.com, which has the full ingredients and method; cite those links."
    ),
    host="0.0.0.0",
    # Platform-hosted components are routed to 8000 (deploy settings set PORT to match).
    port=int(os.environ.get("PORT", 8000)),
    stateless_http=True,
    json_response=True,
    # Reached through the Agent Manager gateway, which forwards an in-cluster Host header
    # that DNS-rebinding protection would reject.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


def _summary(r: dict) -> dict:
    keys = ["id", "name", "url", "description", "cuisine", "course", "method", "difficulty",
            "mainIngredient", "totalTimeMinutes", "servings", "caloriesPerServing", "diets", "restrictions"]
    return {k: r.get(k) for k in keys}


def _pick(value: str | None, allowed: list[str], field: str) -> str | None:
    if not value:
        return None
    match = next((a for a in allowed if a.lower() == value.strip().lower()), None)
    if match is None:
        raise ValueError(f"Unknown {field} {value!r}. Call list_filter_values for the valid values.")
    return match


def _embed(text: str) -> list[float]:
    return llm.embeddings.create(model=CONFIG.embed_model, input=[text]).data[0].embedding


@mcp.tool()
def list_filter_values() -> dict[str, list[str]]:
    """The exact values accepted by search_recipes' filters: cuisines, courses, cooking
    methods, difficulties, and diets (diets include restrictions such as Gluten-Free)."""
    f = store.facets()
    return {
        "cuisine": f["cuisine"],
        "course": f["course"],
        "method": f["method"],
        "difficulty": f["difficulty"],
        "diets": f["diets"] + f["restrictions"],
    }


@mcp.tool()
def search_recipes(
    query: str = "",
    cuisine: str | None = None,
    course: str | None = None,
    method: str | None = None,
    difficulty: str | None = None,
    diets: list[str] | None = None,
    ingredients: list[str] | None = None,
    max_minutes: int | None = None,
    max_calories: int | None = None,
    limit: int = 5,
) -> dict:
    """Find recipes. `query` is free text describing the dish ("something warming for
    winter", "crispy tofu"); the filters are exact. All filters are optional and combined with
    AND. `ingredients` match the recipe's main ingredient, name and description (the catalog
    has no full ingredient lists). If nothing matches every filter, filters are dropped in
    stages and `relaxed` says which. Returns up to `limit` (max 20) recipes."""
    f = store.facets()
    criteria: dict = {}
    for field, value in (("cuisine", cuisine), ("course", course), ("method", method), ("difficulty", difficulty)):
        if picked := _pick(value, f[field], field):
            criteria[field] = picked
    if diets:
        criteria["diets"] = [_pick(d, f["diets"] + f["restrictions"], "diet") for d in diets]
    if ingredients:
        criteria["ingredients"] = [i.strip().lower() for i in ingredients if i.strip()][:4]
    if max_minutes:
        criteria["max_minutes"] = max_minutes
    if max_calories:
        criteria["max_calories"] = max_calories

    text = query.strip() or " ".join(str(v) for v in criteria.values()) or "recipe"
    recipes, relaxed = store.search(_embed(f"search_query: {text}"), criteria, max(1, min(limit, MAX_RESULTS)))
    return {"relaxed": relaxed, "count": len(recipes), "recipes": [_summary(r) for r in recipes]}


@mcp.tool()
def get_recipe(recipe_id: str) -> dict:
    """One recipe's details by its id (e.g. "french-onion-soup", from search_recipes), or by
    name if no recipe has that id. The full ingredients and method are on its `url`."""
    res = qdrant.get(f"/points/{uuid.uuid5(ID_NAMESPACE, recipe_id.strip())}")
    if res.status_code == 200:
        p = res.json()["result"]["payload"]
        return {**_summary(p), "tags": p.get("tags"), "prepTimeMinutes": p.get("prepTimeMinutes"),
                "cookTimeMinutes": p.get("cookTimeMinutes")}
    recipes, _ = store.search(_embed(f"search_query: {recipe_id}"), {}, 1)
    if not recipes:
        raise ValueError(f"No recipe found for {recipe_id!r}")
    return get_recipe(recipes[0]["id"])


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
