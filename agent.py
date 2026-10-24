"""RAG pipeline: route the message, retrieve matching recipes, answer from them.

The small local model only does what it is good at: deciding whether a message is about
cooking, rewriting follow-ups into standalone searches, and writing a short reply.
Filters come from filters.py, and the recipe list is rendered from the data, so names,
links and times are always exact.
"""

from __future__ import annotations

import json
import logging
import re

import filters
from config import Config
from search import RecipeStore

log = logging.getLogger("recipe-chat.agent")

OFF_TOPIC_REPLY = (
    "Sorry, I'm a recipe chatbot, so I can only help with cooking and recipes. "
    "Try asking for a dish by cuisine, course, ingredient, cooking method or diet, "
    'for example "a quick vegetarian Thai curry" or "what can I make with chickpeas?"'
)

GREETING_REPLY = (
    "Hi! I can find recipes from a collection of over 4,600. Tell me what you feel like: "
    "a cuisine, a course, a cooking method, a diet, how much time you have, or the "
    "ingredients you want to use."
)

NO_MATCH_REPLY = "I couldn't find a recipe for that. Try naming a dish, a cuisine or an ingredient."

ROUTER_PROMPT = """You are the router of a recipe chatbot. Read the latest user message and reply
with one JSON object: {"intent": ..., "query": ..., "ingredients": [...]}

intent: "recipes" if the message is about food, dishes, cooking, recipes, ingredients,
  kitchen techniques or meals; "greeting" for hello / what can you do; "off_topic" for
  anything else, including programming, maths, news, sport, travel, people and advice.
query: the food the user wants, as a standalone search. For follow-ups ("which of those is
  easiest?", "make it vegan") include what the earlier messages asked for.
ingredients: ingredients the user explicitly wants to cook with, lowercase. Empty if none.

Examples:
"Quick vegetarian Indian dinners" -> {"intent": "recipes", "query": "quick vegetarian Indian dinner", "ingredients": []}
"What can I make with leeks and potatoes?" -> {"intent": "recipes", "query": "leek and potato dishes", "ingredients": ["leeks", "potatoes"]}
"How long should I rest a steak?" -> {"intent": "recipes", "query": "steak", "ingredients": []}
"Write a Python function to reverse a string" -> {"intent": "off_topic", "query": "", "ingredients": []}
"Who won the election?" -> {"intent": "off_topic", "query": "", "ingredients": []}
"Hello!" -> {"intent": "greeting", "query": "", "ingredients": []}"""

TOPIC_PROMPT = """Decide whether a message belongs in a cooking chatbot. Answer with one word: yes or no.

yes: anything about food, drinks, dishes, recipes, ingredients, substitutions, nutrition,
cooking or baking techniques, kitchen equipment, meal ideas or food storage.
no: everything else, such as places, people, history, sport, travel, money, work,
technology, maths and jokes, even when the message mentions a country or a cuisine.

Examples:
What's the difference between baking soda and baking powder? -> yes
How long should I rest a steak? -> yes
Can I freeze cooked rice? -> yes
What wine goes with salmon? -> yes
What's the capital of France? -> no
Recommend a hotel in Rome -> no
Who won the world cup? -> no
Write a Python function -> no
Tell me a joke -> no"""

ANSWER_PROMPT = """You are Recipe Chat, a friendly cooking assistant. The recipes that best match the
user's request are listed below, and the app shows them to the user as a list right after
your reply. Write a short reply (1 to 3 sentences):
- For a recipe request, name one or two of the matches and say what makes them appealing
  (flavours, ingredients, style). Don't comment on times, diets or difficulty: the list
  shows those.
- For a cooking question, answer it briefly, then point to the matches if they are relevant.
- If the user asks how to make a dish, summarise the best match from its description and say
  the full ingredients and method are on its recipe page.
Write plain sentences only: no lists, no headings, no links. Never invent recipes.

RECIPES:
{recipes}"""

HISTORY_TURNS = 3  # user/assistant pairs passed back to the model


class RecipeAgent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = config.llm_client()
        self.store = RecipeStore(config.qdrant_url, config.collection)

    def reply(self, message: str, history: list[dict]) -> str:
        history = history[-2 * HISTORY_TURNS :]
        route = self._route(message, history)
        intent = route.get("intent", "recipes")
        if intent == "off_topic":
            return OFF_TOPIC_REPLY
        if intent == "greeting":
            return GREETING_REPLY

        query = route.get("query") if isinstance(route.get("query"), str) and route["query"].strip() else message
        criteria = filters.extract(f"{message} {query}", self.store.facets(), route.get("ingredients") or [])
        # A second, narrower check: the router alone lets through things like "capital of France?".
        # Naming a cuisine, course, method or diet ("and a vegan one?") is on topic by definition.
        on_topic_words = {k for k in criteria if k not in ("max_minutes", "max_calories")}
        if not on_topic_words and not self._on_topic(message, query):
            return OFF_TOPIC_REPLY
        log.info("search: %s", json.dumps({"query": query, "criteria": criteria}))
        recipes, relaxed = self.store.search(self._embed(f"search_query: {query}"), criteria)
        if not recipes:
            return NO_MATCH_REPLY
        if relaxed:
            # No model intro here: it tends to echo the request ("under 30 minutes") as if met.
            intro = f"Nothing matched everything you asked for, so these are the closest matches without {relaxed}:"
        else:
            intro = self._answer(message, history, recipes)
        return f"{intro}\n\n{render(recipes)}"

    def _on_topic(self, message: str, query: str) -> bool:
        # The router's standalone query gives context to short follow-ups.
        text = message if query == message else f"{message}\n(In context: {query})"
        res = self.llm.chat.completions.create(
            model=self.config.chat_model,
            temperature=0,
            max_tokens=3,
            messages=[{"role": "system", "content": TOPIC_PROMPT}, {"role": "user", "content": text}],
        )
        answer = (res.choices[0].message.content or "").strip().lower()
        log.info("on topic: %s", answer)
        # Only an explicit yes counts: a small model sometimes rambles instead of answering.
        return answer.startswith("yes")

    def _route(self, message: str, history: list[dict]) -> dict:
        res = self.llm.chat.completions.create(
            model=self.config.chat_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": ROUTER_PROMPT}, *history, {"role": "user", "content": message}],
        )
        try:
            route = json.loads(res.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            log.warning("router returned invalid JSON; treating as a recipe search")
            route = {}
        log.info("route: %s", json.dumps(route))
        return route if isinstance(route, dict) else {}

    def _embed(self, text: str) -> list[float]:
        res = self.llm.embeddings.create(model=self.config.embed_model, input=[text])
        return res.data[0].embedding

    def _answer(self, message: str, history: list[dict], recipes: list[dict]) -> str:
        blocks = [
            f"- {r['name']}: {r.get('cuisine')} {r.get('course')}, {r.get('method')}, {r.get('difficulty')}, "
            f"{r.get('totalTimeMinutes')} min. {r.get('description', '')}"
            for r in recipes
        ]
        system = ANSWER_PROMPT.format(recipes="\n".join(blocks))
        res = self.llm.chat.completions.create(
            model=self.config.chat_model,
            temperature=0.3,
            max_tokens=200,
            messages=[{"role": "system", "content": system}, *history, {"role": "user", "content": message}],
        )
        return prose((res.choices[0].message.content or "").strip())


def prose(text: str) -> str:
    """The model's reply up to its first list, which would duplicate the rendered one."""
    kept = []
    for line in text.splitlines():
        if re.match(r"\s*([-*•]|\d+[.)])\s", line):
            break
        kept.append(line)
    return "\n".join(kept).strip().removesuffix(":").strip()


def render(recipes: list[dict]) -> str:
    """The recipe list shown under the reply, straight from the data."""
    lines = []
    for r in recipes:
        diets = ", ".join([d for d in r.get("diets", []) if d != "Standard"] + r.get("restrictions", [])[:3])
        lines.append(
            f"- **[{r['name']}]({r['url']})**: {r.get('cuisine')} {r.get('course', '').lower()}, "
            f"{r.get('method', '').lower()}, {r.get('totalTimeMinutes')} min, {r.get('difficulty', '').lower()}"
            + (f" · {diets}" if diets else "")
        )
    lines.append("\n_Recipes from [Textual Chef](https://textualchef.com/)._")
    return "\n".join(lines)
