import logging
import os
import threading
import uuid
from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import RecipeAgent
from config import Config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("recipe-chat")

AGENT = RecipeAgent(Config.from_env())

# Demo-grade conversation memory, keyed on the caller-supplied session_id.
MAX_SESSIONS = 500
SESSIONS: "OrderedDict[str, list[dict]]" = OrderedDict()
_sessions_lock = threading.Lock()

app = FastAPI(title="Recipe Chat Agent", version="1.0.0")

# Local only: when deployed, CORS belongs in Agent Manager — two sets of headers break browsers.
_cors_origins = [o for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if o]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


def _history(session_id: str) -> list[dict]:
    with _sessions_lock:
        history = SESSIONS.setdefault(session_id, [])
        SESSIONS.move_to_end(session_id)
        while len(SESSIONS) > MAX_SESSIONS:
            SESSIONS.popitem(last=False)
        return list(history)


def _remember(session_id: str, message: str, answer: str) -> None:
    with _sessions_lock:
        SESSIONS.setdefault(session_id, []).extend(
            [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    session_id = req.session_id or str(uuid.uuid4())
    try:
        answer = AGENT.reply(req.message, _history(session_id))
    except Exception as exc:
        # Upstream errors can carry gateway URLs and provider payloads — log, do not return.
        log.exception("agent invocation failed", extra={"session_id": session_id})
        raise HTTPException(status_code=500, detail="agent invocation failed") from exc

    _remember(session_id, req.message, answer)
    return ChatResponse(response=answer, session_id=session_id)
