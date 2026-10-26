import base64
import json
import logging
import os
import threading
import uuid
from collections import OrderedDict

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent import RecipeAgent
from config import Config
from limits import UsageLimiter

try:  # Injected by Agent Manager's auto-instrumentation; absent in local runs.
    from traceloop.sdk import Traceloop
except ImportError:
    Traceloop = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("recipe-chat")

AGENT = RecipeAgent(Config.from_env())
LIMITER = UsageLimiter()

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
    usage: dict


def _user(authorization: str | None) -> tuple[str, str]:
    """(user id, display name) from the ThunderID token the gateway forwards.

    The gateway has already validated the token (signature, issuer, expiry) before the
    request reaches the agent, so the claims are only decoded here, not re-verified.
    Requests without a token (local runs) share the "anonymous" user.
    """
    try:
        payload = (authorization or "").split()[-1].split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        user_id = claims["sub"]
        name = claims.get("username") or claims.get("given_name") or claims.get("name") or user_id
        return user_id, name
    except (IndexError, KeyError, ValueError):
        return "anonymous", "anonymous"


def _tag_trace(user_id: str, name: str, session_id: str) -> None:
    """Adds the user and session to every span of this request, for per-user views."""
    if Traceloop is None:
        return
    try:
        Traceloop.set_association_properties({"user_id": user_id, "user_name": name, "session_id": session_id})
    except Exception:  # noqa: BLE001 - tracing must never break a request
        log.debug("could not tag trace", exc_info=True)


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
def chat(
    req: ChatRequest,
    authorization: str | None = Header(default=None),
    # The gateway's jwt-auth policy (forwardToken: true) moves the validated token to this
    # header and removes Authorization.
    x_forwarded_authorization: str | None = Header(default=None),
):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    session_id = req.session_id or str(uuid.uuid4())
    user_id, name = _user(x_forwarded_authorization or authorization)

    over = LIMITER.start(user_id)
    if over is not None:
        log.info("usage limit: user=%s name=%s blocked for %ss", user_id, name, over["resetsInSeconds"])
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(over["resetsInSeconds"])},
            content={"detail": f"You've reached your usage limit. Try again in {over['resetsInSeconds']} seconds.",
                     "usage": over},
        )

    _tag_trace(user_id, name, session_id)
    try:
        answer, tokens = AGENT.reply_counted(req.message, _history(session_id))
    except Exception as exc:
        # Upstream errors can carry gateway URLs and provider payloads — log, do not return.
        log.exception("agent invocation failed", extra={"session_id": session_id})
        raise HTTPException(status_code=500, detail="agent invocation failed") from exc

    usage = LIMITER.finish(user_id, tokens)
    log.info("usage: user=%s name=%s tokens=%d requests=%d/%d tokens_total=%d/%d", user_id, name, tokens,
             usage["requests"]["used"], usage["requests"]["limit"], usage["tokens"]["used"], usage["tokens"]["limit"])
    _remember(session_id, req.message, answer)
    return ChatResponse(response=answer, session_id=session_id, usage=usage)
