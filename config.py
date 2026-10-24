"""Instance configuration, read from env at startup."""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class Config:
    llm_base_url: str
    llm_api_key: str
    # True when the URL is Agent Manager's AI Gateway, which wants the key in an API-Key header.
    via_gateway: bool
    chat_model: str
    embed_model: str
    qdrant_url: str
    collection: str

    @classmethod
    def from_env(cls) -> "Config":
        # OPENAI_URL / OPENAI_API_KEY are injected when an LLM provider is bound to the agent
        # in Agent Manager. OPENAI_BASE_URL is for local runs straight against Ollama.
        gateway_url = os.environ.get("OPENAI_URL", "")
        return cls(
            llm_base_url=gateway_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            llm_api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
            via_gateway=bool(gateway_url),
            chat_model=os.environ.get("CHAT_MODEL", "llama3.2"),
            embed_model=os.environ.get("EMBED_MODEL", "nomic-embed-text"),
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/"),
            collection=os.environ.get("COLLECTION", "recipes"),
        )

    def llm_client(self) -> OpenAI:
        if self.via_gateway:
            # The AI Gateway authenticates with an API-Key header. The key is also passed as
            # api_key because openai>=3 refuses to start with an empty one; the gateway ignores
            # the resulting bearer header.
            return OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key,
                          default_headers={"API-Key": self.llm_api_key})
        return OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)
