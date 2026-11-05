"""A small client for the local Ollama server -- localhost only.

PERMISSIONS.md allows patient-derived findings to go to an LLM only because
Ollama runs on this machine. So the client refuses any address that is not
localhost before sending a byte, and ignores proxy environment variables (a
proxy would carry the request off the machine). It never pulls a model.

``chat_json`` asks for JSON that matches a schema (Ollama's structured output),
at temperature 0 with a fixed seed, so the same facts give the same draft.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

DEFAULT_MODEL = "llama3.1:8b"
COMPARISON_MODEL = "qwen2.5"
DEFAULT_URL = "http://localhost:11434"
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class LlmError(RuntimeError):
    """The LLM call failed; callers fall back to the template report."""


class LlmUnavailable(LlmError):
    """Nothing answered at the Ollama address."""


def check_local(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http" or parts.hostname not in LOCAL_HOSTS:
        raise ValueError(f"refusing LLM address {url!r}: only a local Ollama server "
                         f"(http://localhost) is allowed; see PERMISSIONS.md")
    return url.rstrip("/")


@dataclass(frozen=True)
class LlmStatus:
    ok: bool
    model: str
    message: str


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_URL,
                 timeout: float = 120.0, keep_alive: str = "15m") -> None:
        self.base_url = check_local(base_url)
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if data is None else "POST")
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:200]
            raise LlmError(f"Ollama answered HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LlmError(f"no answer from Ollama within {self.timeout:g} s") from exc
            raise LlmUnavailable(f"Ollama is not reachable at {self.base_url} ({exc.reason}); "
                                 f"start it with `ollama serve`") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise LlmError(f"no answer from Ollama within {self.timeout:g} s") from exc
        except json.JSONDecodeError as exc:
            raise LlmError(f"Ollama sent something that is not JSON: {exc}") from exc
        except OSError as exc:
            raise LlmUnavailable(f"Ollama is not reachable at {self.base_url} ({exc}); "
                                 f"start it with `ollama serve`") from exc

    def status(self) -> LlmStatus:
        try:
            tags = self._request("/api/tags")
        except LlmError as exc:
            return LlmStatus(False, self.model, str(exc))
        names = {m.get("name", "") for m in tags.get("models", [])}
        if self.model in names or f"{self.model}:latest" in names:
            return LlmStatus(True, self.model, "ready")
        return LlmStatus(False, self.model,
                         f"model {self.model} is not pulled. `ollama pull {self.model}` downloads "
                         f"several GB and needs the user's OK (PERMISSIONS.md)")

    def chat_json(self, messages: list[dict], schema: dict) -> dict:
        body = {"model": self.model, "messages": messages, "stream": False, "format": schema,
                "options": {"temperature": 0, "seed": 0}, "keep_alive": self.keep_alive}
        reply = self._request("/api/chat", body)
        content = (reply.get("message") or {}).get("content", "")
        try:
            out = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LlmError(f"the model's reply is not valid JSON: {content[:200]!r}") from exc
        if not isinstance(out, dict):
            raise LlmError(f"the model's reply is JSON but not an object: {content[:200]!r}")
        return out
