"""The Ollama client talks to localhost only and fails politely."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentic import llm


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send(self.server.tags)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        self._send({"message": {"role": "assistant", "content": self.server.content}})

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_ollama():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.tags = {"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:latest"}]}
    server.content = '{"summary": "ok"}'
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield server
    server.shutdown()
    server.server_close()


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434/"])
def test_local_addresses_are_accepted(url):
    assert llm.check_local(url).startswith("http://")


@pytest.mark.parametrize("url", ["https://api.openai.com", "http://192.168.1.5:11434",
                                 "http://example.com:11434", "ftp://localhost:11434"])
def test_other_addresses_are_refused_before_any_request(url):
    with pytest.raises(ValueError, match="only a local Ollama"):
        llm.OllamaClient(base_url=url)


def test_status_when_nothing_listens():
    status = llm.OllamaClient(base_url=f"http://127.0.0.1:{_closed_port()}", timeout=2).status()
    assert not status.ok and "ollama serve" in status.message


def test_status_when_model_missing(fake_ollama):
    status = llm.OllamaClient("mistral", base_url=fake_ollama.url).status()
    assert not status.ok and "ollama pull mistral" in status.message


def test_status_accepts_the_latest_tag(fake_ollama):
    assert llm.OllamaClient("qwen2.5", base_url=fake_ollama.url).status().ok


def test_chat_json_sends_deterministic_settings_and_parses(fake_ollama):
    client = llm.OllamaClient(base_url=fake_ollama.url)
    schema = {"type": "object"}
    out = client.chat_json([{"role": "user", "content": "hi"}], schema)
    assert out == {"summary": "ok"}
    sent = fake_ollama.requests[0]
    assert sent["model"] == "llama3.1:8b" and sent["stream"] is False
    assert sent["format"] == schema and sent["options"] == {"temperature": 0, "seed": 0}


@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
def test_bad_reply_raises_llm_error(fake_ollama, content):
    fake_ollama.content = content
    with pytest.raises(llm.LlmError):
        llm.OllamaClient(base_url=fake_ollama.url).chat_json([], {})


@pytest.mark.requires_ollama
def test_real_ollama_answers_json():
    client = llm.OllamaClient()
    if not client.status().ok:
        pytest.skip(client.status().message)
    out = client.chat_json([{"role": "user", "content": 'Reply with {"ok": true}'}],
                           {"type": "object", "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"]})
    assert out == {"ok": True}
