"""The HTTP layer: routes, the event stream, and the local-only guards."""

from __future__ import annotations

import http.client
import json
import socket
import threading

import pytest

from frontend import server as srv


@pytest.fixture
def running(app):
    server = srv.make_server(app, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def _port(server):
    return server.server_address[1]


def request(server, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", _port(server), timeout=60)
    h = {"Host": f"127.0.0.1:{_port(server)}", **(headers or {})}
    conn.request(method, path, body=body, headers=h)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp, data


def sse(data: bytes) -> list[tuple[str, dict]]:
    events = []
    for block in data.decode().strip().split("\n\n"):
        kind = payload = None
        for line in block.splitlines():
            if line.startswith("event: "):
                kind = line[len("event: "):]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
        events.append((kind, payload))
    return events


def chat(server, payload, **headers):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return request(server, "POST", "/api/chat", body,
                   {"Content-Type": "application/json", **headers})


def test_page_and_assets_are_served_with_csp(running):
    resp, body = request(running, "GET", "/")
    assert resp.status == 200 and resp.getheader("Content-Type").startswith("text/html")
    assert "default-src 'self'" in resp.getheader("Content-Security-Policy")
    assert b'src="/static/app.js"' in body and b'href="/static/app.css"' in body
    assert b'rel="icon" href="data:image/svg+xml' in body  # no /favicon.ico 404
    for path, ctype in (("/static/app.js", "text/javascript"), ("/static/app.css", "text/css")):
        resp, body = request(running, "GET", path)
        assert resp.status == 200 and resp.getheader("Content-Type").startswith(ctype) and body


@pytest.mark.parametrize("path", ["/static/../server.py", "/etc/passwd", "/static/nope.js",
                                  "/api/nope", "/frontend/app.py"])
def test_unknown_paths_404(running, path):
    assert request(running, "GET", path)[0].status == 404


def test_options_json(running):
    resp, body = request(running, "GET", "/api/options")
    assert resp.status == 200 and resp.getheader("Cache-Control") == "no-store"
    assert resp.getheader("Access-Control-Allow-Origin") is None
    assert len(json.loads(body)["records"]) == 44


def test_run_streams_events(running):
    resp, body = request(running, "GET", "/api/run?record=100&start=2&seconds=20&stay=101")
    assert resp.getheader("Content-Type").startswith("text/event-stream")
    events = sse(body)
    assert [k for k, _ in events] == ["stage"] * 9 + ["done"]
    assert events[-1][1]["case_id"] == events[8][1]["case_id"]
    bad = sse(request(running, "GET", "/api/run?record=100&seconds=abc")[1])
    assert bad == [("error", {"message": "start and seconds must be numbers"})]


@pytest.mark.parametrize("headers", [{"Host": "evil.example"}, {"Host": "127.0.0.1:1"},
                                     {"Sec-Fetch-Site": "cross-site"}])
def test_forbidden_requests_403(running, headers):
    assert request(running, "GET", "/api/options", headers=headers)[0].status == 403


def test_chat_roundtrip_and_limits(running, app):
    case_id = list(app.run({"stay": ["101"]}))[-1][1]["case_id"]
    resp, body = chat(running, {"case_id": case_id, "question": "What is a V beat?"})
    assert resp.status == 200 and "ollama serve" in json.loads(body)["answer"]
    assert chat(running, b"not json")[0].status == 400
    assert chat(running, {"case_id": case_id})[0].status == 400
    assert chat(running, {"case_id": "nope", "question": "hi"})[0].status == 404
    assert chat(running, b"x" * 5000)[0].status == 413
    plain = request(running, "POST", "/api/chat",
                    json.dumps({"case_id": case_id, "question": "hi"}).encode(),
                    {"Content-Type": "text/plain"})
    assert plain[0].status == 415


def test_outcome_endpoint(running):
    resp, body = request(running, "GET", "/api/outcome?stay=201")
    assert resp.status == 200 and json.loads(body)["outcome"] == "died in hospital"
    assert request(running, "GET", "/api/outcome?stay=999999")[0].status == 404
    assert request(running, "GET", "/api/outcome?stay=abc")[0].status == 400


def test_only_loopback_bind(app):
    with pytest.raises(ValueError, match="127.0.0.1"):
        srv.make_server(app, port=0, host="0.0.0.0")


def test_client_disconnect_mid_stream_keeps_server_alive(running):
    port = _port(running)
    with socket.create_connection(("127.0.0.1", port), timeout=30) as s:
        s.sendall(f"GET /api/run?record=100&start=2&seconds=20&stay=101 HTTP/1.1\r\n"
                  f"Host: 127.0.0.1:{port}\r\n\r\n".encode())
        assert s.recv(64).startswith(b"HTTP/1.0 200")
    # The lock must be free again: a full second run completes.
    events = sse(request(running, "GET", "/api/run?stay=101")[1])
    assert events[-1][0] == "done"
