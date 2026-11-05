#!/usr/bin/env python3
"""Serve the agent pipeline page on this machine only.

    python frontend/server.py --open          # http://127.0.0.1:8765 in your browser
    python frontend/server.py --llm off       # no LLM: template reports, chat off
    python frontend/server.py --port 8766     # if 8765 is busy

Pick an ECG record and an ICU stay, press Run, and watch the nine stages fill
in. The server listens on 127.0.0.1 only and the page loads nothing from the
internet. Patient values stay on this machine. A research demo, not for
patient care.

Routes: GET /  /static/app.js  /static/app.css  /api/options
        GET /api/run?record=&start=&seconds=&stay=&peaks=   (server-sent events)
        GET /api/outcome?stay=      POST /api/chat {case_id, question}
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LOCAL_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/static/app.css": ("app.css", "text/css; charset=utf-8"),
}
MAX_BODY = 4096
CSP = ("default-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
#: Sec-Fetch-Site values a browser sends for this page's own requests.
SAME_ORIGIN = {"", "same-origin", "none"}


def _json_default(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def _dump(obj) -> bytes:
    return json.dumps(obj, default=_json_default, allow_nan=False).encode()


class Handler(BaseHTTPRequestHandler):
    server_version = "AgentDemo/1"

    def log_message(self, fmt, *args):  # quiet: request lines carry stay ids
        pass

    @property
    def app(self):
        return self.server.app

    # -- helpers -------------------------------------------------------------

    def _allowed(self) -> bool:
        """Only this page, on this machine: the Host header must name the
        server (stops DNS rebinding) and browsers must not label the request
        cross-site (stops other pages driving it)."""
        return (self.headers.get("Host", "") in self.server.allowed_hosts
                and self.headers.get("Sec-Fetch-Site", "") in SAME_ORIGIN)

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj) -> None:
        self._send(status, _dump(obj), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    # -- GET -----------------------------------------------------------------

    def do_GET(self):
        if not self._allowed():
            return self._error(403, "forbidden")
        url = urlsplit(self.path)
        if url.path in STATIC:
            name, ctype = STATIC[url.path]
            return self._send(200, (STATIC_DIR / name).read_bytes(), ctype,
                              {"Content-Security-Policy": CSP})
        if url.path == "/api/options":
            return self._json(200, self.app.options())
        if url.path == "/api/run":
            return self._stream(parse_qs(url.query))
        if url.path == "/api/outcome":
            return self._outcome(parse_qs(url.query))
        self._error(404, "not found")

    def _outcome(self, query) -> None:
        try:
            self._json(200, self.app.outcome(int((query.get("stay") or [""])[0])))
        except ValueError:
            self._error(400, "stay must be a number")
        except KeyError as exc:
            self._error(404, str(exc.args[0]) if exc.args else "not found")

    def _stream(self, query) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.close_connection = True
        events = self.app.run(query)
        try:
            for kind, data in events:
                self.wfile.write(b"event: " + kind.encode() + b"\ndata: " + _dump(data) + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page went away; closing the generator below frees the lock
        except Exception as exc:  # last resort: tell the page rather than drop the stream
            message = f"internal error: {type(exc).__name__}: {exc}"
            try:
                self.wfile.write(b"event: error\ndata: " + _dump({"message": message}) + b"\n\n")
            except OSError:
                pass
        finally:
            events.close()

    # -- POST ----------------------------------------------------------------

    def do_POST(self):
        if not self._allowed():
            return self._error(403, "forbidden")
        if urlsplit(self.path).path != "/api/chat":
            return self._error(404, "not found")
        self.close_connection = True
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._error(400, "bad Content-Length")
        if length > MAX_BODY:
            return self._error(413, "the question is too long")
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self._error(415, "send JSON")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._error(400, "the body must be JSON")
        if not (isinstance(body, dict) and isinstance(body.get("case_id"), str)
                and isinstance(body.get("question"), str)):
            return self._error(400, "send {case_id, question}")
        try:
            self._json(200, self.app.chat(body["case_id"], body["question"]))
        except KeyError as exc:
            self._error(404, str(exc.args[0]) if exc.args else "not found")


def make_server(app, port: int = DEFAULT_PORT, host: str = LOCAL_HOST) -> ThreadingHTTPServer:
    if host != LOCAL_HOST:
        raise ValueError(f"the page only listens on {LOCAL_HOST}; see PERMISSIONS.md")
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.app = app
    actual = server.server_address[1]
    server.allowed_hosts = {f"127.0.0.1:{actual}", f"localhost:{actual}"}
    server.url = f"http://127.0.0.1:{actual}/"
    return server


def _warm_up(client) -> None:
    """Load the model now so the first report is not the slow one."""
    from agentic.llm import LlmError

    try:
        client.chat_json([{"role": "user", "content": "Reply with {}"}], {"type": "object"})
    except LlmError:
        pass


def build_parser() -> argparse.ArgumentParser:
    from agentic.llm import DEFAULT_MODEL

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--open", action="store_true", help="open the page in your browser")
    parser.add_argument("--llm", default="on",
                        help=f"'on' = {DEFAULT_MODEL} (default), 'off' = no LLM, or a model name")
    parser.add_argument("--source", choices=("demo", "full"), default="demo")
    parser.add_argument("--mimic-dir", type=Path, default=None)
    parser.add_argument("--mimic-raw-dir", type=Path, default=None)
    parser.add_argument("--mitdb-dir", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from agentic import orchestrator
    from agentic.llm import DEFAULT_MODEL, OllamaClient
    from frontend.app import App
    from frontend.outcome import OutcomeLookup
    from mimic_iii import config as mimic_config

    print("Loading the ECG ensemble and the MIMIC-III outputs...")
    deps = orchestrator.default_deps(
        need_ecg=True, need_chart=True, source=args.source, mimic_dir=args.mimic_dir,
        mimic_raw_dir=args.mimic_raw_dir, mitdb_dir=args.mitdb_dir, bundle_path=args.bundle)
    for what, why in deps.unavailable.items():
        print(f"  {what}: {why}")

    choice = args.llm.strip()
    name = None if choice.lower() == "off" else (DEFAULT_MODEL if choice.lower() == "on" else choice)
    llm = status = None
    if name:
        client = OllamaClient(name)
        status = client.status()
        print(f"LLM: {name} {'ready' if status.ok else '- ' + status.message}")
        if status.ok:
            llm = client
            threading.Thread(target=_warm_up, args=(client,), daemon=True).start()

    out_dir = Path(args.mimic_dir or mimic_config.output_dir(args.source)).expanduser()
    outcome = OutcomeLookup(out_dir) if (out_dir / "features.csv.gz").exists() else None
    app = App(deps, llm=llm, outcome=outcome, llm_status=status)

    try:
        server = make_server(app, port=args.port)
    except OSError as exc:
        print(f"Port {args.port} is not free ({exc}). Try --port {args.port + 1}.", file=sys.stderr)
        return 1
    print(f"Open {server.url}   (Ctrl+C to stop)")
    if args.open:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
