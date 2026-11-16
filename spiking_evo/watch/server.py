"""`spiking-evo watch`: a local web page to watch runs (stdlib http.server, no extra deps).

Listens on 127.0.0.1 only. Reads only inside the run folders it was given (and the
experiment files in ``spiking_evo/experiments``); never writes.

    GET  /                      the page (static/index.html, app.js, style.css)
    GET  /api/runs              runs found under the roots
    GET  /api/configs           experiment files that can be started from scratch
    GET  /api/series?run=PATH   CSV columns of one run
    POST /api/sim               {"run": PATH, "genomes": ...} or {"config": PATH} -> session
    GET  /api/frame?sid&n&arena advance n steps (0 = none) and describe one arena
    GET  /api/agent?sid&i       details of agent i (senses, outputs, brain)
"""

from __future__ import annotations

import itertools
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .data import find_runs, run_series
from .sim import Session, open_session

STATIC = Path(__file__).parent / "static"
EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}
MAX_STEPS = 2000
MAX_SESSIONS = 4


class App:
    def __init__(self, roots: list[Path]) -> None:
        self.roots = [r.resolve() for r in roots]
        self.sessions: dict[str, Session] = {}
        self.ids = itertools.count(1)
        self.lock = threading.Lock()

    def allowed_run(self, path: str) -> Path:
        p = Path(path).resolve()
        if not any(p.is_relative_to(r) for r in self.roots) or not (p / "config.yaml").exists():
            raise PermissionError(f"not a run under the watched folders: {path}")
        return p

    def allowed_config(self, path: str) -> Path:
        p = Path(path).resolve()
        if p.parent != EXPERIMENTS.resolve() or p.suffix not in (".yaml", ".toml"):
            raise PermissionError(f"not an experiment file: {path}")
        return p

    def runs(self) -> list[dict]:
        return find_runs(self.roots)

    def configs(self) -> list[dict]:
        return [
            {"path": str(p.resolve()), "name": p.stem} for p in sorted(EXPERIMENTS.glob("*.yaml"))
        ]

    def series(self, run: str) -> dict:
        return run_series(self.allowed_run(run))

    def new_session(self, body: dict) -> dict:
        if "config" in body:
            source = {"config": str(self.allowed_config(body["config"]))}
        else:
            source = {"run": str(self.allowed_run(body["run"])), "genomes": body.get("genomes")}
            if source["genomes"] is None:
                source.pop("genomes")
        session = open_session(source)
        with self.lock:
            sid = str(next(self.ids))
            self.sessions[sid] = session
            while len(self.sessions) > MAX_SESSIONS:  # forget the oldest
                self.sessions.pop(next(iter(self.sessions)))
        return {"sid": sid, **session.info()}

    def session(self, sid: str) -> Session:
        try:
            return self.sessions[sid]
        except KeyError:
            raise LookupError(f"no session {sid}") from None

    def frame(self, sid: str, n: int, arena: int) -> dict:
        s = self.session(sid)
        with self.lock:
            s.step(max(0, min(n, MAX_STEPS)))
            return s.frame(arena)

    def agent(self, sid: str, i: int) -> dict:
        s = self.session(sid)
        with self.lock:
            if not 0 <= i < s.world.n_agents:
                raise LookupError(f"no agent {i}")
            return s.agent(i)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _guard(self, fn) -> None:
            try:
                self._json(fn())
            except PermissionError as e:
                self._json({"error": str(e)}, 403)
            except (LookupError, FileNotFoundError) as e:
                self._json({"error": str(e)}, 404)
            except Exception as e:  # noqa: BLE001 - report to the page, keep serving
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/api/runs":
                return self._guard(app.runs)
            if url.path == "/api/configs":
                return self._guard(app.configs)
            if url.path == "/api/series":
                return self._guard(lambda: app.series(q.get("run", "")))
            if url.path == "/api/frame":
                return self._guard(
                    lambda: app.frame(q.get("sid", ""), int(q.get("n", 1)), int(q.get("arena", 0)))
                )
            if url.path == "/api/agent":
                return self._guard(lambda: app.agent(q.get("sid", ""), int(q.get("i", -1))))
            name = "index.html" if url.path in ("/", "") else url.path.lstrip("/")
            f = (STATIC / name).resolve()
            if f.parent != STATIC.resolve() or not f.is_file():
                return self._send(404, b"not found", "text/plain")
            return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "text/plain"))

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/sim":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            return self._guard(lambda: app.new_session(body))

    return Handler


def serve(roots: list[Path], port: int = 8765, open_browser: bool = True) -> None:
    app = App(roots)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"[watch] {url}  (Ctrl-C per uscire)")
    print("[watch] cartelle: " + ", ".join(str(r) for r in app.roots))
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
