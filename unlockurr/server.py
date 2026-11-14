#!/usr/bin/env python3
"""Unlockurr local server.

Serves the lock screen and Lock Manager from ./public and exposes
POST /api/identify, which asks Drex (hosted System One) to decide whether the
face readout from the door camera belongs to an enrolled guest.

The browser computes face embeddings on-device (face-api.js) and only sends
numeric match distances here. Drex is text-only, so it never sees an image.

Run:  DREX_API_KEY=... python3 server.py   ->  http://127.0.0.1:8080
"""

from __future__ import annotations

import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE / "public"


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from the project's .env (gitignored). The project file wins."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


load_dotenv(BASE / ".env")
HOST = os.environ.get("UNLOCKURR_HOST", "127.0.0.1")
PORT = int(os.environ.get("UNLOCKURR_PORT", "8080"))

# --- Drex connection -------------------------------------------------------
DREX_URL = "https://drex.nace.ai/v1/systemone"
DREX_MODEL = "drex-v2.0"
DREX_TIMEOUT_S = 60
DREX_ATTEMPTS = 3

# --- Unlock policy (review and tune these on your own enrolment data) -------
MIN_IDENTITY_PROB = 0.80   # Drex probability for the chosen guest
MIN_RELIABLE_PROB = 0.70   # Drex probability that the capture is trustworthy
MAX_MEAN_DISTANCE = 0.55   # hard gate in code: never unlock above this distance

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # refuse every redirect
        return None


def tls_context() -> ssl.SSLContext:
    """Verified TLS. python.org builds on macOS ship without a CA bundle, so fall back
    to the system bundle when Python's own is missing (verification stays on)."""
    ctx = ssl.create_default_context()
    if not Path(ssl.get_default_verify_paths().openssl_cafile).exists() and Path("/etc/ssl/cert.pem").exists():
        ctx.load_verify_locations("/etc/ssl/cert.pem")
    return ctx


OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=tls_context()), NoRedirect)


class DrexUnavailable(Exception):
    pass


def log(msg: str) -> None:
    print(f"[unlockurr] {msg}", file=sys.stderr, flush=True)


def call_drex(body: dict) -> dict:
    key = os.environ.get("DREX_API_KEY", "").strip()
    if not key:
        raise DrexUnavailable("DREX_API_KEY is not set on the server")
    data = json.dumps(body).encode()
    last = "unknown"
    for attempt in range(DREX_ATTEMPTS):
        req = urllib.request.Request(
            DREX_URL,
            data=data,
            method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        wait = None
        try:
            with OPENER.open(req, timeout=DREX_TIMEOUT_S) as r:
                resp = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            rid = e.headers.get("x-request-id", "-")
            last = f"HTTP {e.code}"
            log(f"Drex error {e.code} request_id={rid} attempt={attempt + 1}")
            if e.code not in (429, 500, 529):
                raise DrexUnavailable(last)
            ms = e.headers.get("retry-after-ms")
            if ms and ms.isdigit():
                wait = int(ms) / 1000
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = f"network: {type(e).__name__}"
            log(f"Drex network error {type(e).__name__} attempt={attempt + 1}")
        if attempt == DREX_ATTEMPTS - 1:
            raise DrexUnavailable(last)
        time.sleep(min(wait if wait is not None else 0.5 * 2 ** attempt + random.random() * 0.3, 8))
    else:  # pragma: no cover
        raise DrexUnavailable(last)

    if resp.get("model") != DREX_MODEL:
        log(f"Drex model mismatch: asked {DREX_MODEL}, got {resp.get('model')} request_id={resp.get('request_id')}")
        raise DrexUnavailable("unexpected model")
    return resp


def build_request(room: str, cands: list[dict], capture: dict) -> dict:
    criteria = {}
    for c in cands:
        criteria[c["id"]] = (
            f"The person at the door is {c['name']}. Their enrolled face samples "
            f"({c['samples']}) give mean distance {c['mean']:.3f}, best {c['best']:.3f}, "
            f"worst {c['worst']:.3f} across the captured frames."
        )
    criteria["no_match"] = (
        "Nobody enrolled clearly matches: every distance is too high, two guests are "
        "too close to call, or the readout is inconsistent."
    )
    state = {
        "context": (
            f"Face ID readout from the door lock of hotel room {room}. A face-embedding "
            "model compared the person at the door against each enrolled guest. Distance "
            "is Euclidean between 128-d face embeddings: same person is usually below "
            "0.45, 0.45-0.55 is borderline, above 0.6 is almost always a different person. "
            "A trustworthy match has a low mean, a low worst-case and a clear margin over "
            "the runner-up."
        ),
        "capture": capture,
        "candidates_sorted_by_mean_distance": cands,
    }
    return {
        "model": DREX_MODEL,
        "state": state,
        "questions": {
            "identity": {
                "type": "choice",
                "instructions": "Which enrolled guest, if any, is the person at the door?",
                "criteria": criteria,
            },
            "reliable": {
                "type": "noul",
                "instructions": (
                    "Is this capture reliable enough to act on: a face detected with high "
                    "confidence in every frame, consistent distances between frames, and a "
                    "clear margin between the best and second-best guest?"
                ),
            },
        },
    }


def num(v, lo=0.0, hi=10.0) -> float:
    f = float(v)
    if not (lo <= f <= hi):
        raise ValueError("out of range")
    return round(f, 4)


def parse_identify(payload: dict):
    room = str(payload.get("room", ""))[:12] or "----"
    raw = payload.get("candidates")
    if not isinstance(raw, list) or not 1 <= len(raw) <= 20:
        raise ValueError("candidates must be a list of 1-20")
    cands = []
    for c in raw:
        cid = str(c.get("id", ""))
        if not ID_RE.match(cid) or cid == "no_match":
            raise ValueError("bad candidate id")
        cands.append({
            "id": cid,
            "name": re.sub(r"\s+", " ", str(c.get("name", "Guest"))).strip()[:60] or "Guest",
            "mean": num(c["mean"]), "best": num(c["best"]), "worst": num(c["worst"]),
            "samples": int(num(c.get("samples", 1), 1, 50)),
        })
    if len({c["id"] for c in cands}) != len(cands):
        raise ValueError("duplicate candidate id")
    cands.sort(key=lambda c: c["mean"])
    cap = payload.get("capture") or {}
    capture = {
        "frames": int(num(cap.get("frames", 0), 0, 30)),
        "mean_detection_confidence": num(cap.get("detection", 0), 0, 1),
        "min_detection_confidence": num(cap.get("min_detection", 0), 0, 1),
        "face_width_fraction_of_frame": num(cap.get("face_width", 0), 0, 1),
    }
    if len(cands) > 1:
        capture["margin_best_vs_runner_up"] = round(cands[1]["mean"] - cands[0]["mean"], 4)
    return room, cands, capture


def identify(payload: dict) -> tuple[int, dict]:
    try:
        room, cands, capture = parse_identify(payload)
    except (ValueError, KeyError, TypeError) as e:
        return 400, {"ok": False, "error": "bad_request", "message": str(e)}

    try:
        resp = call_drex(build_request(room, cands, capture))
    except DrexUnavailable as e:
        # Guardrail fails closed: no Drex verdict, no face unlock.
        return 503, {"ok": False, "error": "drex_unavailable", "message": str(e)}

    answers = resp.get("answers") or {}
    ident, rel = answers.get("identity") or {}, answers.get("reliable") or {}
    labels = {c["id"] for c in cands} | {"no_match"}
    if (ident.get("type") != "choice" or ident.get("choice") not in labels
            or rel.get("type") != "noul" or not isinstance(rel.get("noul"), (int, float))):
        log(f"Drex answer shape invalid request_id={resp.get('request_id')}")
        return 503, {"ok": False, "error": "drex_unavailable", "message": "invalid answer shape"}

    choice = ident["choice"]
    prob = float((ident.get("probabilities") or {}).get(choice, 0))
    reliable = float(rel["noul"])
    chosen = next((c for c in cands if c["id"] == choice), None)

    reason = "ok"
    if chosen is None:
        reason = "Drex found no matching user"
    elif prob < MIN_IDENTITY_PROB:
        reason = f"Drex not confident ({prob:.0%})"
    elif reliable < MIN_RELIABLE_PROB:
        reason = f"Capture unreliable ({reliable:.0%})"
    elif chosen["mean"] > MAX_MEAN_DISTANCE:
        reason = "Distance above hard limit"
    unlock = reason == "ok"

    log(f"identify room={room} choice={choice} p={prob:.2f} reliable={reliable:.2f} "
        f"unlock={unlock} request_id={resp.get('request_id')}")
    return 200, {
        "ok": True, "unlock": unlock, "user_id": choice if unlock else None,
        "choice": choice, "probability": round(prob, 4), "reliable": round(reliable, 4),
        "reason": reason, "model": resp.get("model"), "request_id": resp.get("request_id"),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        # make browsers revalidate static files so code updates are picked up on reload
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # keep the console for lock events

    def send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/status":
            return self.send_json(200, {
                "drex_key_configured": bool(os.environ.get("DREX_API_KEY", "").strip()),
                "model": DREX_MODEL,
            })
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/identify":
            return self.send_json(404, {"ok": False, "error": "not_found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 64_000:
                return self.send_json(413, {"ok": False, "error": "too_large"})
            payload = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError
        except ValueError:
            return self.send_json(400, {"ok": False, "error": "bad_json"})
        code, obj = identify(payload)
        self.send_json(code, obj)


def open_in_chrome(urls: list[str]) -> None:
    """Open the lock and manager in Google Chrome (falls back to the default browser)."""
    import subprocess
    import webbrowser
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "-a", "Google Chrome", *urls], check=True)
            return
        for name in ("google-chrome", "chrome", "chromium"):
            try:
                b = webbrowser.get(name)
                for u in urls:
                    b.open_new_tab(u)
                return
            except webbrowser.Error:
                continue
    except (OSError, subprocess.CalledProcessError):
        pass
    log("Google Chrome not found; opening the default browser instead")
    for u in urls:
        webbrowser.open_new_tab(u)


if __name__ == "__main__":
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    key = "configured" if os.environ.get("DREX_API_KEY", "").strip() else "MISSING (face unlock disabled)"
    print(f"Unlockurr running at http://{HOST}:{PORT}\n  Lock:     http://{HOST}:{PORT}/lock.html\n"
          f"  Manager:  http://{HOST}:{PORT}/manager.html\n  Drex key: {key}")
    if "--no-browser" not in sys.argv:
        open_in_chrome([f"http://{HOST}:{PORT}/manager.html", f"http://{HOST}:{PORT}/lock.html"])
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
