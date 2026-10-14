"""Small blocking WebSocket client used by ApiSender (runs on the buffer's sender thread only).

* Connects lazily and reconnects on its own after a drop (short backoff, capped).
* send() raises ConnectionError when the socket is down - the caller keeps the record and retries.
* Needs the `websocket-client` package (import name: websocket).

URL rule: `https://host` -> `wss://host` (encrypted, certificate verified); `http://host` -> `ws://host` (plain text,
logs a warning); no scheme -> `wss://`. If the configured host has no path, DEFAULT_PATH is added
(the camera id is filled in). If it already contains a path, it is used exactly as written.
"""
import json
import logging
import select
import threading
import time
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

import websocket  # websocket-client

log = logging.getLogger("pipeline")

DEFAULT_PATH = "/ws/{camera_id}"     # ASSUMPTION: change to whatever path the command center exposes
CONNECT_TIMEOUT_SECONDS = 5.0
SEND_TIMEOUT_SECONDS = 10.0
BACKOFF_START_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0


def build_ws_url(host: str, camera_id: str) -> str:
    if not host:
        raise ValueError("api.websocket_host is required")
    if "://" not in host:
        host = "wss://" + host                  # no scheme given -> encrypted by default
    parts = urlsplit(host)
    scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
    path = parts.path if parts.path not in ("", "/") else DEFAULT_PATH.format(camera_id=camera_id)
    return urlunsplit((scheme, parts.netloc, path, parts.query, ""))


class PiWebSocketClient:
    def __init__(self, host: str, camera_id: str):
        self.url = build_ws_url(host, camera_id)
        if self.url.startswith("ws://") and urlsplit(self.url).hostname not in ("localhost", "127.0.0.1", "::1"):
            log.warning("[api-ws] %s is NOT encrypted - person/face/vehicle images travel in plain text. "
                        "Use an https:// api.websocket_host (becomes wss://) if the server supports TLS.", self.url)
        self._ws: Optional[websocket.WebSocket] = None
        self._lock = threading.Lock()
        self._backoff = BACKOFF_START_SECONDS
        self._next_attempt = 0.0

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def _connect(self) -> None:
        now = time.monotonic()
        if now < self._next_attempt:
            raise ConnectionError(f"websocket down, next attempt in {self._next_attempt - now:.0f}s")
        try:
            self._ws = websocket.create_connection(self.url, timeout=CONNECT_TIMEOUT_SECONDS)
            self._ws.settimeout(SEND_TIMEOUT_SECONDS)
        except (OSError, websocket.WebSocketException) as error:
            self._ws = None
            self._next_attempt = time.monotonic() + self._backoff
            self._backoff = min(self._backoff * 2, BACKOFF_MAX_SECONDS)
            raise ConnectionError(f"cannot connect to {self.url}: {error}") from error
        self._backoff = BACKOFF_START_SECONDS
        log.info("[api-ws] connected to %s", self.url)

    def _drop(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        self._ws = None

    def _peer_still_there(self) -> bool:
        """Drain anything the server sent (pings, close frames). A closed/reset socket shows up here, so a record is
        not written into a connection we already know is dead (a plain send() on such a socket can 'succeed')."""
        try:
            while select.select([self._ws.sock], [], [], 0)[0]:
                self._ws.settimeout(0.05)
                self._ws.recv()
        except websocket.WebSocketTimeoutException:
            pass
        except (OSError, ValueError, websocket.WebSocketException):
            return False
        finally:
            if self._ws is not None and self._ws.sock is not None:
                self._ws.settimeout(SEND_TIMEOUT_SECONDS)
        return True

    def send(self, payload: Dict[str, Any]) -> None:
        """Send one JSON message. Raises ConnectionError if it could not be delivered."""
        message = json.dumps(payload)           # TypeError/ValueError here = bad payload, not a link problem
        with self._lock:
            if self._ws is not None and not self._peer_still_there():
                log.warning("[api-ws] connection lost")
                self._drop()
            if self._ws is None:
                self._connect()
            try:
                self._ws.send(message)
            except (OSError, websocket.WebSocketException) as error:
                self._drop()
                self._next_attempt = time.monotonic() + BACKOFF_START_SECONDS
                raise ConnectionError(f"send failed: {error}") from error

    def keepalive(self) -> None:
        """Called while idle: pings the socket so a dead link is noticed (and reconnected) before the next record."""
        with self._lock:
            if self._ws is None:
                try:
                    self._connect()
                except ConnectionError:
                    pass
                return
            try:
                if not self._peer_still_there():
                    raise websocket.WebSocketConnectionClosedException("peer closed")
                self._ws.ping()
            except (OSError, websocket.WebSocketException):
                log.warning("[api-ws] connection lost")
                self._drop()

    def close(self) -> None:
        with self._lock:
            self._drop()
