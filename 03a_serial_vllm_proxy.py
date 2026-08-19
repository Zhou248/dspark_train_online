#!/usr/bin/env python3
"""Serialize online hidden-state requests through a local HTTP proxy.

The vLLM response can expose a hidden-state path before its asynchronous disk
write is complete. This proxy holds a global request lock until the companion
file lock is released, preventing the next FSDP rank from overlapping target
execution with the preceding hidden-state export.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx


SERIAL_PATHS = {"/v1/chat/completions", "/v1/completions"}


def hidden_state_handle(body: bytes) -> str | None:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    transfer = payload.get("kv_transfer_params")
    if not isinstance(transfer, dict):
        return None
    value = transfer.get("hidden_states_path") or transfer.get("handle")
    return str(value) if value else None


def wait_for_hidden_state(handle: str, timeout: float) -> None:
    """Wait for vLLM's exclusive writer lock without removing the lock file."""
    path = Path(handle)
    lock_path = Path(f"{handle}.lock")
    deadline = time.monotonic() + timeout

    while not lock_path.exists():
        if path.exists():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for hidden-state lock: {lock_path}")
        time.sleep(0.01)

    descriptor = os.open(lock_path, os.O_RDONLY)
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for hidden-state write: {path}"
                    ) from None
                time.sleep(0.05)
        if not path.exists():
            raise FileNotFoundError(f"vLLM released lock but file is missing: {path}")
    finally:
        os.close(descriptor)


def make_handler(upstream: str, timeout: float):
    serial_lock = threading.Lock()
    client = httpx.Client(timeout=timeout, trust_env=False)

    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, message: str, *args) -> None:
            print(f"proxy client={self.client_address[0]} {message % args}", flush=True)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self._forward(serialized=False)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlsplit(self.path).path
            self._forward(serialized=path in SERIAL_PATHS)

        def _forward(self, serialized: bool) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            request_body = self.rfile.read(length) if length else b""
            lock = serial_lock if serialized else threading.Lock()
            try:
                with lock:
                    response = client.request(
                        self.command,
                        f"{upstream}{self.path}",
                        content=request_body,
                        headers={
                            key: value
                            for key, value in self.headers.items()
                            if key.lower() not in {"host", "content-length", "connection"}
                        },
                    )
                    response_body = response.content
                    if serialized and response.status_code < 300:
                        handle = hidden_state_handle(response_body)
                        if handle is None:
                            raise ValueError(
                                "Successful generation response has no hidden-state handle"
                            )
                        wait_for_hidden_state(handle, timeout)
            except Exception as exc:  # noqa: BLE001 - convert proxy failures to HTTP
                response_body = json.dumps(
                    {"error": {"message": f"serial proxy: {type(exc).__name__}: {exc}"}}
                ).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(response_body)
                return

            self.send_response(response.status_code)
            self.send_header(
                "Content-Type", response.headers.get("content-type", "application/json")
            )
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(response_body)

    return ProxyHandler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(args.upstream.rstrip("/"), args.timeout)
    )
    print(
        f"Serial hidden-state proxy listening on http://{args.host}:{args.port} "
        f"-> {args.upstream}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

