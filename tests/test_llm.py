"""LLM client tests. The live smoke skips when no endpoint is listening."""
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from forgeai.llm import LLMClient, LLMError  # noqa: E402


class _FakeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.bodies.append(body)
        if self.server.fail_remaining:
            self.server.fail_remaining -= 1
            message = b"nope"
            self.send_response(self.server.fail_code)
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)
            return
        payload = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "pong"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        return


class FakeServerTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
        self.server.bodies = []
        self.server.fail_remaining = 0
        self.server.fail_code = 500
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        port = self.server.server_address[1]
        self.tmp = tempfile.TemporaryDirectory()
        self.client = LLMClient(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="fake-model",
            timeout=5,
            run_id="llm-test",
            runs_dir=self.tmp.name,
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def _log_line(self):
        path = Path(self.tmp.name) / "llm-test" / "calls.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        return json.loads(lines[0])

    def test_defaults_and_log_line(self):
        text = self.client.chat([{"role": "user", "content": "ping"}])
        self.assertEqual(text, "pong")
        body = self.server.bodies[0]
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["seed"], 0)
        self.assertEqual(body["model"], "fake-model")
        record = self._log_line()
        self.assertEqual(record["model"], "fake-model")
        self.assertEqual(len(record["prompt_hash"]), 64)
        self.assertEqual(record["prompt_tokens"], 3)
        self.assertEqual(record["completion_tokens"], 1)
        self.assertIsInstance(record["latency_ms"], int)
        self.assertGreaterEqual(record["latency_ms"], 0)
        self.assertEqual(record["attempts"], 1)

    def test_retries_http_500_once(self):
        self.server.fail_remaining = 1
        self.server.fail_code = 500
        text = self.client.chat([{"role": "user", "content": "ping"}])
        self.assertEqual(text, "pong")
        self.assertEqual(len(self.server.bodies), 2)
        self.assertEqual(self._log_line()["attempts"], 2)

    def test_http_400_does_not_retry(self):
        self.server.fail_remaining = 5
        self.server.fail_code = 400
        with self.assertRaises(LLMError):
            self.client.chat([{"role": "user", "content": "ping"}])
        self.assertEqual(len(self.server.bodies), 1)

    def test_connection_refused_tries_three_times(self):
        calls = []

        def boom(*args, **kwargs):
            calls.append(1)
            raise urllib.error.URLError(ConnectionRefusedError(111, "refused"))

        client = LLMClient(
            base_url="http://127.0.0.1:9/v1",
            model="fake-model",
            timeout=0.2,
            run_id="refused",
            runs_dir=self.tmp.name,
        )
        with patch("forgeai.llm.urllib.request.urlopen", boom):
            with self.assertRaises(LLMError):
                client.chat([{"role": "user", "content": "ping"}])
        self.assertEqual(len(calls), 3)


class LiveSmokeTests(unittest.TestCase):
    def test_skips_when_endpoint_is_absent(self):
        base = os.environ.get("FORGEAI_BASE_URL", "http://127.0.0.1:11434/v1")
        parsed = urlparse(base)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=0.5):
                pass
        except OSError:
            self.skipTest(f"endpoint absent: {base}")
        model = os.environ.get("FORGEAI_MODEL")
        if not model:
            self.skipTest("endpoint is up but FORGEAI_MODEL is unset")
        with tempfile.TemporaryDirectory() as tmp:
            client = LLMClient(base_url=base, model=model, timeout=30, run_id="smoke", runs_dir=tmp)
            text = client.chat([{"role": "user", "content": "Reply with the single word pong."}])
        self.assertIsInstance(text, str)
        self.assertTrue(text.strip())


if __name__ == "__main__":
    unittest.main()
