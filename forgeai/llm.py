"""OpenAI-compatible chat client. Stdlib only.

Base URL comes from FORGEAI_BASE_URL (default http://127.0.0.1:11434/v1).
Model name comes from FORGEAI_MODEL (default qwen2.5:3b-instruct).
FORGEAI_API_KEY is sent only when set.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen2.5:3b-instruct"
DEFAULT_TIMEOUT_S = 60
MAX_RETRIES = 2  # two retries after the first attempt


class LLMError(Exception):
    """The endpoint failed, or the response was not a chat completion."""


def prompt_hash(messages) -> str:
    """SHA-256 of the message payload. Temperature and seed are not included."""
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LLMClient:
    def __init__(self, base_url=None, model=None, api_key=None, timeout=DEFAULT_TIMEOUT_S,
                 run_id="default", runs_dir="runs"):
        chosen = base_url if base_url is not None else os.environ.get("FORGEAI_BASE_URL")
        self.base_url = (chosen or DEFAULT_BASE_URL).rstrip("/")
        self.model = model if model is not None else os.environ.get("FORGEAI_MODEL", DEFAULT_MODEL)
        if api_key is not None:
            self.api_key = api_key or None
        else:
            self.api_key = os.environ.get("FORGEAI_API_KEY") or None
        self.timeout = timeout
        self.run_id = run_id
        self.runs_dir = Path(runs_dir)

    def chat(self, messages, temperature=0, seed=0) -> str:
        """Send one chat completion. Retries connection errors, timeouts, and HTTP 5xx."""
        started = time.perf_counter()
        last_error = None
        attempt = 0
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                text, prompt_tokens, completion_tokens = self._post(messages, temperature, seed)
            except urllib.error.HTTPError as exc:
                exc.close()
                last_error = exc
                if 400 <= exc.code < 500:
                    break
                continue
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                continue
            except LLMError:
                self._append_log(messages, None, None, attempt, started)
                raise
            self._append_log(messages, prompt_tokens, completion_tokens, attempt, started)
            return text
        self._append_log(messages, None, None, attempt, started)
        raise LLMError(f"chat completion failed after {attempt} attempt(s): {last_error}")

    def _post(self, messages, temperature, seed):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
        try:
            body = json.loads(raw)
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"malformed chat completion response: {exc}") from exc
        usage = body.get("usage") or {}
        return text, usage.get("prompt_tokens"), usage.get("completion_tokens")

    def _append_log(self, messages, prompt_tokens, completion_tokens, attempts, started):
        path = self.runs_dir / self.run_id / "calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": self.model,
            "prompt_hash": prompt_hash(messages),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": int(round((time.perf_counter() - started) * 1000)),
            "attempts": attempts,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
