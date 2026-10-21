"""Killable math worker with bounded JSON IPC; no pickle or host-thread fallback.

This is resource containment, NOT filesystem/network isolation. Run the service
under the OS/container restrictions described in docs/math-execution.md.
"""

import json
import math
import os
import re
import selectors
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

MAX_SOURCE = 32_768
MAX_MESSAGE = 65_536
MAX_OUTPUT = 8_192
MAX_MODEL_OUTPUT = 131_072
_CODE_MARKERS = re.compile(r"<tool_call>|</tool_call>|<code>|</code>|```python[ \t]*\r?\n|```")


def validate_model_output(text):
    if (
        type(text) is not str
        or len(text) > MAX_MODEL_OUTPUT
        or len(text.encode("utf-8")) > MAX_MODEL_OUTPUT
    ):
        raise ValueError("Model output must be bounded text")


def extract_math_code(text):
    """Linear, bounded fenced-code extraction with no host-side Python parsing."""
    validate_model_output(text)
    blocks = []
    start = None
    closing = None
    for marker in _CODE_MARKERS.finditer(text):
        token = marker.group()
        if start is None:
            if token in ("<tool_call>", "<code>") or token.startswith("```python"):
                closing = {"<tool_call>": "</tool_call>", "<code>": "</code>"}.get(token, "```")
                start = marker.end()
        elif token == closing:
            blocks.append(text[start : marker.start()])
            start = None
        else:
            raise ValueError("Nested or mismatched code markers are not supported")
    if start is not None:
        # Generation can stop at the closing marker without returning it.
        blocks.append(text[start:])
    if not blocks:
        return None
    source = "\n".join(blocks)
    if len(source) > MAX_SOURCE or len(source.encode("utf-8")) > MAX_SOURCE:
        raise ValueError("Math source size limit exceeded")
    return source


@dataclass(frozen=True)
class MathResult:
    output: str
    logs: str
    is_final_answer: bool


class RestrictedMathExecutor:
    """One serial worker per task. Errors and reset destroy all task state."""

    def __init__(self, timeout_seconds=20):
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Math execution timeout must be positive and finite")
        self.timeout_seconds = timeout_seconds
        self._process = None
        self._directory = None

    def _start(self):
        if os.name != "posix":
            raise RuntimeError("Restricted math execution requires POSIX resource limits")
        self._directory = tempfile.TemporaryDirectory(prefix="deep-math-")
        worker = Path(__file__).with_name("math_interpreter.py").resolve()
        self._process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=self._directory.name,
            env={"PYTHONHASHSEED": "0"},
            close_fds=True,
            start_new_session=True,
            bufsize=0,
        )
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)

    def __call__(self, source):
        try:
            if (
                type(source) is not str
                or len(source) > MAX_SOURCE
                or len(source.encode("utf-8")) > MAX_SOURCE
            ):
                raise ValueError("Math source size limit exceeded")
            request = json.dumps({"source": source}, ensure_ascii=False).encode("utf-8") + b"\n"
            if len(request) > MAX_MESSAGE:
                raise ValueError("Math request size limit exceeded")
            deadline = time.monotonic() + self.timeout_seconds
            if self._process is None:
                self._start()
            response = self._exchange(request, deadline)
            return self._decode(response)
        except BaseException:
            self.close()
            raise

    def _exchange(self, request, deadline):
        process = self._process
        response = bytearray()
        sent = 0
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdin, selectors.EVENT_WRITE)
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Math execution timed out; task state was discarded")
                events = selector.select(remaining)
                if not events:
                    raise TimeoutError("Math execution timed out; task state was discarded")
                for key, _ in events:
                    if key.fileobj is process.stdin:
                        sent += os.write(process.stdin.fileno(), request[sent:])
                        if sent == len(request):
                            selector.unregister(process.stdin)
                    else:
                        chunk = os.read(process.stdout.fileno(), 4096)
                        if not chunk:
                            raise RuntimeError("Math worker exited without a response")
                        response.extend(chunk)
                        if len(response) > MAX_MESSAGE:
                            raise ValueError("Math response size limit exceeded")
                        if b"\n" in response:
                            if not response.endswith(b"\n") or response.count(b"\n") != 1:
                                raise ValueError("Malformed math response framing")
                            return bytes(response)

    @staticmethod
    def _decode(response):
        data = json.loads(response)
        if type(data) is not dict or type(data.get("ok")) is not bool:
            raise ValueError("Malformed math response")
        if not data["ok"]:
            if set(data) != {"ok", "error"} or type(data["error"]) is not str:
                raise ValueError("Malformed math error response")
            raise ValueError(data["error"][:1000] + " (task state discarded)")
        if (
            set(data) != {"ok", "output", "logs", "final"}
            or type(data["final"]) is not bool
            or any(
                type(data[key]) is not str or len(data[key].encode("utf-8")) > MAX_OUTPUT
                for key in ("output", "logs")
            )
        ):
            raise ValueError("Malformed math result")
        return MathResult(data["output"], data["logs"], data["final"])

    def reset(self):
        self.close()

    def cleanup(self):
        self.close()

    def close(self):
        process, self._process = self._process, None
        try:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
                process.stdin.close()
                process.stdout.close()
        finally:
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None

    def __del__(self):
        if hasattr(self, "_process"):
            self.close()
