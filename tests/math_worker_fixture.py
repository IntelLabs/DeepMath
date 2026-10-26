"""Transport-only fixture for macOS; NEVER used by the production executor.

macOS rejects RLIMIT_AS in this environment. This fixture removes that one
limit ONLY to test subprocess protocol, state, and cancellation. Linux tests
use the unmodified production worker and all real limits instead.
"""

import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_math.math_interpreter import _worker_main  # noqa: E402

if __name__ == "__main__":
    if sys.platform != "darwin":
        raise RuntimeError("This fixture is only for macOS transport tests")
    original = resource.setrlimit

    def set_test_limit(kind, limits):
        if kind != resource.RLIMIT_AS:
            original(kind, limits)

    resource.setrlimit = set_test_limit
    _worker_main()
