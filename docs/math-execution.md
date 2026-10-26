# Restricted math execution

MathAgent executes a deliberately restricted Python-like math language, not
general Python. This replaces its former broad smolagents import policy and
process/thread timeout wrapper. There is no fallback to a general executor.

## Supported contract

- `import sympy as sp` and `from sympy import ...` resolve fixed language tokens:
  `symbols`, `Symbol`, `Rational`, `Eq`, `Abs`, `solve`, `exp`, `sqrt`, `I`, `pi`.
- `import numpy as np` exposes only `np.argmin`; NumPy is not actually imported.
- Expression methods: `subs` (dictionary or two expressions), `as_real_imag()`.
- Built-ins: `range`, `print`, `abs`, `len`, `final_answer`.
- Numeric literals, Python-style division, arithmetic, assignments/unpacking,
  lists/tuples/dictionaries, indexing, comparisons, bounded `for` loops and
  single list comprehensions. No keyword arguments or arbitrary Python syntax.
- Symbol declarations accept bounded ASCII identifiers separated by spaces or
  commas. They do not accept SymPy's string expression or symbol-range syntax.
  `Rational` takes two integers. `solve` takes equations and explicit symbols.
- Task state persists across successful tool calls, not between `run()` calls
  (even with `reset=False`). Every execution error/timeout discards task state.
- Tools, managed agents, nonempty `additional_args`, additional imports and
  executor overrides are rejected rather than silently weakening the policy.
- Streaming callers must finish or close the returned generator to release the
  worker promptly. A MathAgent instance must not be shared by concurrent runs.

The four worked examples remain the compatibility baseline. This is an
intentional capability reduction; arbitrary existing Python snippets may fail.

## Security boundary

AST nodes are manually dispatched, never compiled/evaluated. Module names and
operations are opaque application tokens, not Python module/function objects.
The registry cannot expose `sympify`, `S`, `parse_expr`, `lambdify`, dynamic
imports, reflection, or arbitrary attributes. Mathematical wrappers recursively
reject strings and capability objects **before** calling SymPy. Typed SymPy
expressions may still use internal numeric/symbolic coercion; that is different
from evaluating model-provided expression strings.

Parsing, interpretation, mathematical operations and string rendering run in a
fresh isolated-mode Python subprocess, with a minimal environment, private
temporary working directory, no inherited descriptors, 2 GiB address-space
limit, 120 seconds cumulative CPU limit, and a per-call wall deadline (20 seconds
by default, checked after startup and during IPC). OS process creation and cleanup
are synchronous; external supervision is required for an end-to-end deadline.
Failure to install resource limits fails closed. Production execution targets
Linux. The current macOS environment rejects the required address-space limit,
so production execution is unavailable there rather than weakened. The parent kills and reaps the worker on failure. No daemon-thread
fallback, pickle deserialization, or join-before-draining queue is used.

Source is limited to 32 KiB; JSON messages to 64 KiB; output/logs to 8 KiB each;
AST/value depth to 64; AST nodes, collection lengths and symbolic traversal to
4096; interpreter steps to 100,000; iterations to 5000; variable names to 256;
symbols to 64; integers to 4096 bits; exponent magnitude to 2048. Solves accept
at most 16 equations/symbols. Expensive operations remain subject to worker
deadlines/resource limits. Replies are schema-checked primitive JSON, not live
symbolic objects. All worker errors are surfaced as observations, including
errors after print calls.

**This worker is not an OS sandbox.** It retains the service account's file and
network privileges. Production operators must additionally run the service with
no credentials available to the worker, an unprivileged identity, read-only
filesystem with isolated scratch space, denied network egress, and externally
enforced CPU/memory/process limits (for example, an appropriately configured
container or sandbox). Python `-I` and resource limits do not provide those
protections. No deployment configuration is installed by this change. Do not
expose this service as a general public code-execution endpoint.

The separate `math_verify` parsing paths in rewards/metrics are outside this
remediation and require their own review. Model/GPU inference and the original
security scanner must also be validated in the deployment environment.

## CPU-only regression tests

The tests use the standard library `unittest`, the declared SymPy and smolagents
versions, and stub only the optional vLLM import for agent integration tests.
They can be run with `python -m unittest discover -s tests -v`, without loading
model weights or a GPU. No exploit test invokes an OS command from model source.
On Linux these tests use the real production worker with resource limits. On
macOS, transport/integration tests explicitly substitute a **test-only** worker
fixture that omits the unsupported address-space limit. A separate test checks
that the unmodified production worker fails closed. Passing macOS tests does not
establish Linux resource-limit enforcement. The Linux CI workflow runs the real
worker on Python 3.11 and 3.12; CI results must be checked before deployment.