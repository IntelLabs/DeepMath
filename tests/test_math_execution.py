import ast
import importlib.util
import json
import re
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import sympy as sp
from smolagents import ChatMessage, FinalAnswerStep, Model
from smolagents.memory import ActionStep

from deep_math.math_executor import (
    MAX_MODEL_OUTPUT,
    MAX_SOURCE,
    RestrictedMathExecutor,
    extract_math_code,
)
from deep_math.math_interpreter import MathInterpreter, MathPolicyError

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = re.findall(
    r"<tool_call>(.*?)</tool_call>",
    (ROOT / "deep_math/fewshot.txt").read_text(),
    re.S,
)


class TransportExecutor(RestrictedMathExecutor):
    """Real production worker on Linux; explicitly transport-only on macOS."""

    def _start(self):
        if sys.platform != "darwin":
            return super()._start()
        popen = subprocess.Popen

        def start_fixture(args, **kwargs):
            return popen([*args[:-1], str(ROOT / "tests/math_worker_fixture.py")], **kwargs)

        with patch("deep_math.math_executor.subprocess.Popen", side_effect=start_fixture):
            return super()._start()


class InterpreterTests(unittest.TestCase):
    def test_documented_examples(self):
        interpreter = MathInterpreter()
        results = [interpreter.execute(source) for source in EXAMPLES]
        self.assertEqual(results[0]["output"], "-7")
        self.assertEqual(results[1]["output"], "[2/25, 3/100, 1/300, 19/700, 9/200]")
        self.assertEqual(results[2]["logs"], "1/6\n")
        self.assertEqual(results[3]["output"], "[-10, -10, 10, 10]")
        self.assertEqual(results[4]["logs"], "sqrt(2)*(1008 + 1008*sqrt(3))\n")

    def test_rejects_evaluation_capabilities_before_invocation(self):
        attacks = [
            "sp.sympify('1+1')",
            "sp.S('1+1')",
            "sp.parse_expr('1+1')",
            "sp.lambdify(x, x)",
            "sp.parsing.sympy_parser.parse_expr('1+1')",
            "from sympy import sympify as parse\nparse('1+1')",
            "from sympy import *",
            "from sympy.core import sympify",
            "import os",
            "import scipy",
            "import sympy.core",
            "np.load('file')",
            "np.frompyfunc(sp.sqrt, 1, 1)",
            "__import__('os')",
            "eval('1+1')",
            "exec('1+1')",
            "getattr(sp, 'sympify')('1+1')",
            "x.__class__",
            "sp.__dict__",
            "sp.sqrt('x+1')",
            "f = sp.sqrt\nf('x+1')",
            "sp.Eq('x', 1)",
            "sp.Rational('1/2', 1)",
            "sp.solve(['x+1'], [x])",
            "sp.solve([{x: 'x+1'}], [x])",
            "x.subs({x: 'x+1'})",
            "x.subs('x', 1)",
            "sp.Symbol('x', cls=sp.sqrt)",
            "sp.Symbol('x+1')",
            "sp.symbols('x:10')",
            "sp.sqrt(**{x: 1})",
            "sp.sqrt(sp.sqrt)",
            "sp.sqrt = 1",
            "a = [1]\na[0] = 2",
            "[1].append(2)",
            "[sp.sqrt]",
            "print({1: sp})",
            "(lambda: 1)()",
            "def f():\n    return 1",
            "while True:\n    pass",
        ]
        for source in attacks:
            with self.subTest(source=source):
                interpreter = MathInterpreter()
                interpreter.execute("import sympy as sp\nimport numpy as np\nx = sp.Symbol('x')")
                with (
                    patch.object(
                        sp, "sympify", side_effect=AssertionError("sympify reached")
                    ) as sympify,
                    patch.object(
                        sp, "lambdify", side_effect=AssertionError("lambdify reached")
                    ) as lambdify,
                    patch(
                        "sympy.parsing.sympy_parser.parse_expr",
                        side_effect=AssertionError("parser reached"),
                    ) as parser,
                ):
                    with self.assertRaises(MathPolicyError):
                        interpreter.execute(source)
                    sympify.assert_not_called()
                    lambdify.assert_not_called()
                    parser.assert_not_called()

    def test_reported_sympify_escape_never_reaches_eval_capable_sink(self):
        # Freeze the reported snippet here; do not depend on the optional PoC bundle.
        source = (
            "from sympy import sympify\n"
            "result = sympify(\"__import__('os').popen('id -u; id -g; uname -m; "
            "echo POC-MARKER > proof_from_payload.txt').read()\")\n"
            "print(result)\n"
            "final_answer(result)\n"
        )
        # Intercept the dangerous sink even if the policy regresses. This test must
        # never execute the report's shell command or write its marker file.
        with (
            patch.object(sp, "sympify", side_effect=AssertionError("sympify reached")) as sink,
            patch("os.popen", side_effect=AssertionError("shell reached")) as shell,
        ):
            with self.assertRaisesRegex(
                MathPolicyError, "Import is not an approved math capability"
            ):
                MathInterpreter().execute(source)
            sink.assert_not_called()
            shell.assert_not_called()

    def test_statement_semantics_and_safe_aliases(self):
        interpreter = MathInterpreter()
        self.assertEqual(interpreter.execute("1/2")["output"], "0.5")
        self.assertEqual(
            interpreter.execute("from sympy import Rational as R\nf = R\nf(1, 2)")["output"], "1/2"
        )
        interpreter.execute("i = 99\na, b = (2, 3)\n[a*i for i in range(b)]")
        self.assertEqual(interpreter.execute("i")["output"], "99")
        self.assertEqual(interpreter.execute("[i for i in range(4) if i > 1]")["output"], "[2, 3]")

    def test_final_answer_unwinds_any_statement(self):
        for source in (
            "final_answer(42)\nmissing",
            "answer = final_answer(42)\nmissing",
            "for i in range(3):\n    answer = final_answer(42)\nmissing",
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    MathInterpreter().execute(source),
                    {
                        "output": "42",
                        "logs": "",
                        "final": True,
                    },
                )

    def test_numeric_and_structural_budgets(self):
        sources = [
            "2 ** 2049",
            "(2 ** 1024) ** 1024",
            "from sympy import Rational\nr = Rational(1, 2) ** 1024\nr ** 2048",
            "from sympy import Rational\nRational(1, 2) / 0",
            "range(4097)",
            "1e309",
            "'x' * 1000000",
            "x = 0\nfor i in range(13):\n    x = [x, x]",
            "for i in range(100):\n    for j in range(100):\n        x = i + j",
            "for i in range(3000):\n    print('abcd')",
            "from sympy import symbols\nsymbols('" + " ".join(f"x{i}" for i in range(65)) + "')",
            "[" * 70 + "1" + "]" * 70,
            "x = 1\n" * 5000,
        ]
        for source in sources:
            with self.subTest(source=source[:100]):
                with self.assertRaises(MathPolicyError):
                    MathInterpreter().execute(source)
        self.assertEqual(MathInterpreter().execute("2 ** 2048 > 0")["output"], "True")

    def test_foreign_objects_never_reach_wrappers(self):
        interpreter = MathInterpreter()
        interpreter.execute("import sympy as sp")
        interpreter.variables["foreign"] = object()
        with patch.object(sp, "sqrt") as sqrt:
            with self.assertRaises(MathPolicyError):
                interpreter.execute("sp.sqrt(foreign)")
            sqrt.assert_not_called()


@unittest.skipUnless(sys.platform in ("darwin", "linux"), "POSIX worker tests")
class WorkerTransportTests(unittest.TestCase):
    def setUp(self):
        self.worker = TransportExecutor(30)
        self.addCleanup(self.worker.close)

    def test_documented_examples_over_json(self):
        results = [self.worker(source) for source in EXAMPLES]
        self.assertEqual(results[0].output, "-7")
        self.assertEqual(results[2].logs, "1/6\n")
        self.assertEqual(results[3].output, "[-10, -10, 10, 10]")
        self.assertEqual(results[4].logs, "sqrt(2)*(1008 + 1008*sqrt(3))\n")

    def test_state_reset_and_errors(self):
        self.worker("x = 7")
        self.assertEqual(self.worker("x + 1").output, "8")
        process = self.worker._process
        with self.assertRaisesRegex(ValueError, "Undefined variable"):
            self.worker("print('before error')\nmissing")
        self.assertIsNotNone(process.poll())
        with self.assertRaises(ValueError):
            self.worker("x")
        self.worker("x = 9")
        self.worker.reset()
        with self.assertRaises(ValueError):
            self.worker("x")

    def test_deadline_kills_and_reaps_worker(self):
        self.worker("x = 1")
        process = self.worker._process
        directory = self.worker._directory.name
        self.worker.timeout_seconds = 1e-9
        with self.assertRaises(TimeoutError):
            self.worker("1 + 1")
        self.assertIsNotNone(process.poll())
        self.assertIsNone(self.worker._process)
        self.assertFalse(Path(directory).exists())
        self.worker.timeout_seconds = 30
        with self.assertRaisesRegex(ValueError, "Undefined variable"):
            self.worker("x")

    def test_cleanup_contract(self):
        self.worker("1")
        process = self.worker._process
        self.worker.cleanup()
        self.worker.cleanup()
        self.assertIsNotNone(process.poll())

    def test_oversized_output_kills_worker(self):
        self.worker("x = 1")
        with self.assertRaisesRegex(ValueError, "Print limit"):
            self.worker("for i in range(3000):\n    print('abcd')")
        self.assertIsNone(self.worker._process)

    def test_startup_failure_never_falls_back(self):
        with patch(
            "deep_math.math_executor.subprocess.Popen", side_effect=OSError("startup failed")
        ):
            with self.assertRaisesRegex(OSError, "startup failed"):
                RestrictedMathExecutor()("1")

    @unittest.skipUnless(sys.platform == "darwin", "macOS fail-closed regression")
    def test_production_macos_resource_limits_fail_closed(self):
        worker = RestrictedMathExecutor(30)
        self.addCleanup(worker.close)
        with self.assertRaisesRegex(ValueError, "Required worker resource limits unavailable"):
            worker("1")
        self.assertIsNone(worker._process)


class ProtocolTests(unittest.TestCase):
    def test_input_limits_before_encoding_or_startup(self):
        with patch.object(RestrictedMathExecutor, "_start") as start:
            for source in ("x" * (MAX_SOURCE + 1), "é" * MAX_SOURCE, object()):
                with self.assertRaises(ValueError):
                    RestrictedMathExecutor()(source)
            start.assert_not_called()

    def test_strict_reply_schema(self):
        for reply in (
            [],
            {"ok": 1},
            {"ok": True},
            {"ok": False, "error": []},
            {"ok": True, "output": [], "logs": "", "final": False},
            {"ok": True, "output": "", "logs": "", "final": 1},
            {"ok": True, "output": "x" * 8193, "logs": "", "final": False},
            {"ok": True, "output": "", "logs": "", "final": False, "extra": 1},
        ):
            with self.subTest(reply=str(reply)[:100]):
                with self.assertRaises(ValueError):
                    RestrictedMathExecutor._decode(json.dumps(reply))

    def test_code_extraction_never_parses_python_in_parent(self):
        with patch.object(ast, "parse", side_effect=AssertionError("host parsed Python")):
            self.assertIsNone(extract_math_code("1 + 1"))
            self.assertEqual(extract_math_code("text\n<tool_call>1 + 1</tool_call>"), "1 + 1")
            self.assertEqual(extract_math_code("```python\n1 + 1\n```"), "1 + 1\n")
            self.assertEqual(extract_math_code("<code>1 + 1"), "1 + 1")
            with self.assertRaises(ValueError):
                extract_math_code("<tool_call>" * 2000)
            with self.assertRaises(ValueError):
                extract_math_code("x" * (MAX_MODEL_OUTPUT + 1))


# Stub only the optional GPU import; all smolagents classes/lifecycle are real.
if importlib.util.find_spec("vllm") is None:
    vllm_stub = types.ModuleType("vllm")
    vllm_stub.SamplingParams = type("SamplingParams", (), {})
    with patch.dict(sys.modules, {"vllm": vllm_stub}):
        from deep_math.agent import MathAgent
else:
    from deep_math.agent import MathAgent


class StubModel(Model):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = iter(outputs)
        self.calls = 0
        self.last_output_token_count = 1
        self.model_id = "test-model"

    def generate(self, messages, **kwargs):
        self.calls += 1
        return ChatMessage(role="assistant", content=next(self.outputs))


class TransportMathAgent(MathAgent):
    def create_python_executor(self):
        return TransportExecutor(self.code_execution_timeout)


class AgentTests(unittest.TestCase):
    def agent(self, outputs, **kwargs):
        agent = TransportMathAgent([], StubModel(outputs), verbosity_level=-1, **kwargs)
        self.addCleanup(agent.cleanup)
        return agent

    def test_actual_agent_constructor_and_bypass_rejection(self):
        agent = MathAgent([], StubModel([]), verbosity_level=-1)
        self.addCleanup(agent.cleanup)
        self.assertIs(type(agent.python_executor), RestrictedMathExecutor)
        cases = [
            {"additional_authorized_imports": ["sympy"]},
            {"executor": object()},
            {"executor_type": "local"},
            {"executor_kwargs": {}},
            {"managed_agents": [object()]},
            {"add_base_tools": True},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    MathAgent([], StubModel([]), **kwargs)
        with self.assertRaises(ValueError):
            MathAgent([object()], StubModel([]))
        with self.assertRaises(ValueError):
            agent.run("task", additional_args={"host": object()})

    def test_model_emitted_sympify_escape_rejected_then_valid_math_succeeds(self):
        # Preserve the report's fenced code, import, and nested __import__ pattern,
        # but use a harmless expression: subprocesses do not inherit parent mocks.
        probe = (
            "Here is the computation:\n```python\n"
            "from sympy import sympify\n"
            "result = sympify(\"__import__('builtins').str(6 * 7)\")\n"
            "print(result)\n"
            "final_answer(result)\n```"
        )
        agent = self.agent(
            [probe, "<tool_call>final_answer(6 * 7)</tool_call>"], max_steps=2
        )
        events = list(agent.run("Security regression probe", stream=True))
        actions = [event for event in events if isinstance(event, ActionStep)]
        finals = [event for event in events if isinstance(event, FinalAnswerStep)]

        self.assertEqual(agent.model.calls, 2)
        self.assertEqual(len(actions), 2)
        # Assert actual execution observations, not the echoed model source.
        self.assertEqual(actions[0].model_output, probe)
        self.assertIn(
            "MathPolicyError: Import is not an approved math capability",
            actions[0].observations,
        )
        self.assertIn("task state discarded", actions[0].observations)
        self.assertFalse(actions[0].is_final_answer)
        self.assertEqual(actions[1].action_output, "42")
        self.assertTrue(actions[1].is_final_answer)
        self.assertEqual(len(finals), 1)
        self.assertIsNone(agent.python_executor._process)

    def test_multistep_state_single_final_event_and_cleanup(self):
        agent = self.agent(
            [
                "<tool_call>x = 6</tool_call>",
                "<tool_call>answer = final_answer(x * 7)</tool_call>",
            ]
        )
        events = list(agent.run("task", stream=True))
        finals = [event for event in events if isinstance(event, FinalAnswerStep)]
        actions = [event for event in events if isinstance(event, ActionStep)]
        self.assertEqual(len(finals), 1)
        self.assertEqual(len(actions), 2)
        self.assertTrue(actions[-1].is_final_answer)
        self.assertEqual(actions[-1].action_output, "42")
        self.assertIn("42", finals[0].output)
        self.assertEqual(agent.model.calls, 2)
        self.assertIsNone(agent.python_executor._process)

    def test_errors_after_print_are_observations_and_clear_state(self):
        agent = self.agent(
            [
                "<tool_call>x = 7\nprint('before')\nmissing</tool_call>",
                "<tool_call>x</tool_call>",
                "\\boxed{7}",
            ]
        )
        output = agent.run("task")
        self.assertIn("Undefined variable: missing", output)
        self.assertIn("Undefined variable: x", output)
        self.assertIn("\\boxed{7}", output)
        self.assertIsNone(agent.python_executor._process)

    def test_stream_cancellation_and_inherited_cleanup(self):
        agent = self.agent(["<tool_call>x = 1</tool_call>"])
        stream = agent.run("task", stream=True)
        next(stream)
        process = agent.python_executor._process
        self.assertIsNotNone(process)
        stream.close()
        self.assertIsNotNone(process.poll())
        agent = self.agent(["<tool_call>x = 1</tool_call>"])
        stream = agent.run("task", stream=True)
        next(stream)
        process = agent.python_executor._process
        agent.cleanup()
        self.assertIsNotNone(process.poll())
        stream.close()

    def test_no_task_state_with_retained_conversation(self):
        agent = self.agent(["<tool_call>x = 1</tool_call>", "<tool_call>x</tool_call>"])
        agent.run("first", max_steps=1)
        agent.run("second", max_steps=1, reset=False)
        self.assertIn("Undefined variable: x", agent.memory.steps[-1].observations)

    def test_final_answer_checks_can_reject_and_retry(self):
        def check(answer, memory, agent):
            return answer == "42"

        agent = self.agent(
            [
                "<tool_call>final_answer(1)</tool_call>",
                "<tool_call>final_answer(42)</tool_call>",
            ],
            final_answer_checks=[check],
        )
        agent.run("task")
        self.assertEqual(agent.model.calls, 2)
        self.assertIsNotNone(agent.memory.steps[0].error)
        self.assertTrue(agent.memory.steps[1].is_final_answer)


if __name__ == "__main__":
    unittest.main()
