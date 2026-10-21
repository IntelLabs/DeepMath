"""A deliberately small math language, not a general Python sandbox.

Model source is interpreted, never compiled or evaluated. Library objects and
Python callables are not language values. Only the wrappers below can call SymPy;
they reject strings (including nested strings) before mathematical coercion.
This file is also the private, JSON-only worker entry point.
"""

import ast
import json
import math
import operator
import re
import sys
from dataclasses import dataclass
from types import MappingProxyType

import sympy as sp

MAX_SOURCE = 32_768
MAX_MESSAGE = 65_536
MAX_OUTPUT = 8_192
MAX_ITEMS = 4_096
MAX_STEPS = 100_000
MAX_ITERATIONS = 5_000
MAX_BITS = 4_096
MAX_POWER = 2_048
MAX_DEPTH = 64
MAX_NAMES = 256
MAX_SYMBOLS = 64
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")


class MathPolicyError(ValueError):
    """Source or values fall outside the supported mathematical language."""


class _FinishedError(Exception):
    def __init__(self, value):
        self.value = value


@dataclass(frozen=True)
class _Namespace:
    name: str


@dataclass(frozen=True)
class _Operation:
    name: str
    receiver: object = None


# These are language tokens, not module/function references.
CAPABILITIES = MappingProxyType(
    {
        "sympy": MappingProxyType(
            {
                **{
                    name: _Operation(name)
                    for name in (
                        "symbols",
                        "Symbol",
                        "Rational",
                        "Eq",
                        "Abs",
                        "solve",
                        "exp",
                        "sqrt",
                    )
                },
                "I": sp.I,
                "pi": sp.pi,
            }
        ),
        "numpy": MappingProxyType({"argmin": _Operation("argmin")}),
    }
)
BUILTINS = MappingProxyType(
    {name: _Operation(name) for name in ("range", "print", "final_answer", "abs", "len")}
)
BINARY = MappingProxyType(
    {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
)
COMPARE = MappingProxyType(
    {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }
)
ALLOWED_NODES = frozenset(
    {
        ast.Module,
        ast.Expr,
        ast.Assign,
        ast.AugAssign,
        ast.For,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.BinOp,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Call,
        ast.keyword,
        ast.Attribute,
        ast.Subscript,
        ast.ListComp,
        ast.comprehension,
        ast.Compare,
        ast.Import,
        ast.ImportFrom,
        ast.alias,
        *BINARY,
        *COMPARE,
    }
)


def _name(name):
    if not IDENTIFIER.fullmatch(name):
        raise MathPolicyError("Names must be bounded public ASCII identifiers")
    return name


def _scalar(value):
    if type(value) in (int, float, complex) or isinstance(value, sp.Expr):
        return value
    raise MathPolicyError("Expected a number or typed symbolic expression, never a string")


def _math_values(value):
    """Recursively reject strings, capabilities and foreign objects BEFORE SymPy calls."""
    if type(value) in (list, tuple):
        for item in value:
            _math_values(item)
    elif type(value) is dict:
        for key, item in value.items():
            _math_values(key)
            _math_values(item)
    elif isinstance(value, sp.Equality):
        return
    else:
        _scalar(value)


class MathInterpreter:
    """Stateful interpreter used only inside the resource-limited worker in production."""

    def __init__(self):
        self.variables = {}
        self.symbol_names = set()

    def _check(self, value, depth=0, budget=None):
        if budget is None:
            budget = [MAX_ITEMS]
        budget[0] -= 1
        if budget[0] < 0:
            raise MathPolicyError("Aggregate value size limit exceeded")
        if depth > MAX_DEPTH:
            raise MathPolicyError("Value nesting limit exceeded")
        if isinstance(value, (_Namespace, _Operation)) or value is None:
            return value
        if type(value) is bool:
            return value
        if type(value) is str:
            if len(value.encode("utf-8")) > MAX_OUTPUT:
                raise MathPolicyError("String limit exceeded")
        elif type(value) is int:
            if value.bit_length() > MAX_BITS:
                raise MathPolicyError("Integer size limit exceeded")
        elif type(value) in (float, complex):
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                raise MathPolicyError("Nonfinite numbers are not supported")
        elif type(value) in (list, tuple, dict, range):
            if len(value) > MAX_ITEMS:
                raise MathPolicyError("Collection size limit exceeded")
            for item in value:
                self._check(item, depth + 1, budget)
                if type(value) is dict:
                    self._check(value[item], depth + 1, budget)
        elif isinstance(value, sp.Basic):
            for item in sp.preorder_traversal(value):
                budget[0] -= 1
                if budget[0] < 0:
                    raise MathPolicyError("Symbolic complexity limit exceeded")
                if (
                    isinstance(item, sp.Rational)
                    and max(int(item.p).bit_length(), int(item.q).bit_length()) > MAX_BITS
                ):
                    raise MathPolicyError("Integer size limit exceeded")
                if item in (sp.oo, -sp.oo, sp.zoo, sp.nan):
                    raise MathPolicyError("Nonfinite numbers are not supported")
            if len(value.free_symbols) > MAX_SYMBOLS:
                raise MathPolicyError("Symbol limit exceeded")
        else:
            raise MathPolicyError("Unsupported value type")
        return value

    def _tick(self):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise MathPolicyError("Execution step limit exceeded")

    def _iterate(self, values):
        if type(values) not in (list, tuple, range, dict):
            raise MathPolicyError("Only bounded collections can be iterated")
        for value in values:
            self.iterations += 1
            if self.iterations > MAX_ITERATIONS:
                raise MathPolicyError("Iteration limit exceeded")
            yield value

    def _bind(self, target, value):
        self._check(value)
        if isinstance(target, ast.Name):
            name = _name(target.id)
            if name in BUILTINS:
                raise MathPolicyError("Cannot replace a built-in operation")
            self.variables[name] = value
            if len(self.variables) > MAX_NAMES:
                raise MathPolicyError("Variable limit exceeded")
        elif isinstance(target, (ast.Tuple, ast.List)):
            if type(value) not in (tuple, list) or len(target.elts) != len(value):
                raise MathPolicyError("Unpacking requires a matching tuple or list")
            for child, item in zip(target.elts, value):
                self._bind(child, item)
        else:
            raise MathPolicyError("Only local variable assignment is allowed")

    def execute(self, source):
        if (
            type(source) is not str
            or len(source) > MAX_SOURCE
            or len(source.encode("utf-8")) > MAX_SOURCE
        ):
            raise MathPolicyError("Source size limit exceeded")
        self.steps = self.iterations = 0
        self.logs = ""
        self.final = False
        tree = ast.parse(source, mode="exec")
        pending = [(tree, 0)]
        count = 0
        while pending:
            node, depth = pending.pop()
            count += 1
            if count > MAX_ITEMS or depth > MAX_DEPTH:
                raise MathPolicyError("AST size or depth limit exceeded")
            if type(node) not in ALLOWED_NODES:
                raise MathPolicyError(f"Unsupported syntax: {type(node).__name__}")
            if isinstance(node, ast.Name):
                _name(node.id)
            pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        result = None
        try:
            for statement in tree.body:
                result = self._statement(statement)
        except _FinishedError as finished:
            result = finished.value
            self.final = True
        output = self._render(result) if result is not None else ""
        return {"output": output, "logs": self.logs, "final": self.final}

    def _statement(self, node):
        self._tick()
        if isinstance(node, ast.Expr):
            return self._expression(node.value)
        if isinstance(node, ast.Assign):
            value = self._expression(node.value)
            for target in node.targets:
                self._bind(target, value)
        elif isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                raise MathPolicyError("Only local augmented assignment is allowed")
            value = self._binary(
                node.op, self._expression(node.target), self._expression(node.value)
            )
            self._bind(node.target, value)
        elif isinstance(node, ast.For):
            if node.orelse:
                raise MathPolicyError("Loop else is not supported")
            for item in self._iterate(self._expression(node.iter)):
                self._bind(node.target, item)
                for statement in node.body:
                    result = self._statement(statement)
                    if self.final:
                        return result
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in CAPABILITIES:
                    raise MathPolicyError("Only restricted sympy and numpy imports are supported")
                self._bind(ast.Name(id=alias.asname or alias.name), _Namespace(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in CAPABILITIES:
                raise MathPolicyError("Import is not an approved math namespace")
            for alias in node.names:
                if alias.name not in CAPABILITIES[node.module]:
                    raise MathPolicyError("Import is not an approved math capability")
                self._bind(
                    ast.Name(id=alias.asname or alias.name), CAPABILITIES[node.module][alias.name]
                )
        else:
            raise MathPolicyError("Unsupported statement")
        return None

    def _expression(self, node):
        self._tick()
        return self._check(self._evaluate_node(node))

    def _evaluate_node(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in BUILTINS:
                return BUILTINS[node.id]
            if node.id not in self.variables:
                raise MathPolicyError(f"Undefined variable: {node.id}")
            return self.variables[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [self._expression(item) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise MathPolicyError("Dictionary expansion is not supported")
            return {
                self._expression(k): self._expression(v) for k, v in zip(node.keys, node.values)
            }
        if isinstance(node, ast.BinOp):
            return self._binary(node.op, self._expression(node.left), self._expression(node.right))
        if isinstance(node, ast.UnaryOp):
            value = _scalar(self._expression(node.operand))
            return -value if isinstance(node.op, ast.USub) else +value
        if isinstance(node, ast.Compare):
            left = _scalar(self._expression(node.left))
            for op, child in zip(node.ops, node.comparators):
                right = _scalar(self._expression(child))
                if not bool(COMPARE[type(op)](left, right)):
                    return False
                left = right
            return True
        if isinstance(node, ast.Attribute):
            owner = self._expression(node.value)
            if isinstance(owner, _Namespace) and node.attr in CAPABILITIES[owner.name]:
                return CAPABILITIES[owner.name][node.attr]
            if isinstance(owner, (sp.Expr, sp.Equality)) and node.attr in ("subs", "as_real_imag"):
                return _Operation(node.attr, owner)
            raise MathPolicyError("Attribute is not an approved math capability")
        if isinstance(node, ast.Subscript):
            owner, key = self._expression(node.value), self._expression(node.slice)
            if type(owner) in (list, tuple, range) and type(key) is int:
                return owner[key]
            if type(owner) is dict:
                _scalar(key)
                return owner[key]
            raise MathPolicyError("Only typed collection indexing is supported")
        if isinstance(node, ast.ListComp):
            if len(node.generators) != 1 or node.generators[0].is_async:
                raise MathPolicyError("Only single bounded comprehensions are supported")
            generator = node.generators[0]
            values = self._expression(generator.iter)
            saved = self.variables.copy()
            result = []
            try:
                for item in self._iterate(values):
                    self._bind(generator.target, item)
                    if all(bool(self._expression(test)) for test in generator.ifs):
                        result.append(self._expression(node.elt))
                        if len(result) > MAX_ITEMS:
                            raise MathPolicyError("Collection size limit exceeded")
            finally:
                self.variables = saved
            return result
        if isinstance(node, ast.Call):
            operation = self._expression(node.func)
            if not isinstance(operation, _Operation):
                raise MathPolicyError("Only approved operations can be called")
            if node.keywords:
                raise MathPolicyError("Keyword arguments and callbacks are not supported")
            args = [self._expression(arg) for arg in node.args]
            return self._call(operation, args)
        raise MathPolicyError("Unsupported expression")

    def _binary(self, op, left, right):
        _scalar(left)
        _scalar(right)
        if isinstance(op, ast.Pow):
            if type(right) not in (int, float) and not isinstance(right, sp.Rational):
                raise MathPolicyError("Powers require a bounded numeric exponent")
            if abs(right) > MAX_POWER:
                raise MathPolicyError("Exponent limit exceeded")
            if type(left) is int or isinstance(left, sp.Rational):
                bits = (
                    int(left).bit_length()
                    if type(left) is int
                    else max(int(left.p).bit_length(), int(left.q).bit_length())
                )
                if bits * float(abs(right)) > MAX_BITS:
                    raise MathPolicyError("Power result size limit exceeded")
        return self._check(BINARY[type(op)](left, right))

    def _render(self, value):
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, (_Namespace, _Operation)):
                raise MathPolicyError("Capabilities cannot be rendered as data")
            if type(item) in (list, tuple):
                pending.extend(item)
            elif type(item) is dict:
                pending.extend(item.keys())
                pending.extend(item.values())
        text = str(value)
        if len(text.encode("utf-8")) > MAX_OUTPUT:
            raise MathPolicyError("Output size limit exceeded")
        return text

    def _call(self, operation, args):
        name = operation.name
        if name in ("symbols", "Symbol"):
            if len(args) != 1 or type(args[0]) is not str:
                raise MathPolicyError("Symbol declarations require one identifier string")
            names = args[0].replace(",", " ").split()
            if not names or (name == "Symbol" and len(names) != 1):
                raise MathPolicyError("Invalid symbol declaration")
            for item in names:
                _name(item)
            self.symbol_names.update(names)
            if len(names) > MAX_SYMBOLS or len(self.symbol_names) > MAX_SYMBOLS:
                raise MathPolicyError("Symbol limit exceeded")
            # Do not pass model strings to sympify, parse_expr, or symbols' mini-language.
            symbols = tuple(sp.Symbol(item) for item in names)
            return symbols[0] if len(symbols) == 1 else symbols
        if name in ("print", "final_answer"):
            if name == "final_answer":
                if len(args) != 1:
                    raise MathPolicyError("final_answer requires one value")
                raise _FinishedError(args[0])
            text = " ".join(self._render(arg) for arg in args) + "\n"
            if len((self.logs + text).encode("utf-8")) > MAX_OUTPUT:
                raise MathPolicyError("Print limit exceeded")
            self.logs += text
            return None
        if name == "len":
            if len(args) != 1 or type(args[0]) not in (list, tuple, dict, range):
                raise MathPolicyError("len requires a collection")
            return len(args[0])
        # Every mathematical wrapper must pass this recursive check before invoking a library.
        for arg in args:
            _math_values(arg)
        if name == "range":
            if not 1 <= len(args) <= 3 or any(type(arg) is not int for arg in args):
                raise MathPolicyError("range requires one to three integers")
            return self._check(range(*args))
        if name == "Rational":
            if len(args) != 2 or any(type(arg) is not int for arg in args):
                raise MathPolicyError("Rational requires two integers")
            if args[1] == 0:
                raise MathPolicyError("Rational denominator must be nonzero")
            return sp.Rational(*args)
        if name == "Eq" and len(args) == 2:
            return sp.Eq(*[_scalar(arg) for arg in args])
        if name in ("Abs", "abs", "sqrt", "exp") and len(args) == 1:
            function = {"Abs": sp.Abs, "abs": sp.Abs, "sqrt": sp.sqrt, "exp": sp.exp}[name]
            return function(_scalar(args[0]))
        if name == "solve" and len(args) == 2:
            equations, symbols = args
            equations = equations if type(equations) in (list, tuple) else [equations]
            symbols = symbols if type(symbols) in (list, tuple) else [symbols]
            if (
                not 1 <= len(equations) <= 16
                or not 1 <= len(symbols) <= 16
                or any(not isinstance(s, sp.Symbol) for s in symbols)
                or any(not isinstance(e, (sp.Expr, sp.Equality)) for e in equations)
            ):
                raise MathPolicyError("solve requires typed equations and symbols (at most 16)")
            return sp.solve(equations, symbols)
        if name == "subs" and operation.receiver is not None:
            if len(args) == 1 and type(args[0]) is dict:
                for key, value in args[0].items():
                    _scalar(key)
                    _scalar(value)
                return operation.receiver.subs(args[0])
            if len(args) == 2:
                return operation.receiver.subs(*[_scalar(arg) for arg in args])
        if name == "as_real_imag" and not args and isinstance(operation.receiver, sp.Expr):
            return operation.receiver.as_real_imag()
        if name == "argmin" and len(args) == 1 and type(args[0]) in (list, tuple):
            values = args[0]
            if not values:
                raise MathPolicyError("argmin requires a nonempty collection")
            for value in values:
                _scalar(value)
                if isinstance(value, sp.Expr) and (value.free_symbols or value.is_real is not True):
                    raise MathPolicyError("argmin requires real numeric values")
            return min(range(len(values)), key=values.__getitem__)
        raise MathPolicyError("Unsupported operation or argument signature")


def _worker_main():
    # A fresh process, not forked model/GPU state. Fail closed if limits cannot be set.
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    except (ImportError, OSError, ValueError) as error:
        print(
            json.dumps(
                {"ok": False, "error": f"Required worker resource limits unavailable: {error}"}
            ),
            flush=True,
        )
        return
    interpreter = MathInterpreter()
    while True:
        request = sys.stdin.buffer.readline(MAX_MESSAGE + 1)
        if not request:
            return
        if len(request) > MAX_MESSAGE or not request.endswith(b"\n"):
            return
        try:
            data = json.loads(request)
            if type(data) is not dict or set(data) != {"source"}:
                raise MathPolicyError("Invalid request")
            result = interpreter.execute(data["source"])
            reply = {"ok": True, **result}
        except Exception as error:
            # Drop partially mutated state on all failures, not just on timeouts.
            interpreter = MathInterpreter()
            reply = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:1000]}"}
        payload = json.dumps(reply, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        if len(payload) > MAX_MESSAGE:
            return
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    _worker_main()
