"""
GreenCode Static Analyzer
=========================
Extracts code complexity features from Python source code using AST parsing 
and radon metrics. Produces a feature vector used by the prediction engine.
"""

import ast
import math
from collections import defaultdict
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze


# ──────────────────────────────────────────────────────────────────────────────
# AST Feature Visitor
# ──────────────────────────────────────────────────────────────────────────────

class CodeFeatureVisitor(ast.NodeVisitor):
    """Walks the AST to extract structural features from Python code."""

    IO_FUNCTIONS = frozenset([
        "open", "read", "write", "readlines", "writelines",
        "print", "input", "readline",
    ])
    NETWORK_MODULES = frozenset([
        "requests", "urllib", "http", "socket", "aiohttp",
        "httpx", "grpc", "websocket",
    ])
    HEAVY_MATH = frozenset([
        "numpy", "scipy", "pandas", "torch", "tensorflow",
        "sklearn", "xgboost", "lightgbm",
    ])

    def __init__(self):
        # Counters
        self.function_count = 0
        self.class_count = 0
        self.loop_count = 0
        self.max_loop_depth = 0
        self.nested_loop_count = 0
        self.conditional_count = 0
        self.try_except_count = 0
        self.io_operations = 0
        self.network_calls = 0
        self.list_comprehensions = 0
        self.dict_comprehensions = 0
        self.generator_expressions = 0
        self.lambda_count = 0
        self.recursion_candidates = 0
        self.heavy_math_imports = 0
        self.decorator_count = 0
        self.global_variables = 0
        self.assert_count = 0
        self.yield_count = 0
        self.await_count = 0

        # Internal state
        self._current_loop_depth = 0
        self._function_names: set[str] = set()
        self._call_targets: list[str] = []
        self._imported_modules: set[str] = set()

    # ── Structural nodes ─────────────────────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_count += 1
        self.decorator_count += len(node.decorator_list)
        self._function_names.add(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_count += 1
        self.decorator_count += len(node.decorator_list)
        self.generic_visit(node)

    # ── Loops ────────────────────────────────────────────────────────────

    def _visit_loop(self, node):
        self.loop_count += 1
        self._current_loop_depth += 1
        if self._current_loop_depth > 1:
            self.nested_loop_count += 1
        self.max_loop_depth = max(self.max_loop_depth, self._current_loop_depth)
        self.generic_visit(node)
        self._current_loop_depth -= 1

    def visit_For(self, node):
        self._visit_loop(node)

    def visit_While(self, node):
        self._visit_loop(node)

    visit_AsyncFor = visit_For

    # ── Conditionals & error handling ────────────────────────────────────

    def visit_If(self, node):
        self.conditional_count += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.try_except_count += 1
        self.generic_visit(node)

    visit_TryStar = visit_Try  # Python 3.11+

    # ── Comprehensions & generators ──────────────────────────────────────

    def visit_ListComp(self, node):
        self.list_comprehensions += 1
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.dict_comprehensions += 1
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.list_comprehensions += 1  # group with list comps
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.generator_expressions += 1
        self.generic_visit(node)

    # ── Calls ────────────────────────────────────────────────────────────

    def visit_Call(self, node):
        name = self._resolve_call_name(node)
        if name:
            self._call_targets.append(name)
            base = name.split(".")[0]
            if base in self.IO_FUNCTIONS or name.split(".")[-1] in self.IO_FUNCTIONS:
                self.io_operations += 1
        self.generic_visit(node)

    # ── Imports ───────────────────────────────────────────────────────────

    def visit_Import(self, node):
        for alias in node.names:
            mod = alias.name.split(".")[0]
            self._imported_modules.add(mod)
            if mod in self.NETWORK_MODULES:
                self.network_calls += 1
            if mod in self.HEAVY_MATH:
                self.heavy_math_imports += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            mod = node.module.split(".")[0]
            self._imported_modules.add(mod)
            if mod in self.NETWORK_MODULES:
                self.network_calls += 1
            if mod in self.HEAVY_MATH:
                self.heavy_math_imports += 1
        self.generic_visit(node)

    # ── Misc nodes ───────────────────────────────────────────────────────

    def visit_Lambda(self, node):
        self.lambda_count += 1
        self.generic_visit(node)

    def visit_Global(self, node):
        self.global_variables += len(node.names)
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.assert_count += 1
        self.generic_visit(node)

    def visit_Yield(self, node):
        self.yield_count += 1
        self.generic_visit(node)

    visit_YieldFrom = visit_Yield

    def visit_Await(self, node):
        self.await_count += 1
        self.generic_visit(node)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_call_name(node: ast.Call) -> str | None:
        """Best-effort name resolution for a Call node."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = []
            current = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return None

    def finalize(self):
        """Post-processing after full tree walk."""
        # Detect potential recursion (function calls its own name)
        for name in self._call_targets:
            if name in self._function_names:
                self.recursion_candidates += 1


# ──────────────────────────────────────────────────────────────────────────────
# Radon metrics helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_cyclomatic_complexity(code: str) -> dict:
    """Average and max cyclomatic complexity via radon."""
    try:
        blocks = cc_visit(code)
        if not blocks:
            return {"avg_complexity": 1.0, "max_complexity": 1, "total_complexity": 1}
        complexities = [b.complexity for b in blocks]
        return {
            "avg_complexity": round(sum(complexities) / len(complexities), 2),
            "max_complexity": max(complexities),
            "total_complexity": sum(complexities),
        }
    except Exception:
        return {"avg_complexity": 1.0, "max_complexity": 1, "total_complexity": 1}


def _safe_halstead(code: str) -> dict:
    """Halstead volume, difficulty, and effort."""
    try:
        report = h_visit(code)
        # h_visit returns a named tuple for the total
        total = report.total if hasattr(report, "total") else report
        return {
            "halstead_volume": round(getattr(total, "volume", 0), 2),
            "halstead_difficulty": round(getattr(total, "difficulty", 0), 2),
            "halstead_effort": round(getattr(total, "effort", 0), 2),
            "halstead_bugs": round(getattr(total, "bugs", 0), 4),
        }
    except Exception:
        return {
            "halstead_volume": 0,
            "halstead_difficulty": 0,
            "halstead_effort": 0,
            "halstead_bugs": 0,
        }


def _safe_maintainability(code: str) -> float:
    """Maintainability index (0-100, higher ⇒ easier to maintain)."""
    try:
        return round(mi_visit(code, multi=False), 2)
    except Exception:
        return 50.0


def _safe_raw_metrics(code: str) -> dict:
    """Raw line-count metrics from radon."""
    try:
        raw = analyze(code)
        return {
            "loc": raw.loc,
            "lloc": raw.lloc,
            "sloc": raw.sloc,
            "comments": raw.comments,
            "blank_lines": raw.blank,
        }
    except Exception:
        lines = code.strip().splitlines()
        return {
            "loc": len(lines),
            "lloc": len(lines),
            "sloc": len([l for l in lines if l.strip()]),
            "comments": 0,
            "blank_lines": 0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def analyze_code(code: str) -> dict:
    """
    Analyze a Python code string and return a comprehensive feature dictionary.

    Parameters
    ----------
    code : str
        Python source code to analyze.

    Returns
    -------
    dict
        Feature dictionary with ~30 features covering structure, complexity,
        I/O patterns, and code quality metrics.
    """
    # Parse AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"SyntaxError: {e.msg} (line {e.lineno})"}

    # Walk AST
    visitor = CodeFeatureVisitor()
    visitor.visit(tree)
    visitor.finalize()

    # Radon metrics
    complexity = _safe_cyclomatic_complexity(code)
    halstead = _safe_halstead(code)
    maintainability = _safe_maintainability(code)
    raw = _safe_raw_metrics(code)

    # Compute a simple "computational intensity" score (0-100)
    intensity = min(100, (
        visitor.loop_count * 10
        + visitor.nested_loop_count * 20
        + visitor.max_loop_depth * 15
        + visitor.heavy_math_imports * 15
        + visitor.recursion_candidates * 10
        + complexity["avg_complexity"] * 3
    ))

    # Determine dominant workload type
    if visitor.network_calls > 0:
        workload_type = "network"
    elif visitor.io_operations >= 3:
        workload_type = "io_heavy"
    elif visitor.heavy_math_imports > 0 or intensity > 50:
        workload_type = "cpu_heavy"
    elif raw["sloc"] < 10:
        workload_type = "trivial"
    else:
        workload_type = "mixed"

    features = {
        # ── Structure ─────────────────────────────────────────────────
        "function_count": visitor.function_count,
        "class_count": visitor.class_count,
        "loop_count": visitor.loop_count,
        "max_loop_depth": visitor.max_loop_depth,
        "nested_loop_count": visitor.nested_loop_count,
        "conditional_count": visitor.conditional_count,
        "try_except_count": visitor.try_except_count,
        "list_comprehensions": visitor.list_comprehensions,
        "dict_comprehensions": visitor.dict_comprehensions,
        "generator_expressions": visitor.generator_expressions,
        "lambda_count": visitor.lambda_count,
        "decorator_count": visitor.decorator_count,
        "recursion_candidates": visitor.recursion_candidates,
        # ── I/O & Network ────────────────────────────────────────────
        "io_operations": visitor.io_operations,
        "network_calls": visitor.network_calls,
        "heavy_math_imports": visitor.heavy_math_imports,
        # ── Async ────────────────────────────────────────────────────
        "yield_count": visitor.yield_count,
        "await_count": visitor.await_count,
        # ── Raw metrics ──────────────────────────────────────────────
        **raw,
        # ── Complexity ───────────────────────────────────────────────
        **complexity,
        **halstead,
        "maintainability_index": maintainability,
        # ── Derived ──────────────────────────────────────────────────
        "computational_intensity": round(intensity, 2),
        "workload_type": workload_type,
    }

    return features


def analyze_file(filepath: str) -> dict:
    """Convenience wrapper: read a .py file and analyze it."""
    with open(filepath, "r", encoding="utf-8") as f:
        return analyze_code(f.read())


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    sample = """
import numpy as np

def matrix_multiply(a, b):
    result = np.dot(a, b)
    for i in range(len(result)):
        for j in range(len(result[0])):
            result[i][j] = round(result[i][j], 2)
    return result

def save_results(data, filename):
    with open(filename, 'w') as f:
        for row in data:
            f.write(','.join(str(x) for x in row))
            f.write('\\n')
"""
    result = analyze_code(sample)
    print(json.dumps(result, indent=2))
