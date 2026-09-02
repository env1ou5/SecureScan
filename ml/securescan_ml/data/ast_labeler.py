"""AST-based labelers for the classes Semgrep does not cover.

Measured Semgrep coverage on held-out synthetic vulnerable samples (n=25/class,
rulesets `r/python.lang.security` + `p/security-audit`):

    COMMAND_INJECTION       25/25
    UNSAFE_DESERIALIZATION  25/25
    SQL_INJECTION           12/25
    XSS                      4/25
    HARDCODED_SECRET         0/25
    PATH_TRAVERSAL           0/25

Two classes are entirely uncovered, so mined real code would contain no
examples of them at all. Falling back to Bandit for those would reintroduce the
circularity the Semgrep choice was made to avoid (Bandit is the benchmark
baseline, proposal §10), so these are independent detectors instead.

Both are deliberately high-precision and low-recall. A missed vulnerable
function costs one training sample; a mislabeled safe function teaches the model
something false.
"""

from __future__ import annotations

import ast
import math
import re

from securescan_ml.labels import Label

# --------------------------------------------------------------------------
# Hardcoded secrets
# --------------------------------------------------------------------------

SECRET_NAME_RE = re.compile(
    r"(api[_-]?key|secret|password|passwd|pwd|token|credential|auth|"
    r"access[_-]?key|private[_-]?key|client[_-]?secret|bearer)",
    re.IGNORECASE,
)

# Recognizable credential formats. A match here is near-certain, regardless of
# the variable's name.
SECRET_VALUE_RE = re.compile(
    r"^(sk_live_[A-Za-z0-9]{16,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
    r"ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"AIza[0-9A-Za-z_-]{35}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)

# Values that look secret-shaped but are not credentials.
SECRET_ALLOWLIST = frozenset(
    {
        "",
        "none",
        "null",
        "changeme",
        "password",
        "secret",
        "token",
        "test",
        "example",
        "your_api_key_here",
        "xxx",
        "todo",
        "placeholder",
        "dummy",
        "fake",
        "sample",
        "<your key>",
        "redacted",
        "*",
        "**",
        "***",
    }
)


def shannon_entropy(value: str) -> float:
    """Bits per character. Random-looking credentials score high; words do not."""
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_secret_literal(name: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in SECRET_ALLOWLIST or len(value) < 8:
        return False
    # A format match is conclusive on its own.
    if SECRET_VALUE_RE.match(value):
        return True
    if not SECRET_NAME_RE.search(name):
        return False
    # Prose clears the entropy bar surprisingly easily -- "please enter your
    # password below" scores above 3.4 bits/char because it uses many distinct
    # characters. Real credentials are single opaque tokens, so require the
    # value to contain no whitespace before trusting entropy at all.
    if any(ch.isspace() for ch in value):
        return False
    # Reject prose-with-separators too (dotted paths, snake_case sentences):
    # a credential is mostly alphanumeric.
    alnum_ratio = sum(ch.isalnum() for ch in value) / len(value)
    if alnum_ratio < 0.8:
        return False
    return len(value) >= 16 and shannon_entropy(value) >= 3.4


def find_hardcoded_secrets(tree: ast.AST) -> list[int]:
    """Line numbers where a credential is assigned from a string literal."""
    hits: list[int] = []

    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.expr | None = None

        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.keyword) and node.arg:
            # e.g. connect(password="hunter2hunter2hunter2")
            if (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and _is_secret_literal(node.arg, node.value.value)
            ):
                hits.append(getattr(node.value, "lineno", 0))
            continue

        if value is None or not isinstance(value, ast.Constant):
            continue
        if not isinstance(value.value, str):
            continue

        for target in targets:
            name = (
                target.id
                if isinstance(target, ast.Name)
                else target.attr
                if isinstance(target, ast.Attribute)
                else ""
            )
            if name and _is_secret_literal(name, value.value):
                hits.append(value.lineno)
                break

    return sorted(set(hits))


# --------------------------------------------------------------------------
# Path traversal
# --------------------------------------------------------------------------

TAINT_ATTRS = frozenset({"args", "form", "GET", "POST", "json", "data", "params", "query_params"})
FILE_SINKS = frozenset({"open", "read_text", "read_bytes", "write_text", "write_bytes"})
PATH_BUILDERS = {("os", "path", "join"), ("os", "path", "abspath"), ("posixpath", "join")}

# Any of these in the function means containment was at least considered, so
# the detector stands down rather than risk a false positive.
GUARD_MARKERS = (
    "is_relative_to",
    "commonpath",
    "commonprefix",
    "startswith",
    "realpath",
    "safe_join",
    "secure_filename",
    "abspath",
    "resolve",
    "relpath",
    "basename",
)


def _attr_chain(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _tainted_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Parameters, plus locals assigned from a request-like source."""
    tainted = {a.arg for a in fn.args.args + fn.args.kwonlyargs if a.arg not in ("self", "cls")}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        for sub in ast.walk(node.value):
            if (
                isinstance(sub, ast.Attribute)
                and sub.attr in TAINT_ATTRS
                or isinstance(sub, ast.Name)
                and sub.id in tainted
            ):
                tainted.add(target.id)
    return tainted


def _flows_from(node: ast.AST, tainted: set[str]) -> bool:
    return any(isinstance(sub, ast.Name) and sub.id in tainted for sub in ast.walk(node))


def find_path_traversal(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[int]:
    """Tainted data reaching a filesystem sink with no containment check.

    Intraprocedural and conservative: it looks for taint flowing into a path
    that is opened, and gives up entirely if the function mentions any
    containment idiom.
    """
    source = ast.unparse(fn)
    if any(marker in source for marker in GUARD_MARKERS):
        return []

    tainted = _tainted_names(fn)
    if not tainted:
        return []

    # Names holding a path built from tainted input.
    path_names: set[str] = set()
    hits: list[int] = []

    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                value = node.value
                builds_path = False
                if isinstance(value, ast.Call) and _attr_chain(value.func) in PATH_BUILDERS:
                    builds_path = True
                elif isinstance(value, (ast.BinOp, ast.JoinedStr)):
                    # "root/" + x, or f"root/{x}"
                    builds_path = True
                if builds_path and _flows_from(value, tainted):
                    path_names.add(target.id)

        if isinstance(node, ast.Call):
            func_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if func_name not in FILE_SINKS or not node.args:
                continue
            arg = node.args[0]
            if (isinstance(arg, ast.Name) and arg.id in path_names) or (
                _flows_from(arg, tainted) and isinstance(arg, (ast.BinOp, ast.JoinedStr, ast.Call))
            ):
                hits.append(node.lineno)

    return sorted(set(hits))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def label_source(source: str) -> dict[int, Label]:
    """Return {line: label} for the classes Semgrep cannot cover."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    findings: dict[int, Label] = {}

    for line in find_hardcoded_secrets(tree):
        findings[line] = Label.HARDCODED_SECRET

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in find_path_traversal(node):
                findings.setdefault(line, Label.PATH_TRAVERSAL)

    return findings
