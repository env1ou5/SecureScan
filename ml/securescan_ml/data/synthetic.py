"""Compositional synthetic corpus generator (the Juliet-equivalent source).

Real Python CVE data is scarce, so vulnerable examples are largely synthesized.
The danger is obvious and stated in proposal §5: a model trained on templated
code learns the template. This generator fights that by composing along five
independent axes so that no two samples share surface form:

    taint source x sink variant x wrapper structure x identifier pool x noise

A template that emitted `query = "SELECT ..." + user_id` every time would teach
the model to look for that exact string. Sampling the library (sqlite3 /
psycopg2 / MySQLdb / SQLAlchemy), the taint source (Flask args / Django GET /
argv / stdin), the surrounding control flow, and every identifier means the
invariant across samples is the *data flow*, which is the thing worth learning.

It is still synthetic, and the honest evaluation is on mined real code
(`mine.py`). This source exists for volume and class balance.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from securescan_ml.data.schema import VulnRecord
from securescan_ml.labels import Label

# --------------------------------------------------------------------------
# Identifier pools -- every sample draws fresh names so the model cannot key
# on a variable being called `user_id`.
# --------------------------------------------------------------------------

FUNCTION_VERBS = [
    "get",
    "fetch",
    "load",
    "read",
    "lookup",
    "find",
    "resolve",
    "handle",
    "process",
    "render",
    "build",
    "collect",
    "retrieve",
    "select",
    "query",
    "serve",
    "export",
    "import",
    "sync",
    "check",
    "update",
    "apply",
]
FUNCTION_NOUNS = [
    "user",
    "account",
    "record",
    "profile",
    "order",
    "invoice",
    "session",
    "document",
    "report",
    "asset",
    "item",
    "entry",
    "config",
    "template",
    "message",
    "ticket",
    "job",
    "task",
    "product",
    "customer",
    "node",
]
PARAM_NAMES = [
    "user_id",
    "record_id",
    "name",
    "key",
    "target",
    "ident",
    "slug",
    "ref",
    "token",
    "value",
    "arg",
    "param",
    "lookup_key",
    "resource",
    "handle",
]
LOCAL_NAMES = [
    "result",
    "data",
    "row",
    "value",
    "output",
    "payload",
    "content",
    "buf",
    "raw",
    "obj",
    "item",
    "response",
    "chunk",
    "blob",
]
TABLE_NAMES = [
    "users",
    "accounts",
    "orders",
    "records",
    "sessions",
    "documents",
    "profiles",
    "invoices",
    "items",
    "customers",
    "products",
    "audit_log",
]
COLUMN_NAMES = ["id", "uuid", "username", "email", "slug", "external_id", "ref"]


@dataclass
class Ctx:
    """Freshly drawn names for one generated sample."""

    rng: random.Random
    func: str
    param: str
    local: str
    table: str
    column: str
    indent: str = "    "

    @classmethod
    def draw(cls, rng: random.Random) -> Ctx:
        return cls(
            rng=rng,
            func=f"{rng.choice(FUNCTION_VERBS)}_{rng.choice(FUNCTION_NOUNS)}",
            param=rng.choice(PARAM_NAMES),
            local=rng.choice(LOCAL_NAMES),
            table=rng.choice(TABLE_NAMES),
            column=rng.choice(COLUMN_NAMES),
        )

    def pick(self, options: list) -> object:
        return self.rng.choice(options)


# --------------------------------------------------------------------------
# Taint sources -- where attacker-controlled data enters.
# --------------------------------------------------------------------------

TAINT_SOURCES: list[tuple[str, str]] = [
    ("flask", "request.args.get({p!r}, '')"),
    ("flask", "request.form[{p!r}]"),
    ("flask", "request.json.get({p!r})"),
    ("django", "request.GET.get({p!r}, '')"),
    ("django", "request.POST[{p!r}]"),
    ("argv", "sys.argv[1]"),
    ("stdin", "input('enter value: ')"),
    ("env", "os.environ.get({p!r}, '')"),
    ("param", None),  # taint arrives as the function parameter itself
]

IMPORTS_FOR_SOURCE = {
    "flask": "from flask import request",
    "django": "from django.http import HttpRequest  # noqa: F401",
    "argv": "import sys",
    "stdin": None,  # builtin
    "env": "import os",
    "param": None,
}


def taint_expr(ctx: Ctx) -> tuple[str, str | None, str]:
    """Return (expression yielding tainted data, needed import, source kind)."""
    kind, template = ctx.pick(TAINT_SOURCES)  # type: ignore[misc]
    if template is None:
        return ctx.param, None, kind
    return template.format(p=ctx.param), IMPORTS_FOR_SOURCE[kind], kind


# --------------------------------------------------------------------------
# Sink templates. Each returns (body_lines, imports) and is written twice:
# once building the sink unsafely, once safely. The safe variant is a genuine
# fix, not a cosmetic change, so SAFE samples are not trivially separable.
# --------------------------------------------------------------------------

SinkFn = Callable[[Ctx, str, bool], tuple[list[str], list[str]]]


def sink_sql(ctx: Ctx, tainted: str, vulnerable: bool) -> tuple[list[str], list[str]]:
    lib = ctx.pick(["sqlite3", "psycopg2", "MySQLdb", "sqlalchemy"])
    placeholder = {"sqlite3": "?", "psycopg2": "%s", "MySQLdb": "%s", "sqlalchemy": ":val"}[lib]
    imports = [f"import {lib}"] if lib != "sqlalchemy" else ["from sqlalchemy import text"]
    base = f"SELECT * FROM {ctx.table} WHERE {ctx.column} = "

    if vulnerable:
        style = ctx.pick(["concat", "fstring", "percent", "format"])
        if style == "concat":
            build = f'{ctx.local}_sql = "{base}" + str({tainted})'
        elif style == "fstring":
            build = f'{ctx.local}_sql = f"{base}{{{tainted}}}"'
        elif style == "percent":
            build = f'{ctx.local}_sql = "{base}%s" % ({tainted},)'
        else:
            build = f'{ctx.local}_sql = "{base}{{}}".format({tainted})'
        exec_line = (
            f"return connection.execute(text({ctx.local}_sql))"
            if lib == "sqlalchemy"
            else f"return cursor.execute({ctx.local}_sql)"
        )
        return [build, exec_line], imports

    if lib == "sqlalchemy":
        return [
            f'{ctx.local}_sql = text("{base}{placeholder}")',
            f"return connection.execute({ctx.local}_sql, {{'val': {tainted}}})",
        ], imports
    return [
        f'{ctx.local}_sql = "{base}{placeholder}"',
        f"return cursor.execute({ctx.local}_sql, ({tainted},))",
    ], imports


def sink_command(ctx: Ctx, tainted: str, vulnerable: bool) -> tuple[list[str], list[str]]:
    tool = ctx.pick(["ping", "git", "convert", "ffmpeg", "curl", "tar"])
    if vulnerable:
        style = ctx.pick(["system", "shell_run", "popen", "check_output"])
        if style == "system":
            return [f'return os.system("{tool} " + {tainted})'], ["import os"]
        if style == "shell_run":
            return [
                f'return subprocess.run(f"{tool} {{{tainted}}}", shell=True, capture_output=True)'
            ], ["import subprocess"]
        if style == "popen":
            return [f'return subprocess.Popen("{tool} " + {tainted}, shell=True)'], [
                "import subprocess"
            ]
        return [f'return subprocess.check_output("{tool} %s" % {tainted}, shell=True)'], [
            "import subprocess"
        ]
    return [
        f'return subprocess.run(["{tool}", str({tainted})], shell=False, capture_output=True)'
    ], ["import subprocess"]


def sink_path(ctx: Ctx, tainted: str, vulnerable: bool) -> tuple[list[str], list[str]]:
    root = ctx.pick(["/var/data/uploads", "/srv/files", "./storage", "/opt/app/media"])
    if vulnerable:
        style = ctx.pick(["join", "concat", "fstring"])
        if style == "join":
            body = [f'{ctx.local}_path = os.path.join("{root}", {tainted})']
        elif style == "concat":
            body = [f'{ctx.local}_path = "{root}/" + {tainted}']
        else:
            body = [f'{ctx.local}_path = f"{root}/{{{tainted}}}"']
        body.append(f'with open({ctx.local}_path, "rb") as fh:')
        body.append(f"{ctx.indent}return fh.read()")
        return body, ["import os"]
    return [
        f'{ctx.local}_root = Path("{root}").resolve()',
        f"{ctx.local}_path = ({ctx.local}_root / {tainted}).resolve()",
        f"if not {ctx.local}_path.is_relative_to({ctx.local}_root):",
        f'{ctx.indent}raise ValueError("path escapes the storage root")',
        f'with open({ctx.local}_path, "rb") as fh:',
        f"{ctx.indent}return fh.read()",
    ], ["from pathlib import Path"]


def sink_deserialize(ctx: Ctx, tainted: str, vulnerable: bool) -> tuple[list[str], list[str]]:
    if vulnerable:
        style = ctx.pick(["pickle", "yaml", "marshal", "shelve"])
        if style == "pickle":
            return [f"return pickle.loads({tainted})"], ["import pickle"]
        if style == "yaml":
            return [f"return yaml.load({tainted}, Loader=yaml.Loader)"], ["import yaml"]
        if style == "marshal":
            return [f"return marshal.loads({tainted})"], ["import marshal"]
        return [f"return dill.loads({tainted})"], ["import dill"]
    style = ctx.pick(["json", "safe_yaml"])
    if style == "json":
        return [f"return json.loads({tainted})"], ["import json"]
    return [f"return yaml.safe_load({tainted})"], ["import yaml"]


def sink_xss(ctx: Ctx, tainted: str, vulnerable: bool) -> tuple[list[str], list[str]]:
    if vulnerable:
        style = ctx.pick(["fstring_response", "mark_safe", "concat", "format_html"])
        if style == "fstring_response":
            return [f"return HttpResponse(f\"<div class='card'>{{{tainted}}}</div>\")"], [
                "from django.http import HttpResponse"
            ]
        if style == "mark_safe":
            return [f'return mark_safe("<span>" + {tainted} + "</span>")'], [
                "from django.utils.safestring import mark_safe"
            ]
        if style == "concat":
            return [f'return "<p>" + {tainted} + "</p>"'], []
        return [f'return Markup("<b>{{}}</b>").format({tainted})'], [
            "from markupsafe import Markup"
        ]
    style = ctx.pick(["escape", "render"])
    if style == "escape":
        return [f'return "<p>" + html.escape({tainted}) + "</p>"'], ["import html"]
    return [f'return render(request, "{ctx.table}.html", {{"value": {tainted}}})'], [
        "from django.shortcuts import render"
    ]


def sink_secret(ctx: Ctx, tainted: str, vulnerable: bool) -> tuple[list[str], list[str]]:
    """Hardcoded secrets do not involve taint -- the credential IS the defect."""
    kind = ctx.pick(["api_key", "password", "token", "secret_key", "aws_secret"])
    fake = "".join(ctx.rng.choice("abcdef0123456789") for _ in range(32))
    if vulnerable:
        prefix = ctx.pick(["sk_live_", "AKIA", "ghp_", "xoxb-", ""])
        return [
            f'{kind.upper()} = "{prefix}{fake}"',
            f"client = ServiceClient(credential={kind.upper()})",
            "return client.connect()",
        ], ["from services import ServiceClient"]
    return [
        f'{kind.upper()} = os.environ["{kind.upper()}"]',
        f"client = ServiceClient(credential={kind.upper()})",
        "return client.connect()",
    ], ["import os", "from services import ServiceClient"]


SINKS: dict[Label, SinkFn] = {
    Label.SQL_INJECTION: sink_sql,
    Label.COMMAND_INJECTION: sink_command,
    Label.PATH_TRAVERSAL: sink_path,
    Label.UNSAFE_DESERIALIZATION: sink_deserialize,
    Label.XSS: sink_xss,
    Label.HARDCODED_SECRET: sink_secret,
}


# --------------------------------------------------------------------------
# Wrapper structures -- the control flow the sink is embedded in. Without
# these, every sample would be a flat three-line function and the model would
# learn shape rather than semantics.
# --------------------------------------------------------------------------

NOISE_LINES = [
    "logger.debug('processing request')",
    "start = time.monotonic()",
    "metrics.increment('handler.calls')",
    "if not isinstance({p}, str):\n        {p} = str({p})",
    "cache_key = f'{f}:{{{p}}}'",
    "attempts = 0",
]


def _indent(lines: list[str], levels: int = 1) -> list[str]:
    pad = "    " * levels
    out = []
    for line in lines:
        out.extend(pad + part if part.strip() else "" for part in line.split("\n"))
    return out


def wrap(ctx: Ctx, body: list[str], assign: str | None) -> list[str]:
    """Embed the sink in one of several surrounding structures."""
    structure = ctx.pick(["plain", "try", "guard", "loop", "docstring", "nested_if"])
    inner: list[str] = []
    if assign:
        inner.append(assign)
    inner.extend(body)

    if structure == "plain":
        return inner
    if structure == "docstring":
        return [f'"""{ctx.pick(FUNCTION_VERBS).title()} a {ctx.pick(FUNCTION_NOUNS)}."""'] + inner
    if structure == "try":
        return (
            ["try:"]
            + _indent(inner)
            + ["except Exception as exc:", "    logger.warning('failed: %s', exc)", "    raise"]
        )
    if structure == "guard":
        return [f"if not {ctx.param}:", "    return None"] + inner
    if structure == "loop":
        return ["for attempt in range(3):"] + _indent(inner) + ["    break", "return None"]
    return [f"if {ctx.param} is not None:"] + _indent(inner) + ["return None"]


def _render_function(ctx: Ctx, body: list[str], imports: list[str], is_async: bool) -> str:
    header_style = ctx.pick(["plain", "method", "decorated", "typed"])
    kw = "async def" if is_async else "def"

    lines: list[str] = []
    seen: set[str] = set()
    for imp in imports:
        if imp == "":
            continue
        if imp not in seen:
            seen.add(imp)
            lines.append(imp)
    if lines:
        lines.append("")

    if header_style == "method":
        lines.append(f"class {ctx.pick(FUNCTION_NOUNS).title()}Service:")
        lines.append(f"    {kw} {ctx.func}(self, {ctx.param}):")
        lines.extend(_indent(body, 2))
    elif header_style == "decorated":
        lines.append(f'@route("/{ctx.table}/<{ctx.param}>")')
        lines.append(f"{kw} {ctx.func}({ctx.param}):")
        lines.extend(_indent(body))
    elif header_style == "typed":
        lines.append(f"{kw} {ctx.func}({ctx.param}: str) -> object:")
        lines.extend(_indent(body))
    else:
        lines.append(f"{kw} {ctx.func}({ctx.param}):")
        lines.extend(_indent(body))

    return "\n".join(lines) + "\n"


def generate_sample(label: Label, rng: random.Random, vulnerable: bool) -> tuple[str, str]:
    """Generate one function. Returns (code, function_name)."""
    ctx = Ctx.draw(rng)
    # `logger` is referenced by several noise/wrapper lines, so bind it.
    imports: list[str] = ["import logging", "", "logger = logging.getLogger(__name__)"]

    if label is Label.HARDCODED_SECRET:
        body, sink_imports = SINKS[label](ctx, "", vulnerable)
        assign = None
    else:
        tainted, taint_import, _kind = taint_expr(ctx)
        if tainted == ctx.param:
            assign = None
            sink_input = ctx.param
        else:
            assign = f"{ctx.local}_input = {tainted}"
            sink_input = f"{ctx.local}_input"
        if taint_import:
            imports.append(taint_import)
        body, sink_imports = SINKS[label](ctx, sink_input, vulnerable)

    imports.extend(sink_imports)

    if rng.random() < 0.35:
        noise = rng.choice(NOISE_LINES).format(p=ctx.param, f=ctx.func)
        body = noise.split("\n") + body
        if "time." in noise:
            imports.append("import time")

    wrapped = wrap(ctx, body, assign)
    is_async = rng.random() < 0.15
    code = _render_function(ctx, wrapped, imports, is_async)
    return code, ctx.func


def generate_corpus(
    per_class: int = 900,
    safe_ratio: float = 1.6,
    seed: int = 1337,
    n_pseudo_repos: int = 60,
) -> list[VulnRecord]:
    """Build the synthetic half of the corpus.

    Samples are assigned to pseudo-repository ids so that repository-level
    splitting has something to work with. Generation is seeded per repository,
    so a repository's samples are correlated the way real ones are -- which is
    exactly the leakage the split is designed to contain.
    """
    rng = random.Random(seed)
    records: list[VulnRecord] = []
    repos = [f"synthetic/pkg_{i:03d}" for i in range(n_pseudo_repos)]

    for label in SINKS:
        for i in range(per_class):
            repo = rng.choice(repos)
            code, name = generate_sample(label, rng, vulnerable=True)
            records.append(
                VulnRecord(
                    repository_id=repo,
                    file_path=f"{repo.split('/')[-1]}/{name}.py",
                    language="python",
                    code=code,
                    vulnerability_type=label,
                    start_line=1,
                    end_line=code.count("\n"),
                    source="juliet",
                    is_synthetic=True,
                    metadata={"generator": "synthetic-v1", "variant": i},
                )
            )

    n_safe = int(per_class * len(SINKS) * safe_ratio / len(SINKS))
    for label in SINKS:
        for _ in range(n_safe):
            repo = rng.choice(repos)
            code, name = generate_sample(label, rng, vulnerable=False)
            records.append(
                VulnRecord(
                    repository_id=repo,
                    file_path=f"{repo.split('/')[-1]}/{name}.py",
                    language="python",
                    code=code,
                    vulnerability_type=Label.SAFE,
                    start_line=1,
                    end_line=code.count("\n"),
                    source="juliet",
                    is_synthetic=True,
                    metadata={"generator": "synthetic-v1", "safe_counterpart": label.value},
                )
            )

    rng.shuffle(records)
    return records
