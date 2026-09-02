"""Vulnerability taxonomy, severity mapping, and remediation templates.

Single source of truth for the label set. The backend imports from here rather
than redefining labels, so the model head, the database enum, and the dashboard
can never drift apart.

Severity is NOT predicted by the model (decision D2-derived, proposal §2). The
model outputs a class and a calibrated confidence; severity is a static property
of the class. Do not add a severity head.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Label(str, Enum):
    SAFE = "SAFE"
    SQL_INJECTION = "SQL_INJECTION"
    XSS = "XSS"
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    COMMAND_INJECTION = "COMMAND_INJECTION"
    UNSAFE_DESERIALIZATION = "UNSAFE_DESERIALIZATION"
    HARDCODED_SECRET = "HARDCODED_SECRET"


class Severity(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Ordering is the model's output layer ordering. Append-only: inserting a label
# in the middle invalidates every existing checkpoint.
LABEL_ORDER: tuple[Label, ...] = (
    Label.SAFE,
    Label.SQL_INJECTION,
    Label.XSS,
    Label.PATH_TRAVERSAL,
    Label.COMMAND_INJECTION,
    Label.UNSAFE_DESERIALIZATION,
    Label.HARDCODED_SECRET,
)

NUM_LABELS = len(LABEL_ORDER)
LABEL_TO_ID: dict[Label, int] = {lab: i for i, lab in enumerate(LABEL_ORDER)}
ID_TO_LABEL: dict[int, Label] = {i: lab for lab, i in LABEL_TO_ID.items()}

SEVERITY: dict[Label, Severity] = {
    Label.SAFE: Severity.NONE,
    Label.SQL_INJECTION: Severity.CRITICAL,
    Label.COMMAND_INJECTION: Severity.CRITICAL,
    Label.UNSAFE_DESERIALIZATION: Severity.HIGH,
    Label.PATH_TRAVERSAL: Severity.HIGH,
    Label.HARDCODED_SECRET: Severity.HIGH,
    Label.XSS: Severity.MEDIUM,
}

# Maps CWE identifiers found in CVEFixes / Juliet metadata onto our taxonomy.
# Anything unmapped is dropped during normalization rather than guessed at --
# a wrong label is worse than a missing sample.
CWE_TO_LABEL: dict[str, Label] = {
    "CWE-89": Label.SQL_INJECTION,
    "CWE-564": Label.SQL_INJECTION,
    "CWE-79": Label.XSS,
    "CWE-80": Label.XSS,
    "CWE-116": Label.XSS,
    "CWE-22": Label.PATH_TRAVERSAL,
    "CWE-23": Label.PATH_TRAVERSAL,
    "CWE-36": Label.PATH_TRAVERSAL,
    "CWE-78": Label.COMMAND_INJECTION,
    "CWE-77": Label.COMMAND_INJECTION,
    "CWE-88": Label.COMMAND_INJECTION,
    "CWE-502": Label.UNSAFE_DESERIALIZATION,
    "CWE-798": Label.HARDCODED_SECRET,
    "CWE-259": Label.HARDCODED_SECRET,
    "CWE-321": Label.HARDCODED_SECRET,
}


@dataclass(frozen=True)
class Remediation:
    """A templated fix suggestion.

    Templated, never generated, and never applied automatically (proposal §9).
    A wrong prediction can surface an irrelevant suggestion; it must never be
    able to rewrite a user's source.
    """

    title: str
    explanation: str
    unsafe_example: str
    safe_example: str


REMEDIATIONS: dict[Label, Remediation] = {
    Label.SQL_INJECTION: Remediation(
        title="Use parameterized queries",
        explanation=(
            "Building SQL by concatenating user input lets an attacker alter the "
            "query's structure. Parameter binding sends the query and the values "
            "over separate channels, so a value can never become syntax."
        ),
        unsafe_example='query = "SELECT * FROM users WHERE id=" + user_id\ncursor.execute(query)',
        safe_example='cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
    ),
    Label.COMMAND_INJECTION: Remediation(
        title="Avoid shell=True; pass an argument list",
        explanation=(
            "With shell=True the string is interpreted by the shell, so characters "
            "like ; | && let an attacker append their own commands. Passing a list "
            "executes the binary directly with no shell involved."
        ),
        unsafe_example='subprocess.run(f"ping {host}", shell=True)',
        safe_example='subprocess.run(["ping", "-c", "1", host], shell=False)',
    ),
    Label.PATH_TRAVERSAL: Remediation(
        title="Resolve and confine paths to a base directory",
        explanation=(
            "User-controlled path components can escape the intended directory with "
            "'..' or an absolute path. Resolve the path fully, then verify the result "
            "is still inside the allowed root before opening it."
        ),
        unsafe_example="open(os.path.join(UPLOAD_DIR, filename))",
        safe_example=(
            "root = Path(UPLOAD_DIR).resolve()\n"
            "target = (root / filename).resolve()\n"
            "if not target.is_relative_to(root):\n"
            "    raise ValueError('path escapes upload directory')\n"
            "open(target)"
        ),
    ),
    Label.UNSAFE_DESERIALIZATION: Remediation(
        title="Do not deserialize untrusted data with pickle",
        explanation=(
            "pickle.loads executes arbitrary code during reconstruction, so any "
            "attacker-controlled payload is remote code execution. Use a data-only "
            "format for untrusted input."
        ),
        unsafe_example="obj = pickle.loads(request.data)",
        safe_example="obj = json.loads(request.data)  # validate against a schema next",
    ),
    Label.XSS: Remediation(
        title="Escape output; do not mark user input as safe",
        explanation=(
            "Interpolating user input into HTML lets an attacker inject script. Let "
            "the template engine escape by default, and never wrap user data in "
            "mark_safe / |safe."
        ),
        unsafe_example='return HttpResponse(f"<div>{comment}</div>")',
        safe_example='return render(request, "comment.html", {"comment": comment})',
    ),
    Label.HARDCODED_SECRET: Remediation(
        title="Load secrets from the environment or a secret manager",
        explanation=(
            "A credential in source is exposed to everyone with repository access "
            "and persists in git history after removal. Inject it at runtime and "
            "rotate anything that was committed."
        ),
        unsafe_example='API_KEY = "sk_live_a1b2c3d4e5f6"',
        safe_example='API_KEY = os.environ["API_KEY"]',
    ),
}


def severity_for(label: Label) -> Severity:
    return SEVERITY[label]


def remediation_for(label: Label) -> Remediation | None:
    return REMEDIATIONS.get(label)


def label_from_cwe(cwe: str) -> Label | None:
    return CWE_TO_LABEL.get(cwe.strip().upper())
