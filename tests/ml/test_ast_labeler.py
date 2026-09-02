"""The AST labeler must be high-precision: a false positive teaches the model
something untrue, while a false negative only costs one training sample.
"""

import ast

from securescan_ml.data.ast_labeler import (
    find_hardcoded_secrets,
    label_source,
    shannon_entropy,
)
from securescan_ml.labels import Label


def labels_of(source: str) -> set[Label]:
    return set(label_source(source).values())


# --------------------------------------------------------------------------
# Hardcoded secrets
# --------------------------------------------------------------------------


def test_detects_recognizable_credential_formats():
    for literal in (
        "sk_live_abcdef0123456789abcdef",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_" + "a" * 36,
        "xoxb-123456789012-abcdefghijkl",
    ):
        source = f'harmless_name = "{literal}"\n'
        assert Label.HARDCODED_SECRET in labels_of(source), literal


def test_detects_secret_name_with_high_entropy_value():
    source = 'API_KEY = "f3a9c2b8d7e64150a1b2c3d4e5f60718"\n'
    assert Label.HARDCODED_SECRET in labels_of(source)


def test_detects_secret_in_keyword_argument():
    source = 'conn = connect(host="db", password="a9F3kd02LmZq81xVbn43")\n'
    assert Label.HARDCODED_SECRET in labels_of(source)


def test_ignores_environment_lookup():
    source = 'API_KEY = os.environ["API_KEY"]\n'
    assert Label.HARDCODED_SECRET not in labels_of(source)


def test_ignores_placeholder_values():
    for placeholder in ("changeme", "your_api_key_here", "TODO", "placeholder", ""):
        source = f'API_KEY = "{placeholder}"\n'
        assert Label.HARDCODED_SECRET not in labels_of(source), placeholder


def test_ignores_low_entropy_prose():
    """A secret-sounding name holding an English phrase is not a credential."""
    source = 'password_prompt_message = "please enter your password below"\n'
    assert Label.HARDCODED_SECRET not in labels_of(source)


def test_ignores_non_secret_name_and_ordinary_value():
    source = 'TEMPLATE_DIR = "application/templates/default"\n'
    assert Label.HARDCODED_SECRET not in labels_of(source)


def test_entropy_ranks_random_above_prose():
    assert shannon_entropy("f3a9c2b8d7e64150a1b2c3d4") > shannon_entropy("aaaaaaaaaaaaaaaa")


def test_reports_the_assignment_line():
    source = "x = 1\n\nSECRET_TOKEN = 'b8d7e64150a1b2c3d4e5f60718a9c2'\n"
    lines = find_hardcoded_secrets(ast.parse(source))
    assert lines == [3]


# --------------------------------------------------------------------------
# Path traversal
# --------------------------------------------------------------------------


def test_detects_join_of_tainted_input():
    source = (
        "import os\n"
        "def read_upload(filename):\n"
        "    p = os.path.join('/srv/files', filename)\n"
        "    with open(p, 'rb') as fh:\n"
        "        return fh.read()\n"
    )
    assert Label.PATH_TRAVERSAL in labels_of(source)


def test_detects_fstring_path():
    source = (
        "def read_upload(name):\n"
        "    with open(f'/srv/files/{name}', 'rb') as fh:\n"
        "        return fh.read()\n"
    )
    assert Label.PATH_TRAVERSAL in labels_of(source)


def test_stands_down_when_containment_is_checked():
    """A guarded function must never be labeled vulnerable."""
    source = (
        "from pathlib import Path\n"
        "def read_upload(name):\n"
        "    root = Path('/srv/files').resolve()\n"
        "    target = (root / name).resolve()\n"
        "    if not target.is_relative_to(root):\n"
        "        raise ValueError('escapes root')\n"
        "    with open(target, 'rb') as fh:\n"
        "        return fh.read()\n"
    )
    assert Label.PATH_TRAVERSAL not in labels_of(source)


def test_ignores_constant_path():
    source = (
        "def read_config():\n"
        "    with open('/etc/app/config.toml', 'rb') as fh:\n"
        "        return fh.read()\n"
    )
    assert Label.PATH_TRAVERSAL not in labels_of(source)


def test_ignores_function_with_no_parameters():
    source = "def dump():\n    with open('out.txt', 'w') as fh:\n        fh.write('x')\n"
    assert Label.PATH_TRAVERSAL not in labels_of(source)


def test_syntax_error_returns_no_labels():
    assert label_source("def broken(:\n    pass\n") == {}


def test_does_not_emit_labels_semgrep_already_covers():
    """The AST labeler exists only for the two uncovered classes; it must not
    start guessing at SQL injection or command injection."""
    source = "import os\ndef run(host):\n    return os.system('ping ' + host)\n"
    assert labels_of(source) <= {Label.HARDCODED_SECRET, Label.PATH_TRAVERSAL}
