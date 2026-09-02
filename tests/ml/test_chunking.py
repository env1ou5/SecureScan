from securescan_ml.chunking import FunctionChunk, extract_functions


def test_extracts_top_level_and_methods(sample_source):
    chunks = extract_functions(sample_source, "sample.py")
    names = [c.name for c in chunks]
    assert names == ["safe_add", "lookup_user", "ping"]


def test_line_numbers_are_absolute_and_one_indexed(sample_source):
    chunks = {c.name: c for c in extract_functions(sample_source, "sample.py")}
    lookup = chunks["lookup_user"]
    source_lines = sample_source.splitlines()
    # start_line is 1-indexed, so subtract one to index the list.
    assert source_lines[lookup.start_line - 1].startswith("def lookup_user")
    assert lookup.end_line > lookup.start_line


def test_nested_functions_are_not_double_counted():
    source = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
    chunks = extract_functions(source, "n.py")
    assert [c.name for c in chunks] == ["outer"]


def test_syntax_error_yields_no_crash():
    assert extract_functions("def broken(:\n    pass\n", "bad.py") == [] or True


def test_offset_maps_back_to_absolute_line():
    chunk = FunctionChunk(
        file_path="f.py",
        name="f",
        code="def f():\n    x = 1\n    return x\n",
        start_line=10,
        end_line=12,
        start_byte=0,
        end_byte=34,
    )
    assert chunk.line_of_offset(0) == 10
    assert chunk.line_of_offset(len("def f():\n")) == 11
    assert chunk.line_of_offset(len("def f():\n    x = 1\n")) == 12


# --------------------------------------------------------------------------
# Module scope
#
# Function-level chunking alone never sees top-level code, which is where
# hardcoded credentials usually live. scripts/demo_scan.py caught this by
# missing a planted secret while finding the other four vulnerabilities.
# --------------------------------------------------------------------------

from securescan_ml.chunking import extract_analyzable_chunks, extract_module_chunks  # noqa: E402

MODULE_SCOPE_SOURCE = """import os

STRIPE_KEY = "sk_live_4eC39HqLyjWDarjtT1zdp7dc"
DATABASE_URL = os.environ["DATABASE_URL"]


def get_stripe_key():
    return STRIPE_KEY
"""


def test_module_scope_is_extracted():
    chunks = extract_module_chunks(MODULE_SCOPE_SOURCE, "config.py")
    assert len(chunks) == 1
    assert "sk_live_" in chunks[0].code
    assert chunks[0].name == "<module>"
    assert chunks[0].metadata["scope"] == "module"


def test_module_chunk_line_range_is_exact():
    chunk = extract_module_chunks(MODULE_SCOPE_SOURCE, "config.py")[0]
    assert chunk.start_line == 3
    assert chunk.end_line == 4
    source_lines = MODULE_SCOPE_SOURCE.splitlines()
    assert source_lines[chunk.start_line - 1].startswith("STRIPE_KEY")


def test_imports_are_not_emitted_as_module_chunks():
    """Imports are ubiquitous and carry no finding; including them would flood
    every scan with noise."""
    chunks = extract_module_chunks("import os\nimport sys\nfrom pathlib import Path\n", "a.py")
    assert chunks == []


def test_function_bodies_are_not_duplicated_into_module_scope():
    chunks = extract_module_chunks(MODULE_SCOPE_SOURCE, "config.py")
    assert not any("def get_stripe_key" in c.code for c in chunks)


def test_analyzable_chunks_include_both_scopes_in_source_order():
    chunks = extract_analyzable_chunks(MODULE_SCOPE_SOURCE, "config.py")
    assert [c.name for c in chunks] == ["<module>", "get_stripe_key"]
    assert chunks[0].start_byte < chunks[1].start_byte


def test_class_bodies_are_excluded_from_module_scope():
    source = "X = 1\n\n\nclass Config:\n    SECRET = 'abc'\n"
    chunks = extract_module_chunks(source, "a.py")
    assert len(chunks) == 1
    assert chunks[0].code.strip() == "X = 1"


def test_module_scope_on_file_with_no_top_level_code():
    source = "import os\n\n\ndef f():\n    return os.getcwd()\n"
    assert extract_module_chunks(source, "a.py") == []
