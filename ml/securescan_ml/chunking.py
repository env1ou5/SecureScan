"""Extract Python functions as classification units (proposal §2, D2).

Functions are the unit of classification because that is how every public
dataset is labeled, they fit the 512-token window, and their byte offsets let
predictions map back to exact source lines.

Two backends:

  * tree-sitter (preferred) -- error-tolerant, so a file with a syntax error
    still yields the functions that did parse, and the same approach extends to
    other languages later.
  * ast (fallback) -- stdlib, no dependency, but rejects a file outright if any
    part of it fails to parse.

The fallback exists so the pipeline and its tests run before tree-sitter
grammars are built. Production uses tree-sitter; `active_backend()` reports
which one is live so a scan can record it.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

try:  # pragma: no cover - depends on install
    import tree_sitter_python as tspython
    from tree_sitter import Language, Node, Parser

    _TS_LANGUAGE = Language(tspython.language())
    _TS_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import/build failure means fall back
    _TS_AVAILABLE = False


@dataclass
class FunctionChunk:
    """One function, with everything needed to map a prediction back to source."""

    file_path: str
    name: str
    code: str
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    start_byte: int  # offset into the original file
    end_byte: int
    # Set when a chunk is a window of an oversized function.
    window_index: int = 0
    window_count: int = 1
    metadata: dict = field(default_factory=dict)

    @property
    def is_window(self) -> bool:
        return self.window_count > 1

    def line_of_offset(self, offset_in_chunk: int) -> int:
        """Map a byte offset within `code` to an absolute line in the file.

        Attribution produces token offsets relative to the chunk; findings need
        absolute file lines.
        """
        if offset_in_chunk < 0:
            offset_in_chunk = 0
        prefix = self.code.encode("utf-8")[:offset_in_chunk]
        return self.start_line + prefix.count(b"\n")


def active_backend() -> str:
    return "tree-sitter" if _TS_AVAILABLE else "ast"


def extract_functions(source: str, file_path: str = "<unknown>") -> list[FunctionChunk]:
    """Extract every function and method in `source`.

    Nested functions are skipped: the enclosing function's chunk already
    contains their text, and emitting both would double-count findings.
    """
    if _TS_AVAILABLE:
        return _extract_tree_sitter(source, file_path)
    return _extract_ast(source, file_path)


def _extract_tree_sitter(source: str, file_path: str) -> list[FunctionChunk]:  # pragma: no cover
    data = source.encode("utf-8")
    parser = Parser(_TS_LANGUAGE)
    tree = parser.parse(data)
    chunks: list[FunctionChunk] = []

    def walk(node: Node, inside_function: bool) -> None:
        if node.type == "function_definition":
            if not inside_function:
                name_node = node.child_by_field_name("name")
                chunks.append(
                    FunctionChunk(
                        file_path=file_path,
                        name=data[name_node.start_byte : name_node.end_byte].decode(
                            "utf-8", "replace"
                        )
                        if name_node
                        else "<anonymous>",
                        code=data[node.start_byte : node.end_byte].decode("utf-8", "replace"),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                    )
                )
            # Descend regardless, but mark that we are inside one now.
            for child in node.children:
                walk(child, True)
            return
        for child in node.children:
            walk(child, inside_function)

    walk(tree.root_node, False)
    return chunks


def _extract_ast(source: str, file_path: str) -> list[FunctionChunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        log.warning("skipping %s: %s", file_path, exc)
        return []

    lines = source.splitlines(keepends=True)
    # Byte offset of the start of each 1-indexed line.
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line.encode("utf-8")))

    def offset_of(lineno: int, col: int) -> int:
        return line_offsets[lineno - 1] + col

    chunks: list[FunctionChunk] = []
    seen_spans: set[tuple[int, int]] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Skip nested functions: enclosed by an already-captured span.
        if any(s <= node.lineno and node.end_lineno <= e for s, e in seen_spans):
            continue
        start_byte = offset_of(node.lineno, node.col_offset)
        end_byte = offset_of(node.end_lineno, node.end_col_offset)
        seen_spans.add((node.lineno, node.end_lineno))
        chunks.append(
            FunctionChunk(
                file_path=file_path,
                name=node.name,
                code=source.encode("utf-8")[start_byte:end_byte].decode("utf-8", "replace"),
                start_line=node.lineno,
                end_line=node.end_lineno,
                start_byte=start_byte,
                end_byte=end_byte,
            )
        )

    chunks.sort(key=lambda c: c.start_byte)
    return chunks


def window_oversized(
    chunk: FunctionChunk,
    tokenizer,
    max_tokens: int = 512,
    stride_tokens: int = 128,
) -> list[FunctionChunk]:
    """Split a function that exceeds the model's window into overlapping windows.

    Returns [chunk] unchanged when it already fits. Window predictions are
    recombined by the predictor, which takes the maximum non-SAFE probability
    across windows (proposal §4).

    Splitting is done on lines rather than tokens so each window stays
    syntactically readable and its line mapping stays exact.
    """
    encoded = tokenizer(chunk.code, add_special_tokens=True, return_attention_mask=False)
    if len(encoded["input_ids"]) <= max_tokens:
        return [chunk]

    lines = chunk.code.splitlines(keepends=True)
    # Budget in tokens, converted to a line count via the observed average.
    tokens_per_line = max(len(encoded["input_ids"]) / max(len(lines), 1), 1e-6)
    lines_per_window = max(int((max_tokens - 8) / tokens_per_line), 1)
    overlap_lines = (
        min(max(int(stride_tokens / tokens_per_line), 1), lines_per_window - 1)
        if lines_per_window > 1
        else 0
    )
    step = max(lines_per_window - overlap_lines, 1)

    windows: list[FunctionChunk] = []
    line_byte = chunk.start_byte
    prefix_bytes = [chunk.start_byte]
    for line in lines:
        line_byte += len(line.encode("utf-8"))
        prefix_bytes.append(line_byte)

    starts = list(range(0, len(lines), step))
    for i, start in enumerate(starts):
        end = min(start + lines_per_window, len(lines))
        windows.append(
            FunctionChunk(
                file_path=chunk.file_path,
                name=chunk.name,
                code="".join(lines[start:end]),
                start_line=chunk.start_line + start,
                end_line=chunk.start_line + end - 1,
                start_byte=prefix_bytes[start],
                end_byte=prefix_bytes[end],
                window_index=i,
                window_count=len(starts),
                metadata=dict(chunk.metadata),
            )
        )
        if end >= len(lines):
            break

    for w in windows:
        w.window_count = len(windows)
    return windows


def extract_module_chunks(source: str, file_path: str = "<unknown>") -> list[FunctionChunk]:
    """Extract contiguous runs of top-level, non-def statements.

    Function-level chunking alone is blind to module scope, which is precisely
    where hardcoded credentials usually live:

        API_KEY = "sk_live_..."        <- never seen by the model
        def get_key():
            return API_KEY             <- seen, and correctly SAFE

    This was caught by scripts/demo_scan.py missing a planted secret while
    finding the other four vulnerabilities.

    Contiguous runs are emitted separately rather than one concatenated blob so
    each chunk keeps an exact, unbroken line range and attribution still maps
    back to real source lines.

    Imports are skipped: they are top-level, ubiquitous, and carry no finding on
    their own, so including them would flood every scan with noise.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        log.warning("skipping module scope in %s: %s", file_path, exc)
        return []

    lines = source.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line.encode("utf-8")))

    skip = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import,
        ast.ImportFrom,
    )

    runs: list[list[ast.stmt]] = []
    current: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, skip):
            if current:
                runs.append(current)
                current = []
            continue
        current.append(node)
    if current:
        runs.append(current)

    chunks: list[FunctionChunk] = []
    for run in runs:
        start_line = run[0].lineno
        end_line = max(getattr(n, "end_lineno", n.lineno) or n.lineno for n in run)
        code = "".join(lines[start_line - 1 : end_line])
        if not code.strip():
            continue
        chunks.append(
            FunctionChunk(
                file_path=file_path,
                name="<module>",
                code=code,
                start_line=start_line,
                end_line=end_line,
                start_byte=line_offsets[start_line - 1],
                end_byte=line_offsets[end_line],
                metadata={"scope": "module"},
            )
        )
    return chunks


def extract_analyzable_chunks(source: str, file_path: str = "<unknown>") -> list[FunctionChunk]:
    """Everything a scan should classify: functions plus module-scope runs.

    This is what the scan worker uses. Dataset mining still calls
    `extract_functions` directly, since public datasets label functions and
    mixing scopes there would change the corpus semantics.
    """
    chunks = extract_functions(source, file_path) + extract_module_chunks(source, file_path)
    chunks.sort(key=lambda c: c.start_byte)
    return chunks
