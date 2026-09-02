# Datasets

Data is **not** committed. Everything here is gitignored except this file.
`datasets/normalized.jsonl` is the build artifact the training pipeline consumes.

Build it with:

```bash
make dataset
```

## Sources actually used

The proposal's original plan named CVEFixes and Juliet. Both turned out to be
impractical for a Python-only project, and what replaced them is documented
here rather than quietly substituted.

| Source | What it is | Label origin | Synthetic |
|---|---|---|---|
| `juliet` | Compositional generator (`data/synthetic.py`) | Known by construction | yes |
| `mined` | Top ~220 PyPI packages by download count | Semgrep + AST detectors | no |

**Why not CVEFixes:** building it requires cloning thousands of repositories
over several hours. Its Python subset is small, and the resulting corpus would
still need human verification to be worth the "real CVE" claim.

**Why not Juliet proper:** the NIST Juliet suites are C/C++ and Java. The
Python material in SARD is thin. `data/synthetic.py` is a Juliet-equivalent
built for this taxonomy instead.

## The synthetic generator

Templated code teaches a model the template. The generator composes along five
independent axes so the invariant across samples is the *data flow* rather than
the surface form:

```
taint source x sink variant x wrapper structure x identifier pool x noise
```

- **Taint sources:** Flask `request.args`/`form`/`json`, Django `GET`/`POST`,
  `sys.argv`, `input()`, `os.environ`, or the bare parameter.
- **Sink variants:** per class — SQL via sqlite3/psycopg2/MySQLdb/SQLAlchemy
  with concat/f-string/`%`/`.format` construction; commands via
  `os.system`/`Popen`/`run(shell=True)`/`check_output`; and so on.
- **Wrappers:** plain, `try/except`, guard clause, loop, docstring, nested `if`.
- **Headers:** module function, class method, decorated route, type-annotated,
  `async`.
- **Identifiers:** every name drawn fresh from pools, so nothing keys on
  `user_id`.

Safe counterparts are genuine fixes (parameter binding, `shell=False`,
`is_relative_to` containment), not cosmetic edits, so SAFE is not trivially
separable.

Measured: 3,120 samples generated, **0 parse failures, 100% unique** code
strings.

## Labeling mined real code

Labels come from **Semgrep** (`r/python.lang.security` + `p/security-audit`),
mapped onto the taxonomy through each finding's CWE metadata using
`CWE_TO_LABEL` in `labels.py`.

**Semgrep is used for labels; Bandit is the benchmark baseline (§10). They are
never the same tool.** Labeling with the tool you benchmark against measures
agreement, not correctness, and would make the headline comparison meaningless.

### Measured Semgrep coverage

Run against held-out synthetic vulnerable samples (n=25/class):

| Class | Detected | Mapped to taxonomy |
|---|---:|---:|
| COMMAND_INJECTION | 25/25 | 25/25 |
| UNSAFE_DESERIALIZATION | 25/25 | 25/25 |
| SQL_INJECTION | 12/25 | 12/25 |
| XSS | 4/25 | 4/25 |
| PATH_TRAVERSAL | 0/25 | 0/25 |
| HARDCODED_SECRET | 0/25 | 0/25 |

Two classes are entirely uncovered. Mined real code would therefore contain no
examples of them at all.

### Filling the gap without reintroducing circularity

Falling back to Bandit for those two classes would put the baseline back into
the labeling path. `data/ast_labeler.py` is a third, independent detector
instead:

- **Hardcoded secrets** — AST scan for string literals assigned to
  secret-named targets or keyword arguments, gated on recognizable credential
  formats (`sk_live_`, `AKIA`, `ghp_`, `xox[bp]-`, PEM headers) or on Shannon
  entropy ≥ 3.4 bits/char with no whitespace and ≥80% alphanumeric.
- **Path traversal** — intraprocedural taint from parameters and request-like
  attributes into `open`/`read_text`/`write_bytes` via `os.path.join`,
  concatenation, or f-strings, standing down entirely if the function mentions
  any containment idiom (`is_relative_to`, `commonpath`, `realpath`, …).

Both are deliberately high-precision, low-recall. A missed vulnerable function
costs one training sample; a mislabeled safe function teaches the model
something false.

Measured on held-out synthetic samples (n=60/class):

| Detector | Recall | False positives on safe counterparts |
|---|---:|---:|
| Hardcoded secrets | 60/60 | 0/60 |
| Path traversal | 37/60 | 0/60 |

## Normalized schema

One JSON object per line, matching `ml/securescan_ml/data/schema.py`:

```json
{
  "repository_id": "pypi/requests",
  "file_path": "requests/sessions.py",
  "language": "python",
  "code": "def send(self, request, **kwargs):\n    ...",
  "vulnerability_type": "COMMAND_INJECTION",
  "start_line": 84,
  "end_line": 92,
  "source": "mined",
  "is_synthetic": false,
  "vulnerable_lines": [86],
  "metadata": {"package": "requests", "labeler": "semgrep"}
}
```

`repository_id` is required and load-bearing: splits are made on it, never on
individual functions. `is_synthetic` must agree with `source` — the schema
raises if they contradict.

## Pipeline

```
synthetic generator          top-220 PyPI packages
        |                             |
        |                    semgrep + AST detectors
        |                             |
        +------------> VulnRecord <---+
                            |
                    MinHash + LSH dedup        <- before splitting, never after
                            |
              repository-disjoint train/val/test
                            |
                    real-code-only test set
                            |
                  datasets/normalized.jsonl
```

## The rule that matters most

The real-code test set contains **no synthetic samples, ever**, and its metrics
are always reported separately from the headline number.

## Honest limitation

These are **static-analysis labels, not human-verified CVE labels**. Semgrep
and the AST detectors have their own false positives and blind spots, and a
model trained on them inherits both. Its ceiling is agreement with those
labelers on held-out repositories, not ground truth.

This is the single biggest caveat on every number this project reports, and it
is stated in the README and the proposal as well.
