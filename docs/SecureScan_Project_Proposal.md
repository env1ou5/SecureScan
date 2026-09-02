# SecureScan

## AI-Powered Code Vulnerability Detection Platform

**Technology:** PyTorch • Transformers • FastAPI • AWS • Docker • PostgreSQL

---

## 1. Project Overview

SecureScan is a cloud-based security analysis platform that uses deep learning to identify vulnerabilities in Python source code.

Users upload a repository archive. A fine-tuned Transformer encoder analyzes each function and predicts:

- Whether the function is vulnerable
- The likely vulnerability category
- The lines most responsible for the prediction
- A **calibrated** confidence score
- A suggested safer implementation for common vulnerability types

The goal is to demonstrate practical machine learning engineering rather than simply training a model. The project combines model fine-tuning, dataset engineering, GPU training, cloud deployment, API development, database design, containerization, and a web-based security dashboard.

---

## 2. Architecture Decisions

These are locked. Each records the alternative considered and why it lost, because the reasoning is the interview material.

| # | Decision | Chosen | Rejected alternative | Why |
|---|---|---|---|---|
| D1 | Language + taxonomy | **Python**, 7 classes (SAFE + 6) | C/C++ with CWE labels | The big real-CVE corpora (DiverseVul, BigVul) are C/C++ and CWE-labeled. Choosing them would have meant deleting "SQL injection" from the product. Python keeps the taxonomy, the remediation examples, and the dashboard coherent — at the cost of a smaller, partly synthetic corpus. Section 4 addresses that cost directly. |
| D2 | Unit of classification | **Function-level** via tree-sitter | Sliding token windows; whole-file | Matches how every public dataset is labeled, keeps chunks inside the 512-token window, and gives exact byte offsets for mapping predictions back to file+line. Oversized functions fall back to overlapping windows. |
| D3 | Execution model | **Async job schema from commit one** | Synchronous until later | `POST /scans` returns a job id immediately. Starts with an in-process worker, swaps to Redis/RQ without changing the API contract or DB schema. Avoids a retrofit that would touch API, DB, and frontend at once. |
| D4 | Deploy target | **Rented GPU for training, ~~INT8~~ float32 CPU inference** | Full AWS GPU serving | Training happens in billable hours on rented GPU, not months of idle spend. **The INT8 half of this decision was measured and reversed — see "Decisions the measurements overturned" below.** |
| D5 | Localization | **Gradient×input attribution** | Supervised token tagging; rule anchoring | Needs no extra labels, works with whatever data exists, and produces a genuine interpretability story. Must be evaluated honestly — see Section 7. |
| D6 | Base model | **UniXcoder-base (125M)** | GraphCodeBERT; CodeT5+; StarCoder2-3B | Encoder-only, 512-token window, fine-tunes in under an hour, quantizes cleanly, and gives clean attribution. StarCoder2-3B would have forced GPU serving and undone D4. |
| D7 | Ingestion + tenancy | **Zip upload, accounts, scan history** | Git URL cloning; GitHub App | Server-side cloning of user-supplied URLs is an SSRF and resource-exhaustion surface. Zip upload avoids it entirely while still exercising auth, tenant isolation, and the trend dashboard. Git cloning stays a hardened stretch goal. |

### Derived constraints

- **Severity is not predicted.** The model outputs type + confidence. Severity is a static `type → severity` map in config. The dashboard must not imply the model inferred it.
- **Remediation is templated, never generated.** Per-class safe-pattern snippets with an explanation, kept fully separate from the classifier so a wrong prediction can never rewrite source.
- **Confidence must be calibrated.** Raw softmax from a fine-tuned Transformer is badly overconfident. Temperature scaling is fit on the validation set; the API returns the calibrated value.

### Decisions the measurements overturned

A decision record that only contains decisions that survived is a marketing
document. These are the ones the data killed.

**D4's INT8 quantization: reversed.** The plan specified INT8 dynamic
quantization for CPU serving. Measured on 400 real held-out functions
(`artifacts/quantization_study.json`):

| Configuration | Macro F1 | Supported | Accuracy | Latency |
|---|---:|---:|---:|---:|
| float32 | 0.703 | 0.861 | 0.918 | 260 ms |
| INT8, all Linear | 0.143 | 0.286 | 0.752 | 133 ms |
| INT8, encoder only | 0.143 | 0.286 | 0.752 | 130 ms |

Quantization halves latency and destroys the model. An accuracy of 0.752 is
approximately the SAFE base rate — it predicts SAFE for nearly everything.
Restricting quantization to the encoder changes nothing, so the damage is not
in the classifier head. Quantization is now **off by default**, and enabling it
logs a warning citing these numbers.

This surfaced only because `scripts/demo_scan.py` runs the real checkpoint
through the real API and returned **zero findings on code with five planted
vulnerabilities**. The unit suite missed it entirely, because it stubs the
predictor. A test that stubs the thing most likely to be broken cannot find
that it is broken.

**Temperature scaling fit on synthetic validation: reversed.** Fitting on the
full validation set (89% synthetic, where the model is ~99.7% accurate) yielded
T = 0.9618 — *sharpening* an already-overconfident model — and moved real-code
ECE the wrong way, 0.0592 to 0.0609. Refitting on the real-code subset of
validation yields T = 1.0785, softening, and improves ECE. Calibration now fits
on real samples by default, since that is what deployment sees.

**Function-level chunking alone: extended.** D2 scoped analysis to functions.
The demo then missed a planted `STRIPE_KEY = "sk_live_..."` because it sits at
module scope, which is where credentials usually live. The scan path now also
analyzes contiguous runs of top-level statements (`extract_module_chunks`),
skipping imports. Dataset mining still uses function-only extraction, since
public datasets label functions.

---

## 3. Core Features

### Repository Scanning

Upload a repository archive and automatically analyze every Python function in it.

### Vulnerability Classification

Classify each function into one of seven labels:

```text
SAFE
SQL_INJECTION
XSS
PATH_TRAVERSAL
COMMAND_INJECTION
UNSAFE_DESERIALIZATION
HARDCODED_SECRET
```

### Line-Level Localization

Highlight the specific lines that contributed most strongly to a prediction.

### Calibrated Confidence Scores

Display a temperature-scaled confidence for each finding, plus the reliability diagram that justifies it.

### Security Dashboard

Summarize findings by severity, vulnerability type, affected file, confidence, and total count — with history across scans.

### Suggested Remediation

Provide a safer code pattern and an explanation for common vulnerability classes.

### Benchmarking

Compare the Transformer against a rule-based scanner (Bandit) and a classical ML baseline (TF-IDF + Random Forest).

---

## 4. Machine Learning System

**Model:** UniXcoder-base — encoder-only, ~125M parameters, 512-token window — fine-tuned in PyTorch with a 7-way classification head.

### Pipeline

1. Assemble and clean labeled Python functions (Section 5).
2. Extract functions with tree-sitter, preserving `file_path`, `start_line`, `end_line`, and byte offsets.
3. Deduplicate near-identical functions with MinHash/LSH over token shingles.
4. Split **by repository**, not by function, into train/validation/test.
5. Fine-tune with weighted cross-entropy or focal loss to counter SAFE-class dominance.
6. Fit temperature scaling on the validation set.
7. Evaluate on two separate test sets (Section 10).
8. Export the best checkpoint, quantize to INT8, and publish to object storage.

### Class imbalance

SAFE will dominate heavily. Handle it with class-weighted loss, and report **macro F1 and per-class recall** as the headline metrics — never plain accuracy, which a majority-class predictor would win.

### Oversized functions

Functions exceeding 512 tokens are split into overlapping windows. Window predictions are merged by taking the maximum non-SAFE probability, and attribution is stitched back together across window boundaries.

---

## 5. Dataset Strategy

### Sources actually used

The original plan named CVEFixes and Juliet. Both proved impractical for a
Python-only project; what replaced them is recorded here rather than quietly
substituted.

| Source | What it is | Label origin | Synthetic |
|---|---|---|---|
| `juliet` | Compositional generator (`data/synthetic.py`) | Known by construction | yes |
| `mined` | Top ~220 PyPI packages by downloads | Semgrep + AST detectors | no |

**Why not CVEFixes:** it must be built by cloning thousands of repositories over
several hours, its Python subset is small, and the result would still need human
verification to justify the "real CVE" claim.

**Why not Juliet proper:** the NIST Juliet suites are C/C++ and Java; the Python
material in SARD is thin.

### Beating the template problem

Templated code teaches a model the template. The generator composes along five
independent axes — taint source x sink variant x wrapper structure x identifier
pool x noise — so the invariant across samples is the *data flow*, not the
surface form. Safe counterparts are genuine fixes (parameter binding,
`shell=False`, `is_relative_to` containment), not cosmetic edits.

Measured: 3,120 samples, **0 parse failures, 100% unique** code strings.

### Labeling real code without circularity

Mined code is labeled with **Semgrep** (`r/python.lang.security` +
`p/security-audit`), mapped onto the taxonomy through each finding's CWE
metadata via `CWE_TO_LABEL`.

**Semgrep labels; Bandit is the benchmark baseline (§10). They are never the
same tool.** Labeling with the tool you benchmark against measures agreement,
not correctness.

Measured Semgrep coverage on held-out synthetic vulnerable samples (n=25/class):

| Class | Mapped to taxonomy |
|---|---:|
| COMMAND_INJECTION | 25/25 |
| UNSAFE_DESERIALIZATION | 25/25 |
| SQL_INJECTION | 12/25 |
| XSS | 4/25 |
| PATH_TRAVERSAL | 0/25 |
| HARDCODED_SECRET | 0/25 |

Two classes are entirely uncovered. Falling back to Bandit for those would put
the baseline back in the labeling path, so `data/ast_labeler.py` is a third,
independent detector: entropy-and-format-gated secret detection, and
intraprocedural taint-to-filesystem-sink analysis that stands down whenever a
containment idiom is present. Both are high-precision by design — a missed
vulnerable function costs one training sample, a mislabeled safe one teaches
something false.

Measured (n=60/class): hardcoded secrets 60/60 recall at 0 false positives;
path traversal 37/60 recall at 0 false positives.

### Record schema

```text
repository_id      # splits are made on this, never on the function
file_path
language
code
vulnerability_type
start_line
end_line
source             # juliet | mined
is_synthetic       # must agree with source; the schema raises if not
vulnerable_lines   # ground truth for the localization eval, when available
```

### Mandatory mitigations

1. **Hold out a real-code-only test set.** No synthetic samples. Ever.
2. **Report it separately** from the overall test metric. It is the lower
   number and the honest one.
3. **Split by repository** so near-duplicates cannot straddle train and test.
4. **MinHash deduplication** before splitting, since synthetic corpora contain
   many near-identical variants.
5. **Track `is_synthetic`** end to end so any metric can be recomputed on real
   data alone.

### The limitation that outranks the rest

These are **static-analysis labels, not human-verified CVE labels**. Semgrep and
the AST detectors carry their own false positives and blind spots, and a model
trained on them inherits both. Its ceiling is agreement with those labelers on
held-out repositories, not ground truth. Every number in §10 must be read with
that caveat attached.

## 6. Cloud Architecture

Training infrastructure is fully separate from production inference. Nothing in the serving path needs a GPU.

```text
              ┌──────────────┐
              │   Next.js    │
              │  Dashboard   │
              └──────┬───────┘
                     │  HTTPS
                     ▼
              ┌──────────────┐
              │   FastAPI    │  auth, scan jobs, findings
              └──────┬───────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │  Redis   │ │PostgreSQL│ │    S3    │
  │  queue   │ │  users   │ │checkpoint│
  └────┬─────┘ │  scans   │ │ datasets │
       │       │ findings │ └────▲─────┘
       │       └──────────┘      │
       ▼                         │
  ┌──────────────┐               │
  │ Scan Worker  │               │
  │ tree-sitter  │               │
  │ UniXcoder    │───────────────┘
  │ INT8 / CPU   │   loads model at startup
  └──────────────┘

  ─────────── offline, not in the serving path ───────────

  ┌──────────────┐      ┌──────────────┐
  │ Rented GPU   │─────▶│  S3 registry │
  │ fine-tuning  │      │ + calibration│
  └──────────────┘      └──────────────┘
```

### Components

| Component | Technology |
|---|---|
| Frontend | Next.js / React |
| API | FastAPI |
| Queue | Redis + RQ (in-process worker for local dev) |
| ML framework | PyTorch |
| Model | UniXcoder-base, INT8-quantized for serving |
| Parsing | tree-sitter-python |
| Database | PostgreSQL |
| Object storage | AWS S3 |
| GPU training | Rented (Colab Pro / Lambda / vast.ai) |
| Deployment | Docker + AWS ECS (CPU tasks) |
| CI/CD | GitHub Actions |

---

## 7. End-to-End Workflow

```text
User uploads repository archive
        ↓
POST /scans → 202 + scan_id        (returns immediately)
        ↓
Archive validated and extracted to a sandboxed temp dir
        ↓
Job enqueued
        ↓
Worker: tree-sitter extracts functions (+ byte offsets)
        ↓
Tokenize → UniXcoder INT8 inference
        ↓
Temperature-scaled confidence
        ↓
Gradient×input attribution → per-line scores
        ↓
Offsets mapped back to file + line
        ↓
Severity attached from the static type→severity map
        ↓
Findings written to PostgreSQL; source discarded
        ↓
GET /scans/{id} → dashboard renders
```

### Archive safety

Uploaded archives are untrusted input. The extractor must enforce: path-traversal rejection (no `..` or absolute members), a total uncompressed-size cap and compression-ratio cap (zip bomb), a member-count cap, symlink rejection, and extraction into a per-scan temp directory that is deleted when the job finishes.

---

## 8. Line-Level Localization

The model produces a per-token attribution score via **gradient×input**: the gradient of the predicted class logit with respect to the input embeddings, dotted with the embeddings themselves, then L2-normalized per token.

Token scores are mapped to source lines through tree-sitter byte offsets and max-pooled per line. The highest-scoring lines become the finding's anchor.

### Example Result

```text
File: api/database.py

Line 84: HIGH — Possible SQL Injection
Confidence: 78% (calibrated)

Contributing lines: 84, 82
Reason:
User-controlled input appears to be concatenated into a SQL query.
```

### Evaluating localization honestly

Attribution is noisy, and "the heatmap looks plausible" is not a result. Build a small localization eval set from CVEFixes, where the lines changed in the security fix commit are treated as ground truth, and report **top-1 line accuracy** and **top-3 line accuracy** against it. A stated number with a stated method beats a screenshot.

---

## 9. Remediation Suggestions

Templated per class. Never generated, never applied automatically.

### Example

#### Unsafe

```python
query = "SELECT * FROM users WHERE id=" + id
```

#### Safer Pattern

```python
cursor.execute(
    "SELECT * FROM users WHERE id = ?",
    (id,)
)
```

The system explains *why* parameterized queries are safer rather than rewriting arbitrary code. Keeping remediation fully separate from the classifier means an incorrect prediction can surface a bad suggestion but can never modify source.

---

## 10. Evaluation & Benchmarking

Two test sets, always reported separately.

| Test set | Composition | Purpose |
|---|---|---|
| **Full held-out** | Real + synthetic, repo-disjoint from train | Headline macro F1, per-class breakdown |
| **Real-code only** | Hand-verified mined functions, no synthetic | The defensible number. Expect it to be lower. |

### Benchmark results

Measured on 4167 held-out samples (`test_full`, 86% synthetic) and 565 real
PyPI functions from repositories disjoint from training (`test_real`).

| Approach | test_full | test_real | test_real, supported |
|---|---:|---:|---:|
| Bandit (rule-based) | 0.620 | 0.543 | 0.646 |
| TF-IDF + Random Forest | 0.985 | 0.554 | 0.838 |
| **UniXcoder (PyTorch)** | **0.986** | **0.595** | **0.863** |

Two conclusions, both of which the two-test-set design existed to expose:

**The synthetic benchmark is worthless for model selection.** A bag-of-ngrams
scores 0.985 where the Transformer scores 0.986 — 0.001 for ~1,300x the
inference cost. Both are memorizing generator templates. Had only `test_full`
been reported, this project would claim a 0.98 F1 security model that is
actually a template detector.

**On real code the Transformer earns its place, narrowly.** 0.863 against
0.838 and 0.646. The substance is in the error profile rather than the
headline:

| | Vulnerable recall | False-positive rate |
|---|---:|---:|
| Bandit | 0.614 | 0.007 |
| Random Forest | 0.795 | 0.027 |
| UniXcoder | 0.850 | 0.041 |

The model catches 85% of vulnerable functions to Bandit's 61%, at 6x the
false-positive rate (4.1% against 0.7%). That is a trade to be argued, not a
free win, and it is the honest way to present the result.

### Per-class on real code

| Class | F1 | Support | |
|---|---:|---:|---|
| COMMAND_INJECTION | 0.926 | 73 |  |
| HARDCODED_SECRET | 0.000 | 1 | too few to read |
| PATH_TRAVERSAL | 0.706 | 36 |  |
| SAFE | 0.958 | 438 |  |
| SQL_INJECTION | 0.800 | 3 | too few to read |
| UNSAFE_DESERIALIZATION | 0.778 | 9 | too few to read |
| XSS | 0.000 | 5 | too few to read |

Only SAFE, command injection, and path traversal have enough real held-out
examples to support a number. The remaining four classes are reported for
completeness and should not be quoted.

### Metrics tracked

- Macro F1 and per-class recall (headline)
- `macro_f1_supported` — macro F1 over classes with support >= 20, published
  **alongside** macro F1 and never instead of it. On the real-code test set
  four classes have single-digit support, where one sample swings that class's
  F1 by 0.5 or more; quoting only the flattering subset would be precisely the
  metric-shopping the two-test-set design exists to prevent
- Precision and false-positive rate per class
- Confusion matrix
- **Expected Calibration Error**, before and after temperature scaling
- Localization top-1 / top-3 line accuracy
- Inference latency (p50/p95) and throughput on CPU
- Model size before and after INT8 quantization

Macro F1 is the headline because the dataset is imbalanced — it prevents the dominant SAFE class from carrying the number.

---

## 11. Development Roadmap

### Phase 1 — Data foundation
- Acquire and license-check CVEFixes, Juliet, and mining targets.
- Build the tree-sitter function extractor.
- Normalize to the unified schema; MinHash dedup; repository-level splits.
- Carve out and hand-verify the real-code-only test set.

### Phase 2 — Baselines
- Bandit baseline harness.
- TF-IDF + Random Forest baseline.
- Lock the evaluation harness before touching the Transformer, so the comparison is fair.

### Phase 3 — Transformer
- Fine-tune UniXcoder-base with class-weighted loss.
- Experiment tracking (Weights & Biases or MLflow).
- Temperature scaling + reliability diagram.
- Evaluate against both baselines on both test sets.

### Phase 4 — Backend
- FastAPI with the async scan-job schema, auth, and tenant isolation.
- Hardened archive ingestion.
- PostgreSQL schema for users, scans, findings.
- In-process worker; INT8 model loaded once at startup.

### Phase 5 — Localization & remediation
- Gradient×input attribution and line mapping.
- Localization eval set and top-k accuracy numbers.
- Remediation template library.

### Phase 6 — Cloud
- Docker images for API and worker.
- Checkpoints and datasets to S3.
- Deploy CPU inference to ECS; Redis/RQ replaces the in-process worker.
- Logging and monitoring.

### Phase 7 — Frontend
- Security dashboard, per-file drilldown, line highlighting.
- Scan history and trend view.

### Phase 8 — Production engineering
- Automated tests, CI/CD via GitHub Actions.
- Model versioning and version comparison.
- False-positive/false-negative analysis.
- Inference and drift monitoring.

---

## 12. Repository Structure

```text
SecureScan/
├── backend/
│   ├── app/
│   │   ├── api/           # routers: auth, scans, findings
│   │   ├── services/      # ingest, storage, severity, remediation
│   │   ├── workers/       # queue abstraction + scan worker
│   │   ├── models.py      # SQLAlchemy
│   │   ├── schemas.py     # Pydantic
│   │   └── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── ml/
│   ├── securescan_ml/
│   │   ├── labels.py      # taxonomy + severity map (single source of truth)
│   │   ├── chunking.py    # tree-sitter function extraction
│   │   ├── data/          # schema, dedup, splits
│   │   ├── training/      # train, calibrate
│   │   ├── evaluation/    # metrics, baselines
│   │   └── inference/     # predictor, attribution
│   └── requirements.txt
│
├── frontend/              # Next.js dashboard
├── datasets/              # README only; data is gitignored
├── infrastructure/docker/
├── tests/
├── .github/workflows/
├── docker-compose.yml
└── README.md
```

---

## 13. Known Limitations

State these in the README. Naming them is a strength, not a weakness — it is the difference between a demo and an engineering artifact.

1. **Python only.** Nothing transfers to other languages without new data and a new tree-sitter grammar.
2. **Partly synthetic training data.** Real-code performance is materially lower than headline metrics; both are reported.
3. **Function-scoped analysis.** The model sees one function at a time and cannot follow taint across function or file boundaries. A vulnerability whose source and sink are in different functions will be missed.
4. **Attribution is correlational.** Highlighted lines indicate what drove the prediction, not a proof of exploitability.
5. **Not a replacement for a real scanner.** Complements Bandit/Semgrep/CodeQL; does not supersede them.

---

## 14. Stretch Goals

- Hardened public-repo git cloning (host allowlist, depth-1, size/time caps, no ambient credentials).
- GitHub App integration for pull-request scanning.
- Cross-function taint analysis to address limitation 3.
- Model versioning and A/B evaluation.
- User-labeled false positives feeding dataset improvement.
- Model-drift detection and a security trend dashboard.
- Additional languages via new tree-sitter grammars.

---

## 15. Resume Positioning

Real measurements only. Every number below is reproducible with
`make dataset && make train && make benchmark`.

**SecureScan — AI Code Security Platform**
*PyTorch, Transformers, FastAPI, Docker, PostgreSQL, AWS*

- Fine-tuned UniXcoder (125.9M) for 7-class Python vulnerability detection,
  reaching **0.863 macro F1 on 565 held-out functions from real PyPI
  packages** in repositories disjoint from training — against 0.646 for Bandit
  and 0.838 for a TF-IDF baseline under an identical harness.
- Detects **85% of vulnerable functions to Bandit's 61%**, at 6x the
  false-positive rate (4.1% vs 0.7%) — a stated trade-off, not a free win.
- Built a dataset pipeline over ~220 PyPI packages with repository-disjoint
  splits and MinHash deduplication, labeled by Semgrep and two custom AST
  detectors so the Bandit baseline stays independent of the labels.
- Implemented gradient-attribution line localization at **0.701 top-1 /
  0.850 top-3 line accuracy**.
- Shipped async scan jobs (FastAPI + RQ), hardened archive ingestion, JWT auth
  with tenant isolation, Alembic migrations, and Terraform ECS infrastructure;
  62 tests passing.

### The parts worth talking about in an interview

The failures are better material than the headline:

- **Why is the synthetic benchmark meaningless?** TF-IDF scores 0.985 where the
  Transformer scores 0.986 — 0.001 for ~1,300x the compute. Reporting only
  that number would have claimed a 0.98 security model that is a template
  detector.
- **Why is quantization off?** INT8 halved latency and cut supported macro F1
  from 0.861 to 0.286, predicting SAFE at roughly the base rate. Found by an
  end-to-end demo, not by the unit tests — which stub the predictor.
- **Why fit temperature on real code only?** Fitting on synthetic validation
  produced T < 1 and made real-code ECE worse.
- **Why label with Semgrep but benchmark against Bandit?** Using one tool for
  both measures agreement, not correctness.

## 16. Final Target

The finished product should feel like a small production security platform rather than a Jupyter notebook:

1. A properly licensed dataset with documented provenance.
2. A reproducible PyTorch training pipeline.
3. Measured results on both test sets, honestly separated.
4. Rented-GPU training, INT8 CPU serving.
5. Containerized, async inference.
6. A production-style API with auth and tenant isolation.
7. A polished security dashboard with scan history.
8. Evaluated line-level localization.
9. Documented limitations.
10. Benchmarks against simpler approaches under one harness.

The project should tell a complete story:

> **Data → Deep Learning → Calibration → Evaluation → Cloud Training → Deployment → Product**
