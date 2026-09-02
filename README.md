# SecureScan

AI-powered vulnerability detection for Python source code. Upload a repository
archive; a fine-tuned Transformer classifies every function, localizes the
lines responsible, and reports a calibrated confidence.

Design rationale and decision record: [docs/SecureScan_Project_Proposal.md](docs/SecureScan_Project_Proposal.md).

## Results

Measured, not projected. Reproduce with `make dataset && make train && make benchmark`.

### The headline number

**Macro F1 0.863 on held-out real PyPI code** (three classes with adequate
support), against 0.646 for Bandit and 0.838 for TF-IDF + Random Forest.

### Both test sets, reported together

| Approach | test_full (n=4167) | test_real (n=565) | test_real, supported |
|---|---:|---:|---:|
| Bandit (rule-based) | 0.620 | 0.543 | 0.646 |
| TF-IDF + Random Forest | 0.985 | 0.554 | 0.838 |
| **UniXcoder (fine-tuned)** | **0.986** | **0.595** | **0.863** |

`test_full` is 86% synthetic. `test_real` is real PyPI functions from
repositories disjoint from training. "Supported" restricts macro F1 to the
three classes with n >= 20; the other four have between 1 and 9 real examples,
where a single sample moves that class's F1 by 0.5 or more.

### What the numbers actually say

**On synthetic data the Transformer is pointless.** It scores 0.986 against
0.985 for a bag-of-ngrams — a gain of 0.001 for roughly 1,300x the
inference cost (131 ms vs 0.1 ms per sample). Both models are memorizing
generator templates. Anyone quoting the 0.98 as a headline is quoting the
generator's regularity, not a security capability.

**On real code the gap is real but modest.** 0.863 vs 0.838 vs 0.646.
The interesting split is recall against false positives:

| | Vulnerable-function recall | False-positive rate |
|---|---:|---:|
| Bandit | 0.614 | 0.007 |
| Random Forest | 0.795 | 0.027 |
| UniXcoder | 0.850 | 0.041 |

The model finds **85% of vulnerable functions where Bandit finds 61%**, at
**6x Bandit's false-positive rate** (4.1% vs 0.7%). Whether that trade is
worth it depends on whether the reviewer would rather chase 4.1% noise or miss
15% of real findings. It is a trade, not a free win.

### Per-class on real code

| Class | F1 | n | |
|---|---:|---:|---|
| COMMAND_INJECTION | 0.926 | 73 |  |
| HARDCODED_SECRET | 0.000 | 1 | too few to read |
| PATH_TRAVERSAL | 0.706 | 36 |  |
| SAFE | 0.958 | 438 |  |
| SQL_INJECTION | 0.800 | 3 | too few to read |
| UNSAFE_DESERIALIZATION | 0.778 | 9 | too few to read |
| XSS | 0.000 | 5 | too few to read |

### Localization

Gradient x input attribution, scored against the labeling analyzer's finding
line on 127 held-out records:

| | |
|---|---:|
| Top-1 line accuracy | **0.701** |
| Top-3 line accuracy | **0.850** |

Ground truth is where Semgrep or the AST detectors fired, not a human
annotation, so this measures agreement with the labeler's notion of the
offending line.

### Three things the measurements overturned

A results section listing only what worked is a sales pitch. These are the
decisions the data killed.

**1. INT8 quantization destroys this model.** Decision D4 specified INT8 CPU
serving. Measured on 400 real held-out functions:

| Configuration | Supported macro F1 | Accuracy | Latency |
|---|---:|---:|---:|
| float32 | **0.861** | 0.917 | 260 ms |
| INT8, all Linear | 0.286 | 0.752 | 133 ms |
| INT8, encoder only | 0.286 | 0.752 | 130 ms |

Half the latency, and 0.752 accuracy is roughly the SAFE base rate — it says
SAFE to almost everything. Quantization is now off by default.

This was caught by `make demo`, which pushes the real checkpoint through the
real API and reported **zero findings on code with five planted
vulnerabilities**. The 62-test suite missed it completely, because those tests
stub the predictor. A test that stubs the component most likely to break cannot
tell you it broke.

**2. Temperature scaling fit on synthetic data made calibration worse.**
Fitting on the full validation set (89% synthetic, model ~99.7% accurate there)
gave T = 0.9618 — sharpening an already-overconfident model — and moved
real-code ECE from 0.0592 to 0.0609, the wrong way. Refitting on the real-code
subset gives T = 1.0785 and real-code ECE 0.0592 -> 0.0558. Calibration now
fits on real samples by default.

It still does not fully solve the problem: the model reports 97.6% mean
confidence at 92.4% accuracy on real code, so it remains overconfident by
5.2 points. One scalar cannot reconcile two very different accuracy
regimes.

**3. Function-level chunking was blind to module scope.** The demo missed a
planted `STRIPE_KEY = "sk_live_..."` because it sits at module level — which is
where credentials actually live — while the function that returns it is
correctly SAFE. The scan path now analyzes contiguous runs of top-level
statements as well. The demo finds all five.

### Training

Stopped after epoch 2 on evidence, not impatience: validation macro F1 went
**down** (0.9959 -> 0.9939) while training loss collapsed (0.378 -> 0.029) and
false-positive rate doubled. The epoch-1 checkpoint was retained. Validation is
89% synthetic, so it saturates almost immediately and is a weak
checkpoint-selection signal — the trainer now warns when this is the case, and
early stopping is wired in.

| | |
|---|---|
| Base model | microsoft/unixcoder-base, 125.9M params |
| Corpus | 19,649 functions, 276 repositories, 3,184 real |
| Hardware | 1x RTX 5060 Laptop (8GB), bf16, ~31 min/epoch |
| CPU inference | 131 ms/function (synthetic), 346 ms (real, longer functions) |


## Architecture

| Layer | Choice | Why |
|---|---|---|
| Model | UniXcoder-base (125.9M), float32 on CPU | Fits a 512-token window; CPU-only serving keeps idle cost ~zero. INT8 was measured and rejected — see below |
| Unit | Functions + module scope via tree-sitter | Matches dataset labeling; module scope added after the demo missed a top-level credential |
| Execution | Async job schema from commit one | In-process worker locally, RQ in production, one API contract |
| Localization | Gradient×input attribution | Needs no line labels; scored with top-k line accuracy |
| Confidence | Temperature scaling | Raw softmax is overconfident and would mislead a developer acting on it |
| Ingestion | Zip upload, hardened extraction | Avoids the SSRF surface of server-side git cloning |
| Labels | Semgrep + custom AST detectors | Bandit is the *baseline*; labeling with it would make the benchmark circular |

`ml/securescan_ml/labels.py` is the single source of truth for the taxonomy,
severity map, and remediation templates. The backend imports from it, and
`GET /api/taxonomy` serves it to the frontend, so the model head, the database,
and the dashboard cannot drift apart.

## Quick start

```bash
make install
make dataset      # generate + mine + label + dedup + split  (~15 min)
make train        # fine-tune on GPU                          (~20 min)
make calibrate    # fit temperature scaling  <-- do not skip
make benchmark    # Bandit vs Random Forest vs Transformer
make demo         # real model through the real API
```

Or run the stack:

```bash
docker compose up                    # API + Postgres, in-process worker
docker compose --profile queue up    # adds Redis + a separate worker
```

API at `http://localhost:8000` (docs at `/docs`), dashboard at
`http://localhost:3000`.

## How it works

```
zip upload
    ↓  hardened extraction (traversal, symlink, bomb, member-count checks)
tree-sitter extracts functions, preserving byte offsets
    ↓
UniXcoder INT8 on CPU, batched
    ↓
temperature-scaled confidence  →  below threshold, dropped
    ↓
gradient×input attribution → per-line scores → absolute file lines
    ↓
severity from the static type→severity map; templated remediation attached
    ↓
findings to PostgreSQL; uploaded source deleted
```

## Dataset

Built by `make dataset`; nothing is committed. Two sources:

- **Synthetic** (`data/synthetic.py`) — a compositional generator, not fixed
  templates. Composes taint source × sink variant × wrapper structure ×
  identifier pool × noise, so the invariant across samples is the data flow
  rather than the surface form. 0 parse failures, 100% unique samples.
- **Mined real code** — top ~220 PyPI packages by downloads, labeled with
  Semgrep plus two independent AST detectors for the classes Semgrep does not
  cover (measured: it detects 0/25 hardcoded secrets and 0/25 path traversals).

Full methodology, coverage measurements, and the anti-circularity argument:
[datasets/README.md](datasets/README.md).

## Layout

```
backend/app/       FastAPI: api/ routers, services/ ingest+storage, workers/ queue+scan
backend/alembic/   migrations
ml/securescan_ml/  labels, chunking, data/, training/, evaluation/, inference/
frontend/          Next.js dashboard
scripts/           demo_scan.py — real model through the real API
infrastructure/    worker image, Terraform (ECS Fargate, RDS, ElastiCache, S3)
tests/             pytest suite
```

## Tests

```bash
make test
```

The suite runs on SQLite with a stubbed model — no database server, no
checkpoint, no GPU. `tests/backend/test_ingest.py` is the one to keep green: it
asserts rejection of path traversal, absolute paths, zip bombs, symlinks (zip
and tar), and member-count overflow. Uploaded archives are untrusted input.

## Limitations

Stated plainly because they are real.

1. **Labels are static-analysis output, not human-verified CVEs.** Semgrep and
   the AST detectors carry their own false positives and blind spots, and the
   model inherits both. Its ceiling is agreement with those labelers on
   held-out repositories, not ground truth. This is the biggest caveat on every
   number here.
2. **The real-code test set is small and skewed.** Four of the seven classes
   have fewer than ten real held-out examples, so their per-class numbers are
   noise. Only SAFE, command injection, and path traversal have enough support
   to read.
3. **Python only.** Nothing transfers without new data and a new grammar.
4. **Chunk-scoped.** The model sees one function (or one run of module-level
   statements) at a time and cannot follow taint across function or file
   boundaries. A vulnerability whose source and sink live in different
   functions will be missed.
5. **Still overconfident on real code.** Even after calibration it reports
   ~97.6% mean confidence at ~92.4% accuracy. Treat the percentage as a
   ranking signal, not a probability.
6. **Attribution is correlational.** Highlighted lines show what drove the
   prediction, not proof of exploitability.
7. **Not a replacement for a real scanner.** Complements Bandit, Semgrep, and
   CodeQL; does not supersede them — and Bandit still has a 6x lower
   false-positive rate.
