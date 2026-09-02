# SecureScan. Run `make help` for the pipeline in order.

PY      := .venv/bin/python
PIP     := .venv/bin/pip
EXPORT  := PYTHONPATH=backend:ml
MODEL   := artifacts/unixcoder-v1
DATA    := datasets/normalized.jsonl

.PHONY: help venv install dataset train calibrate benchmark localization demo test lint format api worker frontend clean

help:
	@echo "Pipeline, in order:"
	@echo "  make install     create .venv and install everything"
	@echo "  make dataset     generate synthetic + mine PyPI, label, dedup, split"
	@echo "  make train       fine-tune UniXcoder (GPU if present)"
	@echo "  make calibrate   fit temperature scaling  <-- do not skip"
	@echo "  make benchmark   Bandit vs Random Forest vs Transformer"
	@echo "  make localization  top-1/top-3 line accuracy for attribution"
	@echo "  make demo        real model + real API on a sample repo"
	@echo ""
	@echo "Development:"
	@echo "  make test        pytest"
	@echo "  make lint        ruff check"
	@echo "  make format      ruff format"
	@echo "  make api         run the API locally"
	@echo "  make worker      run an RQ worker (needs SECURESCAN_REDIS_URL)"

venv:
	python3 -m venv .venv || true
	$(PY) -m ensurepip --upgrade 2>/dev/null || \
	  (curl -sS https://bootstrap.pypa.io/get-pip.py | $(PY) -)

install: venv
	$(PIP) install -q -r requirements-dev.txt
	$(PIP) install -q -r ml/requirements.txt

dataset:
	$(EXPORT) $(PY) -m securescan_ml.data.build --output $(DATA) --packages 220 --per-class 1100

train:
	$(EXPORT) $(PY) -m securescan_ml.training.train --data $(DATA) --output $(MODEL)

calibrate:
	$(EXPORT) $(PY) -m securescan_ml.training.calibrate --model $(MODEL) --data $(DATA)

localization:
	$(EXPORT) $(PY) -m securescan_ml.evaluation.eval_localization --model $(MODEL) --data $(DATA)

benchmark:
	$(EXPORT) $(PY) -m securescan_ml.evaluation.run_benchmark --data $(DATA) --model $(MODEL)

demo:
	$(EXPORT) $(PY) scripts/demo_scan.py

test:
	$(EXPORT) $(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check backend ml tests

format:
	$(PY) -m ruff format backend ml tests

api:
	$(EXPORT) $(PY) -m uvicorn app.main:app --reload --app-dir backend

worker:
	$(EXPORT) $(PY) -m app.workers.rq_worker

frontend:
	cd frontend && npm install && npm run dev

clean:
	find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .storage
