# One-command reproducibility. `make all` runs the full pipeline from a fresh clone.
PY ?= python3
STUDENT_ID ?= 2025em1200044
STUDENT_NAME ?= Aditya Jha
REPO_URL ?= https://github.com/AdityaJha2/Mini_Production_ML_System

.PHONY: all install data train evaluate test robustness serve benchmark report clean

all: install data train evaluate test robustness report

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.generate_data

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

test:
	$(PY) -m pytest -q

robustness:
	$(PY) -m serving.robustness

serve:
	uvicorn serving.app:app --host 127.0.0.1 --port 8000

benchmark:
	$(PY) -m serving.benchmark --mode batch
	$(PY) -m serving.benchmark --mode api --requests 1000

report:
	$(PY) -m report.build_report --student-id "$(STUDENT_ID)" --student-name "$(STUDENT_NAME)" --repo-url "$(REPO_URL)"

clean:
	rm -rf data/processed/*.csv logs/*.log .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
