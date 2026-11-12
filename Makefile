.PHONY: install check

PY ?= python3
DB ?= demo.db

# Reproducible install from the exact pinned versions.
install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

# Lint, type-check, and run the full test suite (no network access needed).
check:
	black --check .
	ruff check .
	mypy src
	pytest
