.PHONY: install check demo clean

PY ?= python3
DB ?= demo.db

install:
	$(PY) -m pip install -e '.[dev]'

# Lint, type-check, and run the full test suite (no network access needed).
check:
	ruff format --check .
	ruff check .
	mypy src
	pytest

# One-command offline demo: create a run, replay a sanitized fixture, print the report.
demo:
	rm -f $(DB)
	alpha-lab init --db $(DB)
	RUN_ID=$$(alpha-lab run create --dataset demo-dataset --region USA --universe TOP1000 \
	  --delay 1 --adapter replay --db $(DB)) && \
	alpha-lab run execute "$$RUN_ID" --adapter replay \
	  --fixture tests/fixtures/basic_replay.json --db $(DB) && \
	alpha-lab report "$$RUN_ID" --db $(DB)

clean:
	rm -f $(DB)
