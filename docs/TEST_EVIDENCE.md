# Test evidence

The suite covers the factory engine, validation, scheduling, LLM response contracts, persistence, replay/fake adapters, CLI paths, and external-error handling. Network-facing tests intercept responses; running the suite does not authenticate to or spend quota on an external service.

Publication verification on 2026-10-19:

- `pytest -q`: 247 passed, 5 skipped; 85.22% combined statement/branch coverage;
- `ruff format --check .` and `ruff check .`: all checks passed;
- `mypy src`: 73 source files passed strict type checking;
- forbidden-source and sensitive-artifact scan: no copied-framework references, production
  databases, generated reports, credentials, real dataset allowlists, or local user paths found.

Test success demonstrates local behavior only; it does not demonstrate predictive validity,
out-of-sample performance, or platform approval.

