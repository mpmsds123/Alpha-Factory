# Test evidence

The suite covers the factory engine, validation, scheduling, LLM response contracts, persistence, replay/fake adapters, CLI paths, and external-error handling. Network-facing tests intercept responses; running the suite does not authenticate to or spend quota on an external service.

Publication verification on 2026-10-18:

- `pytest -q --no-cov`: 247 passed, 5 skipped;
- `ruff check .`: all checks passed;
- forbidden-source and sensitive-artifact scan: no copied-framework references, production
  databases, generated reports, credentials, real dataset allowlists, or local user paths found.

Test success demonstrates local behavior only; it does not demonstrate predictive validity,
out-of-sample performance, or platform approval.

