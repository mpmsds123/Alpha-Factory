# Test Evidence

This release is verified as one installable Python 3.11 package. The evidence below distinguishes
local contract checks from external validation.

## Verification commands

```bash
ruff format --check .
ruff check .
mypy src
pytest -q
python -m build --wheel --no-isolation
python -m pip install --force-reinstall --no-deps \
  dist/alpha_research_orchestrator-0.5.0-py3-none-any.whl
```

## Final local result

Verified on 2026-09-11 with CPython 3.11.11:

- `ruff format --check .`: 65 files already formatted.
- `ruff check .`: all checks passed.
- `mypy src`: success with no issues in 42 source files under strict mode.
- `pytest -q`: 116 passed in 6.45 seconds.
- Branch-aware coverage: 91.79%, above the required 85% floor.
- Wheel: `alpha_research_orchestrator-0.5.0-py3-none-any.whl` built successfully.
- Clean wheel install: version `0.5.0`, `alpha-lab`, and `alpha-lab-mcp` verified.

## Scope of automated evidence

- Unit and integration tests cover domain state, identities, policies, migrations, repositories,
  adapters, orchestration, reports, corpus inventory, method cards, retrieval, LLM contracts, live
  workflow boundaries, MCP tools, and portfolio-document consistency.
- Replay regression fixtures contain synthetic expressions and recreate known failure classes
  without copying private research content.
- All live HTTP tests use intercepted responses. The suite never contacts WorldQuant.
- MCP tests use the official SDK's in-memory client/server transport; they do not open a network
  service.
- Wheel verification checks the installed version and both `alpha-lab` and `alpha-lab-mcp` entry
  points independently of the source tree.

## Evidence not claimed

These tests do not establish predictive validity, out-of-sample performance, economic value,
current WorldQuant endpoint compatibility, or platform approval. Those are external research and
authorized integration questions, not package-test conclusions.
