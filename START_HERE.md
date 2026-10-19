# Start here

For a local, non-networked check:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
alpha-factory --help
```

Do not run live research from the public example configuration. Before an authorized run, create a private configuration, keep the LLM key and platform session outside the repository, run the LLM-only preflight, inspect its proposed hypotheses/formulas, and then choose an explicit backtest cap.

The safe progression is: local tests → LLM-only inspection → small authorized pilot → human review. Local success is not external research success.

