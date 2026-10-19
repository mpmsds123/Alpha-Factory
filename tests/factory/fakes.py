"""An in-memory BRAIN stand-in for pipeline tests (no network)."""

from collections.abc import Callable
from typing import Any

from alpha_orchestrator.brain.api import ChildResult, SimulationProgress
from alpha_orchestrator.brain.client import Quota

Metrics = Callable[[str, dict[str, Any]], dict[str, Any] | None]


class _Client:
    def __init__(self) -> None:
        self.quota = Quota()

    def sleep(self, _s: float) -> None:
        pass


class FakeBrainAPI:
    """`metrics(expression, settings)` returns an IS dict, or None to make the child fail."""

    def __init__(
        self,
        metrics: Metrics,
        *,
        polls_before_done: int = 1,
        refuse_first: int = 0,
        max_running: int | None = None,
    ) -> None:
        self.max_running = max_running
        self.finished: set[str] = set()
        self.metrics = metrics
        self.client = _Client()
        self.polls_before_done = polls_before_done
        self.refuse_first = refuse_first
        self.sims: dict[str, list[dict[str, Any]]] = {}
        self.polls: dict[str, int] = {}
        self.alphas: dict[str, dict[str, Any]] = {}
        self.children: dict[str, ChildResult] = {}
        self.started = 0
        self.checks: dict[str, list[dict[str, Any]]] = {}
        self.pnl: dict[str, dict[str, float]] = {}
        self.yearly: dict[str, list[dict[str, Any]]] = {}
        self.submitted: list[str] = []

    def start_multi_simulation(self, payloads: list[dict[str, Any]]) -> str:
        if self.refuse_first > 0:
            self.refuse_first -= 1
            raise ValueError("CONCURRENT_SIMULATION_LIMIT_EXCEEDED")
        if self.max_running is not None and len(set(self.sims) - self.finished) >= self.max_running:
            raise ValueError("CONCURRENT_SIMULATION_LIMIT_EXCEEDED")
        self.started += len(payloads)
        location = f"/simulations/M{len(self.sims)}"
        self.sims[location] = payloads
        self.polls[location] = 0
        return location

    def poll(self, location: str) -> SimulationProgress:
        self.polls[location] += 1
        if self.polls[location] <= self.polls_before_done:
            return SimulationProgress(done=False, status="RUNNING")
        self.finished.add(location)
        kids = []
        for i, payload in enumerate(self.sims[location]):
            child_id = f"{location.rsplit('/', 1)[1]}c{i}"
            settings = payload["settings"]
            is_ = self.metrics(payload["regular"], settings)
            alpha_id = None
            if is_ is not None:
                alpha_id = f"A{len(self.alphas)}"
                keep = {
                    k: settings[k]
                    for k in (
                        "region",
                        "universe",
                        "delay",
                        "decay",
                        "neutralization",
                        "truncation",
                        "maxTrade",
                    )
                }
                self.alphas[alpha_id] = {
                    "id": alpha_id,
                    "regular": {"code": payload["regular"]},
                    "settings": keep,
                    "is": is_,
                }
            self.children[child_id] = ChildResult(
                child_id, "COMPLETE" if alpha_id else "ERROR", alpha_id, None if alpha_id else "bad"
            )
            kids.append(child_id)
        return SimulationProgress(done=True, status="COMPLETE", children=kids)

    def child(self, child_id: str) -> ChildResult:
        return self.children[child_id]

    def alpha(self, alpha_id: str) -> dict[str, Any]:
        return self.alphas[alpha_id]

    def check(self, alpha_id: str) -> list[dict[str, Any]]:
        return self.checks.get(
            alpha_id,
            [
                {"name": "SELF_CORRELATION", "result": "PASS", "value": 0.2},
                {"name": "PROD_CORRELATION", "result": "PASS", "value": 0.3},
            ],
        )

    def daily_pnl(self, alpha_id: str) -> dict[str, float]:
        return self.pnl.get(alpha_id, {})

    def yearly_stats(self, alpha_id: str) -> list[dict[str, Any]]:
        return self.yearly.get(alpha_id, [])

    def performance_comparison(self, alpha_id: str) -> dict[str, Any]:
        return {}

    def set_properties(self, alpha_id: str, **_: Any) -> None:
        pass

    def submit(self, alpha_id: str) -> bool:
        self.submitted.append(alpha_id)
        return True


def is_metrics(
    sharpe: float,
    fitness: float,
    turnover: float = 0.1,
    fails: str = "",
    margin: float = 0.001,
    returns: float = 0.05,
    long_count: int = 1500,
    short_count: int = 1500,
) -> dict[str, Any]:
    checks = [{"name": n, "result": "FAIL"} for n in fails.split(",") if n]
    return {
        "sharpe": sharpe,
        "fitness": fitness,
        "turnover": turnover,
        "returns": returns,
        "margin": margin,
        "drawdown": 0.05,
        "longCount": long_count,
        "shortCount": short_count,
        "checks": checks,
    }


class FakePlatform(FakeBrainAPI):
    """FakeBrainAPI plus catalog endpoints, for whole-day tests."""

    def __init__(
        self,
        metrics: Metrics,
        datasets: dict[str, list[dict[str, Any]]],
        categories: dict[str, str] | None = None,
        **kw: Any,
    ) -> None:
        kw.setdefault("polls_before_done", 0)
        super().__init__(metrics, **kw)
        self._categories = categories or {}
        self._datasets = datasets

    def simulation_options(self) -> dict[str, Any]:
        return {
            "USA": {
                "universes": ["TOP3000", "TOP1000"],
                "delays": [1],
                "neutralizations": ["NONE", "INDUSTRY", "SUBINDUSTRY", "MARKET", "SECTOR"],
            }
        }

    def datasets(self, region: str, delay: int, universe: str) -> list[dict[str, Any]]:
        return [
            {
                "id": ds,
                "name": ds,
                "category": {"id": self._categories.get(ds, "analyst")},
                "fieldCount": 10 + len(f),
                "pyramidMultiplier": 1.3,
                "alphaCount": 5,
                "coverage": 0.9,
            }
            for ds, f in self._datasets.items()
        ]

    def datafields(
        self, region: str, delay: int, universe: str, dataset_id: str
    ) -> list[dict[str, Any]]:
        return self._datasets[dataset_id]

    def operators(self) -> list[dict[str, Any]]:
        names = [
            "rank",
            "ts_rank",
            "ts_zscore",
            "ts_delta",
            "ts_mean",
            "ts_av_diff",
            "ts_std_dev",
            "ts_backfill",
            "vec_avg",
            "group_rank",
            "group_zscore",
            "group_neutralize",
            "trade_when",
            "multiply",
            "sign",
            "hump",
            "ts_decay_linear",
            "abs",
            "winsorize",
            "bucket",
            "densify",
            "group_zscore",
        ]
        # no signature parsed from "name(...)" with a variadic marker: arity unchecked
        return [{"name": n, "definition": f"{n}(x, ..)", "scope": ["REGULAR"]} for n in names]

    def pyramid_alphas(self) -> list[dict[str, Any]]:
        return []
