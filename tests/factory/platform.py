"""An in-memory stand-in for the platform API, for scheduler and loop tests."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from alpha_orchestrator.brain.api import ChildResult, SimulationProgress

Score = Callable[[str, dict[str, Any]], dict[str, float]]


def default_score(expression: str, settings: dict[str, Any]) -> dict[str, float]:
    return {"sharpe": 0.5, "fitness": 0.3, "turnover": 0.2, "returns": 0.03, "test_sharpe": 0.4}


@dataclass
class Quota:
    remaining: int | None = None


@dataclass
class Client:
    quota: Quota = field(default_factory=Quota)


class FakeAPI:
    def __init__(self, score: Score = default_score, operators: dict[str, str] | None = None):
        self.score = score
        self.client = Client()
        self.batches: dict[str, list[dict[str, Any]]] = {}
        self.alphas: dict[str, dict[str, Any]] = {}
        self.properties: dict[str, dict[str, Any]] = {}
        self.checks: dict[str, list[dict[str, Any]]] = {}
        self.submitted: list[str] = []
        self.ops = operators or {}
        self.reject_next = False
        self._child: dict[str, str | None] = {}

    # metadata
    def simulation_options(self) -> dict[str, Any]:
        return {"R1": {"universes": ["TOP100"], "delays": [1], "neutralizations": ["N1", "N2"]}}

    def datasets(self, region: str, delay: int, universe: str) -> list[dict[str, Any]]:
        return [{"id": "d1", "name": "Data one", "category": {"id": "cat"}, "fieldCount": 5}]

    def datafields(self, *args: Any) -> list[dict[str, Any]]:
        return [
            {"id": "sig_a", "type": "MATRIX", "coverage": 0.9, "description": "a quantity"},
            {"id": "sig_b", "type": "MATRIX", "coverage": 0.9, "description": "another one"},
            {"id": "zone_f", "type": "GROUP", "coverage": 1.0, "description": "groups"},
        ]

    def operators(self) -> list[dict[str, Any]]:
        return [{"name": k, "definition": v, "description": ""} for k, v in self.ops.items()]

    # simulation
    def start_multi_simulation(self, payloads: list[dict[str, Any]]) -> str:
        if self.reject_next:
            self.reject_next = False
            raise ValueError(
                'Simulation refused (400): [{"regular": ["bad"]}'
                + ", {}" * (len(payloads) - 1)
                + "]"
            )
        loc = f"/simulations/b{len(self.batches)}"
        self.batches[loc] = payloads
        return loc

    def poll(self, location: str) -> SimulationProgress:
        payloads = self.batches[location]
        children = []
        for i, p in enumerate(payloads):
            cid = f"{location[-2:]}c{i}"
            aid = f"A{len(self.alphas)}"
            m = self.score(p["regular"], p["settings"])
            block = {k: m[k] for k in ("sharpe", "fitness", "turnover", "returns") if k in m}
            settings = {k: v for k, v in p["settings"].items() if k != "testPeriod"}
            self.alphas[aid] = {
                "regular": {"code": p["regular"]},
                "settings": settings,
                "is": {
                    **block,
                    "checks": m.get("checks", [{"name": "LOW_SHARPE", "result": "PASS"}]),
                },
                "train": block,
                "test": {"sharpe": m.get("test_sharpe")},
            }
            self._child[cid] = aid if not m.get("error") else None
            children.append(cid)
        return SimulationProgress(done=True, status="COMPLETE", children=children)

    def child(self, child_id: str) -> ChildResult:
        aid = self._child[child_id]
        return ChildResult(child_id, "COMPLETE" if aid else "ERROR", aid, None if aid else "bad")

    def alpha(self, alpha_id: str) -> dict[str, Any]:
        return self.alphas.get(alpha_id, {})

    def check(self, alpha_id: str) -> list[dict[str, Any]]:
        return self.checks.get(
            alpha_id,
            [
                {"name": "LOW_SHARPE", "result": "PASS"},
                {"name": "SELF_CORRELATION", "result": "PASS", "value": 0.3},
                {"name": "PROD_CORRELATION", "result": "PASS", "value": 0.4},
            ],
        )

    def submit(self, alpha_id: str) -> bool:
        self.submitted.append(alpha_id)
        return True

    def set_properties(self, alpha_id: str, **kw: Any) -> None:
        self.properties[alpha_id] = kw
