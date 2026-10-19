"""Typed wrappers for the platform endpoints used by the factory.

Every method returns plain dictionaries or lists so the factory does not depend on
HTTP implementation details.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from alpha_orchestrator.brain.client import BrainClient

PAGE = 50
MULTI_SIM_MAX_CHILDREN = 10


def simulation_payload(expression: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    full = {
        "instrumentType": "EQUITY",
        "language": "FASTEXPR",
        "pasteurization": "ON",
        "unitHandling": "VERIFY",
        "nanHandling": "OFF",
        "maxTrade": "OFF",
        "testPeriod": "P0Y0M0D",
        "visualization": False,
        **settings,
    }
    return {"type": "REGULAR", "settings": full, "regular": expression}


@dataclass
class SimulationProgress:
    """State of one simulation request (single or multi)."""

    done: bool
    status: str
    children: list[str] = field(default_factory=list)
    alpha_id: str | None = None
    message: str | None = None


@dataclass
class ChildResult:
    child_id: str
    status: str
    alpha_id: str | None
    message: str | None


class BrainAPI:
    def __init__(self, client: BrainClient) -> None:
        self.client = client

    # ---------- metadata ----------
    def datasets(self, region: str, delay: int, universe: str) -> list[dict[str, Any]]:
        query = {
            "instrumentType": "EQUITY",
            "region": region,
            "delay": delay,
            "universe": universe,
            "limit": PAGE,
        }
        return self._paged("/data-sets", query)

    def datafields(
        self, region: str, delay: int, universe: str, dataset_id: str
    ) -> list[dict[str, Any]]:
        query = {
            "instrumentType": "EQUITY",
            "region": region,
            "delay": delay,
            "universe": universe,
            "dataset.id": dataset_id,
            "limit": PAGE,
        }
        return self._paged("/data-fields", query)

    def operators(self) -> list[dict[str, Any]]:
        payload = self.client.wait_json("/operators")
        return list(payload or [])

    def simulation_options(self) -> dict[str, Any]:
        """Regions, universes, delays and neutralizations this account may use."""
        response = self.client.request("OPTIONS", "/simulations")
        settings = response.json()["actions"]["POST"]["settings"]["children"]
        return parse_simulation_options(settings)

    def pyramid_multipliers(self) -> list[dict[str, Any]]:
        payload = self.client.wait_json("/users/self/activities/pyramid-multipliers")
        return _records(payload)

    def pyramid_alphas(self) -> list[dict[str, Any]]:
        payload = self.client.wait_json("/users/self/activities/pyramid-alphas")
        return _records(payload)

    def user(self) -> dict[str, Any]:
        return dict(self.client.wait_json("/users/self") or {})

    # ---------- simulation ----------
    def start_multi_simulation(self, payloads: list[dict[str, Any]]) -> str:
        """Returns the progress URL. Raises ValueError when BRAIN refuses the batch."""
        if not 1 <= len(payloads) <= MULTI_SIM_MAX_CHILDREN:
            raise ValueError("A multi-simulation holds 1-10 alphas")
        body: Any = payloads if len(payloads) > 1 else payloads[0]
        response = self.client.request("POST", "/simulations", json=body)
        location = response.headers.get("Location")
        if response.status_code // 100 != 2 or not location:
            raise ValueError(f"Simulation refused ({response.status_code}): {response.text[:4000]}")
        return str(location)

    def poll(self, location: str) -> SimulationProgress:
        """One look at a simulation. A multi returns child ids; a single returns its alpha."""
        response = self.client.request("GET", location)
        if float(response.headers.get("Retry-After", 0) or 0) > 0:
            return SimulationProgress(done=False, status="RUNNING")
        data = response.json() if response.content.strip() else {}
        return SimulationProgress(
            done=True,
            status=str(data.get("status", "ERROR")),
            children=[str(c) for c in data.get("children") or []],
            alpha_id=data.get("alpha"),
            message=data.get("message"),
        )

    def child(self, child_id: str) -> ChildResult:
        data = self.client.wait_json(f"/simulations/{child_id}") or {}
        return ChildResult(
            child_id=child_id,
            status=str(data.get("status", "ERROR")),
            alpha_id=data.get("alpha"),
            message=data.get("message"),
        )

    # ---------- results ----------
    def alpha(self, alpha_id: str) -> dict[str, Any]:
        return dict(self.client.wait_json(f"/alphas/{alpha_id}") or {})

    def yearly_stats(self, alpha_id: str) -> list[dict[str, Any]]:
        return _table(self.client.wait_json(f"/alphas/{alpha_id}/recordsets/yearly-stats"))

    def daily_pnl(self, alpha_id: str) -> dict[str, float]:
        rows = _table(self.client.wait_json(f"/alphas/{alpha_id}/recordsets/daily-pnl"))
        out: dict[str, float] = {}
        for row in rows:
            value = row.get("pnl")
            if row.get("date") and isinstance(value, (int, float)):
                out[str(row["date"])] = float(value)
        return out

    def check(self, alpha_id: str) -> list[dict[str, Any]]:
        """Pre-submission checks, including SELF/PROD correlation once the rest pass."""
        payload = self.client.wait_json(f"/alphas/{alpha_id}/check") or {}
        return list((payload.get("is") or {}).get("checks") or [])

    def correlation_max(self, alpha_id: str, kind: str) -> float | None:
        payload = self.client.wait_json(f"/alphas/{alpha_id}/correlations/{kind}") or {}
        value = payload.get("max")
        return float(value) if isinstance(value, (int, float)) else None

    def performance_comparison(self, alpha_id: str) -> dict[str, Any]:
        path = f"/users/self/alphas/{alpha_id}/before-and-after-performance"
        return dict(self.client.wait_json(path) or {})

    def set_properties(
        self,
        alpha_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: Iterable[str] = (),
    ) -> None:
        body: dict[str, Any] = {"tags": list(tags)}
        if name:
            body["name"] = name
        if description:
            body["regular"] = {"description": description}
        self.client.request("PATCH", f"/alphas/{alpha_id}", json=body)

    def submit(self, alpha_id: str) -> bool:
        """Only ever called from the explicit `review approve` command."""
        response = self.client.request("POST", f"/alphas/{alpha_id}/submit")
        while float(response.headers.get("Retry-After", 0) or 0) > 0:
            self.client.sleep(float(response.headers["Retry-After"]))
            response = self.client.request("GET", f"/alphas/{alpha_id}/submit")
        return response.status_code == 200

    # ---------- helpers ----------
    def _paged(self, path: str, query: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        offset, total = 0, None
        while total is None or offset < total:
            payload = self.client.wait_json(f"{path}?{urlencode({**query, 'offset': offset})}")
            if not payload:
                break
            rows = list(payload.get("results") or [])
            total = int(payload.get("count", len(rows)))
            out.extend(rows)
            offset += PAGE
            if not rows:
                break
        return out


def parse_simulation_options(children: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the OPTIONS tree into {region: {universes, delays, neutralizations}}."""

    def choices(name: str) -> Any:
        return children.get(name, {}).get("choices", {}).get("instrumentType", {}).get("EQUITY")

    regions = [c["value"] for c in choices("region") or []]
    out: dict[str, Any] = {}
    universe_tree = choices("universe") or {}
    neut_tree = choices("neutralization") or {}
    delay_tree = choices("delay") or {}
    for region in regions:
        out[region] = {
            "universes": [c["value"] for c in universe_tree.get("region", {}).get(region, [])],
            "neutralizations": [c["value"] for c in neut_tree.get("region", {}).get(region, [])],
            "delays": [c["value"] for c in delay_tree.get("region", {}).get(region, [])],
        }
    return out


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(x) for x in payload]
    if isinstance(payload, dict):
        for key in ("results", "records", "pyramids"):
            if isinstance(payload.get(key), list):
                return [dict(x) if isinstance(x, dict) else {"value": x} for x in payload[key]]
    return []


def _table(payload: Any) -> list[dict[str, Any]]:
    """BRAIN record sets: {schema: {properties: [{name}]}, records: [[...]]}."""
    if not isinstance(payload, dict) or not payload.get("records"):
        return []
    names = [p["name"] for p in payload.get("schema", {}).get("properties", [])]
    return [dict(zip(names, row, strict=False)) for row in payload["records"]]
