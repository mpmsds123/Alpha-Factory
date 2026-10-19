"""Daily catalog refresh: which cells exist, what the platform pays for each, what you own.

A cell is (region, delay, universe, dataset). The universe per region is the broadest one
your account offers, because sub-universe tests are easier to pass from a wide pool
(规则手册) and the IQC runs used the same choice.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.domain.models import utc_now
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import CellORM

PREFERRED_UNIVERSE = {
    "USA": "TOP3000",
    "EUR": "TOP2500",
    "ASI": "MINVOL1M",
    "GLB": "TOP3000",
    "CHN": "TOP2000U",
    "IND": "TOP500",
    "JPN": "TOP1600",
    "KOR": "TOP600",
    "TWN": "TOP500",
    "HKG": "TOP500",
    "AMR": "TOP600",
}


def cell_id(region: str, delay: int, universe: str, dataset_id: str) -> str:
    return f"{region}|{delay}|{universe}|{dataset_id}"


def pick_universe(region: str, offered: list[str]) -> str | None:
    preferred = PREFERRED_UNIVERSE.get(region)
    if preferred in offered:
        return preferred
    tops = sorted(
        (u for u in offered if u.startswith("TOP") and u[3:].isdigit()), key=lambda u: -int(u[3:])
    )
    return tops[0] if tops else (offered[0] if offered else None)


def upsert_cells(
    s: Session, rows: Iterable[Mapping[str, Any]], region: str, delay: int, universe: str
) -> int:
    now = utc_now()
    count = 0
    for row in rows:
        cid = cell_id(region, delay, universe, str(row["id"]))
        cell = s.get(CellORM, cid) or CellORM(
            cell_id=cid, region=region, delay=delay, universe=universe, dataset_id=str(row["id"])
        )
        category = row.get("category") or {}
        cell.dataset_name = str(row.get("name") or "")
        cell.category = str(
            category.get("id") if isinstance(category, dict) else category or "other"
        )
        cell.description = str(row.get("description") or "")[:4000]
        cell.multiplier = float(row.get("pyramidMultiplier") or 1.0)
        cell.value_score = float(row.get("valueScore") or 0.0)
        cell.alpha_count = int(row.get("alphaCount") or 0)
        cell.user_count = int(row.get("userCount") or 0)
        cell.coverage = float(row.get("coverage") or 0.0)
        cell.field_count = int(row.get("fieldCount") or 0)
        cell.themes = list(row.get("themes") or [])
        cell.active = True
        cell.refreshed_at = now
        s.add(cell)
        count += 1
    return count


def refresh(api: BrainAPI, ledger: Ledger, config: FactoryConfig) -> dict[str, Any]:
    options = api.simulation_options()
    regions = config.policy.regions or sorted(options)
    summary: dict[str, Any] = {"regions": {}}
    for region in regions:
        offered = options.get(region)
        if not offered:
            continue
        universe = pick_universe(region, offered["universes"])
        if universe is None:
            continue
        for delay in config.policy.delays:
            if delay not in offered["delays"]:
                continue
            rows = [
                r
                for r in api.datasets(region, delay, universe)
                if int(r.get("fieldCount") or 0) >= 3
            ]
            with ledger.session() as s:
                n = upsert_cells(s, rows, region, delay, universe)
                ledger.log(s, "catalog", region=region, delay=delay, universe=universe, cells=n)
            summary["regions"][f"{region}/D{delay}/{universe}"] = n
    with ledger.session() as s:
        ledger.log(
            s,
            "pyramids",
            counts={
                f"{k[0]}|{k[1]}|{k[2]}": v for k, v in pyramid_counts(api.pyramid_alphas()).items()
            },
        )
    return summary


def pyramid_counts(records: Iterable[Mapping[str, Any]]) -> Counter[tuple[str, int, str]]:
    """Your submissions per (region, delay, category) this quarter. Tolerates both the
    nested ({"region":..., "category": {"id":...}, "alphaCount": n}) and the flat
    ({"pyramid": "USA/D1/ANALYST", "count": n}) shapes the platform has returned."""
    out: Counter[tuple[str, int, str]] = Counter()
    for r in records:
        if "pyramid" in r and isinstance(r["pyramid"], str):
            parts = r["pyramid"].split("/")
            if len(parts) == 3:
                key = (parts[0], int(parts[1].lstrip("Dd") or 1), parts[2].lower())
                out[key] += int(r.get("count") or r.get("alphaCount") or 0)
            continue
        category = r.get("category")
        cat = category.get("id") if isinstance(category, dict) else category
        if r.get("region") and cat is not None:
            key = (str(r["region"]), int(r.get("delay", 1)), str(cat).lower())
            out[key] += int(r.get("alphaCount") or r.get("count") or 0)
    return out


def latest_pyramid_counts(s: Session) -> Counter[tuple[str, int, str]]:
    from alpha_orchestrator.factory.models import EventORM

    event = s.scalar(
        select(EventORM).where(EventORM.kind == "pyramids").order_by(EventORM.id.desc()).limit(1)
    )
    out: Counter[tuple[str, int, str]] = Counter()
    if event:
        for key, value in (event.payload.get("counts") or {}).items():
            region, delay, cat = key.split("|")
            out[(region, int(delay), cat)] = int(value)
    return out
