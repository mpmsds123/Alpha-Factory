"""The catalog cache: which datasets exist per region (delay 1), and their fields.

Dataset lists are refreshed once a day; a dataset's fields are fetched when the sampler
first opens it. The universe per region is the broadest one offered (wide pools pass the
sub-universe test more easily), unless the private vocabulary names one.
"""

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import CatalogORM
from alpha_orchestrator.factory.util import utc_now
from alpha_orchestrator.factory.vocab import Vocabulary

MIN_DATASET_FIELDS = 3  # a dataset with fewer fields offers nothing to sample
STALE_HOURS = 20.0  # refresh once per day


def pick_universe(region: str, offered: list[str], preferred: Mapping[str, str]) -> str | None:
    """The preferred universe when offered, else the largest TOP<n>, else the first."""
    if preferred.get(region) in offered:
        return preferred[region]
    tops = sorted(
        (u for u in offered if u.startswith("TOP") and u[3:].isdigit()), key=lambda u: -int(u[3:])
    )
    return tops[0] if tops else (offered[0] if offered else None)


def _upsert(s: Session, kind: str, key: dict[str, Any], values: dict[str, Any]) -> None:
    row = s.scalar(
        select(CatalogORM).where(
            CatalogORM.kind == kind,
            CatalogORM.region == key["region"],
            CatalogORM.delay == key["delay"],
            CatalogORM.universe == key["universe"],
            CatalogORM.dataset_id == key["dataset_id"],
            CatalogORM.field_id == key.get("field_id", ""),
        )
    )
    if row is None:
        row = CatalogORM(kind=kind, **key)
        s.add(row)
    for k, v in values.items():
        setattr(row, k, v)
    row.refreshed_at = utc_now()


def _category(row: Mapping[str, Any]) -> str:
    cat = row.get("category") or ""
    return str(cat.get("id") if isinstance(cat, dict) else cat).lower()


def upsert_datasets(
    s: Session, rows: Iterable[Mapping[str, Any]], region: str, delay: int, universe: str
) -> int:
    """Insert or update dataset rows; return how many were kept."""
    n = 0
    for r in rows:
        if int(r.get("fieldCount") or 0) < MIN_DATASET_FIELDS:
            continue
        key = {"region": region, "delay": delay, "universe": universe, "dataset_id": str(r["id"])}
        _upsert(
            s,
            "dataset",
            key,
            {
                "dataset_name": str(r.get("name") or ""),
                "category": _category(r),
                "description": str(r.get("description") or "")[:4000],
                "coverage": float(r.get("coverage") or 0.0),
                "alpha_count": int(r.get("alphaCount") or 0),
                "user_count": int(r.get("userCount") or 0),
                "field_count": int(r.get("fieldCount") or 0),
            },
        )
        n += 1
    return n


def refresh(api: BrainAPI, ledger: Ledger, config: FactoryConfig, vocab: Vocabulary) -> int:
    """Refresh the dataset list of every allowed region at the configured delay."""
    options = api.simulation_options()
    regions = config.policy.regions or sorted(options)
    total = 0
    for region in regions:
        offered = options.get(region)
        if not offered or config.policy.delay not in offered["delays"]:
            continue
        universe = pick_universe(region, offered["universes"], vocab.preferred_universe)
        if universe is None:
            continue
        rows = api.datasets(region, config.policy.delay, universe)
        with ledger.session() as s:
            n = upsert_datasets(s, rows, region, config.policy.delay, universe)
        ledger.log("catalog", region=region, universe=universe, datasets=n)
        total += n
    return total


def is_stale(ledger: Ledger, now: datetime | None = None) -> bool:
    """True when the dataset list was never loaded or is older than a day."""
    with ledger.session() as s:
        last = s.scalar(
            select(func.max(CatalogORM.refreshed_at)).where(CatalogORM.kind == "dataset")
        )
    if last is None:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    return (now or datetime.now(UTC)) - last > timedelta(hours=STALE_HOURS)


def load_fields(api: BrainAPI, ledger: Ledger, dataset: CatalogORM) -> int:
    """Fetch and cache one dataset's fields."""
    rows = api.datafields(dataset.region, dataset.delay, dataset.universe, dataset.dataset_id)
    n = 0
    with ledger.session() as s:
        for r in rows:
            if not r.get("id"):
                continue
            key = {
                "region": dataset.region,
                "delay": dataset.delay,
                "universe": dataset.universe,
                "dataset_id": dataset.dataset_id,
                "field_id": str(r["id"]),
            }
            _upsert(
                s,
                "field",
                key,
                {
                    "dataset_name": dataset.dataset_name,
                    "category": dataset.category,
                    "field_type": str(r.get("type") or "").upper(),
                    "description": str(r.get("description") or "")[:1000],
                    "coverage": float(r.get("coverage") or 0.0),
                    "alpha_count": int(r.get("alphaCount") or 0),
                    "user_count": int(r.get("userCount") or 0),
                },
            )
            n += 1
    ledger.log("fields_loaded", dataset=dataset.dataset_id, region=dataset.region, fields=n)
    return n
