"""Read-only startup evidence; configuration health is not live acceptance."""

import sqlite3
from pathlib import Path
from typing import Any

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import today_utc


def inspect_local(config: FactoryConfig) -> dict[str, Any]:
    path = Path(config.db_path).expanduser().resolve()
    cap = min(config.budget.v1_daily_cap, config.budget.daily_sim_cap)
    start = config.budget.window_start()
    report: dict[str, Any] = {
        "day_utc": today_utc(),
        "daily_cap": cap,
        "quota_window_start_utc": start.isoformat(),
        "quota_timezone": config.budget.quota_timezone,
        "quota_reset_hour": config.budget.quota_reset_hour,
        "ledger": str(path),
        "ledger_exists": path.exists(),
        "authenticated": "not_checked",
        "live_acceptance": "not_verified",
    }
    if not path.exists():
        report["blockers"] = ["ledger_missing"]
        return report
    try:
        # SQLite read-only mode also observes committed WAL data. Never use immutable
        # mode against a live ledger; it would silently omit its WAL.
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
            report["reserved_today"] = conn.execute(
                "SELECT coalesce(sum(size),0) FROM batches WHERE created_at >= ? "
                "AND status NOT IN ('REFUSED','REJECTED')",
                (start.replace(tzinfo=None).isoformat(" "),),
            ).fetchone()[0]
            report["unknown_batches"] = conn.execute(
                "SELECT count(*) FROM batches WHERE status='UNKNOWN' "
                "OR (status='RUNNING' AND location='')"
            ).fetchone()[0]
            report["inflight_batches"] = conn.execute(
                "SELECT count(*) FROM batches WHERE status='RUNNING'"
            ).fetchone()[0]
            report["last_completed"] = conn.execute(
                "SELECT max(completed_at) FROM candidates"
            ).fetchone()[0]
        report["remaining_allocation"] = max(0, cap - report["reserved_today"])
        report["blockers"] = ["uncertain_batches"] if report["unknown_batches"] else []
    except sqlite3.Error as exc:
        report["blockers"] = ["ledger_unreadable"]
        report["error"] = type(exc).__name__
    return report
