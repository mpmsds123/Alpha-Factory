import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from alpha_orchestrator.factory import report
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import (
    DecisionORM,
    ExpressionORM,
    HStatus,
    HypothesisORM,
    PriorORM,
    Review,
    SimStatus,
    SimulationORM,
)


def _seed(ledger: Ledger, usd: float, stored: int) -> None:
    now = datetime.now(UTC)
    with ledger.session() as s:
        h = HypothesisORM(
            region="R",
            dataset_id="d",
            round=1,
            status=HStatus.CLOSED,
            best_train_sharpe=1.3,
            closed_at=now,
        )
        s.add(h)
        s.flush()
        good = ExpressionORM(hypothesis_id=h.id, expression="x")
        bad = ExpressionORM(
            hypothesis_id=h.id, expression="y", valid=False, rejection="unknown operator z"
        )
        s.add_all([good, bad])
        s.flush()
        for i in range(max(stored, 1)):
            s.add(
                SimulationORM(
                    expression_id=good.id,
                    hypothesis_id=h.id,
                    region="R",
                    delay=1,
                    settings={"decay": 0},
                    fingerprint=uuid4().hex,
                    status=SimStatus.DONE,
                    completed_at=now,
                    train_pass=True,
                    oos_pass=True,
                    checks_pass=True,
                    review=Review.PENDING if i < stored else Review.NONE,
                    expression="x",
                )
            )
        s.add(
            DecisionORM(
                hypothesis_id=h.id,
                step="diagnosis",
                cost_usd=usd,
                model="m",
                prompt_version="diagnosis.v1",
                output={"label": "idea_supported"},
                flags=["uncited"],
            )
        )
        s.add(
            DecisionORM(
                hypothesis_id=h.id, step="decision", actor="program", output={"action": "drop"}
            )
        )
        s.add(PriorORM(claim=uuid4().hex, source_type="own", history=[{}, {}]))


def test_guard_daily_cap_and_cost_per_alpha() -> None:
    cfg = FactoryConfig()
    cfg.cost.daily_usd_cap = 100.0
    ledger = Ledger("sqlite://")
    assert report.guard(ledger, cfg) == ""
    _seed(ledger, usd=4.0, stored=0)
    assert report.guard(ledger, cfg) == ""  # under the minimum spend
    _seed(ledger, usd=2.0, stored=0)
    assert "per alpha" in report.guard(ledger, cfg)
    _seed(ledger, usd=0.0, stored=4)  # 6.0 for 4 alphas = 1.5 each
    assert report.guard(ledger, cfg) == ""
    cfg.cost.daily_usd_cap = 5.0
    _seed(ledger, usd=0.0, stored=0)
    assert report.guard(ledger, cfg).startswith("daily LLM cap")
    later = datetime.now(UTC) + timedelta(days=10)
    assert report.guard(ledger, cfg, now=later) == ""


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_write_all_csvs(tmp_path: Path) -> None:
    cfg = FactoryConfig(reports_dir=str(tmp_path))
    ledger = Ledger("sqlite://")
    report.write_all(ledger, cfg)  # empty database still writes headers
    _seed(ledger, usd=0.5, stored=1)
    out = report.write_all(ledger, cfg, status="running")
    stages = _read(out / "stages.csv")
    assert len(stages) == 1
    row = stages[0]
    assert (row["expressions_proposed"], row["expressions_valid"]) == ("2", "1")
    assert (row["stored_for_review"], row["diag_idea_supported"]) == ("1", "1")
    assert (row["program_overrides"], row["uncited_decisions"], row["status"]) == (
        "1",
        "1",
        "running",
    )
    assert row["usd_per_stored"] == "0.5" and row["round1_cleared"] == "1"
    assert _read(out / "rejections.csv")[0]["unknown operator"] == "1"
    pm = _read(out / "by_prompt_model.csv")[0]
    assert (pm["passed_hypotheses"], pm["passes_per_usd"]) == ("1", "2.0")
    assert _read(out / "hypotheses.csv")[0]["id"] == "H1"
    assert _read(out / "priors.csv")[0]["updates"] == "1"
    assert _read(out / "review_queue.csv")[0]["review"] == Review.PENDING


def test_rejection_kinds() -> None:
    assert report.rejection_kind("window(s) [17] not in") == "window"
    assert report.rejection_kind("something new") == "other"


def test_quota_day_label() -> None:
    cfg = FactoryConfig()
    # 22:00 Pacific on the 11th opens the window of the 12th
    assert report._day(datetime(2026, 11, 12, 6, 0, tzinfo=UTC), cfg) == "2026-11-12"
    assert report._day(datetime(2026, 11, 12, 4, 0, tzinfo=UTC), cfg) == "2026-11-11"
    assert report._day(None, cfg) == ""
