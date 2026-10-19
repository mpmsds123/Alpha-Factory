from collections.abc import Callable
from pathlib import Path

from alpha_orchestrator.adapters.replay import ReplayBrainAdapter
from alpha_orchestrator.config import LimitPolicy, RoutingPolicy
from alpha_orchestrator.domain.enums import CandidateStatus, PlatformCheckStatus, RunStatus
from alpha_orchestrator.domain.models import RunSpecification
from alpha_orchestrator.orchestration.engine import OfflineOrchestrator
from alpha_orchestrator.reporting.report import build_run_report
from alpha_orchestrator.storage.repository import Repository

ROOT = Path(__file__).parents[2]


def test_sanitized_historical_failure_fixture_is_contained(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
) -> None:
    run = make_run(dataset_id="regression-dataset")
    adapter = ReplayBrainAdapter.from_path(
        ROOT / "tests" / "fixtures" / "historical_regressions.json"
    )
    assert len(adapter.fixture.regression_cases) == 9
    summary = OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        run.run_id, adapter
    )
    assert summary.status is RunStatus.STOPPED
    assert repository.count_candidates(run.run_id) == 2
    assert repository.count_duplicate_events(run.run_id) == 1
    assert repository.count_pending(run.run_id) == 1
    states = {item.status for item in repository.list_candidates(run.run_id)}
    assert CandidateStatus.READY_FOR_PLATFORM_CHECK in states
    assert CandidateStatus.SIMULATING in states
    attempts = repository.list_attempts(run.run_id)
    keys_by_candidate: dict[str, set[str]] = {}
    for attempt in attempts:
        keys_by_candidate.setdefault(attempt.candidate_id, set()).add(attempt.idempotency_key)
    assert all(len(keys) == 1 for keys in keys_by_candidate.values())
    report = build_run_report(repository, run.run_id)
    assert "Live platform-all-pass claims: none" in report
    assert "submittable" not in report.lower()

    ready = next(
        item
        for item in repository.list_candidates(run.run_id)
        if item.status == CandidateStatus.READY_FOR_PLATFORM_CHECK
    )
    checks = adapter.request_platform_checks(ready.candidate_id)
    statuses = {check.status for check in checks}
    assert statuses == {
        PlatformCheckStatus.FAIL,
        PlatformCheckStatus.WARNING,
        PlatformCheckStatus.ERROR,
    }
