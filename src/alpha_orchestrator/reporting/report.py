"""English Markdown reports derived only from the SQLite ledger."""

from collections import Counter
from uuid import UUID

from alpha_orchestrator.domain.enums import AttemptStatus, CandidateStatus
from alpha_orchestrator.storage.repository import Repository


def build_run_report(repository: Repository, run_id: str | UUID) -> str:
    run = repository.get_run(run_id)
    candidates = repository.list_candidates(run_id)
    attempts = repository.list_attempts(run_id)
    decisions = repository.list_decisions(run_id)
    platform_checks = repository.list_platform_checks(run_id)
    llm_requests = repository.list_llm_requests(run_id)
    candidate_counts = Counter(candidate.status for candidate in candidates)
    attempt_counts = Counter(attempt.status for attempt in attempts)
    gate_counts = Counter(decision.outcome for decision in decisions)
    platform_counts = Counter(check.status for check in platform_checks)
    platform_all_pass_count = sum(
        decision.gate_name == "G6_PLATFORM_READINESS"
        and decision.outcome == "PASS"
        and "PLATFORM_ALL_PASS_CONFIRMED" in decision.reason_codes
        for decision in decisions
    )
    llm_status_counts = Counter(request.validation_status for request in llm_requests)
    repair_counts = Counter(
        candidate.changed_dimension for candidate in candidates if candidate.changed_dimension
    )
    infrastructure_failures = (
        attempt_counts[AttemptStatus.RETRYABLE_ERROR] + attempt_counts[AttemptStatus.TERMINAL_ERROR]
    )
    duplicate_count = repository.count_duplicate_events(run_id)
    proposal_events = len(candidates) + duplicate_count
    structural_failures = candidate_counts[CandidateStatus.INVALID]
    structural_passes = len(candidates) - structural_failures
    limits = run.limit_policy_json
    unused_candidates = max(0, limits["maximum_distinct_candidates"] - len(candidates))
    unused_attempts = max(0, limits["maximum_simulation_result_rows"] - len(attempts))
    unused_seconds = max(
        0, int(limits["maximum_wall_clock_seconds"] - repository.age_seconds(run_id))
    )
    recorded_tokens = sum(request.total_tokens or 0 for request in llm_requests)
    unused_llm_requests = max(0, limits["maximum_llm_requests"] - len(llm_requests))

    lines = [
        f"# Run report: {run.run_id}",
        "",
        "## Frozen contract",
        "",
        f"- Dataset: `{run.dataset_id}`",
        f"- Cell: `{run.region} x {run.universe} x Delay {run.delay}`",
        f"- Data type: `{run.data_type}`",
        f"- Routing policy: `{run.routing_policy_version}`",
        f"- Limit policy: `{run.limit_policy_version}`",
        f"- Adapter mode: `{run.adapter_mode}`",
        f"- Generator prompt: `{run.generator_prompt_version}`",
        f"- Repair prompt: `{run.repair_prompt_version}`",
        "",
        "## Reconciliation",
        "",
        f"- Proposal events: {proposal_events}",
        f"- Distinct candidate fingerprints: {len(candidates)}",
        f"- Duplicate proposals reused: {duplicate_count}",
        f"- Exact reconciliation: {proposal_events} = {duplicate_count} duplicate + "
        f"{len(candidates)} distinct current records",
        f"- Structural pass/fail: {structural_passes}/{structural_failures}",
        f"- External simulation requests/attempt rows: {len(attempts)}/{len(attempts)}",
        f"- Completed simulation results: {attempt_counts[AttemptStatus.COMPLETED]}",
        f"- Infrastructure failures: {infrastructure_failures}",
        f"- Unresolved simulation attempts: {attempt_counts[AttemptStatus.PENDING]}",
        f"- Decisions: {len(decisions)}",
        "",
        "## Candidate states",
        "",
    ]
    for candidate_status in CandidateStatus:
        if candidate_counts[candidate_status]:
            lines.append(f"- {candidate_status}: {candidate_counts[candidate_status]}")
    lines.extend(["", "## Deterministic gate outcomes", ""])
    if not gate_counts:
        lines.append("- None")
    for outcome, count in sorted(gate_counts.items()):
        lines.append(f"- {outcome}: {count}")
    lines.extend(["", "## Repair actions", ""])
    if not repair_counts:
        lines.append("- None recorded")
    for dimension, count in sorted(repair_counts.items()):
        lines.append(f"- {dimension}: {count}")
    lines.extend(["", "## Platform checks", ""])
    if not platform_counts:
        lines.append("- None recorded")
    for platform_status, count in sorted(platform_counts.items()):
        lines.append(f"- {platform_status}: {count}")
    lines.extend(
        [
            "",
            "## LLM ledger",
            "",
            f"- Requests: {len(llm_requests)}",
            f"- Recorded tokens: {recorded_tokens}",
        ]
    )
    for validation_status, count in sorted(llm_status_counts.items()):
        lines.append(f"- LLM {validation_status}: {count}")
    lines.extend(
        [
            "",
            "## Unused limits",
            "",
            f"- Distinct candidates: {unused_candidates}",
            f"- Simulation attempt rows: {unused_attempts}",
            f"- LLM requests: {unused_llm_requests}",
            f"- Approximate wall-clock seconds: {unused_seconds}",
        ]
    )
    lines.extend(
        [
            "",
            "## Stop boundary",
            "",
            f"- Run status: `{run.status}`",
            f"- Stop reason: `{run.stop_reason or 'NONE'}`",
            "- Automatic submission: unavailable",
            (
                f"- Confirmed live platform-all-pass decisions: {platform_all_pass_count}"
                if platform_all_pass_count
                else "- Live platform-all-pass claims: none"
            ),
            "",
            "## Surviving lineage",
            "",
        ]
    )
    survivors = [
        candidate
        for candidate in candidates
        if candidate.status
        in {
            CandidateStatus.READY_FOR_PLATFORM_CHECK,
            CandidateStatus.READY_FOR_HUMAN_REVIEW,
        }
    ]
    if not survivors:
        lines.append("No surviving candidates.")
    for candidate in survivors:
        parent = candidate.parent_candidate_id or "ROOT"
        lines.append(
            f"- `{candidate.candidate_id}` parent `{parent}`; "
            f"fingerprint `{candidate.candidate_fingerprint}`"
        )
    return "\n".join(lines) + "\n"
