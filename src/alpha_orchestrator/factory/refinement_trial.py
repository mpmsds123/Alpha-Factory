"""A matched comparison of one LLM proposal and one deterministic repair choice."""

from fractions import Fraction
from math import isfinite
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from alpha_orchestrator.factory import fixes
from alpha_orchestrator.factory.models import CandidateORM, CandidateStatus, ResearchIdeaORM

VERSION = 1


def outcome(trial: dict[str, Any]) -> str:
    result = trial.get("result")
    if result is None:
        return "pending"
    rule, llm = result["rule"], result["llm"]
    if llm["state"] == "abstained":
        return "abstained"
    if any(arm["state"] != "observed" for arm in (rule, llm)):
        return "unusable"
    if trial["choices"]["rule"] == trial["choices"]["llm"]:
        return "same_choice"
    if rule["success"] == llm["success"]:
        return "same_outcome"
    return "llm_won" if llm["success"] else "rule_won"


def rule_choice(names: list[str], problem: str, stats: fixes.Stats) -> str:
    """Smoothed historical success rate. Menu order breaks ties."""

    def score(name: str) -> Fraction:
        wins, losses = stats.get((problem, name), [0, 0])
        return Fraction(wins + 1, wins + losses + 2)

    return max(names, key=score)


def settle(s: Session) -> None:
    """Save a result only after all assigned candidates have terminal states."""
    ideas = s.scalars(
        select(ResearchIdeaORM).where(
            func.json_type(ResearchIdeaORM.extra, "$.refinement_trial") == "object",
            func.json_type(ResearchIdeaORM.extra, "$.refinement_trial.result").is_(None),
        )
    )
    for idea in ideas:
        trial = idea.extra["refinement_trial"]
        rows = {name: s.get(CandidateORM, cid) for name, cid in trial["candidates"].items()}
        if any(
            row and row.status in (CandidateStatus.QUEUED, CandidateStatus.RUNNING)
            for row in rows.values()
        ):
            continue
        targets = set(trial["target_checks"])
        parent_checks = set(trial["parent_checks"])
        arms: dict[str, Any] = {}
        for name, fix in trial["choices"].items():
            row = rows.get(fix)
            if fix is None:
                arms[name] = {"state": "abstained"}
                continue
            if (
                row is None
                or row.status != CandidateStatus.DONE
                or row.sharpe is None
                or not isfinite(row.sharpe)
                or abs(row.sharpe) > trial["suspicious_sharpe"]
            ):
                arms[name] = {"state": "unusable", "candidate_status": row.status if row else None}
                continue
            observed = {
                check["name"]: check.get("result")
                for check in row.checks
                if isinstance(check, dict) and isinstance(check.get("name"), str)
            }
            if any(observed.get(target) not in ("PASS", "FAIL") for target in targets):
                arms[name] = {"state": "unusable", "reason": "target check unavailable"}
                continue
            remaining = fixes.fail_set(row.fails)
            cleared = {target for target in targets if observed[target] == "PASS"}
            new_failures = remaining - parent_checks
            arms[name] = {
                "state": "observed",
                "sharpe": row.sharpe,
                "cleared": sorted(cleared),
                "new_failures": sorted(new_failures),
                "success": cleared == targets and not new_failures,
            }
        idea.extra = {**idea.extra, "refinement_trial": {**trial, "result": arms}}
