"""Turn a researcher-authored template file into root candidates for a live run.

A live BRAIN adapter supplies data, never ideas, so a live run has no candidates unless
something seeds it. This module reads a plain text file of expression templates and expands
them against the frozen metadata snapshot, deterministically and without a model, so a first
run can be executed and defended line by line.

Template file format, one template per line:

    # comments and blank lines are ignored
    rank(<operator>(ts_backfill(<field>, 120), 60))
    ts_rank(ts_backfill(<field>, 120), 250)

`<field>` expands to every field in the snapshot, `<operator>` to every time-series operator.
Chinese placeholders `<字段>` and `<算子>` mean the same thing. A line with no placeholder is
used exactly as written.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from alpha_orchestrator.domain.enums import DataType
from alpha_orchestrator.domain.models import CandidateProposal, MetadataSnapshot

FIELD_PLACEHOLDERS = ("<field>", "<字段>")
OPERATOR_PLACEHOLDERS = ("<operator>", "<算子>")

# `<operator>(x, d)` slots may only receive operators with that exact shape. A `ts_`
# prefix alone is insufficient: correlation/covariance require two inputs plus lookback.
TWO_ARGUMENT_TIME_SERIES_OPERATORS = frozenset(
    {
        "ts_arg_max",
        "ts_arg_min",
        "ts_av_diff",
        "ts_backfill",
        "ts_count_nans",
        "ts_decay_linear",
        "ts_delay",
        "ts_delta",
        "ts_max",
        "ts_mean",
        "ts_median",
        "ts_min",
        "ts_product",
        "ts_rank",
        "ts_scale",
        "ts_std_dev",
        "ts_sum",
        "ts_zscore",
    }
)


class TemplateError(ValueError):
    """A template file could not be read or produced nothing usable."""


def load_templates(path: Path) -> list[str]:
    """Read one template per line, dropping blank lines and `#` comments."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TemplateError(f"Template file could not be read: {path}") from exc
    templates = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not templates:
        raise TemplateError(f"Template file contains no templates: {path}")
    return templates


def _placeholder_in(template: str, placeholders: Sequence[str]) -> str | None:
    return next((token for token in placeholders if token in template), None)


def time_series_operators(operators: Iterable[str]) -> list[str]:
    return sorted(
        name for name in operators if name.casefold() in TWO_ARGUMENT_TIME_SERIES_OPERATORS
    )


def expand_templates(
    templates: Sequence[str],
    metadata: MetadataSnapshot,
    *,
    data_type: DataType,
    settings: Mapping[str, Any],
    limit: int | None = None,
) -> list[CandidateProposal]:
    """Expand templates against the snapshot in a stable, reproducible order.

    Fields and operators are sorted, and templates are walked in file order, so the same
    template file and the same snapshot always produce the same candidates in the same
    sequence. `limit` truncates the result; the engine applies its own ceiling as well.
    """
    fields = sorted(name for name, kind in metadata.fields.items() if kind is data_type)
    if not fields:
        raise TemplateError(f"Metadata snapshot contains no {data_type} fields")
    operators = time_series_operators(metadata.operators)

    proposals: list[CandidateProposal] = []
    seen: set[str] = set()
    for template in templates:
        field_token = _placeholder_in(template, FIELD_PLACEHOLDERS)
        operator_token = _placeholder_in(template, OPERATOR_PLACEHOLDERS)
        if operator_token is not None and not operators:
            raise TemplateError(
                "Template requires a time-series operator but the snapshot contains none"
            )
        field_values = fields if field_token else [None]
        operator_values = operators if operator_token else [None]
        for field in field_values:
            for operator in operator_values:
                expression = template
                if field_token and field is not None:
                    expression = expression.replace(field_token, field)
                if operator_token and operator is not None:
                    expression = expression.replace(operator_token, operator)
                if expression in seen:
                    continue
                seen.add(expression)
                proposals.append(
                    CandidateProposal(
                        expression=expression,
                        settings=dict(settings),
                        mechanism_claim=(
                            f"Seed template {template!r} expanded against dataset "
                            f"{metadata.dataset_id}; no economic claim is asserted yet."
                        ),
                        uncertainty=(
                            "Template expansion is a search, not a hypothesis. Any result "
                            "here is one draw among many and carries no standalone evidence."
                        ),
                    )
                )
                if limit is not None and len(proposals) >= limit:
                    return proposals
    return proposals


def seed_from_file(
    path: Path,
    metadata: MetadataSnapshot,
    *,
    data_type: DataType,
    settings: Mapping[str, Any],
    limit: int | None = None,
) -> list[CandidateProposal]:
    return expand_templates(
        load_templates(path),
        metadata,
        data_type=data_type,
        settings=settings,
        limit=limit,
    )
