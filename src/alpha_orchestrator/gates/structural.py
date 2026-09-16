"""Conservative structural checks against a frozen metadata snapshot."""

import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from alpha_orchestrator.domain.enums import DataType, ErrorClass
from alpha_orchestrator.domain.identity import tokenize_expression
from alpha_orchestrator.domain.models import MetadataSnapshot

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}|<[^>]+>|\b(?:TODO|PLACEHOLDER)\b", re.IGNORECASE)

# BRAIN's group operators (group_neutralize, group_rank, ...) take a built-in
# classification keyword as their group argument, e.g. group_neutralize(x, industry).
# That keyword is never a dataset field or an operator, so without this allowlist the
# validator flags every such call as an unknown field even though BRAIN accepts it.
GROUP_CLASSIFICATION_IDENTIFIERS = frozenset(
    {"market", "sector", "industry", "subindustry", "country", "exchange"}
)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: ErrorClass
    message: str
    repairable: bool = False


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


def _balanced(expression: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for character in expression:
        if character in "([{":
            stack.append(character)
        elif character in pairs and (not stack or stack.pop() != pairs[character]):
            return False
    return not stack


class StructuralValidator:
    def validate(
        self,
        expression: str,
        settings: Mapping[str, Any],
        metadata: MetadataSnapshot,
        data_type: DataType,
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        if PLACEHOLDER_RE.search(expression):
            issues.append(
                ValidationIssue(
                    code=ErrorClass.INVALID_EXPRESSION,
                    message="Expression contains an unresolved placeholder",
                    repairable=True,
                )
            )
        try:
            tokens = tokenize_expression(expression)
        except ValueError as exc:
            issues.append(ValidationIssue(code=ErrorClass.INVALID_EXPRESSION, message=str(exc)))
            tokens = []
        if not _balanced(expression):
            issues.append(
                ValidationIssue(
                    code=ErrorClass.INVALID_EXPRESSION,
                    message="Expression delimiters are unbalanced",
                    repairable=True,
                )
            )

        known_operators = {name.casefold() for name in metadata.operators}
        known_fields = {name.casefold(): kind for name, kind in metadata.fields.items()}
        for index, token in enumerate(tokens):
            if not IDENTIFIER_RE.fullmatch(token):
                continue
            key = token.casefold()
            next_token = tokens[index + 1] if index + 1 < len(tokens) else None
            if next_token == "(":
                if key not in known_operators:
                    issues.append(
                        ValidationIssue(
                            code=ErrorClass.INVALID_OPERATOR,
                            message=f"Unknown operator: {token}",
                        )
                    )
            elif (
                key not in known_fields
                and key not in {"true", "false"}
                and key not in GROUP_CLASSIFICATION_IDENTIFIERS
            ):
                issues.append(
                    ValidationIssue(
                        code=ErrorClass.INVALID_FIELD,
                        message=f"Unknown field: {token}",
                    )
                )
            elif key in known_fields and known_fields[key] is not data_type:
                issues.append(
                    ValidationIssue(
                        code=ErrorClass.INVALID_FIELD,
                        message=f"Field {token} is incompatible with {data_type}",
                    )
                )

        for name, value in settings.items():
            allowed = metadata.settings.get(name)
            if allowed is None or value not in allowed:
                issues.append(
                    ValidationIssue(
                        code=ErrorClass.INVALID_SETTING,
                        message=f"Setting is absent from the frozen snapshot: {name}={value!r}",
                    )
                )
        return ValidationResult(valid=not issues, issues=issues)
