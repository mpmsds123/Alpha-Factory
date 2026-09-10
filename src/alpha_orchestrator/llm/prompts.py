"""Versioned prompt registration and resolution."""

from pathlib import Path

from alpha_orchestrator.domain.enums import LLMRole
from alpha_orchestrator.storage.repository import Repository

PROMPT_FILES: dict[LLMRole, tuple[str, str]] = {
    LLMRole.GENERATOR: ("generator.v1", "generator.v1.md"),
    LLMRole.REPAIR_PLANNER: ("repair-planner.v1", "repair-planner.v1.md"),
}


def default_prompt_directory() -> Path:
    repository_prompts = Path.cwd() / "prompts"
    if repository_prompts.is_dir():
        return repository_prompts
    return Path(__file__).parents[1] / "default_prompts"


def register_default_prompts(
    repository: Repository, directory: Path | None = None
) -> dict[LLMRole, str]:
    prompt_directory = directory or default_prompt_directory()
    versions: dict[LLMRole, str] = {}
    for role, (version, filename) in PROMPT_FILES.items():
        content = (prompt_directory / filename).read_text(encoding="utf-8")
        repository.register_prompt_version(version, role, content)
        versions[role] = version
    return versions
