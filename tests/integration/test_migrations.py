from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

ROOT = Path(__file__).parents[2]
MILESTONE_1_TABLES = {
    "alembic_version",
    "candidates",
    "decisions",
    "events",
    "lineage_edges",
    "platform_checks",
    "runs",
    "simulation_attempts",
}
MILESTONE_2_TABLES = {
    "llm_requests",
    "method_cards",
    "prompt_versions",
    "source_documents",
    "source_sections",
}
MILESTONE_3_TABLES = {"metadata_snapshots"}


def test_migrations_preserve_milestone_schema_boundaries(tmp_path: Path) -> None:
    database_path = tmp_path / "migration-check.sqlite"
    database_url = f"sqlite:///{database_path}"
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    config.config_file_name = None

    command.upgrade(config, "0001_milestone_1")
    engine = create_engine(database_url)
    assert set(inspect(engine).get_table_names()) == MILESTONE_1_TABLES

    command.upgrade(config, "0002_knowledge_llm")
    assert set(inspect(engine).get_table_names()) == MILESTONE_1_TABLES | MILESTONE_2_TABLES
    assert "context_manifest_json" in {
        column["name"] for column in inspect(engine).get_columns("llm_requests")
    }

    command.upgrade(config, "head")
    assert set(inspect(engine).get_table_names()) == (
        MILESTONE_1_TABLES | MILESTONE_2_TABLES | MILESTONE_3_TABLES
    )

    command.downgrade(config, "0002_knowledge_llm")
    assert set(inspect(engine).get_table_names()) == MILESTONE_1_TABLES | MILESTONE_2_TABLES

    command.downgrade(config, "0001_milestone_1")
    assert set(inspect(engine).get_table_names()) == MILESTONE_1_TABLES
    engine.dispose()
