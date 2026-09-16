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

    command.upgrade(config, "0003_live_metadata")
    assert set(inspect(engine).get_table_names()) == (
        MILESTONE_1_TABLES | MILESTONE_2_TABLES | MILESTONE_3_TABLES
    )
    # Not asserting the new column is absent here: 0003's own migration builds
    # this table from the live ORM class (checkfirst=True), so replaying
    # migrations from scratch against current code already has it by this
    # point. Only a database that historically stopped at 0003 lacks it --
    # 0004's upgrade() is what actually needs to handle both cases.

    command.upgrade(config, "head")
    assert set(inspect(engine).get_table_names()) == (
        MILESTONE_1_TABLES | MILESTONE_2_TABLES | MILESTONE_3_TABLES
    )
    assert "operator_definitions_json" in {
        column["name"] for column in inspect(engine).get_columns("metadata_snapshots")
    }

    command.downgrade(config, "0003_live_metadata")
    assert "operator_definitions_json" not in {
        column["name"] for column in inspect(engine).get_columns("metadata_snapshots")
    }

    command.downgrade(config, "0002_knowledge_llm")
    assert set(inspect(engine).get_table_names()) == MILESTONE_1_TABLES | MILESTONE_2_TABLES

    command.downgrade(config, "0001_milestone_1")
    assert set(inspect(engine).get_table_names()) == MILESTONE_1_TABLES
    engine.dispose()


def test_0004_adds_the_column_to_a_database_that_really_stopped_at_0003(
    tmp_path: Path,
) -> None:
    """The scenario 0004's upgrade() actually exists for: a database that was
    never replayed from scratch against current code, so it genuinely lacks the
    column 0003's own migration would otherwise have smuggled in."""
    database_path = tmp_path / "stopped-at-0003.sqlite"
    database_url = f"sqlite:///{database_path}"
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    config.config_file_name = None

    command.upgrade(config, "0003_live_metadata")
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE metadata_snapshots_without_column AS "
            "SELECT snapshot_id, kind, dataset_id, region, universe, delay, data_type, "
            "content_hash, payload_json, source, captured_at FROM metadata_snapshots"
        )
        connection.exec_driver_sql("DROP TABLE metadata_snapshots")
        connection.exec_driver_sql(
            "ALTER TABLE metadata_snapshots_without_column RENAME TO metadata_snapshots"
        )
    assert "operator_definitions_json" not in {
        column["name"] for column in inspect(engine).get_columns("metadata_snapshots")
    }

    command.upgrade(config, "head")
    assert "operator_definitions_json" in {
        column["name"] for column in inspect(engine).get_columns("metadata_snapshots")
    }
    engine.dispose()
