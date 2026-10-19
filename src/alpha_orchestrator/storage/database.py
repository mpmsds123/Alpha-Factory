"""Database construction with SQLite foreign-key enforcement."""

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from alpha_orchestrator.storage.models import Base


class Database:
    def __init__(self, url: str) -> None:
        kwargs: dict[str, object] = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        if url in {"sqlite://", "sqlite:///:memory:"}:
            kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, future=True, **kwargs)
        if url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)
        self.session_factory = sessionmaker(
            bind=self.engine, class_=Session, expire_on_commit=False
        )

    @staticmethod
    def _enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()


def sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.expanduser().resolve()}"


def create_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)
