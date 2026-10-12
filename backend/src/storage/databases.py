"""One PostgreSQL database per knowledge base, all on the DATABASE_URL server."""

from __future__ import annotations

import os
import re

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine


_DATABASE_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class DatabaseAdminError(RuntimeError):
    """Raised when a knowledge base database cannot be created or dropped."""


def server_url_from_env() -> str:
    """DATABASE_URL: the server to use, and its database for admin commands."""

    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("Missing required database configuration: DATABASE_URL")
    return url


def database_url(server_url: str, database: str) -> str:
    """The server URL pointed at another database on the same server."""

    return make_url(server_url).set(database=database).render_as_string(
        hide_password=False
    )


def _checked(database: str) -> str:
    if not _DATABASE_NAME_RE.match(database):
        raise ValueError(f"Invalid database name: {database!r}")
    return database


async def _admin(server_url: str, statement: str, parameters: dict | None = None):
    # CREATE/DROP DATABASE cannot run inside a transaction block.
    engine = create_async_engine(server_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(statement), parameters or {})
            return result.scalar() if result.returns_rows else None
    except SQLAlchemyError as exc:
        raise DatabaseAdminError(f"Database command failed: {exc}") from exc
    finally:
        await engine.dispose()


async def database_exists(server_url: str, database: str) -> bool:
    return bool(
        await _admin(
            server_url,
            "SELECT 1 FROM pg_database WHERE datname = :name",
            {"name": _checked(database)},
        )
    )


async def ensure_database(server_url: str, database: str) -> str:
    """Create the database if it is missing; returns its connection URL."""

    if not await database_exists(server_url, database):
        await _admin(server_url, f'CREATE DATABASE "{_checked(database)}"')
    return database_url(server_url, database)


async def drop_database(server_url: str, database: str) -> bool:
    """Drop the database, closing open connections; False if it did not exist."""

    if not await database_exists(server_url, database):
        return False
    await _admin(server_url, f'DROP DATABASE "{_checked(database)}" WITH (FORCE)')
    return True
