"""Postgres connection pool.

psycopg3 with a connection pool; pgvector types registered per-connection so `vector` columns
round-trip as Python lists/numpy arrays without manual casting.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import get_settings

_pool: ConnectionPool | None = None


def _configure(conn) -> None:
    register_vector(conn)


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=10,
            configure=_configure,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pool


@contextmanager
def connection() -> Iterator:
    """Transactional connection. Commits on success, rolls back on exception."""
    with get_pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def init_schema() -> None:
    """Apply schema.sql, then any pending files in `migrations/`.

    Uses a raw connection, not the pool. `register_vector` looks up the `vector` type OID, which
    does not exist until `CREATE EXTENSION vector` in schema.sql has run. Opening the pool first
    on an empty database fails before that statement can execute.
    """
    from pathlib import Path

    from app.db.migrate import apply_migrations, run_script

    settings = get_settings()
    sql = (Path(__file__).parent / "schema.sql").read_text()
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        run_script(conn, sql)
        apply_migrations(conn)
