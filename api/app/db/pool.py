"""Postgres connection pool.

psycopg3 with a connection pool; pgvector types registered per-connection so `vector` columns
round-trip as Python lists/numpy arrays without manual casting.
"""

from collections.abc import Iterator
from contextlib import contextmanager

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
    """Apply schema.sql. Idempotent — every statement is IF NOT EXISTS."""
    from pathlib import Path

    sql = (Path(__file__).parent / "schema.sql").read_text()
    with connection() as conn:
        conn.execute(sql)
