"""Apply numbered SQL files in `app/db/migrations/`.

`schema.sql` creates the baseline (IF NOT EXISTS). Migrations evolve it. A row in
`schema_migrations` is the receipt — rerunning startup must not re-apply a file.
"""

from __future__ import annotations

from pathlib import Path

_MIGRATIONS = Path(__file__).parent / "migrations"

_ENSURE_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    id          TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def run_script(conn, sql: str) -> None:
    """Run a SQL file one statement at a time.

    psycopg3 `execute` accepts a single statement. These files are DDL — no dollar-quoted
    bodies — so splitting on ';' is enough.
    """
    statement = []
    for line in sql.splitlines():
        if line.strip().startswith("--"):
            continue
        statement.append(line)
        if line.rstrip().endswith(";"):
            chunk = "\n".join(statement).strip()
            if chunk:
                conn.execute(chunk)
            statement = []
    tail = "\n".join(statement).strip()
    if tail:
        conn.execute(tail)


def apply_migrations(conn) -> list[str]:
    run_script(conn, _ENSURE_TABLE)
    applied = {
        row["id"]
        for row in conn.execute("SELECT id FROM schema_migrations").fetchall()
    }
    ran: list[str] = []
    for path in sorted(_MIGRATIONS.glob("*.sql")):
        if path.name in applied:
            continue
        run_script(conn, path.read_text())
        conn.execute("INSERT INTO schema_migrations (id) VALUES (%s)", (path.name,))
        ran.append(path.name)
    return ran
