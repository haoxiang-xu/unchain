"""Startup health supports known schemas without performing migrations."""

import sqlite3

import pytest

from unchain.persistence.sqlite_memory_v2 import SQLiteMemoryV2Store
from unchain.persistence.sqlite_read_v2 import (
    SQLiteContextV2ReadError,
    read_sqlite_context_v2_store_status,
)
from unchain.persistence.sqlite_v2 import SQLiteContextV2Store


@pytest.mark.parametrize("versions", [(1, 2), (1, 2, 3), (1,), (1, 3), (1, 2, 4), (1, 2, 3, 4)])
def test_status_reads_known_versions_and_rejects_unknown_without_writing(tmp_path, versions):
    database = tmp_path / "journal.sqlite3"
    objects = tmp_path / "objects"
    SQLiteContextV2Store(database_path=database, object_directory=objects)
    SQLiteMemoryV2Store(database_path=database, object_directory=objects)
    with sqlite3.connect(database) as connection:
        if versions == (1, 2):
            names = [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND name LIKE 'context_v2_%integrity_%'"
            )]
            for name in names:
                connection.execute(f'DROP TRIGGER "{name}"')
            connection.execute("ALTER TABLE executions DROP COLUMN integrity_revision")
        connection.execute("DELETE FROM context_v2_schema")
        connection.executemany(
            "INSERT INTO context_v2_schema(version) VALUES (?)", [(v,) for v in versions],
        )
    with sqlite3.connect(database) as connection:
        before = tuple(connection.iterdump())
    if versions in ((1, 2), (1, 2, 3)):
        status = read_sqlite_context_v2_store_status(database)
        assert status.available
        assert status.schema_version == max(versions)
    else:
        with pytest.raises(SQLiteContextV2ReadError, match="unsupported"):
            read_sqlite_context_v2_store_status(database)
    with sqlite3.connect(database) as connection:
        assert tuple(connection.iterdump()) == before
