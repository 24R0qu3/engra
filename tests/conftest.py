import logging

import pytest


def pytest_configure(config):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@pytest.fixture(autouse=True)
def _isolate_fts(monkeypatch, tmp_path_factory):
    """Point the SQLite FTS5 keyword index at a fresh temp db per test.

    Keeps the (now default) hybrid search path from touching the real user-data
    directory, and gives each test an empty, independent keyword index.
    """
    from engra import storage

    db = tmp_path_factory.mktemp("fts") / "fts.db"
    monkeypatch.setattr(storage, "FTS_DB_PATH", db)
    monkeypatch.setattr(storage, "_fts_available", None)
    monkeypatch.setattr(storage, "_fts_warned", False)
    yield
