import argparse
import logging
import tomllib
from pathlib import Path

import pytest

from engram.commands import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    _find_seq_index,
    _format_missing_pages,
    _get_chunk_sequence,
    _stale_warning,
    chunk_text,
    default_project,
    expand_paths,
    parse_page_range,
)
from engram.config import _DEFAULT_TOML, DEFAULTS
from engram.log import setup
from engram.readers import SUPPORTED_EXTENSIONS, read_markdown, read_rst, read_text
from engram.storage import clear_session, read_session, write_session

# ── chunk_text ────────────────────────────────────────────────────────────────


def test_short_text_single_chunk():
    text = "Hello world"
    assert chunk_text(text) == [text]


def test_exact_chunk_size_single_chunk():
    text = "x" * CHUNK_SIZE
    assert len(chunk_text(text)) == 1


def test_long_text_multiple_chunks():
    text = "x" * (CHUNK_SIZE * 2)
    chunks = chunk_text(text)
    assert len(chunks) > 1


def test_chunks_have_overlap():
    text = "ab" * CHUNK_SIZE  # long enough to split
    chunks = chunk_text(text)
    if len(chunks) > 1:
        # end of first chunk should appear at start of second
        overlap = chunks[0][CHUNK_SIZE - CHUNK_OVERLAP :]
        assert chunks[1].startswith(overlap)


def test_chunk_max_size():
    text = "x" * (CHUNK_SIZE * 3)
    for chunk in chunk_text(text):
        assert len(chunk) <= CHUNK_SIZE


def test_empty_text():
    assert chunk_text("") == [""]


# ── config ────────────────────────────────────────────────────────────────────


def test_defaults_are_valid_toml():
    parsed = tomllib.loads(_DEFAULT_TOML)
    assert "backend" in parsed
    assert "index" in parsed


def test_defaults_backend_type():
    assert DEFAULTS["backend"]["type"] == "local"


def test_defaults_index_copy():
    assert DEFAULTS["index"]["copy"] is True


def test_load_returns_defaults_when_no_file(tmp_path, monkeypatch):
    from engram import config

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "nonexistent.toml")
    result = config.load()
    assert result["backend"]["type"] == "local"
    assert result["index"]["copy"] is True


def test_load_merges_user_values(tmp_path, monkeypatch):
    from engram import config

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[index]\ncopy = false\n")
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_file)
    result = config.load()
    assert result["index"]["copy"] is False
    assert result["backend"]["type"] == "local"  # default preserved


def test_init_creates_config(tmp_path, monkeypatch):
    from engram import config

    cfg_file = tmp_path / "config.toml"
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_file)
    config.init()
    assert cfg_file.exists()


def test_init_does_not_overwrite(tmp_path, monkeypatch):
    from engram import config

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("custom = true\n")
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_file)
    config.init()
    assert cfg_file.read_text() == "custom = true\n"


# ── log setup ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_root_logger():
    root = logging.getLogger()
    original = list(root.handlers)
    root.handlers.clear()
    yield
    root.handlers.clear()
    root.handlers.extend(original)


def test_root_logger_set_to_debug(tmp_path):
    setup(log_path=str(tmp_path / "engram.log"))
    assert logging.getLogger().level == logging.DEBUG


def test_console_handler_level(tmp_path):
    console, _ = setup(console_level="INFO", log_path=str(tmp_path / "engram.log"))
    assert console.level == logging.INFO


def test_file_handler_level(tmp_path):
    _, file = setup(file_level="WARNING", log_path=str(tmp_path / "engram.log"))
    assert file.level == logging.WARNING


def test_log_file_created(tmp_path):
    log_path = tmp_path / "engram.log"
    setup(log_path=str(log_path))
    assert log_path.exists()


def test_log_directory_created(tmp_path):
    log_path = tmp_path / "subdir" / "engram.log"
    setup(log_path=str(log_path))
    assert log_path.parent.exists()


def test_two_handlers_attached(tmp_path):
    handlers = setup(log_path=str(tmp_path / "engram.log"))
    assert len(handlers) == 2


# ── CLI arg parsing ───────────────────────────────────────────────────────────


def make_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_index = sub.add_parser("index")
    p_index.add_argument("pdfs", nargs="+", type=Path)
    p_index.add_argument("--force", action="store_true")
    store_group = p_index.add_mutually_exclusive_group()
    store_group.add_argument("--link", action="store_true")
    store_group.add_argument("--no-store", action="store_true")

    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("--top", type=int, default=5)
    p_search.add_argument("--min-score", type=float, default=0.0)
    p_search.add_argument("--file", default=None)

    return parser


def test_index_defaults():
    args = make_parser().parse_args(["index", "doc.pdf"])
    assert args.force is False
    assert args.link is False
    assert args.no_store is False


def test_index_force_flag():
    args = make_parser().parse_args(["index", "--force", "doc.pdf"])
    assert args.force is True


def test_index_link_flag():
    args = make_parser().parse_args(["index", "--link", "doc.pdf"])
    assert args.link is True


def test_index_no_store_flag():
    args = make_parser().parse_args(["index", "--no-store", "doc.pdf"])
    assert args.no_store is True


def test_index_link_and_no_store_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        make_parser().parse_args(["index", "--link", "--no-store", "doc.pdf"])


def test_search_defaults():
    args = make_parser().parse_args(["search", "my query"])
    assert args.top == 5
    assert args.min_score == 0.0
    assert args.file is None


def test_search_custom_top():
    args = make_parser().parse_args(["search", "query", "--top", "10"])
    assert args.top == 10


def test_search_min_score():
    args = make_parser().parse_args(["search", "query", "--min-score", "0.4"])
    assert args.min_score == pytest.approx(0.4)


def test_search_file_filter():
    args = make_parser().parse_args(["search", "query", "--file", "doc.pdf"])
    assert args.file == "doc.pdf"


# ── default_project ───────────────────────────────────────────────────────────


def test_default_project_uses_parent_dir(tmp_path):
    pdf = tmp_path / "myproject" / "doc.pdf"
    pdf.parent.mkdir()
    assert default_project(pdf) == "myproject"


def test_default_project_fallback(tmp_path):
    # Root-level path with no meaningful parent
    pdf = Path("/doc.pdf")
    result = default_project(pdf)
    assert result  # not empty


# ── session ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def patched_state(tmp_path, monkeypatch):
    import engram.storage as storage

    monkeypatch.setattr(storage, "STATE_FILE", tmp_path / "state.toml")
    yield tmp_path / "state.toml"


def test_read_session_empty(patched_state):
    assert read_session() == []


def test_write_and_read_session(patched_state):
    write_session(["proj-a", "proj-b"])
    assert read_session() == ["proj-a", "proj-b"]


def test_clear_session(patched_state):
    write_session(["proj-a"])
    clear_session()
    assert read_session() == []


def test_session_expiry(patched_state, monkeypatch):
    from datetime import datetime, timedelta

    write_session(["proj-a"])

    # Write a state file with an old timestamp directly
    old_time = (datetime.now() - timedelta(hours=9)).isoformat()
    patched_state.write_text(
        f'[session]\nactive_projects = ["proj-a"]\nactivated_at = "{old_time}"\n'
    )

    assert read_session() == []
    assert not patched_state.exists()  # expired file removed


# ── readers ───────────────────────────────────────────────────────────────────


def test_supported_extensions_includes_common_types():
    for ext in [".pdf", ".txt", ".md", ".rst", ".html", ".docx", ".pptx", ".epub"]:
        assert ext in SUPPORTED_EXTENSIONS


def test_read_text_single_section(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Hello world\nThis is a test.")
    sections = read_text(f)
    assert len(sections) == 1
    assert sections[0].phys_page == 1
    assert sections[0].total == 1
    assert "Hello world" in sections[0].text


def test_read_markdown_splits_by_headings(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Introduction\nSome intro text.\n\n## Methods\nSome methods.")
    sections = read_markdown(f)
    assert len(sections) == 2
    assert sections[0].page_label == "Introduction"
    assert sections[1].page_label == "Methods"


def test_read_markdown_no_headings_fallback(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("Just plain text with no headings.")
    sections = read_markdown(f)
    assert len(sections) == 1


def test_read_markdown_section_numbering(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# A\ntext\n# B\ntext\n# C\ntext")
    sections = read_markdown(f)
    assert [s.phys_page for s in sections] == [1, 2, 3]
    assert all(s.total == 3 for s in sections)


def test_read_rst_splits_by_sections(tmp_path):
    f = tmp_path / "doc.rst"
    f.write_text("Introduction\n============\nSome intro.\n\nMethods\n-------\nSome methods.")
    sections = read_rst(f)
    assert len(sections) >= 1
    assert any("Introduction" in s.page_label for s in sections)


def test_read_markdown_label_truncated(tmp_path):
    f = tmp_path / "doc.md"
    long_heading = "# " + "A" * 100
    f.write_text(long_heading + "\ntext")
    sections = read_markdown(f)
    assert len(sections[0].page_label) <= 60


# ── parse_page_range ──────────────────────────────────────────────────────────


def test_parse_page_range_single():
    assert parse_page_range("42") == (42, 42)


def test_parse_page_range_range():
    assert parse_page_range("5-10") == (5, 10)


def test_parse_page_range_degenerate():
    assert parse_page_range("7-7") == (7, 7)


def test_parse_page_range_reversed():
    with pytest.raises(ValueError, match="less than or equal"):
        parse_page_range("10-5")


def test_parse_page_range_invalid_string():
    with pytest.raises(ValueError):
        parse_page_range("abc")


def test_parse_page_range_invalid_range_string():
    with pytest.raises(ValueError):
        parse_page_range("a-b")


# ── _format_missing_pages ─────────────────────────────────────────────────────


def test_format_missing_pages_empty():
    assert _format_missing_pages([]) == ""


def test_format_missing_pages_single():
    assert _format_missing_pages([5]) == "5"


def test_format_missing_pages_consecutive():
    assert _format_missing_pages([5, 6, 7]) == "5-7"


def test_format_missing_pages_non_consecutive():
    assert _format_missing_pages([5, 7, 9]) == "5, 7, 9"


def test_format_missing_pages_mixed():
    assert _format_missing_pages([3, 4, 7, 10, 11, 12]) == "3-4, 7, 10-12"


# ── chunk navigation helpers ──────────────────────────────────────────────────


def _make_col_stub(metas):
    """Return a minimal chromadb collection stub for _get_chunk_sequence."""
    from unittest.mock import MagicMock

    col = MagicMock()
    col.get.return_value = {"metadatas": metas, "documents": []}
    return col


def test_get_chunk_sequence_sorted():
    metas = [
        {"page": 2, "chunk": 0, "page_label": "2"},
        {"page": 1, "chunk": 1, "page_label": "1"},
        {"page": 1, "chunk": 0, "page_label": "1"},
    ]
    seq = _get_chunk_sequence(_make_col_stub(metas), "doc.pdf")
    assert seq == [(1, 0, "1"), (1, 1, "1"), (2, 0, "2")]


def test_get_chunk_sequence_deduplicates():
    metas = [
        {"page": 1, "chunk": 0, "page_label": "1"},
        {"page": 1, "chunk": 0, "page_label": "1"},  # duplicate
    ]
    seq = _get_chunk_sequence(_make_col_stub(metas), "doc.pdf")
    assert len(seq) == 1


def test_find_seq_index_found():
    seq = [(1, 0, "1"), (1, 1, "1"), (2, 0, "2")]
    assert _find_seq_index(seq, 2, 0) == 2


def test_find_seq_index_not_found():
    seq = [(1, 0, "1"), (2, 0, "2")]
    assert _find_seq_index(seq, 5, 0) is None


def test_find_seq_index_first():
    seq = [(3, 1, "3"), (4, 0, "4")]
    assert _find_seq_index(seq, 3, 1) == 0


# ── _stale_warning ────────────────────────────────────────────────────────────


def test_stale_warning_missing_file(tmp_path):
    nonexistent = str(tmp_path / "gone.pdf")
    warning = _stale_warning(nonexistent, "2026-01-01T00:00:00", 1234.0)
    assert warning is not None
    assert "no longer exists" in warning
    assert "gone.pdf" in warning


def test_stale_warning_up_to_date(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"data")
    mtime = f.stat().st_mtime
    assert _stale_warning(str(f), "2026-01-01T00:00:00", mtime) is None


def test_stale_warning_changed_file(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"data")
    old_mtime = f.stat().st_mtime - 100  # simulate older stored mtime
    warning = _stale_warning(str(f), "2026-01-01T00:00:00", old_mtime)
    assert warning is not None
    assert "changed since last indexed" in warning
    assert "2026-01-01" in warning


def test_stale_warning_no_mtime_stored(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"data")
    # old index entry: no source_mtime stored
    assert _stale_warning(str(f), None, None) is None


# ── expand_paths ──────────────────────────────────────────────────────────────


def test_expand_paths_passthrough_files(tmp_path):
    f = tmp_path / "doc.pdf"
    f.touch()
    result = expand_paths([f])
    assert result == [f]


def test_expand_paths_expands_directory(tmp_path):
    (tmp_path / "a.pdf").touch()
    (tmp_path / "b.txt").touch()
    (tmp_path / "c.jpg").touch()  # unsupported — included by glob but filtered later
    result = expand_paths([tmp_path])
    names = {p.name for p in result}
    assert "a.pdf" in names
    assert "b.txt" in names
    assert "c.jpg" not in names  # not in SUPPORTED_EXTENSIONS so not returned


def test_expand_paths_mixed(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    (d / "nested.md").touch()
    single = tmp_path / "top.txt"
    single.touch()
    result = expand_paths([single, d])
    names = {p.name for p in result}
    assert "top.txt" in names
    assert "nested.md" in names
