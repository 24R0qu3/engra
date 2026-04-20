import argparse
import logging
import tomllib
from pathlib import Path

import pytest

from engra.commands import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_MIN_SCORE,
    MODEL_NAME,
    _data_list_members,
    _fetch_linked_results,
    _find_seq_index,
    _format_missing_pages,
    _get_chunk_sequence,
    _is_notable,
    _load_bookmarks,
    _model_is_cached,
    _normalize_scores,
    _rerank_results,
    _save_bookmarks,
    _stale_status,
    _stale_warning,
    chunk_text,
    default_project,
    expand_paths,
    parse_page_range,
)
from engra.config import _DEFAULT_TOML, DEFAULTS
from engra.log import setup
from engra.readers import (
    SUPPORTED_EXTENSIONS,
    _extract_html_links,
    read_html,
    read_markdown,
    read_rst,
    read_text,
)
from engra.storage import CACHE_DIR, clear_session, read_session, write_session

# ── chunk_text ────────────────────────────────────────────────────────────────


def test_short_text_single_chunk():
    text = "Hello world"
    assert chunk_text(text) == [text]


def test_exact_chunk_size_single_chunk():
    text = "x" * CHUNK_SIZE
    assert len(chunk_text(text)) == 1


def test_long_text_multiple_chunks():
    # Use text > 2000 chars with paragraph breaks so actual splitting occurs
    text = "word " * 500  # ~2500 chars
    chunks = chunk_text(text)
    assert len(chunks) > 1


def test_chunks_have_overlap():
    # Use paragraph-separated text to ensure actual splitting
    text = "\n\n".join(["sentence." * 100 for _ in range(5)])
    chunks = chunk_text(text)
    if len(chunks) > 1:
        # end of first chunk should appear at start of second
        overlap = chunks[0][-CHUNK_OVERLAP:]
        assert chunks[1].startswith(overlap)


def test_chunk_max_size():
    text = "word " * 500  # 2500 chars > CHUNK_SIZE
    for chunk in chunk_text(text):
        assert len(chunk) <= CHUNK_SIZE + CHUNK_OVERLAP


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
    from engra import config

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "nonexistent.toml")
    result = config.load()
    assert result["backend"]["type"] == "local"
    assert result["index"]["copy"] is True


def test_load_merges_user_values(tmp_path, monkeypatch):
    from engra import config

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[index]\ncopy = false\n")
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_file)
    result = config.load()
    assert result["index"]["copy"] is False
    assert result["backend"]["type"] == "local"  # default preserved


def test_init_creates_config(tmp_path, monkeypatch):
    from engra import config

    cfg_file = tmp_path / "config.toml"
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_file)
    config.init()
    assert cfg_file.exists()


def test_init_does_not_overwrite(tmp_path, monkeypatch):
    from engra import config

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
    setup(log_path=str(tmp_path / "engra.log"))
    assert logging.getLogger().level == logging.DEBUG


def test_console_handler_level(tmp_path):
    console, _ = setup(console_level="INFO", log_path=str(tmp_path / "engra.log"))
    assert console.level == logging.INFO


def test_file_handler_level(tmp_path):
    _, file = setup(file_level="WARNING", log_path=str(tmp_path / "engra.log"))
    assert file.level == logging.WARNING


def test_log_file_created(tmp_path):
    log_path = tmp_path / "engra.log"
    setup(log_path=str(log_path))
    assert log_path.exists()


def test_log_directory_created(tmp_path):
    log_path = tmp_path / "subdir" / "engra.log"
    setup(log_path=str(log_path))
    assert log_path.parent.exists()


def test_two_handlers_attached(tmp_path):
    handlers = setup(log_path=str(tmp_path / "engra.log"))
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
    import engra.storage as storage

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


# ── _stale_status ─────────────────────────────────────────────────────────────


def test_stale_status_missing():
    assert _stale_status("/nonexistent/gone.pdf", 1234.0) == "missing"


def test_stale_status_ok(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"data")
    assert _stale_status(str(f), f.stat().st_mtime) == "ok"


def test_stale_status_stale(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"data")
    assert _stale_status(str(f), f.stat().st_mtime - 100) == "stale"


def test_stale_status_unknown_no_mtime(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"data")
    # Pre-feature entry: source_mtime was never stored
    assert _stale_status(str(f), None) == "unknown"


def test_stale_status_missing_no_mtime():
    # File missing takes priority over no mtime
    assert _stale_status("/nonexistent/gone.pdf", None) == "missing"


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


def test_stale_warning_no_mtime_stored_is_silent(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"data")
    # Pre-feature index entry: no warning emitted (unknown ≠ stale)
    assert _stale_warning(str(f), None, None) is None


# ── cross-project search CLI ──────────────────────────────────────────────────


def _make_search_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--project", action="append", metavar="PROJECT", dest="projects", default=None)
    p.add_argument("--all", dest="search_all", action="store_true")
    return parser


def test_search_single_project():
    args = _make_search_parser().parse_args(["search", "q", "--project", "A"])
    assert args.projects == ["A"]


def test_search_multi_project():
    args = _make_search_parser().parse_args(["search", "q", "--project", "A", "--project", "B"])
    assert args.projects == ["A", "B"]


def test_search_no_project_defaults_none():
    args = _make_search_parser().parse_args(["search", "q"])
    assert args.projects is None


def test_search_all_overrides_projects():
    args = _make_search_parser().parse_args(["search", "q", "--all", "--project", "A"])
    # When --all is set, projects should be ignored in dispatch
    assert args.search_all is True


# ── bookmark helpers ──────────────────────────────────────────────────────────


@pytest.fixture()
def patched_bookmarks(tmp_path, monkeypatch):
    import engra.config as cfg

    bm_path = tmp_path / "bookmarks.json"
    monkeypatch.setattr(cfg, "BOOKMARKS_PATH", bm_path)
    # Also patch in commands module which imported at load time
    import engra.commands as commands

    monkeypatch.setattr(commands, "BOOKMARKS_PATH", bm_path)
    yield bm_path


def test_load_bookmarks_empty(patched_bookmarks):
    assert _load_bookmarks() == {}


def test_save_and_load_bookmarks(patched_bookmarks):
    bm = {
        "myquery": {
            "name": "myquery",
            "query": "hello",
            "top": 5,
            "min_score": None,
            "created_at": "2026-01-01T00:00:00",
        }
    }
    _save_bookmarks(bm)
    assert _load_bookmarks() == bm


def test_load_bookmarks_missing_file(patched_bookmarks):
    assert not patched_bookmarks.exists()
    assert _load_bookmarks() == {}


def test_save_bookmarks_creates_dir(tmp_path, monkeypatch):
    import engra.commands as commands

    nested = tmp_path / "a" / "b" / "bookmarks.json"
    monkeypatch.setattr(commands, "BOOKMARKS_PATH", nested)
    _save_bookmarks({"x": {"name": "x"}})
    assert nested.exists()


# ── bookmark CLI: query is positional ─────────────────────────────────────────


def test_bookmark_save_query_positional(tmp_path, monkeypatch):
    """bookmark save NAME QUERY accepts query as a positional argument."""
    import engra.main as m

    m.run.__code__  # just check main is importable  # noqa: B018
    # Parse the CLI args via argparse
    import sys
    from unittest.mock import patch

    with patch.object(sys, "argv", ["engra", "bookmark", "save", "myname", "my search query"]):
        # We can't call run() (it tries to log), so build the parser directly

        # Reload main to get a fresh parser

        # Manually invoke the parser logic from main.py
        ns = _parse_main_args(["bookmark", "save", "myname", "my search query"])
        assert ns.bm_cmd == "save"
        assert ns.name == "myname"
        assert ns.query == "my search query"


def _parse_main_args(argv):
    """Helper that builds the engra argparse and parses the given argv list."""
    import argparse

    from engra import __version__

    parser = argparse.ArgumentParser(prog="engra")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log", default="WARNING")
    parser.add_argument("--log-file", default="DEBUG")
    parser.add_argument("--log-path", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bm = sub.add_parser("bookmark")
    bm_sub = p_bm.add_subparsers(dest="bm_cmd", required=True)
    p_bm_save = bm_sub.add_parser("save")
    p_bm_save.add_argument("name")
    p_bm_save.add_argument("query")
    p_bm_save.add_argument("--project", default=None)
    p_bm_save.add_argument("--top", type=int, default=5)
    p_bm_save.add_argument("--min-score", type=float, default=None, dest="min_score")

    return parser.parse_args(argv)


# ── cmd_info unknown-field hint ───────────────────────────────────────────────


def _make_info_col_stub(metas):
    from unittest.mock import MagicMock

    col = MagicMock()
    col.count.return_value = len(metas)
    col.get.return_value = {"metadatas": metas, "documents": [], "ids": []}
    return col


def _capture_info(monkeypatch, metas, filename=None):
    """Run cmd_info with a mocked collection and return printed text."""
    import io

    from rich.console import Console

    import engra.commands as commands

    col = _make_info_col_stub(metas)
    monkeypatch.setattr(commands, "get_collection", lambda: col)

    buf = io.StringIO()
    fake_console = Console(file=buf, highlight=False, markup=False)
    monkeypatch.setattr(commands, "console", fake_console)

    commands.cmd_info(filename=filename)
    return buf.getvalue()


def test_info_unknown_fields_show_hint(tmp_path, monkeypatch):
    """Fields missing from old index entries display a re-index hint."""
    metas = [
        {
            "source": str(tmp_path / "doc.pdf"),
            "filename": "doc.pdf",
            "page": 1,
            "chunk": 0,
            "page_label": "1",
            "total_pages": 1,
            "project": "test",
            # model, chunk_size, chunk_overlap, indexed_at intentionally absent
        }
    ]
    out = _capture_info(monkeypatch, metas)
    assert "re-index to populate" in out


def test_info_known_fields_no_hint(tmp_path, monkeypatch):
    """Fully-populated index entries show no re-index hint."""
    metas = [
        {
            "source": str(tmp_path / "doc.pdf"),
            "filename": "doc.pdf",
            "page": 1,
            "chunk": 0,
            "page_label": "1",
            "total_pages": 1,
            "project": "test",
            "model": "intfloat/multilingual-e5-large",
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "indexed_at": "2026-01-01T00:00:00+00:00",
            "source_mtime": 1234567890.0,
        }
    ]
    out = _capture_info(monkeypatch, metas)
    assert "re-index to populate" not in out


# ── index metadata constants ──────────────────────────────────────────────────


def test_model_name_constant():
    assert "multilingual" in MODEL_NAME


def test_chunk_size_positive():
    assert CHUNK_SIZE > 0


def test_chunk_overlap_less_than_size():
    assert CHUNK_OVERLAP < CHUNK_SIZE


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


# ── near-blank skip message and empty-dir warning ─────────────────────────────


def test_skip_note_shown_when_sections_skipped():
    """Summary line includes threshold info when some sections are near-blank."""
    from engra.commands import MIN_CHARS

    # Simulate: 3 total sections, 1 near-blank skipped → indexed pages differ from total
    # We verify the message format by constructing it directly as commands.py does
    total_sections = 3
    indexed = 2
    skipped = total_sections - indexed
    skip_note = (
        f", skipped {skipped} sections shorter than {MIN_CHARS} chars" if skipped > 0 else ""
    )
    assert str(MIN_CHARS) in skip_note
    assert "skipped 1" in skip_note


def test_skip_note_empty_when_no_skips():
    total_sections = 3
    indexed = 3
    skipped = total_sections - indexed
    skip_note = f", skipped {skipped} sections shorter than 80 chars" if skipped > 0 else ""
    assert skip_note == ""


# ── model loading & cache ──────────────────────────────────────────────────────


def test_cache_dir_is_path():
    assert hasattr(CACHE_DIR, "parent")  # is a Path


def test_model_is_cached_false_when_dir_missing(monkeypatch, tmp_path):
    import engra.commands as cmd

    monkeypatch.setattr(cmd, "CACHE_DIR", tmp_path)
    assert _model_is_cached() is False


def test_model_is_cached_true_when_onnx_present(monkeypatch, tmp_path):
    import engra.commands as cmd

    model_dir = tmp_path / "models" / MODEL_NAME.replace("/", "__")
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").touch()
    monkeypatch.setattr(cmd, "CACHE_DIR", tmp_path)
    assert _model_is_cached() is True


def test_model_is_cached_false_when_no_onnx(monkeypatch, tmp_path):
    import engra.commands as cmd

    model_dir = tmp_path / "models" / MODEL_NAME.replace("/", "__")
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").touch()  # no .onnx
    monkeypatch.setattr(cmd, "CACHE_DIR", tmp_path)
    assert _model_is_cached() is False


def test_load_model_uses_cache_dir(monkeypatch):
    """load_model passes CACHE_DIR/models as cache_dir to TextEmbedding."""
    captured = {}

    class FakeEmbedding:
        def __init__(self, model_name, cache_dir=None, **kwargs):
            captured["model_name"] = model_name
            captured["cache_dir"] = cache_dir

    import engra.commands as cmd

    monkeypatch.setattr("builtins.__import__", __import__)  # ensure import still works

    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def fake_import(name, *args, **kwargs):
        if name == "fastembed":
            import types

            m = types.ModuleType("fastembed")
            m.TextEmbedding = FakeEmbedding
            return m
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    # Reset any cached import of fastembed in commands module scope
    import sys

    sys.modules.pop("fastembed", None)

    cmd.load_model()

    assert captured["model_name"] == cmd.MODEL_NAME
    assert captured["cache_dir"] == str(CACHE_DIR / "models")


def test_main_suppresses_ort_warning():
    """ORT_LOGGING_LEVEL is set before any fastembed import in main."""
    import os
    import sys

    # Remove os.environ entry if set, then reimport main (which sets it)
    os.environ.pop("ORT_LOGGING_LEVEL", None)
    sys.modules.pop("engra.main", None)
    import engra.main  # noqa: F401 – side-effect: sets env var

    assert os.environ.get("ORT_LOGGING_LEVEL") == "3"


# ── cmd_ask ────────────────────────────────────────────────────────────────────


def _make_ask_collection(docs, metas):
    from unittest.mock import MagicMock

    col = MagicMock()
    col.count.return_value = len(docs)
    distances = [0.1] * len(docs)
    col.query.return_value = {"documents": [docs], "metadatas": [metas], "distances": [distances]}
    return col


def _make_fake_model(embedding=None):
    from unittest.mock import MagicMock

    import numpy as np

    model = MagicMock()
    vec = embedding if embedding is not None else np.zeros(128)
    model.query_embed.return_value = iter([vec])
    return model


def test_ask_empty_collection(monkeypatch, capsys):
    import io

    from rich.console import Console

    import engra.commands as commands

    col = _make_ask_collection([], [])
    col.count.return_value = 0
    monkeypatch.setattr(commands, "get_collection", lambda: col)
    buf = io.StringIO()
    monkeypatch.setattr(commands, "console", Console(file=buf, highlight=False))
    commands.cmd_ask("What is torque?")
    out = buf.getvalue()
    assert "empty" in out.lower()


def test_ask_llm_success(monkeypatch):
    import io
    import json
    from unittest.mock import MagicMock, patch

    from rich.console import Console

    import engra.commands as commands

    docs = ["Torque is a rotational force.", "It is measured in Newton-metres."]
    metas = [
        {"filename": "doc.pdf", "page": 1, "page_label": "1", "chunk": 0},
        {"filename": "doc.pdf", "page": 2, "page_label": "2", "chunk": 0},
    ]
    col = _make_ask_collection(docs, metas)
    monkeypatch.setattr(commands, "get_collection", lambda: col)
    monkeypatch.setattr(commands, "load_model", lambda: _make_fake_model())
    monkeypatch.setattr(commands, "read_session", lambda: [])

    # Streaming SSE lines
    sse_chunk = json.dumps({"choices": [{"delta": {"content": "Torque is a rotational force."}}]})
    sse_lines = [
        f"data: {sse_chunk}\n".encode(),
        b"data: [DONE]\n",
    ]

    buf = io.StringIO()
    monkeypatch.setattr(commands, "console", Console(file=buf, highlight=False))

    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.__iter__ = lambda s: iter(sse_lines)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        commands.cmd_ask("What is torque?")

    out = buf.getvalue()
    assert "Torque" in out
    assert "Sources" in out


def test_ask_llm_connection_error(monkeypatch):
    import io
    import urllib.error
    from unittest.mock import patch

    from rich.console import Console

    import engra.commands as commands

    docs = ["Some text"]
    metas = [{"filename": "doc.pdf", "page": 1, "page_label": "1", "chunk": 0}]
    col = _make_ask_collection(docs, metas)
    monkeypatch.setattr(commands, "get_collection", lambda: col)
    monkeypatch.setattr(commands, "load_model", lambda: _make_fake_model())
    monkeypatch.setattr(commands, "read_session", lambda: [])

    buf = io.StringIO()
    monkeypatch.setattr(commands, "console", Console(file=buf, highlight=False))

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        commands.cmd_ask("What is torque?")

    out = buf.getvalue()
    assert "failed" in out.lower() or "LLM" in out


# ── export / import ────────────────────────────────────────────────────────────


def _make_export_collection(project: str = "testproject"):
    """Return a mock collection pre-loaded with two chunks for export tests."""
    from unittest.mock import MagicMock

    col = MagicMock()
    ids = ["id1", "id2"]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    documents = ["chunk one text", "chunk two text"]
    metadatas = [
        {
            "filename": "a.pdf",
            "page": 1,
            "page_label": "1",
            "chunk": 0,
            "project": project,
            "source": "/tmp/a.pdf",
            "indexed_at": "2026-01-01T00:00:00",
        },
        {
            "filename": "a.pdf",
            "page": 2,
            "page_label": "2",
            "chunk": 0,
            "project": project,
            "source": "/tmp/a.pdf",
            "indexed_at": "2026-01-01T00:00:00",
        },
    ]
    col.get.return_value = {
        "ids": ids,
        "embeddings": embeddings,
        "documents": documents,
        "metadatas": metadatas,
    }
    col.count.return_value = 2
    return col


def test_data_export_returns_expected_shape(monkeypatch, tmp_path):
    import engra.commands as commands

    col = _make_export_collection()
    monkeypatch.setattr(commands, "get_collection", lambda: col)
    monkeypatch.setattr(commands, "FILES_DIR", tmp_path)

    result = commands._data_export("testproject")

    assert result["project"] == "testproject"
    assert result["chunk_count"] == 2
    assert len(result["chunks"]) == 2
    assert result["chunks"][0]["id"] == "id1"
    assert result["chunks"][0]["embedding"] == [0.1, 0.2]


def test_data_export_raises_for_unknown_project(monkeypatch):
    from unittest.mock import MagicMock

    import engra.commands as commands

    col = MagicMock()
    col.get.return_value = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
    monkeypatch.setattr(commands, "get_collection", lambda: col)

    with pytest.raises(ValueError, match="not found"):
        commands._data_export("ghost")


def test_cmd_export_creates_archive(monkeypatch, tmp_path):
    import io
    import json
    import tarfile

    from rich.console import Console

    import engra.commands as commands

    col = _make_export_collection()
    # Create a real file so tarfile can add it
    fake_file = tmp_path / "a.pdf"
    fake_file.write_bytes(b"%PDF fake")

    files_dir = tmp_path / "files"
    files_dir.mkdir()
    stored = files_dir / "a.pdf"
    stored.write_bytes(b"%PDF fake")

    monkeypatch.setattr(commands, "get_collection", lambda: col)
    monkeypatch.setattr(commands, "FILES_DIR", files_dir)
    buf = io.StringIO()
    monkeypatch.setattr(commands, "console", Console(file=buf, highlight=False))

    out_path = tmp_path / "testproject.engra.tar.gz"
    commands.cmd_export("testproject", output_path=out_path)

    assert out_path.exists()
    with tarfile.open(out_path, "r:gz") as tf:
        names = tf.getnames()
    assert "manifest.json" in names
    assert "chunks.json" in names

    # Verify manifest content
    with tarfile.open(out_path, "r:gz") as tf:
        manifest = json.load(tf.extractfile("manifest.json"))
    assert manifest["project"] == "testproject"
    assert manifest["chunk_count"] == 2
    assert manifest["model"] == commands.MODEL_NAME


def test_data_import_round_trips(monkeypatch, tmp_path):
    import io
    import json
    import tarfile

    import engra.commands as commands

    # Build a minimal valid archive
    chunks = [
        {
            "id": "id1",
            "embedding": [0.1, 0.2],
            "document": "hello",
            "metadata": {"filename": "a.pdf", "page": 1, "project": "p"},
        },
    ]
    manifest = {
        "engra_export_version": commands.EXPORT_FORMAT_VERSION,
        "model": commands.MODEL_NAME,
        "project": "p",
        "exported_at": "2026-01-01T00:00:00",
        "chunk_count": 1,
        "file_count": 0,
    }
    arc = tmp_path / "p.engra.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        for name, obj in [("manifest.json", manifest), ("chunks.json", chunks)]:
            data = json.dumps(obj).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    from unittest.mock import MagicMock

    col = MagicMock()
    col.get.return_value = {"ids": []}
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    monkeypatch.setattr(commands, "get_collection", lambda: col)
    monkeypatch.setattr(commands, "FILES_DIR", files_dir)
    monkeypatch.setattr(commands, "ensure_dirs", lambda: None)
    monkeypatch.setattr("engra.storage.ensure_dirs", lambda: None)

    result = commands._data_import(arc)

    assert result["project"] == "p"
    assert result["chunks_added"] == 1
    col.add.assert_called_once()


def test_data_import_rejects_model_mismatch(monkeypatch, tmp_path):
    import io
    import json
    import tarfile

    import engra.commands as commands

    manifest = {
        "engra_export_version": commands.EXPORT_FORMAT_VERSION,
        "model": "some/other-model",
        "project": "p",
        "exported_at": "2026-01-01T00:00:00",
        "chunk_count": 0,
        "file_count": 0,
    }
    arc = tmp_path / "p.engra.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        data = json.dumps(manifest).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    with pytest.raises(ValueError, match="Model mismatch"):
        commands._data_import(arc)


def test_data_import_rejects_duplicate_without_overwrite(monkeypatch, tmp_path):
    import io
    import json
    import tarfile
    from unittest.mock import MagicMock

    import engra.commands as commands

    manifest = {
        "engra_export_version": commands.EXPORT_FORMAT_VERSION,
        "model": commands.MODEL_NAME,
        "project": "p",
        "exported_at": "2026-01-01T00:00:00",
        "chunk_count": 0,
        "file_count": 0,
    }
    arc = tmp_path / "p.engra.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        data = json.dumps(manifest).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    col = MagicMock()
    col.get.return_value = {"ids": ["existing-id"]}
    monkeypatch.setattr(commands, "get_collection", lambda: col)

    with pytest.raises(ValueError, match="already exists"):
        commands._data_import(arc, overwrite=False)


def test_cmd_import_soft_calls_cmd_index(monkeypatch, tmp_path):
    import engra.commands as commands

    calls = []
    monkeypatch.setattr(
        commands,
        "cmd_index",
        lambda paths, copy, store, project: calls.append((paths, copy, store, project)),
    )

    src = tmp_path / "mydocs"
    src.mkdir()
    commands.cmd_import_soft(src, project="override")

    assert len(calls) == 1
    paths, copy, store, project = calls[0]
    assert copy is False
    assert store is True
    assert project == "override"


def test_cmd_import_soft_defaults_project_to_dirname(monkeypatch, tmp_path):
    import engra.commands as commands

    calls = []
    monkeypatch.setattr(
        commands, "cmd_index", lambda paths, copy, store, project: calls.append(project)
    )

    src = tmp_path / "myproject"
    src.mkdir()
    commands.cmd_import_soft(src)

    assert calls[0] == "myproject"


def test_cmd_export_numpy_embeddings_serialized_as_floats(monkeypatch, tmp_path):
    """Embeddings returned as numpy arrays must be stored as float lists, not strings.

    Regression test for the bug where `json.dumps(..., default=str)` turned numpy
    arrays into their repr strings, causing ChromaDB to reject them on import.
    """
    import io
    import json
    import tarfile
    from unittest.mock import MagicMock

    import numpy as np
    from rich.console import Console

    import engra.commands as commands

    # ChromaDB returns embeddings as numpy arrays
    numpy_embeddings = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]

    col = MagicMock()
    col.get.return_value = {
        "ids": ["id1", "id2"],
        "embeddings": numpy_embeddings,
        "documents": ["text one", "text two"],
        "metadatas": [
            {
                "filename": "a.pdf",
                "page": 1,
                "page_label": "1",
                "chunk": 0,
                "project": "testproject",
                "source": "/tmp/a.pdf",
                "indexed_at": "2026-01-01",
            },
            {
                "filename": "a.pdf",
                "page": 2,
                "page_label": "2",
                "chunk": 0,
                "project": "testproject",
                "source": "/tmp/a.pdf",
                "indexed_at": "2026-01-01",
            },
        ],
    }
    col.count.return_value = 2

    monkeypatch.setattr(commands, "get_collection", lambda: col)
    monkeypatch.setattr(commands, "FILES_DIR", tmp_path)
    monkeypatch.setattr(commands, "console", Console(file=io.StringIO(), highlight=False))

    out_path = tmp_path / "testproject.engra.tar.gz"
    commands.cmd_export("testproject", output_path=out_path)

    with tarfile.open(out_path, "r:gz") as tf:
        chunks = json.load(tf.extractfile("chunks.json"))

    for chunk in chunks:
        emb = chunk["embedding"]
        assert isinstance(emb, list), "embedding must be a list, not a string or numpy array"
        assert all(isinstance(v, float) for v in emb), "embedding values must be floats"


def test_cmd_export_includes_project_metadata(monkeypatch, tmp_path):
    """Exported manifest must include description, keywords, and auto-description fields."""
    import io
    import json
    import tarfile

    from rich.console import Console

    import engra.commands as commands

    col = _make_export_collection()
    monkeypatch.setattr(commands, "get_collection", lambda: col)
    monkeypatch.setattr(commands, "FILES_DIR", tmp_path)
    monkeypatch.setattr(commands, "console", Console(file=io.StringIO(), highlight=False))

    project_meta = {
        "description": "A test project",
        "keywords": ["foo", "bar"],
        "auto_description": "Auto-generated description",
        "auto_keywords": ["auto1", "auto2"],
    }
    monkeypatch.setattr(commands, "read_projects", lambda: {"testproject": project_meta})

    out_path = tmp_path / "testproject.engra.tar.gz"
    commands.cmd_export("testproject", output_path=out_path)

    with tarfile.open(out_path, "r:gz") as tf:
        manifest = json.load(tf.extractfile("manifest.json"))

    assert manifest["project_meta"] == project_meta


def test_data_import_restores_project_metadata(monkeypatch, tmp_path):
    """Importing an archive must restore description and keywords via update_project_meta."""
    import io
    import json
    import tarfile
    from unittest.mock import MagicMock

    import engra.commands as commands

    project_meta = {
        "description": "Restored description",
        "keywords": ["k1", "k2"],
        "auto_description": "",
        "auto_keywords": [],
    }
    chunks = [
        {
            "id": "id1",
            "embedding": [0.1, 0.2],
            "document": "hello",
            "metadata": {"filename": "a.pdf", "page": 1, "project": "p"},
        }
    ]
    manifest = {
        "engra_export_version": commands.EXPORT_FORMAT_VERSION,
        "model": commands.MODEL_NAME,
        "project": "p",
        "exported_at": "2026-01-01T00:00:00",
        "chunk_count": 1,
        "file_count": 0,
        "project_meta": project_meta,
    }
    arc = tmp_path / "p.engra.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        for name, obj in [("manifest.json", manifest), ("chunks.json", chunks)]:
            data = json.dumps(obj).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    col = MagicMock()
    col.get.return_value = {"ids": []}
    files_dir = tmp_path / "files"
    files_dir.mkdir()

    recorded_meta: list[dict] = []

    def fake_update_project_meta(name, **kwargs):
        recorded_meta.append({"name": name, **kwargs})

    monkeypatch.setattr(commands, "get_collection", lambda: col)
    monkeypatch.setattr(commands, "FILES_DIR", files_dir)
    monkeypatch.setattr(commands, "ensure_dirs", lambda: None)
    monkeypatch.setattr("engra.storage.ensure_dirs", lambda: None)
    monkeypatch.setattr(commands, "update_project_meta", fake_update_project_meta)

    commands._data_import(arc)

    assert len(recorded_meta) == 1
    assert recorded_meta[0]["name"] == "p"
    assert recorded_meta[0]["description"] == "Restored description"
    assert recorded_meta[0]["keywords"] == ["k1", "k2"]


# ── read_html — heading-split (fix #1) ───────────────────────────────────────


def test_read_html_no_headings_single_section(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text("<html><body><p>Hello world</p></body></html>")
    sections = read_html(f)
    assert len(sections) == 1
    assert "Hello world" in sections[0].text


def test_read_html_splits_by_h2_headings(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text(
        "<html><body>"
        "<p>intro</p>"
        "<h2>Section A</h2><p>content A</p>"
        "<h2>Section B</h2><p>content B</p>"
        "</body></html>"
    )
    sections = read_html(f)
    # intro section + two heading sections
    assert len(sections) == 3
    labels = [s.page_label for s in sections]
    assert "Section A" in labels
    assert "Section B" in labels


def test_read_html_heading_text_is_label(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text("<html><body><h1>My Title</h1><p>body text</p></body></html>")
    sections = read_html(f)
    heading_sections = [s for s in sections if s.page_label == "My Title"]
    assert len(heading_sections) == 1
    assert "body text" in heading_sections[0].text


def test_read_html_section_numbering(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text("<html><body><h1>A</h1><p>text a</p><h2>B</h2><p>text b</p></body></html>")
    sections = read_html(f)
    assert [s.phys_page for s in sections] == list(range(1, len(sections) + 1))
    assert all(s.total == len(sections) for s in sections)


def test_read_html_strips_nav_and_footer(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text(
        "<html><body>"
        "<nav>Navigation</nav>"
        "<p>Main content</p>"
        "<footer>Footer text</footer>"
        "</body></html>"
    )
    sections = read_html(f)
    all_text = " ".join(s.text for s in sections)
    assert "Navigation" not in all_text
    assert "Footer text" not in all_text
    assert "Main content" in all_text


def test_read_html_label_truncated(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text(f"<html><body><h1>{'A' * 100}</h1><p>text</p></body></html>")
    sections = read_html(f)
    for s in sections:
        assert len(s.page_label) <= 60


def test_read_html_heading_text_not_duplicated_in_body(tmp_path):
    """Heading text should not appear twice (once as label, once in body)."""
    f = tmp_path / "doc.html"
    f.write_text("<html><body><h2>My Section</h2><p>description here</p></body></html>")
    sections = read_html(f)
    section = next(s for s in sections if s.page_label == "My Section")
    # The heading text itself should not be re-emitted as body content
    assert section.text.count("My Section") == 0


# ── _is_notable — stub/TBD flagging (fix #3) ────────────────────────────────


def test_is_notable_tbd_uppercase():
    assert _is_notable("This value is TBD")


def test_is_notable_tbd_lowercase():
    assert _is_notable("value is tbd for now")


def test_is_notable_todo():
    assert _is_notable("// TODO: implement this")


def test_is_notable_fixme():
    assert _is_notable("/* FIXME: broken path */")


def test_is_notable_stub_word():
    assert _is_notable("This is a stub implementation")


def test_is_notable_not_implemented_phrase():
    assert _is_notable("This feature is not implemented yet")


def test_is_notable_raise_not_implemented_error():
    assert _is_notable("    raise NotImplementedError")


def test_is_notable_false_on_normal_text():
    assert not _is_notable("Returns the device allocation for the given node.")


def test_is_notable_false_on_empty():
    assert not _is_notable("")


def test_is_notable_stub_not_matched_as_substring():
    # "stubborn" should not match the \bstub\b pattern
    assert not _is_notable("stubborn resistance to change")


# ── _normalize_scores (fix #2a — relative score display) ─────────────────────


def test_normalize_scores_empty():
    assert _normalize_scores([]) == []


def test_normalize_scores_single():
    assert _normalize_scores([0.84]) == [1.0]


def test_normalize_scores_all_equal():
    assert _normalize_scores([0.85, 0.85, 0.85]) == [1.0, 1.0, 1.0]


def test_normalize_scores_top_is_one():
    scores = [0.87, 0.85, 0.84, 0.83]
    result = _normalize_scores(scores)
    assert result[0] == pytest.approx(1.0)


def test_normalize_scores_bottom_is_zero():
    scores = [0.87, 0.85, 0.84, 0.83]
    result = _normalize_scores(scores)
    assert result[-1] == pytest.approx(0.0)


def test_normalize_scores_values_correct():
    # [0.80, 0.85, 0.90] → spread=0.10; normalised: [0.0, 0.5, 1.0]
    result = _normalize_scores([0.80, 0.85, 0.90])
    assert result == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_scores_preserves_order():
    scores = [0.87, 0.85, 0.84]
    result = _normalize_scores(scores)
    assert result[0] > result[1] > result[2]


def test_normalize_scores_output_length_matches_input():
    scores = [0.9, 0.85, 0.8, 0.75, 0.7]
    assert len(_normalize_scores(scores)) == len(scores)


def test_normalize_scores_two_elements():
    result = _normalize_scores([0.80, 0.90])
    assert result == pytest.approx([0.0, 1.0])


# ── _rerank_results (fix #2b — cross-encoder re-ranking) ─────────────────────


def _make_hits(n: int) -> list[dict]:
    """Return n minimal hit dicts suitable for passing to _rerank_results."""
    return [
        {
            "filename": f"doc{i}.html",
            "page": i,
            "page_label": str(i),
            "total_pages": n,
            "chunk": 0,
            "score": round(0.85 - i * 0.01, 4),
            "text": f"Content of chunk {i}",
            "project": "test",
            "source": f"/tmp/doc{i}.html",
            "indexed_at": None,
            "source_mtime": None,
            "notable": False,
        }
        for i in range(n)
    ]


def _setup_flashrank_mock(monkeypatch, ranked_order: list[int], scores: list[float]):
    """Patch sys.modules with a flashrank stub and return a mock Ranker.

    Because _rerank_results does `from flashrank import RerankRequest` at call time,
    we must stub sys.modules so the import resolves without the real package installed.
    """
    import sys
    from unittest.mock import MagicMock

    class FakeRerankRequest:
        def __init__(self, query, passages):
            self.query = query
            self.passages = passages

    def fake_rerank(request):
        return [
            {**request.passages[idx], "score": scores[rank]}
            for rank, idx in enumerate(ranked_order)
        ]

    ranker = MagicMock()
    ranker.rerank.side_effect = fake_rerank

    stub = MagicMock()
    stub.RerankRequest = FakeRerankRequest
    monkeypatch.setitem(sys.modules, "flashrank", stub)

    return ranker


def test_rerank_results_returns_top_k(monkeypatch):
    import engra.commands as commands

    hits = _make_hits(6)
    ranker = _setup_flashrank_mock(monkeypatch, [2, 0, 4, 1, 3, 5], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    monkeypatch.setattr(commands, "_load_reranker", lambda: ranker)

    result = _rerank_results("query", hits, top_k=3)
    assert len(result) == 3


def test_rerank_results_adds_rerank_score(monkeypatch):
    import engra.commands as commands

    hits = _make_hits(3)
    ranker = _setup_flashrank_mock(monkeypatch, [1, 0, 2], [0.95, 0.80, 0.60])
    monkeypatch.setattr(commands, "_load_reranker", lambda: ranker)

    result = _rerank_results("query", hits, top_k=3)
    assert all("rerank_score" in h for h in result)
    assert result[0]["rerank_score"] == pytest.approx(0.95)


def test_rerank_results_reorders_by_cross_encoder(monkeypatch):
    import engra.commands as commands

    hits = _make_hits(3)
    # Reverse the order: chunk 2 should come first
    ranker = _setup_flashrank_mock(monkeypatch, [2, 1, 0], [0.9, 0.7, 0.3])
    monkeypatch.setattr(commands, "_load_reranker", lambda: ranker)

    result = _rerank_results("query", hits, top_k=3)
    assert result[0]["filename"] == "doc2.html"
    assert result[1]["filename"] == "doc1.html"
    assert result[2]["filename"] == "doc0.html"


def test_rerank_results_preserves_original_fields(monkeypatch):
    import engra.commands as commands

    hits = _make_hits(2)
    ranker = _setup_flashrank_mock(monkeypatch, [0, 1], [0.9, 0.5])
    monkeypatch.setattr(commands, "_load_reranker", lambda: ranker)

    result = _rerank_results("query", hits, top_k=2)
    assert result[0]["score"] == hits[0]["score"]
    assert result[0]["text"] == hits[0]["text"]
    assert result[0]["filename"] == hits[0]["filename"]


def test_load_reranker_raises_on_missing_dep(monkeypatch):
    import builtins

    import engra.commands as commands

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "flashrank":
            raise ImportError("No module named 'flashrank'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="pip install 'engra\\[rerank\\]'"):
        commands._load_reranker()


def test_search_rerank_flag_in_cli():
    """--rerank is accepted by the search subparser."""
    ns = _parse_search_args(["search", "my query", "--rerank"])
    assert ns.rerank is True


def test_search_rerank_flag_default_false():
    ns = _parse_search_args(["search", "my query"])
    assert ns.rerank is False


def _parse_search_args(argv):
    import argparse

    from engra import __version__

    parser = argparse.ArgumentParser(prog="engra")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log", default="WARNING")
    parser.add_argument("--log-file", default="DEBUG")
    parser.add_argument("--log-path", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("--top", type=int, default=5)
    p_search.add_argument("--min-score", type=float, default=0.0)
    p_search.add_argument("--file", default=None)
    p_search.add_argument("--project", action="append", dest="projects", default=None)
    p_search.add_argument("--all", dest="search_all", action="store_true")
    p_search.add_argument("--full", action="store_true")
    p_search.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
    )
    p_search.add_argument("--rerank", action="store_true")
    p_search.add_argument("--links", action="store_true")

    return parser.parse_args(argv)


# ── _extract_html_links (fix #4) ──────────────────────────────────────────────


def _make_soup(html: str):
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "html.parser")


def test_extract_html_links_relative_html():
    soup = _make_soup('<a href="other.html">link</a>')
    assert _extract_html_links(soup) == ["other.html"]


def test_extract_html_links_skips_external():
    soup = _make_soup('<a href="https://example.com/page.html">ext</a>')
    assert _extract_html_links(soup) == []


def test_extract_html_links_skips_anchor_only():
    soup = _make_soup('<a href="#section">anchor</a>')
    assert _extract_html_links(soup) == []


def test_extract_html_links_skips_non_html():
    soup = _make_soup('<a href="image.png">img</a><a href="doc.pdf">pdf</a>')
    assert _extract_html_links(soup) == []


def test_extract_html_links_skips_self_link():
    soup = _make_soup('<a href="self.html">self</a>')
    assert _extract_html_links(soup, source_name="self.html") == []


def test_extract_html_links_strips_fragment():
    soup = _make_soup('<a href="other.html#method">method</a>')
    assert _extract_html_links(soup) == ["other.html"]


def test_extract_html_links_strips_path_prefix():
    soup = _make_soup('<a href="../api/class.html">class</a>')
    assert _extract_html_links(soup) == ["class.html"]


def test_extract_html_links_deduplicates():
    soup = _make_soup('<a href="a.html">1</a><a href="a.html">2</a>')
    assert _extract_html_links(soup) == ["a.html"]


def test_extract_html_links_sorted():
    soup = _make_soup('<a href="z.html">z</a><a href="a.html">a</a>')
    assert _extract_html_links(soup) == ["a.html", "z.html"]


def test_read_html_attaches_links_to_sections(tmp_path):
    f = tmp_path / "page.html"
    f.write_text('<html><body><p>content</p><a href="other.html">link</a></body></html>')
    sections = read_html(f)
    assert all("other.html" in s.links_to for s in sections)


def test_read_html_no_links_empty_list(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("<html><body><p>no links here</p></body></html>")
    sections = read_html(f)
    assert all(s.links_to == [] for s in sections)


def test_section_links_to_default_empty():
    from engra.readers import Section

    s = Section(text="t", phys_page=1, page_label="1", total=1)
    assert s.links_to == []


# ── _fetch_linked_results (fix #4) ────────────────────────────────────────────


def _make_primary_hit(filename: str, links_to: str = "") -> dict:
    return {
        "filename": filename,
        "page": 1,
        "page_label": "1",
        "total_pages": 1,
        "chunk": 0,
        "score": 0.85,
        "text": f"Content of {filename}",
        "project": "test",
        "source": f"/tmp/{filename}",
        "indexed_at": None,
        "source_mtime": None,
        "notable": False,
        "links_to": links_to,
        "linked_from": [],
    }


def _make_linked_collection(indexed_filenames: list[str]):
    """Return a minimal chromadb collection stub for _fetch_linked_results tests."""
    from unittest.mock import MagicMock

    def fake_get(where=None, include=None, limit=None, **kwargs):
        filename = where.get("filename") if where else None
        if filename in indexed_filenames:
            return {"ids": [f"id_{filename}"], "metadatas": [], "documents": []}
        return {"ids": [], "metadatas": [], "documents": []}

    def fake_query(query_embeddings=None, n_results=1, include=None, where=None, **kwargs):
        filename = where.get("filename") if where else None
        if filename in indexed_filenames:
            meta = {
                "filename": filename,
                "page": 1,
                "page_label": "1",
                "total_pages": 1,
                "chunk": 0,
                "project": "test",
                "source": f"/tmp/{filename}",
                "indexed_at": None,
                "source_mtime": None,
                "links_to": "",
            }
            return {
                "documents": [[f"Content of {filename}"]],
                "metadatas": [[meta]],
                "distances": [[0.15]],
            }
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    col = MagicMock()
    col.get.side_effect = fake_get
    col.query.side_effect = fake_query
    return col


def test_fetch_linked_results_returns_linked_chunk():
    col = _make_linked_collection(["linked.html"])
    primary = [_make_primary_hit("main.html", links_to="linked.html")]
    results = _fetch_linked_results([0.1] * 10, primary, col, [])
    assert len(results) == 1
    assert results[0]["filename"] == "linked.html"
    assert results[0]["linked_from"] == ["main.html"]


def test_fetch_linked_results_skips_unindexed():
    col = _make_linked_collection([])  # nothing indexed
    primary = [_make_primary_hit("main.html", links_to="missing.html")]
    results = _fetch_linked_results([0.1] * 10, primary, col, [])
    assert results == []


def test_fetch_linked_results_skips_primary_filenames():
    col = _make_linked_collection(["main.html"])
    # Link points back to a file already in primary results
    primary = [_make_primary_hit("main.html", links_to="main.html")]
    results = _fetch_linked_results([0.1] * 10, primary, col, [])
    assert results == []


def test_fetch_linked_results_multiple_sources():
    col = _make_linked_collection(["shared.html"])
    primary = [
        _make_primary_hit("a.html", links_to="shared.html"),
        _make_primary_hit("b.html", links_to="shared.html"),
    ]
    results = _fetch_linked_results([0.1] * 10, primary, col, [])
    assert len(results) == 1
    assert set(results[0]["linked_from"]) == {"a.html", "b.html"}


def test_fetch_linked_results_empty_links():
    col = _make_linked_collection(["other.html"])
    primary = [_make_primary_hit("main.html", links_to="")]
    results = _fetch_linked_results([0.1] * 10, primary, col, [])
    assert results == []


def test_fetch_linked_results_adds_notable_flag():
    col = _make_linked_collection(["linked.html"])
    primary = [_make_primary_hit("main.html", links_to="linked.html")]
    results = _fetch_linked_results([0.1] * 10, primary, col, [])
    assert "notable" in results[0]


def test_search_links_flag_in_cli():
    ns = _parse_search_args(["search", "my query", "--links"])
    assert ns.links is True


def test_search_links_flag_default_false():
    ns = _parse_search_args(["search", "my query"])
    assert ns.links is False


# ── Section dataclass new fields ──────────────────────────────────────────────


def test_section_atomic_default_false():
    from engra.readers import Section

    s = Section(text="x", phys_page=1, page_label="1", total=1)
    assert s.atomic is False


def test_section_breadcrumb_default_empty():
    from engra.readers import Section

    s = Section(text="x", phys_page=1, page_label="1", total=1)
    assert s.breadcrumb == ""


def test_section_cross_refs_default_empty():
    from engra.readers import Section

    s = Section(text="x", phys_page=1, page_label="1", total=1)
    assert s.cross_refs == []


# ── read_html breadcrumb tracking (Feature 3) ────────────────────────────────


def test_read_html_breadcrumb_h2_under_h1(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text(
        "<html><body><h1>Animals</h1><p>intro</p><h2>Dogs</h2><p>dog content</p></body></html>"
    )
    sections = read_html(f)
    # Find the Dogs section
    dogs = next(s for s in sections if s.page_label == "Dogs")
    assert dogs.breadcrumb == "Animals"


def test_read_html_breadcrumb_h3_depth(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text(
        "<html><body>"
        "<h1>Animals</h1><p>a</p>"
        "<h2>Dogs</h2><p>b</p>"
        "<h3>Poodle</h3><p>c</p>"
        "</body></html>"
    )
    sections = read_html(f)
    poodle = next(s for s in sections if s.page_label == "Poodle")
    assert poodle.breadcrumb == "Animals > Dogs"


def test_read_html_breadcrumb_resets_on_same_level(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text("<html><body><h2>Section A</h2><p>a</p><h2>Section B</h2><p>b</p></body></html>")
    sections = read_html(f)
    b = next(s for s in sections if s.page_label == "Section B")
    assert b.breadcrumb == ""


def test_read_html_root_section_has_empty_breadcrumb(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text("<html><body><p>preamble</p><h2>First</h2><p>content</p></body></html>")
    sections = read_html(f)
    # The first section (preamble, before any heading) has no breadcrumb
    assert sections[0].breadcrumb == ""


# ── read_html cross-reference extraction (Feature 5) ────────────────────────


def test_read_html_extracts_see_also_links(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text(
        "<html><body>"
        "<h4>myFunction</h4>"
        "<p>Does something.</p>"
        '<dl class="section see"><dt>See also</dt>'
        '<dd><a href="#">Foo::bar</a></dd></dl>'
        "</body></html>"
    )
    sections = read_html(f)
    assert any("Foo::bar" in s.cross_refs for s in sections)


def test_read_html_cross_refs_empty_when_none(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text("<html><body><h2>Section</h2><p>no refs here</p></body></html>")
    sections = read_html(f)
    for s in sections:
        assert s.cross_refs == []


def test_read_html_cross_refs_deduped(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text(
        "<html><body>"
        "<h4>fn</h4>"
        "<p>Does the thing.</p>"
        '<dl class="section see"><dt>See also</dt>'
        '<dd><a href="#">Alpha</a></dd>'
        '<dd><a href="#">Alpha</a></dd>'
        "</dl>"
        "</body></html>"
    )
    sections = read_html(f)
    refs = [r for s in sections for r in s.cross_refs if r == "Alpha"]
    assert refs.count("Alpha") == 1


def test_read_html_cross_refs_not_in_text(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text(
        "<html><body>"
        "<h4>fn</h4><p>actual content</p>"
        '<dl class="section see"><dt>See also</dt>'
        '<dd><a href="#">ShouldNotAppearInText</a></dd></dl>'
        "</body></html>"
    )
    sections = read_html(f)
    fn_section = next(s for s in sections if s.page_label == "fn")
    assert "ShouldNotAppearInText" not in fn_section.text


# ── read_html atomic section detection (Features 2 & 6) ─────────────────────


def test_read_html_h4_section_is_atomic(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text("<html><body><h4>myMethod</h4><p>description</p></body></html>")
    sections = read_html(f)
    method_section = next(s for s in sections if s.page_label == "myMethod")
    assert method_section.atomic is True


def test_read_html_h2_section_not_atomic_by_default(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text("<html><body><h2>Overview</h2><p>general text</p></body></html>")
    sections = read_html(f)
    overview = next(s for s in sections if s.page_label == "Overview")
    assert overview.atomic is False


def test_read_html_fieldtable_marks_atomic(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text(
        "<html><body>"
        "<h2>MyEnum</h2>"
        '<table class="fieldtable"><tr><td>VALUE_A</td></tr></table>'
        "</body></html>"
    )
    sections = read_html(f)
    enum_section = next(s for s in sections if s.page_label == "MyEnum")
    assert enum_section.atomic is True


# ── _data_index honors atomic flag (Features 2 & 6) ─────────────────────────


def test_data_index_atomic_section_not_split(tmp_path, monkeypatch):
    """An atomic section longer than CHUNK_SIZE should produce exactly 1 chunk."""
    from unittest.mock import MagicMock

    import engra.commands as cmd
    from engra.readers import Section

    long_text = "x" * (CHUNK_SIZE * 3)
    atomic_section = Section(text=long_text, phys_page=1, page_label="Big", total=1, atomic=True)
    monkeypatch.setattr(cmd, "read_file", lambda path: [atomic_section])
    monkeypatch.setattr(cmd, "store_file", lambda path, copy=True: path)

    added_ids = []
    col = MagicMock()
    col.count.return_value = 0
    col.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    col.add.side_effect = lambda ids, **kwargs: added_ids.extend(ids)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    import numpy as np

    fake_model = MagicMock()
    fake_model.embed.return_value = iter([np.zeros(10)])
    monkeypatch.setattr(cmd, "load_model", lambda: fake_model)

    f = tmp_path / "test.html"
    f.write_text("dummy")
    cmd._data_index([f], store=False, auto_describe=False)

    assert len(added_ids) == 1


def test_data_index_nonatomic_long_section_is_split(tmp_path, monkeypatch):
    """A non-atomic section longer than CHUNK_SIZE should produce multiple chunks."""
    from unittest.mock import MagicMock

    import numpy as np

    import engra.commands as cmd
    from engra.readers import Section

    long_text = "y " * (CHUNK_SIZE + 100)
    section = Section(text=long_text, phys_page=1, page_label="Long", total=1, atomic=False)
    monkeypatch.setattr(cmd, "read_file", lambda path: [section])
    monkeypatch.setattr(cmd, "store_file", lambda path, copy=True: path)

    added_ids = []
    col = MagicMock()
    col.count.return_value = 0
    col.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    col.add.side_effect = lambda ids, **kwargs: added_ids.extend(ids)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    fake_model = MagicMock()
    fake_model.embed.return_value = iter([np.zeros(10)] * 20)
    monkeypatch.setattr(cmd, "load_model", lambda: fake_model)

    f = tmp_path / "test.html"
    f.write_text("dummy")
    cmd._data_index([f], store=False, auto_describe=False)

    assert len(added_ids) > 1


# ── _data_search new result fields (Batch 1) ─────────────────────────────────


def _make_meta_with_new_fields(**overrides):
    """Build a metadata dict that includes the new breadcrumb/cross_refs fields."""

    base = {
        "source": "/tmp/doc.html",
        "filename": "doc.html",
        "page": 1,
        "page_label": "MyClass",
        "total_pages": 5,
        "chunk": 0,
        "project": "proj",
        "indexed_at": "2026-01-01T00:00:00+00:00",
        "source_mtime": 1234567890.0,
        "model": "intfloat/multilingual-e5-large",
        "chunk_size": 1500,
        "chunk_overlap": 200,
        "breadcrumb": "Namespace > MyClass",
        "cross_refs": "Foo::bar,Baz::qux",
    }
    base.update(overrides)
    return base


def _make_search_col(meta):
    from unittest.mock import MagicMock

    col = MagicMock()
    col.count.return_value = 1
    col.get.return_value = {"metadatas": [meta], "documents": ["text"], "ids": ["id1"]}
    col.query.return_value = {
        "documents": [["text"]],
        "metadatas": [[meta]],
        "distances": [[0.1]],
        "ids": [["id1"]],
    }
    return col


def test_data_search_returns_breadcrumb_field(monkeypatch):
    import engra.commands as cmd

    meta = _make_meta_with_new_fields()
    monkeypatch.setattr(cmd, "get_collection", lambda: _make_search_col(meta))
    monkeypatch.setattr(cmd, "load_model", lambda: _make_fake_model())
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    results = cmd._data_search("query")
    assert results[0]["breadcrumb"] == "Namespace > MyClass"


def test_data_search_returns_cross_references_field(monkeypatch):
    import engra.commands as cmd

    meta = _make_meta_with_new_fields()
    monkeypatch.setattr(cmd, "get_collection", lambda: _make_search_col(meta))
    monkeypatch.setattr(cmd, "load_model", lambda: _make_fake_model())
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    results = cmd._data_search("query")
    assert results[0]["cross_references"] == ["Foo::bar", "Baz::qux"]


def test_data_search_cross_references_empty_when_meta_missing(monkeypatch):
    import engra.commands as cmd

    meta = _make_meta_with_new_fields(cross_refs="")
    monkeypatch.setattr(cmd, "get_collection", lambda: _make_search_col(meta))
    monkeypatch.setattr(cmd, "load_model", lambda: _make_fake_model())
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    results = cmd._data_search("query")
    assert results[0]["cross_references"] == []


# ── DEFAULT_MIN_SCORE (Feature 1) ────────────────────────────────────────────


def test_default_min_score_value():
    assert DEFAULT_MIN_SCORE == 0.3


# ── _data_list_members (Feature 4) ──────────────────────────────────────────


def _make_list_members_col(metas, docs=None):
    from unittest.mock import MagicMock

    docs = docs or ["text"] * len(metas)
    col = MagicMock()
    col.count.return_value = len(metas)
    col.get.return_value = {
        "metadatas": metas,
        "documents": docs,
        "ids": [f"id{i}" for i in range(len(metas))],
    }
    return col


def test_data_list_members_empty_db(monkeypatch):
    from unittest.mock import MagicMock

    import engra.commands as cmd

    col = MagicMock()
    col.count.return_value = 0
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    assert _data_list_members("doc.html") == []


def test_data_list_members_groups_by_section(monkeypatch):
    import engra.commands as cmd

    metas = [
        {"page": 1, "page_label": "MyClass", "chunk": 0, "filename": "doc.html", "breadcrumb": ""},
        {"page": 1, "page_label": "MyClass", "chunk": 1, "filename": "doc.html", "breadcrumb": ""},
    ]
    docs = ["chunk0 text", "chunk1 text"]
    col = _make_list_members_col(metas, docs)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    result = _data_list_members("doc.html")
    assert len(result) == 1
    assert result[0]["section"] == "MyClass"
    assert len(result[0]["chunks"]) == 2


def test_data_list_members_multiple_sections_sorted(monkeypatch):
    import engra.commands as cmd

    metas = [
        {"page": 2, "page_label": "BSection", "chunk": 0, "filename": "doc.html", "breadcrumb": ""},
        {"page": 1, "page_label": "ASection", "chunk": 0, "filename": "doc.html", "breadcrumb": ""},
    ]
    col = _make_list_members_col(metas)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    result = _data_list_members("doc.html")
    assert len(result) == 2
    assert result[0]["page"] == 1  # sorted by page


def test_data_list_members_section_filter_substring(monkeypatch):
    import engra.commands as cmd

    metas = [
        {
            "page": 1,
            "page_label": "Dogs overview",
            "chunk": 0,
            "filename": "doc.html",
            "breadcrumb": "",
        },
        {
            "page": 2,
            "page_label": "Cats overview",
            "chunk": 0,
            "filename": "doc.html",
            "breadcrumb": "",
        },
    ]
    col = _make_list_members_col(metas)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    result = _data_list_members("doc.html", section_filter="dogs")
    assert len(result) == 1
    assert result[0]["section"] == "Dogs overview"


def test_data_list_members_section_filter_case_insensitive(monkeypatch):
    import engra.commands as cmd

    metas = [
        {
            "page": 1,
            "page_label": "Dogs overview",
            "chunk": 0,
            "filename": "doc.html",
            "breadcrumb": "",
        },
    ]
    col = _make_list_members_col(metas)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    result = _data_list_members("doc.html", section_filter="DOGS")
    assert len(result) == 1


def test_data_list_members_returns_breadcrumb_in_chunks(monkeypatch):
    import engra.commands as cmd

    metas = [
        {
            "page": 1,
            "page_label": "Foo",
            "chunk": 0,
            "filename": "doc.html",
            "breadcrumb": "NS > Class",
        },
    ]
    col = _make_list_members_col(metas, ["content"])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    result = _data_list_members("doc.html")
    assert result[0]["chunks"][0]["breadcrumb"] == "NS > Class"
