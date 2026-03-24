# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install for development
pip install -e ".[test]"      # with test deps
pip install -e ".[dev]"       # with dev + version bump tools

# Run tests
pytest                         # all tests
pytest tests/test_engram.py::test_chunking  # single test

# Lint and format
ruff check .                   # lint
ruff format .                  # format
ruff format --check .          # check formatting (CI mode)

# Release
bump-my-version bump patch     # edits __init__.py and tags
git push --follow-tags         # triggers release workflow → PyInstaller binaries
```

## Architecture

Engram is a local-first semantic search CLI. Documents are chunked, embedded with `fastembed`, and stored in a local `chromadb` vector database.

### Module Roles

| Module | Role |
|--------|------|
| `main.py` | `argparse` CLI parser; dispatches to `commands.py` |
| `commands.py` | All command logic (~966 lines): index, search, get, info, list, remove, bookmark, project |
| `readers.py` | Document format readers returning `Section(text, phys_page, page_label, total)` |
| `storage.py` | File copy/symlink to data dir; session state (TOML, 8-hour TTL) |
| `config.py` | Loads `~/.config/engram/config.toml`, merges with defaults |
| `log.py` | Rotating file + console logging |

### Data Flow

**Indexing:** `File → readers.py (Sections) → chunk (1500 chars / 200 overlap) → fastembed → chromadb`

**Searching:** `Query → fastembed → chromadb cosine similarity → filter (project/filename/score) → rich display`

### Key Design Decisions

- **Embedding model:** hardcoded to `intfloat/multilingual-e5-large` (multilingual, CPU-friendly via ONNX)
- **Chunking:** fixed 1500-char chunks with 200-char overlap; no dynamic sizing
- **Projects:** session-scoped groupings with 8-hour TTL, stored in `~/.local/share/engram/state.toml`
- **Bookmarks:** named saved searches stored as JSON in the data directory
- **Staleness detection:** compares file `mtime` against `indexed_at` stored in chromadb metadata
- **Storage:** files copied or symlinked to `~/.local/share/engram/files/`; index at `~/.local/share/engram/db/`

### Adding a New Document Format

1. Add a `read_<format>(path) -> list[Section]` function to `readers.py`
2. Register it in the `READERS` dict in `commands.py` (maps file extension → reader function)

### CI

`.github/workflows/test.yaml` runs `ruff check`, `ruff format --check`, and `pytest` on every push/PR to `main`.
`.github/workflows/release.yaml` builds standalone binaries for Linux, macOS, and Windows via PyInstaller on version tags.
