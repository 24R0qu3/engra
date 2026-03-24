# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install for development
pip install -e ".[test]"      # test tools (pytest, ruff)
pip install -e ".[dev]"       # dev tools (bump-my-version, ruff)

# Lint and format
ruff check .
ruff format .
ruff format --check .         # CI mode

# Tests
pytest                        # all tests
pytest tests/test_engra.py::TestClassName::test_name  # single test

# Version bump (creates git tag, triggers release CI)
bump-my-version bump patch|minor|major
git push --follow-tags
```

## Architecture

**engra** is a local-first semantic search CLI. Documents are chunked, embedded via `intfloat/multilingual-e5-large` (fastembed), and stored in a local chromadb vector database. Search returns semantically similar chunks ranked by cosine similarity.

### Module layout

```
src/engra/
├── main.py       — CLI definition (argparse), routes subcommands
├── commands.py   — all command implementations (index, search, get, info, list, remove, bookmark, project)
├── storage.py    — chromadb client, file copy/link, session management
├── readers.py    — per-format document parsers → Section dataclasses
├── config.py     — loads ~/.config/engra/config.toml with defaults
└── log.py        — logging setup
```

### Data flow

1. **Indexing**: `readers.py` parses a document into `Section` objects (one per page/slide/heading) → `commands.py` chunks sections into 1500-char overlapping windows → fastembed embeds in batches of 32 → chromadb stores vectors + metadata.
2. **Search**: query embedded with same model → chromadb cosine similarity lookup → results filtered by active project session and optional `--min-score`.

### Key constants (`commands.py`)

- `MODEL_NAME = "intfloat/multilingual-e5-large"`
- `CHUNK_SIZE = 1500`, `CHUNK_OVERLAP = 200`, `MIN_CHARS = 80`

### Storage locations (via platformdirs)

| Purpose | Path |
|---------|------|
| Vector index | `~/.local/share/engra/db/` |
| Stored files | `~/.local/share/engra/files/` |
| Session state | `~/.local/share/engra/state.toml` |
| User config | `~/.config/engra/config.toml` |

### Projects & sessions

Documents belong to a **project** (derived from parent directory name or `--project` flag). Projects must be *activated* before search scopes to them. Active projects are stored in `state.toml` with an 8-hour TTL. Multiple projects can be active simultaneously.

### `Section` dataclass (`readers.py`)

The core abstraction passed from readers to the chunker:
```python
@dataclass
class Section:
    text: str
    phys_page: int      # 1-based physical index
    page_label: str     # human-readable (e.g., "i", "Slide 2")
    total: int          # total sections in document
```

Each format has its own reader function (`read_pdf`, `read_markdown`, `read_docx`, etc.) registered in a format dispatch dict. Adding a new format means writing a reader and adding it to the registry.

### CI

`.github/workflows/test.yaml` runs `ruff check`, `ruff format --check`, and `pytest` on every push/PR to `main`.
`.github/workflows/release.yaml` builds standalone binaries for Linux, macOS, and Windows via PyInstaller on version tags.
