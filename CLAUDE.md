# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install for development
pip install -e ".[test]"      # test tools (pytest, ruff)
pip install -e ".[dev]"       # dev tools (bump-my-version, ruff)
pip install -e ".[full,gpu]"  # everything: MCP + Claude AI + GPU

# Lint and format (ruff, line-length 100, rules E/F/I)
ruff check .
ruff format .
ruff format --check .         # CI mode

# Tests
pytest                                                # all tests
pytest tests/test_engra.py::TestClassName::test_name  # single test

# Version bump (edits __init__.py, commits, creates tag v{version})
bump-my-version bump patch|minor|major
git push --follow-tags        # triggers release CI
```

## Architecture

**engra** is a local-first semantic search CLI. Documents are chunked, embedded via
`intfloat/multilingual-e5-large` (fastembed / ONNX Runtime), and stored in a local chromadb
vector DB alongside a SQLite FTS5 keyword index. Search fuses dense (vector) and keyword (BM25)
retrieval via Reciprocal Rank Fusion by default (`mode: dense|keyword|hybrid`).

### Module layout

```
src/engra/
├── main.py       — argparse CLI, one subparser per command, routes to cmd_* funcs
├── commands.py   — ALL command implementations + embedding model setup (~2900 lines)
├── storage.py    — chromadb client + SQLite FTS5 keyword index, file copy/link, session/project state
├── mcp_server.py — MCP stdio server: tool manifest + _dispatch to the _data_* functions
├── readers.py    — per-format parsers → list[Section]; READERS dispatch dict
├── config.py     — DEFAULTS dict + config.toml loader (tomllib)
└── log.py        — logging setup
```

`main.py` defines flags, `commands.py` does the work — keep that split. Adding a subcommand
means: add a subparser in `main.py`, a `cmd_*` implementation in `commands.py`, and route it
in the dispatch block near the bottom of `main.py`. Each retrieval `cmd_*` has a matching
`_data_*` function (`_data_search`, `_data_get_chunks`, ...) that's the one MCP's `_dispatch`
calls too — CLI output formatting lives in `cmd_*`, everything else lives in `_data_*`.

### Data flow

1. **Index**: `readers.read_file()` dispatches on extension → `list[Section]` (one per
   page/slide/heading) → `commands.py` chunks into token-budgeted, overlapping windows
   (measured against the embedding model's real tokenizer, not characters) → fastembed embeds
   in batches → chromadb stores vectors + metadata (file, page, project, chunk index, `doc_id`)
   → the same chunk text is mirrored into the FTS5 keyword index (`storage.fts_add`).
2. **Search** (`_data_search`): query embedded → dense arm (chromadb cosine) and keyword arm
   (FTS5 BM25) each produce a ranked candidate list → fused via `_reciprocal_rank_fusion` in
   hybrid mode → reranked by a cross-encoder by default (`rerank`, degrades gracefully if the
   optional `flashrank` dependency is missing) → MMR-diversified to the final `top_k`
   (`diversify`, default on) → each result gets a `confidence` (min-max normalized *within that
   response*, not comparable across queries) alongside the raw `score`.

### Document identity (`doc_id`)

Two files sharing a basename in different projects are different documents. `doc_id_prefix(path)`
(`f"{name}_{md5(resolved_path)[:8]}"`) is the authoritative identity, stored as its own chromadb
metadata field and as the on-disk stored-file name (`storage.stored_name`). Filename-only lookups
(`_resolve_doc_scope`) raise `AmbiguousFilenameError` when a bare filename matches more than one
distinct `doc_id`; pass `doc_id` explicitly to disambiguate. Legacy chunks indexed before this
feature lack a `doc_id` and are treated as one shared bucket, not falsely flagged as ambiguous.

### Keyword index sync (`storage.py`)

The FTS5 `chunks` table must be kept in lockstep with chromadb on every mutation: add (index),
delete (reindex/remove/project remove), and update (project rename). See `fts_add`,
`fts_delete_by_ids`, `fts_delete_by_doc_id`, `fts_delete_by_project`, `fts_update_project` — any
new chromadb mutation path needs a matching FTS5 call. FTS5 is a compile-time-optional SQLite
module; `get_fts_connection()` degrades to `None` (warns once) if unavailable, and every keyword
path treats that as "no keyword results," not a crash.

### Key constants (`commands.py`, top of file)

- `MODEL_NAME = "intfloat/multilingual-e5-large"`, `MODEL_TOKEN_LIMIT = 512`
- `CHUNK_SIZE = 450` (target chunk size in **tokens**, not characters), `CHUNK_OVERLAP = 45`,
  `MIN_CHARS = 80`
- `DEFAULT_MIN_SCORE = 0.3` — the MCP `engra_search` confidence floor below which the response's
  `not_found` flag is set

Embedding batch size / threads / provider / device_id / max_tokens come from the `[embedding]`
config section, not constants (`batch_size=64`, `threads=0`→cpu_count, `provider="cpu"`,
`device_id=0`, `max_tokens=450`).

### `Section` dataclass (`readers.py`)

Core abstraction between readers and the chunker:

```python
@dataclass
class Section:
    text: str
    phys_page: int      # 1-based physical index
    page_label: str     # human-readable (e.g. "i", "Slide 2")
    total: int          # total sections in document
```

Each format has a `read_*` function registered in `READERS` (`.pdf`, `.md`, `.txt`, `.rst`,
`.html/.htm`, `.docx`, `.pptx`, `.epub`). Adding a format = write a reader and add it to
`READERS`; `SUPPORTED_EXTENSIONS` derives from it automatically.

### Projects & sessions

Documents belong to a **project** (parent directory name, or `--project`). A project must be
*activated* before search scopes to it; active projects live in `state.toml` with an 8-hour
TTL and multiple can be active at once. `--all` bypasses the session for a global search.

### GPU

CPU inference is the default. `[gpu]` extra swaps fastembed for fastembed-gpu; after
installing, `engra setup-gpu` replaces the stub `onnxruntime-gpu` with the full CUDA 12 wheel.
Provider is selected in `commands.py` via the `[embedding] provider` config
(`cpu`/`cuda`/…), which maps to an ONNX Runtime execution provider and falls back to CPU with a
warning if unavailable. Requires CUDA compute capability ≥ 7.0.

### Optional dependencies (extras)

Core install is CPU-only search/index. Extras gate features and their imports are lazy:
`mcp` (stdio server, `engra mcp`), `ai` (Claude auto-description, needs `ANTHROPIC_API_KEY`),
`gpu`, `rerank` (flashrank cross-encoder), `build` (pyinstaller). `full` = mcp + ai.

### Storage locations (platformdirs)

| Purpose | Path |
|---------|------|
| Vector index (chromadb) | `~/.local/share/engra/db/` |
| Keyword index (SQLite FTS5) | `~/.local/share/engra/fts.db` |
| Stored file copies | `~/.local/share/engra/files/` |
| Session state | `~/.local/share/engra/state.toml` |
| Project metadata | `~/.local/share/engra/projects.json` |
| User config | `~/.config/engra/config.toml` (auto-created first run) |
| Bookmarks | `~/.config/engra/bookmarks.json` |
| Model cache | `~/.cache/engra/models/` |
| Logs | `~/.local/state/engra/log/engra.log` |

### Commands (subparsers in `main.py`)

`index` (with `--check`, `--profile`, `--link`, `--no-store`, `--force`), `search` (with
`--mode dense|keyword|hybrid`, `--rerank`), `get` (with `--doc-id`), `info`, `ask` (RAG via
OpenAI-compatible endpoint or native Claude, `[ask] backend`), `list`, `remove` (with
`--doc-id`), `mcp`, `setup-gpu` (also installs `mcp`/`anthropic` if missing), `export`/`import`
(portable project archives that reuse embeddings), `bookmark` (save/run/list/remove), `project`
(list/active/activate/deactivate/rename/remove/describe/autodescribe).

### MCP server (`mcp_server.py`)

`engra_search`'s response is an envelope `{results, not_found, reason?}`, not a bare list —
`not_found` is set when nothing matched or the best `confidence` is below `DEFAULT_MIN_SCORE`.
`engra_list_files`/`engra_list_members` are paginated (`offset`/`limit` → `{items, total,
offset, limit}`). `engra_get_chunk`/`engra_get_neighbors`/`engra_list_members` accept `doc_id`.
`engra_index` only accepts paths under `[mcp] index_allowlist`. Tool failures return
`CallToolResult(isError=True)`, not a plain result. Payload pruning/pagination/error-wrapping
happens **only** at the `_dispatch` layer — the underlying `_data_*` functions stay full-detail
and unpaginated for CLI/`cmd_ask` reuse; don't change their return shapes for MCP-only needs.

## CI

- `.github/workflows/test.yaml` — `ruff check`, `ruff format --check`, `pytest` on push/PR to `main`.
- `.github/workflows/release.yaml` — PyInstaller standalone binaries (Linux/macOS/Windows) on version tags.
