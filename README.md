# engra

Local-first semantic search over your documents. Index PDFs once, search them instantly from anywhere. Organize files into projects and switch context with a single command.

> *An engra is the biological term for a stored memory trace.*

## Installation

### From source (recommended for now)

```bash
pipx install .                        # CPU
pipx install ".[gpu]"                 # GPU (replaces fastembed with fastembed-gpu)
```

Or with pip into a venv:

```bash
pip install -e .
pip install -e ".[gpu]"               # for GPU support
```

## Quick start

```bash
engra index ./iso11783_6/report.pdf  # index a PDF — project = parent dir name
engra search "torque sensor"         # search all indexed documents
```

## Indexing

```bash
engra index report.pdf                        # project = parent directory name
engra index report.pdf --project iso-std      # override project name
engra index ./docs/                           # index all PDFs in a directory
engra index report.pdf --force                # re-index even if already present
engra index report.pdf --link                 # symlink instead of copying the file
engra index report.pdf --no-store             # index without storing a copy
```

## Searching

```bash
engra search "torque sensor calibration"      # search (respects active project session)
engra search "Drehmomentsensor" --top 10      # more results
engra search "query" --min-score 0.4          # filter low-quality matches
engra search "query" --file report.pdf        # restrict to one file
engra search "query" --project iso-std        # restrict to a project (overrides session)
engra search "query" --all                    # global search, ignore active session
```

## Projects

Projects group related files together. Activate a project to scope all searches to it — the session lasts 8 hours or until you deactivate.

```bash
# Activate / deactivate
engra project activate iso-std                # activate a single project
engra project activate iso-std machinery      # activate multiple projects
engra project deactivate                      # clear session, back to global search
engra project active                          # show currently active project(s)

# Manage
engra project list                            # all projects with file/chunk counts
engra project rename iso-std iso11783         # rename across all indexed chunks
engra project remove iso-std                  # remove project from index
```

Once a project is active, searches are automatically scoped to it:

```bash
engra project activate iso-std
engra search "hydraulic pressure"             # searches only in iso-std
engra search "hydraulic pressure" --all       # override: search everywhere
engra project deactivate
```

## Listing and removing documents

```bash
engra list                                    # show all indexed files with project, chunks, path
engra remove report.pdf                       # remove by filename (if unambiguous)
engra remove ./docs/report.pdf                # remove by full path
```

## Data locations

| Purpose | Path |
|---|---|
| Index (chromadb) | `~/.local/share/engra/db/` |
| Stored file copies | `~/.local/share/engra/files/` |
| Session state | `~/.local/share/engra/state.toml` |
| Config | `~/.config/engra/config.toml` |
| Logs | `~/.cache/engra/log/engra.log` |

## Config

`~/.config/engra/config.toml` is created automatically on first run:

```toml
[backend]
type = "local"
# server_url = "http://host:8000"   # uncomment to use a shared server (coming soon)

[index]
copy = true   # set to false to symlink instead of copying files
```

## GPU support

The default install uses CPU inference via ONNX Runtime (fastembed).
For GPU acceleration, install the GPU variant:

```bash
pip install ".[gpu]"
```

> **Note:** Requires CUDA compute capability ≥ 7.0.

## Windows notes

- `--link` requires Admin privileges or Developer Mode — engra automatically falls back to `--copy` with a warning if symlinks are unavailable
- Binaries are available for Windows via the one-liner installer above
- Terminal colors require Windows Terminal or a modern console (cmd.exe has limited support)

## Development

```bash
pip install -e ".[test]"    # install with test dependencies
pip install -e ".[dev]"     # install with dev dependencies
pytest                      # run tests
ruff check .                # lint
ruff format .               # format
```

## Releasing

```bash
bump-my-version bump patch   # 0.1.0 → 0.1.1
bump-my-version bump minor   # 0.1.1 → 0.2.0
git push --follow-tags       # triggers release workflow
```

## CI/CD

- **test.yaml** — ruff + pytest on every push/PR to `main`
- **release.yaml** — builds standalone binaries for Linux and macOS on version tag push
- **Dependabot** — weekly pip and GitHub Actions dependency updates
