# engram

Local-first semantic search over your documents. Index PDFs once, search them instantly from anywhere. Organize files into projects and switch context with a single command.

> *An engram is the biological term for a stored memory trace.*

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

### One-liner (once published to GitHub releases)

**Linux / macOS**
```bash
curl -fsSL https://raw.githubusercontent.com/<owner>/engram/main/install.sh | bash
```

**Windows (PowerShell)**
```powershell
irm https://raw.githubusercontent.com/<owner>/engram/main/install.ps1 | iex
```

## Quick start

```bash
engram index ./iso11783_6/report.pdf  # index a PDF — project = parent dir name
engram search "torque sensor"         # search all indexed documents
```

## Indexing

```bash
engram index report.pdf                        # project = parent directory name
engram index report.pdf --project iso-std      # override project name
engram index ./docs/                           # index all PDFs in a directory
engram index report.pdf --force                # re-index even if already present
engram index report.pdf --link                 # symlink instead of copying the file
engram index report.pdf --no-store             # index without storing a copy
```

## Searching

```bash
engram search "torque sensor calibration"      # search (respects active project session)
engram search "Drehmomentsensor" --top 10      # more results
engram search "query" --min-score 0.4          # filter low-quality matches
engram search "query" --file report.pdf        # restrict to one file
engram search "query" --project iso-std        # restrict to a project (overrides session)
engram search "query" --all                    # global search, ignore active session
```

## Projects

Projects group related files together. Activate a project to scope all searches to it — the session lasts 8 hours or until you deactivate.

```bash
# Activate / deactivate
engram project activate iso-std                # activate a single project
engram project activate iso-std machinery      # activate multiple projects
engram project deactivate                      # clear session, back to global search
engram project active                          # show currently active project(s)

# Manage
engram project list                            # all projects with file/chunk counts
engram project rename iso-std iso11783         # rename across all indexed chunks
engram project remove iso-std                  # remove project from index
```

Once a project is active, searches are automatically scoped to it:

```bash
engram project activate iso-std
engram search "hydraulic pressure"             # searches only in iso-std
engram search "hydraulic pressure" --all       # override: search everywhere
engram project deactivate
```

## Listing and removing documents

```bash
engram list                                    # show all indexed files with project, chunks, path
engram remove report.pdf                       # remove by filename (if unambiguous)
engram remove ./docs/report.pdf                # remove by full path
```

## Data locations

| Purpose | Path |
|---|---|
| Index (chromadb) | `~/.local/share/engram/db/` |
| Stored file copies | `~/.local/share/engram/files/` |
| Session state | `~/.local/share/engram/state.toml` |
| Config | `~/.config/engram/config.toml` |
| Logs | `~/.cache/engram/log/engram.log` |

## Config

`~/.config/engram/config.toml` is created automatically on first run:

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

- `--link` requires Admin privileges or Developer Mode — engram automatically falls back to `--copy` with a warning if symlinks are unavailable
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
