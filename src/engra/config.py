import tomllib
from pathlib import Path

from platformdirs import user_config_dir

CONFIG_PATH = Path(user_config_dir("engra", appauthor=False)) / "config.toml"
BOOKMARKS_PATH = Path(user_config_dir("engra", appauthor=False)) / "bookmarks.json"

DEFAULTS: dict = {
    "backend": {
        "type": "local",
        # "server_url": "http://host:8000",
    },
    "index": {
        "copy": True,  # False = symlink
    },
    "ask": {
        "api_base": "http://localhost:11434/v1",
        "model": "llama3",
        "api_key": "ollama",
        "context_chunks": 5,
        "system_prompt": (
            "You are a helpful assistant. Answer the question using only the provided context. "
            "If the context does not contain enough information, say so."
        ),
    },
    "autodescribe": {
        # "openai" = OpenAI-compatible endpoint (Ollama or any compat server, no extra dep)
        # "claude" = Anthropic native API (requires: pip install 'engra[ai]' + ANTHROPIC_API_KEY)
        # "disabled" = skip auto-description entirely
        "backend": "openai",
        "api_base": "http://localhost:11434/v1",
        "model": "llama3",
        "api_key": "ollama",
        "claude_model": "claude-haiku-4-5-20251001",
    },
}

_DEFAULT_TOML = """\
[backend]
type = "local"
# server_url = "http://host:8000"

[index]
copy = true  # set to false to symlink instead of copying files

[ask]
# OpenAI-compatible LLM endpoint (default: local Ollama)
api_base = "http://localhost:11434/v1"
model = "llama3"
api_key = "ollama"
context_chunks = 5

[autodescribe]
# backend: "openai" (Ollama/any OpenAI-compat) | "claude" (Anthropic API) | "disabled"
backend = "openai"
api_base = "http://localhost:11434/v1"
model = "llama3"
api_key = "ollama"
# claude_model = "claude-haiku-4-5-20251001"  # used when backend = "claude"
"""


def load() -> dict:
    if not CONFIG_PATH.exists():
        return DEFAULTS
    with open(CONFIG_PATH, "rb") as f:
        user = tomllib.load(f)
    merged: dict = {}
    for section, defaults in DEFAULTS.items():
        user_section = user.get(section, {})
        merged[section] = {**defaults, **user_section}
    for section in user:
        if section not in merged:
            merged[section] = user[section]
    return merged


def init() -> None:
    """Write default config on first run."""
    if CONFIG_PATH.exists():
        return
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(_DEFAULT_TOML)
