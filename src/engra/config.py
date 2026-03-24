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
}

_DEFAULT_TOML = """\
[backend]
type = "local"
# server_url = "http://host:8000"

[index]
copy = true  # set to false to symlink instead of copying files
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
