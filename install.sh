#!/usr/bin/env bash
# install.sh — download and install engram from the latest GitHub release.
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/engram/main/install.sh | bash
#   INSTALL_DIR=/usr/local/bin bash install.sh   # custom location (needs sudo)
set -euo pipefail

REPO="<owner>/engram"
BIN_NAME="engram"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

# ── Detect platform ───────────────────────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
  Linux)  PLATFORM="linux"  ;;
  Darwin) PLATFORM="darwin" ;;
  *)
    echo "Unsupported OS: $OS" >&2
    exit 1
    ;;
esac

# ── Resolve latest release tag ────────────────────────────────────────────────
echo "Fetching latest release info..."
TAG="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
  | grep '"tag_name"' | head -1 \
  | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')"

if [ -z "$TAG" ]; then
  echo "Could not determine latest release tag." >&2
  exit 1
fi

# ── Download binary ───────────────────────────────────────────────────────────
URL="https://github.com/$REPO/releases/download/$TAG/${BIN_NAME}-${TAG}-${PLATFORM}"
echo "Downloading $BIN_NAME $TAG ($PLATFORM)..."

mkdir -p "$INSTALL_DIR"
curl -fsSL "$URL" -o "$INSTALL_DIR/$BIN_NAME"
chmod +x "$INSTALL_DIR/$BIN_NAME"

echo "Installed to $INSTALL_DIR/$BIN_NAME"

# ── PATH hint ─────────────────────────────────────────────────────────────────
if ! echo ":$PATH:" | grep -q ":$INSTALL_DIR:"; then
  echo ""
  echo "  $INSTALL_DIR is not in your PATH."
  echo "  Add it by running:"
  echo ""
  echo "    echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc  # bash"
  echo "    echo 'fish_add_path $INSTALL_DIR' >> ~/.config/fish/config.fish  # fish"
  echo ""
  echo "  Then restart your terminal."
fi

echo "Done. Run: $BIN_NAME --help"
