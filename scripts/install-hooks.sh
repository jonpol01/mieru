#!/bin/bash
# Install git hooks for this repo.
# Run once after cloning: ./scripts/install-hooks.sh

HOOK_DIR="$(git rev-parse --show-toplevel)/.git/hooks"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cp "$SCRIPT_DIR/pre-commit" "$HOOK_DIR/pre-commit"
chmod +x "$HOOK_DIR/pre-commit"

echo "Hooks installed."
