#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/release.sh <version> <release-notes-file>
# Example: ./scripts/release.sh 0.3.0 notes.md
#
# Steps:
#   1. Bump version in pyproject.toml and __init__.py
#   2. Commit and tag
#   3. Push to GitHub with tag
#   4. Build sdist + wheel
#   5. Upload to PyPI
#   6. Create GitHub release

VERSION="${1:-}"
NOTES_FILE="${2:-}"

if [[ -z "$VERSION" ]]; then
  echo "Usage: ./scripts/release.sh <version> [release-notes-file]"
  echo "  version:            e.g. 0.3.0 (no 'v' prefix)"
  echo "  release-notes-file: optional markdown file for GitHub release body"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYPROJECT="$REPO_ROOT/pyproject.toml"
INIT_PY="$REPO_ROOT/src/genmedia/__init__.py"
ENV_FILE="$REPO_ROOT/.env"

# --- Preflight checks ---

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Error: Working tree has uncommitted changes. Commit or stash first."
  exit 1
fi

if git tag -l "v$VERSION" | grep -q .; then
  echo "Error: Tag v$VERSION already exists."
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: .env file not found (needed for PyPI token)."
  exit 1
fi

PYPI_TOKEN="$(grep '^PyPI_token=' "$ENV_FILE" | head -1 | sed 's/^PyPI_token="//' | sed 's/"$//')"
if [[ -z "$PYPI_TOKEN" ]]; then
  echo "Error: PyPI_token not found in .env"
  exit 1
fi

command -v gh >/dev/null 2>&1 || { echo "Error: gh CLI not found. Install: brew install gh"; exit 1; }

# --- Bump version ---

CURRENT_VERSION="$(grep '^version = ' "$PYPROJECT" | sed 's/version = "//' | sed 's/"//')"
echo "Bumping $CURRENT_VERSION -> $VERSION"

sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" "$PYPROJECT"
sed -i '' "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" "$INIT_PY"

# --- Commit, tag, push ---

git add "$PYPROJECT" "$INIT_PY"
git commit -m "bump: v$VERSION"
git tag "v$VERSION"
echo "Pushing to GitHub..."
git push origin main --tags

# --- Build ---

DIST_DIR="$REPO_ROOT/dist2"
rm -rf "$DIST_DIR"
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi
echo "Building..."
"$PYTHON" -m build --outdir "$DIST_DIR" 2>&1 | tail -2

# --- Upload to PyPI ---

echo "Uploading to PyPI..."
.venv/bin/twine upload "$DIST_DIR"/genmedia-"$VERSION"* \
  -u __token__ -p "$PYPI_TOKEN" 2>&1 | grep -E '(Uploading|View at)'

# --- GitHub release ---

if [[ -n "$NOTES_FILE" && -f "$NOTES_FILE" ]]; then
  gh release create "v$VERSION" --title "v$VERSION" --notes-file "$NOTES_FILE"
else
  # Auto-generate from git log since last tag
  PREV_TAG="$(git tag --sort=-v:refname | grep -v "v$VERSION" | head -1)"
  NOTES="$(git log "$PREV_TAG"..HEAD~1 --oneline --no-decorate)"
  gh release create "v$VERSION" --title "v$VERSION" --notes "$NOTES"
fi

echo ""
echo "Released v$VERSION"
echo "  PyPI:   https://pypi.org/project/genmedia/$VERSION/"
echo "  GitHub: https://github.com/tennyson-mccalla/genmedia/releases/tag/v$VERSION"
