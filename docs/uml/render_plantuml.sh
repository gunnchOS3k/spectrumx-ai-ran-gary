#!/usr/bin/env bash
# Render all PlantUML sources in this directory to SVG (optional local workflow).
# Requires: plantuml in PATH (https://plantuml.com/starting) or:
#   docker run --rm -v "$PWD:/work" -w /work plantuml/plantuml:latest *.puml
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
if command -v plantuml >/dev/null 2>&1; then
  plantuml -tsvg ./*.puml
  echo "Wrote SVG next to .puml files in $DIR"
else
  echo "plantuml not found. Install from https://plantuml.com/starting or use Docker image plantuml/plantuml."
  exit 1
fi
