#!/usr/bin/env bash
# Render all PlantUML sources in docs/uml to committed SVGs under docs/uml/rendered/.
# Stable output names: <diagram_id>.svg (from @startuml <id>).
#
# Requirements:
#   - Java 17+ (java on PATH)
#   - plantuml.jar at docs/uml/.tools/plantuml.jar
#
# Download jar (example):
#   mkdir -p docs/uml/.tools && curl -L -o docs/uml/.tools/plantuml.jar \
#     https://github.com/plantuml/plantuml/releases/download/v1.2024.7/plantuml-1.2024.7.jar
#
# Layout: sources use !pragma layout smetana where Graphviz would be required, so a
# system Graphviz "dot" install is optional for this repo's diagrams.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
JAR="$DIR/.tools/plantuml.jar"
OUT="$DIR/rendered"
mkdir -p "$OUT"

if ! command -v java >/dev/null 2>&1; then
  echo "render_plantuml.sh: Java not found. Install a JDK and ensure 'java' is on PATH." >&2
  exit 1
fi

if [[ ! -f "$JAR" ]]; then
  echo "render_plantuml.sh: Missing $JAR" >&2
  echo "  Download PlantUML JAR into docs/uml/.tools/ (see header comments)." >&2
  exit 1
fi

java -jar "$JAR" -tsvg -o rendered ./*.puml
echo "Wrote SVGs under $OUT"

if command -v dot >/dev/null 2>&1; then
  echo "Graphviz 'dot' detected (optional for this repo; smetana pragma covers most diagrams)."
else
  echo "Note: Graphviz 'dot' not on PATH — OK if all diagrams use smetana or sequence/activity engines."
fi
