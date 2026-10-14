#!/bin/sh
set -eu
cd "$(dirname "$0")"
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error preprint.tex
bibtex preprint
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error preprint.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error preprint.tex
