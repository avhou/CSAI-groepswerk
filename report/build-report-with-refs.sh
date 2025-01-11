#!/usr/bin/env bash
set -euv

rm *.aux *.bbl *.blg *.brf *.log *.out || true
rm report.pdf || true
docker run --rm -v "$PWD":/data texlive/texlive bash -c "cd /data && pdflatex report.tex && biber report && pdflatex report.tex && pdflatex report.tex"
open report.pdf
