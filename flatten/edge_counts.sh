#!/bin/sh

Echo "Cubic bezier curves"

echo "inkscape"
FLATTEN_INPUT=inkscape FLATTEN_OUTPUT=notes/results/edge-count-cubic-inkscape.md cargo test --release edge_count_cubic
flatten-helper normalize -i notes/results/edge-count-cubic-inkscape.md -o notes/results/edge-count-cubic-inkscape-normalized.md
flatten-helper graph -i notes/results/edge-count-cubic-inkscape-normalized.md -o notes/results/edge-count-cubic-inkscape.svg -t Inkscape -s "Normalized edge count score (lower is better)"

echo "nehab"
FLATTEN_INPUT=nehab FLATTEN_OUTPUT=notes/results/edge-count-cubic-nehab.md cargo test --release edge_count_cubic
flatten-helper normalize -i notes/results/edge-count-cubic-nehab.md -o notes/results/edge-count-cubic-nehab-normalized.md
flatten-helper graph -i notes/results/edge-count-cubic-nehab-normalized.md -o notes/results/edge-count-cubic-nehab.svg -t Nehab -s "Normalized edge count score (lower is better)"

echo "font"
FLATTEN_INPUT=font FLATTEN_OUTPUT=notes/results/edge-count-cubic-font.md cargo test --release edge_count_cubic
flatten-helper normalize -i notes/results/edge-count-cubic-font.md -o notes/results/edge-count-cubic-font-normalized.md
flatten-helper graph -i notes/results/edge-count-cubic-font-normalized.md -o notes/results/edge-count-cubic-font.svg -t Font -s "Normalized edge count score (lower is better)"

echo "tiger"
FLATTEN_INPUT=tiger FLATTEN_OUTPUT=notes/results/edge-count-cubic-tiger.md cargo test --release edge_count_cubic
flatten-helper normalize -i notes/results/edge-count-cubic-tiger.md -o notes/results/edge-count-cubic-tiger-normalized.md
flatten-helper graph -i notes/results/edge-count-cubic-tiger-normalized.md -o notes/results/edge-count-cubic-tiger.svg -t Tiger -s "Normalized edge count score (lower is better)"

echo "all"
FLATTEN_OUTPUT=notes/results/edge-count-cubic-all.md cargo test --release edge_count_cubic
flatten-helper normalize -i notes/results/edge-count-cubic-all.md -o notes/results/edge-count-cubic-all-normalized.md
flatten-helper graph -i notes/results/edge-count-cubic-all-normalized.md -o notes/results/edge-count-cubic-all.svg -t All -s "Normalized edge count score (lower is better)"

Echo "Quadratic bezier curves"

echo "inkscape"
FLATTEN_INPUT=inkscape FLATTEN_OUTPUT=notes/results/edge-count-quadratic-inkscape.md cargo test --release edge_count_quadratic
flatten-helper normalize -i notes/results/edge-count-quadratic-inkscape.md -o notes/results/edge-count-quadratic-inkscape-normalized.md
flatten-helper graph -i notes/results/edge-count-quadratic-inkscape-normalized.md -o notes/results/edge-count-quadratic-inkscape.svg -t Inkscape -s "Normalized edge count score (lower is better)"

echo "nehab"
FLATTEN_INPUT=nehab FLATTEN_OUTPUT=notes/results/edge-count-quadratic-nehab.md cargo test --release edge_count_quadratic
flatten-helper normalize -i notes/results/edge-count-quadratic-nehab.md -o notes/results/edge-count-quadratic-nehab-normalized.md
flatten-helper graph -i notes/results/edge-count-quadratic-nehab-normalized.md -o notes/results/edge-count-quadratic-nehab.svg -t Nehab -s "Normalized edge count score (lower is better)"

echo "font"
FLATTEN_INPUT=font FLATTEN_OUTPUT=notes/results/edge-count-quadratic-font.md cargo test --release edge_count_quadratic
flatten-helper normalize -i notes/results/edge-count-quadratic-font.md -o notes/results/edge-count-quadratic-font-normalized.md
flatten-helper graph -i notes/results/edge-count-quadratic-font-normalized.md -o notes/results/edge-count-quadratic-font.svg -t Font -s "Normalized edge count score (lower is better)"

echo "tiger"
FLATTEN_INPUT=tiger FLATTEN_OUTPUT=notes/results/edge-count-quadratic-tiger.md cargo test --release edge_count_quadratic
flatten-helper normalize -i notes/results/edge-count-quadratic-tiger.md -o notes/results/edge-count-quadratic-tiger-normalized.md
flatten-helper graph -i notes/results/edge-count-quadratic-tiger-normalized.md -o notes/results/edge-count-quadratic-tiger.svg -t Tiger -s "Normalized edge count score (lower is better)"

echo "all"
FLATTEN_OUTPUT=notes/results/edge-count-quadratic-all.md cargo test --release edge_count_quadratic
flatten-helper normalize -i notes/results/edge-count-quadratic-all.md -o notes/results/edge-count-quadratic-all-normalized.md
flatten-helper graph -i notes/results/edge-count-quadratic-all-normalized.md -o notes/results/edge-count-quadratic-all.svg -t All -s "Normalized edge count score (lower is better)"

echo "done."
