#!/bin/sh

echo "-------------------------------------"
echo "$1"
echo "Running the benchmarks..."
FLATTEN_INPUT="$1" cargo bench cubic
echo "Extracting benchmark results..."
flatten-helper criterion -i . -o notes/results/bench-cubic-$1-$2.md
echo "Making graphs..."
flatten-helper graph -i notes/results/bench-cubic-$1-$2.md -o notes/results/bench-cubic-$1-$2.svg
echo "$1 done."
