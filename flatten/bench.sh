#!/bin/sh

echo "-------------------------------------"
echo "$1: $2"
echo "Running the benchmarks..."
FLATTEN_INPUT="$2" cargo bench $1
echo "Extracting benchmark results..."
flatten-helper criterion -i . -o notes/results/bench-$1-$2-$3.md
echo "Making graphs..."
flatten-helper graph -i notes/results/bench-$1-$2-$3.md -o notes/results/bench-$1-$2-$3.svg
echo "$2 done."
