./bench.sh nehab $1
./bench.sh font $1
./bench.sh inkscape $1
./bench.sh tiger $1

echo "-------------------------------------"
echo "all"
echo "Running the benchmarks..."
cargo bench cubic
echo "Extracting benchmarks results..."
flatten-helper criterion -i . -o notes/results/bench-cubic-all-$1.md
echo "Making graphs..."
flatten-helper graph -i notes/results/bench-cubic-all-$1.md -o notes/results/bench-cubic-all-$1.svg -t All
echo "All done."
