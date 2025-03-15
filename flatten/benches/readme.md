# Caveats

- I suspect that there may be a bug in `hain`'s implementation, so the results for that one are probably wrong.
- `levien-simd` was a half-baked attempt at sprinkling some sse in `levien`'s algorithm with underwhelming results. Raph levien has [a work in progress](https://gist.github.com/raphlinus/5f4e9feb85fd79bafc72da744571ec0e) avx2 version that is more promising.

# Performance

To run the benchmarks: `cargo bench`

Benchmark results on an AMD Ryzen 7 PRO 6850U:

![Cubic bézier benchmark results](results-cubic.svg)

![Quadratic bézier benchmark results](results-quadratic.svg)

# Edge count

To produce the edge counts: `cargo test -- --nocapture`

Cubic bézier curves:

| tolerance |  0.01 |  0.03 |  0.05 |  0.08 |  0.10 |  0.15 |  0.20 |  0.25 |  0.50 |  1.00 |
|-----------| -----:| -----:| -----:| -----:| -----:| -----:| -----:| -----:| -----:| -----:|
| recursive | 2288129 | 1363020 | 1053942 | 823670 | 682280 | 587594 | 527508 | 467414 | 312494 | 236479 |
| linear    | 2255375 | 1413185 | 1000862 | 809227 | 703986 | 578230 | 498198 | 442722 | 316320 | 222504 |
| levien-19 | 1708590 | 1085991 | 772649 | 633755 | 550317 | 451987 | 393241 | 353367 | 254431 | 183622 |
| levien-37 | 1917284 | 1217976 | 865097 | 708784 | 615646 | 504475 | 438236 | 392778 | 281205 | 202492 |
| levien-55 | 2258444 | 1433147 | 1017367 | 832680 | 722399 | 591598 | 513389 | 459935 | 328333 | 235734 |
| hain      | 3135836 | 1988972 | 1411183 | 1155309 | 1002741 | 821859 | 714078 | 640544 | 458168 | 329181 |
| sedeberg  | 3206955 | 2032625 | 1440607 | 1178565 | 1022101 | 836656 | 726281 | 651107 | 463860 | 331359 |
| fwd-diff  | 3206955 | 2032625 | 1440607 | 1178565 | 1022101 | 836656 | 726281 | 651107 | 463860 | 331359 |
| hfd       | 2616409 | 1734954 | 1215441 | 1034111 | 871491 | 681503 | 611250 | 562894 | 379582 | 284342 |

Quadratic bézier curves:

| tolerance |  0.01 |  0.03 |  0.05 |  0.08 |  0.10 |  0.15 |  0.20 |  0.25 |  0.50 |  1.00 |
|-----------| -----:| -----:| -----:| -----:| -----:| -----:| -----:| -----:| -----:| -----:|
| recursive | 2269505 | 1351843 | 1046285 | 818341 | 676775 | 583200 | 524826 | 465572 | 311350 | 237241 |
| linear    | 2236352 | 1401608 | 992695 | 802519 | 698060 | 573823 | 494841 | 440007 | 314960 | 223257 |
| sedeberg  | 1667590 | 1059932 | 753645 | 618369 | 536873 | 440975 | 383910 | 344832 | 248102 | 180451 |
| levien    | 1584190 | 1007222 | 716390 | 587726 | 510680 | 419477 | 365296 | 328333 | 236813 | 172735 |
| fwd-diff  | 1667590 | 1059932 | 753645 | 618369 | 536873 | 440975 | 383910 | 344832 | 248102 | 180451 |

![Cubic bézier benchmark results](cubic-vis.svg)
