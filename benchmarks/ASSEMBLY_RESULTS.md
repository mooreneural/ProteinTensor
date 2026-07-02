# Full feature-assembly benchmark

Traditional (mmCIF parse + A3M parse + distance-matrix compute) vs
ProteinTensor (read structure + MSA + distance matrix + embedding from `.ptt`).
Tests the README's headline feature-assembly speedup on the current code.

- **Platform:** Windows-10-10.0.26100-SP0
- **proteintensor:** 0.2.0 | numpy 1.26.4 | zarr 2.18.7
- **Run:** 2026-07-02T21:08:43+00:00

| Structure | Res | MSA depth | Traditional | ProteinTensor | Speedup |
|---|---|---|---|---|---|
| 1UBQ | 76 | 512 | 14.062 ms | 7.142 ms | **2.0x** |
| 6LU7 | 312 | 1024 | 48.709 ms | 13.57 ms | **3.6x** |
| 4HHB | 574 | 2048 | 117.961 ms | 22.736 ms | **5.2x** |
| 6M0J | 791 | 2048 | 196.423 ms | 38.305 ms | **5.1x** |
| 6VXX | 2916 | 8192 | 1395.119 ms | 308.731 ms | **4.5x** |
| 6OHW | 3525 | 8192 | 1462.067 ms | 381.472 ms | **3.8x** |

**Average speedup across 6 structures: 4.0x**

> Structure is real; MSA depth and embedding shape are realistic but the
> numeric content is synthetic (timing depends on dimensions, not values).
> MSA generation and ESM2 inference are one-time costs and are not measured.

