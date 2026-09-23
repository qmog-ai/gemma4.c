# Minimal CPU optimization study

This study starts from the current batched FP32-scale runtime at `86d7a27`.
Each directory below is a complete implementation that can be built and read on
its own.

## Implementations

| Stage | Source | Change from previous stage | Runtime lines | Diff |
| --- | --- | --- | ---: | ---: |
| Batched scalar | [`00-batched-scalar/gemma4.c`](../f32-optimization-snapshots/00-batched-scalar/gemma4.c) | Existing baseline | 582 | — |
| SIMD matmul | [`01-simd-matmul/gemma4.c`](../f32-optimization-snapshots/01-simd-matmul/gemma4.c) | One `dot64()` helper, with AVX-512 VNNI and AVX2 implementations | 607 | +28/−3 |
| SIMD attention | [`02-simd-attention/gemma4.c`](../f32-optimization-snapshots/02-simd-attention/gemma4.c) | Vectorize query-key dots and weighted value sums | 617 | +18/−8 |
| Row-major tiling | [`03-row-major-tiling/gemma4.c`](../f32-optimization-snapshots/03-row-major-tiling/gemma4.c) | Compute several token rows together without changing weight layout | 636 | +40/−21 |
| Packed weights | [`04-packed-weights/gemma4.c`](../f32-optimization-snapshots/04-packed-weights/gemma4.c) | Pack 16 weight rows in the exporter and compute their outputs together | 640 | C +66/−62; exporter +10/−2 |

Final `main` is 700 lines. The clean packed endpoint is 640 lines.

## Method

Ryzen 7 7700, GCC 15.2, eight OpenMP threads, and `-O3 -march=native`.
Each number is the median of three runs after one discarded warmup. `pp512`
measures a 512-token prompt. Decode generates 128 tokens after filling the KV
cache to the stated depth. Model loading and cache filling are excluded.

## Performance

| Stage | pp512 | tg128@d0 | tg128@d512 | tg128@d1024 |
| --- | ---: | ---: | ---: | ---: |
| Batched scalar | 86.05 | 25.01 | 23.15 | 22.48 |
| SIMD matmul | 101.05 | 25.01 | 23.17 | 22.50 |
| SIMD attention | 114.52 | 25.21 | 24.41 | 24.09 |
| Row-major tiling | 116.32 | 23.73 | 23.12 | 22.95 |
| Packed weights | 609.62 | 25.25 | 24.47 | 24.14 |

All values are tokens per second.

## Numerical validation

Each snapshot was compared directly with the Hugging Face BF16 model over the
same 64 positions.

| Stage | Mean KL | Top-1 agreement |
| --- | ---: | ---: |
| Batched scalar | 0.005276 | 59/64 |
| SIMD matmul | 0.005276 | 59/64 |
| SIMD attention | 0.005244 | 60/64 |
| Row-major tiling | 0.005244 | 60/64 |
| Packed weights | 0.005374 | 60/64 |

The SIMD matmul is numerically identical to the scalar dot product in this
check. Row-major tiling is identical to the preceding SIMD-attention runtime.
SIMD attention and packing change floating-point accumulation order but do not
increase divergence materially. A longer validation is still required before
promoting any experiment into the real history.

## AVX2 fallback

The packed endpoint was also compiled with `-march=x86-64-v3`, excluding
AVX-512 and VNNI. It reached 313.76 tok/s for pp512 and 25.25 tok/s for shallow
decode. Its 64-position Hugging Face metrics were identical to the native
build: mean KL 0.005374 and 60/64 top-1 agreement.

## Conclusions

- SIMD matmul is worth keeping. It adds 25 lines and improves pp512 by 17.4%.
  Decode does not move because it is already constrained by memory bandwidth.
- SIMD attention is also worth keeping. It adds 10 lines, improves pp512 by
  another 13.3%, and improves decode by 5.4% at depth 512 and 7.1% at depth
  1024.
- Row-major tiling is not worth keeping by itself. It improves pp512 by only
  1.6%, loses roughly 5–6% decode throughput, and adds 19 lines.
- Packing is the decisive prefill change. Relative to the row-major tile it
  improves pp512 by 5.24× and restores decode performance.
- The 640-line packed endpoint reaches 96.6% of final `main`'s published pp512
  throughput while remaining substantially easier to read.

For the real history, the evidence supports SIMD matmul next, then SIMD
attention. Row-major tiling should remain a documented experiment rather than
a permanent endpoint. Packing can introduce the tile and matching layout
together once the article has shown why the row-major experiment is weak.
