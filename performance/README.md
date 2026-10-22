# Performance

Build the benchmark:

```sh
make benchmark
```

Measure prompt processing at specific prompt lengths:

```sh
./benchmark -m MODEL --pp 64,256,512 -r 3
```

Measure 128 generated tokens at specific starting context lengths:

```sh
./benchmark -m MODEL --tg 128 --d 64,256,512 -r 3
```

`--pp` selects prompt lengths, `--tg` selects a decode length, `--d` selects starting context lengths, and `-r` sets the number of measured repetitions. Each workload is warmed up once. Model loading, tokenization, sampling, and the initial context fill are excluded from timing.

Profile one measured workload with function inlining disabled:

```sh
./performance/profile.sh -m MODEL --tg 128 --d 512 -r 1
perf report --stdio --no-children --percent-limit 0
```

The profiler enables cycle counting immediately before the measured workload and disables it immediately afterward.
