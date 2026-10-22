CC = cc
CFLAGS = -std=c11 -O3 -Wall -Wextra -march=native
LDFLAGS = -lm

all: run benchmark

run: gemma4.c
	$(CC) $(CFLAGS) gemma4.c -o run $(LDFLAGS)

benchmark: performance/benchmark.c gemma4.c
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) -g performance/benchmark.c -o benchmark $(LDFLAGS)

libgemma4.so: gemma4.c
	$(CC) $(CFLAGS) -fPIC -shared -DGEMMA4_NO_MAIN gemma4.c -o libgemma4.so $(LDFLAGS)

.PHONY: all benchmark clean
clean:
	rm -f run benchmark libgemma4.so perf.data
