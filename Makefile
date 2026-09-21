CC = cc
CFLAGS = -std=c11 -O3 -Wall -Wextra -march=native
LDFLAGS = -lm

all: run benchmark

run: gemma4.c
	$(CC) $(CFLAGS) gemma4.c -o run $(LDFLAGS)

benchmark: benchmark.c gemma4.c
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) -g benchmark.c -o benchmark $(LDFLAGS)

.PHONY: all benchmark clean
clean:
	rm -f run benchmark perf.data
