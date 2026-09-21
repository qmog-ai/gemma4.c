CC = cc
CFLAGS = -std=c11 -O3 -Wall -Wextra -march=native
LDFLAGS = -lm

all: run benchmark

run: gemma4.c
	$(CC) $(CFLAGS) gemma4.c -o run $(LDFLAGS)

benchmark: benchmark.c gemma4.c
	$(CC) $(CFLAGS) -g benchmark.c -o benchmark $(LDFLAGS)

profile: benchmark.c gemma4.c
	$(CC) $(CFLAGS) -g -fno-inline benchmark.c -o benchmark $(LDFLAGS)

.PHONY: all benchmark profile clean
clean:
	rm -f run benchmark perf.data
