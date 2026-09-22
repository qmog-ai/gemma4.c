CC = cc
CFLAGS = -std=c11 -O3 -Wall -Wextra -march=native -fopenmp
LDFLAGS = -lm

all: run benchmark

run: gemma4.c
	$(CC) $(CFLAGS) gemma4.c -o run $(LDFLAGS)

benchmark: benchmark.c gemma4.c
	$(CC) $(CFLAGS) benchmark.c -o benchmark $(LDFLAGS)

.PHONY: all clean
clean:
	rm -f run benchmark
