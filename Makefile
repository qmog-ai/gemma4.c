CC = cc
CFLAGS = -std=c11 -O3 -Wall -Wextra -march=native -fopenmp
LDFLAGS = -lm

run: gemma4-e2b.c
	$(CC) $(CFLAGS) gemma4-e2b.c -o run $(LDFLAGS)

.PHONY: clean
clean:
	rm -f run
