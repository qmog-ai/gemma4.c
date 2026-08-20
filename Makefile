CC = cc
CFLAGS = -std=c11 -O3 -Wall -Wextra
LDFLAGS = -lm

run: gemma4.c
	$(CC) $(CFLAGS) gemma4.c -o run $(LDFLAGS)

.PHONY: clean
clean:
	rm -f run
