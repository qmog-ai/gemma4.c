CC = cc
CFLAGS = -std=c11 -O3 -Wall -Wextra -march=native -fopenmp
LDFLAGS = -lm
WINCC = x86_64-w64-mingw32-gcc

run: gemma4.c
	$(CC) $(CFLAGS) gemma4.c -o run $(LDFLAGS)

win64: gemma4.c win.c win.h
	$(WINCC) $(CFLAGS) -static -D_WIN32 gemma4.c win.c -o run.exe $(LDFLAGS) -lshell32

.PHONY: clean win64
clean:
	rm -f run run.exe
