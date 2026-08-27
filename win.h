#ifndef GEMMA4_WIN_H
#define GEMMA4_WIN_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <fcntl.h>
#include <io.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>

#define open _open
#define close _close
#define fstat _fstat64
#define stat _stat64
#define isatty _isatty
#define O_RDONLY _O_RDONLY
#define STDOUT_FILENO 1

#define PROT_READ 1
#define PROT_WRITE 2
#define MAP_PRIVATE 2
#define MAP_FAILED ((void *)-1)

void *mmap(void *addr, size_t length, int protection, int flags, int fd, int64_t offset);
int munmap(void *address, size_t length);
void argv_utf8(int *argc, char ***argv);

#endif
