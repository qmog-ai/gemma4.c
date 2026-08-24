#include "win.h"

#include <errno.h>

void *mmap(void *addr, size_t length, int protection, int flags, int fd, int64_t offset) {
    (void)addr;
    if (!length || flags != MAP_PRIVATE || !(protection & PROT_READ)) {
        errno = EINVAL;
        return MAP_FAILED;
    }

    HANDLE file = (HANDLE)_get_osfhandle(fd);
    if (file == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return MAP_FAILED;
    }

    int writable = protection & PROT_WRITE;
    DWORD page = writable ? PAGE_WRITECOPY : PAGE_READONLY;
    DWORD access = writable ? FILE_MAP_COPY : FILE_MAP_READ;
    HANDLE mapping = CreateFileMappingA(file, NULL, page, 0, 0, NULL);
    if (!mapping) {
        errno = EIO;
        return MAP_FAILED;
    }

    uint64_t position = (uint64_t)offset;
    void *view = MapViewOfFile(mapping, access, (DWORD)(position >> 32), (DWORD)position, length);
    CloseHandle(mapping);
    if (!view) {
        errno = EIO;
        return MAP_FAILED;
    }
    return view;
}

int munmap(void *address, size_t length) {
    (void)length;
    if (UnmapViewOfFile(address)) return 0;
    errno = EIO;
    return -1;
}
