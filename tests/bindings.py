"""ctypes declarations for gemma4.c."""

import ctypes
from pathlib import Path


def load_library(path: Path):
    lib = ctypes.CDLL(str(path))
    lib.load_model.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.load_model.restype = ctypes.c_void_p
    lib.free_model.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.create_state.restype = ctypes.c_void_p
    lib.free_state.argtypes = [ctypes.c_void_p]
    lib.forward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.logits.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.tokenize_prompt.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
    ]
    lib.tokenize_prompt.restype = ctypes.c_int
    return lib
