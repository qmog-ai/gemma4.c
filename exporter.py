#!/usr/bin/env python3
"""Export Gemma 4 E2B to the binary consumed by gemma4.c."""

import argparse
import json
import math
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
import safetensors


N_LAYERS = 35
N_KV_LAYERS = 15
TOKEN_MAX = 94
VOCAB_MAX = 262144
BYTE_TOKEN_BASE = 238
SPECIAL_MAX = 256
ENCODE_VOCAB_MAX = 32768
MERGES_MAX = 514906
GROUP_SIZE = 64
BLOCK_ROWS = 16
BLOCK_WIDTH = 4

TOKENIZER_SPAN = 33_429_932
# data, scales, shape[4] — matches C Tensor (32 bytes)
TENSOR_RECORD = struct.Struct("<QQ4i")
LOOKUP_RECORD = struct.Struct("<8sii")
TEXT_TENSOR_SPAN = 55_008
HEADER_SIZE = 4  # MOG\0
GELU_TABLE_N = 4096
GELU_TABLE_LO = -8.0
GELU_TABLE_HI = 8.0

cursor = 0


def align64(value):
    return (value + 63) & ~63


def read_json(root, name):
    with (root / name).open(encoding="utf-8") as file:
        return json.load(file)


def tensor(weights, path, synthetic=None):
    global cursor
    shape = tuple(synthetic.shape) if synthetic is not None else tuple(
        int(n) for n in weights.get_slice(path).get_shape()
    )
    if not 1 <= len(shape) <= 2 or any(size <= 0 for size in shape):
        raise ValueError(f"unsupported shape for {path}: {shape}")
    quantized = synthetic is None and len(shape) == 2
    if quantized and (shape[0] % BLOCK_ROWS or shape[1] % GROUP_SIZE):
        raise ValueError(f"cannot pack {path} with shape {shape}")
    count = math.prod(shape)
    result = {
        "path": path, "shape": shape, "synthetic": synthetic,
        "data": cursor, "scales": 0,
    }
    cursor = align64(cursor + count * (1 if quantized else 4))
    if quantized:
        result["scales"] = cursor
        cursor = align64(cursor + count // GROUP_SIZE * 2)
    return result


def gelu_table():
    x = np.linspace(GELU_TABLE_LO, GELU_TABLE_HI, GELU_TABLE_N, dtype=np.float32)
    return (0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))).astype(np.float32)


def rope_tables(config):
    def table(head_dim, fraction, theta):
        pairs = int(head_dim * fraction) // 2
        exponent = 2 * np.arange(pairs, dtype=np.float32) / np.float32(head_dim)
        frequency = np.float32(1) / np.power(np.float32(theta), exponent)
        angles = np.arange(config["max_position_embeddings"], dtype=np.float32)[:, None] * frequency
        cosine = np.cos(angles)
        np.sin(angles, out=angles)
        return cosine, angles

    sliding = table(config["head_dim"], 1,
                    config["rope_parameters"]["sliding_attention"]["rope_theta"])
    full = config["rope_parameters"]["full_attention"]
    global_ = table(config["global_head_dim"], full["partial_rotary_factor"], full["rope_theta"])
    return *sliding, *global_


def text_tensors(weights, config):
    tables = rope_tables(config)
    rope_cache = {}

    def shared_rope(index):
        if index not in rope_cache:
            rope_cache[index] = tensor(weights, None, synthetic=tables[index])
        return rope_cache[index]

    slots = [
        tensor(weights, "model.language_model.embed_tokens.weight"),
        tensor(weights, "model.language_model.embed_tokens_per_layer.weight"),
    ]
    for layer in range(N_LAYERS):
        p = f"model.language_model.layers.{layer}."
        shared_kv = layer < N_KV_LAYERS
        rope_index = 0 if config["layer_types"][layer] == "sliding_attention" else 2
        layer_slots = [
            tensor(weights, p + "input_layernorm.weight"), tensor(weights, p + "layer_scalar"),
            tensor(weights, p + "pre_feedforward_layernorm.weight"),
            tensor(weights, p + "post_attention_layernorm.weight"),
            tensor(weights, p + "post_feedforward_layernorm.weight"),
            tensor(weights, p + "post_per_layer_input_norm.weight"),
            tensor(weights, p + "per_layer_input_gate.weight"),
            tensor(weights, p + "per_layer_projection.weight"),
            tensor(weights, p + "self_attn.q_norm.weight"),
            tensor(weights, p + "self_attn.k_norm.weight") if shared_kv else None,
            tensor(weights, p + "self_attn.q_proj.weight"),
            tensor(weights, p + "self_attn.k_proj.weight") if shared_kv else None,
            tensor(weights, p + "self_attn.v_proj.weight") if shared_kv else None,
            tensor(weights, p + "self_attn.o_proj.weight"),
            tensor(weights, p + "mlp.gate_proj.weight"),
            tensor(weights, p + "mlp.up_proj.weight"),
            tensor(weights, p + "mlp.down_proj.weight"),
            shared_rope(rope_index), shared_rope(rope_index + 1),
        ]
        intermediate_size = 6144 if layer < N_KV_LAYERS else 12288
        if (layer_slots[14]["shape"] != (intermediate_size, 1536)
                or layer_slots[15]["shape"] != (intermediate_size, 1536)
                or layer_slots[16]["shape"] != (1536, intermediate_size)):
            raise ValueError(f"unexpected MLP shapes in layer {layer}")
        if shared_kv:
            head_width = 512 if rope_index == 2 else 256
            if (layer_slots[11]["shape"] != (head_width, 1536)
                    or layer_slots[12]["shape"] != (head_width, 1536)):
                raise ValueError(f"unexpected K/V shapes in layer {layer}")
        slots += layer_slots
    gelu = tensor(weights, None, synthetic=gelu_table())
    gelu["shape"] = (GELU_TABLE_N, int(GELU_TABLE_LO), int(GELU_TABLE_HI))
    slots += [
        tensor(weights, "model.language_model.norm.weight"),
        tensor(weights, "model.language_model.per_layer_model_projection.weight"),
        tensor(weights, "model.language_model.per_layer_projection_norm.weight"),
        gelu,
    ]
    return slots


def validate_text_config(model):
    if model.get("model_type") != "gemma4":
        raise ValueError("checkpoint is not Gemma 4")
    text = model["text_config"]
    if ((text["hidden_size"], text["vocab_size"], text["num_hidden_layers"],
            text["num_hidden_layers"] - text["num_kv_shared_layers"],
            text["num_key_value_heads"], text["head_dim"],
            text["global_head_dim"], text["intermediate_size"],
            text["use_double_wide_mlp"]) !=
            (1536, VOCAB_MAX, N_LAYERS, N_KV_LAYERS, 1, 256, 512, 6144, True)):
        raise ValueError("expected Gemma 4 E2B")
    if (text["max_position_embeddings"], text["sliding_window"],
            text["hidden_size_per_layer_input"]) != (131072, 512, 256):
        raise ValueError("expected Gemma 4 E2B text configuration")
    if (np.float32(text["rms_norm_eps"]) != np.float32(float.fromhex("0x1.0c6f7ap-20"))
            or np.float32(text["final_logit_softcapping"]) != np.float32(30.0)):
        raise ValueError("expected Gemma 4 E2B text configuration")
    layer_types = ["sliding_attention"] * 4 + ["full_attention"]
    if text["layer_types"] != layer_types * 7:
        raise ValueError("expected Gemma 4 E2B attention sequence")
    return text


def put_string(buffer, offset, width, value):
    encoded = value.encode("utf-8")
    if b"\0" in encoded or len(encoded) >= width:
        raise ValueError(f"token cannot fit char[{width}]")
    buffer[offset:offset + len(encoded)] = encoded


def decoded_token(token_id, token, special_ids):
    if token_id in special_ids:
        payload = b""
    elif len(token) == 6 and token[:3] == "<0x" and token[5] == ">" \
            and all(char in "0123456789abcdefABCDEF" for char in token[3:5]):
        payload = bytes([int(token[3:5], 16)])
    else:
        payload = token.replace("▁", " ").encode("utf-8")
        if b"\0" in payload:
            raise ValueError(f"token cannot fit char[{TOKEN_MAX}]")
    if len(payload) >= TOKEN_MAX:
        raise ValueError(f"token cannot fit char[{TOKEN_MAX}]")
    return payload


def compile_tokenizer(data):
    raw_vocab = data["model"]["vocab"]
    raw_merges = data["model"]["merges"]
    added = data["added_tokens"]
    vocab = dict(raw_vocab)
    for token in added:
        content, token_id = token["content"], token["id"]
        if content in vocab and vocab[content] != token_id:
            raise ValueError(f"conflicting token ID for {content!r}")
        vocab[content] = token_id
    ids = [None] * (max(vocab.values()) + 1)
    for token, token_id in vocab.items():
        if token_id < 0 or token_id >= VOCAB_MAX or ids[token_id] is not None:
            raise ValueError("token IDs must be unique and dense")
        ids[token_id] = token
    if len(ids) != VOCAB_MAX or any(token is None for token in ids):
        raise ValueError("token IDs must be unique and dense")

    merges, results, pairs = [], set(), set()
    for rank, merge in enumerate(raw_merges):
        left, right = merge if isinstance(merge, list) else merge.split(" ", 1)
        pair = vocab[left], vocab[right]
        if pair in pairs or left + right not in vocab:
            raise ValueError(f"invalid merge {merge!r}")
        pairs.add(pair)
        results.add(left + right)
        merges.append((struct.pack("<2i", *pair), vocab[left + right], rank))
    specials = sorted(
        ((token["id"], token["content"]) for token in added if token["special"]),
        key=lambda item: (-len(item[1].encode()), item[1].encode()),
    )
    special_ids = {token_id for token_id, _ in specials}
    special_tokens = {token for _, token in specials}
    for byte in range(256):
        token = f"<0x{byte:02X}>"
        if (vocab.get(token) != BYTE_TOKEN_BASE + byte
                or token in special_tokens or token in results):
            raise ValueError("byte tokens must be contiguous non-special non-merge entries")
    encode = [(token.encode().ljust(8, b"\0"), token_id, 0)
              for token_id, token in enumerate(ids)
              if token not in special_tokens and token not in results
              and len(token.encode()) <= 4]
    if len(specials) > SPECIAL_MAX or len(encode) > ENCODE_VOCAB_MAX \
            or len(merges) > MERGES_MAX:
        raise ValueError("tokenizer exceeds fixed MOG limits")

    buffer = bytearray(TOKENIZER_SPAN)
    struct.pack_into("<3i", buffer, 0, len(merges), len(encode), len(specials))
    for token_id, token in enumerate(ids):
        payload = decoded_token(token_id, token, special_ids)
        offset = 12 + token_id * TOKEN_MAX
        buffer[offset:offset + len(payload)] = payload
    special_offset = 12 + VOCAB_MAX * TOKEN_MAX
    encode_offset = special_offset + SPECIAL_MAX * 100
    merge_offset = encode_offset + ENCODE_VOCAB_MAX * LOOKUP_RECORD.size
    for index, (token_id, token) in enumerate(specials):
        put_string(buffer, special_offset + index * 100, TOKEN_MAX, token)
        struct.pack_into("<i", buffer, special_offset + index * 100 + 96, token_id)
    for base, entries in ((encode_offset, encode), (merge_offset, merges)):
        for index, entry in enumerate(sorted(entries)):
            LOOKUP_RECORD.pack_into(buffer, base + index * LOOKUP_RECORD.size, *entry)
    return bytes(buffer)


def tensor_record(tensor):
    shape = tensor["shape"] + (0,) * (4 - len(tensor["shape"]))
    return TENSOR_RECORD.pack(tensor["data"], tensor["scales"], *shape)


def quantize(array):
    groups = array.reshape(-1, GROUP_SIZE)
    # Round the scale to its fp16 storage precision first, then quantize against the rounded value so the runtime dequantizes with the exact scale used here. Floor at the smallest normal fp16.
    scales = np.maximum(np.abs(groups).max(axis=1) / 127.0, 6.104e-05).astype("<f2")
    wide = scales.astype("<f4")
    values = np.rint(groups / wide[:, None]).clip(-127, 127).astype(np.int8)
    return values, scales


def read_weight(weights, path, rows=None):
    tensor = weights.get_tensor(path) if rows is None else weights.get_slice(path)[rows]
    return tensor.float().contiguous().numpy()


def write_tensor(output, weights, tensor):
    if tensor["synthetic"] is not None:
        arrays = [np.asarray(tensor["synthetic"], dtype="<f4", order="C")]
    elif len(tensor["shape"]) == 2:
        total_rows, columns = tensor["shape"]
        chunk_rows = max(1, (16 * 1024 * 1024) // columns)
        chunk_rows = max(BLOCK_ROWS, chunk_rows - chunk_rows % BLOCK_ROWS)
        arrays = (read_weight(weights, tensor["path"],
                              slice(start, min(total_rows, start + chunk_rows)))
                  for start in range(0, total_rows, chunk_rows))
    else:
        arrays = [read_weight(weights, tensor["path"])]

    data_at, scales_at = tensor["data"], tensor["scales"]
    for array in arrays:
        array = np.asarray(array, dtype="<f4", order="C")
        if not scales_at:
            output.seek(data_at)
            output.write(array.tobytes())
            data_at += array.nbytes
            continue

        values, scales = quantize(array)
        rows, columns = array.shape
        blocks, groups = rows // BLOCK_ROWS, columns // GROUP_SIZE
        values = values.reshape(rows, columns).reshape(
            blocks, BLOCK_ROWS, groups, GROUP_SIZE // BLOCK_WIDTH, BLOCK_WIDTH
        ).transpose(0, 2, 3, 1, 4).reshape(-1)
        scales = scales.reshape(blocks, BLOCK_ROWS, groups).transpose(0, 2, 1).reshape(-1)
        output.seek(data_at)
        output.write(values.tobytes())
        data_at += values.nbytes
        output.seek(scales_at)
        output.write(scales.tobytes())
        scales_at += scales.nbytes


def export(checkpoint_path, output_path):
    global cursor
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = read_json(checkpoint_path, "config.json")
    tokenizer = compile_tokenizer(read_json(checkpoint_path, "tokenizer.json"))
    text_config = validate_text_config(model)

    weights_path = checkpoint_path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError("checkpoint is missing model.safetensors")
    cursor = align64(HEADER_SIZE + TOKENIZER_SPAN + TEXT_TENSOR_SPAN)
    with safetensors.safe_open(weights_path, framework="pt") as weights:
        text = text_tensors(weights, text_config)
        file_size = cursor

        with tempfile.TemporaryDirectory(
            prefix=f".{output_path.name}.", dir=output_path.parent,
        ) as directory:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                temporary_path = Path(output.name)
                output.truncate(file_size)
                output.write(b"MOG\0")
                output.write(tokenizer)
                for item in text:
                    record = tensor_record(item) if item else bytes(TENSOR_RECORD.size)
                    output.write(record)

                written = set()
                for item in text:
                    if item is None or item["data"] in written:
                        continue
                    written.add(item["data"])
                    write_tensor(output, weights, item)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, output_path)

    text_count = len({item["data"] for item in text if item})
    print(f"Wrote {output_path} ({file_size / 1024**3:.2f} GiB): {text_count} text tensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
