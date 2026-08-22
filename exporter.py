#!/usr/bin/env python3
"""Export Gemma 4 E2B to the binary consumed by gemma4-e2b.c."""

import argparse
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import warnings

import numpy as np
import safetensors


N_LAYERS = 35
N_KV_LAYERS = 15
TOKEN_MAX = 94
VOCAB_MAX = 262144
SPECIAL_MAX = 256
ENCODE_VOCAB_MAX = 32768
MERGES_MAX = 514906
GROUP_SIZE = 64
BLOCK_ROWS = 16
BLOCK_WIDTH = 4

TOKENIZER_SPAN = 36_182_444
# data, scales, shape[4] — matches C Tensor (32 bytes)
TENSOR_RECORD = struct.Struct("<QQ4i")
SECTION = struct.Struct("<8sII")
FOOTER = struct.Struct("<8sII6Q")
TEXT_TENSOR_SPAN = 55_008
VISION_WEIGHTS_SIZE = 11_920
AUDIO_WEIGHTS_SIZE = 15_072
MOG_VERSION = 10
HEADER_SIZE = 4 + 4  # MOG\0 + uint32 version
GELU_TABLE_N = 4096
GELU_TABLE_LO = -8.0
GELU_TABLE_HI = 8.0

F32 = 0
INT8 = 2
cursor = 0


def align64(value):
    return (value + 63) & ~63


def read_json(root, name):
    with (root / name).open(encoding="utf-8") as file:
        return json.load(file)


def tensor(weights, path, dtype=None, packed=False, synthetic=None):
    global cursor
    shape = tuple(synthetic.shape) if synthetic is not None else tuple(
        int(n) for n in weights.get_slice(path).get_shape()
    )
    count = math.prod(shape)
    if dtype is None:
        dtype = F32 if synthetic is not None or count == 1 or "norm" in path.lower() else INT8
    if not 1 <= len(shape) <= 4 or any(size <= 0 for size in shape):
        raise ValueError(f"unsupported shape for {path}: {shape}")
    if dtype == INT8 and count % GROUP_SIZE:
        raise ValueError(f"cannot quantize {path} with shape {shape}")
    if packed and (len(shape) != 2 or shape[0] % BLOCK_ROWS or shape[1] % GROUP_SIZE):
        raise ValueError(f"cannot pack {path} with shape {shape}")
    result = {
        "path": path, "shape": shape, "dtype": dtype,
        "block_rows": BLOCK_ROWS if packed else 0,
        "synthetic": synthetic, "clips": None,
        "data": cursor, "scales": 0,
    }
    cursor = align64(cursor + count * (4 if dtype == F32 else 1))
    if dtype == INT8:
        result["scales"] = cursor
        cursor = align64(cursor + count // GROUP_SIZE * 2)
    return result


def clipped_linear(weights, path):
    result = tensor(weights, path + ".linear.weight")
    result["clips"] = tuple(float(weights.get_tensor(path + "." + field))
                            for field in ("input_min", "input_max", "output_min", "output_max"))
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
        tensor(weights, "model.language_model.embed_tokens.weight", INT8, True),
        tensor(weights, "model.language_model.embed_tokens_per_layer.weight", INT8),
    ]
    for layer in range(N_LAYERS):
        p = f"model.language_model.layers.{layer}."
        shared_kv = layer < N_KV_LAYERS
        rope_index = 0 if config["layer_types"][layer] == "sliding_attention" else 2
        slots += [
            tensor(weights, p + "input_layernorm.weight"), tensor(weights, p + "layer_scalar"),
            tensor(weights, p + "pre_feedforward_layernorm.weight"),
            tensor(weights, p + "post_attention_layernorm.weight"),
            tensor(weights, p + "post_feedforward_layernorm.weight"),
            tensor(weights, p + "post_per_layer_input_norm.weight"),
            tensor(weights, p + "per_layer_input_gate.weight", INT8, True),
            tensor(weights, p + "per_layer_projection.weight", INT8, True),
            tensor(weights, p + "self_attn.q_norm.weight"),
            tensor(weights, p + "self_attn.k_norm.weight") if shared_kv else None,
            tensor(weights, p + "self_attn.q_proj.weight", INT8, True),
            tensor(weights, p + "self_attn.k_proj.weight", INT8, True) if shared_kv else None,
            tensor(weights, p + "self_attn.v_proj.weight", INT8, True) if shared_kv else None,
            tensor(weights, p + "self_attn.o_proj.weight", INT8, True),
            tensor(weights, p + "mlp.gate_proj.weight", INT8, True),
            tensor(weights, p + "mlp.up_proj.weight", INT8, True),
            tensor(weights, p + "mlp.down_proj.weight", INT8, True),
            shared_rope(rope_index), shared_rope(rope_index + 1),
        ]
    gelu = tensor(weights, None, synthetic=gelu_table())
    gelu["shape"] = (GELU_TABLE_N, int(GELU_TABLE_LO), int(GELU_TABLE_HI))
    slots += [
        tensor(weights, "model.language_model.norm.weight"),
        tensor(weights, "model.language_model.per_layer_model_projection.weight", INT8, True),
        tensor(weights, "model.language_model.per_layer_projection_norm.weight"),
        gelu,
    ]
    return slots


def vision_tensors(weights):
    result = [
        tensor(weights, "model.vision_tower.patch_embedder.input_proj.weight"),
        tensor(weights, "model.vision_tower.patch_embedder.position_embedding_table"),
    ]
    for layer in range(16):
        p = f"model.vision_tower.encoder.layers.{layer}"
        result += [tensor(weights, p + path) for path in (
            ".input_layernorm.weight", ".post_attention_layernorm.weight",
            ".pre_feedforward_layernorm.weight", ".post_feedforward_layernorm.weight",
            ".self_attn.q_norm.weight", ".self_attn.k_norm.weight",
        )]
        result += [clipped_linear(weights, p + ".self_attn." + name)
                   for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
        result += [clipped_linear(weights, p + ".mlp." + name)
                   for name in ("gate_proj", "up_proj", "down_proj")]
    result.append(tensor(weights, "model.embed_vision.embedding_projection.weight"))
    return result


def audio_tensors(weights, tables):
    p = "model.audio_tower"
    result = [tensor(weights, p + path) for path in (
        ".subsample_conv_projection.layer0.conv.weight",
        ".subsample_conv_projection.layer0.norm.weight",
        ".subsample_conv_projection.layer1.conv.weight",
        ".subsample_conv_projection.layer1.norm.weight",
        ".subsample_conv_projection.input_proj_linear.weight",
    )]
    for layer in range(12):
        p = f"model.audio_tower.layers.{layer}"
        for ff in ("feed_forward1", "feed_forward2"):
            base = p + "." + ff
            result += [clipped_linear(weights, base + ".ffw_layer_1"),
                       clipped_linear(weights, base + ".ffw_layer_2"),
                       tensor(weights, base + ".pre_layer_norm.weight"),
                       tensor(weights, base + ".post_layer_norm.weight")]
        result.append(tensor(weights, p + ".norm_pre_attn.weight"))
        result += [clipped_linear(weights, p + ".self_attn." + name)
                   for name in ("q_proj", "k_proj", "v_proj", "post")]
        result += [tensor(weights, p + path) for path in (
            ".self_attn.per_dim_scale", ".self_attn.relative_k_proj.weight",
            ".norm_post_attn.weight", ".lconv1d.pre_layer_norm.weight",
        )]
        result += [clipped_linear(weights, p + ".lconv1d.linear_start"),
                   tensor(weights, p + ".lconv1d.depthwise_conv1d.weight"),
                   tensor(weights, p + ".lconv1d.conv_norm.weight"),
                   clipped_linear(weights, p + ".lconv1d.linear_end"),
                   tensor(weights, p + ".norm_out.weight")]
    result += [
        tensor(weights, "model.audio_tower.output_proj.weight"),
        tensor(weights, "model.audio_tower.output_proj.bias"),
        tensor(weights, "model.embed_audio.embedding_projection.weight"),
        tensor(weights, None, synthetic=tables["@window"]),
        tensor(weights, None, synthetic=tables["@mel_filters"]),
    ]
    return result


def pack_text_config(model):
    if model.get("model_type") != "gemma4":
        raise ValueError("checkpoint is not Gemma 4")
    text = model["text_config"]
    if ((text["hidden_size"], text["vocab_size"], text["num_hidden_layers"],
            text["num_hidden_layers"] - text["num_kv_shared_layers"]) !=
            (1536, VOCAB_MAX, N_LAYERS, N_KV_LAYERS) or
            text["head_dim"] % 64 or text["global_head_dim"] % 64):
        raise ValueError("expected Gemma 4 E2B")
    kinds = {"sliding_attention": "s", "full_attention": "f"}
    layer_types = "".join(kinds[kind] for kind in text["layer_types"])
    layer_types = layer_types.encode("ascii").ljust(64, b"\0")
    return struct.pack(
        "<if64s2if",
        text["max_position_embeddings"],
        text["rms_norm_eps"], layer_types,
        text["sliding_window"], text["hidden_size_per_layer_input"],
        text["final_logit_softcapping"],
    )


def pack_modality_configs(model, processor):
    vision = model["vision_config"]
    audio = model["audio_config"]
    feature = processor["feature_extractor"]
    image = processor["image_processor"]
    vision_config = struct.pack(
        "<11i3f", vision["hidden_size"], vision["intermediate_size"],
        vision["num_hidden_layers"], vision["num_attention_heads"],
        vision["num_key_value_heads"], vision["head_dim"],
        vision["patch_size"], vision["pooling_kernel_size"],
        vision["default_output_length"], vision["position_embedding_size"], 3,
        vision["rope_parameters"]["rope_theta"], vision["rms_norm_eps"],
        image["rescale_factor"],
    )
    audio_config = struct.pack(
        "<17i7f", audio["hidden_size"], audio["num_hidden_layers"],
        audio["num_attention_heads"], audio["attention_chunk_size"],
        audio["attention_context_left"], audio["attention_context_right"],
        audio["conv_kernel_size"], *audio["subsampling_conv_channels"],
        audio["output_proj_dims"], feature["sampling_rate"],
        feature["fft_length"], feature["frame_length"], feature["hop_length"],
        feature["feature_size"], processor["audio_seq_length"],
        processor["audio_ms_per_token"], audio["attention_invalid_logits_value"],
        audio["attention_logit_cap"], audio["residual_weight"],
        audio["rms_norm_eps"], feature["min_frequency"],
        feature["max_frequency"], feature["mel_floor"],
    )
    return vision_config, audio_config, feature


def audio_tables(feature):
    from transformers.audio_utils import mel_filter_bank, window_function

    window = window_function(feature["frame_length"]).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filters = mel_filter_bank(
            num_frequency_bins=feature["fft_length"] // 2 + 1,
            num_mel_filters=feature["feature_size"],
            min_frequency=feature["min_frequency"],
            max_frequency=feature["max_frequency"],
            sampling_rate=feature["sampling_rate"], norm=None, mel_scale="htk",
        ).astype(np.float32)
    return {"@window": window, "@mel_filters": filters}


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
        merges.append((pair[0], pair[1], vocab[left + right], rank))
    specials = sorted(
        ((token["id"], token["content"]) for token in added if token["special"]),
        key=lambda item: (-len(item[1].encode()), item[1].encode()),
    )
    special_ids = {token_id for token_id, _ in specials}
    special_tokens = {token for _, token in specials}
    encode = [(token_id, token) for token_id, token in enumerate(ids)
              if token not in special_tokens and token not in results]
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
    merge_offset = encode_offset + ENCODE_VOCAB_MAX * 100
    for base, entries in ((special_offset, specials),
                          (encode_offset, sorted(encode,
                                                key=lambda item: item[1].encode()))):
        for index, (token_id, token) in enumerate(entries):
            put_string(buffer, base + index * 100, TOKEN_MAX, token)
            struct.pack_into("<i", buffer, base + index * 100 + 96, token_id)
    for index, merge in enumerate(sorted(merges)):
        struct.pack_into("<4i", buffer, merge_offset + index * 16, *merge)
    return bytes(buffer)


def tensor_record(tensor):
    shape = tensor["shape"] + (0,) * (4 - len(tensor["shape"]))
    return TENSOR_RECORD.pack(tensor["data"], tensor["scales"], *shape)


def write_tensor_records(output, tensors):
    for item in tensors:
        output.write(tensor_record(item))
        if item["clips"]:
            output.write(struct.pack("<4f", *item["clips"]))


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
        if tensor["block_rows"]:
            chunk_rows = max(BLOCK_ROWS,
                             chunk_rows - chunk_rows % BLOCK_ROWS)
        arrays = (read_weight(weights, tensor["path"],
                              slice(start, min(total_rows, start + chunk_rows)))
                  for start in range(0, total_rows, chunk_rows))
    else:
        arrays = [read_weight(weights, tensor["path"])]

    data_at, scales_at = tensor["data"], tensor["scales"]
    for array in arrays:
        array = np.asarray(array, dtype="<f4", order="C")
        if tensor["dtype"] == 0:
            output.seek(data_at)
            output.write(array.tobytes())
            data_at += array.nbytes
            continue

        values, scales = quantize(array)
        if tensor["block_rows"]:
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
    processor = read_json(checkpoint_path, "processor_config.json")
    tokenizer = compile_tokenizer(read_json(checkpoint_path, "tokenizer.json"))
    config = pack_text_config(model)
    vision_config, audio_config, feature = pack_modality_configs(model, processor)

    weights_path = checkpoint_path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError("checkpoint is missing model.safetensors")
    cursor = align64(HEADER_SIZE + len(config) + TOKENIZER_SPAN + TEXT_TENSOR_SPAN)
    with safetensors.safe_open(weights_path, framework="pt") as weights:
        text = text_tensors(weights, model["text_config"])

        vision_start = align64(cursor)
        cursor = align64(vision_start + SECTION.size + len(vision_config)
                         + VISION_WEIGHTS_SIZE)
        vision = vision_tensors(weights)
        vision_section = (vision_start, cursor - vision_start)

        audio_start = align64(cursor)
        cursor = align64(audio_start + SECTION.size + len(audio_config)
                         + AUDIO_WEIGHTS_SIZE)
        audio = audio_tensors(weights, audio_tables(feature))
        audio_section = (audio_start, cursor - audio_start)
        footer_at = align64(cursor)
        file_size = footer_at + FOOTER.size

        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.", suffix=".tmp",
            dir=output_path.parent, delete=False,
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        try:
            with temporary_path.open("w+b") as output:
                output.truncate(file_size)
                output.write(b"MOG\0" + struct.pack("<I", MOG_VERSION))
                output.write(config)
                output.write(tokenizer)
                for item in text:
                    record = tensor_record(item) if item else bytes(TENSOR_RECORD.size)
                    output.write(record)

                sections = (
                    (vision_section[0], vision_config, vision, b"G4VIS\0\0\0"),
                    (audio_section[0], audio_config, audio, b"G4AUD\0\0\0"),
                )
                for start, section_config, tensors, magic in sections:
                    output.seek(start)
                    output.write(SECTION.pack(magic, 0, len(tensors)))
                    output.write(section_config)
                    write_tensor_records(output, tensors)

                written = set()
                for tensors in (text, vision, audio):
                    for item in tensors:
                        if item is None or item["data"] in written:
                            continue
                        written.add(item["data"])
                        write_tensor(output, weights, item)

                output.seek(footer_at)
                output.write(FOOTER.pack(
                    b"G4MEND\0\0", 0, FOOTER.size,
                    0, 0, *vision_section, *audio_section,
                ))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, output_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    text_count = len({item["data"] for item in text if item})
    print(f"Wrote {output_path} ({file_size / 1024**3:.2f} GiB): "
          f"{text_count} text + {len(vision)} vision + {len(audio)} audio tensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
