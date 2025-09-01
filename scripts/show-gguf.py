import argparse
import struct
import sys
from pathlib import Path

GGUF_MAGIC = 0x46554747

class GGUFValueType:
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

GGML_TYPE_MAP = {
    0: "F32",
    1: "F16",
    8: "Q8_0",
    12: "Q4_K",
}

def read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")

def read_value(f, value_type):
    if value_type == GGUFValueType.UINT32:
        return struct.unpack("<I", f.read(4))[0]
    if value_type == GGUFValueType.FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    if value_type == GGUFValueType.BOOL:
        return bool(struct.unpack("?", f.read(1))[0])
    if value_type == GGUFValueType.STRING:
        return read_string(f)
    if value_type == GGUFValueType.ARRAY:
        array_type = struct.unpack("<I", f.read(4))[0]
        array_len = struct.unpack("<Q", f.read(8))[0]
        values = [read_value(f, array_type) for _ in range(array_len)]
        return values
    
    raise NotImplementedError(f"Reading for GGUF value type {value_type} is not implemented")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model-path", type=Path, required=True, help="Path to the GGUF model file to inspect.")
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: Input model file not found at {args.model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Inspecting GGUF file: {args.model_path}")
    print("-" * 80)

    with open(args.model_path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            print(f"Error: Invalid GGUF magic number. Expected {GGUF_MAGIC:#x}, got {magic:#x}", file=sys.stderr)
            sys.exit(1)

        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        print(f"GGUF Version: V{version}")
        print(f"Tensor Count: {tensor_count}")
        print(f"Metadata KV Count: {kv_count}")
        print("-" * 80)
        
        print("Metadata:")
        for _ in range(kv_count):
            key = read_string(f)
            value_type_enum = struct.unpack("<I", f.read(4))[0]
            try:
                value = read_value(f, value_type_enum)
                print(f"  - {key}: {value}")
            except NotImplementedError as e:
                print(f"  - {key}: [Could not read value: {e}]")

        print("-" * 80)
        print("Tensors:")
        for i in range(tensor_count):
            name = read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            
            shape_rev = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            shape = list(reversed(shape_rev))
            
            ggml_type_enum = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]

            type_str = GGML_TYPE_MAP.get(ggml_type_enum, f"Unknown ({ggml_type_enum})")

            print(f"  - [{i:03d}] Name: {name:<40} | Shape: {str(shape):<25} | DType: {type_str:<8}")

    print("-" * 80)

if __name__ == "__main__":
    main()
