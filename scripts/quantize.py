import argparse
import struct
from pathlib import Path
import sys
import json
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download
from tqdm import tqdm


GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class GGUFValueType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

class GGUFWriter:
    def __init__(self, path: Path, arch: str):
        self.f = path.open("wb")
        self.arch = arch
        self.kv = {}
        self.tensors = []
        self.tensor_data = bytearray()

    def write_u32(self, val: int):
        self.f.write(struct.pack("<I", val))

    def write_u64(self, val: int):
        self.f.write(struct.pack("<Q", val))

    def write_f32(self, val: float):
        self.f.write(struct.pack("<f", val))

    def write_string(self, val: str):
        encoded = val.encode("utf-8")
        self.write_u64(len(encoded))
        self.f.write(encoded)

    def add_kv(self, key: str, value_type: int, value):
        self.kv[key] = (value_type, value)

    def add_tensor_info(self, name: str, tensor: np.ndarray, dtype_str: str, override_shape=None):
        shape = override_shape if override_shape is not None else tensor.shape
        offset = len(self.tensor_data)
        pad = (GGUF_DEFAULT_ALIGNMENT - (offset % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
        self.tensor_data.extend(b'\x00' * pad)
        
        self.tensors.append({
            "name": name,
            "shape": shape,
            "dtype": dtype_str,
            "offset": len(self.tensor_data),
        })
        self.tensor_data.extend(tensor.tobytes())

    def write_header(self):
        self.write_u32(GGUF_MAGIC)
        self.write_u32(GGUF_VERSION)
        self.write_u64(len(self.tensors))
        self.write_u64(len(self.kv))

        for key, (value_type, value) in self.kv.items():
            self.write_string(key)
            self.write_u32(value_type)
            if value_type == GGUFValueType.STRING:
                self.write_string(value)
            elif value_type == GGUFValueType.UINT32:
                self.write_u32(value)
            elif value_type == GGUFValueType.FLOAT32:
                 self.write_f32(value)
            elif value_type == GGUFValueType.BOOL:
                 self.f.write(struct.pack("<B", 1 if value else 0))
            else:
                raise NotImplementedError(f"Value type {value_type} not implemented")

    def write_tensor_info(self):
        for tensor in self.tensors:
            self.write_string(tensor["name"])
            shape = tensor["shape"]
            self.write_u32(len(shape))
            for dim in reversed(shape):
                self.write_u64(dim)
            
            ggml_dtype = {
                "f16": 1, "f32": 0, "q8_0": 8, "q4_k": 12
            }[tensor["dtype"]]
            self.write_u32(ggml_dtype)
            self.write_u64(tensor["offset"])

    def write_tensor_data(self):
        pad = (GGUF_DEFAULT_ALIGNMENT - (self.f.tell() % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
        self.f.write(b'\x00' * pad)
        self.f.write(self.tensor_data)

    def close(self):
        self.f.close()

def load_model_weights(identifier: str, model_name: str, index_file: str) -> dict:
    print(f"{Colors.HEADER}--- Loading {model_name}: '{identifier}' ---{Colors.ENDC}")
    model_directory = None
    safetensors_files = []

    as_path = Path(identifier)
    if as_path.exists() and as_path.is_dir():
        model_directory = as_path
        print(f"Identifier is a local directory: {model_directory}")
    elif as_path.exists() and as_path.is_file():
        print(f"Identifier is a local file: {as_path}")
        safetensors_files = [as_path]
        model_directory = as_path.parent
    else:
        print(f"Identifier is not a local path. Assuming it's a Hugging Face model ID and attempting download...")
        try:
            model_directory = Path(snapshot_download(repo_id=identifier))
            print(f"Model successfully located at: {model_directory}")
        except Exception as e:
            print(f"{Colors.FAIL}Error: Could not download model '{identifier}'.\nDetails: {e}{Colors.ENDC}", file=sys.stderr)
            sys.exit(1)

    if not safetensors_files:
        index_path = model_directory / index_file
        if index_path.exists():
            print(f"Found index file, loading sharded model: {index_path.name}")
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map")
                if not weight_map:
                    print(f"{Colors.FAIL}Error: 'weight_map' not found in {index_path}{Colors.ENDC}", file=sys.stderr)
                    sys.exit(1)

                unique_files = sorted(list(set(weight_map.values())))
                for filename in unique_files:
                    file_path = model_directory / filename
                    if not file_path.exists():
                        print(f"{Colors.FAIL}Error: Shard file specified in index not found: {file_path}{Colors.ENDC}", file=sys.stderr)
                        sys.exit(1)
                    safetensors_files.append(file_path)
            except Exception as e:
                print(f"{Colors.FAIL}Error processing index file {index_path}: {e}{Colors.ENDC}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Could not find index file. Looking for single safetensors file...")
            possible_files = list(model_directory.glob("*.safetensors"))
            if len(possible_files) == 1:
                safetensors_files.append(possible_files[0])
            elif len(possible_files) > 1:
                print(f"{Colors.FAIL}Error: Found multiple .safetensors files but no index. Cannot determine which to load: {possible_files}{Colors.ENDC}", file=sys.stderr)
                sys.exit(1)

    if not safetensors_files:
        print(f"{Colors.FAIL}Error: No safetensors files were found for '{identifier}'.{Colors.ENDC}", file=sys.stderr)
        sys.exit(1)

    model_tensors = {}
    print(f"Loading tensors from: {[str(p.name) for p in safetensors_files]}")
    for model_file in safetensors_files:
        try:
            with safe_open(model_file, framework="pt", device="cpu") as f:
                for key in tqdm(f.keys(), desc=f"  > Loading {model_file.name}"):
                    model_tensors[key] = f.get_tensor(key)
        except Exception as e:
            print(f"{Colors.FAIL}Error loading {model_file}: {e}{Colors.ENDC}", file=sys.stderr)
            sys.exit(1)

    print(f"{Colors.HEADER}--- Finished loading {model_name}. Found {len(model_tensors)} tensors. ---\n{Colors.ENDC}")
    return model_tensors

def quantize_q8_0(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim != 2:
        raise ValueError("Q8_0 quantization only supports 2D tensors")
    
    QK8_0 = 32
    rows, cols = tensor.shape
    if cols % QK8_0 != 0:
        raise ValueError(f"Last dimension must be divisible by {QK8_0}")

    tensor = tensor.to(torch.float32)
    n_blocks = cols // QK8_0
    
    quantized_data = np.empty((rows, n_blocks, 34), dtype=np.uint8)

    for r in range(rows):
        for i in range(n_blocks):
            block = tensor[r, i*QK8_0:(i+1)*QK8_0]
            
            amax = torch.max(torch.abs(block))
            d = amax / 127.0 if amax != 0 else 0.0
            
            d_f16 = d.to(torch.float16).numpy().tobytes()
            quantized_data[r, i, 0:2] = np.frombuffer(d_f16, dtype=np.uint8)

            qs = torch.round(block / d).clamp(-128, 127).to(torch.int8)
            quantized_data[r, i, 2:] = qs.numpy().view(np.uint8)

    return quantized_data.reshape(rows, n_blocks * 34)

def quantize_q4_k(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim != 2:
        raise ValueError("Q4_K quantization only supports 2D tensors")

    QK_K = 256
    K_SCALE_SIZE = 12
    rows, cols = tensor.shape
    if cols % QK_K != 0:
        raise ValueError(f"Last dimension must be divisible by {QK_K}")

    tensor = tensor.to(torch.float32)
    n_blocks = cols // QK_K
    
    quantized_data = np.empty((rows, n_blocks, 144), dtype=np.uint8)

    for r in range(rows):
        for i in range(n_blocks):
            block = tensor[r, i*QK_K:(i+1)*QK_K]
            
            sub_blocks = block.reshape(16, 16)
            
            p_maxs = sub_blocks.max(dim=1).values
            p_mins = sub_blocks.min(dim=1).values
            
            deltas = (p_maxs - p_mins) / 15.0
            
            if (deltas < 1e-9).all():
                quantized_data[r, i, :] = 0
                continue
            
            deltas[deltas < 1e-9] = 1.0
            inv_deltas = 1.0 / deltas
            
            qs_4bit = torch.round((sub_blocks - p_mins[:, None]) * inv_deltas[:, None]).clamp(0, 15).to(torch.uint8)
            qs_flat = qs_4bit.flatten().numpy()
            
            qs_packed = np.empty(QK_K // 2, dtype=np.uint8)
            for j in range(QK_K // 2):
                qs_packed[j] = qs_flat[2*j] | (qs_flat[2*j + 1] << 4)

            d = deltas.max()
            dmin = p_mins.min()

            inv_d = 1.0 / d
            quantized_scales_6bit = torch.round(deltas * inv_d * 63.0).clamp(0, 63).to(torch.uint8).numpy()
            
            packed_scales = np.zeros(K_SCALE_SIZE, dtype=np.uint8)
            sl = quantized_scales_6bit
            
            for j in range(8):
                packed_scales[j] = (sl[j] & 0x3F) | ((sl[j + 8] & 0x3) << 6)
            for j in range(4):
                packed_scales[j + 8] = ((sl[j * 2 + 8] >> 2) & 0xF) | (((sl[j * 2 + 9] >> 2) & 0xF) << 4)

            current_block_data = quantized_data[r, i]
            current_block_data[0:2] = np.frombuffer(d.to(torch.float16).numpy().tobytes(), dtype=np.uint8)
            current_block_data[2:4] = np.frombuffer(dmin.to(torch.float16).numpy().tobytes(), dtype=np.uint8)
            current_block_data[4:16] = packed_scales
            current_block_data[16:144] = qs_packed

    return quantized_data.reshape(rows, n_blocks * 144)

def main():
    parser = argparse.ArgumentParser(description="Quantize a model to GGUF format.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-id", type=str, help="The Hugging Face model ID to quantize (e.g., 'sesame/csm-1b').")
    group.add_argument("--model-path", type=Path, help="Path to a local model directory or a specific safetensors file.")
    parser.add_argument("--index-file", type=str, default="model.safetensors.index.json", help="The name of the index file for sharded models.")
    parser.add_argument("--output-path", type=Path, required=True, help="Path to save quantized GGUF model.")
    parser.add_argument("--qtype", type=str, default="q8_0", choices=["q8_0", "q4_k"], help="Quantization type.")
    args = parser.parse_args()

    identifier = args.model_id if args.model_id else args.model_path
    model = load_model_weights(str(identifier), "model", args.index_file)

    writer = GGUFWriter(args.output_path, "csm")
    
    writer.add_kv("general.architecture", GGUFValueType.STRING, "csm")
    writer.add_kv("general.quantization_version", GGUFValueType.UINT32, 2) 
    writer.add_kv("csm.embedding_length", GGUFValueType.UINT32, 2048)
    writer.add_kv("csm.backbone.block_count", GGUFValueType.UINT32, 16)
    writer.add_kv("csm.decoder.block_count", GGUFValueType.UINT32, 4)
    writer.add_kv("csm.audio_num_codebooks", GGUFValueType.UINT32, 32)
    writer.add_kv("csm.audio_vocab_size", GGUFValueType.UINT32, 2051)
    writer.add_kv("csm.text_vocab_size", GGUFValueType.UINT32, 128256)

    print("Quantizing and adding tensors...")
    for name, tensor in model.items():
        if name.endswith(".weight") and tensor.ndim == 2 and "norm" not in name and "embeddings" not in name:
            print(f"  Quantizing {name} {tensor.shape} to {args.qtype.upper()}")
            if args.qtype == "q8_0":
                quantized = quantize_q8_0(tensor)
                writer.add_tensor_info(name, quantized, "q8_0", override_shape=tensor.shape)
            elif args.qtype == "q4_k":
                quantized = quantize_q4_k(tensor)
                writer.add_tensor_info(name, quantized, "q4_k", override_shape=tensor.shape)
            else:
                raise ValueError(f"Unknown quantization type {args.qtype}")
        else:
            print(f"  Adding {name} {tensor.shape} as F16")
            tensor_f16 = tensor.to(torch.float16).numpy()
            writer.add_tensor_info(name, tensor_f16, "f16")
            
    print("Writing GGUF file...")
    writer.write_header()
    writer.write_tensor_info()
    writer.write_tensor_data()
    writer.close()
    
    print(f"Successfully quantized model to {args.output_path}")

if __name__ == "__main__":
    main()