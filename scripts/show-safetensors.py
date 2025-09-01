import argparse
import sys
from pathlib import Path

from safetensors import safe_open

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the input model.safetensors file.")
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: Input model file not found at {args.model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from: {args.model_path}")

    with safe_open(args.model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"found key: '{key}' with shape: '{tensor.shape}'")

if __name__ == "__main__":
    main()