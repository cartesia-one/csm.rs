import argparse
import sys
import json
from pathlib import Path

from huggingface_hub import snapshot_download, HfFolder
from safetensors import safe_open

def main():
    parser = argparse.ArgumentParser(
        description="Load and inspect safetensors files from a local path or the Hugging Face Hub."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model-id", 
        type=str, 
        help="The Hugging Face model ID to download (e.g., 'meta-llama/Llama-2-7b-hf')."
    )
    group.add_argument(
        "--model-path", 
        type=Path, 
        help="Path to a local model directory or a specific safetensors file within it."
    )
    args = parser.parse_args()

    model_directory = None

    if args.model_id:
        print(f"Attempting to download model '{args.model_id}' from Hugging Face Hub...")
        try:
            model_directory = Path(snapshot_download(repo_id=args.model_id))
            print(f"Model successfully located at: {model_directory}")
        except Exception as e:
            print(f"Error: Could not download model '{args.model_id}'.\nDetails: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.model_path:
        print(f"Using local model path: {args.model_path}")
        input_path = args.model_path
        if not input_path.exists():
            print(f"Error: Input path not found at {input_path}", file=sys.stderr)
            sys.exit(1)
        if input_path.is_dir():
            model_directory = input_path
        else:
            model_directory = input_path.parent

    safetensors_files = []
    index_path = model_directory / "model.safetensors.index.json"

    if index_path.exists():
        print(f"Found index file: {index_path}")
        print("Attempting to load sharded weights.")
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            
            weight_map = index_data.get("weight_map")
            if not weight_map:
                print(f"Error: 'weight_map' not found in {index_path}", file=sys.stderr)
                sys.exit(1)

            unique_files = sorted(list(set(weight_map.values())))
            
            for filename in unique_files:
                file_path = model_directory / filename
                if not file_path.exists():
                    print(f"Error: Shard file specified in index not found: {file_path}", file=sys.stderr)
                    sys.exit(1)
                safetensors_files.append(file_path)

        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON from {index_path}", file=sys.stderr)
            sys.exit(1)
            
    else:
        print(f"Could not find '{index_path}'.")
        print("Falling back to single-file model.")
        
        single_file_path = model_directory / "model.safetensors"
        if single_file_path.exists():
             safetensors_files.append(single_file_path)
        else:
            print(f"Could not find 'model.safetensors'. Searching for any '*.safetensors' file...")
            possible_files = list(model_directory.glob("*.safetensors"))
            if len(possible_files) == 1:
                print(f"Found a single alternative file: {possible_files[0]}")
                safetensors_files.append(possible_files[0])
            elif len(possible_files) > 1:
                print(f"Error: Found multiple .safetensors files but no index. Cannot determine which to load: {possible_files}", file=sys.stderr)
                sys.exit(1)

    if not safetensors_files:
        print(f"Error: No safetensors files were found in {model_directory}.", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading model from: {[str(p.name) for p in safetensors_files]}")
    total_keys = 0
    for model_file in safetensors_files:
        print(f"\n--- Loading shard: {model_file.name} ---")
        try:
            with safe_open(model_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    print(f"  found key: '{key}' with shape: {tensor.shape}")
                    total_keys += 1
        except Exception as e:
            print(f"Error loading {model_file}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\nFinished. Found a total of {total_keys} keys across {len(safetensors_files)} file(s).")

if __name__ == "__main__":
    main()