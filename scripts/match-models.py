import argparse
import sys
import json
from pathlib import Path
import torch

from huggingface_hub import snapshot_download
from safetensors import safe_open
from tqdm import tqdm

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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


def compare_by_name(weights1: dict, weights2: dict, args):
    """Compares models based on matching tensor names (keys)."""
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())

    common_keys = sorted(list(keys1.intersection(keys2)))
    unique_to_1 = sorted(list(keys1.difference(keys2)))
    unique_to_2 = sorted(list(keys2.difference(keys1)))

    equal_tensors = []
    different_tensors = []

    for key in common_keys:
        t1 = weights1[key]
        t2 = weights2[key]

        if t1.shape != t2.shape:
            different_tensors.append((key, f"Shapes differ - M1: {t1.shape}, M2: {t2.shape}"))
            continue

        if torch.allclose(t1, t2, rtol=args.rtol, atol=args.atol):
             equal_tensors.append((key, f"Shape: {t1.shape}"))
        else:
            diff = torch.abs(t1 - t2).max()
            different_tensors.append((key, f"Values differ - Shape: {t1.shape}, Max Diff: {diff:.4e}"))

    print("="*80)
    print(" " * 25 + f"{Colors.BOLD}MODEL COMPARISON REPORT (BY NAME){Colors.ENDC}")
    print("="*80)
    print(f"{Colors.OKBLUE}Model 1:{Colors.ENDC} {args.model1} ({len(keys1)} total tensors)")
    print(f"{Colors.OKBLUE}Model 2:{Colors.ENDC} {args.model2} ({len(keys2)} total tensors)")
    print(f"{Colors.OKBLUE}Tolerances:{Colors.ENDC} rtol={args.rtol}, atol={args.atol}")
    print("-" * 80)

    print(f"\n{Colors.OKGREEN}✅ Equal Tensors ({len(equal_tensors)} / {len(common_keys)} common tensors){Colors.ENDC}")
    if equal_tensors:
        for key, info in equal_tensors:
            print(f"  - {key:<50} | {info}")
    else:
        print("  None")

    print(f"\n{Colors.FAIL}❌ Different Tensors ({len(different_tensors)} / {len(common_keys)} common tensors){Colors.ENDC}")
    if different_tensors:
        for key, info in different_tensors:
            print(f"  - {key:<50} | {info}")
    else:
        print("  None")

    print(f"\n{Colors.WARNING}1️⃣ Tensors only in Model 1 ({len(unique_to_1)} tensors){Colors.ENDC}")
    if unique_to_1:
        for key in unique_to_1:
            print(f"  - {key:<50} | Shape: {weights1[key].shape}")
    else:
        print("  None")

    print(f"\n{Colors.WARNING}2️⃣ Tensors only in Model 2 ({len(unique_to_2)} tensors){Colors.ENDC}")
    if unique_to_2:
        for key in unique_to_2:
            print(f"  - {key:<50} | Shape: {weights2[key].shape}")
    else:
        print("  None")
    print("-" * 80)


def compare_by_content(weights1: dict, weights2: dict, args):
    print(f"{Colors.OKBLUE}Preparing for content-based comparison...{Colors.ENDC}")
    
    w1_list = list(weights1.items())
    w2_list = list(weights2.items())
    
    matched_pairs = []
    
    w1_matched = [False] * len(w1_list)
    w2_matched = [False] * len(w2_list)

    for i, (name1, t1) in enumerate(tqdm(w1_list, desc="Comparing Tensors")):
        for j, (name2, t2) in enumerate(w2_list):
            if w2_matched[j]:
                continue
            
            if t1.shape == t2.shape:
                if torch.allclose(t1, t2, rtol=args.rtol, atol=args.atol):
                    matched_pairs.append(((name1, t1.shape), (name2, t2.shape)))
                    w1_matched[i] = True
                    w2_matched[j] = True
                    break
    
    unmatched_1 = [item for i, item in enumerate(w1_list) if not w1_matched[i]]
    unmatched_2 = [item for j, item in enumerate(w2_list) if not w2_matched[j]]

    print("\n" + "="*80)
    print(" " * 22 + f"{Colors.BOLD}MODEL COMPARISON REPORT (BY CONTENT){Colors.ENDC}")
    print("="*80)
    print(f"{Colors.OKBLUE}Model 1:{Colors.ENDC} {args.model1} ({len(weights1)} total tensors)")
    print(f"{Colors.OKBLUE}Model 2:{Colors.ENDC} {args.model2} ({len(weights2)} total tensors)")
    print(f"{Colors.OKBLUE}Tolerances:{Colors.ENDC} rtol={args.rtol}, atol={args.atol}")
    print("-" * 80)

    print(f"\n{Colors.OKGREEN}✅ Matched Tensors by Content ({len(matched_pairs)} pairs){Colors.ENDC}")
    if matched_pairs:
        for (name1, shape1), (name2, _) in sorted(matched_pairs, key=lambda x: x[0][0]):
            print(f"  - {Colors.OKBLUE}M1:{Colors.ENDC} {name1:<47} | {Colors.HEADER}Shape: {shape1}{Colors.ENDC}")
            print(f"    {Colors.OKBLUE}M2:{Colors.ENDC} {name2}")
    else:
        print("  None")

    print(f"\n{Colors.WARNING}1️⃣ Unmatched Tensors in Model 1 ({len(unmatched_1)} tensors){Colors.ENDC}")
    if unmatched_1:
        for name, tensor in sorted(unmatched_1, key=lambda x: x[0]):
            print(f"  - {name:<50} | Shape: {tensor.shape}")
    else:
        print("  None")

    print(f"\n{Colors.WARNING}2️⃣ Unmatched Tensors in Model 2 ({len(unmatched_2)} tensors){Colors.ENDC}")
    if unmatched_2:
        for name, tensor in sorted(unmatched_2, key=lambda x: x[0]):
            print(f"  - {name:<50} | Shape: {tensor.shape}")
    else:
        print("  None")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare the weights of two models from Hugging Face Hub or local paths.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model1", type=str, help="Identifier for the first model (e.g., 'org/model-name' or '/path/to/model1').")
    parser.add_argument("model2", type=str, help="Identifier for the second model (e.g., 'org/model-name' or '/path/to/model2').")
    parser.add_argument(
        "--match-by-content",
        action="store_true",
        help="Match tensors by their content (using torch.allclose) instead of their names.\n"
             "This is useful for comparing models with different layer naming conventions\n"
             "or minor floating-point variations."
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for tensor comparison (default: 1e-5)."
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for tensor comparison (default: 1e-4)."
    )
    parser.add_argument("--index-file-m1", default="model.safetensors.index.json", help="Index filename for model 1.")
    parser.add_argument("--index-file-m2", default="model.safetensors.index.json", help="Index filename for model 2.")
    args = parser.parse_args()

    weights1 = load_model_weights(args.model1, "Model 1", args.index_file_m1)
    weights2 = load_model_weights(args.model2, "Model 2", args.index_file_m2)

    if args.match_by_content:
        compare_by_content(weights1, weights2, args)
    else:
        compare_by_name(weights1, weights2, args)


if __name__ == "__main__":
    main()