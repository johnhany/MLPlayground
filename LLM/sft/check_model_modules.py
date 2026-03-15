"""
Check what modules are available in the model for LoRA.
Run this to debug which modules should be in target_modules.
"""

import sys
import os
from pathlib import Path

def check_model_modules(model_path: str, local_only: bool = True):
    """Check available modules in the model."""

    if local_only:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Error: Unsloth not installed. Install with: pip install unsloth")
        sys.exit(1)

    print(f"Loading model from: {model_path}")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            local_files_only=local_only,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("\n" + "="*80)
    print("MODEL STRUCTURE - All modules:")
    print("="*80)

    # Find all linear/attention modules
    linear_modules = set()
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if "Linear" in module_type or "Attention" in module_type:
            # Extract the last part of the module name
            parts = name.split(".")
            if len(parts) > 0:
                last_part = parts[-1]
                if last_part not in linear_modules and any(
                    kw in last_part for kw in ["proj", "gate", "up", "down", "attn", "attention"]
                ):
                    linear_modules.add(last_part)
                    print(f"  {name:80} -> {last_part:30} ({module_type})")

    print("\n" + "="*80)
    print("RECOMMENDED target_modules for LoRA:")
    print("="*80)
    print("\nUnique module suffixes found:")
    for module in sorted(linear_modules):
        print(f"  - {module}")

    # Try common module names
    print("\n" + "="*80)
    print("CHECKING COMMON MODULE NAMES:")
    print("="*80)

    common_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "dense", "intermediate",
    ]

    model_dict = dict(model.named_modules())
    found_modules = []

    for module_name in common_modules:
        found = False
        for name in model_dict.keys():
            if module_name in name:
                found = True
                break
        status = "✓ FOUND" if found else "✗ NOT FOUND"
        print(f"  {module_name:20} {status}")
        if found:
            found_modules.append(module_name)

    print("\n" + "="*80)
    print("PEFT RECOMMENDATION:")
    print("="*80)
    print(f"\ntarget_modules = {found_modules}")

    print("\n" + "="*80)
    print("MODEL INFO:")
    print("="*80)
    print(f"Model type: {model.config.model_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model config: {model.config}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_model_modules.py <model_path> [--online]")
        print("Example: python check_model_modules.py /home/john/models/qwen3-4b")
        sys.exit(1)

    model_path = sys.argv[1]
    local_only = "--online" not in sys.argv

    check_model_modules(model_path, local_only=local_only)
