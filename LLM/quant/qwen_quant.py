"""
Qwen3.5-9B 4-bit Quantization using Unsloth.
Outputs vLLM-compatible AWQ format (default) or GGUF format.

Usage:
    # AWQ (vLLM recommended)
    python qwen_quant.py --model /path/to/Qwen3.5-9B --output_dir ./output

    # GGUF
    python qwen_quant.py --model /path/to/Qwen3.5-9B --output_dir ./output --quant_format gguf

    # vLLM loading
    vllm serve ./output/awq --quantization awq --dtype half
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# ==================== Argument Parsing ====================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B 4-bit quantization with Unsloth (vLLM-compatible output)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--quant_format",
        type=str,
        default="awq",
        choices=["awq", "gguf"],
        help="Quantization output format: awq (vLLM recommended) or gguf (llama.cpp)",
    )
    parser.add_argument(
        "--awq_bits",
        type=int,
        default=4,
        choices=[4],
        help="AWQ quantization bits (currently only 4-bit supported)",
    )
    parser.add_argument(
        "--awq_group_sz",
        type=int,
        default=128,
        help="AWQ group size (128 recommended)",
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=128,
        help="Number of calibration samples for AWQ",
    )
    parser.add_argument(
        "--gguf_method",
        type=str,
        default="q4_k_m",
        choices=["q4_k_m", "q4_k_s", "q5_k_m", "q8_0", "f16"],
        help="GGUF quantization method (only used with --quant_format gguf)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--local_only",
        action="store_true",
        help="Disable HuggingFace Hub network requests (use local model only)",
    )
    parser.add_argument(
        "--calib_data_dir",
        type=str,
        default=None,
        help="Local path to pre-downloaded calibration dataset (saved via --download_calib_data)",
    )
    parser.add_argument(
        "--download_calib_data",
        action="store_true",
        help="Download wikitext calibration dataset to --calib_data_dir and exit",
    )
    return parser.parse_args()


# ==================== Model Loading ====================


def load_model_and_tokenizer(args):
    """Load model with Unsloth optimization."""
    from unsloth import FastLanguageModel

    model_path = args.model

    if args.local_only:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        # Verify local path exists
        if not Path(model_path).exists():
            print(f"[ERROR] Local model path not found: {model_path}")
            print("  Remove --local_only to download from HuggingFace Hub.")
            sys.exit(1)
        print(f"[INFO] Local-only mode: loading from {model_path}")

    print(f"[INFO] Loading model: {model_path}")
    print(f"[INFO] Max sequence length: {args.max_seq_len}")

    # For AWQ: load in 16-bit (AWQ needs original weights for calibration)
    # For GGUF: Unsloth handles quantization natively
    load_in_4bit = args.quant_format == "gguf"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=args.max_seq_len,
        dtype=None,          # auto-detect bf16/fp16
        load_in_4bit=load_in_4bit,
    )

    print(f"[INFO] Model loaded. Device: {next(model.parameters()).device}")
    print(f"[INFO] Model dtype: {next(model.parameters()).dtype}")

    return model, tokenizer


# ==================== AWQ Quantization (via llm-compressor) ====================


def download_calib_data(save_dir: str):
    """Download wikitext-2 to a local directory for offline use."""
    from datasets import load_dataset

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"[CALIB] Downloading wikitext-2-raw-v1 → {save_dir} ...")
    from datasets import load_dataset  # type: ignore[import]
    # Use cache_dir= directly — env var may be ignored if datasets was already imported
    load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", cache_dir=save_dir)
    total = sum(f.stat().st_size for f in Path(save_dir).rglob("*") if f.is_file()) / (1024 ** 2)
    print(f"[CALIB] Done ({total:.1f} MB). Use with: --calib_data_dir {save_dir}")



def quantize_awq(args):
    """
    W4A16 group-wise 4-bit quantization using llm-compressor (vLLM project).
    autoawq is officially deprecated; llm-compressor is the vLLM-recommended
    successor for producing AWQ-compatible models.

    Output is saved in HuggingFace safetensors format and can be loaded by
    vLLM directly without --quantization flag (compressed-tensors format).
    """
    # Compatibility shim: TORCH_INIT_FUNCTIONS was removed in transformers>=4.52.
    # llm-compressor imports it at module level; an empty dict is safe here
    # since no weight init is performed when loading pre-trained weights.
    import importlib
    _mu = importlib.import_module("transformers.modeling_utils")
    if not hasattr(_mu, "TORCH_INIT_FUNCTIONS"):
        setattr(_mu, "TORCH_INIT_FUNCTIONS", {})

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ImportError as e:
        if "TORCH_INIT_FUNCTIONS" in str(e):
            print("[ERROR] llm-compressor is incompatible with the installed transformers version.")
            print("  The TORCH_INIT_FUNCTIONS shim did not help. Try pinning transformers:")
            print("  pip install 'transformers==4.51.3'")
        else:
            print("[ERROR] llm-compressor not installed.")
            print("  Run: pip install llmcompressor")
        sys.exit(1)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(args.output_dir) / "awq"
    output_path.mkdir(parents=True, exist_ok=True)

    # W4A16 with group quantization — equivalent to AWQ scheme
    # group_size=128 is the AWQ standard; matches --awq_group_sz
    scheme = f"W{args.awq_bits}A16"
    print(f"\n[AWQ] Starting {scheme} quantization via llm-compressor")
    print(f"[AWQ] Group size: {args.awq_group_sz}")
    print(f"[AWQ] Calibration samples: {args.calib_samples}")
    print(f"[AWQ] Output: {output_path}")

    print("[AWQ] Loading model and tokenizer...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    except ValueError as e:
        if "does not recognize this architecture" in str(e) or "model_type" in str(e):
            import json
            cfg = json.loads((Path(args.model) / "config.json").read_text())
            model_type = cfg.get("model_type", "unknown")
            import transformers
            print(f"\n[ERROR] transformers {transformers.__version__} does not support model_type='{model_type}'.")
            print("  Fix: pip install --upgrade transformers")
            print("  Or:  pip install git+https://github.com/huggingface/transformers.git")
            sys.exit(1)
        raise
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=["lm_head"],
    )

    print("[AWQ] Running quantization (this may take 10-30 minutes)...")
    t0 = time.time()

    # If a local calib cache dir is given, redirect the datasets cache at runtime.
    # We patch datasets.config directly because env vars may be ignored if the
    # datasets module was already imported (e.g. by llm-compressor).
    if args.calib_data_dir:
        import datasets.config as _ds_cfg  # type: ignore[import]
        _ds_cfg.HF_DATASETS_CACHE = Path(args.calib_data_dir).resolve()
        print(f"[AWQ] Calibration dataset cache: {args.calib_data_dir}")
    elif args.local_only:
        print("[AWQ] WARN: --local_only set but no --calib_data_dir provided.")
        print("[AWQ]       Pre-download first:  python qwen_quant.py "
              "--download_calib_data --calib_data_dir ./calib_data")
        sys.exit(1)

    oneshot(
        model=model,
        recipe=recipe,
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.calib_samples,
    )
    elapsed = time.time() - t0
    print(f"[AWQ] Quantization done in {elapsed:.1f}s")

    print(f"[AWQ] Saving to {output_path}...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    safetensors = list(output_path.glob("*.safetensors"))
    if safetensors:
        total_size = sum(f.stat().st_size for f in safetensors) / (1024 ** 3)
        print(f"[AWQ] Model size: {total_size:.2f} GB")

    print(f"\n[AWQ] Done! Quantized model saved to: {output_path}")
    print(f"[AWQ] Load with vLLM (compressed-tensors format, no extra flag needed):")
    print(f"      vllm serve {output_path}")

    return str(output_path)


# ==================== GGUF Quantization ====================


def quantize_gguf(args, model, tokenizer):
    """
    GGUF quantization using Unsloth's native export.
    Supports q4_k_m, q5_k_m, q8_0, f16 etc.
    """
    output_path = Path(args.output_dir) / "gguf"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[GGUF] Starting GGUF quantization: {args.gguf_method}")
    print(f"[GGUF] Output: {output_path}")

    t0 = time.time()
    model.save_pretrained_gguf(
        str(output_path),
        tokenizer,
        quantization_method=args.gguf_method,
    )
    elapsed = time.time() - t0
    print(f"[GGUF] Export done in {elapsed:.1f}s")

    # Find the output file and report size
    gguf_files = list(output_path.glob("*.gguf"))
    if gguf_files:
        gguf_file = gguf_files[0]
        size_gb = gguf_file.stat().st_size / (1024 ** 3)
        print(f"[GGUF] File: {gguf_file.name} ({size_gb:.2f} GB)")
        print(f"\n[GGUF] Done! GGUF model saved to: {output_path}")
        print(f"[GGUF] Load with llama.cpp:")
        print(f"       ./llama-cli -m {gguf_file}")
        print(f"[GGUF] Load with vLLM (limited support):")
        print(f"       vllm serve {gguf_file} --quantization gguf")
        return str(gguf_file)
    else:
        print(f"[GGUF] Done! Output: {output_path}")
        return str(output_path)


# ==================== Main ====================


def main():
    args = parse_args()

    print("=" * 60)
    print("  Qwen 4-bit Quantization Pipeline (Unsloth)")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Format:      {args.quant_format.upper()}")
    print(f"  Output dir:  {args.output_dir}")
    if args.quant_format == "awq":
        print(f"  AWQ bits:    {args.awq_bits}")
        print(f"  Group size:  {args.awq_group_sz}")
        print(f"  Calib samples: {args.calib_samples}")
    else:
        print(f"  GGUF method: {args.gguf_method}")
    print("=" * 60)

    # Handle one-shot download mode
    if args.download_calib_data:
        if not args.calib_data_dir:
            print("[ERROR] --download_calib_data requires --calib_data_dir <path>")
            sys.exit(1)
        download_calib_data(args.calib_data_dir)
        sys.exit(0)

    # Check GPU
    if not torch.cuda.is_available():
        print("[WARN] No GPU detected. Quantization will be very slow on CPU.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[INFO] GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Check transformers supports this model's architecture
    model_path = Path(args.model)
    if model_path.exists():
        import json
        import transformers
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        config_file = model_path / "config.json"
        if config_file.exists():
            cfg = json.loads(config_file.read_text())
            model_type = cfg.get("model_type", "")
            if model_type and model_type not in CONFIG_MAPPING:
                print(f"\n[WARN] transformers {transformers.__version__} does not natively support model_type='{model_type}'.")
                print("[WARN] This may cause loading errors. If so, upgrade transformers:")
                print("       pip install --upgrade transformers")
                print("       pip install git+https://github.com/huggingface/transformers.git")
                print("[WARN] Continuing anyway (trust_remote_code=True may still work)...\n")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.quant_format == "awq":
        # AWQ: use autoawq directly (loads model internally)
        # Still call load_model_and_tokenizer to get tokenizer + validate path
        print("\n[INFO] AWQ mode: model will be loaded by autoawq for calibration")
        if args.local_only:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            if not Path(args.model).exists():
                print(f"[ERROR] Local model path not found: {args.model}")
                sys.exit(1)

        quantize_awq(args)

    elif args.quant_format == "gguf":
        # GGUF: use Unsloth's native export
        model, tokenizer = load_model_and_tokenizer(args)
        quantize_gguf(args, model, tokenizer)

    print("\n[INFO] Quantization complete.")


if __name__ == "__main__":
    main()
