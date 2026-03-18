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


# ==================== AWQ Quantization ====================


def quantize_awq(args):
    """
    AWQ 4-bit quantization using autoawq.
    AWQ requires loading the model independently (not via Unsloth's wrapper),
    so we use AutoAWQForCausalLM directly on the model path.
    Tokenizer is loaded after the model to reuse the properly parsed config object,
    working around a transformers bug where AutoConfig returns a raw dict for
    custom architectures.
    """
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("[ERROR] autoawq not installed. Run: pip install autoawq")
        sys.exit(1)
    from transformers import AutoTokenizer

    output_path = Path(args.output_dir) / "awq"
    output_path.mkdir(parents=True, exist_ok=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": args.awq_group_sz,
        "w_bit": args.awq_bits,
        "version": "GEMM",
    }

    print(f"\n[AWQ] Starting 4-bit AWQ quantization")
    print(f"[AWQ] Config: {quant_config}")
    print(f"[AWQ] Calibration samples: {args.calib_samples}")
    print(f"[AWQ] Output: {output_path}")

    # Load model via AutoAWQ (uses transformers under the hood)
    print("[AWQ] Loading model for calibration...")
    awq_model = AutoAWQForCausalLM.from_pretrained(
        args.model,
        safetensors=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer using awq_model.config (a proper PretrainedConfig object),
    # bypassing the transformers bug that returns a raw dict for custom archs.
    print("[AWQ] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        config=awq_model.config,
        trust_remote_code=True,
    )

    # Run AWQ quantization with calibration
    print("[AWQ] Running quantization (this may take 10-30 minutes)...")
    t0 = time.time()
    awq_model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data="wikitext",
        split="train",
        text_column="text",
        n_samples=args.calib_samples,
        max_seq_len=args.max_seq_len,
    )
    elapsed = time.time() - t0
    print(f"[AWQ] Quantization done in {elapsed:.1f}s")

    # Save quantized model
    print(f"[AWQ] Saving to {output_path}...")
    awq_model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Print model size
    gguf_files = list(output_path.glob("*.safetensors"))
    if gguf_files:
        total_size = sum(f.stat().st_size for f in gguf_files) / (1024 ** 3)
        print(f"[AWQ] Model size: {total_size:.2f} GB")

    print(f"\n[AWQ] Done! Quantized model saved to: {output_path}")
    print(f"[AWQ] Load with vLLM:")
    print(f"      vllm serve {output_path} --quantization awq --dtype half")

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

    # Check GPU
    if not torch.cuda.is_available():
        print("[WARN] No GPU detected. Quantization will be very slow on CPU.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[INFO] GPU: {gpu_name} ({gpu_mem:.1f} GB)")

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
