"""
Qwen3-4B SFT (Supervised Fine-Tuning) on RTX 3090 using LoRA.
Memory-efficient training with Unsloth optimization.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
import warnings

import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers.trainer_callback import TrainerCallback
from datasets import Dataset, load_dataset

warnings.filterwarnings("ignore")

# ==================== Argument Parsing ====================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-4B SFT with LoRA on RTX 3090"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data (JSONL or JSON format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./qwen3_sft_output",
        help="Output directory for checkpoints and final model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for training",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit training to N samples (for debugging)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (use with large datasets)",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log metrics every N steps",
    )
    parser.add_argument(
        "--local_only",
        action="store_true",
        help="Only load local model files, disable HuggingFace hub connection",
    )

    return parser.parse_args()


# ==================== Data Processing ====================


def load_training_data(data_path: str, max_samples: Optional[int] = None) -> Dataset:
    """
    Load training data from JSONL/JSON file.
    Expected format:
    {
        "uuid": "xxx",
        "input": "user question",
        "output": "assistant answer",
        "domain": "math",
        "meta": {...}
    }
    """
    data_path = Path(data_path)

    # Load data
    if data_path.suffix == ".jsonl":
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    elif data_path.suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    else:
        raise ValueError(f"Unsupported format: {data_path.suffix}")

    # Limit samples for debugging
    if max_samples:
        data = data[:max_samples]

    print(f"Loaded {len(data)} samples from {data_path}")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict({
        "uuid": [d.get("uuid", "") for d in data],
        "input": [d.get("input", "") for d in data],
        "output": [d.get("output", "") for d in data],
        "domain": [d.get("domain", "") for d in data],
    })

    return dataset


def format_chat_prompt(example, tokenizer):
    """
    Format data into Qwen3 chat template.
    Only the assistant output is used for loss calculation.
    """
    # Qwen3 expects messages in this format
    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


# ==================== Model Loading ====================


def load_model_and_tokenizer(model_id: str, max_seq_length: int, local_only: bool = False):
    """
    Load Qwen3-4B model with Unsloth optimization.
    Uses bfloat16 precision for efficiency on RTX 3090.

    Args:
        model_id: HuggingFace model ID or local path to model directory
        max_seq_length: Maximum sequence length for training
        local_only: If True, only load from local cache/path, disable HF hub
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "Unsloth not found. Install with: pip install unsloth"
        )

    # If local_only is True, validate local path exists
    if local_only:
        model_path = Path(model_id)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Local model path not found: {model_id}\n"
                f"Please ensure the model directory exists with config.json and model files."
            )
        print(f"Loading model from local path: {model_id}")

        # Disable HuggingFace Hub connection
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        print(f"Loading model: {model_id} (HF Hub enabled)")

    # Load model with 4-bit quantization option (if Unsloth supports it)
    # Default: load in bf16 for better precision
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=False,  # Use bf16 instead of 4-bit for better precision
            local_files_only=local_only,  # Only use cached/local files
        )
    except Exception as e:
        if local_only:
            print(f"\n❌ Failed to load local model from: {model_id}")
            print(f"Error: {str(e)}")
            print("\n💡 Tips:")
            print("  1. Check if the model directory exists")
            print("  2. Verify it contains: config.json, model files (*.bin or *.safetensors)")
            print("  3. Model structure should be like: /path/to/Qwen3-4B/")
            print("     ├── config.json")
            print("     ├── model.safetensors (or *.bin)")
            print("     ├── tokenizer.json")
            print("     └── tokenizer_config.json")
        raise

    return model, tokenizer


# ==================== LoRA Configuration ====================


def apply_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float, seed: int = 42):
    """
    Apply LoRA using Unsloth's native FastLanguageModel.get_peft_model().
    This is required when the base model was loaded with Unsloth, as Unsloth
    patches the training loop and expects its own LoRA setup.
    Target modules verified for Qwen3-4B.
    """
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )

    model.print_trainable_parameters()

    return model


# ==================== Training ====================


class LoggingCallback(TrainerCallback):
    """Custom callback for detailed logging."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            print(
                f"Step {state.global_step}: "
                f"loss={state.log_history[-1].get('loss', 'N/A'):.4f}"
            )


def train(
    model,
    tokenizer,
    train_dataset: Dataset,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.05,
    max_seq_length: int = 2048,
    seed: int = 42,
    save_steps: int = 500,
    save_total_limit: int = 3,
    logging_steps: int = 10,
):
    """
    Train model using standard Transformer Trainer.
    Loss is only computed on assistant tokens (input tokens masked with label=-100).
    """

    # Prepare dataset - format text and tokenize
    def format_and_tokenize(example):
        """Format into chat template and tokenize."""
        messages = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

        # Mask loss for input (user) tokens
        # Find where assistant response starts
        user_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["input"]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        user_tokens = tokenizer(user_text, truncation=True, max_length=max_seq_length)
        user_token_count = len(user_tokens["input_ids"])

        # Create labels: -100 for prompt tokens, real labels for response
        labels = tokenized["input_ids"].copy()
        labels[:user_token_count] = [-100] * user_token_count

        tokenized["labels"] = labels
        return tokenized

    print("Tokenizing dataset...")
    train_dataset = train_dataset.map(
        format_and_tokenize,
        remove_columns=["uuid", "input", "output", "domain"],
        desc="Tokenizing",
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        optim="paged_adamw_8bit",
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_strategy="steps",       # Save by steps, not epoch (safer for large datasets)
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        bf16=True,
        gradient_checkpointing=False,  # Handled by Unsloth's use_gradient_checkpointing="unsloth"
        max_grad_norm=1.0,
        dataloader_num_workers=4,      # Speed up data loading
    )

    # Create trainer with data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not using masked language modeling, this is causal LM
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[LoggingCallback()],
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final adapter
    print(f"\nSaving LoRA adapter to {output_dir}...")
    model.save_pretrained(f"{output_dir}/final_adapter")
    tokenizer.save_pretrained(f"{output_dir}/final_adapter")

    print("✓ Training completed!")
    print(f"Adapter saved to: {output_dir}/final_adapter")


# ==================== Main ====================


def main():
    args = parse_args()

    print("=" * 60)
    print("Qwen3-4B SFT (LoRA) on RTX 3090")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} (grad accum: {args.grad_accum})")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    if args.local_only:
        print("📁 Mode: LOCAL ONLY (HuggingFace Hub disabled)")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load data
    train_dataset = load_training_data(args.data, args.max_samples)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, args.max_seq_length, local_only=args.local_only
    )

    # Apply LoRA using Unsloth's native interface
    print("Applying LoRA configuration...")
    model = apply_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Train
    train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()
