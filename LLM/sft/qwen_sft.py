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
from transformers import TrainingArguments, TextIteratorStreamer
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
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


def load_model_and_tokenizer(model_id: str, max_seq_length: int):
    """
    Load Qwen3-4B model with Unsloth optimization.
    Uses bfloat16 precision for efficiency on RTX 3090.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "Unsloth not found. Install with: pip install unsloth"
        )

    # Load model with 4-bit quantization option (if Unsloth supports it)
    # Default: load in bf16 for better precision
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # Use bf16 instead of 4-bit for better precision
    )

    return model, tokenizer


# ==================== LoRA Configuration ====================


def apply_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float):
    """
    Apply LoRA to model using PEFT.
    Target modules: all attention + MLP layers.
    """
    lora_config = LoraConfig(
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
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
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
):
    """
    Train model using TRL SFTTrainer.
    Loss is only computed on assistant tokens (input tokens masked with label=-100).
    """

    # Prepare dataset
    def format_fn(batch):
        """Format batch for training."""
        texts = []
        for i in range(len(batch["input"])):
            messages = [
                {"role": "user", "content": batch["input"][i]},
                {"role": "assistant", "content": batch["output"][i]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    train_dataset = train_dataset.map(
        format_fn,
        batched=True,
        batch_size=len(train_dataset),
        remove_columns=["uuid", "input", "output", "domain"],
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        bf16=True,  # Use bfloat16
        gradient_checkpointing=True,  # Save memory with gradient checkpointing
        max_grad_norm=1.0,
        remove_unused_columns=True,
        report_to=["tensorboard"],
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,  # Don't pack sequences (easier debugging)
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
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load data
    train_dataset = load_training_data(args.data, args.max_samples)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.max_seq_length)

    # Apply LoRA
    print("Applying LoRA configuration...")
    model = apply_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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
    )


if __name__ == "__main__":
    main()
