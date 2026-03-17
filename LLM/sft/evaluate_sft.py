"""
Evaluate SFT model by comparing base model vs fine-tuned model.

Generates:
- HTML/Markdown comparison reports (side-by-side outputs)
- Automatic metrics (BLEU, ROUGE-L, Perplexity)
- Detailed JSON results for further analysis
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Dict, List
import warnings

import torch
from peft import PeftModel
from tqdm import tqdm

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT model")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to base model (Qwen/Qwen3-4B or local path)",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter (e.g., ./qwen3_sft_output/final_adapter)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data (JSONL format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (for debugging)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--local_only",
        action="store_true",
        help="Only load local models, disable HuggingFace Hub",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    return parser.parse_args()


def load_models(base_model_path: str, adapter_path: str, local_only: bool = False):
    """Load base model and fine-tuned model with LoRA adapter."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from qwen_sft import load_model_and_tokenizer

    print("Loading base model...")
    base_model, tokenizer = load_model_and_tokenizer(
        base_model_path, max_seq_length=2048, local_only=local_only
    )

    print("Loading fine-tuned model (base + LoRA)...")
    finetuned_model, _ = load_model_and_tokenizer(
        base_model_path, max_seq_length=2048, local_only=local_only
    )

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    finetuned_model = PeftModel.from_pretrained(finetuned_model, adapter_path)

    return base_model, finetuned_model, tokenizer


def load_test_data(data_path: str, max_samples: Optional[int] = None) -> tuple:
    """Load test data from JSONL. Supports two formats:

    Format 1 (SFT training data):
    {"uuid": "...", "input": "...", "output": "...", "domain": "..."}

    Format 2 (AIME2025):
    {"question": "...", "answer": "..."}
    """
    import json

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    if max_samples:
        data = data[:max_samples]

    # Detect format
    if data and "question" in data[0]:
        data_format = "aime"
        print(f"Detected AIME2025 format")
    elif data and "input" in data[0]:
        data_format = "sft"
        print(f"Detected SFT training format")
    else:
        raise ValueError("Unknown data format. Expected 'question'/'answer' or 'input'/'output'")

    print(f"Loaded {len(data)} test samples ({data_format} format)")
    return data, data_format


def generate_response(
    model, tokenizer, input_text: str, max_new_tokens: int = 512, device: str = "cuda"
) -> str:
    """Generate response from model."""
    messages = [{"role": "user", "content": input_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response (remove prompt)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    return response


def compute_metrics(
    base_outputs: List[str],
    finetuned_outputs: List[str],
    references: List[str],
) -> Dict:
    """Compute BLEU, ROUGE-L, and other metrics."""
    try:
        from sacrebleu import corpus_bleu
    except ImportError:
        print("Warning: sacrebleu not installed, skipping BLEU calculation")
        corpus_bleu = None

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Warning: rouge_score not installed, skipping ROUGE calculation")
        rouge_scorer = None

    metrics = {
        "base_model": {},
        "finetuned_model": {},
        "improvement": {},
    }

    # BLEU
    if corpus_bleu:
        try:
            bleu_base = corpus_bleu(base_outputs, [references]).score
            bleu_ft = corpus_bleu(finetuned_outputs, [references]).score
            metrics["base_model"]["bleu"] = round(bleu_base, 2)
            metrics["finetuned_model"]["bleu"] = round(bleu_ft, 2)
            metrics["improvement"]["bleu"] = f"+{round(bleu_ft - bleu_base, 2)}"
        except Exception as e:
            print(f"Warning: BLEU calculation failed: {e}")

    # ROUGE-L
    if rouge_scorer:
        try:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
            rouge_base = [
                scorer.score(ref, pred)["rougeL"].fmeasure
                for ref, pred in zip(references, base_outputs)
            ]
            rouge_ft = [
                scorer.score(ref, pred)["rougeL"].fmeasure
                for ref, pred in zip(references, finetuned_outputs)
            ]
            avg_rouge_base = sum(rouge_base) / len(rouge_base)
            avg_rouge_ft = sum(rouge_ft) / len(rouge_ft)
            metrics["base_model"]["rouge_l"] = round(avg_rouge_base, 4)
            metrics["finetuned_model"]["rouge_l"] = round(avg_rouge_ft, 4)
            metrics["improvement"]["rouge_l"] = f"+{round(avg_rouge_ft - avg_rouge_base, 4)}"
        except Exception as e:
            print(f"Warning: ROUGE calculation failed: {e}")

    return metrics


def generate_html_report(
    test_data: List[Dict],
    base_outputs: List[str],
    finetuned_outputs: List[str],
    output_path: str,
    data_format: str = "sft",
):
    """Generate HTML comparison report. Supports SFT and AIME formats."""
    col_header = "Question" if data_format == "aime" else "Input"

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>SFT Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .base {{ background-color: #ffe6e6; }}
            .finetuned {{ background-color: #e6f3ff; }}
            h1 {{ color: #333; }}
            .sample-num {{ font-weight: bold; color: #666; }}
            .question {{ font-style: italic; color: #555; }}
            .answer {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>SFT Model Evaluation Report</h1>
        <p>Comparing Base Model vs Fine-tuned Model</p>
        <p>Format: {data_format.upper()}</p>
        <table>
            <tr>
                <th>#</th>
                <th>{col_header}</th>
                <th>Reference</th>
                <th class="base">Base Model</th>
                <th class="finetuned">Fine-tuned Model</th>
            </tr>
    """

    for i, (sample, base_out, ft_out) in enumerate(
        zip(test_data, base_outputs, finetuned_outputs), 1
    ):
        if data_format == "aime":
            input_text = sample.get("question", "")
            reference = sample.get("answer", "")
        else:
            input_text = sample.get("input", "")
            reference = sample.get("output", "")

        html += f"""
            <tr>
                <td class="sample-num">{i}</td>
                <td>{input_text}</td>
                <td>{reference}</td>
                <td class="base">{base_out}</td>
                <td class="finetuned">{ft_out}</td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML report saved to {output_path}")


def generate_markdown_report(
    test_data: List[Dict],
    base_outputs: List[str],
    finetuned_outputs: List[str],
    output_path: str,
    data_format: str = "sft",
):
    """Generate Markdown comparison report. Supports SFT and AIME formats."""
    md = "# SFT Model Evaluation Report\n\n"
    md += "Comparing Base Model vs Fine-tuned Model\n\n"
    md += f"**Format**: {data_format.upper()}\n\n"

    for i, (sample, base_out, ft_out) in enumerate(
        zip(test_data, base_outputs, finetuned_outputs), 1
    ):
        if data_format == "aime":
            input_text = sample.get("question", "")
            reference = sample.get("answer", "")
            md += f"## Sample {i}\n\n"
            md += f"**Question**: {input_text}\n\n"
            md += f"**Reference Answer**: {reference}\n\n"
        else:
            input_text = sample.get("input", "")
            reference = sample.get("output", "")
            md += f"## Sample {i}\n\n"
            md += f"**Input**: {input_text}\n\n"
            md += f"**Reference Output**: {reference}\n\n"

        md += f"**Base Model Output**:\n> {base_out}\n\n"
        md += f"**Fine-tuned Model Output**:\n> {ft_out}\n\n"
        md += "---\n\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Markdown report saved to {output_path}")


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("\n" + "=" * 60)
    print("Loading Models")
    print("=" * 60)
    base_model, finetuned_model, tokenizer = load_models(
        args.base_model, args.adapter_path, local_only=args.local_only
    )

    # Load test data
    print("\n" + "=" * 60)
    print("Loading Test Data")
    print("=" * 60)
    test_data, data_format = load_test_data(args.test_data, args.max_samples)

    # Generate responses
    print("\n" + "=" * 60)
    print("Generating Responses")
    print("=" * 60)

    base_outputs = []
    finetuned_outputs = []
    references = []

    for sample in tqdm(test_data, desc="Evaluating"):
        if data_format == "aime":
            # AIME2025 format: question -> answer
            input_text = sample.get("question", "")
            reference = sample.get("answer", "")
        else:
            # SFT format: input -> output
            input_text = sample.get("input", "")
            reference = sample.get("output", "")

        base_out = generate_response(
            base_model,
            tokenizer,
            input_text,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        ft_out = generate_response(
            finetuned_model,
            tokenizer,
            input_text,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )

        base_outputs.append(base_out)
        finetuned_outputs.append(ft_out)
        references.append(reference)

    # Compute metrics
    print("\n" + "=" * 60)
    print("Computing Metrics")
    print("=" * 60)
    metrics = compute_metrics(base_outputs, finetuned_outputs, references)

    # Generate reports
    print("\n" + "=" * 60)
    print("Generating Reports")
    print("=" * 60)

    html_path = os.path.join(args.output_dir, "comparison.html")
    md_path = os.path.join(args.output_dir, "comparison.md")
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    detailed_path = os.path.join(args.output_dir, "detailed_results.json")

    generate_html_report(test_data, base_outputs, finetuned_outputs, html_path, data_format)
    generate_markdown_report(test_data, base_outputs, finetuned_outputs, md_path, data_format)

    # Save metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {metrics_path}")

    # Save detailed results
    detailed_results = []
    for i, (sample, base_out, ft_out, ref) in enumerate(
        zip(test_data, base_outputs, finetuned_outputs, references), 1
    ):
        if data_format == "aime":
            result = {
                "sample_id": i,
                "question": sample.get("question", ""),
                "reference_answer": ref,
                "base_model_answer": base_out,
                "finetuned_model_answer": ft_out,
            }
        else:
            result = {
                "sample_id": i,
                "input": sample.get("input", ""),
                "reference_output": ref,
                "base_model_output": base_out,
                "finetuned_model_output": ft_out,
            }
        detailed_results.append(result)

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {detailed_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total samples evaluated: {len(test_data)}")
    print(f"\nMetrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
