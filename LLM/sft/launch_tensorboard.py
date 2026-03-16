#!/usr/bin/env python3
"""
Launch TensorBoard to monitor SFT training metrics.

Usage:
    python launch_tensorboard.py [output_dir]

Example:
    python launch_tensorboard.py ./qwen3_sft_output
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    log_dir = os.path.join(output_dir, "logs")

    if not os.path.exists(log_dir):
        print(f"❌ TensorBoard log directory not found: {log_dir}")
        print("Make sure training has started and created logs.")
        sys.exit(1)

    print("🚀 Launching TensorBoard...")
    print(f"📊 Log directory: {log_dir}")
    print("🌐 Open browser: http://localhost:6006")
    print("")
    print("Available metrics:")
    print("  - Training Loss")
    print("  - Learning Rate")
    print("  - Gradient Norm")
    print("  - Training Speed (samples/sec)")
    print("")
    print("Press Ctrl+C to stop TensorBoard")
    print("")

    try:
        subprocess.run(
            ["tensorboard", f"--logdir={log_dir}", "--port=6006"],
            check=False
        )
    except KeyboardInterrupt:
        print("\n\n✅ TensorBoard stopped.")
    except FileNotFoundError:
        print("❌ TensorBoard not found. Install with: pip install tensorboard")
        sys.exit(1)

if __name__ == "__main__":
    main()
