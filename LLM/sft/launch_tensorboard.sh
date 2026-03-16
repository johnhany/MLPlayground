#!/bin/bash
# Launch TensorBoard to monitor training metrics

OUTPUT_DIR="${1:-.}"
LOG_DIR="$OUTPUT_DIR/logs"

if [ ! -d "$LOG_DIR" ]; then
    echo "❌ TensorBoard log directory not found: $LOG_DIR"
    echo "Make sure training has started and created logs."
    exit 1
fi

echo "🚀 Launching TensorBoard..."
echo "📊 Log directory: $LOG_DIR"
echo "🌐 Open browser: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""

tensorboard --logdir="$LOG_DIR" --port=6006
