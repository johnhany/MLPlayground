# Qwen3-4B SFT on RTX 3090

高效的单卡 LoRA 微调脚本，基于 Unsloth 优化框架。

## 快速开始

### 1. 环境要求

```bash
pip install -U unsloth[colab-new] @git+https://github.com/unslothai/unsloth.git
pip install transformers peft trl datasets accelerate bitsandbytes
```

### 2. 准备数据

数据格式为 JSONL（每行一个 JSON）或 JSON 数组：

```json
{
  "uuid": "unique_id",
  "input": "用户问题",
  "output": "模型回答",
  "domain": "数据域（如 math/coding/etc）",
  "meta": {额外信息}
}
```

示例文件：`sample_data.jsonl`

### 3. 基础训练（最小化测试）

```bash
python qwen_sft.py \
  --data sample_data.jsonl \
  --output ./qwen3_sft_output \
  --epochs 1 \
  --max_samples 5 \
  --batch_size 1 \
  --grad_accum 4
```

预期时间：~1-2 分钟（5个样本，GPU 显存占用 ~8-10GB）

### 4. 完整训练（真实场景）

```bash
python qwen_sft.py \
  --data your_data.jsonl \
  --output ./qwen3_sft_output \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 8 \
  --max_seq_length 2048 \
  --lr 2e-4
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `Qwen/Qwen3-4B` | HuggingFace 模型 ID |
| `--data` | - | 训练数据路径（JSONL/JSON）|
| `--output` | `./qwen3_sft_output` | 输出目录（保存 adapter） |
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 2 | 批次大小（RTX 3090 推荐值） |
| `--grad_accum` | 8 | 梯度积累步数 |
| `--lr` | 2e-4 | 学习率 |
| `--max_seq_length` | 2048 | 最大序列长度 |
| `--max_samples` | None | 限制样本数（调试用） |
| `--lora_r` | 16 | LoRA 秩（rank） |
| `--lora_alpha` | 32 | LoRA alpha 系数 |
| `--warmup_ratio` | 0.05 | 预热比例 |

## 显存占用（RTX 3090 = 24GB）

| 配置 | 估计显存 | 样本吞吐量 |
|------|---------|----------|
| bs=1, grad_accum=4 | ~6 GB | 低（调试用） |
| bs=2, grad_accum=8 | ~12 GB | 中（推荐） |
| bs=4, grad_accum=4 | ~16 GB | 高（可能 OOM） |

## 输出文件结构

```
qwen3_sft_output/
├── checkpoint-1/               # 第1个epoch的检查点
├── checkpoint-2/
├── checkpoint-3/
├── final_adapter/              # 最终 LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── events.out.tfevents.*       # TensorBoard 日志
```

## 推理（使用微调后的模型）

```python
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import TextStreamer

# 加载微调的 LoRA adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-4B",
    max_seq_length=2048,
    dtype="bfloat16",
)
model = PeftModel.from_pretrained(model, "qwen3_sft_output/final_adapter")

# 格式化输入
messages = [
    {"role": "user", "content": "1+1=?"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False)

# 推理
inputs = tokenizer(text, return_tensors="pt").to("cuda")
streamer = TextStreamer(tokenizer)
model.generate(**inputs, streamer=streamer, max_new_tokens=128)
```

## 故障排除

### OOM（显存溢出）
1. 减小 `--batch_size`（2→1）
2. 减小 `--max_seq_length`（2048→1024）
3. 增大 `--grad_accum`（8→16）以保持等效批量

### 数据加载报错
- 确保 JSONL 格式正确（每行一个完整 JSON）
- 检查必需字段：`input` 和 `output`

### 模型下载缓慢
```bash
# 使用阿里镜像（中国用户）
export HF_ENDPOINT=https://huggingface.co
huggingface-cli download Qwen/Qwen3-4B
```

## 性能优化

### 1. Unsloth 加速
- 自动启用 Flash Attention 2
- 2x 训练速度，70% 显存节省

### 2. 梯度检查点（Gradient Checkpointing）
- 自动启用，减少激活值显存 ~50%

### 3. 4-bit 量化（可选）
修改 `load_model_and_tokenizer()` 中的 `load_in_4bit=True`（精度损失，但显存 <5GB）

## 参考资源

- Qwen3 官方文档：https://github.com/QwenLM/Qwen3
- Unsloth：https://github.com/unslothai/unsloth
- LoRA 论文：https://arxiv.org/abs/2106.09714
