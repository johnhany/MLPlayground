# Qwen3-4B SFT on RTX 3090

高效的单卡 LoRA 微调脚本，基于 Unsloth 优化框架。

**特别说明**：本脚本支持 4-bit 量化模型的 QLoRA 训练（如果模型以 4-bit 加载）。

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

## QLoRA 训练（4-bit 量化模型）

如果你的模型以 4-bit 量化加载（由 Unsloth 或 bitsandbytes 处理），脚本会自动进行 QLoRA 训练：

- ✅ **自动检测**：脚本自动识别 4-bit 量化层
- ✅ **冻结基础权重**：4-bit 量化的权重保持冻结
- ✅ **训练 LoRA 适配器**：只有 LoRA 参数可训练
- ✅ **显存高效**：4-bit 量化 + LoRA = 极低显存占用

### 显存对比（RTX 3090）

| 方案 | 模型精度 | LoRA | 显存占用 | 速度 |
|------|---------|------|---------|------|
| Full Fine-Tune | bf16 | ✗ | ~24 GB | 1x |
| LoRA | bf16 | ✓ | ~12 GB | 2x (Unsloth) |
| **QLoRA** | 4-bit | ✓ | **~6 GB** | **2x** |

你的 Qwen3-4B 正在使用 **QLoRA**，这是最显存高效的选项。

## 监控训练指标

### 实时查看指标（TensorBoard）

训练过程中，所有指标自动保存到 TensorBoard 日志。启动 TensorBoard 查看：

```bash
# 方法 1：Python 脚本（推荐）
python launch_tensorboard.py ./qwen3_sft_output

# 方法 2：直接命令
tensorboard --logdir=./qwen3_sft_output/logs --port=6006
```

然后在浏览器打开：**http://localhost:6006**

### 可监控的指标

| 指标 | 说明 | 用途 |
|------|------|------|
| **loss** | 训练损失 | 监控模型学习进度，应该逐步下降 |
| **learning_rate** | 学习率变化 | 验证学习率调度器（cosine）工作正常 |
| **grad_norm** | 梯度范数 | 检查梯度爆炸/消失（应 < 1.0） |
| **epoch** | 当前轮数 | 追踪训练进度 |
| **steps** | 全局步数 | 总训练步数 |

### 指标解读

**正常训练迹象**：
- ✅ Loss 平稳下降（不一定单调，但总体趋势向下）
- ✅ Learning rate 按 cosine 曲线衰减
- ✅ Grad norm 稳定在 0.1~1.0 范围
- ✅ 每步耗时 ~7-8 秒（RTX 3090 + QLoRA）

**异常迹象**：
- ❌ Loss 不下降或上升 → 学习率过高，尝试 `--lr 5e-5`
- ❌ Grad norm 爆炸（> 10） → 梯度裁剪失效，检查数据质量
- ❌ Loss 震荡剧烈 → 批量大小过小，增加 `--grad_accum`

## 本地模型加载（无网络连接）

### 快速开始

如果网络无法连接到 HuggingFace Hub，可以使用 `--local_only` 标志仅从本地加载模型：

```bash
python qwen_sft.py \
  --model /path/to/local/Qwen3-4B \
  --data sample_data.jsonl \
  --output ./output \
  --local_only
```

### 准备本地模型文件

#### 方法 1：从 HuggingFace 缓存目录复制

```bash
# 模型通常缓存在：
~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/

# 或使用环境变量指定缓存位置
export HF_HOME=/custom/path
python qwen_sft.py --model $HF_HOME/models--Qwen--Qwen3-4B --local_only
```

#### 方法 2：从磁盘下载模型

```bash
# 下载整个模型目录到本地
mkdir -p ./models
cd ./models

# 使用 huggingface-cli（需要网络，但可以一次性完成）
huggingface-cli download Qwen/Qwen3-4B --local-dir ./Qwen3-4B

# 或者手动下载后复制
# 确保本地目录包含以下文件：
# ├── config.json
# ├── model.safetensors (或多个 *.bin 文件)
# ├── tokenizer.json
# ├── tokenizer_config.json
# └── generation_config.json
```

#### 方法 3：使用阿里镜像（中国网络）

```bash
# 设置阿里镜像源
export HF_ENDPOINT=https://huggingface.co

# 下载模型
huggingface-cli download Qwen/Qwen3-4B --local-dir ./Qwen3-4B

# 后续训练使用本地路径
python qwen_sft.py \
  --model ./Qwen3-4B \
  --data sample_data.jsonl \
  --local_only
```

### 验证本地模型文件完整性

```bash
ls -la ./Qwen3-4B/
# 应该看到：
# -rw-r--r--  config.json
# -rw-r--r--  model.safetensors (或 *.bin)
# -rw-r--r--  tokenizer.json
# -rw-r--r--  tokenizer_config.json
# -rw-r--r--  generation_config.json
```

### 排查本地加载错误

如果 `--local_only` 仍报错：

```bash
# 1. 检查 HF_HOME 缓存
export HF_HOME=~/.cache/huggingface
find $HF_HOME -name "config.json" | grep Qwen

# 2. 检查文件权限
chmod -R 755 /path/to/Qwen3-4B/

# 3. 尝试不用 --local_only（允许网络，使用缓存加速）
python qwen_sft.py --model Qwen/Qwen3-4B --data sample_data.jsonl
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `Qwen/Qwen3-4B` | HuggingFace 模型 ID 或本地路径 |
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
| `--local_only` | False | 仅加载本地模型，禁用 HuggingFace Hub |

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
