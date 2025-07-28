# Local LLM Dialogue System

A robust private offline dialogue system for local deployment of quantized Large Language Models using llama-cpp-python. Features advanced streaming text generation, intelligent content filtering, and secure local AI interactions.

## Features

- **Private Local Inference**: Run quantized LLMs entirely on your machine without external API dependencies
- **Advanced Streaming**: Real-time text generation with dual-method approach (API streaming + token fallback)
- **Intelligent Content Filtering**: Prevents generation of irrelevant or off-topic content
- **ChatML Support**: Proper conversation formatting with system prompts and dialogue history
- **Robust Error Handling**: Comprehensive error recovery with automatic fallback mechanisms
- **GPU Acceleration**: Optional GPU offloading for improved performance
- **Flexible Configuration**: Customizable temperature, context size, and generation parameters

## Requirements

- Python 3.8+
- llama-cpp-python
- Compatible quantized model in GGUF format (Qwen, LLaMA, etc.)

## Model Download

Due to the large size of LLM models (typically 4-15GB), model files are not included in this repository. You'll need to download a compatible GGUF model separately.

### Recommended Models

#### Qwen Models (Recommended)
- **Qwen 1.5 7B Chat Q4_0**: [Download from Hugging Face](https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GGUF/resolve/main/qwen1_5-7b-chat-q4_0.gguf)
- **Qwen 1.5 7B Chat Q8_0**: [Download from Hugging Face](https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GGUF/resolve/main/qwen1_5-7b-chat-q8_0.gguf)
- **Full Qwen Collection**: [Browse all Qwen GGUF models](https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GGUF/tree/main)

#### LLaMA Models
- **LLaMA 2 7B Chat**: [TheBloke's GGUF Collection](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
- **Code Llama 7B**: [TheBloke's Code Llama GGUF](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF)

#### Other Compatible Models
- **Mistral 7B**: [TheBloke's Mistral GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
- **OpenChat 3.5**: [TheBloke's OpenChat GGUF](https://huggingface.co/TheBloke/openchat_3.5-GGUF)

### Quantization Levels

Choose based on your hardware and quality requirements:

| Quantization | File Size | RAM Usage | Quality | Speed |
|-------------|-----------|-----------|---------|-------|
| Q4_0        | ~4GB      | ~6GB      | Good    | Fast  |
| Q4_1        | ~4.5GB    | ~6.5GB    | Better  | Fast  |
| Q5_0        | ~5GB      | ~7GB      | High    | Medium|
| Q8_0        | ~7GB      | ~9GB      | Highest | Slow  |

### Download Instructions

1. **Create models directory**:
   ```bash
   mkdir models
   cd models
   ```

2. **Download using wget** (Linux/Mac):
   ```bash
   wget https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GGUF/resolve/main/qwen1_5-7b-chat-q4_0.gguf
   ```

3. **Download using curl**:
   ```bash
   curl -L -o qwen1_5-7b-chat-q4_0.gguf https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GGUF/resolve/main/qwen1_5-7b-chat-q4_0.gguf
   ```

4. **Or download manually** from the Hugging Face links above and place in the `models/` directory

### Verify Download
```bash
ls -la models/
# Should show your downloaded .gguf file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Dialogue Session

```bash
python dialogue.py --model /path/to/your/model.gguf --interactive
```

### Single Question Mode

```bash
python dialogue.py --model /path/to/your/model.gguf --question "How are you today?"
```

### Advanced Configuration

```bash
python dialogue.py --model /path/to/your/model.gguf \
    --interactive \
    --temp 0.3 \
    --max_tokens 1024 \
    --n_ctx 4096 \
    --n_gpu_layers 10
```

## Command Line Options

- `--model`: Path to the GGUF model file (required)
- `--interactive`: Start an interactive dialogue session
- `--question`: Ask a single question (non-interactive mode)
- `--temp`: Temperature for sampling (default: 0.3)
- `--max_tokens`: Maximum tokens to generate (default: 512)
- `--n_ctx`: Context size for the model (default: 2048)
- `--n_gpu_layers`: Number of layers to offload to GPU (default: 0)

## Key Features

### Content Filtering
The system includes intelligent content filtering that:
- Prevents generation of irrelevant or off-topic content
- Stops inappropriate academic/exam content from appearing
- Maintains conversation quality and relevance
- Uses minimal filtering to avoid blocking legitimate responses

### Robust Streaming
- **Primary Method**: llama-cpp-python's streaming completion API
- **Automatic Fallback**: Switches to token-by-token generation if streaming fails
- **Error Recovery**: Graceful handling of generation errors
- **Real-time Display**: Text appears as it's generated for better user experience

### Conversation Management
- **ChatML Format**: Proper conversation structure with system/user/assistant roles
- **History Tracking**: Maintains conversation context across turns
- **Memory Management**: Automatic history truncation to stay within context limits

## Examples

### Running with Qwen Model
```bash
python dialogue.py --model models/qwen1_5-7b-chat-q4_0.gguf --interactive --temp 0.3
```

### GPU Acceleration
```bash
python dialogue.py --model models/qwen1_5-7b-chat.q4_0.gguf --interactive --n_gpu_layers 32
```

### Custom Configuration for Long Conversations
```bash
python dialogue.py --model models/your-model.gguf \
    --interactive \
    --temp 0.5 \
    --max_tokens 2048 \
    --n_ctx 8192
```

## Troubleshooting

### Model Loading Issues
- Ensure your model is in GGUF format
- Check that you have enough RAM/VRAM for the model size
- Try reducing `n_ctx` if running out of memory

### Generation Problems
- Adjust `--temp` for different response styles (lower = more deterministic)
- Increase `--max_tokens` for longer responses
- Use `--n_gpu_layers` to offload computation to GPU if available

### Content Quality
The system automatically handles:
- Stopping at appropriate conversation boundaries
- Filtering out irrelevant content
- Maintaining high-quality responses

## Architecture

The system uses a modular architecture with:
- **Dual Generation Strategy**: Streaming API with token-by-token fallback
- **Content Pipeline**: Real-time filtering during generation
- **Memory Management**: Efficient conversation history handling
- **Error Recovery**: Multiple layers of fallback mechanisms

## License

This project is open source and available under the MIT License. 