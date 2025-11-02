# Transformer Language Model Experiments

Production-ready implementation of modern transformer architectures for character-level language modeling, featuring RoPE, GQA, SwiGLU, and RMSNorm.

## Features

- **Multiple Architectures**: Baseline, RoPE, GQA, SwiGLU, RMSNorm, and combined configurations
- **Optimizer Comparison**: Adam, AdamW, and SGD implementations
- **Comprehensive Metrics**: Loss, bits-per-character (BPC), and perplexity
- **Experiment Tracking**: Automatic logging and checkpointing
- **Text Generation**: Interactive and batch generation modes
- **Evaluation Tools**: Model comparison and performance analysis

## Project Structure

```
.
├── model.py                 # Transformer model implementations
├── config.py                # Model and training configurations
├── train_experiments.py     # Training script with experiment tracking
├── evaluate.py              # Evaluation and generation tools
├── utils.py                 # Helper functions
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Download the Shakespeare dataset:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O shakespeare.txt
```

Or use your own text file and update `DATA_PATH` in `config.py`.

## Usage

### Training a Single Model

Train a specific model with chosen optimizer:

```bash
# Train baseline model with AdamW
python train_experiments.py --model baseline --optimizer adamw

# Train full model (all features) with Adam
python train_experiments.py --model full --optimizer adam

# Train with custom iterations
python train_experiments.py --model rope --optimizer adamw --max_iters 10000
```

**Available Models:**
- `baseline` - Standard transformer with absolute positional encoding
- `rope` - Rotary Position Embedding (RoPE)
- `gqa` - Grouped Query Attention (2 KV heads)
- `swiglu` - SwiGLU activation function
- `rmsnorm` - RMS normalization instead of LayerNorm
- `full` - All features combined

**Available Optimizers:**
- `adam` - Standard Adam optimizer
- `adamw` - Adam with weight decay (recommended)
- `sgd` - Stochastic Gradient Descent with momentum

### Running All Experiments

To run a complete experiment suite comparing all architectures:

```bash
python train_experiments.py --all
```

This will train all model architectures with both Adam and AdamW optimizers, then generate a comparison table.

### Evaluation

Evaluate a trained model:

```bash
# Basic evaluation with text generation
python evaluate.py checkpoints/baseline_adamw_20241102_120000/best_model.pt

# Generate more samples
python evaluate.py checkpoints/full_adamw_20241102_120000/best_model.pt --samples 10 --max_tokens 1000

# Interactive generation mode
python evaluate.py checkpoints/full_adamw_20241102_120000/best_model.pt --interactive --temperature 0.8

# Compare multiple models
python evaluate.py --compare \
    checkpoints/baseline_adamw_20241102_120000/best_model.pt \
    checkpoints/rope_adamw_20241102_120000/best_model.pt \
    checkpoints/full_adamw_20241102_120000/best_model.pt
```

## Configuration

Edit `config.py` to customize:

```python
# Training parameters
MAX_ITERS = 5000          # Training iterations
BATCH_SIZE = 64           # Batch size
BLOCK_SIZE = 256          # Context length
LEARNING_RATE = 3e-4      # Learning rate

# Model architecture
MODELS = {
    'custom': {
        'n_embd': 512,     # Embedding dimension
        'n_head': 8,       # Number of attention heads
        'n_layer': 8,      # Number of transformer layers
        'dropout': 0.2,
        'use_rope': True,
        'use_gqa': True,
        'n_kv_head': 4,    # GQA key-value heads
        'use_swiglu': True,
        'use_rmsnorm': True,
    }
}
```

## Architecture Details

### Rotary Position Embedding (RoPE)
- Encodes position information via rotation in complex space
- Better length generalization than absolute positional encoding
- Multiplicative attention bias instead of additive

### Grouped Query Attention (GQA)
- Reduces KV cache size by sharing key-value heads
- Interpolates between multi-head and multi-query attention
- Faster inference with minimal quality loss

### SwiGLU Activation
- Gated Linear Unit variant with Swish activation
- Improves model capacity without adding parameters
- Used in modern LLMs (LLaMA, PaLM)

### RMSNorm
- Simpler than LayerNorm, normalizes by RMS only
- Faster computation, no mean subtraction
- Equivalent performance in practice

## Metrics

### Cross-Entropy Loss
Natural logarithm of average probability assigned to correct character.

### Bits Per Character (BPC)
```
BPC = Loss / log(2)
```
Measures compression efficiency. Lower is better.

### Perplexity
```
Perplexity = exp(Loss)
```
Inverse probability of correct prediction. Lower is better.

## Output Files

Each training run creates:
```
checkpoints/
└── {model}_{optimizer}_{timestamp}/
    ├── best_model.pt              # Best checkpoint
    ├── training_log.json          # Metrics history
    └── {model}_eval_results.json  # Evaluation results
```

Training log contains:
- Loss and BPC curves
- Model configuration
- Training hyperparameters
- Total training time

## Example Results

After running experiments, you'll see output like:

```
Results Summary:
--------------------------------------------------------------------------------
Model           Optimizer  Best Val Loss   Best Val BPC    Params      
--------------------------------------------------------------------------------
full            adamw      1.2456          1.7982          2,104,320   
rope            adamw      1.2891          1.8609          2,104,320   
gqa             adamw      1.2934          1.8671          1,893,376   
swiglu          adamw      1.3012          1.8783          2,367,488   
rmsnorm         adamw      1.3089          1.8894          2,104,320   
baseline        adamw      1.3245          1.9119          2,104,320   
--------------------------------------------------------------------------------
```

## Interactive Generation

```bash
python evaluate.py checkpoints/full_adamw_20241102_120000/best_model.pt --interactive
```

```
Prompt: ROMEO:
Generated:
ROMEO:
I will not be so much as I am not
The king of men, and the world shall be
As well as you, my lord, I will not be
...
```

## Tips for Better Results

1. **Longer Training**: Increase `MAX_ITERS` for better convergence
2. **Larger Models**: Increase `n_embd`, `n_head`, or `n_layer` for more capacity
3. **Learning Rate**: Try learning rate warmup for stable training
4. **Data Quality**: Clean and diverse training data improves generation
5. **Temperature**: Lower temperature (0.7-0.8) for more coherent generation

## Troubleshooting

**CUDA Out of Memory:**
- Reduce `BATCH_SIZE` or `BLOCK_SIZE`
- Use smaller model (fewer layers/dimensions)

**Poor Generation Quality:**
- Train longer (`MAX_ITERS`)
- Try lower dropout rate
- Ensure data is properly formatted

**Slow Training:**
- Enable GPU acceleration
- Increase batch size if memory allows
- Use mixed precision training

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

## License

This implementation is for educational purposes. The architecture patterns are based on published research papers.

## Acknowledgments

Based on Andrej Karpathy's nanoGPT and incorporates modern architecture improvements from recent LLM research.
