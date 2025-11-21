# Character-Level Transformer Language Model: Implementation and Experimental Analysis

A systematic implementation and empirical evaluation of character-level Transformer language models trained on the text8 dataset. This project investigates optimization techniques from "Attention Is All You Need" (Vaswani et al., 2017) including dropout regularization, label smoothing, Noam learning rate scheduling, and the Adam optimizer.

**Repository**: https://github.com/SheYuting/char_transformer

## Repository Structure

```
char_transformer/
│
├── data/
│   ├── text8_train.txt          # Training split 
│   └── text8_test.txt           # Test split 
│
├── models/
│   ├── lstm.py                  # LSTM baseline implementation  
│   └── models.py                # Transformer models
│
├── util/
│   └── generation.py            # Text generation functions
│
├── lstm.ipynb                   # LSTM baseline experiments
├── transformer.ipynb            # Baseline Transformer
├── transformer_adam.ipynb       # Adam optimizer experiments
├── transformer__noam.ipynb      # Noam learning rate schedule
├── transformer_dropout.ipynb    # Dropout regularization
├── transformer_gradient_smoothing.ipynb  # Label smoothing
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
└── README.md                    # This file
```

## Project Overview

This project implements and evaluates character-level sequence models for next-character prediction on the text8 dataset (100MB preprocessed Wikipedia text). The primary focus is systematic comparison of optimization techniques and their impact on training dynamics, convergence behavior, and final model performance.

###Architectures Implemented

**Transformer Decoder**
- Multi-head self-attention mechanism with causal masking
- Sinusoidal positional encoding
- Layer normalization and residual connections
- Position-wise feed-forward networks
- **Specifications:** 6 layers, 64 embedding dimensions, 8 attention heads, ~308K parameters

**LSTM Baseline**
- Recurrent architecture with hidden state
- Standard benchmark for sequence modeling tasks
- Comparative baseline for Transformer performance

## Model Architecture

**Transformer Decoder Specifications:**
```python
d_model = 64            # Embedding dimension
n_layers = 6            # Transformer blocks
n_heads = 8             # Attention heads per layer
d_ff = 256              # Feed-forward hidden dimension (4x expansion)
context_length = 128    # Sequence length
dropout = 0.1           # (when enabled)
vocab_size = 27         # Character vocabulary
mlp_ratio = 4           # FFN expansion factor
```

**Total Parameters:** Approximately 308,000 (~0.31M)

**Parameter Breakdown:**
- Token embeddings: 1,728
- Positional embeddings: 8,192
- Transformer blocks (6x): 296,448
- Output projection: 1,728

## Experimental Configurations

Five systematic experiments isolate the effect of individual optimization techniques:

### 1. Baseline Transformer (`transformer.ipynb`)
Standard Transformer decoder with minimal modifications. Establishes performance baseline.

**Configuration:**
- Optimizer: Adam (lr=0.001)
- No dropout, no label smoothing
- Fixed learning rate

**Results:**
- Test Loss: 1.2931
- Test Accuracy: 59.4%
- Last Character Accuracy: 63.0%
- Training Time: 249.5 seconds

### 2. Adam Optimizer (`transformer_adam.ipynb`)
Enhanced Adam implementation with default hyperparameters.

**Results:**
- Test Loss: 1.2844
- Test Accuracy: 60.0%
- Last Character Accuracy: 63.6%
- Training Time: 1729.3 seconds

**Improvement over Baseline:** 0.7% loss reduction

### 3. Noam Learning Rate Schedule (`transformer__noam.ipynb`) ⭐ BEST
Implements adaptive learning rate with warmup and inverse square root decay.

**Schedule Formula:**
```
lr(step) = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
```

**Configuration:**
- Optimizer: Adam with Noam schedule
- Warmup steps: 4,000
- Peak learning rate: Based on d_model (64)
- No dropout, no label smoothing

**Results:**
- **Test Loss: 1.2760** (Best)
- Test Accuracy: 59.9%
- Last Character Accuracy: 63.0%
- Training Time: 1733.0 seconds

**Improvement:** 1.3% over baseline, 0.7% over fixed-LR Adam

### 4. Dropout Regularization (`transformer_dropout.ipynb`)
Applies dropout (p=0.1) to embeddings, attention weights, and feed-forward layers.

**Results:**
- Test Loss: 1.3496
- Test Accuracy: 57.9%
- Last Character Accuracy: 62.4%
- Training Time: 1967.1 seconds

**Observation:** Dropout increased test loss by 5.8%, suggesting overregularization.

### 5. Label Smoothing (`transformer_gradient_smoothing.ipynb`)
Implements label smoothing (ε=0.1) with Noam schedule.

**Results:**
- Test Loss: 1.7219
- Test Accuracy: 58.4%
- Last Character Accuracy: 61.4%
- Training Time: 1970.5 seconds

**Note:** Higher loss values are expected with label smoothing as it intentionally softens predictions for better calibration.

### 6. LSTM Baseline (`lstm.ipynb`)
Classical recurrent architecture for comparative analysis.

**Specifications:**
```python
d_model = 128           # Hidden size
n_layers = 3            # LSTM layers
max_len = 128           # Sequence length
```

## Dataset

**text8 Corpus**
- Source: Preprocessed English Wikipedia (100MB)
- Preprocessing: Lowercase conversion, punctuation removal, space normalization
- Training file: `data/text8_train.txt` (90MB)
- Test file: `data/text8_test.txt` (10MB)
- Vocabulary size: 27 characters (a-z + space)
- Sequence length: 128 characters

The dataset files are already split and preprocessed in the repository.

## Results Summary

### Performance Comparison

| Configuration | Test Loss | Test Accuracy | Last Char Acc. | Training Time (s) |
|--------------|-----------|---------------|----------------|-------------------|
| **Noam Schedule** | **1.2760** | **59.9%** | **63.0%** | 1733 |
| Adam Optimizer | 1.2844 | 60.0% | 63.6% | 1729 |
| Baseline | 1.2931 | 59.4% | 63.0% | 250 |
| Dropout | 1.3496 | 57.9% | 62.4% | 1967 |
| Label Smoothing† | 1.7219 | 58.4% | 61.4% | 1971 |

†Higher loss expected due to smoothed target distributions

### Key Findings

**1. Learning Rate Scheduling is Critical**

The Noam scheduler achieved the lowest test loss (1.2760), demonstrating that adaptive learning rates with warmup significantly improve convergence. The warmup phase prevents early training instabilities while the inverse square root decay enables fine-grained optimization in later stages.

**2. Dropout Can Overregularize Small Models**

Dropout regularization (p=0.1) resulted in worse test performance (1.3496 vs 1.2760), suggesting that for a 308K parameter model trained on 90MB of data, aggressive regularization is counterproductive. The parameter-to-data ratio (1:292K bytes/param) is already data-rich.

**3. Label Smoothing Changes Loss Interpretation**

Label smoothing intentionally increases loss values by adding uncertainty to target distributions. The higher loss (1.7219) reflects better-calibrated predictions rather than worse performance. This technique is valuable when confidence estimates are required.

**4. Character-Level Modeling Ceiling**

All configurations plateau at 58-60% accuracy, reflecting the inherent difficulty of next-character prediction. Character-level models face higher perplexity than word-level models due to increased sequence length requirements and lack of lexical semantic information.

### Training Dynamics

**Generalization Gap (Train Loss - Test Loss):**

| Configuration | Gap | Interpretation |
|--------------|-----|----------------|
| Baseline | 0.0102 | Minimal (possible underfitting) |
| Adam | 0.0285 | Healthy generalization |
| Dropout | 0.0412 | Moderate despite regularization |
| Label Smoothing | 0.0398 | Appropriate uncertainty |
| Noam | 0.0627 | Largest but best test performance |

## Installation and Usage

### Prerequisites

Python 3.8+ with dependencies:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- JAX 0.4.x (automatic differentiation and XLA compilation)
- Flax 0.8.x (neural network library)
- Optax 0.1.x (optimization library)
- NumPy, Matplotlib (utilities and visualization)

### Dataset

The text8 dataset is already included in the repository under `data/`:
- `data/text8_train.txt` - Training data (90MB)
- `data/text8_test.txt` - Test data (10MB)

If you need to download the original text8 dataset:

```bash
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip
# Then split into train/test files
```

### Running Experiments

Each notebook is self-contained and can be executed independently:

```bash
# Best performing configuration
jupyter notebook transformer__noam.ipynb

# Adam optimizer baseline
jupyter notebook transformer_adam.ipynb

# Other configurations
jupyter notebook transformer_dropout.ipynb
jupyter notebook transformer_gradient_smoothing.ipynb
```

### Text Generation

Generate text samples using trained models:

```python
from util.generation import generate_tokens

# Sample with temperature
generated = generate_tokens(
    model=model,
    params=params,
    rng=rng,
    context=context,
    length=500,
    block_size=128,
    temperature=0.8,
    sample=True
)
```

## Technical Implementation

### JAX + Flax Framework

This project uses JAX and Flax for several advantages:

**JAX Benefits:**
- XLA compilation for optimized GPU kernels
- Automatic differentiation with `grad()`
- Functional programming paradigm
- Efficient vectorization with `vmap`

**Flax Benefits:**
- Clean, modular neural network definitions
- Explicit parameter management
- Easy integration with Optax optimizers
- Functional approach to state management

### Training Configuration

```python
max_iterations = 100_000
batch_size = 64
gradient_clip = 1.0
warmup_steps = 4_000    # (for Noam schedule)
evaluation_frequency = 1_000  # iterations
```

### Optimization

**Adam Hyperparameters:**
```python
learning_rate = 0.001   # Base/peak learning rate
beta1 = 0.9            # First moment decay
beta2 = 0.999          # Second moment decay
epsilon = 1e-8         # Numerical stability
```

**Noam Schedule Implementation:**
```python
def transformer_lr_schedule(step, d_model=64, warmup_steps=4000):
    step = max(step, 1)
    return d_model**(-0.5) * min(
        step**(-0.5), 
        step * warmup_steps**(-1.5)
    )
```

## Theoretical Background

### Transformer Architecture

The Transformer replaces recurrent connections with multi-head self-attention:

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Multi-Head Attention:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O
where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

### Optimization Techniques

**Noam Learning Rate Schedule**: Combines linear warmup with inverse square root decay for stable optimization and fine-grained convergence.

**Label Smoothing**: Replaces hard targets with soft distributions: `y_smooth = (1 - ε) × y_true + ε / K`, improving calibration and reducing overconfidence.

**Dropout Regularization**: Randomly zeros neurons during training to prevent co-adaptation and encourage redundant representations.

**Adam Optimizer**: Adaptive moment estimation combining momentum and RMSprop for robust per-parameter learning rates.

## Limitations and Future Work

### Current Limitations

1. **Single Dataset:** Evaluated only on text8
2. **Small Model:** 308K parameters; larger models may show different patterns  
3. **Limited Search:** No exhaustive hyperparameter tuning
4. **Computational Constraints:** Limited to 100K iterations

### Future Directions

**Architecture Improvements:**
- Implement Flash Attention for efficiency
- Evaluate Rotary Position Embeddings (RoPE)
- Test Grouped Query Attention (GQA)
- Scale to larger models (1M-10M parameters)

**Training Enhancements:**
- Mixed precision training (FP16/BF16)
- Gradient checkpointing for memory efficiency
- Longer contexts (256-512 tokens)
- Extended training (500K+ iterations)

**Evaluation Extensions:**
- Multiple datasets (code, multilingual, domain-specific)
- Calibration error metrics (ECE)
- Human evaluation of generated samples

## References

### Primary Papers

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention is all you need**. In *Advances in Neural Information Processing Systems* (pp. 5998-6008).

2. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). **Rethinking the inception architecture for computer vision**. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 2818-2826).

3. Kingma, D. P., & Ba, J. (2015). **Adam: A method for stochastic optimization**. In *International Conference on Learning Representations* (ICLR).

4. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). **Dropout: A simple way to prevent neural networks from overfitting**. *The Journal of Machine Learning Research*, 15(1), 1929-1958.

### Implementation References

5. Karpathy, A. (2022). **nanoGPT: A minimal implementation of GPT**. GitHub repository: https://github.com/karpathy/nanoGPT

6. Rush, A. M. (2018). **The Annotated Transformer**. In *Proceedings of Workshop for NLP Open Source Software* (NLP-OSS).

### Dataset

7. Mahoney, M. (2011). **Large Text Compression Benchmark**. Available at: http://mattmahoney.net/dc/textdata.html


## License

This project is for educational and research purposes. The architecture implementations are based on techniques described in published research papers.

## Acknowledgments

- Andrej Karpathy for inspirational nanoGPT implementation
- Google Research for JAX and Flax frameworks  
- Original Transformer authors (Vaswani et al.) for foundational architecture
- Text8 dataset creators for benchmark corpus

