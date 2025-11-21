# Character-Level Transformer Language Model: Implementation and Experimental Analysis

This project implements and evaluates character-level sequence models for next-character prediction on the text8 dataset. The main objective is to study how different optimisation techniques affect the training behaviour and performance of a small decoder-only Transformer, and to compare it to an LSTM baseline.

## Repository Structure

```text
char_transformer/
│
├── data/
│   ├── text8_train.txt          # Training split
│   └── text8_test.txt           # Test split
│
├── models/
│   ├── lstm.py                  # LSTM baseline implementation
│   └── models.py                # Decoder-only Transformer
│
├── util/
│   └── generation.py            # Text generation functions
│
├── lstm.ipynb                   # LSTM baseline experiment
├── transformer.ipynb            # Baseline Transformer (Optax default Adam)
├── transformer_adam.ipynb       # Transformer with Transformer-paper Adam
├── transformer _noam.ipynb      # Transformer with Noam LR schedule
├── transformer_dropout.ipynb    # Noam + Adam + dropout RNG wiring
├── transformer_gradient_smoothing.ipynb  # Noam + Adam + label smoothing
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
└── README.md                    # This file
```

## Model Architectures

### Transformer Decoder

The main model is a GPT-style decoder-only Transformer:

- Multi-head self-attention with causal masking
- Learned token and positional embeddings
- Layer normalisation and residual connections
- Position-wise feed-forward networks with expansion ratio 4
- Shared input and output embeddings

Configuration used in all Transformer experiments:

```python
vocab_size = 27      # a–z plus space
d_model    = 256     # embedding / hidden dimension
n_heads    = 8       # attention heads
n_layers   = 2       # Transformer blocks
max_len    = 128     # maximum supported sequence length
B, T       = 128, 32 # training batch size and context length
```

The model supports sequences up to length 128, but all training and evaluation windows use T = 32 characters.

### LSTM Baseline

The recurrent baseline is a stacked LSTM defined in `lstm.py`:

```python
vocab_size  = 27
hidden_size = 128
n_layers    = 3
max_len     = 128
B, T        = 128, 32
niter       = 10_000
```

It uses the same character-level data pipeline and window length as the Transformer.

## Dataset

The experiments use the text8 corpus, a 100MB cleaned English Wikipedia dump. Two pre-split files are provided:

- `data/text8_train.txt` – training text
- `data/text8_test.txt`  – held-out test text

The vocabulary consists of 27 characters (a–z and space). Sequences of length 32 are sampled from the training text; at each position the model predicts the next character.

## Experiments

Each Transformer notebook isolates one main change on top of the shared architecture and data pipeline.

### 1. Baseline Transformer (`transformer.ipynb`)

Decoder-only Transformer with Optax default Adam and fixed learning rate.

Configuration:

- Architecture: Transformer (vocab_size = 27, d_model = 256, n_layers = 2, n_heads = 8, max_len = 128)
- Optimiser: Adam with Optax defaults

  ```python
  learning_rate = 0.001
  tx = optax.adam(learning_rate=learning_rate)  # b1 = 0.9, b2 = 0.999, eps = 1e-8
  ```

- Learning rate: constant
- Regularisation: no explicit dropout; no label smoothing
- Training loop: `niter = 100_000`, `B, T = 128, 32`

Results (test split):

- Test Loss: 1.2931
- Test Accuracy: 59.4%
- Last Character Accuracy: 63.0%
- Training Time: 249.5 s

This configuration serves as the reference point for all subsequent experiments.

### 2. Transformer-paper Adam (`transformer_adam.ipynb`)

Uses the same architecture and data pipeline as the baseline, but changes the Adam hyperparameters to those used in the original Transformer paper.

Configuration:

- Architecture: same as baseline
- Optimiser: Adam with Transformer-paper settings

  ```python
  learning_rate = 0.001
  tx = optax.adam(
      learning_rate = learning_rate,
      b1 = 0.9,
      b2 = 0.98,
      eps = 1e-9,
  )
  ```

- Learning rate: constant
- Regularisation: no dropout; no label smoothing
- Training loop: `niter = 100_000`, `B, T = 128, 32`

Results:

- Test Loss: 1.2844
- Test Accuracy: 60.0%
- Last Character Accuracy: 63.6%
- Training Time: 1729.3 s

### 3. Noam Learning Rate Schedule (`transformer _noam.ipynb`)

Combines the Transformer-paper Adam with the Noam learning rate schedule:

```python
def transformer_lr_schedule(d_model, warmup_steps=4000):
    def schedule(step):
        step = jnp.maximum(step, 1)
        inv_sqrt_step = step ** -0.5
        scaled_step   = step * (warmup_steps ** -1.5)
        return (d_model ** -0.5) * jnp.minimum(inv_sqrt_step, scaled_step)
    return schedule
```

Configuration:

- Architecture: same as baseline
- Learning rate: `learning_rate_schedule = transformer_lr_schedule(d_model=256, warmup_steps=4000)`
- Optimiser:

  ```python
  tx = optax.adam(
      learning_rate = learning_rate_schedule,
      b1 = 0.9,
      b2 = 0.98,
      eps = 1e-9,
  )
  ```

- Regularisation: no explicit dropout; no label smoothing
- Training loop: `niter = 100_000`, `B, T = 128, 32`

Results:

- Test Loss: 1.2760
- Test Accuracy: 59.9%
- Last Character Accuracy: 63.0%
- Training Time: 1733.0 s

This configuration achieves the lowest test loss among all Transformer variants.

### 4. “Dropout” Experiment (`transformer_dropout.ipynb`)

In this notebook, the training step is modified to pass a dropout RNG into the model:

```python
logits = model.apply(
    {"params": params},
    x,
    deterministic=False,
    rngs={"dropout": dropout_rng},
)
```

The optimiser and Noam schedule remain the same as in the previous experiment. The current `DecoderOnlyTransformer` definition does not include `nn.Dropout` layers, so this run mainly validates the RNG wiring rather than applying actual stochastic dropout inside the Transformer blocks.

Configuration:

- Architecture: same as Noam experiment
- Optimiser: Adam with Noam schedule and (b1 = 0.9, b2 = 0.98, eps = 1e-9)
- Training loop: `niter = 100_000`, `B, T = 128, 32`

Results:

- Test Loss: 1.3496
- Test Accuracy: 57.9%
- Last Character Accuracy: 62.4%
- Training Time: 1967.1 s

### 5. Label Smoothing (`transformer_gradient_smoothing.ipynb`)

This notebook adds label smoothing to the loss function, while keeping the Noam schedule and Transformer-paper Adam. The training step also passes a dropout RNG as in the previous notebook.

Label smoothing is implemented as:

```python
one_hot  = jax.nn.one_hot(flat_targets, V)
smoothed = one_hot * (1.0 - epsilon_ls) + epsilon_ls / V
loss     = optax.softmax_cross_entropy(flat_logits, smoothed).mean()
```

Configuration:

- Architecture: same as Noam experiment
- Optimiser: Adam with Noam schedule (b1 = 0.9, b2 = 0.98, eps = 1e-9)
- Label smoothing: epsilon_ls = 0.1
- Training loop: `niter = 100_000`, `B, T = 128, 32`

Results:

- Test Loss: 1.7219
- Test Accuracy: 58.4%
- Last Character Accuracy: 61.4%
- Training Time: 1970.5 s

With label smoothing, the training objective changes and the loss cannot be directly compared quantitatively to the non-smoothed runs, although accuracy remains comparable.

### 6. LSTM Baseline (`lstm.ipynb`)

The LSTM baseline uses the same data and evaluation protocol as the Transformer, but with a recurrent architecture.

Configuration:

```python
hidden_size = 128
n_layers    = 3
max_len     = 128
B, T        = 128, 32
niter       = 10_000
learning_rate = 0.001
tx = optax.adam(learning_rate=learning_rate)
```

The LSTM models serve as a classical benchmark for sequence modelling; their performance can be compared against the Transformer variants once evaluated under the same metrics.

## Results Summary

A summary of the Transformer experiments on the test split:

| Configuration               | Test Loss | Test Accuracy | Last Char Accuracy | Time (s) |
|----------------------------|-----------|---------------|--------------------|----------|
| Noam schedule              | 1.2760    | 59.9%         | 63.0%              | 1733.0   |
| Transformer-paper Adam     | 1.2844    | 60.0%         | 63.6%              | 1729.3   |
| Baseline (Optax Adam)      | 1.2931    | 59.4%         | 63.0%              | 249.5    |
| “Dropout” RNG wiring       | 1.3496    | 57.9%         | 62.4%              | 1967.1   |
| Label smoothing (ε = 0.1)  | 1.7219    | 58.4%         | 61.4%              | 1970.5   |

Key observations:

- The Noam schedule, combined with Transformer-paper Adam, achieves the lowest test loss.
- Changing only the Adam hyperparameters from Optax defaults to the Transformer-paper settings yields a modest but consistent improvement.
- The label smoothing run shows higher loss by construction, since the targets are softened; accuracy remains close to other configurations.
- The current codebase wires in dropout RNG but does not yet include explicit dropout layers inside the Transformer.

## Running the Code

### Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Running Experiments

Launch any of the notebooks in Jupyter:

```bash
jupyter notebook transformer.ipynb
jupyter notebook transformer_adam.ipynb
jupyter notebook "transformer _noam.ipynb"
jupyter notebook transformer_dropout.ipynb
jupyter notebook transformer_gradient_smoothing.ipynb
jupyter notebook lstm.ipynb
```

### Text Generation

Text generation is implemented in `util/generation.py`. A typical usage pattern is:

```python
from util.generation import generate_tokens

# context_ids: (B, T) int32 array of initial character indices
generated_ids = generate_tokens(
    model=trained_model,
    params=trained_params,
    rng=jax.random.PRNGKey(0),
    context=context_ids,
    length=500,
    block_size=64,
    temperature=0.8,
    sample=True,
)
```

The generated token IDs can be mapped back to characters using the `int_to_char` dictionary constructed in the notebooks.

## Conclusion

This repository provides a compact, fully working character-level language modelling setup with a small Transformer and an LSTM baseline. The experiments demonstrate how optimiser hyperparameters, learning rate scheduling, and label smoothing influence the optimisation dynamics of a Transformer at modest scale, and serve as a foundation for further architectural or training improvements.
