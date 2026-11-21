# Simple GPT

A minimal character-level GPT implementation built with PyTorch, designed for learning how simple GPT language models work end to end. It trains on a plain text file, learns to predict the next character and then generates text interactively from a saved checkpoint.

## Overview

This project demonstrates the core mechanics of a simple GPT (decoder-only Transformer):

- Character-level tokenization (every unique character in the training text becomes a token)
- Causal multi-head attention implemented from scratch
- Residual connections and layer normalization
- Next-character prediction via cross-entropy loss
- Autoregressive sampling with multinomial sampling

Because the model operates on characters and is intentionally small, it runs comfortably on CPU.

## Project Structure

| File                   | Purpose                                                                 |
| ---------------------- | ----------------------------------------------------------------------- |
| `model.py`             | Model definition: attention `Block` and the `SimpleGPT` class           |
| `trainer.py`           | Reads `input.txt`, trains the model, saves `simple_gpt_model.pt`        |
| `chat.py`              | Loads the checkpoint and generates text from your prompts               |
| `input.txt`            | Training text (provide your own — any plain-text file works)            |
| `simple_gpt_model.pt`  | Saved checkpoint (created by `trainer.py`)                              |

## Requirements

- Python 3.8+
- PyTorch

Install PyTorch if needed:

pip install torch

CUDA is used automatically when available; otherwise training and inference run on CPU.

## Usage

### 1. Prepare the training data

Place a plain-text file named `input.txt` in the project directory. Larger and more varied texts produce noticeably better samples. Project Gutenberg books are a good starting corpus.

### 2. Train

python trainer.py

Training runs for 3000 steps and prints the loss every 250 steps. When it finishes, the script writes `simple_gpt_model.pt`, which contains both the model weights and the character vocabulary (both are required for inference).

### 3. Chat

python chat.py

Type a prompt and the model continues it with 300 sampled characters. Exit with `quit()`, `exit()`, or `bye()`.

Notes:

- Prompts containing characters that were not in `input.txt` are rejected with a warning, since the model can only produce characters from its vocabulary.
- Output is sampled stochastically, so the same prompt can yield different results each time.

## Model Architecture

| Parameter     | Value |
| ------------- | ----- |
| Embedding dim | 64    |
| Layers        | 2     |
| Context size  | 64    |
| Vocabulary    | Unique characters in `input.txt` |

Each block consists of causal self-attention followed by an MLP with a 4x expansion factor, both wrapped in residual connections with pre-normalization (LayerNorm). Attention is computed jointly for query, key, and value with a single linear projection, and future positions are masked so the model cannot peek ahead during training.

## How Training Works

Each step draws 32 random 64-character windows from the text. The input `x` is the window itself and the target `y` is the same window shifted left by one position, so every position is trained to predict the following character. Optimization uses AdamW with a learning rate of 1e-3.

## Ideas for Extension

- Increase `embed_dim`, `n_layers`, or `block_size` for a stronger model
- Add a temperature parameter or top-k sampling to `generate()` for controllable outputs
- Switch to subword (BPE) tokenization for better long-text handling
- Track validation loss to monitor overfitting

## License

Feel free to modify and distribute for educational purposes.