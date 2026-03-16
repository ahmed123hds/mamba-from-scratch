# Mamba-from-scratch 🐍

A clean, documented, and mathematics-first implementation of the [Mamba architecture](https://arxiv.org/abs/2312.00752) (Selective State Space Models) from scratch in PyTorch.

This repository is built to **show exactly how Mamba works under the hood**—from the continuous-time state space equations to the zero-order hold (ZOH) discretization, selective gating, and the parallel scan algorithm.

## Features
- **S6 (Selective SSM)**: Input-dependent parameterization of \(\Delta\), \(B\), and \(C\).
- **ZOH Discretization**: Exact conversion of continuous SSM logic to discrete recurrences.
- **Sequential vs. Parallel Scans**: Includes a slow Python-loop sequential scan (for debugging/learning) and a fast $O(\log L)$ Hillis-Steele parallel prefix scan algorithm (for performance).
- **Trainable Character LM**: Includes a minimal `train.py` script to train the model on a small text document (e.g. Shakespeare) from scratch.
- **Gated Architecture**: Implements the dual-path SwiGLU-style block that pairs the short-range `Conv1d` with the long-range active memory of the SSM.

## Project Structure
```text
mamba/
├── ops.py       # Core SSM math: Discretization, Parallel/Sequential Scans
├── block.py     # Mamba component: SSM + Conv1d + Gating mechanism
├── model.py     # Full Language Model wiring: Embedding, layers, loss
├── __init__.py  
train.py         # Tiny Shakespeare training loop
generate.py      # Autoregressive inference script
```

## Running the Code

### 1. Train the model
Train a tiny character-level language model from scratch. By default, it uses a built-in snippet of Shakespeare.

```bash
python train.py
```
*You can also point it to your own generic text file using `--data my_text.txt`*.

### 2. Generate Text
Once training completes (it saves `mamba_ckpt.pt`), generate text given a prompt:

```bash
python generate.py --prompt "To be,"
```

## Mathematical Overview of the Implementation

Unlike traditional Transformers that use $O(N^2)$ Self-Attention, Mamba operates via a line-time recurrent process.

1. **The Math**: The core of a **State Space Model (SSM)** is an ordinary differential equation:
   $$ h'(t) = Ah(t) + Bx(t) $$
   $$ y(t) = Ch(t) + Dx(t) $$
   
2. **Discretization (ZOH)**: Computers operate on discrete tokens. We apply the *Zero-Order Hold* using a learned step size $\Delta$ per token. See `discretize()` in `mamba/ops.py`.
   $$ \bar{A} = \exp(\Delta A) $$
   $$ \bar{B} \approx \Delta B $$

3. **Selectivity**: The breakthrough of Mamba. Prior SSMs kept A, B, and C rigid. In this implementation (see `SelectiveSSM` in `mamba/block.py`), $\Delta, B,$ and $C$ are projected directly from the current input token $x_k$, enabling the model to decide dynamically when to remember, forget, or update context.

4. **Hillis-Steele Parallel Scan**: While the recurrence $h_k = \bar{A}_k h_{k-1} + \bar{B}_k x_k$ looks serial, it's linear. We implement a tree-based parallel prefix scan (`ssm_scan_parallel`) to compute all sequence states simultaneously, drastically speeding up training over standard loops.
