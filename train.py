"""
train.py -- Character-level language model training demo

Trains Mamba on a plain text file at the character level.
This is the classic "tiny Shakespeare" setup -- good for validating
that everything works end to end without needing GPUs or large data.

Usage:
    python train.py                           # uses built-in sample text
    python train.py --data path/to/text.txt  # use your own file

The model learns to predict the next character given the previous ones,
which is exactly what Mamba's autoregressive training does.
"""

import os
import math
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW

from mamba.model import Mamba, MambaConfig


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
""".strip() * 40   # repeat to give the model more data to learn from


def build_dataset(text, seq_len, train_frac=0.9):
    """Encode text to integers, split into train/val, build (input, target) tensors."""
    chars    = sorted(set(text))
    ch2id    = {c: i for i, c in enumerate(chars)}
    data     = torch.tensor([ch2id[c] for c in text], dtype=torch.long)

    n        = int(len(data) * train_frac)
    train    = data[:n]
    val      = data[n:]

    def make_pairs(d):
        # sliding window: input = d[i:i+seq_len], target = d[i+1:i+seq_len+1]
        n_chunks = (len(d) - 1) // seq_len
        x = torch.stack([d[i * seq_len    : i * seq_len + seq_len] for i in range(n_chunks)])
        y = torch.stack([d[i * seq_len + 1: i * seq_len + seq_len + 1] for i in range(n_chunks)])
        return x, y

    return make_pairs(train), make_pairs(val), ch2id, {i: c for c, i in ch2id.items()}


def get_batch(x, y, batch_size, device):
    idx = torch.randint(len(x), (batch_size,))
    return x[idx].to(device), y[idx].to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, train_xy, val_xy, batch_size, eval_iters, device):
    model.eval()
    results = {}
    for split, (x, y) in [("train", train_xy), ("val", val_xy)]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(x, y, batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────
    if args.data and os.path.exists(args.data):
        with open(args.data, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded {len(text):,} characters from {args.data}")
    else:
        text = SAMPLE_TEXT
        print(f"Using built-in sample text ({len(text):,} characters)")

    train_xy, val_xy, ch2id, id2ch = build_dataset(text, args.seq_len)
    vocab_size = len(ch2id)
    print(f"Vocabulary size: {vocab_size} unique characters")

    # ── Model ─────────────────────────────────────────────────────────────
    config = MambaConfig(
        vocab_size  = vocab_size,
        d_model     = args.d_model,
        n_layers    = args.n_layers,
        d_state     = args.d_state,
        expand      = 2,
    )
    model = Mamba(config).to(device)
    print(f"Mamba parameters: {model.count_parameters():,}")

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Cosine LR schedule with linear warmup
    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # ── Training ───────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        xb, yb = get_batch(*train_xy, args.batch_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_steps:
            losses = estimate_loss(model, train_xy, val_xy,
                                   args.batch_size, args.eval_iters, device)
            elapsed = time.time() - t0
            print(f"step {step:5d}/{args.max_steps} | "
                  f"train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | "
                  f"lr {lr:.2e} | {elapsed:.1f}s")
            t0 = time.time()

    # Save checkpoint
    ckpt = {"model": model.state_dict(), "config": config, "id2ch": id2ch, "ch2id": ch2id}
    torch.save(ckpt, args.ckpt_path)
    print(f"\nCheckpoint saved to {args.ckpt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Mamba character-level LM")
    p.add_argument("--data",          type=str,   default=None)
    p.add_argument("--seq_len",       type=int,   default=128)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--d_model",       type=int,   default=128)
    p.add_argument("--n_layers",      type=int,   default=4)
    p.add_argument("--d_state",       type=int,   default=16)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--max_steps",     type=int,   default=2000)
    p.add_argument("--warmup_steps",  type=int,   default=100)
    p.add_argument("--eval_interval", type=int,   default=200)
    p.add_argument("--eval_iters",    type=int,   default=20)
    p.add_argument("--ckpt_path",     type=str,   default="mamba_ckpt.pt")
    args = p.parse_args()
    train(args)
