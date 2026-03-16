"""
model.py -- Full Mamba Language Model

Stacks MambaBlocks on top of a token embedding layer, followed by a
final LayerNorm and a linear head that projects back to vocabulary logits.

The embedding and the output head share weights (weight tying) -- this is
standard practice in language models since Inan et al. (2016) showed it
reduces parameters and improves perplexity.

Usage:
    config = MambaConfig(vocab_size=256, d_model=128, n_layers=4)
    model  = Mamba(config)
    logits = model(token_ids)   # (batch, length, vocab_size)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .block import MambaBlock


@dataclass
class MambaConfig:
    """
    Hyperparameters for a Mamba language model.

    d_model  : embedding / hidden dimension
    n_layers : how many MambaBlocks to stack
    d_state  : SSM state size N (paper uses 16)
    d_conv   : local conv width (paper uses 4)
    expand   : expansion factor inside each block (paper uses 2)
    dt_rank  : rank of Δ projection; None → ceil(d_model / 16)
    vocab_size: number of tokens
    pad_id   : token id to ignore in loss (optional)
    """
    vocab_size        : int   = 256
    d_model           : int   = 128
    n_layers          : int   = 4
    d_state           : int   = 16
    d_conv            : int   = 4
    expand            : int   = 2
    dt_rank           : Optional[int] = None
    pad_id            : int   = -1
    use_parallel_scan : bool  = True


class Mamba(nn.Module):
    """
    Mamba language model:
        Embedding → [MambaBlock × n_layers] → LayerNorm → LM Head

    The LM head shares weights with the embedding matrix (weight tying).
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            MambaBlock(
                d_model           = config.d_model,
                d_state           = config.d_state,
                d_conv            = config.d_conv,
                expand            = config.expand,
                dt_rank           = config.dt_rank,
                use_parallel_scan = config.use_parallel_scan,
            )
            for _ in range(config.n_layers)
        ])

        self.norm_f = nn.LayerNorm(config.d_model)

        # LM head -- project hidden states to vocab logits
        # Weight tying: share weights with embedding
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight   # tie weights

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Standard LM weight initialization:
          - Linear and Embedding: N(0, 0.02)
          - LayerNorm: bias=0, weight=1
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        Args:
            idx     : (batch, length)  -- token indices
            targets : (batch, length)  -- target token indices for loss (optional)

        Returns:
            logits  : (batch, length, vocab_size)
            loss    : scalar cross-entropy loss, or None if targets not provided
        """
        x = self.embedding(idx)     # (B, L, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)          # final layer norm
        logits = self.lm_head(x)    # (B, L, vocab_size)

        loss = None
        if targets is not None:
            # standard cross-entropy over flattened sequence
            B, L, V = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(B * L, V),
                targets.view(B * L),
                ignore_index=self.config.pad_id,
            )

        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check -- run this file directly to verify shapes
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = MambaConfig(
        vocab_size = 256,
        d_model    = 128,
        n_layers   = 4,
        d_state    = 16,
    )
    model = Mamba(config)
    print(f"Parameters: {model.count_parameters():,}")

    x = torch.randint(0, 256, (2, 64))      # batch=2, length=64
    t = torch.randint(0, 256, (2, 64))

    logits, loss = model(x, t)
    print(f"Logits shape : {logits.shape}")  # expect (2, 64, 256)
    print(f"Loss         : {loss.item():.4f}")
