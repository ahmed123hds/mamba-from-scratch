"""
block.py -- The Mamba Block

This is the building block of the Mamba architecture. Each block takes an
input sequence, runs it through a selective SSM (S6) with gating, and adds
a residual connection. Stack N of these to make a full model.

Architecture (one block):

    x  ─────────────────────────────────────────────────────────────┐
    │                                                               │ (residual)
    LayerNorm                                                       │
    │                                                               │
    ├──── Linear (expand) ──── Conv1d ──── SiLU ──── S6 ──── ┐    │
    │                                                          ×    │
    └──── Linear (expand) ──────────────────────── SiLU ──── ┘    │
                                                               │    │
                                                 Linear (project)   │
                                                               │    │
                                                               └────┘ → output

The two-branch structure is where "gated" comes from: the SSM output is
multiplied element-wise by a learned gate. This is similar to Gated MLPs
and SwiGLU activations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import discretize, ssm_scan_parallel, ssm_scan_sequential


class SelectiveSSM(nn.Module):
    """
    The Selective State Space Model (S6) from the Mamba paper.

    What makes it "selective": the key parameters Δ, B, C are computed
    from the input x_k at each position -- so the model can dynamically
    decide what to remember, what to ignore, and how fast to transition.

    In contrast, the matrix A is a global learned parameter (not selective).
    It's parameterized as log(-A) to ensure A stays negative, keeping
    the system stable (eigenvalues with negative real parts → decaying state).

    Dimensions:
        d_inner  : expanded inner dimension  (= expand * d_model)
        d_state  : SSM state dimension N     (typically 16)
        dt_rank  : rank for Δ projection     (≈ d_model / 16)
    """

    def __init__(self, d_inner, d_state=16, dt_rank=None, dt_min=0.001,
                 dt_max=0.1, dt_init="random", use_parallel_scan=True):
        super().__init__()
        self.d_inner  = d_inner
        self.d_state  = d_state
        self.dt_rank  = dt_rank or math.ceil(d_inner / 16)
        self.use_parallel_scan = use_parallel_scan

        # x_proj maps inner activations to [Δ (dt_rank), B (d_state), C (d_state)]
        # This is the "selection" mechanism -- all three depend on the input
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)

        # dt_proj expands Δ from dt_rank back up to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # Initialize dt_proj following the paper: dt ~ Uniform(dt_min, dt_max)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A is the continuous-time state matrix, shape (d_inner, d_state)
        # Parameterize as log so that A = -exp(log_A) stays negative always
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_inner, 1)
        self.log_A = nn.Parameter(torch.log(A))    # will negate during forward

        # D is the skip/residual connection (direct feedthrough x → y)
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, u):
        """
        Args:
            u : (batch, length, d_inner)
        Returns:
            y : (batch, length, d_inner)
        """
        batch, length, d_inner = u.shape

        # ── Step 1: Compute input-dependent parameters Δ, B, C ──────────────
        x_dbl = self.x_proj(u)   # (B, L, dt_rank + 2*d_state)

        delta, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        # delta: (B, L, dt_rank), B: (B, L, d_state), C: (B, L, d_state)

        # expand delta from dt_rank to d_inner, then ensure positivity with softplus
        delta = F.softplus(self.dt_proj(delta))   # (B, L, d_inner)

        # ── Step 2: Get discrete Ā, B̄ via ZOH ──────────────────────────────
        A = -torch.exp(self.log_A)   # keep A negative for stability
        A_bar, B_bar = discretize(A, B, delta)
        # A_bar, B_bar: (B, L, d_inner, d_state)

        # ── Step 3: Run the SSM recurrence ──────────────────────────────────
        if self.use_parallel_scan:
            y = ssm_scan_parallel(u, A_bar, B_bar, C)
        else:
            y = ssm_scan_sequential(u, A_bar, B_bar, C)
        # y: (B, L, d_inner)

        # ── Step 4: Skip connection through D ───────────────────────────────
        # y = C h + D x  (additive feedthrough, D is a learned per-channel scalar)
        y = y + u * self.D

        return y


class MambaBlock(nn.Module):
    """
    One complete Mamba block: normalization + SSM branch + gate branch + residual.

    The expand factor (default 2) determines d_inner = expand * d_model.
    The conv1d (width d_conv) provides a short-range local context window
    before the SSM handles long-range dependencies.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 dt_rank=None, use_parallel_scan=True):
        super().__init__()
        self.d_model  = d_model
        self.d_inner  = int(expand * d_model)
        self.d_state  = d_state
        self.d_conv   = d_conv

        # Pre-normalization (RMS norm or LayerNorm -- using LayerNorm for simplicity)
        self.norm = nn.LayerNorm(d_model)

        # Single linear that produces both SSM-branch input AND gate in one shot
        # Output is split into [ssm_input (d_inner), gate (d_inner)]
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Short-range local convolution -- operates on the SSM branch only
        # groups=d_inner makes it a depthwise conv (one filter per channel)
        self.conv1d = nn.Conv1d(
            in_channels  = self.d_inner,
            out_channels = self.d_inner,
            kernel_size  = d_conv,
            padding      = d_conv - 1,      # causal padding on the left
            groups       = self.d_inner,
            bias         = True,
        )

        # The selective SSM (S6)
        self.ssm = SelectiveSSM(
            d_inner           = self.d_inner,
            d_state           = d_state,
            dt_rank           = dt_rank,
            use_parallel_scan = use_parallel_scan,
        )

        # Project back down to d_model after gating
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x : (batch, length, d_model)
        Returns:
            out : (batch, length, d_model)
        """
        residual = x
        x = self.norm(x)

        # Split into SSM branch (u) and gate branch (v)
        u, v = self.in_proj(x).chunk(2, dim=-1)
        # u, v : (batch, length, d_inner)

        # ── SSM branch ──────────────────────────────────────────────────────
        # Conv1d expects (batch, channels, length), so we transpose
        u = u.transpose(1, 2)                        # (B, d_inner, L)
        u = self.conv1d(u)[..., :u.shape[-1]]        # crop causal padding
        u = u.transpose(1, 2)                        # (B, L, d_inner)
        u = F.silu(u)

        u = self.ssm(u)    # (B, L, d_inner)

        # ── Gate branch ─────────────────────────────────────────────────────
        v = F.silu(v)

        # Multiplicative gating: SSM output × gate
        y = u * v

        # Project back and add residual
        out = self.out_proj(y) + residual
        return out
