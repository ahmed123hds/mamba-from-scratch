"""
ops.py -- Core SSM operations

This file implements the mathematical heart of Mamba:
    1. ZOH discretization  (continuous → discrete)
    2. Sequential scan     (simple, easy to follow)
    3. Parallel scan       (fast, Hillis-Steele algorithm)

All the math here follows the notation in the Mamba paper:
    "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    Gu & Dao, 2023  (https://arxiv.org/abs/2312.00752)

The key recurrence is:
    h_k = Ā_k * h_{k-1} + B̄_k * x_k
    y_k = C_k · h_k

where Ā, B̄ come from discretizing the continuous-time SSM via ZOH.
"""

import math
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ZOH Discretization
# ─────────────────────────────────────────────────────────────────────────────

def discretize(A, B, delta):
    """
    Convert continuous-time SSM params (A, B) to discrete (Ā, B̄) using
    Zero-Order Hold (ZOH) discretization with step size delta.

    Continuous system:
        h'(t) = A h(t) + B x(t)

    ZOH formulas (exact):
        Ā = exp(Δ · A)
        B̄ = (ΔA)^{-1} (exp(ΔA) - I) ΔB

    In practice, Mamba uses the simplified Euler approximation for B̄:
        B̄ ≈ Δ · B
    This is fine because the learned Δ values turn out small, and it
    avoids a potentially ill-conditioned matrix inverse.

    Args:
        A     : (d_inner, d_state)         -- continuous state matrix
        B     : (batch, length, d_state)   -- input-dependent (selective)
        delta : (batch, length, d_inner)   -- step sizes (also selective)

    Returns:
        A_bar : (batch, length, d_inner, d_state)
        B_bar : (batch, length, d_inner, d_state)
    """
    # delta * A needs shape (B, L, d_inner, d_state)
    # delta: (B, L, d_inner), A: (d_inner, d_state)
    delta_A = torch.einsum('b l d, d n -> b l d n', delta, A)
    A_bar = torch.exp(delta_A)

    # Euler approximation: B̄ = Δ · B  →  shape (B, L, d_inner, d_state)
    # delta: (B, L, d_inner), B: (B, L, d_state)
    B_bar = torch.einsum('b l d, b l n -> b l d n', delta, B)

    return A_bar, B_bar


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Sequential Scan  (reference implementation)
# ─────────────────────────────────────────────────────────────────────────────

def ssm_scan_sequential(u, A_bar, B_bar, C):
    """
    Run the SSM recurrence step-by-step in a plain Python loop.
    Slow (O(L) serial steps), but easy to follow and good for debugging.

    Recurrence:
        h_k  = Ā_k * h_{k-1} + B̄_k * u_k    (element-wise on d_inner × d_state)
        y_k  = C_k @ h_k                       (inner product over d_state)

    We start with h_0 = 0 (zero initial state).

    Args:
        u     : (batch, length, d_inner)
        A_bar : (batch, length, d_inner, d_state)
        B_bar : (batch, length, d_inner, d_state)
        C     : (batch, length, d_state)

    Returns:
        y     : (batch, length, d_inner)
    """
    batch, length, d_inner, d_state = A_bar.shape
    h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []

    for k in range(length):
        # input contribution: B̄_k * u_k  →  outer over (d_inner, d_state)
        Bu = torch.einsum('b d n, b d -> b d n', B_bar[:, k], u[:, k])
        h  = A_bar[:, k] * h + Bu           # (batch, d_inner, d_state)

        # read from memory: y_k = C_k · h  (sum over d_state)
        y_k = torch.einsum('b d n, b n -> b d', h, C[:, k])
        ys.append(y_k)

    return torch.stack(ys, dim=1)           # (batch, length, d_inner)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Parallel Scan  (fast, O(log L) depth)
# ─────────────────────────────────────────────────────────────────────────────

def ssm_scan_parallel(u, A_bar, B_bar, C):
    """
    Compute all SSM states in parallel using a prefix-scan.

    The recurrence h_k = a_k * h_{k-1} + b_k is a LINEAR recurrence,
    which means it has an associative binary operator:

        (a₂, b₂) ⊗ (a₁, b₁)  =  (a₂·a₁,  a₂·b₁ + b₂)

    Because ⊗ is associative, we can use the Hillis-Steele parallel prefix
    scan to compute all prefix products h_1, h_2, ..., h_L simultaneously
    in O(log L) parallel steps instead of O(L) serial steps.

    Args:
        u     : (batch, length, d_inner)
        A_bar : (batch, length, d_inner, d_state)
        B_bar : (batch, length, d_inner, d_state)
        C     : (batch, length, d_state)

    Returns:
        y     : (batch, length, d_inner)
    """
    # b_k = B̄_k * u_k  →  (batch, length, d_inner, d_state)
    b = torch.einsum('b l d n, b l d -> b l d n', B_bar, u)

    # parallel prefix scan returns all h_k
    h_all = _hillis_steele_scan(A_bar, b)   # (batch, length, d_inner, d_state)

    # y_k = C_k · h_k  (inner product over d_state)
    y = torch.einsum('b l d n, b l n -> b l d', h_all, C)
    return y


def _hillis_steele_scan(a, b):
    """
    Hillis-Steele parallel inclusive prefix scan for the recurrence:
        h_k = a_k * h_{k-1} + b_k,   h_0 = 0

    At each doubling step s, we combine position k with position k-s:
        a[k] ← a[k] * a[k-s]
        b[k] ← a[k] * b[k-s] + b[k]

    After ceil(log2 L) steps, b[k] holds the cumulative state h_k.

    The identity element for the operator is (1, 0):  any pair composed
    with (1, 0) on the right is unchanged -- used to pad shifted values.

    a, b : (batch, length, d_inner, d_state)
    returns h_all : (batch, length, d_inner, d_state)
    """
    batch, L, d_inner, d_state = a.shape

    # work on clones so we don't touch the original tensors
    a = a.clone()
    b = b.clone()

    step = 1
    while step < L:
        # Shift a and b right by 'step' positions, padding left with identity (1, 0)
        a_shifted = torch.cat([
            torch.ones (batch, step, d_inner, d_state, device=a.device, dtype=a.dtype),
            a[:, :-step]
        ], dim=1)
        b_shifted = torch.cat([
            torch.zeros(batch, step, d_inner, d_state, device=b.device, dtype=b.dtype),
            b[:, :-step]
        ], dim=1)

        # Combine: (a[k], b[k]) ⊗ (a_shifted[k], b_shifted[k])
        # NOTE: we must read a_shifted/b_shifted from BEFORE updating a/b
        new_a = a * a_shifted
        new_b = a * b_shifted + b

        a, b = new_a, new_b
        step *= 2

    # b now holds h_1, h_2, ..., h_L
    return b
