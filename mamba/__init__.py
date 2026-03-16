"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces
Implemented from scratch following the paper by Albert Gu and Tri Dao (2023).

Modules:
    ops   -- core SSM math: discretization, sequential and parallel scans
    block -- the Mamba block (SSM + gating + normalization)
    model -- full language model stacking Mamba blocks
"""

from .block import MambaBlock
from .model import Mamba, MambaConfig
