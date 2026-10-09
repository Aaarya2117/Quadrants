"""ML Model Swap: compare, atomically swap, and roll back models with identical architecture.

This package holds the CLI and demo orchestration. The swap engine sits behind
the SwapEngine contract in modelswap.engine, so the real runtime can be plugged
in without changing the CLI or the demo.
"""

__version__ = "0.1.0"
