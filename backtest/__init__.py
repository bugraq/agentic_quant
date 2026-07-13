"""Deterministik backtest çekirdeği — evaluator + vectorized motor."""
from backtest.engine import run_backtest
from backtest.evaluator import evaluate_signal

__all__ = ["run_backtest", "evaluate_signal"]
