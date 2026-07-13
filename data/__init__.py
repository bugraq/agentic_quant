"""Veri katmanı — sentetik üreteçler + DataAdapter (sentetik/gerçek tak-çalıştır)."""
from data.adapter import DataAdapter, SyntheticAdapter, YFinanceAdapter, make_adapter
from data.synthetic import (
    MarketData,
    gen_cross_sectional_momentum,
    gen_random,
    gen_short_term_reversal,
    split_by_fraction,
)

__all__ = [
    "MarketData",
    "split_by_fraction",
    "gen_random",
    "gen_cross_sectional_momentum",
    "gen_short_term_reversal",
    "DataAdapter",
    "SyntheticAdapter",
    "YFinanceAdapter",
    "make_adapter",
]
