"""Veri katmanı — şimdilik sentetik üreteçler; ileride point-in-time adaptörü."""
from data.synthetic import (
    MarketData,
    gen_cross_sectional_momentum,
    gen_random,
    gen_short_term_reversal,
)

__all__ = [
    "MarketData",
    "gen_random",
    "gen_cross_sectional_momentum",
    "gen_short_term_reversal",
]
