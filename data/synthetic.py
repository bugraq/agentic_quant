"""
Sentetik veri üreteci — bilinen özelliklere sahip veri (Doküman 23.1).

Amaç: motoru sınamak. Motor, veri içine GÖMÜLÜ gerçek sinyali bulabilmeli
(momentum/reversal) ve TAMAMEN RASTGELE veride sahte alpha üretmemeli.

MarketData: alan adı -> DataFrame(index=tarih, columns=varlık). Wide panel.
Bu, ileride gerçek point-in-time veri adaptörüyle aynı arayüzü paylaşacak.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    fields: dict[str, pd.DataFrame]

    def get(self, name: str) -> pd.DataFrame:
        if name not in self.fields:
            raise KeyError(f"Veri alanı yok: {name}")
        return self.fields[name]

    @property
    def dates(self) -> pd.Index:
        return next(iter(self.fields.values())).index


def _prices_from_returns(returns: np.ndarray, dates, tickers) -> MarketData:
    """Getirilerden fiyat paneli ve türev alanları kur."""
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    close = pd.DataFrame(prices, index=dates, columns=tickers)
    rng = np.random.default_rng(0)
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, size=prices.shape),
                          index=dates, columns=tickers)
    return MarketData(fields={
        "close": close,
        "adjusted_close": close,
        "open": close.shift(1).bfill(),   # basitleştirme: açılış ~ önceki kapanış
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": volume,
        "dollar_volume": close * volume,
        "market_cap": close * 1e7,
    })


def split_by_fraction(md: MarketData, research_frac: float = 0.7) -> tuple[MarketData, MarketData]:
    """Zaman çizgisini araştırma / KİLİTLİ holdout olarak böl.

    Araştırma ajanı yalnızca ilk parçayı görür; holdout (son parça) ayrı bir
    servise kalır ve asla LLM'e/araştırmaya sızmaz (Doküman 2.2, 10.3).
    """
    n = len(md.dates)
    cut = int(n * research_frac)
    research = MarketData(fields={k: v.iloc[:cut].copy() for k, v in md.fields.items()})
    holdout = MarketData(fields={k: v.iloc[cut:].copy() for k, v in md.fields.items()})
    return research, holdout


def gen_random(n_sec=20, n_days=750, seed=0) -> MarketData:
    """Öngörülemez rastgele yürüyüş — hiçbir alpha OLMAMALI."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.02, size=(n_days, n_sec))
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_sec)]
    return _prices_from_returns(returns, dates, tickers)


def gen_cross_sectional_momentum(n_sec=20, n_days=750, seed=0,
                                 drift_spread=0.0008) -> MarketData:
    """
    Kalıcı kesitsel momentum: her varlığın gizli bir drift'i var. Geçmiş
    getiri, gelecekteki getiriyi kesitsel olarak öngörür. 'Geçmiş getiriye
    göre sırala, kazananı tut' stratejisi POZİTİF Sharpe vermeli.
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, drift_spread, size=n_sec)          # varlık başına gizli drift
    noise = rng.normal(0.0, 0.02, size=(n_days, n_sec))
    returns = mu[None, :] + noise
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_sec)]
    return _prices_from_returns(returns, dates, tickers)


def gen_short_term_reversal(n_sec=20, n_days=750, seed=0, phi=0.25) -> MarketData:
    """
    Kısa vadeli reversal: r_{t} = -phi * r_{t-1} + gürültü. 'Dünün getirisini
    tersine çevir' (negate) stratejisi POZİTİF Sharpe vermeli.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.02, size=(n_days, n_sec))
    returns = np.zeros_like(noise)
    returns[0] = noise[0]
    for t in range(1, n_days):
        returns[t] = -phi * returns[t - 1] + noise[t]
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_sec)]
    return _prices_from_returns(returns, dates, tickers)
