"""
DataAdapter — veri kaynağı soyutlaması (değiştirilebilir kutu).

Sentetik üreteç ve gerçek piyasa verisi AYNI arayüzü paylaşır: load() ->
MarketData (wide panel: alan -> DataFrame[tarih, varlık]). Kaynak config'ten
seçilir; backtest/evaluator hangi kaynağı kullandığını bilmez.

SURVIVORSHIP UYARISI (Doküman 7): yfinance sabit bir güncel ticker listesiyle
çalışır — delist olmuş şirketler eksiktir, bu da survivorship bias yaratır.
Gerçek point-in-time sistem için tarihsel endeks üyeliği gerekir. Bu adapter
bir İSKELET/demo'dur ve bu sınırı açıkça taşır.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from data.synthetic import (
    MarketData,
    gen_cross_sectional_momentum,
    gen_random,
    gen_short_term_reversal,
)

_STD_FIELDS = ["open", "high", "low", "close", "adjusted_close",
               "volume", "dollar_volume", "market_cap"]


class DataAdapter(Protocol):
    def load(self) -> MarketData: ...


class SyntheticAdapter:
    """Sentetik üreteçleri DataAdapter arayüzüne sarar."""

    _GENERATORS = {
        "momentum": gen_cross_sectional_momentum,
        "reversal": gen_short_term_reversal,
        "random": gen_random,
    }

    def __init__(self, kind: str = "momentum", **kwargs) -> None:
        if kind not in self._GENERATORS:
            raise ValueError(f"Bilinmeyen sentetik tür: {kind}")
        self._gen = self._GENERATORS[kind]
        self._kwargs = kwargs

    def load(self) -> MarketData:
        return self._gen(**self._kwargs)


class YFinanceAdapter:
    """
    Yahoo Finance OHLCV (yfinance). SURVIVORSHIP BIAS taşır (yukarıdaki uyarı).
    Fundamental/market_cap yoktur; market_cap = close (placeholder).
    """

    def __init__(self, tickers: list[str], start: str, end: str) -> None:
        self.tickers = tickers
        self.start = start
        self.end = end

    def load(self) -> MarketData:
        import yfinance as yf

        raw = yf.download(self.tickers, start=self.start, end=self.end,
                          progress=False, auto_adjust=False)
        if raw.empty:
            raise RuntimeError("yfinance boş veri döndürdü (rate-limit veya ticker hatası?).")

        def _fld(name: str) -> pd.DataFrame:
            df = raw[name].copy()
            if isinstance(df, pd.Series):        # tek ticker durumu
                df = df.to_frame(self.tickers[0])
            return df

        close = _fld("Close")
        volume = _fld("Volume")
        fields = {
            "open": _fld("Open"),
            "high": _fld("High"),
            "low": _fld("Low"),
            "close": close,
            "adjusted_close": _fld("Adj Close"),
            "volume": volume,
            "dollar_volume": close * volume,
            "market_cap": close,   # placeholder — shares outstanding yok
        }
        # Temizlik: iş günlerine hizala, kısıtlı ffill (delist boşluklarını kapatma)
        fields = {k: v.sort_index().ffill(limit=5) for k, v in fields.items()}
        return MarketData(fields=fields)


def make_adapter(config: dict) -> DataAdapter:
    """configs/data.yaml'a göre veri adaptörü kur."""
    config = config or {}
    source = config.get("source", "synthetic")
    if source == "synthetic":
        s = config.get("synthetic", {})
        return SyntheticAdapter(
            kind=s.get("kind", "momentum"),
            n_sec=int(s.get("n_sec", 20)),
            n_days=int(s.get("n_days", 750)),
            seed=int(s.get("seed", 7)))
    if source == "yfinance":
        y = config.get("yfinance", {})
        return YFinanceAdapter(tickers=y["tickers"], start=y["start"], end=y["end"])
    raise NotImplementedError(f"Bilinmeyen veri kaynağı: {source!r}")
