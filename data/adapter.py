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

import os
from typing import Protocol

import pandas as pd

from data.synthetic import (
    MarketData,
    gen_cross_sectional_momentum,
    gen_random,
    gen_short_term_reversal,
)

# market_cap standart alan DEĞİL: kaynakta shares outstanding yoksa üretilmez.
_STD_FIELDS = ["open", "high", "low", "close", "adjusted_close",
               "volume", "dollar_volume"]


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
    Fundamental/market_cap yoktur — market_cap alanı BİLEREK üretilmez.
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
            # market_cap BİLEREK YOK: shares outstanding verisi olmadan
            # market_cap üretilemez; fiyatı market_cap diye sunmak sahte
            # "size faktörü" testine yol açar (etiket-sinyal dürüstlüğü).
        }
        # Temizlik: iş günlerine hizala, kısıtlı ffill (delist boşluklarını kapatma)
        fields = {k: v.sort_index().ffill(limit=5) for k, v in fields.items()}
        return MarketData(fields=fields)


def _inject_delisting_shock(fields: dict, price_keys: list[str], shock: float) -> int:
    """Delist olan hisselere (verisi pencere sonundan önce biten) delisting şoku uygula.

    Her fiyat alanı için: hissenin gerçek son geçerli barından SONRAKİ ilk bara
    son_fiyat*(1-şok) yazılır (o an NaN olmalı — yani gerçek delisting). Pozisyon
    o barda şok getirisini (-şok) alır. Değişiklik yerinde (in-place) yapılır.
    Döndürür: şok uygulanan (delist olan) hisse sayısı.
    """
    close = fields["close"]
    dates = close.index
    last_date = dates[-1]
    delisted = 0
    for tkr in close.columns:
        lv = close[tkr].last_valid_index()
        if lv is None or lv >= last_date:
            continue                       # hiç veri yok ya da sona kadar yaşıyor
        pos = dates.get_loc(lv)
        if pos + 1 >= len(dates):
            continue
        shock_bar = dates[pos + 1]
        # Yalnızca gerçek boşluk (o bar zaten NaN) ise şok yaz — veri varsa dokunma
        if pd.notna(close.at[shock_bar, tkr]):
            continue
        for key in price_keys:
            if tkr in fields[key].columns:
                base = fields[key][tkr].loc[lv]
                if pd.notna(base):
                    fields[key].at[shock_bar, tkr] = base * (1.0 - shock)
        delisted += 1
    return delisted


class SP500PointInTimeAdapter:
    """
    Point-in-time S&P 500 evreni (Doküman 4/7 — survivorship DÜZELTMESİ).

    Wikipedia değişiklik tarihçesinden her tarihteki GERÇEK üye kümesi kurulur
    (data.pit_universe); pencere içinde bir gün bile üye olmuş HER ticker'ın
    fiyatı indirilir (bugün endekste olmayanlar dahil) ve `index_membership`
    alanı üretilir. Backtest motoru bu alanla hisseyi yalnızca üye olduğu
    günlerde işleme sokar.

    KALAN DÜRÜST SINIRLAR:
      - Delist olmuş bazı ticker'ların fiyatı Yahoo'da hiç yok; kaçının
        verisiz kaldığı yüklemede raporlanır (tam çözüm CRSP ister).
      - Delisting return modellenmez: veri kesilince pozisyon son fiyattan
        0 getiriyle çıkar (iyimser taraf).
    """

    _BATCH = 100   # yfinance tek istekte çok ticker'ı sever ama batch daha sağlam

    def __init__(self, start: str, end: str, cache_dir: str = "data",
                 delisting_shock: float = 0.30) -> None:
        self.start, self.end, self.cache_dir = str(start), str(end), cache_dir
        # Delist olan hisse pozisyondan bu kadar KAYIPLA çıkar (Doküman 7 /
        # "Beyond Agent Architecture" eleştirisi). CRSP delisting return'ün
        # tutucu placeholder'ı; 0.0 = eski iyimser davranış (kayıpsız çıkış).
        self.delisting_shock = float(delisting_shock)

    def _download_prices(self, tickers: list[str]) -> pd.DataFrame:
        cache = os.path.join(self.cache_dir,
                             f"sp500_pit_prices_{self.start}_{self.end}.pkl")
        if os.path.exists(cache):
            return pd.read_pickle(cache)
        import time

        import yfinance as yf

        # (field, ticker) -> seri; tekrar denemede yalnızca verisi olmayan dolar.
        cols: dict[tuple, pd.Series] = {}

        def absorb(raw: pd.DataFrame) -> None:
            if raw is None or raw.empty:
                return
            for c in raw.columns:
                s = raw[c]
                if c not in cols or cols[c].isna().all():
                    cols[c] = s

        for i in range(0, len(tickers), self._BATCH):
            batch = tickers[i:i + self._BATCH]
            absorb(yf.download(batch, start=self.start, end=self.end,
                               progress=False, auto_adjust=False))
            print(f"  [pit] fiyat indiriliyor: {min(i+self._BATCH, len(tickers))}"
                  f"/{len(tickers)} ticker", flush=True)
        if not cols:
            raise RuntimeError("yfinance hiç veri döndürmedi (rate-limit?).")

        # RETRY: 'no timezone found' çoğu zaman delist değil RATE-LIMIT demektir
        # (aktif hisseler de düşüyor). Verisi hiç gelmeyenleri bekleyip tekrar
        # dene; gerçekten delist olanlar zaten yine boş döner.
        def _missing() -> list[str]:
            got = {t for (f, t), s in cols.items()
                   if f == "Close" and not s.isna().all()}
            return [t for t in tickers if t not in got]

        for attempt in (1, 2):
            miss = _missing()
            if not miss:
                break
            print(f"  [pit] retry {attempt}: {len(miss)} ticker verisiz "
                  f"(rate-limit olabilir), 20 sn bekleyip tekrar...", flush=True)
            time.sleep(20)
            for i in range(0, len(miss), 50):
                absorb(yf.download(miss[i:i + 50], start=self.start, end=self.end,
                                   progress=False, auto_adjust=False))

        merged = pd.DataFrame(cols).sort_index()
        merged.columns = pd.MultiIndex.from_tuples(merged.columns)
        merged.to_pickle(cache)
        return merged

    def load(self) -> MarketData:
        from data.pit_universe import load_membership, load_sectors

        membership = load_membership(self.start, self.end, self.cache_dir)
        tickers = list(membership.columns)
        raw = self._download_prices(tickers)

        def _fld(name: str) -> pd.DataFrame:
            df = raw[name]
            return df.to_frame(tickers[0]) if isinstance(df, pd.Series) else df

        close, volume = _fld("Close"), _fld("Volume")
        # Verisi hiç olmayan üyeleri raporla (DÜRÜSTLÜK: sessizce yutma)
        have = [t for t in tickers if t in close.columns and close[t].notna().any()]
        missing = sorted(set(tickers) - set(have))
        if missing:
            print(f"  [pit] UYARI: {len(missing)}/{len(tickers)} üyenin fiyatı "
                  f"Yahoo'da yok (delist; kalan survivorship kalıntısı). "
                  f"Örnek: {missing[:8]}")

        memb = (membership.reindex(close.index).ffill()
                .fillna(False).astype(float))
        fields = {
            "open": _fld("Open")[have],
            "high": _fld("High")[have],
            "low": _fld("Low")[have],
            "close": close[have],
            "adjusted_close": _fld("Adj Close")[have],
            "volume": volume[have],
            "dollar_volume": (close * volume)[have],
            "index_membership": memb[have],
        }
        # DELISTING ŞOKU (ffill'DEN ÖNCE): pencere sonundan önce verisi biten
        # (delist olan) hisse için, gerçek son fiyattan sonraki ilk bara
        # fiyat*(1-şok) yaz. Pozisyon o barda kaybı yer, sonra ffill kısa kuyruğu
        # taşır. Böylece batan şirketi tutmanın maliyeti modellenir (survivorship
        # kalıntısını azaltır). index_membership ve volume'a dokunulmaz.
        if self.delisting_shock > 0:
            n_delisted = _inject_delisting_shock(
                fields, ["open", "high", "low", "close", "adjusted_close"],
                self.delisting_shock)
            if n_delisted:
                print(f"  [pit] delisting şoku (-%{self.delisting_shock*100:.0f}) "
                      f"{n_delisted} delist olan hisseye uygulandı.")
        # Kısıtlı ffill (delist boşluklarını KAPATMAZ, kısa boşlukları düzeltir)
        fields = {k: (v.sort_index().ffill(limit=5) if k != "index_membership" else v)
                  for k, v in fields.items()}
        # Sektör haritası (GICS) — gerçek sektör-nötralizasyonu için (yalnızca
        # işlem gören ticker'lar). Çekilemezse None (piyasa-nötre düşer).
        try:
            all_sectors = load_sectors(self.cache_dir)
            sectors = {t: all_sectors[t] for t in have if t in all_sectors} or None
        except Exception as e:  # noqa: BLE001 — sektör yoksa araştırma durmasın
            print(f"  [pit] sektör haritası çekilemedi ({type(e).__name__}); "
                  f"sektör-nötr piyasa-nötre düşecek.")
            sectors = None
        return MarketData(fields=fields, sectors=sectors)


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
    if source == "sp500_pit":
        p = config.get("sp500_pit", {})
        return SP500PointInTimeAdapter(start=p["start"], end=p["end"],
                                       cache_dir=p.get("cache_dir", "data"),
                                       delisting_shock=float(p.get("delisting_shock", 0.30)))
    raise NotImplementedError(f"Bilinmeyen veri kaynağı: {source!r}")
