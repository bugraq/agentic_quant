"""
Point-in-time S&P 500 üyeliği (Doküman 4/7 — survivorship bias düzeltmesi).

Bugünün endeks listesini geçmişe uygulamak survivorship bias'tır: bugün hayatta
olanlar tanım gereği "kazananlar"dır. Bu modül, Wikipedia'nın S&P 500 bileşen
DEĞİŞİKLİKLERİ tablosundan geriye yürüyerek her tarih için gerçek üye kümesini
kurar: bir hisse yalnızca O TARİHTE endeksteyken işlem görebilir.

Yöntem: bugünkü listeden başla; değişiklik olaylarını tarihe göre GERİYE doğru
uygula (ekleme olayının öncesinde hisse endekste YOKTU -> çıkar; çıkarma
olayının öncesinde VARDI -> geri ekle).

DÜRÜST SINIRLAR (bunlar hâlâ açık):
  - Kaynak Wikipedia'dır (kürasyonlu ama resmi değil; erken yıllar eksik
    olabilir). Akademik nihai sonuç için CRSP/Compustat gerekir.
  - Endeksten çıkmış hisselerin FİYATI Yahoo'da olmayabilir (özellikle iflas/
    satın alma sonrası). Adapter kaç ticker'ın verisiz kaldığını raporlar.
  - Delisting return (CRSP tarzı son gün kaybı) modellenmiyor; pozisyon son
    fiyattan 0 getiriyle çıkar (iyimser taraf — belgelendi).

Rekonstrüksiyon saf fonksiyondur (offline test edilir); yalnızca
fetch_wikipedia_sp500() ağa çıkar ve sonuç CSV'ye cache'lenir.
"""
from __future__ import annotations

import os
import re

import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def yahoo_symbol(ticker: str) -> str:
    """Wikipedia/S&P sembolü -> Yahoo sembolü (BRK.B -> BRK-B)."""
    return ticker.strip().replace(".", "-")


def _clean_ticker(x) -> str | None:
    if not isinstance(x, str):
        return None
    t = x.strip().upper()
    # dipnot işaretleri / boşluklar
    t = re.sub(r"\[.*?\]", "", t).strip()
    return t or None


def normalize_changes(raw: pd.DataFrame) -> pd.DataFrame:
    """Wikipedia 'Selected changes' tablosunu düz olay listesine çevir.

    Girdi: multi-index kolonlu ham tablo (Date / Added.Ticker / Removed.Ticker).
    Çıktı: DataFrame[date, added, removed] — added/removed tek ticker ya da None.
    """
    df = raw.copy()
    # Multi-index kolonları düzleştir: ('Date','Date') -> 'date' vb.
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for a, b in df.columns:
            a, b = str(a).lower(), str(b).lower()
            if a == b or "unnamed" in b:
                cols.append(a)
            else:
                cols.append(f"{a}_{b}")
        df.columns = cols
    else:
        df.columns = [str(c).lower() for c in df.columns]

    date_col = next(c for c in df.columns if "date" in c)   # 'date'/'effective date'
    added_col = next((c for c in df.columns if c.startswith("added") and "ticker" in c),
                     next((c for c in df.columns if c == "added"), None))
    removed_col = next((c for c in df.columns if c.startswith("removed") and "ticker" in c),
                       next((c for c in df.columns if c == "removed"), None))

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "added": df[added_col].map(_clean_ticker) if added_col else None,
        "removed": df[removed_col].map(_clean_ticker) if removed_col else None,
    })
    out = out.dropna(subset=["date"])
    out = out[~(out["added"].isna() & out["removed"].isna())]
    return out.sort_values("date").reset_index(drop=True)


def reconstruct_membership(current_tickers: list[str], changes: pd.DataFrame,
                           start: str, end: str) -> pd.DataFrame:
    """Bugünkü liste + değişiklik olayları -> point-in-time üyelik matrisi.

    Döndürür: DataFrame[iş günü × ticker] (bool). Ticker'lar Yahoo formatında.
    Olay semantiği: `date` gününden İTİBAREN eklenen üyedir / çıkarılan değildir.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    dates = pd.bdate_range(start_ts, end_ts)

    events = [(row.date, _clean_ticker(row.added), _clean_ticker(row.removed))
              for row in changes.itertuples()]
    events.sort(key=lambda e: e[0], reverse=True)   # yeniden eskiye

    members = {t for t in (_clean_ticker(x) for x in current_tickers) if t}

    # 1) Bugünden pencere sonuna GERİ sar (pencere dışındaki olayları geri al)
    i = 0
    while i < len(events) and events[i][0] > end_ts:
        _, added, removed = events[i]
        if added:
            members.discard(added)
        if removed:
            members.add(removed)
        i += 1

    # 2) Pencere içinde günleri geriye yürü; olay tarihini geçince geri al
    seen: set = set(members)
    day_sets: dict = {}
    for d in reversed(dates):
        while i < len(events) and events[i][0] > d:
            _, added, removed = events[i]
            if added:
                members.discard(added)
            if removed:
                members.add(removed)
            i += 1
        day_sets[d] = frozenset(members)
        seen |= members

    cols = sorted(yahoo_symbol(t) for t in seen)
    data = {c: [] for c in cols}
    for d in dates:
        s = {yahoo_symbol(t) for t in day_sets[d]}
        for c in cols:
            data[c].append(c in s)
    return pd.DataFrame(data, index=dates, columns=cols)


def fetch_wikipedia_sp500() -> tuple[list[str], pd.DataFrame]:
    """Wikipedia'dan (bugünkü üyeler, değişiklik olayları) çek. AĞ GEREKTİRİR."""
    import io

    import requests

    # Wikipedia, kütüphane varsayılan user-agent'larını 403'ler; kimlik ver.
    resp = requests.get(WIKI_URL, timeout=30, headers={
        "User-Agent": "agentic-quant-research/0.1 (akademik staj projesi; "
                      "point-in-time S&P 500 uyeligi)"})
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    current = tables[0]
    sym_col = next(c for c in current.columns if str(c).lower() in ("symbol", "ticker"))
    current_tickers = [t for t in current[sym_col].map(_clean_ticker) if t]
    changes = normalize_changes(tables[1])
    return current_tickers, changes


def load_membership(start: str, end: str, cache_dir: str = "data") -> pd.DataFrame:
    """Üyelik matrisini cache'ten yükle; yoksa Wikipedia'dan kurup cache'le."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"sp500_pit_membership_{start}_{end}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.astype(bool)
    current, changes = fetch_wikipedia_sp500()
    mem = reconstruct_membership(current, changes, start, end)
    mem.to_csv(path)
    return mem
