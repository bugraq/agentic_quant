"""
Benzerlik ve yenilik kontrolü (Doküman 14).

İki stratejinin doğal dil açıklaması farklı olsa bile, YAPISI veya ÜRETTİĞİ
SİNYAL çok benziyorsa aynı araştırma sayılır. Tekrarları elemek hem bütçeyi
korur hem de multiple-testing muhasebesini dürüst tutar (aynı fikri N kez
saymak istatistiği bozar).

Üç seviye (Doküman 14.1-14.3):
  - Metinsel (text): hipotez açıklamasının (claim+mekanizma) leksikal cosine
    benzerliği. Aynı fikri farklı kelimelerle tekrar etmeyi yakalar. NOT: bu
    leksikal bir yaklaşımdır; tam anlamsal embedding (pgvector) ileriki iş
    (Doküman 14.1). Yanlış-pozitif reddi önlemek için eşik ÇOK yüksek (near-verbatim).
  - Yapısal (ast): sinyal ağacının operatör çok-kümesi Jaccard benzerliği.
    Backtest'ten ÖNCE, veri gerektirmez -> bütçe korur.
  - Davranışsal (corr): üretilen sinyal panellerinin korelasyonu.

Kural (Doküman 14):  ast > 0.90  VEYA  |corr| > 0.95  VEYA  text > 0.97 -> duplicate.
"""
from __future__ import annotations

import re
from collections import Counter
from math import sqrt
from typing import Optional

import numpy as np
import pandas as pd

from contracts.dsl import Expression
from contracts.hypothesis_spec import HypothesisSpec

AST_THRESHOLD = 0.90
CORR_THRESHOLD = 0.95
TEXT_THRESHOLD = 0.97   # near-verbatim; leksikal olduğu için yüksek tutulur

# Çok sık geçen, ayırt etmeyen kelimeler (leksikal gürültü)
_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "with", "for",
         "is", "are", "that", "this", "by", "as", "be", "will", "tend", "over",
         "ve", "ile", "bir", "bu", "de", "da", "olan", "için", "gibi"}


def _tokens(expr: Expression) -> Counter:
    """Sinyal ağacını operatör çok-kümesine indir (sıra bağımsız)."""
    tok: Counter = Counter()
    if expr.op == "field":
        tok[f"field:{expr.field}"] += 1
    elif expr.op == "const":
        tok["const"] += 1
    elif expr.op == "feature_ref":
        tok[f"ref:{expr.name}"] += 1
    else:
        key = f"{expr.op}:{expr.window}" if expr.window is not None else expr.op
        tok[key] += 1
    for inp in expr.inputs:
        sub = inp if isinstance(inp, Expression) else Expression(op="feature_ref", name=inp)
        tok.update(_tokens(sub))
    return tok


def _jaccard(a: Counter, b: Counter) -> float:
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


def _text_tokens(text: str) -> Counter:
    """Metni kelime frekans sayacına indir (küçük harf, stopword'süz, kısa atılır)."""
    words = re.findall(r"[a-zçğıöşü0-9]+", (text or "").lower())
    return Counter(w for w in words if len(w) > 2 and w not in _STOP)


def _hyp_text(hyp: HypothesisSpec) -> Counter:
    """Hipotezin metinsel kimliği: başlık + iddia + ekonomik mekanizma açıklaması."""
    mech = hyp.economic_mechanism
    return _text_tokens(" ".join([hyp.title, hyp.claim, mech.type, mech.description]))


def _cosine(a: Counter, b: Counter) -> float:
    """İki kelime-frekans sayacının cosine benzerliği."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[w] * b[w] for w in common)
    na = sqrt(sum(v * v for v in a.values()))
    nb = sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _signal_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """İki sinyal panelinin ortak hücreleri üzerinde İŞARETLİ Pearson korelasyonu.

    İşaretli (mutlak değil): ters işaretli sinyal (corr≈-1) farklı bir bahistir
    (bir faktörün tersi = 'inversion'), duplicate SAYILMAZ. Doküman 14 kuralı da
    işaretli korelasyondur (signal_correlation > 0.95).
    """
    cols = a.columns.intersection(b.columns)
    idx = a.index.intersection(b.index)
    if len(cols) == 0 or len(idx) == 0:
        return 0.0
    av = a.loc[idx, cols].to_numpy().ravel()
    bv = b.loc[idx, cols].to_numpy().ravel()
    mask = ~(np.isnan(av) | np.isnan(bv))
    if mask.sum() < 30:
        return 0.0
    av, bv = av[mask], bv[mask]
    if av.std() == 0 or bv.std() == 0:
        return 0.0
    return float(np.corrcoef(av, bv)[0, 1])


class NoveltyIndex:
    """Kampanya boyunca görülen sinyallerin yapısal + davranışsal kaydı."""

    def __init__(self, ast_threshold: float = AST_THRESHOLD,
                 corr_threshold: float = CORR_THRESHOLD,
                 text_threshold: float = TEXT_THRESHOLD) -> None:
        self.ast_threshold = ast_threshold
        self.corr_threshold = corr_threshold
        self.text_threshold = text_threshold
        # (hid, ast_tokens, text_tokens, signal_df)
        self._entries: list[tuple[str, Counter, Counter, Optional[pd.DataFrame]]] = []

    def check_textual(self, hyp: HypothesisSpec) -> Optional[str]:
        """Backtest'ten önce (bedava). Açıklama near-verbatim aynıysa hid döner.

        Leksikal cosine — aynı fikri hemen hemen aynı kelimelerle tekrar etmeyi
        yakalar (yapı biraz değişse bile). Eşik yüksek: farklı stratejilerin
        ortak kelime (momentum/volume) paylaşması yanlış-pozitif YAPMAZ.
        """
        txt = _hyp_text(hyp)
        for hid, _, ptxt, _ in self._entries:
            if _cosine(txt, ptxt) > self.text_threshold:
                return hid
        return None

    def check_structural(self, hyp: HypothesisSpec) -> Optional[str]:
        """Backtest'ten önce (bedava). Duplicate ise eşleşen hypothesis_id döner."""
        tok = _tokens(hyp.signal)
        for hid, ptok, _, _ in self._entries:
            if _jaccard(tok, ptok) > self.ast_threshold:
                return hid
        return None

    def check_behavioral(self, signal: pd.DataFrame) -> Optional[str]:
        """Sinyal hesaplandıktan sonra: korelasyonla tekrar tespiti."""
        for hid, _, _, psig in self._entries:
            if psig is not None and _signal_corr(signal, psig) > self.corr_threshold:
                return hid
        return None

    def add(self, hyp: HypothesisSpec, signal: Optional[pd.DataFrame] = None) -> None:
        self._entries.append((hyp.hypothesis_id, _tokens(hyp.signal),
                              _hyp_text(hyp), signal))
