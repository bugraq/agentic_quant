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


def structure_signature(expr: Expression) -> str:
    """Sinyal ağacının PENCEREDEN BAĞIMSIZ yapısal imzası.

    Yalnızca pencere uzunluğunda farklılaşan iki strateji (60g vs 90g momentum)
    AYNI yapıdır. 'Farklı strateji yapısı sayısı' metriği bununla ölçülür
    (MVP kriter 3). Operatör adları, alanlar ve ağaç şekli korunur; window değil.
    """
    if expr.op == "field":
        return f"field:{expr.field}"
    if expr.op == "const":
        return "const"
    if expr.op == "feature_ref":
        return f"ref:{expr.name}"
    inner = ",".join(
        structure_signature(c if isinstance(c, Expression)
                            else Expression(op="feature_ref", name=c))
        for c in expr.inputs)
    return f"{expr.op}({inner})"


def hypothesis_structure(hyp: HypothesisSpec) -> str:
    """Hipotezin tam yapısal imzası: sinyal + (varsa) feature ifadeleri."""
    feats = ";".join(sorted(structure_signature(f.expression) for f in hyp.features))
    return structure_signature(hyp.signal) + ("|" + feats if feats else "")


def count_distinct_structures(hyps: "list[HypothesisSpec]") -> int:
    """Bir hipotez listesindeki FARKLI yapı sayısı (pencereden bağımsız)."""
    return len({hypothesis_structure(h) for h in hyps})


# Her stratejide olan/ayırt etmeyen operatörler (aşırı-kullanım sinyali değil)
_UBIQUITOUS_OPS = {"cross_sectional_rank", "field", "const", "feature_ref"}


def _ops_of(expr: Expression, out: set) -> None:
    if expr.op not in ("field", "const", "feature_ref"):
        out.add(expr.op)
    for c in expr.inputs:
        if isinstance(c, Expression):
            _ops_of(c, out)


def dominant_operators(hyps: "list[HypothesisSpec]", min_frac: float = 0.5,
                       k: int = 3) -> list[str]:
    """Hipotezlerin > min_frac oranında AŞIRI kullandığı operatörler.

    Etiket-tabanlı çeşitlilik (family) LLM'ce kandırılabiliyor ('reversal'ı
    'regime_conditioned' etiketlemek). Yapı-tabanlı bu sinyal gerçek rutu
    yakalar: ör. negate/volatility/conditional hipotezlerin çoğunda ise LLM
    volatilite-koşullu reversal'a sıkışmış demektir.
    """
    if not hyps:
        return []
    counts: Counter = Counter()
    for h in hyps:
        ops: set = set()
        _ops_of(h.signal, ops)
        for f in h.features:
            _ops_of(f.expression, ops)
        for op in ops - _UBIQUITOUS_OPS:
            counts[op] += 1
    n = len(hyps)
    dominant = [op for op, c in counts.most_common() if c / n > min_frac]
    return dominant[:k]


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
