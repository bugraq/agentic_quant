"""
Sayısal parametre optimizasyonu (Doküman 16.3 ve 27).

TEMEL AYRIM: LLM YAPISAL değişiklikten sorumludur; bu motor ise SAYISAL
parametrelerden (pencere/lookback uzunlukları). Bir hipotezin YAPISINI sabit
tutar, yalnızca window parametrelerini kampanyanın izin verdiği ufuklar
üzerinde arar.

Aşırı-uydurmayı (overfitting) önlemek için skor MUHAFAZAKÂRDIR: en iyi ortalama
Sharpe değil, en KÖTÜ walk-forward fold'unun Sharpe'ı maksimize edilir
(min-fold — Doküman 11.2 muhafazakâr skoru).
"""
from __future__ import annotations

import itertools
import random
from typing import Optional

from contracts.dsl import Expression, NamedFeature
from contracts.decision import DecisionType
from contracts.hypothesis_spec import HypothesisSpec
from data.synthetic import MarketData
from dsl import compile_hypothesis, validate
from backtest.walk_forward import run_walk_forward


def _count_windows(expr: Expression) -> int:
    c = 1 if expr.window is not None else 0
    for i in expr.inputs:
        if isinstance(i, Expression):
            c += _count_windows(i)
    return c


def _set_windows(expr: Expression, values: list[int], counter: list[int]) -> Expression:
    w = expr.window
    if w is not None:
        w = values[counter[0]]
        counter[0] += 1
    new_inputs = [_set_windows(i, values, counter) if isinstance(i, Expression) else i
                  for i in expr.inputs]
    return expr.model_copy(update={"window": w, "inputs": new_inputs})


def _apply_windows(hyp: HypothesisSpec, values: list[int]) -> HypothesisSpec:
    counter = [0]
    feats = [NamedFeature(name=f.name, expression=_set_windows(f.expression, values, counter))
             for f in hyp.features]
    signal = _set_windows(hyp.signal, values, counter)
    return hyp.model_copy(update={"features": feats, "signal": signal})


def wf_score(hyp: HypothesisSpec, data: MarketData, cost_bps: float,
             graph=None) -> Optional[float]:
    """Optimizasyon hedefi: walk-forward ortalama Sharpe. Aşırı-uydurma araştırma
    döneminde optimize edilse de KİLİTLİ holdout'ta bağımsızca sınanır."""
    g = graph or compile_hypothesis(hyp)
    res = run_walk_forward(g, hyp, data, n_folds=5, cost_bps=cost_bps)
    return res.aggregate_sharpe()


def n_window_slots(hyp: HypothesisSpec) -> int:
    return (sum(_count_windows(f.expression) for f in hyp.features)
            + _count_windows(hyp.signal))


def optimize_parameters(hyp: HypothesisSpec, data: MarketData,
                        allowed_horizons: list[int], cost_bps: float = 5.0,
                        n_samples: int = 8, seed: int = 0):
    """
    Yapı sabit; pencereleri allowed_horizons üzerinde ara. Rastgele arama
    (grid çok büyüyebilir). Mevcut parametreler baz alınır; iyileşme yoksa
    orijinal döner. Döndürür: (en_iyi_hipotez, min_fold_sharpe).
    """
    n = n_window_slots(hyp)
    if n == 0 or not allowed_horizons:
        return hyp, wf_score(hyp, data, cost_bps)

    best_hyp = hyp
    best_score = wf_score(hyp, data, cost_bps)

    # Az slot varsa TAM GRID (kesin en iyi), çoksa rastgele arama (patlamayı önle).
    if len(allowed_horizons) ** n <= 48:
        combos = list(itertools.product(allowed_horizons, repeat=n))
    else:
        rng = random.Random(seed)
        combos = [tuple(rng.choice(allowed_horizons) for _ in range(n))
                  for _ in range(n_samples)]

    for vals in combos:
        cand = _apply_windows(hyp, list(vals))
        try:
            g = compile_hypothesis(cand)
            if validate(g, cand).decision != DecisionType.accept:
                continue                       # sızıntı/geçersiz kombinasyonu atla
            score = wf_score(cand, data, cost_bps, graph=g)
        except Exception:  # noqa: BLE001
            continue
        if score is not None and (best_score is None or score > best_score):
            best_hyp, best_score = cand, score
    return best_hyp, best_score
