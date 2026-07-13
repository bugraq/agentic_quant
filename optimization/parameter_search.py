"""
Sayısal parametre optimizasyonu (Doküman 16.3 ve 27).

TEMEL AYRIM: LLM YAPISAL değişiklikten sorumludur; bu motor ise SAYISAL
parametrelerden (pencere/lookback uzunlukları). Bir hipotezin YAPISINI sabit
tutar, yalnızca window parametrelerini kampanyanın izin verdiği ufuklar
üzerinde arar.

Aşırı-uydurmayı (overfitting) önlemek için skor MUHAFAZAKÂRDIR: en iyi ortalama
Sharpe değil, en KÖTÜ walk-forward fold'unun Sharpe'ı maksimize edilir
(min-fold — Doküman 11.2 muhafazakâr skoru).

DÜRÜST SAYIM (Doküman 10/12): burada yapılan HER backtest bir denemedir ve
multiple-testing muhasebesine girmek zorundadır. Bu yüzden değerlendirilen
bütün adaylar (hipotez + sonuç) çağırana geri verilir; orchestrator hepsini
hafızaya kaydeder. Kaydedilmeyen deneme = gizli arama = geçersiz istatistik.
"""
from __future__ import annotations

import itertools
import random
from typing import Optional

from contracts.backtest_result import BacktestResult
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


def _min_fold_sharpe(res: BacktestResult) -> Optional[float]:
    """Muhafazakâr skor: en KÖTÜ fold'un Sharpe'ı (Doküman 11.2)."""
    if not res.per_fold_metrics:
        return None
    return min(m.sharpe for m in res.per_fold_metrics)


def wf_score(hyp: HypothesisSpec, data: MarketData, cost_bps: float,
             graph=None) -> tuple[Optional[float], BacktestResult]:
    """Optimizasyon hedefi: walk-forward MIN-FOLD Sharpe (muhafazakâr).

    Sonucu da döndürür ki her deneme hafızaya kaydedilebilsin (dürüst sayım).
    """
    g = graph or compile_hypothesis(hyp)
    res = run_walk_forward(g, hyp, data, n_folds=5, cost_bps=cost_bps)
    return _min_fold_sharpe(res), res


def n_window_slots(hyp: HypothesisSpec) -> int:
    return (sum(_count_windows(f.expression) for f in hyp.features)
            + _count_windows(hyp.signal))


def optimize_parameters(hyp: HypothesisSpec, data: MarketData,
                        allowed_horizons: list[int], cost_bps: float = 5.0,
                        n_samples: int = 8, seed: int = 0):
    """
    Yapı sabit; pencereleri allowed_horizons üzerinde ara. Az slot varsa tam
    grid, çoksa rastgele arama. Mevcut parametreler baz alınır; iyileşme yoksa
    orijinal döner.

    Döndürür: (en_iyi_hipotez, min_fold_sharpe, trials)
      trials: değerlendirilen HER aday için (aday_hipotez, BacktestResult) —
      orchestrator bunları hafızaya yazar (multiple-testing sayımı).
    """
    trials: list[tuple[HypothesisSpec, BacktestResult]] = []
    n = n_window_slots(hyp)
    if n == 0 or not allowed_horizons:
        score, _ = wf_score(hyp, data, cost_bps)
        return hyp, score, trials

    best_hyp = hyp
    best_score, _ = wf_score(hyp, data, cost_bps)

    # Az slot varsa TAM GRID (kesin en iyi), çoksa rastgele arama (patlamayı önle).
    if len(allowed_horizons) ** n <= 48:
        combos = list(itertools.product(allowed_horizons, repeat=n))
    else:
        rng = random.Random(seed)
        combos = [tuple(rng.choice(allowed_horizons) for _ in range(n))
                  for _ in range(n_samples)]

    for k, vals in enumerate(combos):
        cand = _apply_windows(hyp, list(vals))
        try:
            g = compile_hypothesis(cand)
            if validate(g, cand).decision != DecisionType.accept:
                continue                       # sızıntı/geçersiz kombinasyonu atla
            score, res = wf_score(cand, data, cost_bps, graph=g)
        except Exception:  # noqa: BLE001
            continue
        # Her deneme sayılır: tekil kimlikle kaydedilmek üzere dışarı ver.
        cand_id = f"{hyp.hypothesis_id}_p{k+1}"
        trials.append((cand.model_copy(update={"hypothesis_id": cand_id}), res))
        if score is not None and (best_score is None or score > best_score):
            best_hyp, best_score = cand, score
    return best_hyp, best_score, trials
