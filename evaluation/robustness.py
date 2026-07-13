"""
Sağlamlık testleri (Doküman 9.2) — bir strateji tesadüf mü, dayanıklı mı?

  - Permutation testi: sinyali kesitsel karıştır (tahmin ilişkisini boz).
    Gerçek alpha permütasyonda YOK OLMALI. Olmuyorsa spurious.
  - Maliyet 2x: işlem maliyeti iki katına çıkınca strateji hayatta kalmalı.
  - Parametre perturbasyonu: pencereleri ±%20 kaydır; strateji bıçak sırtı
    olmamalı (küçük değişimde çökmemeli).

Bir aday ancak bu testlerden geçerse gerçekten 'promote' edilmeye değer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.dsl import Expression, NamedFeature
from contracts.hypothesis_spec import HypothesisSpec
from data.synthetic import MarketData
from backtest.engine import compute_pnl, fold_metrics, run_backtest
from backtest.evaluator import evaluate_signal
from dsl import compile_hypothesis

TRADING_DAYS = 252


@dataclass
class RobustnessResult:
    permutation_pvalue: float      # gerçek Sharpe'ı geçen permütasyon oranı
    cost2x_sharpe: float
    param_min_sharpe: float        # perturbasyonlar arası en kötü Sharpe
    robust: bool


def _sharpe(net_pnl) -> float:
    s = net_pnl.std()
    return float(net_pnl.mean() / s * np.sqrt(TRADING_DAYS)) if s > 0 else 0.0


def _permutation_pvalue(signal, hyp, data, cost_bps, n_perm=50, seed=0) -> tuple[float, float]:
    real_pnl, _ = compute_pnl(signal, hyp, data, cost_bps)
    real_sharpe = _sharpe(real_pnl)
    rng = np.random.default_rng(seed)
    vals = signal.to_numpy()
    ge = 0
    for _ in range(n_perm):
        perm = vals.copy()
        for r in range(perm.shape[0]):        # her satırda kesitsel karıştır
            rng.shuffle(perm[r])
        psig = signal.copy()
        psig[:] = perm
        ppnl, _ = compute_pnl(psig, hyp, data, cost_bps)
        if _sharpe(ppnl) >= real_sharpe:
            ge += 1
    return ge / n_perm, real_sharpe


def _scale_windows(expr: Expression, factor: float) -> Expression:
    new_inputs = [_scale_windows(i, factor) if isinstance(i, Expression) else i
                  for i in expr.inputs]
    w = expr.window
    if w is not None:
        w = max(1, int(round(w * factor)))
    return expr.model_copy(update={"window": w, "inputs": new_inputs})


def _perturb_windows(hyp: HypothesisSpec, factor: float) -> HypothesisSpec:
    feats = [NamedFeature(name=f.name, expression=_scale_windows(f.expression, factor))
             for f in hyp.features]
    return hyp.model_copy(update={"signal": _scale_windows(hyp.signal, factor),
                                  "features": feats})


def run_robustness(graph, hyp: HypothesisSpec, data: MarketData,
                   cost_bps: float = 5.0, signal=None) -> RobustnessResult:
    if signal is None:
        signal = evaluate_signal(graph, data)

    perm_p, real_sharpe = _permutation_pvalue(signal, hyp, data, cost_bps)

    # Maliyet 2x
    pnl2x, _ = compute_pnl(signal, hyp, data, cost_bps * 2)
    cost2x_sharpe = _sharpe(pnl2x)

    # Parametre perturbasyonu (±%20)
    param_sharpes = []
    for factor in (0.8, 1.25):
        try:
            g2 = compile_hypothesis(_perturb_windows(hyp, factor))
            res = run_backtest(g2, hyp, data, cost_bps=cost_bps)
            param_sharpes.append(res.aggregate_sharpe() or 0.0)
        except Exception:
            param_sharpes.append(0.0)
    param_min = min(param_sharpes) if param_sharpes else 0.0

    robust = (perm_p < 0.10) and (cost2x_sharpe > 0) and (param_min > 0)
    return RobustnessResult(permutation_pvalue=perm_p, cost2x_sharpe=cost2x_sharpe,
                            param_min_sharpe=param_min, robust=robust)
