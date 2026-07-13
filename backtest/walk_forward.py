"""
Walk-forward değerlendirme (Doküman 9.1).

Stratejilerimizin eğitilen parametresi yoktur (yapı LLM+DSL ile sabit), bu
yüzden walk-forward burada: AYNI stratejiyi ardışık zaman dilimlerinde ayrı
ayrı değerlendirip TUTARLILIĞA bakmaktır. Bir strateji sadece bir dönemde
değil, çoğu fold'da pozitif olmalı (rejimler arası kararlılık).

Falsification.minimum_positive_walk_forward_folds bu orana karşı kontrol edilir.
"""
from __future__ import annotations

import numpy as np

from contracts.backtest_result import BacktestResult, CostBreakdown
from contracts.hypothesis_spec import HypothesisSpec
from contracts.strategy_graph import StrategyGraph
from data.synthetic import MarketData
from backtest.engine import compute_pnl, fold_metrics
from backtest.evaluator import evaluate_signal


def run_walk_forward(graph: StrategyGraph, hyp: HypothesisSpec, data: MarketData,
                     n_folds: int = 5, cost_bps: float = 5.0, seed: int = 42,
                     signal=None) -> BacktestResult:
    if signal is None:
        signal = evaluate_signal(graph, data)
    net_pnl, turnover_t = compute_pnl(signal, hyp, data, cost_bps)

    idx = net_pnl.index
    chunks = np.array_split(np.arange(len(idx)), n_folds)
    folds = []
    for k, ch in enumerate(chunks):
        if len(ch) < 20:   # çok kısa fold'u atla
            continue
        seg = net_pnl.iloc[ch[0]:ch[-1] + 1]
        folds.append(fold_metrics(seg, turnover_t, f"fold_{k}", "validation"))

    positive_frac = (sum(1 for f in folds if f.sharpe > 0) / len(folds)) if folds else 0.0

    return BacktestResult(
        hypothesis_id=hyp.hypothesis_id,
        per_fold_metrics=folds,
        net_returns=[float(x) for x in net_pnl.to_numpy()],
        cost_breakdown=CostBreakdown(spread=float((turnover_t.shift(1) * (cost_bps/1e4)).sum())),
        exposures={"positive_fold_fraction": positive_frac, "n_folds": len(folds)},
        data_version=getattr(data, "version", "synthetic-v0"),
        engine_version="v0.2-walk-forward",
        seed=seed,
    )
