"""
Vectorized backtest motoru (MVP — günlük frekans).

Zamanlama (sızıntısızlık burada hayata geçer):
  - Sinyal close_t'de hesaplanır  -> weights_t
  - Pozisyon bir bar SONRA getiri kazanır: pnl_{t+1} = weights_t . ret_{t+1}
Yani ağırlık yalnızca <= close_t bilgisini kullanır, getiri kesinlikle
gelecektedir. (weights.shift(1) ile hizalama tam olarak bunu sağlar.)

Portföy: kesitsel long-short, dollar-neutral, eşit ağırlık, gross=1.
Maliyet: basit turnover * cost_bps modeli.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from contracts.backtest_result import BacktestResult, CostBreakdown, FoldMetrics
from contracts.hypothesis_spec import HypothesisSpec
from contracts.strategy_graph import StrategyGraph
from data.synthetic import MarketData
from backtest.evaluator import evaluate_signal

ENGINE_VERSION = "v0.1-vectorized-daily"
TRADING_DAYS = 252


def _build_weights(signal: pd.DataFrame, long_q: float, short_q: float) -> pd.DataFrame:
    """Kesitsel long-short ağırlıklar: üst quantile long, alt quantile short, gross=1."""
    ranks = signal.rank(axis=1, pct=True)
    longs = (ranks >= 1.0 - long_q).astype(float)
    shorts = (ranks <= short_q).astype(float)
    # Her bacağı kendi içinde normalize et; long +0.5, short -0.5 -> gross 1, net 0
    longs = longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    shorts = shorts.div(shorts.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    return (longs.fillna(0) - shorts.fillna(0))


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(-dd.min()) if len(dd) else 0.0


def compute_pnl(signal, hyp: HypothesisSpec, data: MarketData, cost_bps: float):
    """Sinyal -> (net_pnl, turnover_t) serileri. Motor çekirdeği; tekrar kullanılır."""
    long_q = hyp.portfolio.long_quantile or 0.1
    short_q = hyp.portfolio.short_quantile or 0.1
    weights = _build_weights(signal, long_q, short_q)

    asset_ret = data.get("close").pct_change()
    # ★ execution gecikmesi: weights_t, ret_{t+1} kazanır (shift(1) ile hizalama)
    gross_pnl = (weights.shift(1) * asset_ret).sum(axis=1)
    turnover_t = (weights - weights.shift(1)).abs().sum(axis=1)
    cost_t = turnover_t.shift(1) * (cost_bps / 1e4)
    net_pnl = (gross_pnl - cost_t).dropna()
    return net_pnl, turnover_t


def fold_metrics(net_pnl, turnover_t, fold_id: str, split: str) -> FoldMetrics:
    """Bir getiri diliminden metrikler."""
    mean, std = net_pnl.mean(), net_pnl.std()
    sharpe = float(mean / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    equity = (1.0 + net_pnl).cumprod()
    turn = turnover_t.reindex(net_pnl.index)
    return FoldMetrics(
        fold_id=fold_id, split=split, sharpe=sharpe,
        annualized_return=float(mean * TRADING_DAYS),
        volatility=float(std * np.sqrt(TRADING_DAYS)),
        max_drawdown=_max_drawdown(equity),
        turnover=float(turn.mean() * TRADING_DAYS))


def run_backtest(graph: StrategyGraph, hyp: HypothesisSpec, data: MarketData,
                 cost_bps: float = 5.0, seed: int = 42,
                 split_name: str = "research", signal=None) -> BacktestResult:
    """StrategyGraph + veri -> BacktestResult (tek fold, iskelet).

    signal önceden hesaplanmışsa (novelty kontrolünde) yeniden hesaplamaz.
    """
    if signal is None:
        signal = evaluate_signal(graph, data)
    net_pnl, turnover_t = compute_pnl(signal, hyp, data, cost_bps)
    fold = fold_metrics(net_pnl, turnover_t, "fold_0", split_name)
    return BacktestResult(
        hypothesis_id=hyp.hypothesis_id,
        per_fold_metrics=[fold],
        net_returns=[float(x) for x in net_pnl.to_numpy()],
        cost_breakdown=CostBreakdown(spread=float((turnover_t.shift(1) * (cost_bps/1e4)).sum())),
        data_version=getattr(data, "version", "synthetic-v0"),
        engine_version=ENGINE_VERSION,
        seed=seed,
    )
