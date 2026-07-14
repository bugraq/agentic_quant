"""
Vectorized backtest motoru (günlük frekans).

ŞEMA = ÇALIŞTIRILAN ŞEY (Doküman 7): hipotezin beyan ettiği execution ve
portföy kuralları burada GERÇEKTEN uygulanır:

  - execution.trade_time: sinyal close_t'de; işlem beyan edilen anda yapılır
    (varsayılan open_t_plus_1). PnL, işlem anının fiyat serisiyle hesaplanır:
    open_t_plus_1 -> düzeltilmiş open-to-open getiri, weights.shift(2) hizası
    (w_t open_{t+1}'de kurulur, open_{t+1}->open_{t+2} getirisini kazanır).
  - execution.rebalance + holding_period_days: ağırlıklar yalnızca etkin
    aralıkta (max(rebalance_gün, holding_period)) güncellenir; arada sabit
    tutulur (drift ihmali — vectorized yaklaşım).
  - portfolio.type: cross_sectional_long_short (dollar-neutral) veya long_only.
  - portfolio.weighting: equal | rank_weight. gross_exposure ölçekler.

GETİRİLER DÜZELTİLMİŞ FİYATTAN: temettü + split dahil (adjusted_close; open
için düzeltme faktörü uygulanır). Sinyal ise LLM'in beyan ettiği ham alanları
kullanır (close vs.) — o ayrım DSL'in işidir.

Maliyet: turnover * cost_bps, işlem anına yazılır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from contracts.backtest_result import BacktestResult, CostBreakdown, FoldMetrics
from contracts.hypothesis_spec import (
    REBALANCE_DAYS,
    HypothesisSpec,
    Portfolio,
)
from contracts.strategy_graph import StrategyGraph
from data.synthetic import MarketData
from dsl.operators import parse_time_token
from backtest.evaluator import evaluate_signal

ENGINE_VERSION = "v0.3-declared-execution"
TRADING_DAYS = 252


def _build_weights(signal: pd.DataFrame, port: Portfolio) -> pd.DataFrame:
    """Portföy kurallarını uygula: tip (LS/long-only), ağırlıklama, gross."""
    ranks = signal.rank(axis=1, pct=True)
    long_q = port.long_quantile or 0.1
    short_q = port.short_quantile or 0.1

    if port.type == "long_only":
        if port.weighting == "rank_weight":
            raw = (ranks - (1.0 - long_q)).clip(lower=0)
        else:
            raw = (ranks >= 1.0 - long_q).astype(float)
        w = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        return w * port.gross_exposure

    # cross_sectional_long_short: her bacak 0.5 gross -> toplam gross 1, net 0
    if port.weighting == "rank_weight":
        lraw = (ranks - (1.0 - long_q)).clip(lower=0)
        sraw = (short_q - ranks).clip(lower=0)
    else:
        lraw = (ranks >= 1.0 - long_q).astype(float)
        sraw = (ranks <= short_q).astype(float)
    longs = lraw.div(lraw.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    shorts = sraw.div(sraw.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    return (longs.fillna(0) - shorts.fillna(0)) * port.gross_exposure


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(-dd.min()) if len(dd) else 0.0


def _universe_mask(hyp: HypothesisSpec, data: MarketData, signal):
    """Universe filtrelerini (min_price, min_dollar_volume) point-in-time uygula.

    Filtreyi geçmeyen varlıklar o tarihte işlemden çıkarılır (sinyal NaN).
    Doküman 4.4/2.4 — deklare edilen evren kısıtları gerçekten uygulanmalı.
    """
    u = hyp.universe
    mask = signal.notna()
    # ★ Point-in-time endeks üyeliği (survivorship düzeltmesi): hisse yalnızca
    # O TARİHTE endekstyken işlem görebilir. Bugünün listesini geçmişe uygulamak
    # yasak (Doküman 4/7); membership alanı varsa koşulsuz uygulanır.
    if "index_membership" in data.fields:
        mask &= (data.get("index_membership") > 0)
    if u.minimum_price is not None:
        mask &= (data.get("close") >= u.minimum_price)
    if u.minimum_median_dollar_volume is not None:
        med_dv = data.get("dollar_volume").rolling(20, min_periods=5).median()
        mask &= (med_dv >= u.minimum_median_dollar_volume)
    return signal.where(mask)


def _execution_prices(data: MarketData, phase: str) -> pd.DataFrame:
    """İşlem anının DÜZELTİLMİŞ fiyat serisi (temettü+split dahil getiri için).

    adjusted_close yoksa (eski sentetik) ham close'a düşer (faktör=1).
    """
    close = data.get("close")
    try:
        adj_close = data.get("adjusted_close")
    except KeyError:
        adj_close = close
    if phase == "close":
        return adj_close
    factor = adj_close / close
    return data.get("open") * factor


def _effective_interval(hyp: HypothesisSpec) -> int:
    """Ağırlık güncelleme aralığı (iş günü): rebalance ve holding period'un uzunu."""
    rebal = REBALANCE_DAYS.get(hyp.execution.rebalance, 1)
    hold = max(1, int(hyp.execution.holding_period_days or 1))
    return max(rebal, hold)


def _apply_rebalance(weights: pd.DataFrame, interval: int) -> pd.DataFrame:
    """Ağırlıkları yalnızca her `interval` barda bir güncelle; arada sabit tut."""
    if interval <= 1:
        return weights
    keep = np.zeros(len(weights), dtype=bool)
    keep[::interval] = True
    return weights.where(pd.Series(keep, index=weights.index), np.nan).ffill().fillna(0)


def _apply_sector_neutral(weights: pd.DataFrame, data: MarketData) -> pd.DataFrame:
    """portfolio.sector_neutral: her sektör grubunu net-sıfır yap, gross'u koru.

    Beyan edilen sector_neutral GERÇEKTEN uygulanır (Doküman 7 şema=çalıştırılan).
    Sektör haritası yoksa değişiklik yapılmaz (LS zaten piyasa-nötr; dürüst).
    """
    from backtest.evaluator import sector_groups
    if not data.sectors:
        return weights
    out = weights.copy()
    for _sec, cols in sector_groups(weights.columns, data.sectors).items():
        sub = weights[cols]
        out[cols] = sub.sub(sub.mean(axis=1), axis=0)   # sektör içi net-sıfır
    # Gross'u yeniden 1'e ölçekle (sektör-demean gross'u düşürebilir)
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    return out.div(gross, axis=0).fillna(0)


def compute_pnl(signal, hyp: HypothesisSpec, data: MarketData, cost_bps: float):
    """Sinyal -> (net_pnl, turnover_t) serileri. Motor çekirdeği; tekrar kullanılır."""
    signal = _universe_mask(hyp, data, signal)   # evren filtresi (point-in-time)
    weights = _build_weights(signal, hyp.portfolio)
    # sector_neutral yalnızca long-short'ta anlamlı (long-only'de demean sahte
    # short üretir); orada beyanı net-sıfır-per-sektör olarak uygula.
    if getattr(hyp.portfolio, "sector_neutral", False) \
            and hyp.portfolio.type == "cross_sectional_long_short":
        weights = _apply_sector_neutral(weights, data)
    weights = _apply_rebalance(weights, _effective_interval(hyp))

    # ★ Beyan edilen işlem anı UYGULANIR: trade_time -> bar gecikmesi + faz.
    #   w_t (close_t bilgisi) işlem barında kurulur, bir sonraki işlem barına
    #   kadar tutar: pnl hizası weights.shift(bar_offset + 1).
    trade_tick = parse_time_token(hyp.execution.trade_time)
    bar_offset = trade_tick // 2                       # open_t_plus_1 -> 1
    phase = "open" if trade_tick % 2 == 0 else "close"
    # fill_method=None: delist/eksik veri boşluğu getiriyi NaN bırakır (pad ile
    # sahte 0-getiri üretmez); NaN'lar pnl toplamında zaten atlanır.
    exec_ret = _execution_prices(data, phase).pct_change(fill_method=None)

    gross_pnl = (weights.shift(bar_offset + 1) * exec_ret).sum(axis=1)
    turnover_t = (weights - weights.shift(1)).abs().sum(axis=1)
    # Maliyet, o turnover'ı yaratan pozisyonun GETİRİ kazanmaya başladığı barla
    # AYNI hizada yazılır: gross weights.shift(bar_offset+1) kullandığı için
    # maliyet de turnover.shift(bar_offset+1) olmalı (aksi halde maliyet, ilgili
    # pozisyon getiriyi kazanmadan BİR BAR önce düşer — sayısal test doğruladı).
    cost_t = turnover_t.shift(bar_offset + 1) * (cost_bps / 1e4)
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
        cost_breakdown=CostBreakdown(spread=float((turnover_t * (cost_bps/1e4)).sum())),
        data_version=getattr(data, "version", "synthetic-v0"),
        engine_version=ENGINE_VERSION,
        seed=seed,
    )
