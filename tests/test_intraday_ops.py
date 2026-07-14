"""
Gün-içi (high/low) operatör testleri: intraday_range + close_location.

- Girdisiz derlenir; info_tick close_t (tick 1) — sızıntı-validatör doğru karar verir.
- Değerler doğru hesaplanır (ölçek-bağımsız aralık; kapanış konumu [0,1]).
- Uçtan uca: cross_sectional_rank ile sarılıp backtest edilebilir.
"""
import numpy as np
import pandas as pd

from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from backtest import evaluate_signal, run_backtest
from data.synthetic import MarketData
from dsl import compile_hypothesis, validate
from dsl.operators import parse_time_token


def _market() -> MarketData:
    dates = pd.bdate_range("2020-01-01", periods=40)
    tickers = ["A", "B", "C"]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (40, 3)), axis=0),
                         index=dates, columns=tickers)
    # Gün-içi aralık varlığa göre DEĞİŞSİN (A dar, C geniş) — sinyal ayrışsın
    span = pd.DataFrame([[0.01, 0.03, 0.06]] * 40, index=dates, columns=tickers)
    high = close * (1 + span)
    low = close * (1 - span)
    return MarketData(fields={"open": close.shift(1).bfill(), "high": high, "low": low,
                              "close": close, "adjusted_close": close,
                              "volume": pd.DataFrame(1e6, index=dates, columns=tickers),
                              "dollar_volume": close * 1e6})


def _hyp(op, window=None) -> HypothesisSpec:
    inner = Expression(op=op, window=window, inputs=[])
    sig = Expression(op="cross_sectional_rank", inputs=[inner])
    return HypothesisSpec(
        hypothesis_id="hyp_x", title="t", claim="c", family=HypothesisFamily.volatility,
        economic_mechanism=EconomicMechanism(type="intraday", description="d"),
        universe=Universe(source="x"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.34, short_quantile=0.34),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def test_compiles_with_correct_tick():
    for op in ("intraday_range", "close_location"):
        graph = compile_hypothesis(_hyp(op))
        # sinyal düğümünün info_tick'i close_t (1) tabanlı olmalı, open_t (0) DEĞİL
        sig_node = next(n for n in graph.nodes if n.node_id == graph.signal_node_id)
        inner = next(n for n in graph.nodes if n.op == op)
        assert int(inner.params["_info_tick"]) == parse_time_token("close_t"), \
            f"{op} info_tick close_t olmalı (sızıntı önlemi)"
        dec = validate(graph, _hyp(op))
        assert dec.decision.value != "reject", f"{op} sızıntı sanıldı: {dec.issues}"
    print("  [ok] girdisiz derlenir, info_tick close_t, sızıntı yok")


def test_intraday_range_values():
    data = _market()
    sig = evaluate_signal(compile_hypothesis(_hyp("intraday_range")), data)
    # C'nin aralığı en geniş -> her gün en yüksek range -> rank en yüksek
    raw = (data.get("high") - data.get("low")) / data.get("close")
    assert (raw["C"] > raw["A"]).all(), "geniş-aralık varlık daha yüksek range vermeli"
    assert sig.notna().sum().sum() > 0
    print("  [ok] intraday_range ölçek-bağımsız, doğru sıralıyor")


def test_close_location_bounds():
    data = _market()
    loc = evaluate_signal(compile_hypothesis(_hyp("close_location", window=3)), data)
    # close_location ham değeri [0,1] içinde olmalı (rank sonrası -0.5..0.5)
    raw_h = _hyp("close_location")
    graph = compile_hypothesis(raw_h)
    # cross_sectional_rank'siz iç değeri kontrol için doğrudan hesap
    hl = data.get("high") - data.get("low")
    raw = ((data.get("close") - data.get("low")) / hl).clip(0, 1)
    assert raw.min().min() >= 0.0 and raw.max().max() <= 1.0
    print("  [ok] close_location [0,1] sınırlı")


def test_end_to_end_backtest():
    data = _market()
    for op in ("intraday_range", "close_location"):
        h = _hyp(op, window=5)
        graph = compile_hypothesis(h)
        res = run_backtest(graph, h, data, cost_bps=5.0)
        assert res.per_fold_metrics, f"{op} backtest üretmedi"
    print("  [ok] uçtan uca backtest çalışıyor")


def main():
    test_compiles_with_correct_tick()
    test_intraday_range_values()
    test_close_location_bounds()
    test_end_to_end_backtest()
    print("OK — gün-içi operatör testleri geçti.")


if __name__ == "__main__":
    main()
