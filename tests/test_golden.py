"""
Golden backtest testleri (Doküman 23.2).

Önceden doğrulanmış basit stratejilerin SABİT sentetik veri üzerindeki Sharpe
değerleri saklanır. Motor kodu değiştiğinde bu değerler değişmemeli — değişirse
motorda istenmeyen bir kayma vardır. Regresyon güvencesi.

Referans veri: gen_cross_sectional_momentum(seed=1), cost_bps=1.0.
"""
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from dsl import compile_hypothesis
from data import gen_cross_sectional_momentum
from backtest import run_backtest

# Saklanan referans Sharpe değerleri (bu veri + motor için doğrulanmış).
# v0.3-declared-execution: beyan edilen trade_time (open_t+1) + düzeltilmiş
# fiyattan getiri + rebalance/holding uygulaması ile yeniden saptandı.
GOLDEN = {
    "mom60": 0.5626,
    "mom20": -0.0807,
    "rev5": -1.0304,
}
TOL = 0.02   # motor determinist; küçük tolerans


def _H(sig, fam):
    return HypothesisSpec(
        hypothesis_id="g", title="g", claim="g", family=fam,
        economic_mechanism=EconomicMechanism(type="x", description="y"),
        universe=Universe(source="sp500_point_in_time"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def _mom(w):
    return Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=w, inputs=[Expression(op="field", field="close")])])


def _rev(w):
    return Expression(op="cross_sectional_rank", inputs=[
        Expression(op="negate", inputs=[
            Expression(op="return", window=w, inputs=[Expression(op="field", field="close")])])])


def test_golden_values():
    d = gen_cross_sectional_momentum(seed=1)
    cases = {"mom60": (_mom(60), HypothesisFamily.momentum),
             "mom20": (_mom(20), HypothesisFamily.momentum),
             "rev5": (_rev(5), HypothesisFamily.reversal)}
    for name, (sig, fam) in cases.items():
        h = _H(sig, fam)
        got = run_backtest(compile_hypothesis(h), h, d, cost_bps=1.0).aggregate_sharpe()
        assert abs(got - GOLDEN[name]) < TOL, \
            f"{name}: golden {GOLDEN[name]:.4f} ile eşleşmedi, got {got:.4f} (motor kaymış?)"
        print(f"  [ok] {name}: Sharpe {got:.4f} ~ golden {GOLDEN[name]:.4f}")


def main():
    test_golden_values()
    print("OK — golden backtest testleri geçti (motor sabit).")


if __name__ == "__main__":
    main()
