"""
Koşul-canlılık + rapor tekilleştirme testleri (gerçek koşu bulgularından).

Bulgu 1: 'Momentum + Low Vol' diye kabul edilen strateji, koşulu %0.25
tetiklendiği için fiilen saf reversal'dı (rolling_std(close) fiyat ölçeğinde,
0.02 getiri-ölçeğinde — birim hatası). Ölü koşul artık yakalanmalı.

Bulgu 2: Optimizer'ın ölü-parametreli 6'lı özdeş denemeleri raporu boğuyor ve
n_trials'ı şişiriyordu. Özdeş getiri serileri tek strateji sayılmalı (xN).
"""
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from data.synthetic import gen_cross_sectional_momentum
from dsl import compile_hypothesis
from backtest import evaluate_signal
from evaluation.multiple_testing import build_report, dedup_records


def _hyp(signal: Expression) -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id="hyp_lv", title="t", claim="t",
        family=HypothesisFamily.regime_conditioned,
        economic_mechanism=EconomicMechanism(type="x", description="y"),
        universe=Universe(source="s"), features=[], signal=signal,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def _cond_signal(condition: Expression) -> Expression:
    ret5 = Expression(op="return", window=5,
                      inputs=[Expression(op="field", field="close")])
    return Expression(op="cross_sectional_rank", inputs=[
        Expression(op="conditional", inputs=[
            condition, ret5, Expression(op="negate", inputs=[ret5])])])


def test_dead_condition_detected():
    # Gerçek koşudaki birim hatasının kopyası: fiyat-std'si < 0.02 (asla olmaz)
    data = gen_cross_sectional_momentum(seed=1)
    dead_cond = Expression(op="less_than", inputs=[
        Expression(op="rolling_std", window=10,
                   inputs=[Expression(op="field", field="close")]),
        Expression(op="const", value=0.02)])
    g = compile_hypothesis(_hyp(_cond_signal(dead_cond)))
    liveness: list = []
    evaluate_signal(g, data, liveness_out=liveness)
    assert liveness, "conditional için canlılık ölçülmedi"
    _, frac = liveness[0]
    assert frac < 0.02, f"ölü koşul tespit edilemedi (tetiklenme %{frac*100:.1f})"
    print(f"  [ok] ölü koşul yakalandı (tetiklenme %{frac*100:.2f})")


def test_live_condition_not_flagged():
    # Canlı koşul: getiri > 0 — zamanın ~yarısında doğru
    data = gen_cross_sectional_momentum(seed=1)
    live_cond = Expression(op="greater_than", inputs=[
        Expression(op="return", window=1,
                   inputs=[Expression(op="field", field="close")]),
        Expression(op="const", value=0.0)])
    g = compile_hypothesis(_hyp(_cond_signal(live_cond)))
    liveness: list = []
    evaluate_signal(g, data, liveness_out=liveness)
    _, frac = liveness[0]
    assert 0.02 < frac < 0.98, f"canlı koşul yanlış işaretlendi (%{frac*100:.1f})"
    print(f"  [ok] canlı koşul serbest (tetiklenme %{frac*100:.1f})")


def test_report_dedups_identical_returns():
    rets_a = [0.001, -0.002, 0.003, 0.001, -0.001] * 20
    rets_b = [0.002, -0.001, 0.001, -0.002, 0.002] * 20
    records = [
        ("hyp_x", "x", "accept", 0.5, rets_a),
        ("hyp_x_p1", "x", "reject", 0.5, list(rets_a)),   # özdeş kopya (ölü param)
        ("hyp_x_p2", "x", "reject", 0.5, list(rets_a)),   # özdeş kopya
        ("hyp_y", "y", "reject", -0.2, rets_b),
    ]
    distinct, copies = dedup_records(records)
    assert len(distinct) == 2 and copies["hyp_x"] == 3 and copies["hyp_y"] == 1
    rows = build_report(records)
    assert len(rows) == 2, f"rapor tekilleştirmedi: {len(rows)} satır"
    by_id = {r.hypothesis_id: r for r in rows}
    assert by_id["hyp_x"].n_copies == 3
    print("  [ok] özdeş getiri serileri tek strateji sayıldı (x3 etiketiyle)")


def main():
    test_dead_condition_detected()
    test_live_condition_not_flagged()
    test_report_dedups_identical_returns()
    print("OK — ölü koşul + rapor tekilleştirme testleri geçti.")


if __name__ == "__main__":
    main()
