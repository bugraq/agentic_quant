"""
Sayısal parametre optimizasyonu testleri (Doküman 16.3/27).
Kötü pencereli bir stratejiyi yapıyı bozmadan iyileştirmeli.
"""
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from data import gen_cross_sectional_momentum
from optimization import n_window_slots, optimize_parameters


def _mom(window: int) -> HypothesisSpec:
    sig = Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=window, inputs=[Expression(op="field", field="close")])])
    return HypothesisSpec(
        hypothesis_id="hyp_opt", title=f"{window}g momentum", claim="t",
        family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="momentum", description="y"),
        universe=Universe(source="sp500_point_in_time"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def test_slot_count():
    assert n_window_slots(_mom(5)) == 1
    print("  [ok] pencere slotu sayımı doğru")


def test_optimizer_improves_bad_window():
    data = gen_cross_sectional_momentum(seed=1)
    bad = _mom(5)                       # kısa pencere: momentum verisinde zayıf
    from optimization.parameter_search import wf_score
    base, _ = wf_score(bad, data, cost_bps=1.0)
    horizons = [5, 10, 20, 60, 90, 120]
    best_hyp, best_score, trials = optimize_parameters(
        bad, data, allowed_horizons=horizons, cost_bps=1.0, n_samples=8)
    assert best_score >= base, f"optimizer kötüleştirdi: {base:.2f} -> {best_score:.2f}"
    # yapı korunmalı (hâlâ cross_sectional_rank(return(...)))
    assert best_hyp.signal.op == "cross_sectional_rank"
    # DÜRÜST SAYIM: her değerlendirilen kombinasyon deneme olarak dönmeli
    # (1 slot × 6 horizon = tam grid = 6 deneme), hepsinde sonuç + tekil id olmalı.
    assert len(trials) == len(horizons), f"deneme sayısı eksik: {len(trials)}"
    ids = [t[0].hypothesis_id for t in trials]
    assert len(set(ids)) == len(ids) and all(i.startswith("hyp_opt_p") for i in ids)
    assert all(t[1].net_returns for t in trials), "denemede getiri serisi yok"
    print(f"  [ok] optimizer min-fold Sharpe'ı iyileştirdi/korudu: {base:.2f} -> {best_score:.2f}, "
          f"seçilen pencere={best_hyp.signal.inputs[0].window}; "
          f"{len(trials)} deneme sayıma girdi")


def main():
    test_slot_count()
    test_optimizer_improves_bad_window()
    print("OK — parametre optimizasyonu testleri geçti.")


if __name__ == "__main__":
    main()
