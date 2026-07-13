"""
Backtest motoru testleri.

İki grup (Doküman 23.1 + 23.3):
  A) Bilinen sinyal testleri: motor, veriye gömülü gerçek momentum/reversal
     sinyalini bulmalı; RASTGELE veride sahte alpha üretmemeli.
  B) Property testleri: gelecek fiyatı değiştirmek geçmiş sinyali
     değiştirmemeli; maliyet artışı net getiriyi artırmamalı.
"""
import numpy as np

from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from dsl import compile_hypothesis
from data import (
    gen_cross_sectional_momentum, gen_random, gen_short_term_reversal,
)
from backtest import evaluate_signal, run_backtest


def _hyp(signal: Expression, fam=HypothesisFamily.momentum) -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id="hyp_bt", title="t", claim="t", family=fam,
        economic_mechanism=EconomicMechanism(type="x", description="y"),
        universe=Universe(source="sp500_point_in_time"),
        features=[], signal=signal,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def _momentum_signal() -> Expression:
    # geçmiş 60g getiriye göre kesitsel sırala (kazananı long)
    return Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=60, inputs=[Expression(op="field", field="close")])])


def _reversal_signal() -> Expression:
    # dünün getirisini tersine çevir (kaybedeni long)
    return Expression(op="cross_sectional_rank", inputs=[
        Expression(op="negate", inputs=[
            Expression(op="return", window=1, inputs=[Expression(op="field", field="close")])])])


def test_finds_momentum():
    h = _hyp(_momentum_signal())
    g = compile_hypothesis(h)
    res = run_backtest(g, h, gen_cross_sectional_momentum(seed=1), cost_bps=1.0)
    sharpe = res.aggregate_sharpe()
    assert sharpe > 0.5, f"momentum bulunamadı, Sharpe={sharpe:.2f}"
    print(f"  [ok] momentum verisinde momentum sinyali: Sharpe={sharpe:.2f}")


def test_finds_reversal():
    h = _hyp(_reversal_signal(), fam=HypothesisFamily.reversal)
    g = compile_hypothesis(h)
    res = run_backtest(g, h, gen_short_term_reversal(seed=1), cost_bps=1.0)
    sharpe = res.aggregate_sharpe()
    assert sharpe > 0.5, f"reversal bulunamadı, Sharpe={sharpe:.2f}"
    print(f"  [ok] reversal verisinde reversal sinyali: Sharpe={sharpe:.2f}")


def test_no_fake_alpha_on_random():
    # Rastgele veride momentum stratejisi sistematik alpha ÜRETMEMELİ.
    h = _hyp(_momentum_signal())
    g = compile_hypothesis(h)
    sharpes = [run_backtest(g, h, gen_random(seed=s), cost_bps=1.0).aggregate_sharpe()
               for s in range(8)]
    mean_sharpe = float(np.mean(sharpes))
    assert abs(mean_sharpe) < 0.5, f"rastgele veride sahte alpha! ort Sharpe={mean_sharpe:.2f}"
    print(f"  [ok] rastgele veride sahte alpha yok: ort Sharpe={mean_sharpe:.2f}")


def test_future_prices_do_not_change_past_signal():
    # Property (Doküman 23.3): geleceği değiştir -> geçmiş sinyal aynı kalmalı.
    data = gen_cross_sectional_momentum(seed=2)
    g = compile_hypothesis(_hyp(_momentum_signal()))
    sig1 = evaluate_signal(g, data)

    perturbed = gen_cross_sectional_momentum(seed=2)
    perturbed.fields["close"].iloc[-5:] *= 1.5   # son 5 günü boz
    sig2 = evaluate_signal(g, perturbed)

    past1 = sig1.iloc[:-5].dropna(how="all")
    past2 = sig2.iloc[:-5].dropna(how="all")
    assert np.allclose(past1.values, past2.values, equal_nan=True), \
        "gelecek fiyat değişimi geçmiş sinyali etkiledi — SIZINTI!"
    print("  [ok] gelecek fiyatı değiştirmek geçmiş sinyali değiştirmedi")


def test_higher_cost_lowers_return():
    # Property: maliyet artışı net getiriyi ARTIRMAMALI.
    h = _hyp(_momentum_signal())
    g = compile_hypothesis(h)
    data = gen_cross_sectional_momentum(seed=3)
    r_lo = run_backtest(g, h, data, cost_bps=0.0).per_fold_metrics[0].annualized_return
    r_hi = run_backtest(g, h, data, cost_bps=50.0).per_fold_metrics[0].annualized_return
    assert r_hi <= r_lo, f"maliyet arttı ama getiri arttı ({r_lo:.4f} -> {r_hi:.4f})"
    print(f"  [ok] maliyet artışı getiriyi düşürdü: {r_lo:.4f} -> {r_hi:.4f}")


def main():
    test_finds_momentum()
    test_finds_reversal()
    test_no_fake_alpha_on_random()
    test_future_prices_do_not_change_past_signal()
    test_higher_cost_lowers_return()
    print("OK — backtest motoru bilinen sinyalleri buluyor, sızıntı üretmiyor.")


if __name__ == "__main__":
    main()
