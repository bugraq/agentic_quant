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


def test_rebalance_lowers_turnover():
    # Şema = çalıştırılan şey: weekly rebalance / uzun holding period turnover'ı
    # gerçekten DÜŞÜRMELİ (daha önce bu alanlar motor tarafından yok sayılıyordu).
    data = gen_cross_sectional_momentum(seed=4)
    h_daily = _hyp(_momentum_signal())
    h_weekly = h_daily.model_copy(update={"execution": Execution(
        signal_time="close_t", trade_time="open_t_plus_1",
        holding_period_days=5, rebalance="weekly")})
    g = compile_hypothesis(h_daily)
    t_daily = run_backtest(g, h_daily, data).per_fold_metrics[0].turnover
    t_weekly = run_backtest(g, h_weekly, data).per_fold_metrics[0].turnover
    assert t_weekly < t_daily * 0.5, \
        f"weekly rebalance turnover'ı düşürmedi ({t_daily:.1f} -> {t_weekly:.1f})"
    print(f"  [ok] weekly rebalance turnover'ı düşürdü: {t_daily:.1f} -> {t_weekly:.1f}")


def test_long_only_weights_nonnegative():
    # long_only tipinde kısa pozisyon OLMAMALI ve gross ~1 olmalı.
    from backtest.engine import _build_weights
    data = gen_cross_sectional_momentum(seed=5)
    g = compile_hypothesis(_hyp(_momentum_signal()))
    sig = evaluate_signal(g, data)
    port = Portfolio(type="long_only", long_quantile=0.3)
    w = _build_weights(sig, port).dropna(how="all")
    assert (w.fillna(0) >= 0).all().all(), "long_only'de negatif ağırlık var!"
    gross = w.abs().sum(axis=1).iloc[100:]
    assert np.allclose(gross, 1.0, atol=1e-6), "long_only gross 1 değil"
    print("  [ok] long_only: ağırlıklar >= 0, gross = 1")


def test_unsupported_portfolio_rejected():
    # Motorun uygulayamadığı beyan (ör. beta_neutral) static validator'da RED.
    from dsl import validate
    h = _hyp(_momentum_signal())
    h_bad = h.model_copy(update={"portfolio": Portfolio(type="beta_neutral")})
    g = compile_hypothesis(h_bad)
    dec = validate(g, h_bad)
    assert dec.decision.value == "reject", "desteklenmeyen portföy tipi kabul edildi!"
    assert any(i.type == "unsupported_portfolio_type" for i in dec.issues)
    print("  [ok] desteklenmeyen portföy tipi (beta_neutral) reddedildi")


def main():
    test_finds_momentum()
    test_finds_reversal()
    test_no_fake_alpha_on_random()
    test_future_prices_do_not_change_past_signal()
    test_higher_cost_lowers_return()
    test_rebalance_lowers_turnover()
    test_long_only_weights_nonnegative()
    test_unsupported_portfolio_rejected()
    print("OK — backtest motoru bilinen sinyalleri buluyor, sızıntı üretmiyor.")


if __name__ == "__main__":
    main()
