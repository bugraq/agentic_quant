"""
Sızıntı (leakage) testleri — Doküman 23.4 'leakage mutation testleri'.

Kasıtlı hatalı stratejileri validator'ın YAKALADIĞINI, geçerli stratejiyi
ise KABUL ettiğini doğrular. Bu, projenin en kritik güvencesi:
sızıntıyı 'test ederek' değil, 'ifade edilemez kılarak + eşitsizlikle' önlüyoruz.
"""
from contracts.decision import DecisionType
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism,
    Execution,
    Falsification,
    HypothesisFamily,
    HypothesisSpec,
    Portfolio,
    Universe,
)
from dsl import CompileError, compile_hypothesis, validate


def _hyp(signal: Expression, trade_time: str = "open_t_plus_1", **feat) -> HypothesisSpec:
    """Test için minimal ama geçerli bir HypothesisSpec kabuğu."""
    return HypothesisSpec(
        hypothesis_id="hyp_test",
        title="test",
        claim="test",
        family=HypothesisFamily.reversal,
        economic_mechanism=EconomicMechanism(type="x", description="y"),
        universe=Universe(source="sp500_point_in_time"),
        features=feat.get("features", []),
        signal=signal,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.1, short_quantile=0.1),
        execution=Execution(signal_time="close_t", trade_time=trade_time,
                            holding_period_days=5),
        falsification=Falsification(),
    )


def _cs_rank_of(inner: Expression) -> Expression:
    return Expression(op="cross_sectional_rank", inputs=[inner])


def test_valid_strategy_accepted():
    # negate(return(close,5)) @ close_t, işlem open_t+1 -> GEÇERLİ
    sig = _cs_rank_of(Expression(op="negate", inputs=[
        Expression(op="return", window=5, inputs=[Expression(op="field", field="close")])]))
    graph = compile_hypothesis(_hyp(sig, trade_time="open_t_plus_1"))
    dec = validate(graph, _hyp(sig, trade_time="open_t_plus_1"))
    assert dec.decision == DecisionType.accept, dec.issues
    print("  [ok] geçerli strateji kabul edildi")


def test_same_bar_execution_leak():
    # close_t sinyal + close_t execution -> SIZINTI (revise)
    sig = _cs_rank_of(Expression(op="return", window=5,
                                 inputs=[Expression(op="field", field="close")]))
    h = _hyp(sig, trade_time="close_t")
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.revise
    assert any(i.type == "temporal_leakage" for i in dec.issues)
    print("  [ok] close_t sinyal + close_t execution sızıntısı yakalandı")


def test_execution_before_signal_leak():
    # sinyal close_t (tick 1), işlem open_t (tick 0) -> işlem sinyalden ÖNCE
    sig = _cs_rank_of(Expression(op="field", field="close"))
    h = _hyp(sig, trade_time="open_t")
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.revise
    assert any(i.type == "temporal_leakage" for i in dec.issues)
    print("  [ok] işlemin sinyalden önce olması yakalandı")


def test_negative_window_rejected():
    # lag(close, -3) = 3 gün İLERİ bak -> negatif pencere reddi
    sig = _cs_rank_of(Expression(op="lag", window=-3,
                                 inputs=[Expression(op="field", field="close")]))
    h = _hyp(sig)
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.reject
    assert any(i.type == "invalid_parameter" for i in dec.issues)
    print("  [ok] negatif pencere (ileri bakış) reddedildi")


def test_lag_is_safe():
    # lag(close, 1) tick'i geriye çeker (close_{t-1}); close_t işlemde bile güvenli
    sig = _cs_rank_of(Expression(op="lag", window=1,
                                 inputs=[Expression(op="field", field="close")]))
    h = _hyp(sig, trade_time="close_t")
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.accept, dec.issues
    print("  [ok] lag geriye kaydırıyor, güvenli")


def test_unknown_operator_rejected():
    sig = Expression(op="magic_alpha", inputs=[Expression(op="field", field="close")])
    try:
        compile_hypothesis(_hyp(sig))
        raise AssertionError("bilinmeyen operatör derlendi")
    except CompileError:
        print("  [ok] bilinmeyen operatör derlemede reddedildi")


def test_unknown_field_rejected():
    # 'insider_tip' diye bir alan yok -> alternatif/izinsiz veri
    sig = _cs_rank_of(Expression(op="field", field="insider_tip"))
    try:
        compile_hypothesis(_hyp(sig))
        raise AssertionError("bilinmeyen alan derlendi")
    except CompileError:
        print("  [ok] izinsiz veri alanı derlemede reddedildi")


def test_degenerate_conditional_rejected():
    # conditional'ın iki dalı aynı -> sahte koşullama (reward hacking) -> reddedilmeli
    same = Expression(op="field", field="close")
    cond = Expression(op="greater_than", inputs=[
        Expression(op="volatility", window=20, inputs=[Expression(op="field", field="close")]),
        Expression(op="const", value=0.02)])
    sig = _cs_rank_of(Expression(op="conditional", inputs=[cond, same, same]))
    h = _hyp(sig)
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.reject
    assert any(i.type == "degenerate_conditional" for i in dec.issues)
    print("  [ok] dejenere conditional (iki dalı aynı) reddedildi")


def test_excessive_complexity_rejected():
    # 45 iç içe negate -> karmaşıklık sınırı aşımı
    e: Expression = Expression(op="field", field="close")
    for _ in range(45):
        e = Expression(op="negate", inputs=[e])
    h = _hyp(_cs_rank_of(e))
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.reject
    assert any(i.type == "excessive_complexity" for i in dec.issues)
    print("  [ok] aşırı karmaşık strateji reddedildi")


def main():
    test_valid_strategy_accepted()
    test_same_bar_execution_leak()
    test_execution_before_signal_leak()
    test_negative_window_rejected()
    test_lag_is_safe()
    test_unknown_operator_rejected()
    test_unknown_field_rejected()
    test_degenerate_conditional_rejected()
    test_excessive_complexity_rejected()
    print("OK — tüm sızıntı/geçerlilik testleri geçti.")


if __name__ == "__main__":
    main()
