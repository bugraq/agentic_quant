"""
Benzerlik/yenilik kontrolü testleri (Doküman 14).
Aynı yapı -> yapısal duplicate; farklı yapı -> değil.
"""
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from memory.similarity import NoveltyIndex


def _hyp(hid: str, signal: Expression) -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id=hid, title=hid, claim="t", family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="x", description="y"),
        universe=Universe(source="sp500_point_in_time"), features=[], signal=signal,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def _mom(w: int) -> Expression:
    return Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=w, inputs=[Expression(op="field", field="close")])])


def test_identical_structure_is_duplicate():
    idx = NoveltyIndex()
    idx.add(_hyp("hyp_a", _mom(60)))
    assert idx.check_structural(_hyp("hyp_b", _mom(60))) == "hyp_a"
    print("  [ok] aynı yapı yapısal duplicate olarak yakalandı")


def test_different_window_not_duplicate():
    # 60g vs 5g: token 'return:60' vs 'return:5' farklı -> Jaccard < 0.90
    idx = NoveltyIndex()
    idx.add(_hyp("hyp_a", _mom(60)))
    assert idx.check_structural(_hyp("hyp_b", _mom(5))) is None
    print("  [ok] farklı pencere duplicate sayılmadı")


def test_first_is_never_duplicate():
    idx = NoveltyIndex()
    assert idx.check_structural(_hyp("hyp_a", _mom(60))) is None
    print("  [ok] ilk hipotez duplicate değil")


def main():
    test_identical_structure_is_duplicate()
    test_different_window_not_duplicate()
    test_first_is_never_duplicate()
    print("OK — benzerlik/yenilik kontrolü çalışıyor.")


if __name__ == "__main__":
    main()
