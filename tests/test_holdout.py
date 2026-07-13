"""
Holdout servisi testleri (Doküman 10.3).
One-shot (tekrar değerlendirme yasak), aday kotası, sadece özet döner.
"""
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from data import gen_cross_sectional_momentum
from holdout import HoldoutError, HoldoutService


def _hyp(hid: str) -> HypothesisSpec:
    sig = Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=60, inputs=[Expression(op="field", field="close")])])
    return HypothesisSpec(
        hypothesis_id=hid, title="60g momentum", claim="t",
        family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="momentum", description="y"),
        universe=Universe(source="sp500_point_in_time"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def _service(max_candidates=20):
    return HoldoutService(gen_cross_sectional_momentum(seed=9), audit_path=":memory:",
                          max_candidates=max_candidates, min_sharpe=0.3, cost_bps=1.0)


def test_evaluate_returns_summary():
    svc = _service()
    res = svc.evaluate(_hyp("hyp_a"))
    assert res.hypothesis_id == "hyp_a"
    assert isinstance(res.passed, bool)
    assert len(svc.audit_log()) == 1
    print(f"  [ok] holdout değerlendirdi: Sharpe={res.sharpe:.2f}, geçti={res.passed}")


def test_one_shot_enforced():
    svc = _service()
    svc.evaluate(_hyp("hyp_a"))
    try:
        svc.evaluate(_hyp("hyp_a"))
        raise AssertionError("one-shot ihlali kabul edildi")
    except HoldoutError:
        print("  [ok] aynı adayın tekrar değerlendirilmesi engellendi (one-shot)")


def test_candidate_quota():
    svc = _service(max_candidates=1)
    svc.evaluate(_hyp("hyp_a"))
    try:
        svc.evaluate(_hyp("hyp_b"))
        raise AssertionError("kota aşımı kabul edildi")
    except HoldoutError:
        print("  [ok] aday kotası uygulandı")


def main():
    test_evaluate_returns_summary()
    test_one_shot_enforced()
    test_candidate_quota()
    print("OK — holdout servisi testleri geçti.")


if __name__ == "__main__":
    main()
