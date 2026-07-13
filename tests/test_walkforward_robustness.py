"""
Walk-forward + sağlamlık testleri (Doküman 9.1, 9.2).
Gerçek momentum: fold'lar tutarlı + permutation'da alpha yok olur (robust).
Rastgele veri: permutation gerçek sinyalden ayırt edilemez (robust DEĞİL).
"""
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from dsl import compile_hypothesis
from data import gen_cross_sectional_momentum, gen_random
from backtest.walk_forward import run_walk_forward
from evaluation.robustness import run_robustness


def _mom_hyp() -> HypothesisSpec:
    sig = Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=60, inputs=[Expression(op="field", field="close")])])
    return HypothesisSpec(
        hypothesis_id="hyp_wf", title="60g momentum", claim="t",
        family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="momentum", description="y"),
        universe=Universe(source="sp500_point_in_time"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def test_walk_forward_folds():
    h = _mom_hyp()
    res = run_walk_forward(compile_hypothesis(h), h, gen_cross_sectional_momentum(seed=1),
                           n_folds=5, cost_bps=1.0)
    assert len(res.per_fold_metrics) >= 4
    frac = res.exposures["positive_fold_fraction"]
    assert frac >= 0.6, f"momentum fold tutarsız: {frac:.0%}"
    print(f"  [ok] walk-forward: {len(res.per_fold_metrics)} fold, pozitif oran {frac:.0%}")


def test_momentum_is_robust():
    h = _mom_hyp()
    rob = run_robustness(compile_hypothesis(h), h, gen_cross_sectional_momentum(seed=1),
                         cost_bps=1.0)
    assert rob.permutation_pvalue < 0.2, f"perm_p yüksek: {rob.permutation_pvalue}"
    assert rob.cost2x_sharpe > 0, f"maliyet 2x'te çöktü: {rob.cost2x_sharpe}"
    print(f"  [ok] momentum robust: perm_p={rob.permutation_pvalue:.2f}, "
          f"cost2x={rob.cost2x_sharpe:.2f}, param_min={rob.param_min_sharpe:.2f}")


def test_random_is_not_robust():
    h = _mom_hyp()
    rob = run_robustness(compile_hypothesis(h), h, gen_random(seed=3), cost_bps=1.0)
    assert not rob.robust, "rastgele veri robust çıktı — sahte alpha!"
    print(f"  [ok] rastgele veri robust değil: perm_p={rob.permutation_pvalue:.2f}")


def main():
    test_walk_forward_folds()
    test_momentum_is_robust()
    test_random_is_not_robust()
    print("OK — walk-forward + sağlamlık testleri geçti.")


if __name__ == "__main__":
    main()
