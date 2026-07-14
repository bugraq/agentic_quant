"""
Bağımsız reviewer ajanları testi (Doküman 15).

Backtest Auditor ve Statistical Reviewer yapılandırılmış ReviewReport üretmeli;
verdict kontrollerin en kötüsü olmalı; nesnel koşullara doğru tepki vermeli.
"""
from agents.backtest_auditor import BacktestAuditor
from agents.statistical_reviewer import StatisticalReviewer
from contracts.backtest_result import BacktestResult, FoldMetrics
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from contracts.review import CheckStatus
from data.synthetic import gen_cross_sectional_momentum


def _hyp(trade_time="open_t_plus_1") -> HypothesisSpec:
    sig = Expression(op="cross_sectional_rank",
                     inputs=[Expression(op="return", window=20,
                                        inputs=[Expression(op="field", field="close")])])
    return HypothesisSpec(
        hypothesis_id="hyp_r1", title="t", claim="c", family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="momentum", description="d"),
        universe=Universe(source="x"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.2, short_quantile=0.2),
        execution=Execution(signal_time="close_t", trade_time=trade_time,
                            holding_period_days=1),
        falsification=Falsification())


def _result(sharpes) -> BacktestResult:
    folds = [FoldMetrics(fold_id=f"f{i}", split="research", sharpe=s,
                         annualized_return=0.1, volatility=0.1, max_drawdown=0.1,
                         turnover=20.0) for i, s in enumerate(sharpes)]
    return BacktestResult(hypothesis_id="hyp_r1", per_fold_metrics=folds,
                          net_returns=[0.001, -0.002, 0.003] * 40)


def test_auditor_flags_leakage():
    data = gen_cross_sectional_momentum(n_sec=10, n_days=200, seed=1)
    au = BacktestAuditor()
    # sağlam: open_t_plus_1
    rep = au.audit(_hyp(), _result([0.5, 0.6]), data, cost_bps=5.0)
    ex = {c.name: c.status for c in rep.checks}
    assert ex["execution_delay"] == CheckStatus.ok
    assert ex["price_adjustment"] == CheckStatus.ok      # sentetikte adjusted_close var
    # sızıntılı: close_t execution
    bad = au.audit(_hyp(trade_time="close_t"), _result([0.5]), data, cost_bps=5.0)
    exb = {c.name: c.status for c in bad.checks}
    assert exb["execution_delay"] == CheckStatus.fail
    assert bad.verdict == CheckStatus.fail               # en kötü = fail
    print("  [ok] Backtest Auditor sızıntı/execution'ı yakalıyor")


def test_auditor_cost_zero_warns():
    data = gen_cross_sectional_momentum(n_sec=10, n_days=200, seed=1)
    rep = BacktestAuditor().audit(_hyp(), _result([0.5]), data, cost_bps=0.0)
    cost = next(c for c in rep.checks if c.name == "cost_applied")
    assert cost.status == CheckStatus.warn
    print("  [ok] maliyet 0 -> DİKKAT")


class _Row:
    def __init__(self, survives_fdr, dsr, ci_low, ci_high, raw_p):
        self.hypothesis_id = "hyp_r1"
        self.survives_fdr = survives_fdr
        self.dsr = dsr
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.raw_p = raw_p


def test_statistical_reviewer():
    sr = StatisticalReviewer()
    # güçlü: FDR geçer, DSR>0.95, CI sıfırı dışlar
    strong = sr.review(_Row(True, 0.99, 0.2, 1.4, 0.01), _result([0.5, 0.6, 0.4]))
    assert strong.verdict == CheckStatus.ok
    # zayıf: FDR geçmez, DSR düşük, CI sıfırı içerir
    weak = sr.review(_Row(False, 0.10, -0.8, 0.9, 0.6), _result([-0.1, 0.2]))
    assert weak.verdict == CheckStatus.warn
    names = {c.name for c in weak.checks}
    assert {"false_discovery_rate", "deflated_sharpe", "confidence_interval"} <= names
    print("  [ok] Statistical Reviewer FDR/DSR/CI'yı doğru yargılıyor")


def main():
    test_auditor_flags_leakage()
    test_auditor_cost_zero_warns()
    test_statistical_reviewer()
    print("OK — reviewer ajan testleri geçti.")


if __name__ == "__main__":
    main()
