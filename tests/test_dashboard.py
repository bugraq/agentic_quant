"""
Dashboard smoke testi — mini bir hafızadan HTML üretiliyor mu.
"""
import os
import tempfile

from contracts.backtest_result import BacktestResult, FoldMetrics
from contracts.decision import Decision, DecisionSource, DecisionType
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from dashboard import generate_dashboard
from memory import MemoryStore


def _hyp() -> HypothesisSpec:
    sig = Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=60, inputs=[Expression(op="field", field="close")])])
    return HypothesisSpec(
        hypothesis_id="hyp_d1", title="60g momentum", claim="t",
        family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="momentum", description="y"),
        universe=Universe(source="sp500_point_in_time"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def test_generate_dashboard():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "mem.sqlite")
        store = MemoryStore(db)
        result = BacktestResult(
            hypothesis_id="hyp_d1",
            per_fold_metrics=[FoldMetrics(fold_id="f0", split="research", sharpe=0.8,
                                          annualized_return=0.1, volatility=0.12,
                                          max_drawdown=0.09, turnover=5.0)],
            net_returns=[0.001, -0.002, 0.003] * 40)
        dec = Decision(hypothesis_id="hyp_d1", decision=DecisionType.accept,
                       source=DecisionSource.gate)
        store.record(_hyp(), dec, "accepted", result=result)
        store.close()

        out = os.path.join(d, "dash.html")
        generate_dashboard(db, os.path.join(d, "yok.sqlite"), out, campaign_name="test")
        assert os.path.exists(out)
        content = open(out, encoding="utf-8").read()
        for token in ["Araştırma Paneli", "En İyi Stratejiler", "Araştırma Hunisi",
                      "Çoklu Test", "Holdout", "hyp_d1"]:
            assert token in content, f"eksik bölüm: {token}"
        print("  [ok] dashboard tüm bölümlerle üretildi")


def main():
    test_generate_dashboard()
    print("OK — dashboard testi geçti.")


if __name__ == "__main__":
    main()
