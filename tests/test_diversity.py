"""
Çeşitlilik testleri: yapısal çeşitlilik metriği + az-keşfedilmiş aile hesabı.

- structure_signature pencereden BAĞIMSIZ olmalı (60g vs 90g momentum = aynı yapı).
- count_distinct_structures farklı yapıları saymalı.
- _build_context underexplored_regions'ı hiç/az denenen ailelerle doldurmalı.
"""
import os
import tempfile

from contracts.dsl import Expression, NamedFeature
from contracts.decision import Decision, DecisionSource, DecisionType
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from contracts.backtest_result import BacktestResult, FoldMetrics
from contracts.research_context import GenerationMode
from memory import MemoryStore
from memory.similarity import (
    count_distinct_structures, hypothesis_structure, structure_signature,
)
from orchestrator.loop import _build_context, CampaignConfig


def _hyp(hid, window, fam=HypothesisFamily.momentum, op="return") -> HypothesisSpec:
    sig = Expression(op="cross_sectional_rank",
                     inputs=[Expression(op=op, window=window,
                                        inputs=[Expression(op="field", field="close")])])
    return HypothesisSpec(
        hypothesis_id=hid, title="t", claim="c", family=fam,
        economic_mechanism=EconomicMechanism(type=fam.value, description="d"),
        universe=Universe(source="x"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.2, short_quantile=0.2),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def test_structure_window_independent():
    a = _hyp("h1", 60)
    b = _hyp("h2", 90)   # yalnız pencere farkı
    assert structure_signature(a.signal) == structure_signature(b.signal)
    assert hypothesis_structure(a) == hypothesis_structure(b)
    # farklı operatör -> farklı yapı
    c = _hyp("h3", 60, op="zscore")
    assert hypothesis_structure(c) != hypothesis_structure(a)
    print("  [ok] yapı imzası pencereden bağımsız, operatör farkına duyarlı")


def test_count_distinct():
    hyps = [_hyp("h1", 60), _hyp("h2", 90), _hyp("h3", 20),          # hepsi return -> 1 yapı
            _hyp("h4", 60, op="zscore"),                              # 2. yapı
            _hyp("h5", 60, fam=HypothesisFamily.volume, op="volatility")]  # 3. yapı
    assert count_distinct_structures(hyps) == 3
    print("  [ok] farklı yapı sayısı doğru (3 yapı, 5 hipotez)")


def test_underexplored_regions():
    with tempfile.TemporaryDirectory() as d:
        mem = MemoryStore(os.path.join(d, "m.sqlite"))
        # sadece momentum ailesini 2 kez backtest et
        for hid in ("h1", "h2"):
            res = BacktestResult(hypothesis_id=hid, per_fold_metrics=[
                FoldMetrics(fold_id="f0", split="research", sharpe=0.1,
                            annualized_return=0.1, volatility=0.1,
                            max_drawdown=0.1, turnover=10.0)],
                net_returns=[0.001] * 60)
            dec = Decision(hypothesis_id=hid, decision=DecisionType.reject,
                           source=DecisionSource.gate)
            mem.record(_hyp(hid, 60), dec, "gate_rejected", result=res)
        ctx = _build_context(CampaignConfig(goal="g"), mem, 5, GenerationMode.new, None)
        # momentum denendi -> underexplored'da OLMAMALI; reversal/volume/... OLMALI
        assert "momentum" not in ctx.underexplored_regions
        assert "reversal" in ctx.underexplored_regions
        assert "composite" in ctx.underexplored_regions
        mem.close()
    print("  [ok] az-keşfedilmiş aileler doğru hesaplandı")


def main():
    test_structure_window_independent()
    test_count_distinct()
    test_underexplored_regions()
    print("OK — çeşitlilik testleri geçti.")


if __name__ == "__main__":
    main()
