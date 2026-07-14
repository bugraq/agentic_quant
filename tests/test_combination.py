"""
Combination modu testi (Doküman 16.3).

_decide_mode, en az 2 kabul edilmiş hipotez varken uygun turda combination
modunu İKİ ebeveyple döndürmeli; generator prompt'u iki ebeveyni de içermeli.
"""
import os
import tempfile

from contracts.dsl import Expression
from contracts.decision import Decision, DecisionSource, DecisionType
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from contracts.backtest_result import BacktestResult, FoldMetrics
from contracts.research_context import GenerationMode
from memory import MemoryStore
from orchestrator.loop import _decide_mode, _build_context, CampaignConfig
from agents.hypothesis_generator import _build_user_prompt


def _accepted(hid, w, op="return") -> HypothesisSpec:
    sig = Expression(op="cross_sectional_rank",
                     inputs=[Expression(op=op, window=w,
                                        inputs=[Expression(op="field", field="close")])])
    return HypothesisSpec(
        hypothesis_id=hid, title=f"{w}g {op}", claim="c", family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="momentum", description="d"),
        universe=Universe(source="x"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.2, short_quantile=0.2),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def _result(sharpe) -> BacktestResult:
    return BacktestResult(
        hypothesis_id="x",
        per_fold_metrics=[FoldMetrics(fold_id="f0", split="research", sharpe=sharpe,
                                      annualized_return=0.1, volatility=0.1,
                                      max_drawdown=0.1, turnover=20.0)],
        net_returns=[0.001, -0.002, 0.003] * 40)


def test_combination_triggers_with_distinct_accepts():
    with tempfile.TemporaryDirectory() as d:
        mem = MemoryStore(os.path.join(d, "m.sqlite"))
        # YAPISAL OLARAK FARKLI iki kabul (return vs zscore) -> combination anlamlı
        for hid, w, op, s in [("hyp_0001", 60, "return", 1.2),
                              ("hyp_0002", 20, "zscore", 0.8)]:
            dec = Decision(hypothesis_id=hid, decision=DecisionType.accept,
                           source=DecisionSource.gate)
            mem.record(_accepted(hid, w, op), dec, "accepted", result=_result(s))
        # _OP_CYCLE[4] = combination (keşif fazından sonra)
        mode, pa, pb, _cs = _decide_mode(4, mem)
        assert mode == GenerationMode.combination, f"combination bekleniyordu, {mode}"
        assert pa is not None and pb is not None, "iki ebeveyn de dolu olmalı"
        assert pa.hypothesis_id != pb.hypothesis_id
        cfg = CampaignConfig(goal="g")
        ctx = _build_context(cfg, mem, 5, mode, pa, parent_b=pb)
        prompt = _build_user_prompt(ctx)
        assert "BİRLEŞTİRME" in prompt
        assert pa.hypothesis_id in prompt and pb.hypothesis_id in prompt
        mem.close()
    print("  [ok] yapısal FARKLI 2 kabul -> combination (iki ebeveyn)")


def test_no_combination_with_identical_structures():
    """Tüm kabuller AYNI yapıdaysa combination anlamsız — keşfe düşmeli
    (gerçek koşuda 27 birleştirme denemesinin 0'ı kabuldü, hepsi aynı reversal)."""
    with tempfile.TemporaryDirectory() as d:
        mem = MemoryStore(os.path.join(d, "m.sqlite"))
        # İkisi de return -> AYNI yapı
        for hid, w in [("hyp_0001", 60), ("hyp_0002", 20)]:
            dec = Decision(hypothesis_id=hid, decision=DecisionType.accept,
                           source=DecisionSource.gate)
            mem.record(_accepted(hid, w, "return"), dec, "accepted", result=_result(1.0))
        mode, pa, pb, _cs = _decide_mode(4, mem)   # combination turu
        assert mode != GenerationMode.combination, \
            "aynı yapıdaki kabullerle combination tetiklenmemeli"
        mem.close()
    print("  [ok] aynı yapıdaki kabullerle combination tetiklenmiyor (israf önlendi)")


def test_no_combination_with_one_accept():
    with tempfile.TemporaryDirectory() as d:
        mem = MemoryStore(os.path.join(d, "m.sqlite"))
        dec = Decision(hypothesis_id="hyp_0001", decision=DecisionType.accept,
                       source=DecisionSource.gate)
        mem.record(_accepted("hyp_0001", 60), dec, "accepted", result=_result(1.0))
        mode, pa, pb, _cs = _decide_mode(4, mem)
        assert mode != GenerationMode.combination, "tek kabulle combination olmamalı"
        assert pb is None
        mem.close()
    print("  [ok] tek kabulle combination tetiklenmiyor")


def main():
    test_combination_triggers_with_distinct_accepts()
    test_no_combination_with_identical_structures()
    test_no_combination_with_one_accept()
    print("OK — combination modu testleri geçti.")


if __name__ == "__main__":
    main()
