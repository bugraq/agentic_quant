"""
Inversion kara deliği testi (gerçek koşuda görülen bug).

Senaryo: hiç kabul yok, en kötü başarısız hep aynı hipotez. Eski davranış:
sistem HER turda aynı hipotezi ters çevirmeye çalışıyor, LLM aynı sinyali
üretiyor, novelty duplicate'e atıyor -> bütçenin yarısı çöp (16 deneyin 9'u).

Yeni davranış: bir parent BİR KEZ ters çevrilir (lineage'dan bakılır);
ters çevrilecek yeni aday kalmayınca keşfe (new) dönülür.
"""
from contracts.decision import Decision, DecisionSource, DecisionType, Issue, Severity
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from contracts.backtest_result import BacktestResult, FoldMetrics
from contracts.research_context import GenerationMode
from memory import MemoryStore
from orchestrator.loop import _decide_mode


def _hyp(hid: str) -> HypothesisSpec:
    sig = Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=5, inputs=[Expression(op="field", field="close")])])
    return HypothesisSpec(
        hypothesis_id=hid, title="t", claim="t", family=HypothesisFamily.reversal,
        economic_mechanism=EconomicMechanism(type="x", description="y"),
        universe=Universe(source="s"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.2, short_quantile=0.2),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=5),
        falsification=Falsification())


def _reject(hid: str) -> Decision:
    return Decision(hypothesis_id=hid, decision=DecisionType.reject,
                    source=DecisionSource.gate, severity=Severity.medium,
                    issues=[Issue(type="below_acceptance_sharpe", description="x")])


def _result(hid: str, sharpe: float) -> BacktestResult:
    return BacktestResult(hypothesis_id=hid, per_fold_metrics=[
        FoldMetrics(fold_id="f0", split="research", sharpe=sharpe,
                    annualized_return=0.0, volatility=0.1,
                    max_drawdown=0.1, turnover=10.0)],
        net_returns=[0.001, -0.002, 0.001])


def test_inversion_not_repeated():
    m = MemoryStore(":memory:")
    # En kötü başarısız: hyp_0001 (Sharpe -2.0)
    m.record(_hyp("hyp_0001"), _reject("hyp_0001"), "gate_rejected",
             result=_result("hyp_0001", -2.0))

    # Inversion turu (_OP_CYCLE[7]=inversion, keşif fazından sonra): hyp_0001 seçilmeli
    mode, parent, _pb, _ = _decide_mode(iteration=7, memory=m)
    assert mode == GenerationMode.inversion and parent.hypothesis_id == "hyp_0001"

    # Inversion denendi (duplicate'e düştü diyelim) — lineage'a yazıldı
    m.record(_hyp("hyp_0002"), _reject("hyp_0002"), "duplicate",
             parent_hypothesis_id="hyp_0001", relation_type="inversion")

    # Sonraki inversion turu (_OP_CYCLE[13]=inversion): AYNI parent seçilMEmeli;
    # başka aday yok -> keşfe (new) dön
    mode2, parent2, _pb2, _ = _decide_mode(iteration=13, memory=m)
    assert mode2 == GenerationMode.new and parent2 is None, \
        f"aynı parent tekrar ters çevrildi: {mode2}, {parent2}"

    # Yeni bir kötü başarısız gelirse (hyp_0003) inversion ona geçmeli (_OP_CYCLE[18]=inversion)
    m.record(_hyp("hyp_0003"), _reject("hyp_0003"), "gate_rejected",
             result=_result("hyp_0003", -1.5))
    mode3, parent3, _pb3, _ = _decide_mode(iteration=18, memory=m)
    assert mode3 == GenerationMode.inversion and parent3.hypothesis_id == "hyp_0003"
    m.close()
    print("  [ok] aynı parent bir kez ters çevriliyor; aday bitince keşfe dönülüyor")


def _accept(hid: str) -> Decision:
    return Decision(hypothesis_id=hid, decision=DecisionType.accept,
                    source=DecisionSource.gate, severity=Severity.low)


def test_revision_quarantine():
    """Revizyonları 3 kez duplicate üretmiş champion karantinaya alınmalı;
    sıradaki en iyi kabule geçilmeli (gerçek koşu: 24 slotun ~8'i aynı
    champion'ın duplicate revizyonuydu)."""
    m = MemoryStore(":memory:")
    m.record(_hyp("hyp_0001"), _accept("hyp_0001"), "accepted",
             result=_result("hyp_0001", 0.9))      # champion
    m.record(_hyp("hyp_0002"), _accept("hyp_0002"), "accepted",
             result=_result("hyp_0002", 0.6))      # ikinci kabul

    # Champion normalde hyp_0001 (Sharpe 0.9 > 0.6). Revision turu seç (_OP_CYCLE[5]=revision).
    mode, parent, _pb, _ = _decide_mode(iteration=5, memory=m)
    assert mode == GenerationMode.revision and parent.hypothesis_id == "hyp_0001"

    # hyp_0001'in revizyonları 3 kez duplicate üretti -> karantina
    for k in range(3):
        dup = Decision(hypothesis_id=f"hyp_d{k}", decision=DecisionType.duplicate,
                       source=DecisionSource.novelty, severity=Severity.low)
        m.record(_hyp(f"hyp_d{k}"), dup, "duplicate",
                 parent_hypothesis_id="hyp_0001", relation_type="refinement")

    assert "hyp_0001" in m.exhausted_revision_parent_ids()
    mode2, parent2, _pb2, _ = _decide_mode(iteration=8, memory=m)  # _OP_CYCLE[8]=revision
    assert mode2 == GenerationMode.revision and parent2.hypothesis_id == "hyp_0002", \
        f"karantina çalışmadı: {mode2}, {parent2 and parent2.hypothesis_id}"
    m.close()
    print("  [ok] tükenen champion karantinada; revizyon sıradaki kabule geçti")


def main():
    test_inversion_not_repeated()
    test_revision_quarantine()
    print("OK — inversion kara deliği kapatıldı.")


if __name__ == "__main__":
    main()
