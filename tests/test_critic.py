"""
Quant Critic testleri (Doküman 15) — LLM'siz, sahte client ile parse mantığı.
"""
from contracts.decision import DecisionSource, DecisionType
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from agents.quant_critic import DummyCritic, LLMCritic
from llm.openai_client import LLMResponse


def _hyp() -> HypothesisSpec:
    sig = Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=60, inputs=[Expression(op="field", field="close")])])
    return HypothesisSpec(
        hypothesis_id="hyp_c", title="t", claim="momentum", family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="momentum", description="y"),
        universe=Universe(source="sp500_point_in_time"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


class _FakeClient:
    def __init__(self, text: str):
        self._text = text
    def chat(self, *a, **k) -> LLMResponse:
        return LLMResponse(text=self._text, prompt_tokens=10, completion_tokens=5)


def test_dummy_accepts():
    dec = DummyCritic().review(_hyp())
    assert dec.decision == DecisionType.accept and dec.source == DecisionSource.critic
    print("  [ok] DummyCritic her şeyi geçiriyor")


def test_llm_reject_parsed():
    client = _FakeClient('{"decision":"reject","severity":"high",'
                         '"issues":[{"type":"mechanism","description":"tutarsız"}]}')
    dec = LLMCritic(client, model="x").review(_hyp())
    assert dec.decision == DecisionType.reject
    assert dec.source == DecisionSource.critic
    assert dec.issues and dec.issues[0].type == "mechanism"
    print("  [ok] LLMCritic reject kararını doğru parse etti")


def test_llm_garbage_fails_open():
    dec = LLMCritic(_FakeClient("bu JSON değil, saçmalık"), model="x").review(_hyp())
    assert dec.decision == DecisionType.accept   # fail-open: araştırmayı bloklama
    print("  [ok] bozuk çıktıda fail-open (accept)")


def main():
    test_dummy_accepts()
    test_llm_reject_parsed()
    test_llm_garbage_fails_open()
    print("OK — critic testleri geçti.")


if __name__ == "__main__":
    main()
