"""
LLM memorization önlemi testi.

Anonimleştirme AÇIKKEN (varsayılan) LLM'e giden hiçbir prompta ticker adı
veya tarih aralığı sızmamalı — LLM eğitim verisinden dönemin kazananlarını
ezbere bilir (parametre-içi look-ahead). Kapalıyken (ablation) sızar.
"""
from agents.hypothesis_generator import _build_system_prompt, _build_user_prompt
from contracts.research_context import GenerationMode
from memory import MemoryStore
from orchestrator.loop import ANONYMOUS_UNIVERSE, CampaignConfig, _build_context

_DESC = "50 büyük ABD hissesi (AAPL, MSFT, NVDA vb.), Yahoo Finance, 2015-2023"
_SECRETS = ["AAPL", "MSFT", "NVDA", "2015", "2023"]


def _ctx(anonymize: bool):
    cfg = CampaignConfig(universe_description=_DESC, anonymize_universe=anonymize)
    memory = MemoryStore(":memory:")
    ctx = _build_context(cfg, memory, remaining=5, mode=GenerationMode.new, parent=None)
    memory.close()
    return ctx


def test_anonymize_on_hides_tickers_and_dates():
    ctx = _ctx(anonymize=True)
    assert ctx.universe_description == ANONYMOUS_UNIVERSE
    full_prompt = _build_system_prompt(ctx) + _build_user_prompt(ctx)
    for secret in _SECRETS:
        assert secret not in full_prompt, f"prompta sızdı: {secret}"
    print("  [ok] anonimleştirme açık: prompta ticker/tarih sızmıyor")


def test_anonymize_off_is_ablation():
    ctx = _ctx(anonymize=False)
    assert "AAPL" in ctx.universe_description
    print("  [ok] anonimleştirme kapalı (ablation): gerçek tarif gidiyor")


def main():
    test_anonymize_on_hides_tickers_and_dates()
    test_anonymize_off_is_ablation()
    print("OK — memorization önlemi testleri geçti.")


if __name__ == "__main__":
    main()
