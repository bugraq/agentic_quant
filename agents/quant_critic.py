"""
Quant Critic ajanı (Doküman 15) — BAĞIMSIZ eleştirmen.

Üreten LLM kendi stratejisini onaylamamalı. Bu ajan farklı prompt (ve
tercihen farklı model) ile ekonomik mekanizmayı denetler:
  - Ekonomik mekanizma mantıklı mı?
  - Sinyal gerçekten hipotezi temsil ediyor mu?
  - Alternatif (daha sıradan) bir açıklama var mı?
  - Bilinen bir faktörün yeniden adlandırılmış hali mi?

Sonuç yapılandırılmış Decision'dır. İstatistik/sızıntı DETERMİNİSTİK
katmanların işi; critic yalnızca ekonomik muhakeme yapar (sonuç görmeden).
"""
from __future__ import annotations

import json
from typing import Protocol

from contracts.decision import (
    Decision, DecisionSource, DecisionType, Issue, Severity,
)
from contracts.hypothesis_spec import HypothesisSpec
from llm.openai_client import OpenAICompatibleClient


class Critic(Protocol):
    def review(self, hyp: HypothesisSpec) -> Decision: ...


class DummyCritic:
    """LLM yokken: her şeyi geçirir (pipeline'ı bloklamaz)."""

    def review(self, hyp: HypothesisSpec) -> Decision:
        return Decision(hypothesis_id=hyp.hypothesis_id, decision=DecisionType.accept,
                        source=DecisionSource.critic, severity=Severity.low)


_SYSTEM = """Sen kıdemli, şüpheci bir kantitatif araştırma eleştirmenisin. Sana bir
hipotez veriliyor; SONUÇLARI görmeden yalnızca EKONOMİK muhakemeyi denetliyorsun.
Değerlendir:
  - Ekonomik mekanizma tutarlı ve makul mü?
  - Sinyal (DSL) gerçekten iddiayı uyguluyor mu? (yön, ufuk, mantık)
  - Bilinen bir faktörün (momentum/reversal/value...) yeniden adlandırılması mı?
  - Daha sıradan/rakip bir açıklama mekanizmayı geçersiz kılıyor mu?

ÖNEMLİ: Momentum, reversal gibi KLASİK faktörler meşrudur; sırf basit diye
reddetme. Sadece mekanizma tutarsızsa veya sinyal iddiayla ÇELİŞİYORSA
'reject'/'revise' ver. Aksi halde 'accept'.

SADECE şu şemada JSON döndür:
{"decision": "accept|revise|reject", "severity": "low|medium|high",
 "issues": [{"type": "...", "description": "...", "required_action": "..."}]}"""


def _user(hyp: HypothesisSpec) -> str:
    return (f"Başlık: {hyp.title}\n"
            f"İddia: {hyp.claim}\n"
            f"Aile: {hyp.family.value}\n"
            f"Ekonomik mekanizma: {hyp.economic_mechanism.type} — "
            f"{hyp.economic_mechanism.description}\n"
            f"Sinyal (DSL): {hyp.signal.model_dump_json()}\n"
            f"Portföy: {hyp.portfolio.type}\n\n"
            f"Bu hipotezi ekonomik açıdan değerlendir ve JSON kararını ver.")


class LLMCritic:
    def __init__(self, client: OpenAICompatibleClient, model: str,
                 temperature: float = 0.2) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def review(self, hyp: HypothesisSpec) -> Decision:
        resp = self.client.chat(self.model, _SYSTEM, _user(hyp),
                                temperature=self.temperature, max_tokens=800)
        self.total_prompt_tokens += resp.prompt_tokens
        self.total_completion_tokens += resp.completion_tokens
        try:
            data = _extract_json(resp.text)
            decision = DecisionType(data.get("decision", "accept"))
            severity = Severity(data.get("severity", "low"))
            issues = [Issue(type=i.get("type", "critic"),
                            description=i.get("description", ""),
                            required_action=i.get("required_action"))
                      for i in data.get("issues", [])]
        except Exception:
            # Eleştirmen bozuk çıktı verirse araştırmayı bloklama (fail-open)
            return Decision(hypothesis_id=hyp.hypothesis_id, decision=DecisionType.accept,
                            source=DecisionSource.critic, severity=Severity.low)
        return Decision(hypothesis_id=hyp.hypothesis_id, decision=decision,
                        source=DecisionSource.critic, severity=severity, issues=issues)


def _extract_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    return json.loads(s[start:end + 1])
