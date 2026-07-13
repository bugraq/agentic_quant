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
from contracts.dsl import Expression
from contracts.hypothesis_spec import HypothesisSpec
from llm.openai_client import OpenAICompatibleClient


def _collect_ops(expr: Expression, acc: set) -> set:
    """Sinyal ağacındaki tüm operatör adlarını topla (deterministik analiz)."""
    if expr.op not in ("field", "const", "feature_ref"):
        acc.add(expr.op)
    for inp in expr.inputs:
        if isinstance(inp, Expression):
            _collect_ops(inp, acc)
    return acc


def _signal_facts(hyp: HypothesisSpec) -> str:
    """Critic'e verilecek deterministik yapı özeti — LLM DSL'i kendi parse etmesin."""
    ops = _collect_ops(hyp.signal, set())
    conditioning = bool(ops & {"conditional", "greater_than", "less_than"})
    volatility = bool(ops & {"volatility", "rolling_std"})
    combiner = bool(ops & {"multiply", "add", "subtract", "divide", "ratio"})
    return (f"Operatörler: {sorted(ops)}\n"
            f"  - Koşullama/rejim yapısı var mı: {'EVET' if conditioning else 'HAYIR'}\n"
            f"  - Volatilite ölçümü var mı: {'EVET' if volatility else 'HAYIR'}\n"
            f"  - Birden çok sinyal birleştiriliyor mu: {'EVET' if combiner or conditioning else 'HAYIR'}")


class Critic(Protocol):
    def review(self, hyp: HypothesisSpec) -> Decision: ...


class DummyCritic:
    """LLM yokken: her şeyi geçirir (pipeline'ı bloklamaz)."""

    def review(self, hyp: HypothesisSpec) -> Decision:
        return Decision(hypothesis_id=hyp.hypothesis_id, decision=DecisionType.accept,
                        source=DecisionSource.critic, severity=Severity.low)


_SYSTEM = """Sen kıdemli, şüpheci bir kantitatif araştırma eleştirmenisin. Sana bir
hipotez veriliyor; SONUÇLARI görmeden yalnızca EKONOMİK muhakemeyi ve etiketlerin
dürüstlüğünü denetliyorsun. Değerlendir:

  1. ETİKET-SİNYAL UYUŞMASI (en önemli): title/claim/family, sinyalin GERÇEKTE
     yaptığıyla tutarlı mı? Sana SİNYAL YAPISI bölümünde deterministik bir analiz
     verilecek; sinyali KENDİN parse etme, o analize güven. Kurallar:
       - claim 'regime'/'rejim' diyorsa: 'Koşullama var mı' EVET olmalı.
       - claim 'volatility' diyorsa: 'Volatilite var mı' EVET olmalı.
       - claim 'composite'/'birleşik' diyorsa: 'Birden çok sinyal' EVET olmalı.
     Gerekli özellik analizde EVET ise etiket DÜRÜSTTÜR -> mismatch verme, 'accept'.
     Sadece iddia bir yapı vaat edip analizde o yapı HAYIR ise
     -> decision='revise', type='claim_signal_mismatch'.
  2. Ekonomik mekanizma tutarlı ve makul mü?
  3. Bilinen bir faktörün yeniden adlandırılması mı?

Klasik faktörler (momentum/reversal) meşrudur AMA doğru etiketlenmeli: sade
momentum'a momentum de, 'regime-conditioned' deme. Yalın ve dürüst etiketli bir
momentum -> 'accept'. İddiası sinyalini aşan (abartılı etiket) -> 'revise'.

SADECE şu şemada JSON döndür:
{"decision": "accept|revise|reject", "severity": "low|medium|high",
 "issues": [{"type": "...", "description": "...", "required_action": "..."}]}"""


def _user(hyp: HypothesisSpec) -> str:
    return (f"Başlık: {hyp.title}\n"
            f"İddia: {hyp.claim}\n"
            f"Aile: {hyp.family.value}\n"
            f"Ekonomik mekanizma: {hyp.economic_mechanism.type} — "
            f"{hyp.economic_mechanism.description}\n\n"
            f"SİNYAL YAPISI (deterministik analiz — buna güven):\n{_signal_facts(hyp)}\n\n"
            f"Bu hipotezi değerlendir ve JSON kararını ver.")


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
