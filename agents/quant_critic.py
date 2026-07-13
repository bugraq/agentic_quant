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


def _label_honesty_issue(hyp: HypothesisSpec) -> "Issue | None":
    """
    DETERMİNİSTİK etiket-sinyal uyuşma kontrolü (LLM'e bırakılmaz).

    Yalnızca YAPISAL bir söz veren aileler denetlenir: regime_conditioned
    (koşullama gerekir), composite/cross_sectional_interaction (birleştirme
    gerekir). Momentum/reversal/volume/volatility/liquidity TEK-faktör
    ailelerdir; sade sinyalle TAM eşleşir, ekstra yapı GEREKTİRMEZ.
    """
    ops = _collect_ops(hyp.signal, set())
    conditioning = bool(ops & {"conditional", "greater_than", "less_than"})
    combiner = bool(ops & {"multiply", "add", "subtract", "divide", "ratio"}) or conditioning
    fam = hyp.family.value
    if fam == "regime_conditioned" and not conditioning:
        return Issue(type="claim_signal_mismatch",
                     description="family 'regime_conditioned' ama sinyalde koşullama "
                                 "(conditional/greater_than) yok.",
                     required_action="Sinyale rejim koşulu ekle ya da family'yi düzelt.")
    if fam in ("composite", "cross_sectional_interaction") and not combiner:
        return Issue(type="claim_signal_mismatch",
                     description=f"family '{fam}' ama sinyal tek bir faktör — birleştirme yok.",
                     required_action="Birden çok sinyali birleştir ya da family'yi düzelt.")
    return None


class Critic(Protocol):
    def review(self, hyp: HypothesisSpec) -> Decision: ...


def _mismatch_decision(hyp: HypothesisSpec, issue: "Issue") -> Decision:
    return Decision(hypothesis_id=hyp.hypothesis_id, decision=DecisionType.revise,
                    source=DecisionSource.critic, severity=Severity.medium, issues=[issue])


class DummyCritic:
    """LLM yok: sadece deterministik etiket-sinyal kontrolü (yapısal aileler)."""

    def review(self, hyp: HypothesisSpec) -> Decision:
        issue = _label_honesty_issue(hyp)
        if issue is not None:
            return _mismatch_decision(hyp, issue)
        return Decision(hypothesis_id=hyp.hypothesis_id, decision=DecisionType.accept,
                        source=DecisionSource.critic, severity=Severity.low)


_SYSTEM = """Sen kıdemli bir kantitatif araştırma eleştirmenisin. Bir hipotezin
yalnızca EKONOMİK muhakemesini değerlendir (sonuçları görmeden).

VARSAYILAN KARAR 'accept'tir. Momentum, reversal, volume, volatility, liquidity
gibi KLASİK faktörler tamamen MEŞRUDUR; sade olmaları reddi GEREKTİRMEZ. Etiketin
sinyal yapısına uyup uymadığı AYRI bir deterministik sistemce kontrol edilir; sen
onunla ilgilenme.

Yalnızca şu durumlarda 'revise'/'reject' ver:
  - Ekonomik mekanizma açıkça tutarsız/anlamsız/kendisiyle çelişkili ise, VEYA
  - Sinyalin yönü iddianın yönüyle açıkça ters ise (ör. claim 'kazananlar kazanır'
    ama sinyal kaybedenleri long yapıyor).
Şüphedeysen 'accept'. Basitlik, sadelik, klasik-faktör olması RED SEBEBİ DEĞİLDİR.

SADECE şu şemada JSON döndür:
{"decision": "accept|revise|reject", "severity": "low|medium|high",
 "issues": [{"type": "...", "description": "...", "required_action": "..."}]}"""


def _user(hyp: HypothesisSpec) -> str:
    return (f"Başlık: {hyp.title}\n"
            f"İddia: {hyp.claim}\n"
            f"Aile: {hyp.family.value}\n"
            f"Ekonomik mekanizma: {hyp.economic_mechanism.type} — "
            f"{hyp.economic_mechanism.description}\n\n"
            f"Bu hipotezin ekonomik muhakemesini değerlendir ve JSON kararını ver.")


class LLMCritic:
    def __init__(self, client: OpenAICompatibleClient, model: str,
                 temperature: float = 0.2) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def review(self, hyp: HypothesisSpec) -> Decision:
        # 1) DETERMİNİSTİK etiket-sinyal kontrolü (yapısal aileler) — LLM'e sorma
        issue = _label_honesty_issue(hyp)
        if issue is not None:
            return _mismatch_decision(hyp, issue)
        # 2) LLM yalnızca ÖZNEL ekonomik yargı için (varsayılan accept)
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
