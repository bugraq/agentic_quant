"""
Decision — yönlendirme kararı (Doküman 15).

İki tür karar birleşir:
  - Deterministik: hard gate + istatistiksel test (geçti/kaldı, tartışmasız)
  - LLM Critic: ekonomik mekanizma mantıklı mı, gizli faktör mü
Orchestrator ikisini birleştirip nihai rotayı seçer. Deterministik gate
her zaman LLM'i ezer (look-ahead varsa LLM "güzel" dese bile red).

Reviewer çıktısı serbest metin DEĞİL, yapılandırılmıştır.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    accept = "accept"
    revise = "revise"
    reject = "reject"
    duplicate = "duplicate"
    promote_to_holdout = "promote_to_holdout"


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class DecisionSource(str, Enum):
    gate = "gate"                # deterministik hard gate
    statistical = "statistical"  # multiple testing / istatistik
    critic = "critic"            # LLM eleştirmen
    novelty = "novelty"          # benzerlik/tekrar kontrolü


class Issue(BaseModel):
    """Tespit edilen tek bir sorun."""

    type: str = Field(..., description="Örn: execution_bias, survivorship, duplicate")
    description: str
    required_action: Optional[str] = None


class Decision(BaseModel):
    """Bir hipotez hakkında verilen yapılandırılmış karar."""

    hypothesis_id: str
    decision: DecisionType
    source: DecisionSource
    severity: Severity = Severity.low
    issues: list[Issue] = Field(default_factory=list)
    revision_direction: Optional[str] = None

    model_config = {"extra": "forbid"}
