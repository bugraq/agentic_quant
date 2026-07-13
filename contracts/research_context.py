"""
ResearchContext — LLM'e giden HER ŞEY (Doküman: pipeline istasyon 1).

Bu obje prompt'un kendisidir; deterministik olarak metne serialize edilir.
LLM'in gördüğü tek dünya budur. Ham fiyat verisi ASLA burada değildir —
varlıklar sadece `universe_description` metnine "indirgenir". Bu, LLM'in
veriye overfit etmesini ve sızıntı yapmasını yapısal olarak engeller.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from contracts.decision import Decision
from contracts.hypothesis_spec import HypothesisSpec


class GenerationMode(str, Enum):
    """Hipotez üretim modu (Doküman 4.3)."""

    new = "new"
    revision = "revision"
    combination = "combination"
    inversion = "inversion"
    structural_mutation = "structural_mutation"
    regime_adaptation = "regime_adaptation"


class ExperimentSummary(BaseModel):
    """Geçmiş bir denemenin sıkıştırılmış özeti (başarılı VEYA başarısız)."""

    hypothesis_id: str
    title: str
    family: str
    outcome: str = Field(..., description="accepted / rejected / duplicate ...")
    headline_metric: Optional[str] = None  # örn: "OOS Sharpe 0.42"
    lesson: Optional[str] = None            # semantic memory'den çıkarım


class ResearchContext(BaseModel):
    """Orchestrator'ın kurup Hypothesis Generator'a verdiği bağlam."""

    campaign_goal: str
    universe_description: str = Field(
        ..., description="Varlıkların metne indirgenmiş hali — LLM'in gördüğü tek evren"
    )

    # Hipotez uzayını kısıtlayan sınırlar
    allowed_operators: list[str] = Field(default_factory=list)
    allowed_horizons: list[int] = Field(default_factory=list)
    allowed_rebalance: list[str] = Field(default_factory=list)
    allowed_portfolio_types: list[str] = Field(default_factory=list)

    # Hafızadan gelen bağlam (tekrarı önler, yön verir)
    prior_experiments: list[ExperimentSummary] = Field(default_factory=list)
    factor_families_seen: list[str] = Field(default_factory=list)
    underexplored_regions: list[str] = Field(default_factory=list)
    lessons: list[str] = Field(
        default_factory=list,
        description="Semantic memory'den çıkarılan dersler — LLM'e geri besleme")

    # Bandit'in bu tur için önerdiği aile (bütçe tahsisi)
    suggested_family: Optional[str] = None

    # Üretim modu
    generation_mode: GenerationMode = GenerationMode.new
    parent_hypothesis: Optional[HypothesisSpec] = None   # revision/combination için
    critic_feedback: Optional[Decision] = None            # "şunu düzelt"

    # Bütçe bilgisi
    experiments_remaining: Optional[int] = None

    model_config = {"extra": "forbid"}
