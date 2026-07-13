"""
StrategyGraph — HypothesisSpec'in derlenmiş, deterministik hali.

Compiler, DSL ifadelerini hesaplanabilir bir operatör DAG'ına çevirir.
Static Validator bu graph üzerinde `max_info_time` yayarak sızıntı kontrolü
yapar. Aynı HypothesisSpec her zaman aynı graph'a derlenir — tekrar
üretilebilirliğin temeli budur.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """
    Derlenmiş graph'ta tek bir operatör düğümü.

    max_info_time: bu düğümün değerinin en erken hangi (göreli) an'da
    bilinebileceği. Sızıntı kontrolünün taşıyıcısı. Referans an 't' iken
    örn. "close_t", "close_t+5", "open_t+1" gibi sembolik bir zaman etiketi.
    """

    node_id: str
    op: str
    params: dict = Field(default_factory=dict)
    input_ids: list[str] = Field(default_factory=list)

    output_type: str = Field(..., description="series / cross_section / scalar / boolean")
    time_direction: str = Field(..., description="backward / pointwise / forward")
    min_lookback: int = 0
    max_info_time: Optional[str] = Field(
        None, description="En geç bilgi anı, sembolik etiket (örn. close_t)"
    )


class ComplexityMetrics(BaseModel):
    """Karmaşıklık ölçüleri (Doküman 6.4) — overfitting cezası için."""

    node_count: int = 0
    depth: int = 0
    free_parameters: int = 0
    conditions: int = 0
    data_sources: int = 0


class StrategyGraph(BaseModel):
    """Derlenmiş strateji — backtest motorunun girdisi."""

    hypothesis_id: str
    nodes: list[GraphNode] = Field(default_factory=list)

    feature_node_ids: dict[str, str] = Field(
        default_factory=dict, description="feature adı -> node_id"
    )
    signal_node_id: str

    required_data_fields: list[str] = Field(default_factory=list)
    complexity: ComplexityMetrics = Field(default_factory=ComplexityMetrics)

    model_config = {"extra": "forbid"}
