"""
Pipeline contract'ları — istasyonlar arası akan veri objeleri.

Akış:
  ResearchContext -> [LLM] -> HypothesisSpec -> [Compiler] -> StrategyGraph
  -> [Backtest] -> BacktestResult -> [Gate+Critic] -> Decision -> Memory

Bu şemalar sabit kaldığı sürece kutuların içi (LLM sağlayıcısı, backtest
motoru, veri kaynağı) serbestçe değiştirilebilir.
"""
from contracts.backtest_result import BacktestResult, CostBreakdown, FoldMetrics
from contracts.decision import Decision, DecisionType, Issue, Severity
from contracts.dsl import Expression, NamedFeature
from contracts.hypothesis_spec import (
    EconomicMechanism,
    Execution,
    Falsification,
    HypothesisFamily,
    HypothesisSpec,
    Portfolio,
    Universe,
)
from contracts.research_context import (
    ExperimentSummary,
    GenerationMode,
    ResearchContext,
)
from contracts.strategy_graph import ComplexityMetrics, GraphNode, StrategyGraph

__all__ = [
    "Expression",
    "NamedFeature",
    "HypothesisSpec",
    "HypothesisFamily",
    "EconomicMechanism",
    "Universe",
    "Portfolio",
    "Execution",
    "Falsification",
    "ResearchContext",
    "GenerationMode",
    "ExperimentSummary",
    "StrategyGraph",
    "GraphNode",
    "ComplexityMetrics",
    "BacktestResult",
    "FoldMetrics",
    "CostBreakdown",
    "Decision",
    "DecisionType",
    "Issue",
    "Severity",
]
