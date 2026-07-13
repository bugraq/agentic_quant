"""
BacktestResult — deterministik backtest çıktısı.

Metrikler çok boyutludur (sadece Sharpe değil, Doküman 2.5) ve HEPSİ işlem
maliyeti sonrasıdır. Holdout sonuçları burada BİLİNÇLİ olarak yoktur —
holdout ayrı bir servistir ve LLM'in eline asla geçmez.

Her sonuç, kendini üreten bağlamı (data/engine version, seed) taşır ki
deney tekrar üretilebilsin (Doküman 25).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class FoldMetrics(BaseModel):
    """Tek bir walk-forward fold (veya tek split) için metrikler."""

    fold_id: str
    split: str = Field(..., description="research / validation")
    sharpe: float
    annualized_return: float
    volatility: float
    max_drawdown: float
    turnover: float
    hit_rate: Optional[float] = None


class CostBreakdown(BaseModel):
    """İşlem maliyeti dökümü (asset-class'a göre değişir)."""

    commission: float = 0.0
    spread: float = 0.0
    impact: float = 0.0
    borrow_or_funding: float = 0.0


class BacktestResult(BaseModel):
    """Backtest motorunun tam çıktısı."""

    hypothesis_id: str

    per_fold_metrics: list[FoldMetrics] = Field(default_factory=list)
    cost_breakdown: CostBreakdown = Field(default_factory=CostBreakdown)

    # Net günlük getiri serisi — istatistiksel testlerin (DSR, bootstrap, FDR)
    # ham girdisi. İskelette doğrudan tutuyoruz; ölçekte artefakta taşınır.
    net_returns: list[float] = Field(default_factory=list)
    returns_artifact_id: Optional[str] = None
    exposures: dict = Field(default_factory=dict)

    # Reproducibility — bu alanlar olmadan deney yeniden üretilemez.
    data_version: str = "v0"
    engine_version: str = "v0"
    seed: int = 0

    model_config = {"extra": "forbid"}

    def aggregate_sharpe(self) -> Optional[float]:
        """Basit özet: fold Sharpe'larının ortalaması (iskelet için yeterli)."""
        if not self.per_fold_metrics:
            return None
        return sum(m.sharpe for m in self.per_fold_metrics) / len(self.per_fold_metrics)
