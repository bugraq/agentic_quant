"""
HypothesisSpec — LLM'in yapılandırılmış çıktısı (Doküman 4.4).

LLM serbest metin değil, bu şemaya uygun bir obje üretir. Şema doğrulaması
Pydantic ile yapılır; geçersiz çıktı reddedilir (Doküman 17.2).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from contracts.dsl import Expression, NamedFeature


# Motorun GERÇEKTEN uyguladığı değer kümeleri — şema bunların dışına çıkarsa
# static validator reddeder (beyan edilen != çalıştırılan olmasın, Doküman 7).
SUPPORTED_PORTFOLIO_TYPES = {"cross_sectional_long_short", "long_only"}
SUPPORTED_WEIGHTINGS = {"equal", "rank_weight"}
REBALANCE_DAYS = {"daily": 1, "weekly": 5, "monthly": 21}


class HypothesisFamily(str, Enum):
    """Multiple-testing gruplaması için hipotez ailesi (Doküman 10.1)."""

    momentum = "momentum"
    reversal = "reversal"
    volume = "volume"
    volatility = "volatility"
    liquidity = "liquidity"
    cross_sectional_interaction = "cross_sectional_interaction"
    regime_conditioned = "regime_conditioned"
    composite = "composite"


class EconomicMechanism(BaseModel):
    """Hipotezin arkasındaki ekonomik gerekçe."""

    type: str = Field(..., description="Örn: behavioral_reversal, risk_premium")
    description: str
    expected_failure_conditions: list[str] = Field(default_factory=list)


class Universe(BaseModel):
    """İşlem evreni ve likidite filtreleri. Asset-class buradan seçilir."""

    source: str = Field(..., description="Örn: sp500_point_in_time, crypto_top100")
    minimum_price: Optional[float] = None
    minimum_median_dollar_volume: Optional[float] = None


class Portfolio(BaseModel):
    """Portföy oluşturma kuralları."""

    type: str = Field(..., description="cross_sectional_long_short, long_only ...")
    long_quantile: Optional[float] = None
    short_quantile: Optional[float] = None
    weighting: str = "equal"
    sector_neutral: bool = False
    gross_exposure: float = 1.0


class Execution(BaseModel):
    """
    Yürütme zamanlaması — sızıntı kontrolünün merkezindeki blok.
    signal_time < trade_time olmalı (Static Validator zorlar).
    """

    signal_time: str = Field(..., description="Örn: close_t, bar_t")
    trade_time: str = Field(..., description="Örn: open_t_plus_1, bar_t_plus_1")
    holding_period_days: int = Field(..., gt=0)
    rebalance: str = "daily"


class Falsification(BaseModel):
    """
    Ön kayıt (pre-registration): hipotezi neyin ÖLDÜRECEĞİ, sonuçlar
    görülmeden önce taahhüt edilir. Backtest overfitting'e karşı disiplin.
    """

    minimum_oos_sharpe: float = 0.5
    maximum_turnover: Optional[float] = None
    maximum_drawdown: Optional[float] = None
    minimum_positive_walk_forward_folds: Optional[float] = None


class HypothesisSpec(BaseModel):
    """LLM'in ürettiği tam hipotez tanımı."""

    hypothesis_id: str
    title: str
    claim: str
    family: HypothesisFamily

    economic_mechanism: EconomicMechanism
    universe: Universe
    features: list[NamedFeature] = Field(default_factory=list)
    signal: Expression
    portfolio: Portfolio
    execution: Execution
    falsification: Falsification

    model_config = {"extra": "forbid"}
