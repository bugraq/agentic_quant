"""
Operatör kaydı — DSL'in tek doğruluk kaynağı.

Her operatör; tip bilgisi, zaman yönü, arite ve parametre kısıtlarını taşır.
Compiler ve static validator kararlarını BU metadata'dan üretir. Yeni bir
operatör eklemek = buraya bir OperatorSpec eklemek.

ZAMAN MODELİ (sızıntı kontrolünün temeli)
-----------------------------------------
Zamanı yarım-bar "tick"leriyle sayıyoruz; referans bar = t:
    open_t = 0,  close_t = 1,  open_{t+1} = 2,  close_{t+1} = 3, ...
    open_{t-1} = -2, close_{t-1} = -1
Bir düğümün `info_tick`i, değerinin en erken bilinebileceği tick'tir.
Sızıntı kontrolü tek eşitsizliktir:  signal.info_tick < execution.info_tick
Bu model asset-agnostiktir: "bar" günlük de olabilir 4-saatlik kripto da;
sadece veri adaptörü barın ne olduğunu söyler.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ---- Tipler --------------------------------------------------------------
SERIES = "series"            # tek varlık, zaman serisi
CROSS = "cross_section"      # bir andaki tüm varlıklar
SCALAR = "scalar"
BOOLEAN = "boolean"
NUMERIC = "numeric"          # polimorfik: series | cross_section | scalar

# ---- Zaman yönü ----------------------------------------------------------
BACKWARD = "backward"    # sadece geçmişe bakar (güvenli)
POINTWISE = "pointwise"  # aynı an
FORWARD = "forward"      # GELECEĞE bakar — DSL'de YASAK (sızıntı)


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    output_type: str
    time_direction: str
    min_arity: int
    max_arity: int
    needs_window: bool = False
    window_min: int = 1
    window_max: int = 500
    is_lag: bool = False           # info_tick'i k bar GERİYE kaydırır
    accepts: tuple[str, ...] = field(default_factory=tuple)  # kabul edilen input tipleri


# ---- Ham veri alanlarının availability tick'i ----------------------------
# open t günü açılışında; kapanış/hacim/vs t günü kapanışında bilinir.
FIELD_BASE_TICK: dict[str, int] = {
    "open": 0,
    "close": 1,
    "high": 1,
    "low": 1,
    "adjusted_close": 1,
    "volume": 1,
    "dollar_volume": 1,
    "market_cap": 1,
    "index_membership": 1,
}
DATA_FIELDS = set(FIELD_BASE_TICK.keys())

# GİRDİSİZ (0-arity) türetilmiş-alan operatörlerinin taban tick'i. Compiler
# child_tick'i olmayan düğüme normalde 0 (open_t) verir; bu operatörler high/low/
# close okuduğu için değerleri close_t'de (tick 1) bilinir — yoksa sızıntı-validatör
# yanlışlıkla 'open_t'de biliniyor' der ve close_t işlemine izin verirdi.
NO_INPUT_BASE_TICK: dict[str, int] = {
    "intraday_range": FIELD_BASE_TICK["close"],
    "close_location": FIELD_BASE_TICK["close"],
}


def _reg(*specs: OperatorSpec) -> dict[str, OperatorSpec]:
    return {s.name: s for s in specs}


# ---- Operatör kaydı (Doküman 5) -----------------------------------------
REGISTRY: dict[str, OperatorSpec] = _reg(
    # Zaman serisi (geçmişe bakar)
    OperatorSpec("lag", SERIES, BACKWARD, 1, 1, needs_window=True, is_lag=True, accepts=(SERIES,)),
    OperatorSpec("delta", SERIES, BACKWARD, 1, 1, needs_window=True, accepts=(SERIES,)),
    OperatorSpec("return", SERIES, BACKWARD, 1, 1, needs_window=True, accepts=(SERIES,)),
    OperatorSpec("rolling_mean", SERIES, BACKWARD, 1, 1, needs_window=True, accepts=(SERIES,)),
    OperatorSpec("rolling_std", SERIES, BACKWARD, 1, 1, needs_window=True, accepts=(SERIES,)),
    OperatorSpec("rolling_min", SERIES, BACKWARD, 1, 1, needs_window=True, accepts=(SERIES,)),
    OperatorSpec("rolling_max", SERIES, BACKWARD, 1, 1, needs_window=True, accepts=(SERIES,)),
    OperatorSpec("rolling_rank", SERIES, BACKWARD, 1, 1, needs_window=True, accepts=(SERIES,)),
    OperatorSpec("ewma", SERIES, BACKWARD, 1, 1, needs_window=True, accepts=(SERIES,)),
    OperatorSpec("zscore", SERIES, BACKWARD, 1, 1, needs_window=True, accepts=(SERIES,)),
    OperatorSpec("volatility", SERIES, BACKWARD, 1, 1, needs_window=True, accepts=(SERIES,)),
    OperatorSpec("correlation", SERIES, BACKWARD, 2, 2, needs_window=True, accepts=(SERIES,)),
    # Gün-içi (high/low) türetilmiş — GİRDİ ALMAZ, high/low/close'u kendisi okur.
    # window verilirse rolling ortalama (ATR-benzeri düzleştirme). Aynı-bar bilgisi
    # close_t'de bilinir (NO_INPUT_BASE_TICK), gün-içi aralık YENİ bilgi kaynağıdır.
    OperatorSpec("intraday_range", SERIES, BACKWARD, 0, 0, needs_window=False, accepts=()),
    OperatorSpec("close_location", SERIES, BACKWARD, 0, 0, needs_window=False, accepts=()),
    OperatorSpec("residual_return", SERIES, BACKWARD, 0, 1, needs_window=True, accepts=(SERIES,)),
    # Kesitsel (aynı an, tüm varlıklar)
    OperatorSpec("cross_sectional_rank", CROSS, POINTWISE, 1, 1, accepts=(SERIES, CROSS)),
    OperatorSpec("winsorize", CROSS, POINTWISE, 1, 1, accepts=(SERIES, CROSS)),
    OperatorSpec("normalize", CROSS, POINTWISE, 1, 1, accepts=(SERIES, CROSS)),
    OperatorSpec("demean", CROSS, POINTWISE, 1, 1, accepts=(SERIES, CROSS)),
    OperatorSpec("quantile", CROSS, POINTWISE, 1, 1, accepts=(SERIES, CROSS)),
    OperatorSpec("neutralize_market", CROSS, POINTWISE, 1, 1, accepts=(SERIES, CROSS)),
    OperatorSpec("neutralize_sector", CROSS, POINTWISE, 1, 1, accepts=(SERIES, CROSS)),
    # Aritmetik (polimorfik, aynı an)
    OperatorSpec("multiply", NUMERIC, POINTWISE, 2, 2, accepts=(SERIES, CROSS, SCALAR)),
    OperatorSpec("divide", NUMERIC, POINTWISE, 2, 2, accepts=(SERIES, CROSS, SCALAR)),
    OperatorSpec("ratio", NUMERIC, POINTWISE, 2, 2, accepts=(SERIES, CROSS, SCALAR)),
    OperatorSpec("add", NUMERIC, POINTWISE, 2, 2, accepts=(SERIES, CROSS, SCALAR)),
    OperatorSpec("subtract", NUMERIC, POINTWISE, 2, 2, accepts=(SERIES, CROSS, SCALAR)),
    OperatorSpec("negate", NUMERIC, POINTWISE, 1, 1, accepts=(SERIES, CROSS, SCALAR)),
    # Mantıksal
    OperatorSpec("greater_than", BOOLEAN, POINTWISE, 2, 2, accepts=(SERIES, CROSS, SCALAR)),
    OperatorSpec("less_than", BOOLEAN, POINTWISE, 2, 2, accepts=(SERIES, CROSS, SCALAR)),
    OperatorSpec("and", BOOLEAN, POINTWISE, 2, 2, accepts=(BOOLEAN,)),
    OperatorSpec("or", BOOLEAN, POINTWISE, 2, 2, accepts=(BOOLEAN,)),
    OperatorSpec("not", BOOLEAN, POINTWISE, 1, 1, accepts=(BOOLEAN,)),
    OperatorSpec("conditional", NUMERIC, POINTWISE, 3, 3, accepts=(SERIES, CROSS, SCALAR, BOOLEAN)),
)


def get_operator(name: str) -> Optional[OperatorSpec]:
    return REGISTRY.get(name)


# ---- Zaman etiketi <-> tick dönüşümleri ---------------------------------
_TIME_RE = re.compile(r"^(open|close|bar)_t(?:_(plus|minus)_(\d+))?$")


def parse_time_token(token: str) -> int:
    """
    'close_t' -> 1, 'open_t_plus_1' -> 2, 'close_t_plus_2' -> 5, 'open_t' -> 0.
    'bar_t' kripto/asset-agnostik: close gibi davranır (bar kapanışı).
    """
    m = _TIME_RE.match(token.strip())
    if not m:
        raise ValueError(f"Anlaşılamayan zaman etiketi: {token!r}")
    phase = {"open": 0, "close": 1, "bar": 1}[m.group(1)]
    k = int(m.group(3)) if m.group(3) else 0
    if m.group(2) == "minus":
        k = -k
    return phase + 2 * k


def tick_to_label(tick: int) -> str:
    """Tick -> okunabilir sembolik etiket (örn. 5 -> 'close_t+2')."""
    phase = tick % 2
    bar = tick // 2
    name = "open" if phase == 0 else "close"
    if bar == 0:
        return f"{name}_t"
    return f"{name}_t{bar:+d}"
