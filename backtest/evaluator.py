"""
Evaluator — StrategyGraph'ı MarketData üstünde çalıştırıp sinyal panelini üretir.

Graph düğümleri topolojik sırada (çocuklar ebeveynden önce) olduğundan
düğümleri sırayla değerlendirip sonuçları node_id'ye göre saklarız.
Her düğüm ya DataFrame (tarih×varlık) ya da skaler üretir.

Operatör anlamları operator kaydıyla tutarlıdır; hepsi geçmişe/aynı-ana
bakar (shift, rolling), böylece hesaplama tarafında da sızıntı olmaz.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from contracts.strategy_graph import GraphNode, StrategyGraph
from data.synthetic import MarketData

Value = Union[pd.DataFrame, float]


def _demean_xs(df: pd.DataFrame) -> pd.DataFrame:
    """Kesitsel (satır bazında, varlıklar arası) demean."""
    return df.sub(df.mean(axis=1), axis=0)


def sector_groups(columns, sectors: "dict | None") -> "dict[str, list]":
    """Kolonları (ticker) sektöre göre grupla; bilinmeyenler tek grupta."""
    groups: dict[str, list] = {}
    for c in columns:
        groups.setdefault((sectors or {}).get(c, "_UNKNOWN"), []).append(c)
    return groups


def demean_by_sector(df: pd.DataFrame, sectors: "dict | None") -> pd.DataFrame:
    """Her sektör grubu İÇİNDE kesitsel demean (sektör-nötr).

    Sektör haritası yoksa piyasa-nötre (tek grup) düşer — dürüstçe belgelendi.
    """
    if not sectors:
        return _demean_xs(df)
    out = df.copy()
    for _sec, cols in sector_groups(df.columns, sectors).items():
        sub = df[cols]
        out[cols] = sub.sub(sub.mean(axis=1), axis=0)
    return out


def _eval_node(node: GraphNode, vals: dict[str, Value], data: MarketData) -> Value:
    op = node.op
    p = node.params
    ins = [vals[i] for i in node.input_ids]

    if op == "field":
        return data.get(p["field"])
    if op == "const":
        return float(p["value"])

    w = int(p.get("window", 0) or 0)
    x = ins[0] if ins else None

    # --- zaman serisi (geçmişe bakar) ---
    if op == "lag":
        return x.shift(w)
    if op == "delta":
        return x - x.shift(w)
    if op == "return":
        return x / x.shift(w) - 1.0
    if op == "rolling_mean":
        return x.rolling(w).mean()
    if op == "rolling_std":
        return x.rolling(w).std()
    if op == "rolling_min":
        return x.rolling(w).min()
    if op == "rolling_max":
        return x.rolling(w).max()
    if op == "rolling_rank":
        rmin, rmax = x.rolling(w).min(), x.rolling(w).max()
        return (x - rmin) / (rmax - rmin)
    if op == "ewma":
        return x.ewm(span=w).mean()
    if op == "zscore":
        m, s = x.rolling(w).mean(), x.rolling(w).std()
        return (x - m) / s
    if op == "volatility":
        return x.pct_change(fill_method=None).rolling(w).std()
    if op == "correlation":
        return ins[0].rolling(w).corr(ins[1])
    if op == "residual_return":
        ret = x / x.shift(w) - 1.0
        return _demean_xs(ret)   # piyasa (kesit ortalaması) çıkarılmış artık getiri

    # --- kesitsel (aynı an, varlıklar arası) ---
    if op == "cross_sectional_rank":
        return x.rank(axis=1, pct=True) - 0.5
    if op == "quantile":
        return x.rank(axis=1, pct=True)
    if op == "winsorize":
        lo = x.quantile(0.05, axis=1); hi = x.quantile(0.95, axis=1)
        return x.clip(lower=lo, upper=hi, axis=0)
    if op == "normalize":
        return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1) + 1e-12, axis=0)
    if op in ("demean", "neutralize_market"):
        return _demean_xs(x)   # piyasa-nötr (kesit ortalaması çıkar)
    if op == "neutralize_sector":
        return demean_by_sector(x, data.sectors)   # gerçek sektör-nötr (harita varsa)

    # --- aritmetik (elementwise) ---
    if op == "negate":
        return -x
    if op in ("multiply",):
        return ins[0] * ins[1]
    if op in ("divide", "ratio"):
        return ins[0] / ins[1]
    if op == "add":
        return ins[0] + ins[1]
    if op == "subtract":
        return ins[0] - ins[1]

    # --- mantıksal ---
    if op == "greater_than":
        return (ins[0] > ins[1]).astype(float)
    if op == "less_than":
        return (ins[0] < ins[1]).astype(float)
    if op == "and":
        return ((ins[0] != 0) & (ins[1] != 0)).astype(float)
    if op == "or":
        return ((ins[0] != 0) | (ins[1] != 0)).astype(float)
    if op == "not":
        return (ins[0] == 0).astype(float)
    if op == "conditional":
        cond, a, b = ins
        # a veya b skaler olabilir; sağlam koşullu seçim (a if cond else b)
        if isinstance(cond, pd.DataFrame):
            mask = cond != 0
            if isinstance(a, pd.DataFrame):
                return a.where(mask, b)
            if isinstance(b, pd.DataFrame):
                return b.where(~mask, a)
            return mask.astype(float) * a + (~mask).astype(float) * b
        return a if cond != 0 else b

    raise NotImplementedError(f"Evaluator'da uygulanmamış operatör: {op}")


def evaluate_signal(graph: StrategyGraph, data: MarketData,
                    liveness_out: "list[tuple[str, float]] | None" = None) -> pd.DataFrame:
    """Graph'ı çalıştır, sinyal düğümünün panelini döndür.

    liveness_out verilirse her conditional için (node_id, tetiklenme_oranı)
    eklenir — KOŞUL-CANLILIK teşhisi. Koşul neredeyse hiç (veya hep)
    tetikleniyorsa 'rejim koşullaması' etiketi sahtedir: strateji fiilen tek
    dalıdır (gerçek koşuda görüldü: rolling_std(close,10) < 0.02 fiyat/getiri
    birim hatası yüzünden %0.25 tetikleniyordu; 'low-vol momentum' diye kabul
    edilen şey saf reversal çıktı).
    """
    vals: dict[str, Value] = {}
    for node in graph.nodes:
        vals[node.node_id] = _eval_node(node, vals, data)
    signal = vals[graph.signal_node_id]
    if not isinstance(signal, pd.DataFrame):
        raise ValueError("Sinyal skaler çıktı verdi; kesitsel bir panel bekleniyordu.")
    signal = signal.replace([np.inf, -np.inf], np.nan)

    if liveness_out is not None:
        valid = signal.notna()
        for node in graph.nodes:
            if node.op != "conditional" or not node.input_ids:
                continue
            cond = vals[node.input_ids[0]]
            if isinstance(cond, pd.DataFrame):
                # Yalnızca sinyalin tanımlı olduğu hücrelerde ölç (warmup hariç)
                frac = float(cond.where(valid).stack().mean()) if valid.any().any() else 0.0
            else:
                frac = 1.0 if cond else 0.0   # derleme-sabiti koşul: kesin ölü
            liveness_out.append((node.node_id, frac))
    return signal
