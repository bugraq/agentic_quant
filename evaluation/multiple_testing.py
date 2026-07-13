"""
Multiple Testing Controller (Doküman 10) — yapılan TÜM denemeleri hesaba katar.

Bir strateji tek başına iyi Sharpe gösterse bile, 100 deneme içinden seçildiyse
bu tesadüf olabilir. Bu modül tüm backtest'lenen deneyleri alıp:
  - ham p-value (PSR tabanlı, H0: SR<=0)
  - Deflated Sharpe Ratio (deneme sayısı düzeltmesi)
  - Benjamini-Hochberg FDR (hangileri hayatta kalıyor)
  - bootstrap Sharpe güven aralığı
hesaplar. "Kabul edildi" != "istatistiksel olarak geçerli"; asıl süzgeç budur.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.statistics import (
    benjamini_hochberg,
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_moments,
)

TRADING_DAYS = 252


@dataclass
class ReportRow:
    hypothesis_id: str
    title: str
    decision: str
    ann_sharpe: float
    raw_p: float
    dsr: float
    ci_low: float
    ci_high: float
    survives_fdr: bool


def build_report(records: list[tuple], fdr_alpha: float = 0.10) -> list[ReportRow]:
    """records: (hid, title, decision, sharpe, returns_list). n_trials = len(records)."""
    if not records:
        return []

    moments = [sharpe_moments(r[4]) for r in records]
    n_trials = len(records)
    # Deneme SR'lerinin varyansı (seçilim düzeltmesi için)
    var_sr = float(np.var([m.sr for m in moments], ddof=1)) if n_trials > 1 else 0.0

    raw_p = [1.0 - probabilistic_sharpe_ratio(m, 0.0) for m in moments]
    survive = benjamini_hochberg(raw_p, alpha=fdr_alpha)

    rows: list[ReportRow] = []
    for i, (hid, title, decision, _sharpe, returns) in enumerate(records):
        m = moments[i]
        dsr = deflated_sharpe_ratio(m, n_trials, var_sr)
        lo, hi = bootstrap_sharpe_ci(returns, n_boot=500, seed=i)
        rows.append(ReportRow(
            hypothesis_id=hid, title=title, decision=decision,
            ann_sharpe=m.sr * (TRADING_DAYS ** 0.5),
            raw_p=raw_p[i], dsr=dsr, ci_low=lo, ci_high=hi,
            survives_fdr=survive[i]))
    rows.sort(key=lambda r: r.dsr, reverse=True)
    return rows


def print_report(rows: list[ReportRow], n_trials: int) -> None:
    print(f"\n=== MULTIPLE TESTING RAPORU (toplam deneme: {n_trials}) ===")
    if not rows:
        print("  (backtest'lenen deney yok)")
        return
    print(f"{'hipotez':12s} {'Sharpe':>7s} {'ham p':>7s} {'DSR':>6s} "
          f"{'Sharpe %95 CI':>18s}  FDR")
    for r in rows:
        ci = f"[{r.ci_low:.2f}, {r.ci_high:.2f}]"
        flag = "GEÇTİ" if r.survives_fdr else "-"
        dsr_flag = "*" if r.dsr > 0.95 else " "
        print(f"{r.hypothesis_id:12s} {r.ann_sharpe:7.2f} {r.raw_p:7.3f} "
              f"{r.dsr:5.2f}{dsr_flag} {ci:>18s}  {flag}")
    print("  (* = DSR>0.95: deneme sayısı düzeltildikten sonra bile anlamlı)")
