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
    n_copies: int = 1          # birebir aynı getiri serisini üreten deneme sayısı
    n_param_variants: int = 0  # bu stratejinin optimizer'ca aranan pencere varyantı sayısı


def dedup_records(records: list[tuple]) -> tuple[list[tuple], dict]:
    """Birebir özdeş getiri serilerini tekilleştir (ÖLÜ PARAMETRE teşhisi).

    Optimizer bir pencereyi değiştirip getiri hiç değişmiyorsa bu ayrı bir
    deneme DEĞİL, aynı stratejidir — ve o pencere ölü parametredir (gerçek
    koşuda görüldü: 6'şarlı özdeş Sharpe grupları). Aynı seriyi N kez saymak
    hem n_trials'ı şişirir hem raporu boğar.
    Döndürür: (tekil kayıtlar, temsilci_hid -> kopya sayısı).
    """
    seen: dict = {}
    distinct: list[tuple] = []
    copies: dict = {}
    for rec in records:
        key = tuple(round(float(x), 12) for x in rec[4])
        if key in seen:
            copies[seen[key]] += 1
        else:
            seen[key] = rec[0]
            copies[rec[0]] = 1
            distinct.append(rec)
    return distinct, copies


def _normalize(records: list[tuple]) -> list[tuple]:
    """Kayıtları 7-tuple'a normalize et (eski 5-tuple çağrılarla geriye uyum).

    (hid, title, decision, sharpe, returns, relation_type, parent_id)
    """
    out = []
    for r in records:
        rel = r[5] if len(r) > 5 else None
        parent = r[6] if len(r) > 6 else None
        out.append((r[0], r[1], r[2], r[3], r[4], rel, parent))
    return out


def build_report(records: list[tuple], fdr_alpha: float = 0.10) -> list[ReportRow]:
    """records: (hid, title, decision, sharpe, returns_list[, relation_type, parent_id]).

    PARAMETRE-ARAMA VARYANTLARI GRUPLANIR (Doküman 10.1): optimizer'ın bir
    stratejinin pencerelerini tarayan denemeleri BAĞIMSIZ keşif DEĞİLDİR (yüksek
    korelasyonlu, aynı strateji). Bunları N ayrı deneme saymak DSR/FDR paydasını
    haksız şişirir. Bu yüzden çoklu-test yalnızca birincil (distinct) stratejiler
    üzerinden yürütülür; varyantlar parent'a 'aranan varyant sayısı' olarak katlanır.
    Kayıtta yine dururlar (dürüstlük, Doküman 2.3) — sadece istatistik paydası değil.
    """
    if not records:
        return []
    norm = _normalize(records)
    # Parametre varyantlarını ayır ve parent başına say.
    variant_counts: dict = {}
    primary = []
    for rec in norm:
        if rec[5] == "parameter_variant":
            variant_counts[rec[6]] = variant_counts.get(rec[6], 0) + 1
        else:
            primary.append(rec)
    if not primary:   # her şey varyantsa (beklenmez), varyantlara düş
        primary = norm

    records, copies = dedup_records(primary)

    moments = [sharpe_moments(r[4]) for r in records]
    n_trials = len(records)
    # Deneme SR'lerinin varyansı (seçilim düzeltmesi için)
    var_sr = float(np.var([m.sr for m in moments], ddof=1)) if n_trials > 1 else 0.0

    raw_p = [1.0 - probabilistic_sharpe_ratio(m, 0.0) for m in moments]
    survive = benjamini_hochberg(raw_p, alpha=fdr_alpha)

    rows: list[ReportRow] = []
    for i, rec in enumerate(records):
        hid, title, decision, _sharpe, returns = rec[0], rec[1], rec[2], rec[3], rec[4]
        m = moments[i]
        dsr = deflated_sharpe_ratio(m, n_trials, var_sr)
        lo, hi = bootstrap_sharpe_ci(returns, n_boot=500, seed=i)
        rows.append(ReportRow(
            hypothesis_id=hid, title=title, decision=decision,
            ann_sharpe=m.sr * (TRADING_DAYS ** 0.5),
            raw_p=raw_p[i], dsr=dsr, ci_low=lo, ci_high=hi,
            survives_fdr=survive[i], n_copies=copies.get(hid, 1),
            n_param_variants=variant_counts.get(hid, 0)))
    rows.sort(key=lambda r: r.dsr, reverse=True)
    return rows


def print_report(rows: list[ReportRow], n_trials: int) -> None:
    """n_trials: HAM backtest sayısı; tabloda birincil (distinct) stratejiler.

    Parametre varyantları paydadan çıkarıldı (parent'a katlandı, +Nvar);
    n_trials (ham) ile tekil strateji sayısı arasındaki fark bunu gösterir.
    """
    n_distinct = len(rows)
    print(f"\n=== MULTIPLE TESTING RAPORU "
          f"(ham backtest: {n_trials}, BİRİNCİL strateji: {n_distinct}; "
          f"parametre varyantları paydaya girmez) ===")
    if not rows:
        print("  (backtest'lenen deney yok)")
        return
    print(f"{'hipotez':16s} {'Sharpe':>7s} {'ham p':>7s} {'DSR':>6s} "
          f"{'Sharpe %95 CI':>18s} {'var':>4s}  FDR")
    dead_param = False
    for r in rows:
        ci = f"[{r.ci_low:.2f}, {r.ci_high:.2f}]"
        flag = "GEÇTİ" if r.survives_fdr else "-"
        dsr_flag = "*" if r.dsr > 0.95 else " "
        copy_tag = f" x{r.n_copies}" if r.n_copies > 1 else ""
        var_tag = f"+{r.n_param_variants}" if r.n_param_variants else ""
        if r.n_copies > 1:
            dead_param = True
        print(f"{r.hypothesis_id + copy_tag:16s} {r.ann_sharpe:7.2f} {r.raw_p:7.3f} "
              f"{r.dsr:5.2f}{dsr_flag} {ci:>18s} {var_tag:>4s}  {flag}")
    print("  (* = DSR>0.95: deneme sayısı düzeltildikten sonra bile anlamlı)")
    print("  (+N = optimizer bu stratejinin N pencere varyantını aradı; "
          "korelasyonlu oldukları için bağımsız deneme sayılmaz — Doküman 10.1)")
    if dead_param:
        print("  (xN = N deneme BİREBİR aynı getiriyi üretti -> ÖLÜ PARAMETRE)")
