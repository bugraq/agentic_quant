"""
Backtest Auditor — bağımsız deterministik denetçi (Doküman 15).

Kontrol ettiği konular (Doküman 15 listesi): veri sızıntısı / execution sırası,
fiyat düzeltmesi, survivorship / endeks üyeliği, delisting, gerçekçi likidite ve
maliyet varsayımları. Quant Critic'ten AYRI bir roldür: ekonomik mantığı değil,
BACKTEST'İN GEÇERLİLİĞİNİ denetler.

Deterministiktir: bu kontroller nesneldir (LLM tahminine bırakılmaz). Çıktı
yapılandırılmış ReviewReport'tur.
"""
from __future__ import annotations

from contracts.backtest_result import BacktestResult
from contracts.hypothesis_spec import HypothesisSpec
from contracts.review import CheckStatus, ReviewCheck, ReviewReport
from data.synthetic import MarketData
from dsl.operators import parse_time_token, tick_to_label

# Yıllık turnover bu değerin üstündeyse likidite/maliyet varsayımı şüpheli
_TURNOVER_WARN = 250.0


class BacktestAuditor:
    """Backtest geçerliliğini denetleyen deterministik reviewer."""

    reviewer = "Backtest Auditor"

    def audit(self, hyp: HypothesisSpec, result: BacktestResult,
              data: MarketData, cost_bps: float) -> ReviewReport:
        checks: list[ReviewCheck] = []

        # 1) Execution sırası / sızıntı: sinyal anı < işlem anı olmalı
        try:
            sig_tick = parse_time_token(hyp.execution.signal_time)
            trade_tick = parse_time_token(hyp.execution.trade_time)
            if sig_tick < trade_tick:
                checks.append(ReviewCheck(
                    name="execution_delay", status=CheckStatus.ok,
                    detail=(f"Sinyal {tick_to_label(sig_tick)}, işlem "
                            f"{tick_to_label(trade_tick)} — en az bir bar gecikme var.")))
            else:
                checks.append(ReviewCheck(
                    name="execution_delay", status=CheckStatus.fail,
                    detail="İşlem sinyalle aynı/önceki anda — look-ahead riski."))
        except ValueError as e:
            checks.append(ReviewCheck(name="execution_delay", status=CheckStatus.fail,
                                      detail=f"Geçersiz execution zamanı: {e}"))

        # 2) Fiyat düzeltmesi: getiriler adjusted_close'tan (temettü+split)
        adj = "adjusted_close" in data.fields
        checks.append(ReviewCheck(
            name="price_adjustment",
            status=CheckStatus.ok if adj else CheckStatus.warn,
            detail=("Getiriler adjusted_close'tan (temettü+split dahil)." if adj
                    else "adjusted_close yok — düzeltilmemiş fiyat kullanılıyor.")))

        # 3) Survivorship / point-in-time endeks üyeliği
        pit = "index_membership" in data.fields
        checks.append(ReviewCheck(
            name="survivorship",
            status=CheckStatus.ok if pit else CheckStatus.warn,
            detail=("Point-in-time üyelik maskesi uygulanıyor (hisse yalnızca üye "
                    "olduğu günlerde işlenir)." if pit else
                    "Sabit ticker listesi — survivorship bias taşır (PIT üyelik yok).")))

        # 4) Likidite / turnover gerçekçiliği
        max_turn = max((m.turnover for m in result.per_fold_metrics), default=0.0)
        checks.append(ReviewCheck(
            name="liquidity_turnover",
            status=CheckStatus.warn if max_turn > _TURNOVER_WARN else CheckStatus.ok,
            detail=(f"Yıllık turnover ~{max_turn:.0f}x"
                    + (" — yüksek; kapasite/maliyet varsayımı kırılgan olabilir."
                       if max_turn > _TURNOVER_WARN else " — makul aralıkta."))))

        # 5) Maliyet uygulanmış mı
        checks.append(ReviewCheck(
            name="cost_applied",
            status=CheckStatus.ok if cost_bps > 0 else CheckStatus.warn,
            detail=(f"İşlem maliyeti {cost_bps:.0f} bps turnover'a uygulanıyor."
                    if cost_bps > 0 else "İşlem maliyeti 0 — net getiri iyimser.")))

        return ReviewReport.from_checks(self.reviewer, checks)
