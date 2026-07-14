"""
Statistical Reviewer — bağımsız deterministik istatistik denetçisi (Doküman 15).

Kontrol ettiği konular: çoklu-test (FDR), güven aralıkları, fold kararlılığı,
Deflated Sharpe (seçim yanlılığı düzeltmesi). "Kabul edildi" ile "istatistiksel
olarak doğrulandı"yı AYIRIR — bir strateji hard gate'i geçse bile deneme sayısı
düzeltmesinden sonra anlamsız olabilir.

Rapor zamanında (kampanya sonu) çalışır: çoklu-test satırı (DSR/FDR/CI) tüm
backtest'lenen deneyler üzerinden hesaplandığı için per-hipotez değil kampanya
geneli bir görünüm gerektirir. Çıktı yapılandırılmış ReviewReport'tur.
"""
from __future__ import annotations

from contracts.backtest_result import BacktestResult
from contracts.review import CheckStatus, ReviewCheck, ReviewReport

_MIN_POSITIVE_FOLDS = 0.5


class StatisticalReviewer:
    """Çoklu-test/CI/fold/DSR'ı yapılandırılmış bir istatistik yargısına bağlar."""

    reviewer = "Statistical Reviewer"

    def review(self, report_row, result: "BacktestResult | None" = None) -> ReviewReport:
        """report_row: evaluation.multiple_testing.ReportRow; result: fold kararlılığı için."""
        checks: list[ReviewCheck] = []

        # 1) FDR — deneme sayısı düzeltmesinden sonra hayatta mı
        checks.append(ReviewCheck(
            name="false_discovery_rate",
            status=CheckStatus.ok if report_row.survives_fdr else CheckStatus.warn,
            detail=(f"ham p={report_row.raw_p:.3f}; Benjamini-Hochberg FDR'ı "
                    + ("GEÇTİ." if report_row.survives_fdr
                       else "GEÇMEDİ — çoklu-test sonrası anlamlı değil."))))

        # 2) Deflated Sharpe (seçim yanlılığı düzeltilmiş)
        checks.append(ReviewCheck(
            name="deflated_sharpe",
            status=CheckStatus.ok if report_row.dsr > 0.95 else CheckStatus.warn,
            detail=(f"DSR={report_row.dsr:.2f} "
                    + (">0.95 — deneme sayısı düzeltilse bile anlamlı."
                       if report_row.dsr > 0.95 else
                       "≤0.95 — seçim yanlılığı düzeltmesinden sonra zayıf."))))

        # 3) Güven aralığı sıfırı içeriyor mu
        excludes_zero = report_row.ci_low > 0
        checks.append(ReviewCheck(
            name="confidence_interval",
            status=CheckStatus.ok if excludes_zero else CheckStatus.warn,
            detail=(f"Sharpe %95 CI [{report_row.ci_low:.2f}, {report_row.ci_high:.2f}] "
                    + ("sıfırı DIŞLIYOR — yön güvenilir." if excludes_zero
                       else "sıfırı İÇERİYOR — işaret bile kesin değil."))))

        # 4) Fold kararlılığı (walk-forward tutarlılık)
        if result is not None and result.per_fold_metrics:
            pos = sum(1 for m in result.per_fold_metrics if m.sharpe > 0)
            frac = pos / len(result.per_fold_metrics)
            checks.append(ReviewCheck(
                name="fold_stability",
                status=CheckStatus.ok if frac >= _MIN_POSITIVE_FOLDS else CheckStatus.warn,
                detail=(f"{pos}/{len(result.per_fold_metrics)} fold pozitif "
                        f"(%{frac*100:.0f}) — "
                        + ("dönemler arası tutarlı." if frac >= _MIN_POSITIVE_FOLDS
                           else "dönem bağımlı, kırılgan."))))

        return ReviewReport.from_checks(self.reviewer, checks)
