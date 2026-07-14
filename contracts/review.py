"""
ReviewReport — bağımsız reviewer ajanlarının YAPILANDIRILMIŞ çıktısı (Doküman 15).

Doküman 15 en az iki bağımsız değerlendirme rolü ister ve çıktının serbest
metin DEĞİL yapılandırılmış olmasını şart koşar. Quant Critic ekonomik yargıyı
verir; bu modül Backtest Auditor (sızıntı/survivorship/maliyet/likidite) ve
Statistical Reviewer (çoklu-test/CI/fold/DSR) rollerinin ortak şemasıdır.

Bu iki reviewer DETERMİNİSTİKtir (LLM değil): sızıntı ve istatistik denetimi
nesnel olduğundan deterministik kontrol LLM'den hem daha güvenilir hem projenin
'doğrulama deterministik sistemin işidir' ilkesine (Doküman 27) uygundur.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CheckStatus(str, Enum):
    ok = "ok"       # kontrol geçti
    warn = "warn"   # dikkat: zayıflık/varsayım
    fail = "fail"   # ciddi sorun


class ReviewCheck(BaseModel):
    name: str
    status: CheckStatus
    detail: str


class ReviewReport(BaseModel):
    reviewer: str                       # "Backtest Auditor" / "Statistical Reviewer"
    verdict: CheckStatus                # kontrollerin en kötüsü (özet)
    checks: list[ReviewCheck] = Field(default_factory=list)

    @classmethod
    def from_checks(cls, reviewer: str, checks: list[ReviewCheck]) -> "ReviewReport":
        order = {CheckStatus.ok: 0, CheckStatus.warn: 1, CheckStatus.fail: 2}
        verdict = max((c.status for c in checks), key=lambda s: order[s],
                      default=CheckStatus.ok)
        return cls(reviewer=reviewer, verdict=verdict, checks=checks)
