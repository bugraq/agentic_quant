"""
Holdout Servisi (Doküman 10.3) — araştırmadan TAMAMEN ayrı son değerlendirme.

İlkeler:
  - LLM'den ve araştırma orchestrator'ından bağımsızdır (bu modül LLM import ETMEZ).
  - Holdout tarihlerini/serisini dışarı açıklamaz; yalnızca özet metrik döndürür.
  - Önceden belirlenmiş sayıda aday kabul eder (maximum_candidates).
  - Her aday için sonucu BİR KEZ üretir (one-shot); tekrar değerlendirme yasak.
  - Holdout sonucuyla strateji revizyonuna izin vermez (çağıran taraf uygular).
  - Bütün erişimleri audit log'a kaydeder (ayrı veritabanı).

Bu servis deterministik altyapıdır (derleme + backtest); yaratıcı hiçbir
bileşen içermez.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from contracts.hypothesis_spec import HypothesisSpec
from data.synthetic import MarketData
from dsl import compile_hypothesis
from backtest import run_backtest

_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS holdout_access (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT UNIQUE NOT NULL,     -- one-shot: tekil kısıt
    sharpe        REAL,
    passed        INTEGER,
    accessed_at   TEXT
);
"""


class HoldoutError(Exception):
    """One-shot ihlali veya aday kotası aşımı."""


@dataclass
class HoldoutResult:
    hypothesis_id: str
    sharpe: float
    passed: bool


class HoldoutService:
    def __init__(self, holdout_data: MarketData, audit_path: str = "holdout_audit.sqlite",
                 max_candidates: int = 20, min_sharpe: float = 0.5,
                 cost_bps: float = 5.0) -> None:
        self._data = holdout_data          # KİLİTLİ — dışarı verilmez
        self._max = max_candidates
        self._min_sharpe = min_sharpe
        self._cost_bps = cost_bps
        self._audit = sqlite3.connect(audit_path)
        self._audit.execute(_AUDIT_SCHEMA)
        self._audit.commit()

    def _count(self) -> int:
        return self._audit.execute("SELECT COUNT(*) FROM holdout_access").fetchone()[0]

    def evaluate(self, hyp: HypothesisSpec) -> HoldoutResult:
        """Bir adayı kilitli dönemde BİR KEZ değerlendir. Yalnızca özet döner."""
        # One-shot kontrolü
        seen = self._audit.execute(
            "SELECT 1 FROM holdout_access WHERE hypothesis_id=?",
            (hyp.hypothesis_id,)).fetchone()
        if seen:
            raise HoldoutError(
                f"{hyp.hypothesis_id} zaten holdout'ta değerlendirildi (one-shot).")
        # Kota kontrolü
        if self._count() >= self._max:
            raise HoldoutError(f"Holdout aday kotası doldu ({self._max}).")

        graph = compile_hypothesis(hyp)
        result = run_backtest(graph, hyp, self._data, cost_bps=self._cost_bps)
        sharpe = result.aggregate_sharpe() or 0.0
        passed = sharpe >= self._min_sharpe

        self._audit.execute(
            "INSERT INTO holdout_access (hypothesis_id, sharpe, passed, accessed_at) "
            "VALUES (?,?,?,?)",
            (hyp.hypothesis_id, sharpe, int(passed),
             datetime.now(timezone.utc).isoformat()))
        self._audit.commit()
        # NOT: holdout tarihleri/serisi ASLA döndürülmez; sadece özet.
        return HoldoutResult(hyp.hypothesis_id, sharpe, passed)

    def audit_log(self) -> list[tuple]:
        return self._audit.execute(
            "SELECT hypothesis_id, sharpe, passed, accessed_at "
            "FROM holdout_access ORDER BY id").fetchall()

    def close(self) -> None:
        self._audit.close()
