"""
Research Memory — episodic katman (Doküman 12.1).

Her deneyin TAM kaydı saklanır: hipotez, karar, metrikler, seed, aşama.
Başarısız deneyler DAHİL her şey kaydedilir (Doküman 2.3) — sistemin toplam
arama miktarı bilinmeden multiple testing düzeltmesi yapılamaz.

İskelet: SQLite (bağımlılık yok). Arayüz sabit kaldığı sürece ileride
PostgreSQL + pgvector'a taşınabilir.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from contracts.backtest_result import BacktestResult
from contracts.decision import Decision
from contracts.hypothesis_spec import HypothesisSpec

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id   TEXT NOT NULL,
    title           TEXT,
    family          TEXT,
    stage           TEXT,          -- hangi aşamada sonlandı
    decision        TEXT,          -- accept / reject / revise / duplicate
    decision_source TEXT,          -- gate / statistical / critic
    sharpe          REAL,
    max_drawdown    REAL,
    turnover        REAL,
    seed            INTEGER,
    issues_json     TEXT,          -- tespit edilen sorunlar
    hypothesis_json TEXT,          -- tam hipotez (reproducibility)
    returns_json    TEXT,          -- net günlük getiri serisi (istatistik için)
    created_at      TEXT
);
"""


@dataclass
class ExperimentRecord:
    hypothesis_id: str
    title: str
    family: str
    stage: str
    decision: str
    decision_source: str
    sharpe: Optional[float]
    max_drawdown: Optional[float]
    turnover: Optional[float]
    seed: Optional[int]


class MemoryStore:
    def __init__(self, path: str = "research_memory.sqlite") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(_SCHEMA)
        self.conn.commit()

    def record(self, hyp: HypothesisSpec, decision: Decision, stage: str,
               result: Optional[BacktestResult] = None) -> int:
        sharpe = max_dd = turnover = None
        seed = None
        returns_json = None
        if result is not None:
            sharpe = result.aggregate_sharpe()
            max_dd = max((m.max_drawdown for m in result.per_fold_metrics), default=None)
            turnover = max((m.turnover for m in result.per_fold_metrics), default=None)
            seed = result.seed
            returns_json = json.dumps(result.net_returns)
        cur = self.conn.execute(
            """INSERT INTO experiment
               (hypothesis_id, title, family, stage, decision, decision_source,
                sharpe, max_drawdown, turnover, seed, issues_json, hypothesis_json,
                returns_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (hyp.hypothesis_id, hyp.title, hyp.family.value, stage,
             decision.decision.value, decision.source.value,
             sharpe, max_dd, turnover, seed,
             json.dumps([i.model_dump() for i in decision.issues]),
             hyp.model_dump_json(), returns_json,
             datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()
        return cur.lastrowid

    def total_experiments(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM experiment").fetchone()[0]

    def leaderboard(self, limit: int = 10) -> list[tuple]:
        """Kabul edilenler, Sharpe'a göre (Doküman: campaign leaderboard)."""
        return self.conn.execute(
            """SELECT hypothesis_id, title, sharpe, max_drawdown
               FROM experiment WHERE decision='accept' AND sharpe IS NOT NULL
               ORDER BY sharpe DESC LIMIT ?""", (limit,)).fetchall()

    def accepted_hypotheses(self, limit: int = 20) -> list[tuple]:
        """Holdout adayları: kabul edilenler, Sharpe'a göre. (hid, hypothesis_json, sharpe)."""
        return self.conn.execute(
            """SELECT hypothesis_id, hypothesis_json, sharpe
               FROM experiment WHERE decision='accept'
               ORDER BY sharpe DESC LIMIT ?""", (limit,)).fetchall()

    def summary_by_decision(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT decision, COUNT(*) FROM experiment GROUP BY decision").fetchall()
        return {d: c for d, c in rows}

    def prior_summaries(self, limit: int = 20) -> list[tuple]:
        """Üreticiye bağlam: ne denendi (başarılı+başarısız)."""
        return self.conn.execute(
            """SELECT hypothesis_id, title, family, decision, sharpe
               FROM experiment ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()

    def family_stats(self) -> list[tuple]:
        """Aile bazında performans: (family, count, avg_sharpe, best_sharpe)."""
        return self.conn.execute(
            """SELECT family, COUNT(*), AVG(sharpe), MAX(sharpe)
               FROM experiment WHERE sharpe IS NOT NULL
               GROUP BY family ORDER BY AVG(sharpe) DESC""").fetchall()

    def backtested_experiments(self) -> list[tuple]:
        """Backtest edilmiş TÜM deneyler (accept+reject) — multiple testing için.
        Döndürür: (hypothesis_id, title, decision, sharpe, returns_list)."""
        rows = self.conn.execute(
            """SELECT hypothesis_id, title, decision, sharpe, returns_json
               FROM experiment WHERE returns_json IS NOT NULL""").fetchall()
        return [(h, t, d, s, json.loads(rj)) for h, t, d, s, rj in rows]

    def best_by_sharpe(self) -> Optional[tuple]:
        """En yüksek Sharpe'lı deney (kabul edilmese bile) — champion adayı.
        Döndürür: (hypothesis_json, sharpe, decision) veya None."""
        return self.conn.execute(
            """SELECT hypothesis_json, sharpe, decision
               FROM experiment WHERE sharpe IS NOT NULL
               ORDER BY sharpe DESC LIMIT 1""").fetchone()

    def close(self) -> None:
        self.conn.close()
