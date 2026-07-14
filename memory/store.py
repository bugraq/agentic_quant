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
    -- Reproducibility metadata (Doküman 17.3 / 25.5)
    model_name      TEXT,
    temperature     REAL,
    prompt_hash     TEXT,
    output_hash     TEXT,
    -- Lineage (Doküman 13) — hipotez soy ağacı
    parent_hypothesis_id TEXT,
    relation_type   TEXT,          -- refinement / inversion / combination / parameter_variant
    reviews_json    TEXT,          -- reviewer ajan raporları (Doküman 15)
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
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Eski DB'lere sonradan eklenen kolonları getir (geriye uyum)."""
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(experiment)")}
        for col, decl in [("reviews_json", "TEXT")]:
            if col not in have:
                self.conn.execute(f"ALTER TABLE experiment ADD COLUMN {col} {decl}")

    def record(self, hyp: HypothesisSpec, decision: Decision, stage: str,
               result: Optional[BacktestResult] = None,
               llm_meta: Optional[dict] = None,
               parent_hypothesis_id: Optional[str] = None,
               relation_type: Optional[str] = None,
               reviews: Optional[list] = None) -> int:
        sharpe = max_dd = turnover = None
        seed = None
        returns_json = None
        if result is not None:
            sharpe = result.aggregate_sharpe()
            max_dd = max((m.max_drawdown for m in result.per_fold_metrics), default=None)
            turnover = max((m.turnover for m in result.per_fold_metrics), default=None)
            seed = result.seed
            returns_json = json.dumps(result.net_returns)
        m = llm_meta or {}
        reviews_json = json.dumps(
            [r.model_dump() if hasattr(r, "model_dump") else r for r in reviews]
        ) if reviews else None
        cur = self.conn.execute(
            """INSERT INTO experiment
               (hypothesis_id, title, family, stage, decision, decision_source,
                sharpe, max_drawdown, turnover, seed, issues_json, hypothesis_json,
                returns_json, model_name, temperature, prompt_hash, output_hash,
                parent_hypothesis_id, relation_type, reviews_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (hyp.hypothesis_id, hyp.title, hyp.family.value, stage,
             decision.decision.value, decision.source.value,
             sharpe, max_dd, turnover, seed,
             json.dumps([i.model_dump() for i in decision.issues]),
             hyp.model_dump_json(), returns_json,
             m.get("model_name"), m.get("temperature"), m.get("prompt_hash"),
             m.get("output_hash"), parent_hypothesis_id, relation_type,
             reviews_json, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()
        return cur.lastrowid

    def lineage_edges(self) -> list[tuple]:
        """Soy ağacı kenarları: (parent, child, relation_type, child_decision)."""
        return self.conn.execute(
            """SELECT parent_hypothesis_id, hypothesis_id, relation_type, decision
               FROM experiment WHERE parent_hypothesis_id IS NOT NULL""").fetchall()

    def total_experiments(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM experiment").fetchone()[0]

    def max_hypothesis_number(self) -> int:
        """Devam (resume) için: en yüksek hyp_XXXX numarası. Yoksa 0."""
        row = self.conn.execute(
            "SELECT MAX(CAST(SUBSTR(hypothesis_id, 5) AS INTEGER)) "
            "FROM experiment WHERE hypothesis_id LIKE 'hyp_%'").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def all_hypothesis_jsons(self) -> list[str]:
        """NoveltyIndex'i koşular-arası yeniden kurmak için tüm hipotezler."""
        return [r[0] for r in self.conn.execute(
            "SELECT hypothesis_json FROM experiment WHERE hypothesis_json IS NOT NULL")]

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

    def accepted_full(self) -> list[tuple]:
        """Pareto için: (hid, title, sharpe, max_drawdown, turnover, returns_list)."""
        rows = self.conn.execute(
            """SELECT hypothesis_id, title, sharpe, max_drawdown, turnover, returns_json
               FROM experiment WHERE decision='accept' AND returns_json IS NOT NULL""").fetchall()
        return [(h, t, s, dd, tn, json.loads(rj)) for h, t, s, dd, tn, rj in rows]

    def stage_counts(self) -> dict[str, int]:
        """Aşama bazında sayım (funnel / model karşılaştırma metrikleri)."""
        return dict(self.conn.execute(
            "SELECT stage, COUNT(*) FROM experiment GROUP BY stage").fetchall())

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
        Döndürür: (hid, title, decision, sharpe, returns_list, relation_type, parent_id).
        relation_type/parent, çoklu-testte parametre varyantlarını parent'a katlar."""
        rows = self.conn.execute(
            """SELECT hypothesis_id, title, decision, sharpe, returns_json,
                      relation_type, parent_hypothesis_id
               FROM experiment WHERE returns_json IS NOT NULL""").fetchall()
        return [(h, t, d, s, json.loads(rj), rel, par)
                for h, t, d, s, rj, rel, par in rows]

    def family_outcome_counts(self) -> dict[str, tuple[int, int]]:
        """Bandit için: family -> (kabul, toplam_backtest). Sadece backtest'lenenler."""
        rows = self.conn.execute(
            """SELECT family,
                      SUM(CASE WHEN decision='accept' THEN 1 ELSE 0 END) AS accepts,
                      COUNT(*) AS total
               FROM experiment WHERE sharpe IS NOT NULL
               GROUP BY family""").fetchall()
        return {fam: (int(acc or 0), int(tot or 0)) for fam, acc, tot in rows}

    def best_by_sharpe(self) -> Optional[tuple]:
        """En yüksek Sharpe'lı deney (kabul edilmese bile) — champion adayı.
        Döndürür: (hypothesis_json, sharpe, decision) veya None."""
        return self.conn.execute(
            """SELECT hypothesis_json, sharpe, decision
               FROM experiment WHERE sharpe IS NOT NULL
               ORDER BY sharpe DESC LIMIT 1""").fetchone()

    def best_accepted(self, exclude: "set | None" = None) -> Optional[tuple]:
        """En iyi KABUL EDİLMİŞ hipotez — revision champion'ı (Doküman 16.1).
        Ham Sharpe yerine 'doğrulanmış' (gate+fold+robustness geçmiş) olanı seçer,
        böylece reddedilmiş yüksek-Sharpe peşinde koşulmaz. exclude'dakiler
        (revizyonu tükenmiş champion'lar) atlanır. (json, sharpe) | None."""
        rows = self.conn.execute(
            """SELECT hypothesis_id, hypothesis_json, sharpe FROM experiment
               WHERE decision='accept' AND sharpe IS NOT NULL
               ORDER BY sharpe DESC""").fetchall()
        for hid, hjson, sharpe in rows:
            if not exclude or hid not in exclude:
                return (hjson, sharpe)
        return None

    def exhausted_revision_parent_ids(self, max_duplicates: int = 3) -> set:
        """Revizyonları >= max_duplicates kez duplicate üretmiş champion'lar.

        LLM bir champion'ın etrafında hep aynı yapıyı döndürüyorsa o komşuluk
        TÜKENMİŞTİR; ısrar bütçe israfı (gerçek koşuda 24 slotun ~8'i böyle
        yandı). Bu champion'lar revision için karantinaya alınır — kabul
        kayıtları ve leaderboard'daki yerleri etkilenmez.
        """
        return {r[0] for r in self.conn.execute(
            """SELECT parent_hypothesis_id FROM experiment
               WHERE relation_type='refinement' AND decision='duplicate'
                 AND parent_hypothesis_id IS NOT NULL
               GROUP BY parent_hypothesis_id
               HAVING COUNT(*) >= ?""", (max_duplicates,))}

    def inverted_parent_ids(self) -> set:
        """Daha önce inversion denenen parent'lar (sonuç ne olursa olsun).

        Aynı hipotezi tekrar tekrar ters çevirmek bütçe kara deliğidir: tersi
        zaten denendi (başarısız/duplicate olsa bile). Lineage'dan okunur.
        """
        return {r[0] for r in self.conn.execute(
            """SELECT DISTINCT parent_hypothesis_id FROM experiment
               WHERE relation_type='inversion'
                 AND parent_hypothesis_id IS NOT NULL""")}

    def worst_failed_hypothesis(self, exclude: "set | None" = None) -> Optional[tuple]:
        """En negatif Sharpe'lı reddedilen deney — inversion (ters çevirme) adayı.
        exclude'dakiler (zaten ters çevrilenler) atlanır.
        Döndürür: (hypothesis_json, sharpe) veya None."""
        rows = self.conn.execute(
            """SELECT hypothesis_id, hypothesis_json, sharpe FROM experiment
               WHERE decision='reject' AND sharpe IS NOT NULL AND sharpe < -0.3
               ORDER BY sharpe ASC""").fetchall()
        for hid, hjson, sharpe in rows:
            if not exclude or hid not in exclude:
                return (hjson, sharpe)
        return None

    def close(self) -> None:
        self.conn.close()
