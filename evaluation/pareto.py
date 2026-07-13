"""
Çok amaçlı değerlendirme ve Pareto sıralaması (Doküman 11.2).

Hard gate'i geçen stratejiler TEK bir skorla değil, birden çok boyutta
değerlendirilir. Amaç reward hacking'i zorlaştırmak: yüksek Sharpe tek başına
yetmez; drawdown, turnover ve Sharpe'ın ALT güven sınırı da hesaba katılır.

  maximize:  sharpe_lb (bootstrap alt güven sınırı — tesadüfi yüksek Sharpe'ı eler)
  minimize:  max_drawdown, turnover

Pareto-optimal = hiçbir başka strateji tarafından TÜM boyutlarda domine edilmeyen.
Ayrıca bütçe tahsisi için muhafazakâr bir skaler skor da üretilir (yardımcı sinyal).
"""
from __future__ import annotations

from dataclasses import dataclass

from evaluation.statistics import bootstrap_sharpe_ci

# Muhafazakâr skor ağırlıkları (Doküman 11.2)
LAMBDA_DD = 0.5
LAMBDA_TURN = 0.002


@dataclass
class StrategyEval:
    hypothesis_id: str
    title: str
    sharpe: float
    sharpe_lb: float      # bootstrap alt güven sınırı (yıllık)
    max_drawdown: float
    turnover: float
    score: float          # muhafazakâr skaler skor
    pareto_optimal: bool = False


def _objectives(ev: StrategyEval) -> tuple[float, float, float]:
    """Tümü 'büyük daha iyi' formuna çevrilmiş amaçlar (minimize -> negate)."""
    return (ev.sharpe_lb, -ev.max_drawdown, -ev.turnover)


def _dominates(a: StrategyEval, b: StrategyEval) -> bool:
    """a, b'yi domine eder mi: tüm amaçlarda >=, en az birinde > ."""
    oa, ob = _objectives(a), _objectives(b)
    return all(x >= y for x, y in zip(oa, ob)) and any(x > y for x, y in zip(oa, ob))


def conservative_score(sharpe_lb: float, max_dd: float, turnover: float) -> float:
    return sharpe_lb - LAMBDA_DD * max_dd - LAMBDA_TURN * turnover


def evaluate_strategies(records: list[tuple]) -> list[StrategyEval]:
    """
    records: (hid, title, sharpe, max_drawdown, turnover, returns_list).
    Pareto-optimal bayrağı + muhafazakâr skor hesaplar, skora göre sıralı döner.
    """
    evals: list[StrategyEval] = []
    for hid, title, sharpe, max_dd, turnover, returns in records:
        lo, _ = bootstrap_sharpe_ci(returns or [], n_boot=400, seed=0)
        sharpe_lb = lo if lo == lo else (sharpe or 0.0)   # NaN ise Sharpe'a düş
        max_dd = max_dd or 0.0
        turnover = turnover or 0.0
        evals.append(StrategyEval(
            hypothesis_id=hid, title=title, sharpe=sharpe or 0.0, sharpe_lb=sharpe_lb,
            max_drawdown=max_dd, turnover=turnover,
            score=conservative_score(sharpe_lb, max_dd, turnover)))

    # Pareto front: hiçbir başkası tarafından domine edilmeyenler
    for a in evals:
        a.pareto_optimal = not any(b is not a and _dominates(b, a) for b in evals)

    evals.sort(key=lambda e: (e.pareto_optimal, e.score), reverse=True)
    return evals
