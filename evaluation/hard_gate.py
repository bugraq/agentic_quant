"""
Hard Gate (Doküman 11.1) — deterministik, tartışmasız red kapısı.

Çok amaçlı Pareto sıralamasından ÖNCE gelir: temel şartları geçemeyen
strateji doğrudan elenir. Bu kapı LLM'i her zaman ezer (look-ahead ya da
negatif Sharpe varsa LLM 'güzel' dese bile reddedilir).

İki kaynak birleşir:
  - Kampanya sabit kuralları (Sharpe<0, DD>40%, aşırı turnover)
  - Hipotezin kendi ön-kayıtlı falsification eşikleri (min OOS Sharpe vs.)
"""
from __future__ import annotations

from contracts.backtest_result import BacktestResult
from contracts.decision import (
    Decision, DecisionSource, DecisionType, Issue, Severity,
)
from contracts.hypothesis_spec import HypothesisSpec

# Kampanya düzeyi sabit kurallar.
# Not: turnover cezası ZATEN maliyet olarak Sharpe'a yansıyor; bu kap yalnızca
# absürt turnover'ı (ör. kapasite dışı) eler. Günlük long-short için 300 gerçekçi.
HARD_MAX_DRAWDOWN = 0.40
HARD_MAX_TURNOVER = 300.0   # yıllık


def evaluate(result: BacktestResult, hyp: HypothesisSpec,
             min_acceptance_sharpe: float = 0.5,
             min_positive_folds: float = 0.5,
             max_drawdown: float = HARD_MAX_DRAWDOWN,
             max_turnover: float = HARD_MAX_TURNOVER) -> Decision:
    """
    Kabul/red KAMPANYANIN sabit eşiğiyle belirlenir (LLM gameleyemez).
    Hipotezin kendi falsification eşiği yalnızca 'kendi iddiasını tutturdu mu'
    bilgisi olarak KAYDEDİLİR — kabul kapısını LLM'e vermeyiz.
    """
    issues: list[Issue] = []
    sharpe = result.aggregate_sharpe() or 0.0
    worst_dd = max((m.max_drawdown for m in result.per_fold_metrics), default=0.0)
    max_turn = max((m.turnover for m in result.per_fold_metrics), default=0.0)

    # --- Kampanya sabit kuralları (kabul kapısı) ---
    if sharpe < min_acceptance_sharpe:
        issues.append(Issue(
            type="below_acceptance_sharpe",
            description=f"OOS Sharpe {sharpe:.2f} < kampanya eşiği {min_acceptance_sharpe:.2f}."))
    if worst_dd > max_drawdown:
        issues.append(Issue(type="excessive_drawdown",
                            description=f"Max drawdown %{worst_dd*100:.0f} > %{max_drawdown*100:.0f}."))
    if max_turn > max_turnover:
        issues.append(Issue(type="excessive_turnover",
                            description=f"Turnover {max_turn:.1f} > {max_turnover}."))

    # --- Walk-forward tutarlılığı (rejimler arası kararlılık) ---
    # Hipotezin kendi taahhüdü varsa onu, yoksa kampanya varsayılanını kullan.
    fold_frac = result.exposures.get("positive_fold_fraction")
    if fold_frac is not None:
        req = hyp.falsification.minimum_positive_walk_forward_folds or min_positive_folds
        if fold_frac < req:
            issues.append(Issue(
                type="fold_inconsistency",
                description=(f"Pozitif fold oranı {fold_frac:.0%} < gerekli {req:.0%} "
                            f"— strateji dönemler arası tutarsız.")))

    if issues:
        return Decision(hypothesis_id=hyp.hypothesis_id, decision=DecisionType.reject,
                        source=DecisionSource.gate, severity=Severity.medium, issues=issues)

    # Kabul edildi. Kendi ön-kayıtlı iddiasını tutturdu mu? (bilgi amaçlı)
    met_own = sharpe >= hyp.falsification.minimum_oos_sharpe
    info = [] if met_own else [Issue(
        type="below_own_claim",
        description=(f"Kabul edildi ama kendi ön-kayıtlı eşiğinin "
                     f"({hyp.falsification.minimum_oos_sharpe:.2f}) altında — iddia tam tutmadı."))]
    return Decision(hypothesis_id=hyp.hypothesis_id, decision=DecisionType.accept,
                    source=DecisionSource.gate, severity=Severity.low, issues=info)
