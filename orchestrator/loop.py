"""
Research Orchestrator — döngünün kendisi (basit Python loop, LangGraph DEĞİL).

Her iterasyon bir hipotezi pipeline'dan geçirir:
  Context -> [LLM] Hipotez -> Derle -> Statik Doğrula (sızıntı) ->
  Backtest -> Hard Gate -> Hafızaya Yaz -> (yönlendir)

Her deney — reddedilenler dahil — hafızaya kaydedilir (Doküman 2.3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from contracts.decision import Decision, DecisionSource, DecisionType, Issue, Severity
from contracts.hypothesis_spec import HypothesisFamily, HypothesisSpec
from contracts.research_context import (
    ExperimentSummary, GenerationMode, ResearchContext,
)
from data.synthetic import MarketData
from dsl import CompileError, compile_hypothesis, validate
from evaluation import hard_gate_evaluate
from llm import HypothesisProvider
from memory import MemoryStore
from memory.semantic import build_lessons
from memory.similarity import NoveltyIndex
from orchestrator.budget import ThompsonBandit
from backtest import evaluate_signal
from backtest.walk_forward import run_walk_forward
from evaluation.robustness import run_robustness

EXPLORE_ROUNDS = 3   # ilk turlar: keşif (sıfırdan yeni). Sonra: champion'ı geliştir.

# Deney yaşam döngüsü aşamaları (Doküman 22, iskelet alt kümesi)
STAGE_COMPILE_ERROR = "compile_error"
STAGE_STATIC_REJECTED = "static_rejected"
STAGE_CRITIC_REJECTED = "critic_rejected"
STAGE_DUPLICATE = "duplicate"
STAGE_GATE_REJECTED = "gate_rejected"
STAGE_ROBUSTNESS_REJECTED = "robustness_rejected"
STAGE_ACCEPTED = "accepted"


def _duplicate_decision(hyp, dup_of: str, kind: str) -> Decision:
    return Decision(
        hypothesis_id=hyp.hypothesis_id, decision=DecisionType.duplicate,
        source=DecisionSource.novelty, severity=Severity.low,
        issues=[Issue(type=f"{kind}_duplicate",
                      description=f"{kind} olarak {dup_of} ile aynı — tekrar test edilmedi.")])


@dataclass
class CampaignConfig:
    goal: str = "Kesitsel günlük alpha ara"
    universe_description: str = "20 ABD hissesi, günlük bar, point-in-time (sentetik)"
    allowed_operators: list[str] = field(default_factory=list)
    max_experiments: int = 10
    cost_bps: float = 5.0
    min_acceptance_sharpe: float = 0.5   # kampanya kabul eşiği (LLM gameleyemez)


def _decide_mode(iteration: int, memory: MemoryStore):
    """Mod kararı: keşif turlarında yeni; sonra pozitif champion varsa onu geliştir.

    Döndürür: (GenerationMode, parent_hypothesis | None, champion_sharpe | None)
    """
    champion = memory.best_by_sharpe()   # (json, sharpe, decision) | None
    if iteration < EXPLORE_ROUNDS or champion is None or (champion[1] or -1) <= 0:
        return GenerationMode.new, None, (champion[1] if champion else None)
    parent = HypothesisSpec.model_validate_json(champion[0])
    return GenerationMode.revision, parent, champion[1]


def _build_context(cfg: CampaignConfig, memory: MemoryStore, remaining: int,
                   mode: GenerationMode, parent: HypothesisSpec | None,
                   suggested_family: str | None = None) -> ResearchContext:
    priors = [
        ExperimentSummary(hypothesis_id=h, title=t, family=f, outcome=d,
                          headline_metric=(f"Sharpe {s:.2f}" if s is not None else None))
        for (h, t, f, d, s) in memory.prior_summaries()
    ]
    lessons = build_lessons(memory.family_stats())
    return ResearchContext(
        campaign_goal=cfg.goal,
        universe_description=cfg.universe_description,
        allowed_operators=cfg.allowed_operators,
        prior_experiments=priors,
        lessons=lessons,
        generation_mode=mode,
        parent_hypothesis=parent,
        suggested_family=suggested_family,
        experiments_remaining=remaining,
    )


def run_campaign(provider: HypothesisProvider, data: MarketData,
                 memory: MemoryStore, cfg: CampaignConfig, critic=None) -> None:
    from agents.quant_critic import DummyCritic
    critic = critic or DummyCritic()
    novelty = NoveltyIndex()   # kampanya boyunca görülen sinyaller
    bandit = ThompsonBandit([f.value for f in HypothesisFamily], seed=0)
    for i in range(cfg.max_experiments):
        remaining = cfg.max_experiments - i
        mode, parent, champ_sharpe = _decide_mode(i, memory)
        # Yeni hipotez modunda bandit aile seçer (bütçe tahsisi); revision'da champion'ın ailesi
        suggested = bandit.select(memory.family_outcome_counts()) \
            if mode == GenerationMode.new else None
        ctx = _build_context(cfg, memory, remaining, mode, parent, suggested)

        # 0) Hipotez üret — LLM geçerli çıktı veremezse turu atla (kampanya çökmesin)
        try:
            hyp = provider.next(ctx)
        except Exception as e:  # noqa: BLE001 — sağlayıcıya özgü hatalar dahil
            print(f"[{i+1}/{cfg.max_experiments}] ÜRETİM HATASI (atlandı): "
                  f"{type(e).__name__}: {str(e)[:160]}")
            continue
        mode_tag = mode.value + (f"<-{parent.hypothesis_id}" if parent else "")
        tag = f"[{i+1}/{cfg.max_experiments}] ({mode_tag}) {hyp.hypothesis_id} {hyp.title}"

        # 1) Derle (yapısal hatalar burada)
        try:
            graph = compile_hypothesis(hyp)
        except CompileError as e:
            dec = Decision(hypothesis_id=hyp.hypothesis_id, decision=DecisionType.reject,
                           source=DecisionSource.gate, severity=Severity.high,
                           issues=[Issue(type="compile_error", description=str(e))])
            memory.record(hyp, dec, STAGE_COMPILE_ERROR)
            print(f"{tag} -> REDDEDİLDİ (derleme): {e}")
            continue

        # 2) Statik doğrula (SIZINTI kontrolü)
        static = validate(graph, hyp)
        if static.decision != DecisionType.accept:
            memory.record(hyp, static, STAGE_STATIC_REJECTED)
            reason = static.issues[0].type if static.issues else "?"
            print(f"{tag} -> {static.decision.value.upper()} (statik): {reason}")
            continue

        # 3a) Yapısal yenilik kontrolü (backtest'ten ÖNCE — bütçe korur)
        dup = novelty.check_structural(hyp)
        if dup:
            memory.record(hyp, _duplicate_decision(hyp, dup, "yapısal"), STAGE_DUPLICATE)
            print(f"{tag} -> DUPLICATE (yapısal, ~{dup}) — backtest atlandı")
            continue

        # 3a2) Quant Critic — BAĞIMSIZ ekonomik inceleme (backtest'ten önce, bütçe korur)
        try:
            crit = critic.review(hyp)
        except Exception:  # noqa: BLE001 — eleştirmen hatası araştırmayı bloklamasın
            crit = None
        if crit is not None and crit.decision != DecisionType.accept:
            memory.record(hyp, crit, STAGE_CRITIC_REJECTED)
            reason = crit.issues[0].type if crit.issues else "?"
            print(f"{tag} -> {crit.decision.value.upper()} (critic): {reason}")
            continue

        # 3b) Sinyali hesapla, davranışsal yenilik kontrolü (korelasyon)
        signal = evaluate_signal(graph, data)
        dup = novelty.check_behavioral(signal)
        if dup:
            memory.record(hyp, _duplicate_decision(hyp, dup, "davranışsal"), STAGE_DUPLICATE)
            print(f"{tag} -> DUPLICATE (davranışsal, ~{dup})")
            continue

        # 3c) Walk-forward backtest (çoklu fold, önceden hesaplanan sinyalle)
        result = run_walk_forward(graph, hyp, data, n_folds=5,
                                  cost_bps=cfg.cost_bps, signal=signal)
        novelty.add(hyp, signal)
        sharpe = result.aggregate_sharpe() or 0.0

        # 4) Hard gate (kampanya sabit eşiği + fold tutarlılığı)
        gate = hard_gate_evaluate(result, hyp, cfg.min_acceptance_sharpe)
        if gate.decision != DecisionType.accept:
            memory.record(hyp, gate, STAGE_GATE_REJECTED, result=result)
            reason = gate.issues[0].type if gate.issues else "?"
            print(f"{tag} -> RED (gate, Sharpe {sharpe:.2f}): {reason}")
            continue

        # 5) Sağlamlık testleri (permutation, maliyet 2x, parametre perturbasyonu)
        rob = run_robustness(graph, hyp, data, cost_bps=cfg.cost_bps, signal=signal)
        if not rob.robust:
            dec = Decision(
                hypothesis_id=hyp.hypothesis_id, decision=DecisionType.reject,
                source=DecisionSource.statistical, severity=Severity.medium,
                issues=[Issue(type="not_robust",
                              description=(f"perm_p={rob.permutation_pvalue:.2f}, "
                                           f"cost2x_Sharpe={rob.cost2x_sharpe:.2f}, "
                                           f"param_min_Sharpe={rob.param_min_sharpe:.2f}"))])
            memory.record(hyp, dec, STAGE_ROBUSTNESS_REJECTED, result=result)
            print(f"{tag} -> RED (sağlamlık, Sharpe {sharpe:.2f}): "
                  f"perm_p={rob.permutation_pvalue:.2f}")
            continue

        memory.record(hyp, gate, STAGE_ACCEPTED, result=result)
        print(f"{tag} -> KABUL (Sharpe {sharpe:.2f}, perm_p={rob.permutation_pvalue:.2f}, "
              f"cost2x={rob.cost2x_sharpe:.2f})")
