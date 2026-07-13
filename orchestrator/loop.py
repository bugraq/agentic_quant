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
from optimization import n_window_slots, optimize_parameters

EXPLORE_ROUNDS = 3   # ilk turlar: keşif (sıfırdan yeni). Sonra: champion'ı geliştir.

# Üretim modu -> lineage relation_type (Doküman 13)
_RELATION = {
    GenerationMode.revision: "refinement",
    GenerationMode.inversion: "inversion",
    GenerationMode.combination: "combination",
}

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


# LLM'e gösterilen anonim evren tarifi (memorization önlemi, aşağıya bak).
ANONYMOUS_UNIVERSE = (
    "Likit, büyük ölçekli hisse senetlerinden oluşan kesitsel bir evren; "
    "günlük OHLCV barlar. Hangi piyasa, hangi şirketler ve hangi tarih aralığı "
    "olduğu BİLİNÇLİ olarak verilmiyor — genel geçer, mekanizma temelli "
    "hipotezler üret (belirli şirket/dönem bilgisine dayanma).")


@dataclass
class CampaignConfig:
    goal: str = "Kesitsel günlük alpha ara"
    universe_description: str = "20 ABD hissesi, günlük bar, point-in-time (sentetik)"
    # MEMORIZATION ÖNLEMİ (Look-Ahead-Bench / Memorization Problem literatürü):
    # LLM eğitim verisinden "2015-2023'te NVDA uçtu" gibi geleceği ezbere bilir.
    # Ticker adları + tarih aralığı prompta girerse backtest dönemine dair
    # parametre-içi sızıntı olur. Açıkken LLM'e yalnızca anonim tarif gider.
    anonymize_universe: bool = True
    # İzin verilen strateji uzayı (Campaign Manager kısıtları)
    allowed_fields: list[str] = field(default_factory=list)
    allowed_operators: list[str] = field(default_factory=list)
    allowed_horizons: list[int] = field(default_factory=list)
    allowed_rebalance: list[str] = field(default_factory=list)
    portfolio_types: list[str] = field(default_factory=list)
    # Bütçe
    max_experiments: int = 10
    max_llm_tokens: int = 300000
    cost_bps: float = 5.0
    # Risk kısıtları (hard gate; LLM gameleyemez)
    min_acceptance_sharpe: float = 0.5
    max_drawdown: float = 0.40
    max_turnover: float = 300.0
    min_positive_folds: float = 0.5
    # Deney protokolü
    research_fraction: float = 0.7
    # Sayısal optimizasyon (Doküman 27: LLM yapısal, motor sayısal)
    parameter_optimization: bool = False


def _decide_mode(iteration: int, memory: MemoryStore):
    """Mod kararı: keşif turlarında yeni; sonra pozitif champion varsa onu geliştir.

    Döndürür: (GenerationMode, parent_hypothesis | None, champion_sharpe | None)
    """
    # Champion = KABUL EDİLMİŞ en iyi hipotez (ham Sharpe peşinde koşma yok, Doküman 16.1).
    champion = memory.best_accepted()   # (json, sharpe) | None
    champ_sharpe = champion[1] if champion else None
    # Keşif turları: sıfırdan yeni yön dene.
    if iteration < EXPLORE_ROUNDS:
        return GenerationMode.new, None, champ_sharpe
    # Exploit fazı. İki durumda başarısız bir hipotezi TERS ÇEVİR (inversion):
    #   (a) hiç kabul yoksa (champion None) — pes etme, naive sinyalin tersini dene
    #       (ör. momentum kaybediyorsa kısa-vadeli reversal kazanıyor olabilir);
    #   (b) champion varsa her 3. turda çeşitlilik için.
    # Aksi halde kabul edilmiş champion'ı geliştir (revision).
    if champion is None or iteration % 3 == 2:
        failed = memory.worst_failed_hypothesis()
        if failed:
            return (GenerationMode.inversion,
                    HypothesisSpec.model_validate_json(failed[0]), champ_sharpe)
        # Ters çevrilecek belirgin başarısız yoksa keşfe devam.
        return GenerationMode.new, None, champ_sharpe
    return GenerationMode.revision, HypothesisSpec.model_validate_json(champion[0]), champ_sharpe


def _build_context(cfg: CampaignConfig, memory: MemoryStore, remaining: int,
                   mode: GenerationMode, parent: HypothesisSpec | None,
                   suggested_family: str | None = None,
                   literature: list[str] | None = None) -> ResearchContext:
    priors = [
        ExperimentSummary(hypothesis_id=h, title=t, family=f, outcome=d,
                          headline_metric=(f"Sharpe {s:.2f}" if s is not None else None))
        for (h, t, f, d, s) in memory.prior_summaries()
    ]
    lessons = build_lessons(memory.family_stats())
    # LLM'e giden evren tarifi: anonimleştirme açıksa ticker/tarih İÇERMEZ.
    llm_universe = ANONYMOUS_UNIVERSE if cfg.anonymize_universe else cfg.universe_description
    return ResearchContext(
        campaign_goal=cfg.goal,
        universe_description=llm_universe,
        allowed_fields=cfg.allowed_fields,
        allowed_operators=cfg.allowed_operators,
        allowed_horizons=cfg.allowed_horizons,
        allowed_rebalance=cfg.allowed_rebalance,
        allowed_portfolio_types=cfg.portfolio_types,
        prior_experiments=priors,
        lessons=lessons,
        generation_mode=mode,
        parent_hypothesis=parent,
        suggested_family=suggested_family,
        literature_mechanisms=literature or [],
        experiments_remaining=remaining,
    )


def run_campaign(provider: HypothesisProvider, data: MarketData,
                 memory: MemoryStore, cfg: CampaignConfig, critic=None,
                 literature: list[str] | None = None) -> None:
    from agents.quant_critic import DummyCritic
    critic = critic or DummyCritic()

    # DEVAM (resume): ID sayacını ve NoveltyIndex'i hafızadan yeniden kur, böylece
    # önceki koşularla aynı hipotez tekrar üretilmez ve numaralar çakışmaz.
    start = memory.max_hypothesis_number()
    if hasattr(provider, "_counter"):
        provider._counter = start
    novelty = NoveltyIndex()
    seeded = 0
    for hj in memory.all_hypothesis_jsons():
        try:
            novelty.add(HypothesisSpec.model_validate_json(hj))   # yapısal (sinyal df yok)
            seeded += 1
        except Exception:  # noqa: BLE001
            pass
    if start:
        print(f"[devam] {start} önceki deney hafızada; novelty {seeded} sinyalle kuruldu.\n")

    bandit = ThompsonBandit([f.value for f in HypothesisFamily], seed=0)
    for i in range(cfg.max_experiments):
        # Token bütçesi kontrolü (Campaign Manager) — aşılınca kampanya durur
        used = (getattr(provider, "total_prompt_tokens", 0)
                + getattr(provider, "total_completion_tokens", 0))
        if used >= cfg.max_llm_tokens:
            print(f"[bütçe] LLM token bütçesi ({cfg.max_llm_tokens}) doldu "
                  f"({used}); kampanya durduruldu.")
            break
        remaining = cfg.max_experiments - i
        mode, parent, champ_sharpe = _decide_mode(i, memory)
        # Yeni hipotez modunda bandit aile seçer (bütçe tahsisi); revision'da champion'ın ailesi
        suggested = bandit.select(memory.family_outcome_counts()) \
            if mode == GenerationMode.new else None
        ctx = _build_context(cfg, memory, remaining, mode, parent, suggested, literature)

        # 0) Hipotez üret — LLM geçerli çıktı veremezse turu atla (kampanya çökmesin)
        try:
            hyp = provider.next(ctx)
        except Exception as e:  # noqa: BLE001 — sağlayıcıya özgü hatalar dahil
            print(f"[{i+1}/{cfg.max_experiments}] ÜRETİM HATASI (atlandı): "
                  f"{type(e).__name__}: {str(e)[:160]}")
            continue
        mode_tag = mode.value + (f"<-{parent.hypothesis_id}" if parent else "")
        tag = f"[{i+1}/{cfg.max_experiments}] ({mode_tag}) {hyp.hypothesis_id} {hyp.title}"

        # Reproducibility (17.3) + lineage (13): her kayda eklenecek ortak metadata
        parent_id = parent.hypothesis_id if parent else None
        relation = _RELATION.get(mode)
        meta = dict(llm_meta=getattr(provider, "last_meta", None),
                    parent_hypothesis_id=parent_id, relation_type=relation)
        _mrec = memory.record

        def rec(h, d, s, result=None):
            return _mrec(h, d, s, result=result, **meta)

        # 1) Derle (yapısal hatalar burada)
        try:
            graph = compile_hypothesis(hyp)
        except CompileError as e:
            dec = Decision(hypothesis_id=hyp.hypothesis_id, decision=DecisionType.reject,
                           source=DecisionSource.gate, severity=Severity.high,
                           issues=[Issue(type="compile_error", description=str(e))])
            rec(hyp, dec, STAGE_COMPILE_ERROR)
            print(f"{tag} -> REDDEDİLDİ (derleme): {e}")
            continue

        # 2) Statik doğrula (SIZINTI + izin verilen alan/rebalance/portföy kontrolü)
        static = validate(graph, hyp, allowed_fields=cfg.allowed_fields or None,
                          allowed_rebalance=cfg.allowed_rebalance or None,
                          allowed_portfolio_types=cfg.portfolio_types or None)
        if static.decision != DecisionType.accept:
            rec(hyp, static, STAGE_STATIC_REJECTED)
            reason = static.issues[0].type if static.issues else "?"
            print(f"{tag} -> {static.decision.value.upper()} (statik): {reason}")
            continue

        # 3a) Yapısal yenilik kontrolü (backtest'ten ÖNCE — bütçe korur)
        dup = novelty.check_structural(hyp)
        if dup:
            rec(hyp, _duplicate_decision(hyp, dup, "yapısal"), STAGE_DUPLICATE)
            print(f"{tag} -> DUPLICATE (yapısal, ~{dup}) — backtest atlandı")
            continue

        # 3a2) Quant Critic — BAĞIMSIZ ekonomik inceleme (backtest'ten önce, bütçe korur)
        try:
            crit = critic.review(hyp)
        except Exception:  # noqa: BLE001 — eleştirmen hatası araştırmayı bloklamasın
            crit = None
        if crit is not None and crit.decision != DecisionType.accept:
            rec(hyp, crit, STAGE_CRITIC_REJECTED)
            reason = crit.issues[0].type if crit.issues else "?"
            print(f"{tag} -> {crit.decision.value.upper()} (critic): {reason}")
            continue

        # 3b) Sinyali hesapla, davranışsal yenilik kontrolü (korelasyon)
        signal = evaluate_signal(graph, data)
        dup = novelty.check_behavioral(signal)
        if dup:
            rec(hyp, _duplicate_decision(hyp, dup, "davranışsal"), STAGE_DUPLICATE)
            print(f"{tag} -> DUPLICATE (davranışsal, ~{dup})")
            continue

        # 3c) Walk-forward backtest (çoklu fold, önceden hesaplanan sinyalle)
        result = run_walk_forward(graph, hyp, data, n_folds=5,
                                  cost_bps=cfg.cost_bps, signal=signal)
        novelty.add(hyp, signal)
        sharpe = result.aggregate_sharpe() or 0.0

        # 4) Hard gate (kampanya risk kısıtları — config'ten)
        gate = hard_gate_evaluate(result, hyp, cfg.min_acceptance_sharpe,
                                  min_positive_folds=cfg.min_positive_folds,
                                  max_drawdown=cfg.max_drawdown,
                                  max_turnover=cfg.max_turnover)
        if gate.decision != DecisionType.accept:
            rec(hyp, gate, STAGE_GATE_REJECTED, result=result)
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
            rec(hyp, dec, STAGE_ROBUSTNESS_REJECTED, result=result)
            print(f"{tag} -> RED (sağlamlık, Sharpe {sharpe:.2f}): "
                  f"perm_p={rob.permutation_pvalue:.2f}")
            continue

        # 5b) SAYISAL PARAMETRE OPTİMİZASYONU (Doküman 27) — SADECE İYİLEŞTİRME.
        # Kabul+robust bir stratejinin pencerelerini arar; optimize versiyon HEM
        # daha iyi HEM robust ise onu al, değilse orijinali koru (asla bozma).
        if cfg.parameter_optimization and cfg.allowed_horizons and n_window_slots(hyp) > 0:
            opt_hyp, opt_score, opt_trials = optimize_parameters(
                hyp, data, cfg.allowed_horizons, cost_bps=cfg.cost_bps, n_samples=8)
            # DÜRÜST SAYIM: optimizer'ın yaptığı HER backtest bir denemedir ve
            # multiple-testing muhasebesine girer (Doküman 10/12). Kaydet.
            for t_hyp, t_res in opt_trials:
                t_dec = Decision(
                    hypothesis_id=t_hyp.hypothesis_id, decision=DecisionType.reject,
                    source=DecisionSource.statistical, severity=Severity.low,
                    issues=[Issue(type="parameter_search_trial",
                                  description="Parametre arama denemesi (seçilmedi; "
                                              "yalnızca çoklu-test sayımı için).")])
                memory.record(t_hyp, t_dec, "parameter_search", result=t_res,
                              parent_hypothesis_id=hyp.hypothesis_id,
                              relation_type="parameter_variant")
            if opt_trials:
                print(f"    (parametre arama: {len(opt_trials)} deneme sayıma eklendi)")
            # Muhafazakâr karşılaştırma: min-fold vs min-fold (elma-elma).
            orig_min_fold = min((m.sharpe for m in result.per_fold_metrics), default=0.0)
            if opt_hyp is not hyp and (opt_score or -99) > orig_min_fold + 0.05:
                g2 = compile_hypothesis(opt_hyp)
                sig2 = evaluate_signal(g2, data)
                res2 = run_walk_forward(g2, opt_hyp, data, n_folds=5,
                                        cost_bps=cfg.cost_bps, signal=sig2)
                gate2 = hard_gate_evaluate(res2, opt_hyp, cfg.min_acceptance_sharpe,
                                           min_positive_folds=cfg.min_positive_folds,
                                           max_drawdown=cfg.max_drawdown,
                                           max_turnover=cfg.max_turnover)
                rob2 = run_robustness(g2, opt_hyp, data, cost_bps=cfg.cost_bps, signal=sig2)
                if gate2.decision == DecisionType.accept and rob2.robust:
                    # Kriter MUHAFAZAKÂR (min-fold); ortalama Sharpe düşebilir —
                    # bilinçli takas: tutarlılık > tepe performans (Doküman 11.2).
                    print(f"{tag} -> parametre optimize edildi (min-fold "
                          f"{orig_min_fold:.2f} -> {opt_score:.2f}; ort Sharpe "
                          f"{sharpe:.2f} -> {res2.aggregate_sharpe():.2f})")
                    hyp, result, gate = opt_hyp, res2, gate2
                    sharpe = result.aggregate_sharpe() or 0.0

        rec(hyp, gate, STAGE_ACCEPTED, result=result)
        print(f"{tag} -> KABUL (Sharpe {sharpe:.2f}, perm_p={rob.permutation_pvalue:.2f}, "
              f"cost2x={rob.cost2x_sharpe:.2f})")
