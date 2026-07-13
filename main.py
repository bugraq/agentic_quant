"""
Walking Skeleton — uçtan uca araştırma döngüsü.

Bir kampanyayı başlatır: dummy LLM sabit katalogdan hipotez üretir, her biri
derlenir -> sızıntı kontrolünden geçer -> backtest edilir -> hard gate ->
hafızaya (SQLite) yazılır. Sonunda özet + leaderboard basılır.

Modeli gerçek LLM'e çevirmek için: configs/models.yaml -> provider: anthropic.
Kod DEĞİŞMEZ.

Çalıştır:  ./.venv/Scripts/python.exe main.py
"""
from __future__ import annotations

import argparse
import os

import yaml
from dotenv import load_dotenv

from contracts.hypothesis_spec import HypothesisSpec
from dashboard import generate_dashboard
from data import make_adapter, split_by_fraction
from evaluation import build_report, print_report
from holdout import HoldoutError, HoldoutService
from llm import make_critic, make_provider
from memory import MemoryStore
from orchestrator import CampaignConfig, run_campaign

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "research_memory.sqlite")
HOLDOUT_DB = os.path.join(HERE, "holdout_audit.sqlite")


def load_yaml(name: str) -> dict:
    with open(os.path.join(HERE, "configs", name), encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Otonom quant araştırma kampanyası")
    parser.add_argument("--fresh", action="store_true",
                        help="Yeni kampanya: hafızayı SIFIRLA. Varsayılan: mevcut kampanyaya DEVAM et.")
    args = parser.parse_args()

    load_dotenv(os.path.join(HERE, ".env"))   # API key'i ortama yükle (koda girmez)
    campaign = load_yaml("campaign.yaml")["campaign"]
    models = load_yaml("models.yaml")["models"]
    data_cfg = load_yaml("data.yaml")["data"]

    budget = campaign.get("budget", {})
    risk = campaign.get("risk_constraints", {})
    hpol = campaign.get("holdout_policy", {})
    cfg = CampaignConfig(
        goal=campaign["goal"],
        universe_description=campaign["universe_description"],
        allowed_fields=campaign.get("allowed_fields", []),
        allowed_operators=campaign.get("allowed_operators", []),
        allowed_horizons=campaign.get("allowed_horizons", []),
        allowed_rebalance=campaign.get("allowed_rebalance", []),
        portfolio_types=campaign.get("portfolio_types", []),
        max_experiments=int(budget.get("maximum_experiments", 8)),
        max_llm_tokens=int(budget.get("maximum_llm_tokens", 300000)),
        cost_bps=float(budget.get("cost_bps", 5.0)),
        min_acceptance_sharpe=float(risk.get("min_acceptance_sharpe", 0.5)),
        max_drawdown=float(risk.get("max_drawdown", 0.40)),
        max_turnover=float(risk.get("max_turnover", 300.0)),
        min_positive_folds=float(risk.get("min_positive_folds", 0.5)),
        research_fraction=float(hpol.get("research_fraction", 0.7)),
        parameter_optimization=bool(budget.get("parameter_optimization", False)),
        anonymize_universe=bool(campaign.get("anonymize_universe", True)),
    )

    # Model TAK-ÇALIŞTIR: üretici + bağımsız eleştirmen config'ten kurulur
    gen_cfg = models["hypothesis_generator"]
    provider = make_provider(gen_cfg)
    critic = make_critic(models.get("quant_critic", {"provider": "dummy"}))

    # Literatür grounding (Doküman 4.3): web araması ile gerçek faktörleri çek.
    # Kampanya başına BİR KEZ (maliyet). models.yaml -> web_search: true ile açılır.
    literature: list[str] = []
    if gen_cfg.get("web_search") and hasattr(provider, "client"):
        from agents.literature import fetch_literature_mechanisms
        from orchestrator.loop import ANONYMOUS_UNIVERSE
        print("Literatür aranıyor (web_search, en fazla ~90 sn; olmazsa literatürsüz devam)...")
        # Anonimleştirme açıkken literatür ajanı da ticker/tarih GÖRMEZ.
        lit_universe = (ANONYMOUS_UNIVERSE if cfg.anonymize_universe
                        else campaign["universe_description"])
        literature = fetch_literature_mechanisms(
            provider.client, provider.model, lit_universe)
        for m in literature:
            print(f"  • {m[:100]}")
        print()

    # Veri: adaptörden (sentetik/gerçek config'ten) yüklenir, sonra araştırma /
    # KİLİTLİ holdout olarak bölünür. Kampanya yalnızca araştırma verisini görür.
    # Tarih aralığı tek kaynak: campaign.yaml (yfinance adaptörüne enjekte edilir).
    src = data_cfg.get("source")
    if src in ("yfinance", "sp500_pit"):
        data_cfg.setdefault(src, {})
        data_cfg[src]["start"] = str(campaign["start_date"])
        data_cfg[src]["end"] = str(campaign["end_date"])
    adapter = make_adapter(data_cfg)
    full = adapter.load()
    data, holdout_data = split_by_fraction(full, cfg.research_fraction)

    # DEVAM (varsayılan) veya SIFIRLA (--fresh). Devam: novelty/çoklu-test/öğrenme
    # koşular arası birikir; aynı hipotez tekrar üretilmez (Doküman: campaign = çok deney).
    if args.fresh:
        for p in (DB_PATH, HOLDOUT_DB):
            if os.path.exists(p):
                os.remove(p)
        print("(--fresh) Yeni kampanya: hafıza sıfırlandı.")
    memory = MemoryStore(DB_PATH)

    print(f"=== Kampanya: {campaign['name']} ===")
    print(f"Evren: {cfg.universe_description}")
    print(f"Sağlayıcı: {models['hypothesis_generator']['provider']} | "
          f"Bütçe: {cfg.max_experiments} deney\n")

    run_campaign(provider, data, memory, cfg, critic=critic, literature=literature)

    print("\n--- ÖZET ---")
    print(f"Toplam deney (multiple-testing muhasebesi): {memory.total_experiments()}")
    print(f"Karar dağılımı: {memory.summary_by_decision()}")
    print("\nLeaderboard (kabul edilenler, Sharpe'a göre):")
    for hid, title, sharpe, dd in memory.leaderboard():
        print(f"  {hid}  {title:32s}  Sharpe={sharpe:.2f}  MaxDD=%{(dd or 0)*100:.0f}")

    # Multiple testing raporu — "kabul" != "istatistiksel geçerli"
    backtested = memory.backtested_experiments()
    rows = build_report(backtested)
    print_report(rows, n_trials=len(backtested))

    # Holdout değerlendirmesi — araştırmadan AYRI, kilitli dönem, one-shot.
    policy = campaign.get("holdout_policy", {}) or {}
    max_cand = int(policy.get("maximum_candidates", 20))
    candidates = memory.accepted_hypotheses(limit=max_cand)
    if candidates:
        # Holdout audit KORUNUR (one-shot koşular arası): zaten değerlendirilmiş
        # aday tekrar test edilmez (Doküman 10.3).
        holdout = HoldoutService(holdout_data, audit_path=HOLDOUT_DB,
                                 max_candidates=max_cand,
                                 min_sharpe=cfg.min_acceptance_sharpe, cost_bps=cfg.cost_bps)
        print(f"\n=== HOLDOUT (kilitli dönem, one-shot, {len(candidates)} aday) ===")
        for hid, hjson, research_sharpe in candidates:
            hyp = HypothesisSpec.model_validate_json(hjson)
            try:
                res = holdout.evaluate(hyp)
            except HoldoutError:
                print(f"  {hid}  (zaten değerlendirildi — one-shot, atlandı)")
                continue
            flag = "GEÇTİ" if res.passed else "KALDI"
            print(f"  {hid}  araştırma Sharpe={research_sharpe:.2f} -> "
                  f"holdout Sharpe={res.sharpe:.2f}  [{flag}]")
        holdout.close()

    # Token/maliyet görünürlüğü (Doküman 17.3) — üretici + eleştirmen
    pt = getattr(provider, "total_prompt_tokens", 0) + getattr(critic, "total_prompt_tokens", 0)
    ct = getattr(provider, "total_completion_tokens", 0) + getattr(critic, "total_completion_tokens", 0)
    if pt or ct:
        print(f"\nToken kullanımı (üretici+critic): prompt={pt}, completion={ct}, toplam={pt+ct}")

    memory.close()

    # Research dashboard (tek dosya, offline) — hocaya göstermek için
    out = generate_dashboard(DB_PATH, HOLDOUT_DB, os.path.join(HERE, "dashboard.html"),
                             campaign_name=campaign["name"])
    print(f"\nDashboard: {out}")


if __name__ == "__main__":
    main()
