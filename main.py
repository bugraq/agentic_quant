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

import os

import yaml
from dotenv import load_dotenv

from contracts.hypothesis_spec import HypothesisSpec
from data import gen_cross_sectional_momentum
from data.synthetic import split_by_fraction
from evaluation import build_report, print_report
from holdout import HoldoutService
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
    load_dotenv(os.path.join(HERE, ".env"))   # API key'i ortama yükle (koda girmez)
    campaign = load_yaml("campaign.yaml")["campaign"]
    models = load_yaml("models.yaml")["models"]

    cfg = CampaignConfig(
        goal=campaign["goal"],
        universe_description=campaign["universe_description"],
        allowed_operators=campaign.get("allowed_operators", []),
        max_experiments=campaign["maximum_experiments"],
        cost_bps=float(campaign.get("cost_bps", 5.0)),
        min_acceptance_sharpe=float(campaign.get("min_acceptance_sharpe", 0.5)),
    )

    # Model TAK-ÇALIŞTIR: üretici + bağımsız eleştirmen config'ten kurulur
    provider = make_provider(models["hypothesis_generator"])
    critic = make_critic(models.get("quant_critic", {"provider": "dummy"}))

    # Veri: araştırma / KİLİTLİ holdout olarak bölünür. Kampanya yalnızca
    # araştırma verisini görür; holdout ayrı serviste kalır (Doküman 2.2).
    full = gen_cross_sectional_momentum(n_sec=20, n_days=750, seed=7)
    data, holdout_data = split_by_fraction(full, research_frac=0.7)

    # Temiz başlangıç için eski hafızayı sil
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    memory = MemoryStore(DB_PATH)

    print(f"=== Kampanya: {campaign['name']} ===")
    print(f"Evren: {cfg.universe_description}")
    print(f"Sağlayıcı: {models['hypothesis_generator']['provider']} | "
          f"Bütçe: {cfg.max_experiments} deney\n")

    run_campaign(provider, data, memory, cfg, critic=critic)

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
        if os.path.exists(HOLDOUT_DB):
            os.remove(HOLDOUT_DB)
        holdout = HoldoutService(holdout_data, audit_path=HOLDOUT_DB,
                                 max_candidates=max_cand,
                                 min_sharpe=cfg.min_acceptance_sharpe, cost_bps=cfg.cost_bps)
        print(f"\n=== HOLDOUT (kilitli dönem, one-shot, {len(candidates)} aday) ===")
        for hid, hjson, research_sharpe in candidates:
            hyp = HypothesisSpec.model_validate_json(hjson)
            res = holdout.evaluate(hyp)
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


if __name__ == "__main__":
    main()
