"""
Model karşılaştırma koşucusu (Doküman 15/26 — "LLM gerçekten daha iyi mi?").

Aynı kampanya kısıtları + aynı veri + aynı deney bütçesi altında birden çok
hipotez üreticisini (LLM'ler + random-search baseline) yarıştırır ve
ARAŞTIRMA VERİMLİLİĞİ tablosu üretir. Ölçülen şey "en iyi Sharpe" değil
(Doküman 26): kabul oranı, tekrar oranı, derleme hatası, çoklu-test sonrası
en iyi DSR, deney başına token maliyeti.

Kullanım:
    python compare.py                 # configs/compare.yaml'daki yarışmacılar

Adalet kuralları:
  - Her yarışmacıya AYRI, TAZE hafıza (runs/compare_<label>.sqlite).
  - Critic varsayılan dummy (deterministik) — tek değişken üretici olsun.
  - Literatür araması kapalı (koşular arası varyans katmasın).
  - Holdout'a ASLA dokunulmaz; karşılaştırma araştırma dönemi metrikleriyledir.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from evaluation import build_report
from llm import make_provider
from memory import MemoryStore
from orchestrator import run_campaign
from main import HERE, build_config, load_data, load_yaml


@dataclass
class ContestantResult:
    label: str
    total_records: int          # üretilen her şey (duplicate/hata dahil)
    accepts: int
    duplicates: int
    compile_errors: int
    backtested: int             # ham backtest sayısı (optimizer dahil)
    distinct: int               # tekil strateji (özdeş getiriler tekilleştirilmiş)
    best_accept_sharpe: float | None
    best_dsr: float | None
    fdr_survivors: int
    tokens: int
    llm_calls_per_accept: str   # okunabilir özet


def _metrics(label: str, memory: MemoryStore, provider) -> ContestantResult:
    stages = memory.stage_counts()
    decisions = memory.summary_by_decision()
    backtested = memory.backtested_experiments()
    rows = build_report(backtested)
    accepts = decisions.get("accept", 0)
    lb = memory.leaderboard(limit=1)
    tokens = (getattr(provider, "total_prompt_tokens", 0)
              + getattr(provider, "total_completion_tokens", 0))
    return ContestantResult(
        label=label,
        total_records=memory.total_experiments(),
        accepts=accepts,
        duplicates=decisions.get("duplicate", 0),
        compile_errors=stages.get("compile_error", 0),
        backtested=len(backtested),
        distinct=len(rows),
        best_accept_sharpe=(lb[0][2] if lb else None),
        best_dsr=(max((r.dsr for r in rows), default=None) if rows else None),
        fdr_survivors=sum(1 for r in rows if r.survives_fdr),
        tokens=tokens,
        llm_calls_per_accept=(f"{tokens/accepts:,.0f} token/kabul" if accepts and tokens
                              else ("-" if not tokens else "kabul yok")),
    )


def print_table(results: list[ContestantResult], budget: int) -> None:
    print(f"\n=== MODEL KARŞILAŞTIRMASI (deney bütçesi: {budget}/yarışmacı) ===")
    hdr = (f"{'yarışmacı':22s} {'kayıt':>6s} {'kabul':>6s} {'tekrar':>7s} "
           f"{'derl.hata':>9s} {'backtest':>9s} {'tekil':>6s} "
           f"{'en iyi Sharpe':>13s} {'en iyi DSR':>10s} {'FDR':>4s} {'token':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        sh = f"{r.best_accept_sharpe:.2f}" if r.best_accept_sharpe is not None else "-"
        dsr = f"{r.best_dsr:.2f}" if r.best_dsr is not None else "-"
        print(f"{r.label:22s} {r.total_records:6d} {r.accepts:6d} {r.duplicates:7d} "
              f"{r.compile_errors:9d} {r.backtested:9d} {r.distinct:6d} "
              f"{sh:>13s} {dsr:>10s} {r.fdr_survivors:4d} {r.tokens:9,d}")
    print("\nOkuma rehberi: 'kabul' tek başına başarı DEĞİL — DSR ve FDR'a bak "
          "(çoklu-test sonrası anlamlılık). 'tekrar' yüksekse model çeşitlilik "
          "üretemiyor; 'derl.hata' yüksekse şemaya uyamıyor. random-search alt "
          "çıtadır: LLM ondan iyi değilse LLM katkısı yok demektir (Doküman 26).")


def run_contestant(contestant: dict, data, cfg, critic, db_path: str) -> ContestantResult:
    """Tek yarışmacıyı taze hafızayla koştur, metrikleri topla."""
    label = contestant.get("label") or contestant.get("model") or contestant["provider"]
    if os.path.exists(db_path):
        os.remove(db_path)
    provider = make_provider(contestant)
    memory = MemoryStore(db_path)
    print(f"\n########  Yarışmacı: {label}  ########")
    run_campaign(provider, data, memory, cfg, critic=critic, literature=[])
    res = _metrics(label, memory, provider)
    memory.close()
    return res


def main() -> None:
    load_dotenv(os.path.join(HERE, ".env"))
    campaign = load_yaml("campaign.yaml")["campaign"]
    data_cfg = load_yaml("data.yaml")["data"]
    comp = load_yaml("compare.yaml")["compare"]
    cfg = build_config(campaign)

    out_dir = os.path.join(HERE, comp.get("output_dir", "runs"))
    os.makedirs(out_dir, exist_ok=True)

    # Critic: adalet için varsayılan dummy; istenirse models.yaml'daki kullanılır.
    if comp.get("critic") == "models_yaml":
        from llm import make_critic
        critic = make_critic(load_yaml("models.yaml")["models"].get("quant_critic", {}))
    else:
        from agents.quant_critic import DummyCritic
        critic = DummyCritic()

    # Veri BİR KEZ yüklenir; bütün yarışmacılar aynı araştırma dilimini görür.
    data, _holdout = load_data(campaign, data_cfg, cfg.research_fraction)

    results = []
    for contestant in comp["contestants"]:
        label = contestant.get("label", "?")
        db = os.path.join(out_dir, f"compare_{label}.sqlite")
        try:
            results.append(run_contestant(contestant, data, cfg, critic, db))
        except Exception as e:  # noqa: BLE001 — bir yarışmacı çökerse diğerleri koşsun
            print(f"[{label}] KOŞU HATASI: {type(e).__name__}: {str(e)[:200]}")

    if results:
        print_table(results, budget=cfg.max_experiments)
        # Makale/rapor için markdown çıktısı
        md_path = os.path.join(out_dir, "comparison.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("| yarışmacı | kayıt | kabul | tekrar | derleme hatası | "
                    "backtest | tekil | en iyi Sharpe | en iyi DSR | FDR | token |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
            for r in results:
                sh = f"{r.best_accept_sharpe:.2f}" if r.best_accept_sharpe is not None else "-"
                dsr = f"{r.best_dsr:.2f}" if r.best_dsr is not None else "-"
                f.write(f"| {r.label} | {r.total_records} | {r.accepts} | "
                        f"{r.duplicates} | {r.compile_errors} | {r.backtested} | "
                        f"{r.distinct} | {sh} | {dsr} | {r.fdr_survivors} | "
                        f"{r.tokens:,} |\n")
        print(f"\nMarkdown tablo: {md_path}")


if __name__ == "__main__":
    main()
