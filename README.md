# Agentic Quant — LLM Tabanlı Otonom Quant Araştırmacısı

Finansal araştırma sürecini kapalı bir döngüde otomatikleştiren sistem:
hipotez üret → stratejiye derle → sızıntısız backtest et → istatistiksel
değerlendir → kabul/red/geliştir → hafızaya yaz → yeni hipotez.

## Temel ilke (3 katman kesin ayrı)
- **LLM** → hipotez, ekonomik mekanizma, yapısal değişiklik, yorum
- **Deterministik sistem** → veri, derleme, backtest, metrik, istatistik, holdout
- **Sayısal optimizasyon** → sürekli parametreler (pencere, eşik, ağırlık)

LLM asla backtest/veriye dokunmaz; serbest Python yazmaz, sadece onaylı bir
DSL ile yapısal strateji tanımı üretir.

## Pipeline akışı (contract'lar)
```
ResearchContext -> [LLM] -> HypothesisSpec -> [Compiler] -> StrategyGraph
-> [Backtest] -> BacktestResult -> [Gate+Critic] -> Decision -> Memory
```

## Yapı
```
contracts/    # istasyonlar arası akan veri objeleri (Pydantic)  ← ŞU AN BURADAYIZ
configs/      # kampanya, model card, veri, değerlendirme (YAML)
llm/          # LLM soyutlaması (anthropic / vllm / dummy) — değiştirilebilir
agents/       # LLM'i kullanan roller (hypothesis generator, critic)
dsl/          # operatörler + compiler + static_validator (sızıntı kontrolü)
data/         # asset-class adaptörü (sp500 / crypto), point-in-time
backtest/     # motor, portföy, maliyet, execution, walk-forward
evaluation/   # hard gate, robustness, istatistik (FDR/Deflated Sharpe), pareto
memory/       # episodic/semantic/procedural + similarity (tekrar kontrolü)
orchestrator/ # döngünün kendisi (basit Python loop)
```

## Kurulum
```
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m tests.test_contracts_smoke   # duman testi
```

## Yol haritası (walking skeleton)
Bütün döngüyü önce en aptal haliyle uçtan uca çalıştır, sonra kutuları
tek tek `dummy → gerçek` yap. Baştan gerçek yapılacak iki şey:
**static validator (sızıntı)** ve **reproducibility (seed + versiyon)**.

- [x] Contract'lar (Pydantic modelleri) + duman testi
- [x] DSL operatör kaydı + compiler + static validator (sızıntı kontrolü) — 8 leakage testi geçiyor
- [x] Sentetik veri + vectorized backtest (tek fold) — bilinen sinyali buluyor, sahte alpha yok, property testleri geçiyor
- [x] Hard gate + SQLite hafıza (her deney kaydediliyor)
- [x] Orchestrator loop (dummy LLM) — **iskelet uçtan uca dönüyor** (`python main.py`)
- [x] Gerçek LLM (OpenRouter/OpenAI-uyumlu, `models.yaml` provider switch) — **otonom döngü çalışıyor**
- [x] Memory-güdümlü öğrenme — semantic memory (aile bazında ders), champion/revision modu; LLM momentum'a kilitleniyor, champion'ı geliştiriyor, leaderboard doluyor
- [x] Similarity/novelty (duplicate kontrolü) — yapısal (AST token, backtest öncesi) + davranışsal (işaretli korelasyon); tekrarları eleyip bütçe koruyor (Doküman 14)
- [x] İstatistiksel yönetişim — Deflated Sharpe Ratio + PSR + bootstrap CI + Benjamini-Hochberg FDR (multiple testing raporu; "kabul" != "istatistiksel geçerli")
- [x] Walk-forward (çoklu fold, tutarlılık) + robustness (permutation testi, maliyet 2x, parametre perturbasyonu)
- [x] Holdout servisi (LLM'den ayrı, kilitli dönem, one-shot, audit log, aday kotası)
- [x] Critic ajanı (bağımsız LLM, farklı prompt+düşük sıcaklık, ekonomik mekanizma denetimi)
- [x] DataAdapter (sentetik <-> gerçek tak-çalıştır); yfinance ile gerçek S&P 500 (survivorship uyarısıyla)
- [x] Bandit bütçe tahsisi (Thompson sampling) — araştırma bütçesini aileler arasında başarıya göre dağıtır
- [x] Research dashboard — funnel, leaderboard, hipotez detayı, multiple-testing, holdout, aile perf., soy ağacı
- [x] Reproducibility metadata (model, prompt/output hash, seed) — her deney yeniden üretilebilir (Doküman 17.3/25.5)
- [x] Lineage (soy ağacı) + inversion modu (başarısızı ters çevir) + universe filtreleri + ek-gecikme robustness
- [ ] Kripto adaptörü (ccxt), point-in-time endeks üyeliği, delisting
- [ ] (opsiyonel) Pareto çok-amaçlı sıralama, lineage grafiği

`main.py` çalışınca `dashboard.html` üretilir (offline, tarayıcıda aç). Ayrıca:
`python -m dashboard.report`

### LLM sağlayıcısı (esnek)
`configs/models.yaml` → `provider: openrouter|vllm|openai_compatible|dummy`. Hepsi
tek OpenAI-uyumlu istemci; geçiş = base_url + model + api_key ortam değişkeni.
API key `.env`'de (`OPENROUTER_API_KEY`), koda/log'a asla girmez.
