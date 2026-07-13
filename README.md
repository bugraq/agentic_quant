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
- [x] Pareto çok-amaçlı sıralama (Sharpe alt-sınır + drawdown + turnover) + muhafazakâr skor
- [x] Tam kampanya config'i (izin verilen alan/ufuk/operatör, bütçe, risk kısıtları) — hepsi koda bağlı
- [x] Kampanya kalıcılığı (varsayılan devam / `--fresh` sıfırla)
- [x] Motor-şema hizalaması — beyan edilen `trade_time`/`rebalance`/`holding_period`/
  `portfolio.type`/`weighting`/`gross_exposure` motor tarafından GERÇEKTEN uygulanır;
  uygulanamayan beyan static validator'da reddedilir (şema = çalıştırılan şey)
- [x] Getiriler düzeltilmiş fiyattan (temettü+split; adjusted_close, open'a faktör)
- [x] Optimizer denemeleri multiple-testing sayımında (her backtest = 1 deneme;
  `parameter_search` stage'i ile hafızaya yazılır) + min-fold muhafazakâr skor
- [x] market_cap placeholder'ı kaldırıldı (sahte size faktörünü önler)
- [x] LLM memorization önlemi — `anonymize_universe: true` iken prompta ticker/tarih
  gitmez (parametre-içi look-ahead kontrolü); `false` = ablation deneyi
- [x] Random-search baseline (Deney A) — `models.yaml -> provider: random`; aynı
  pipeline, aynı bütçe, ekonomik gerekçe yok; LLM'in katkısı ölçülebilir
- [x] Point-in-time S&P 500 evreni (survivorship düzeltmesi) — Wikipedia değişiklik
  tarihçesinden her tarihteki GERÇEK üye kümesi (`data/pit_universe.py`); pencerede
  üye olmuş ~700 ticker (bugün endekste olmayanlar dahil) indirilir; motor
  `index_membership` maskesiyle hisseyi yalnızca üye olduğu günlerde işleme sokar.
  Kalan dürüst sınırlar: Yahoo'da verisi hiç olmayan delist ticker'lar (yüklemede
  raporlanır) ve delisting return modeli yok — tam çözüm CRSP ister.
- [ ] Kripto adaptörü (ccxt)
- [ ] (opsiyonel) lineage grafiği görselleştirmesi

`main.py` çalışınca `dashboard.html` üretilir (offline, tarayıcıda aç). Ayrıca:
`python -m dashboard.report`

### Kampanya kalıcılığı (devam vs sıfırla) ve holdout
- **Varsayılan:** `python main.py` mevcut kampanyaya **DEVAM eder** — novelty, champion,
  dersler, çoklu-test sayımı koşular arası birikir; aynı hipotez tekrar üretilmez.
  Bir kampanya = çok deney (Doküman 4.1). Tekrar tekrar çalıştırıp büyütebilirsin.
- **Yeni kampanya:** `python main.py --fresh` hafızayı sıfırlar.
- **Holdout AYRI komuttur:** `python main.py --holdout` — kampanya koşusu kilitli
  döneme ASLA dokunmaz; kabul edilen adaylar ancak kampanya bitti kararıyla, bir kez
  (one-shot, audit log'lu) sınanır (Doküman 10.3 — insan-döngüsü sızıntısını da kapatır).
- Hafıza: `research_memory.sqlite` (episodic), `holdout_audit.sqlite` (one-shot log).

### LLM sağlayıcısı (esnek)
`configs/models.yaml` → `provider: openrouter|vllm|openai_compatible|dummy`. Hepsi
tek OpenAI-uyumlu istemci; geçiş = base_url + model + api_key ortam değişkeni.
API key `.env`'de (`OPENROUTER_API_KEY`), koda/log'a asla girmez.
