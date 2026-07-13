# LLM Tabanlı Otonom Quant Araştırmacısı — Proje Özeti

## Bir cümlede
LLM'in finansal araştırma sürecini **kapalı bir döngüde** otomatikleştiren bir
sistem: hipotez üret → stratejiye derle → sızıntısız backtest et → istatistiksel
değerlendir → kabul/red/geliştir → hafızaya yaz → öğren → yeni hipotez. Çıktısı
yalnızca bir strateji değil, **tekrar üretilebilir bir araştırma kaydı**.

## Temel tasarım kararı (sistemin kalbi)
Üç katman kesin biçimde ayrıldı:

| Katman | Sorumluluk |
|---|---|
| **LLM** | Hipotez, ekonomik mekanizma, yapısal strateji değişikliği, yorum |
| **Deterministik sistem** | Veri erişimi, derleme, backtest, metrik, istatistik, holdout |
| **Sayısal optimizasyon** | Bandit ile bütçe tahsisi, parametreler |

LLM **asla** backtest'e veya veriye dokunmaz; serbest Python yazmaz. Sadece
önceden onaylanmış bir **DSL** (domain-specific language) ile yapılandırılmış
strateji tanımı üretir. Bu ayrım; veri sızıntısını, reward hacking'i ve
tekrar-üretilemezliği baştan engeller.

## Araştırma döngüsü (pipeline)
Her hipotez şu istasyonlardan geçer; herhangi birinde elenebilir:

```
LLM üretir → DSL'e derlenir → SIZINTI kontrolü → bağımsız Critic (ekonomik) →
tekrar (novelty) kontrolü → walk-forward backtest → hard gate (+fold tutarlılık) →
sağlamlık testleri → hafıza → öğrenme → bandit bütçe → çoklu-test → HOLDOUT
```

Öne çıkan güvence mekanizmaları:

- **Sızıntısızlık** — Her ifadeye "bu bilgi en erken ne zaman bilinebilir"
  (info_tick) etiketi atanır; `sinyal < işlem zamanı` eşitsizliği zorlanır.
  Sızıntı "test edilerek" değil, DSL'de **ifade edilemez kılınarak** önlenir.
- **Bağımsız Critic** — Üreten LLM kendi stratejisini onaylamaz; ayrı bir LLM
  ekonomik mekanizmayı denetler ("gizli bir faktörün yeniden adı mı?").
- **Multiple testing** — Binlerce backtest sonrası tek yüksek Sharpe anlamsızdır.
  Deflated Sharpe Ratio + FDR + bootstrap ile "kabul" ile "istatistiksel geçerli"
  ayrılır. Her deney (başarısız dahil) sayılır.
- **Kilitli Holdout** — Araştırmadan tamamen ayrı, one-shot, audit log'lu son sınav.
  Araştırma ajanı bu veriye asla erişemez.
- **Öğrenme + bandit** — Sistem geçmiş deneylerden ders çıkarır (hangi aile
  çalışıyor) ve araştırma bütçesini başarılı ailelere Thompson sampling ile dağıtır.

## En güçlü kanıt: dürüstlük
- **Sentetik veride** (bilinen bir sinyal gömülü) sistem sinyali **buluyor**.
- **Gerçek S&P 500 verisinde** basit stratejilerin hepsi eleniyor — sistem
  dürüstçe "kolay alpha yok" diyor, **sahte bir kazanan uydurmuyor**.

Kötü tasarlanmış bir sistem gerçek veride de "alpha buldum" diye kendini
kandırır. Bu sistemin gerçek veride negatif sonuç üretmesi, tasarımının
sağlam olduğunun kanıtıdır.

## Modülerlik (pipeline önce, model tak-çalıştır)
Her şey config'ten sürülür, kod değişmez:
- **LLM sağlayıcısı** — `configs/models.yaml` (OpenRouter bugün, vLLM yarın; aynı kod;
  `provider: random` = LLM'siz random-search baseline, Deney A)
- **Veri kaynağı** — `configs/data.yaml` (sentetik ↔ yfinance ↔ **point-in-time S&P 500**)
- **Kampanya sınırları** — `configs/campaign.yaml` (bütçe, eşikler, operatörler)

## Survivorship düzeltmesi (point-in-time evren)
Bugünün endeks listesini geçmişe uygulamak, "kazananlarla backtest" demektir.
Sistem artık Wikipedia'nın S&P 500 değişiklik tarihçesinden **her tarihteki
gerçek üye kümesini** kurar (2015-2023 penceresinde ~700 farklı ticker; bugün
endekste olmayan ~150 isim dahil) ve bir hisse yalnızca **o tarihte üyeyken**
işlem görebilir. Kalan dürüst sınırlar belgelidir: Yahoo'da verisi hiç olmayan
delist ticker'lar yüklemede raporlanır; delisting return modellenmez (CRSP yok).

## Teknik durum
- ~30 modül, **11 test paketi** (sızıntı, backtest, istatistik, holdout, critic,
  bandit, dashboard...), hepsi geçiyor.
- Kampanya maliyeti ~birkaç sent (gpt-4o-mini). GitHub'da versiyonlu.

---

## Nasıl çalıştırılır (kendin dene)

Terminalde proje klasöründe (`agentic_quant`):

**1. Kurulum (bir kez):**
```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**2. API anahtarı** — `.env` dosyası (zaten var):
```
OPENROUTER_API_KEY=sk-or-...
```

**3. Kampanyayı çalıştır:**
```
.\.venv\Scripts\python.exe main.py
```
İlk çalıştırma point-in-time S&P 500 verisini indirir (~700 ticker, birkaç
dakika); sonraki koşular `data/` altındaki cache'ten saniyeler içinde açılır.
Terminalde her hipotezin kararını, leaderboard'u, multiple-testing raporunu ve
holdout sonuçlarını görürsün.

**4. Dashboard'u aç:**
Çalışma bitince proje klasöründe **`dashboard.html`** oluşur.
Üstüne çift tıkla → tarayıcıda açılır. (Funnel, leaderboard, istatistik, holdout,
aile performansı görsel olarak.) Ayrıca tek başına:
```
.\.venv\Scripts\python.exe -m dashboard.report
```

**5. Testleri çalıştır:**
```
.\.venv\Scripts\python.exe -m tests.test_leakage
.\.venv\Scripts\python.exe -m tests.test_backtest
# (tests/ altındaki her test_*.py aynı şekilde)
```

## Ayarlarla oynamak
- **Gerçek veriye geç:** `configs/data.yaml` içinde `source: yfinance` yap.
- **Modeli değiştir:** `configs/models.yaml` içinde `model:` satırını değiştir
  (ücretsiz denemek için `:free` biten modeller).
- **Deney sayısı / eşikler:** `configs/campaign.yaml`.
