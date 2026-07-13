"""
Hypothesis Generator ajanı — ResearchContext'i prompta çevirir, LLM'i çağırır,
çıktıyı HypothesisSpec'e parse eder.

'Varlıkları prompta indirgeme' burada olur: LLM ham veri görmez, yalnızca
universe_description metnini görür. Çıktı serbest metin değil, şemaya uygun
JSON'dur; geçersizse bir kez düzeltme istenir, yine olmazsa reddedilir.

DSL sınırlaması hipotez uzayını kısıtlar: LLM yalnızca izin verilen
operatörlerden ağaç kurabilir.
"""
from __future__ import annotations

import hashlib
import json

from contracts.hypothesis_spec import HypothesisFamily, HypothesisSpec
from contracts.research_context import ResearchContext
from dsl.operators import DATA_FIELDS, REGISTRY, get_operator
from llm.openai_client import OpenAICompatibleClient


class LLMGenerationError(Exception):
    """LLM geçerli bir HypothesisSpec üretemedi."""


# ZENGİN örnek: iki farklı sinyali BİRLEŞTİREN composite (hacimle-teyitli reversal).
# features bloğunda adlandırılmış sinyaller kurulur, signal'de çarpılır.
_EXAMPLE = """{
  "hypothesis_id": "hyp_example",
  "title": "Volume-confirmed short-term reversal",
  "claim": "Recent losers with abnormally high volume tend to reverse over the next days.",
  "family": "composite",
  "economic_mechanism": {"type": "behavioral_reversal",
                          "description": "Panic selling on high volume overshoots fair value.",
                          "expected_failure_conditions": ["information-driven selloffs"]},
  "universe": {"source": "sp500_point_in_time", "minimum_price": 5.0},
  "features": [
    {"name": "reversal_5d", "expression":
      {"op": "negate", "inputs": [
        {"op": "return", "window": 5, "inputs": [{"op": "field", "field": "close"}]}]}},
    {"name": "abnormal_volume", "expression":
      {"op": "zscore", "window": 60, "inputs": [{"op": "field", "field": "volume"}]}}
  ],
  "signal": {"op": "cross_sectional_rank", "inputs": [
    {"op": "multiply", "inputs": [
      {"op": "feature_ref", "name": "reversal_5d"},
      {"op": "feature_ref", "name": "abnormal_volume"}]}]},
  "portfolio": {"type": "cross_sectional_long_short", "long_quantile": 0.2,
                "short_quantile": 0.2, "weighting": "equal", "sector_neutral": false},
  "execution": {"signal_time": "close_t", "trade_time": "open_t_plus_1",
                "holding_period_days": 5, "rebalance": "daily"},
  "falsification": {"minimum_oos_sharpe": 0.5, "maximum_turnover": 30.0,
                    "maximum_drawdown": 0.25, "minimum_positive_walk_forward_folds": 0.6}
}"""


def _operator_reference(allowed: list[str]) -> str:
    names = allowed or list(REGISTRY.keys())
    lines = []
    for name in names:
        spec = get_operator(name)
        if spec is None:
            continue
        w = " (window gerekli)" if spec.needs_window else ""
        lines.append(f"  - {name}: arite {spec.min_arity}..{spec.max_arity}{w}")
    return "\n".join(lines)


def _build_system_prompt(ctx: ResearchContext) -> str:
    return f"""Sen deneyimli bir kantitatif araştırmacısın. Görevin, verilen evren
için test edilebilir bir ekonomik hipotez üretmek ve onu KATI bir DSL şemasında
ifade etmek. SADECE geçerli JSON döndür — açıklama, markdown, kod bloğu YOK.

DSL yaprakları:
  - {{"op": "field", "field": <alan>}}  — izin verilen alanlar: {sorted(ctx.allowed_fields) or sorted(DATA_FIELDS)}
  - {{"op": "const", "value": <sayı>}}
  - {{"op": "feature_ref", "name": <feature_adı>}}
İzin verilen operatörler (op + inputs listesi, gerekirse window):
{_operator_reference(ctx.allowed_operators)}

İzin verilen family değerleri (SADECE bunlardan biri):
  {[f.value for f in HypothesisFamily]}

Kampanya kısıtları (bunlara UY):
  - Pencere/ufuk (window) SADECE şunlardan: {ctx.allowed_horizons or 'serbest'}
  - execution.rebalance SADECE şunlardan: {ctx.allowed_rebalance or 'serbest'}
  - portfolio.type SADECE şunlardan: {ctx.allowed_portfolio_types or 'serbest'}

KRİTİK KURALLAR (yoksa hipotez reddedilir):
  - signal kesitsel bir ifade olmalı; genelde en dışta cross_sectional_rank kullan.
  - VERİ SIZINTISI YASAK: execution.signal_time = "close_t",
    execution.trade_time = "open_t_plus_1" olmalı (asla close_t'de işlem yok).
  - Gelecek bilgisi kullanma; tüm operatörler geçmişe bakar.
  - falsification.minimum_oos_sharpe gerçekçi olsun (0.3–0.8 arası).
  - TURNOVER KONTROLÜ: aşırı sık işlemden kaçın; çok kısa pencereler (1–3 gün)
    turnover'ı patlatır. Daha uzun pencereler (20+) tercih et.
  - volatility, rolling_std, zscore, return vb. OPERATÖRDÜR — veri alanı DEĞİL.
    Girdi olarak bir alan alırlar: {{"op":"volatility","window":20,"inputs":[
    {{"op":"field","field":"close"}}]}}. 'volatility'yi field olarak KULLANMA.
  - BİRİM UYARISI (ölü koşul tuzağı): rolling_std(close) FİYAT ölçeğindedir
    (dolar; büyük hissede onlarca dolar) — onu 0.02 gibi getiri-ölçeğinde bir
    sabitle karşılaştırırsan koşul HİÇ tetiklenmez ve hipotez reddedilir.
    Volatilite eşiği için volatility operatörünü (getiri bazlı) kullan; daha da
    iyisi sabit eşik yerine KESİTSEL karşılaştırma kur (örn. quantile/
    cross_sectional_rank ile "evrenin medyanından yüksek volatilite").
  - conditional TAM 3 girdi alır ve sırası şudur: [koşul, koşul-doğruysa-değer,
    koşul-yanlışsa-değer]. Örn. yüksek volatilitede reversal, düşükte momentum:
    {{"op":"conditional","inputs":[
      {{"op":"greater_than","inputs":[<volatilite>, <eşik>]}},
      <reversal_sinyali>, <momentum_sinyali>]}}. 2 girdi verme — derleme hatası olur.
  - greater_than / less_than / multiply / subtract TAM 2 girdi alır.
  - DÜRÜST ETİKET: title, claim ve family, sinyalin GERÇEKTE yaptığıyla uyuşmalı.
    'regime-conditioned' diyorsan sinyalde conditional/greater_than/volatilite
    OLMALI; 'composite' diyorsan birden çok sinyali birleştirmelisin. Sade
    momentum'a 'momentum' de, abartılı etiket yapıştırma (critic reddeder).
  - ZENGİNLİK ve ÇEŞİTLİLİK (önemli): Yalnızca cross_sectional_rank(return(N)) gibi
    TEK-faktörlü basit yapılar üretme. Mümkün olduğunda features bloğunda 2+ FARKLI
    sinyal kur ve anlamlı biçimde BİRLEŞTİR (multiply/subtract/conditional).
    Örnek fikirler: momentum × düşük-volatilite, hacimle-teyitli reversal,
    rejim-koşullu (yüksek/düşük volatilitede farklı davranan) strateji,
    volatilite-ayarlı momentum, likidite-filtreli değer. Farklı ekonomik
    mekanizmalar dene — her tur aynı momentum'u tekrarlama.
  - Şema tam olarak şu örnekteki gibi olmalı:

{_EXAMPLE}
"""


def _build_user_prompt(ctx: ResearchContext) -> str:
    priors = "\n".join(
        f"  - {e.title} [{e.family}] -> {e.outcome} ({e.headline_metric or '-'})"
        for e in ctx.prior_experiments
    ) or "  (henüz yok)"
    lessons = "\n".join(f"  - {l}" for l in ctx.lessons) or "  (henüz yok)"
    literature = "\n".join(f"  - {m}" for m in ctx.literature_mechanisms)

    # Inversion: başarısız hipotezi ters çevir. Revision: champion'ı geliştir. Yeni: keşfet.
    if ctx.generation_mode.value == "inversion" and ctx.parent_hypothesis is not None:
        parent_json = ctx.parent_hypothesis.model_dump_json(indent=0)
        task = f"""GÖREV — TERS ÇEVİRME (inversion): Aşağıdaki hipotez NEGATİF Sharpe ile
başarısız oldu. Sinyalin YÖNÜNÜ ters çevir (örn. en dışa negate ekle ya da long/short
mantığını çevir) — ters yön kazanıyor olabilir. claim ve title'ı da tersine güncelle.

BAŞARISIZ HİPOTEZ:
{parent_json}"""
    elif ctx.generation_mode.value == "revision" and ctx.parent_hypothesis is not None:
        parent_json = ctx.parent_hypothesis.model_dump_json(indent=0)
        task = f"""GÖREV — REVİZYON: Aşağıdaki en iyi hipotezi (champion) GELİŞTİR.
Çalışan temel yapıyı KORU, TEK bir şeyi anlamlı biçimde değiştir (örn. pencere
uzunluğunu artır, nötralizasyon ekle, falsification eşiğini gerçekçileştir).
Aynısını kopyalama; ölçülebilir bir iyileştirme hedefle.

CHAMPION:
{parent_json}"""
    elif ctx.suggested_family:
        task = (f"GÖREV — YENİ: Araştırma bütçesi bu tur '{ctx.suggested_family}' "
                f"ailesine ayrıldı. Bu aileye UYAN GERÇEK bir mekanizma kur — sinyal "
                f"gerçekten o aileyi uygulamalı (etiket-sinyal uyuşmalı). Örn. "
                f"'regime_conditioned' için conditional/volatilite kullan. Eğer bu "
                f"aileyi dürüstçe uygulayamıyorsan, sinyaline UYAN family'yi seç.")
    else:
        task = """GÖREV — YENİ: Derslere göre UMUT VERİCİ aileyi seç ya da hiç
denenmemiş bir yön keşfet. ZAYIF aileleri (dersteki uyarılar) tekrarlama."""

    lit_block = (f"\nLİTERATÜR (web'den gerçek faktörler — bunlardan İLHAM al, "
                 f"DSL ile uygula):\n{literature}\n" if literature else "")

    # Duplicate geri bildirimi: bu slotta üretilenler tekrar çıktı; LLM'in
    # AYNI yapıyı bir kez daha döndürmesini açıkça yasakla.
    dup_block = ""
    if ctx.duplicate_feedback:
        dups = "\n".join(f"  - {d}" for d in ctx.duplicate_feedback)
        dup_block = (f"\n⚠ BU TURDA ÜRETTİKLERİN TEKRAR ÇIKTI (kabul edilmedi):\n{dups}\n"
                     f"Bunlarla ve önceki denemelerle YAPISAL olarak farklı bir strateji "
                     f"üret: farklı operatör kombinasyonu, farklı veri alanı veya farklı "
                     f"aile dene. Sadece pencere/başlık değiştirmek YETMEZ.\n")

    return f"""Kampanya hedefi: {ctx.campaign_goal}
Evren: {ctx.universe_description}
Üretim modu: {ctx.generation_mode.value}
{lit_block}
DERSLER (geçmiş deneylerden — bunlara UY):
{lessons}

Daha önce denenen hipotezler (aynısını tekrarlama):
{priors}
{dup_block}
{task}

Yukarıdaki şemaya uygun, geçerli bir hipotez JSON'u üret."""


def _extract_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        raise LLMGenerationError(f"Çıktıda JSON bulunamadı: {text[:200]}")
    return json.loads(s[start:end + 1])


class LLMHypothesisProvider:
    """HypothesisProvider arayüzü: gerçek LLM ile hipotez üretir."""

    def __init__(self, client: OpenAICompatibleClient, model: str,
                 temperature: float = 0.9, max_tokens: int = 4000) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._counter = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_meta: dict = {}   # son çağrının reproducibility metadata'sı

    def _set_meta(self, system: str, user: str, resp) -> None:
        self.last_meta = {
            "model_name": resp.model or self.model,
            "temperature": self.temperature,
            "prompt_hash": hashlib.sha256((system + user).encode()).hexdigest()[:16],
            "output_hash": hashlib.sha256(resp.text.encode()).hexdigest()[:16],
        }

    def next(self, context: ResearchContext) -> HypothesisSpec:
        from dsl import CompileError, compile_hypothesis

        self._counter += 1
        hid = f"hyp_{self._counter:04d}"
        system = _build_system_prompt(context)
        user = _build_user_prompt(context)

        resp = self.client.chat(self.model, system, user,
                                temperature=self.temperature, max_tokens=self.max_tokens)
        self._track(resp)
        self._set_meta(system, user, resp)

        try:
            hyp = self._parse(resp.text, hid)
            compile_hypothesis(hyp)   # şema geçerli ama DERLENEMEZ olabilir (arite vb.)
            return hyp
        except (LLMGenerationError, ValueError, json.JSONDecodeError, CompileError) as e:
            # Bir kez düzeltme iste (Doküman 17.2) — şema VEYA derleme hatası.
            # (Gerçek koşuda görüldü: arite hatası onarımsız çöpe gidiyordu.)
            repair_user = (f"{user}\n\nÖnceki çıktın geçersizdi. Hata: {e}\n"
                           f"Şemaya ve operatör aritelerine birebir uyan, "
                           f"SADECE geçerli JSON döndür.")
            resp2 = self.client.chat(self.model, system, repair_user,
                                     temperature=0.2, max_tokens=self.max_tokens)
            self._track(resp2)
            self._set_meta(system, repair_user, resp2)
            hyp = self._parse(resp2.text, hid)
            try:
                compile_hypothesis(hyp)
            except CompileError:
                # Onarım da derlenemedi: yine de döndür — loop derleme aşamasında
                # yakalayıp compile_error olarak KAYDEDER (her deney kaydedilir).
                pass
            return hyp

    def _parse(self, text: str, hid: str) -> HypothesisSpec:
        data = _extract_json(text)
        data["hypothesis_id"] = hid   # kimliği biz atarız (tekillik garantisi)
        try:
            return HypothesisSpec.model_validate(data)
        except Exception as e:
            raise LLMGenerationError(f"Şema doğrulaması başarısız: {e}") from e

    def _track(self, resp) -> None:
        self.total_prompt_tokens += resp.prompt_tokens
        self.total_completion_tokens += resp.completion_tokens
