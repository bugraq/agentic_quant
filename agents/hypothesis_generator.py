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


_EXAMPLE = """{
  "hypothesis_id": "hyp_example",
  "title": "60-day cross-sectional momentum",
  "claim": "Past 60-day winners keep outperforming.",
  "family": "momentum",
  "economic_mechanism": {"type": "momentum", "description": "Underreaction to news.",
                          "expected_failure_conditions": ["sharp reversals"]},
  "universe": {"source": "sp500_point_in_time", "minimum_price": 5.0},
  "features": [],
  "signal": {"op": "cross_sectional_rank", "inputs": [
      {"op": "return", "window": 60, "inputs": [{"op": "field", "field": "close"}]}]},
  "portfolio": {"type": "cross_sectional_long_short", "long_quantile": 0.3,
                "short_quantile": 0.3, "weighting": "equal", "sector_neutral": false},
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
  - DÜRÜST ETİKET: title, claim ve family, sinyalin GERÇEKTE yaptığıyla uyuşmalı.
    'regime-conditioned' diyorsan sinyalde conditional/greater_than/volatilite
    OLMALI; 'composite' diyorsan birden çok sinyali birleştirmelisin. Sade
    momentum'a 'momentum' de, abartılı etiket yapıştırma (critic reddeder).
  - Şema tam olarak şu örnekteki gibi olmalı:

{_EXAMPLE}
"""


def _build_user_prompt(ctx: ResearchContext) -> str:
    priors = "\n".join(
        f"  - {e.title} [{e.family}] -> {e.outcome} ({e.headline_metric or '-'})"
        for e in ctx.prior_experiments
    ) or "  (henüz yok)"
    lessons = "\n".join(f"  - {l}" for l in ctx.lessons) or "  (henüz yok)"

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

    return f"""Kampanya hedefi: {ctx.campaign_goal}
Evren: {ctx.universe_description}
Üretim modu: {ctx.generation_mode.value}

DERSLER (geçmiş deneylerden — bunlara UY):
{lessons}

Daha önce denenen hipotezler (aynısını tekrarlama):
{priors}

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
        self._counter += 1
        hid = f"hyp_{self._counter:04d}"
        system = _build_system_prompt(context)
        user = _build_user_prompt(context)

        resp = self.client.chat(self.model, system, user,
                                temperature=self.temperature, max_tokens=self.max_tokens)
        self._track(resp)
        self._set_meta(system, user, resp)

        try:
            return self._parse(resp.text, hid)
        except (LLMGenerationError, ValueError, json.JSONDecodeError) as e:
            # Bir kez düzeltme iste (Doküman 17.2)
            repair_user = (f"{user}\n\nÖnceki çıktın geçersizdi. Hata: {e}\n"
                           f"Şemaya birebir uyan, SADECE geçerli JSON döndür.")
            resp2 = self.client.chat(self.model, system, repair_user,
                                     temperature=0.2, max_tokens=self.max_tokens)
            self._track(resp2)
            self._set_meta(system, repair_user, resp2)
            return self._parse(resp2.text, hid)

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
