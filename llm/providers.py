"""
LLM sağlayıcıları — değiştirilebilir kutu.

Arayüz sabittir: HypothesisProvider.next(context) -> HypothesisSpec.
İçi config'ten seçilir (dummy / anthropic / vllm). Bu iskelette yalnızca
DummyProvider gerçektir; sabit bir hipotez katalogundan üretir. Gerçek
sağlayıcılar modeli çağırıp çıktıyı HypothesisSpec şemasına parse edecek
(Doküman 17.2: JSON repair -> schema validate -> bir kez düzeltme -> red).
"""
from __future__ import annotations

from typing import Protocol

from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from contracts.research_context import ResearchContext


class HypothesisProvider(Protocol):
    def next(self, context: ResearchContext) -> HypothesisSpec: ...


# ---- Yardımcı kısayollar -------------------------------------------------
def _field(name: str) -> Expression:
    return Expression(op="field", field=name)


def _ret(w: int) -> Expression:
    return Expression(op="return", window=w, inputs=[_field("close")])


def _cs_rank(inner: Expression) -> Expression:
    return Expression(op="cross_sectional_rank", inputs=[inner])


def _spec(hid, title, claim, fam, signal, trade_time="open_t_plus_1",
          min_sharpe=0.5) -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id=hid, title=title, claim=claim, family=fam,
        economic_mechanism=EconomicMechanism(type=fam.value, description=claim),
        universe=Universe(source="sp500_point_in_time", minimum_price=5.0),
        features=[], signal=signal,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time=trade_time,
                            holding_period_days=1),
        falsification=Falsification(minimum_oos_sharpe=min_sharpe))


# ---- Dummy katalog (LLM'i taklit eder) ----------------------------------
def _catalog() -> list[HypothesisSpec]:
    return [
        # 1) Geçerli momentum — momentum verisinde geçmeli
        _spec("hyp_001", "60g kesitsel momentum",
              "Geçmiş 60 gün kazananları kazanmaya devam eder.",
              HypothesisFamily.momentum, _cs_rank(_ret(60))),
        # 2) Geçerli reversal
        _spec("hyp_002", "1g kısa vadeli reversal",
              "Dünün kaybedenleri toparlanır.",
              HypothesisFamily.reversal,
              _cs_rank(Expression(op="negate", inputs=[_ret(1)]))),
        # 3) SIZINTILI — close_t sinyal + close_t execution (validator yakalamalı)
        _spec("hyp_003", "Sızıntılı momentum (aynı bar)",
              "Aynı bar bilgisiyle aynı barda işlem.",
              HypothesisFamily.momentum, _cs_rank(_ret(20)),
              trade_time="close_t"),
        # 4) Zayıf/çürütülecek — çok yüksek eşik taahhüdü, gate çürütür
        _spec("hyp_004", "20g momentum, agresif eşik",
              "20g momentum, min Sharpe 3.0 taahhüdüyle.",
              HypothesisFamily.momentum, _cs_rank(_ret(20)), min_sharpe=3.0),
    ]


class DummyProvider:
    """Sabit katalogdan sırayla hipotez üretir (deterministik)."""

    def __init__(self) -> None:
        self._catalog = _catalog()
        self._i = 0
        self.last_meta = {"model_name": "dummy", "temperature": None,
                          "prompt_hash": None, "output_hash": None}

    def next(self, context: ResearchContext) -> HypothesisSpec:
        spec = self._catalog[self._i % len(self._catalog)]
        self._i += 1
        return spec.model_copy(deep=True)


# OpenAI-uyumlu sağlayıcılar için varsayılanlar (base_url + api_key ortam değişkeni)
_OPENAI_COMPATIBLE_DEFAULTS = {
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "api_key_env": "OPENROUTER_API_KEY"},
    "vllm": {"base_url": "http://localhost:8000/v1", "api_key_env": "VLLM_API_KEY"},
    "openai_compatible": {"base_url": "http://localhost:8000/v1",
                          "api_key_env": "OPENAI_API_KEY"},
}


def make_provider(config: dict) -> HypothesisProvider:
    """configs/models.yaml'daki 'provider' alanına göre sağlayıcı kur.

    dummy    -> sabit katalog (LLM yok)
    random   -> random-search baseline (LLM yok, ekonomik gerekçe yok; Deney A)
    openrouter / vllm / openai_compatible -> tek OpenAI-uyumlu istemci;
        aralarındaki tek fark base_url + api_key ortam değişkeni. vLLM'e geçiş
        = models.yaml'da provider'ı değiştir + endpoint ver. Kod değişmez.
    """
    config = config or {}
    provider = config.get("provider", "dummy")
    if provider == "dummy":
        return DummyProvider()

    if provider == "random":
        # Random-search baseline (Deney A): LLM'siz, ekonomik gerekçesiz,
        # aynı pipeline'dan geçen rastgele hipotez üreteci. Karşılaştırma için.
        from baselines import RandomHypothesisProvider
        return RandomHypothesisProvider(seed=int(config.get("seed", 0)))

    if provider in ("gp", "genetic"):
        # Genetic-programming baseline (Deney A): fitness'a göre evrimleşen DSL
        # ağaçları (crossover+mutasyon), LLM'siz/gerekçesiz. Random'dan güçlü alt-çıta.
        from baselines import GPHypothesisProvider
        return GPHypothesisProvider(seed=int(config.get("seed", 0)))

    if provider in ("bayesopt", "bayesian"):
        # Bayesian-optimization baseline (Deney A): TPE ile sabit şablonların
        # hiperparametrelerini fitness modelleyerek arar. MVP kriter 9'un 3. baseline'ı.
        from baselines import BayesianOptProvider
        return BayesianOptProvider(seed=int(config.get("seed", 0)))

    if provider in _OPENAI_COMPATIBLE_DEFAULTS:
        # Gecikmeli import: döngüsel bağımlılığı önler
        from llm.openai_client import OpenAICompatibleClient
        from agents.hypothesis_generator import LLMHypothesisProvider

        defaults = _OPENAI_COMPATIBLE_DEFAULTS[provider]
        base_url = config.get("endpoint") or config.get("base_url") or defaults["base_url"]
        api_key_env = config.get("api_key_env") or defaults["api_key_env"]
        model = config.get("model")
        if not model:
            raise ValueError(f"'{provider}' için models.yaml'da 'model' belirtilmeli.")
        headers = {"X-Title": "agentic-quant"} if provider == "openrouter" else None
        client = OpenAICompatibleClient(base_url, api_key_env, default_headers=headers)
        return LLMHypothesisProvider(
            client, model=model,
            temperature=float(config.get("temperature", 0.9)),
            max_tokens=int(config.get("max_tokens", 4000)))

    raise NotImplementedError(f"Bilinmeyen sağlayıcı: {provider!r}")


def make_critic(config: dict):
    """configs/models.yaml'daki quant_critic bloğuna göre eleştirmen kur.

    dummy -> her şeyi geçirir. openai-uyumlu -> bağımsız LLM eleştirmen.
    """
    from agents.quant_critic import DummyCritic

    config = config or {}
    provider = config.get("provider", "dummy")
    if provider == "dummy":
        return DummyCritic()

    if provider in _OPENAI_COMPATIBLE_DEFAULTS:
        from llm.openai_client import OpenAICompatibleClient
        from agents.quant_critic import LLMCritic

        defaults = _OPENAI_COMPATIBLE_DEFAULTS[provider]
        base_url = config.get("endpoint") or config.get("base_url") or defaults["base_url"]
        api_key_env = config.get("api_key_env") or defaults["api_key_env"]
        model = config.get("model")
        if not model:
            raise ValueError(f"'{provider}' critic için 'model' belirtilmeli.")
        headers = {"X-Title": "agentic-quant"} if provider == "openrouter" else None
        client = OpenAICompatibleClient(base_url, api_key_env, default_headers=headers)
        return LLMCritic(client, model=model, temperature=float(config.get("temperature", 0.2)))

    raise NotImplementedError(f"Bilinmeyen critic sağlayıcısı: {provider!r}")
