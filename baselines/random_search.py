"""
Random-search baseline (Doküman 15 / 26 Deney A).

Projenin ana akademik sorusu "LLM gerçekten klasik otomatik aramadan daha iyi
mi?"dir. Bu baseline, LLM ile AYNI kısıt yüzeyini (izinli alanlar, operatörler,
ufuklar) ve AYNI pipeline'ı (validator, novelty, gate, robustness, çoklu-test)
kullanır; tek farkı hipotezleri EKONOMİK GEREKÇESİZ, rastgele DSL ağaçları
olarak üretmesidir. Sabit deney bütçesi altında LLM'e karşı koşulur:

    configs/models.yaml -> hypothesis_generator: {provider: random, seed: 0}

Aile etiketi sinyalin GERÇEK yapısından türetilir (etiket-sinyal dürüstlüğü);
mekanizma açıklaması bilinçli olarak "mekanizma yok" der — baseline'ın tanımı bu.
"""
from __future__ import annotations

import random

from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from contracts.research_context import ResearchContext

# Rastgele ağacın kullanabileceği zaman-serisi operatörleri (window'lu).
_TS_OPS = ["return", "rolling_mean", "rolling_std", "zscore", "ewma",
           "volatility", "delta", "rolling_rank"]
_COMBINERS = ["multiply", "subtract"]
_DEFAULT_FIELDS = ["close", "volume", "dollar_volume"]
_DEFAULT_HORIZONS = [5, 10, 20, 60]


def _family_for(ops: set[str], fields: set[str], combined: bool) -> HypothesisFamily:
    """Aile etiketini sinyalin GERÇEK içeriğinden türet (dürüst etiketleme)."""
    if combined:
        return HypothesisFamily.composite
    if "volatility" in ops or "rolling_std" in ops:
        return HypothesisFamily.volatility
    if fields & {"volume", "dollar_volume"}:
        return HypothesisFamily.volume
    if "negate" in ops:
        return HypothesisFamily.reversal
    return HypothesisFamily.momentum


class RandomHypothesisProvider:
    """HypothesisProvider arayüzü: rastgele (mekanizmasız) hipotez üretir.

    Deterministiktir (seed) — aynı seed aynı hipotez dizisini verir
    (reproducibility, Doküman 18). Araştırma bağlamını (dersler, geçmiş)
    BİLEREK kullanmaz: saf rastgele arama baseline'ı budur.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
        self._counter = 0
        self.last_meta = {"model_name": f"random-search(seed={seed})",
                          "temperature": None, "prompt_hash": None,
                          "output_hash": None}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # ---- rastgele ağaç parçaları ----------------------------------------
    def _feature(self, fields: list[str], ts_ops: list[str],
                 horizons: list[int], ops_used: set, fields_used: set) -> Expression:
        rng = self._rng
        field = rng.choice(fields)
        op = rng.choice(ts_ops)
        fields_used.add(field)
        ops_used.add(op)
        expr = Expression(op=op, window=rng.choice(horizons),
                          inputs=[Expression(op="field", field=field)])
        if rng.random() < 0.5:
            ops_used.add("negate")
            expr = Expression(op="negate", inputs=[expr])
        return expr

    def next(self, context: ResearchContext) -> HypothesisSpec:
        self._counter += 1
        hid = f"hyp_{self._counter:04d}"
        rng = self._rng

        fields = [f for f in (context.allowed_fields or _DEFAULT_FIELDS)
                  if f in _DEFAULT_FIELDS] or _DEFAULT_FIELDS
        ts_ops = [o for o in (context.allowed_operators or _TS_OPS)
                  if o in _TS_OPS] or ["return", "rolling_mean"]
        horizons = context.allowed_horizons or _DEFAULT_HORIZONS

        ops_used: set = set()
        fields_used: set = set()
        core = self._feature(fields, ts_ops, horizons, ops_used, fields_used)
        combined = rng.random() < 0.4
        if combined:
            other = self._feature(fields, ts_ops, horizons, ops_used, fields_used)
            comb = rng.choice(_COMBINERS)
            ops_used.add(comb)
            core = Expression(op=comb, inputs=[core, other])
        signal = Expression(op="cross_sectional_rank", inputs=[core])

        fam = _family_for(ops_used, fields_used, combined)
        rebalance = rng.choice(context.allowed_rebalance or ["daily"])
        holding = rng.choice([1, 5, 10])
        ptype = rng.choice(context.allowed_portfolio_types
                           or ["cross_sectional_long_short"])
        q = rng.choice([0.1, 0.2, 0.3])

        title = f"rastgele-{self._counter:03d} ({fam.value})"
        claim = ("Rastgele arama baseline'ı: ekonomik gerekçesi olmayan, "
                 "rastgele örneklenmiş bir sinyal.")
        return HypothesisSpec(
            hypothesis_id=hid, title=title, claim=claim, family=fam,
            economic_mechanism=EconomicMechanism(
                type="random_baseline",
                description="Mekanizma yok — baseline tanımı gereği rastgele."),
            universe=Universe(source="sp500_point_in_time", minimum_price=5.0),
            features=[], signal=signal,
            portfolio=Portfolio(type=ptype, long_quantile=q, short_quantile=q),
            execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                                holding_period_days=holding, rebalance=rebalance),
            falsification=Falsification(minimum_oos_sharpe=0.5))
