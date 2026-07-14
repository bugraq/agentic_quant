"""
Genetic-programming baseline (Doküman 26 Deney A / MVP kriter 9).

Random-search "kör örnekleme" iken; GP baseline'ı geçmiş fitness'a (araştırma
Sharpe'ı) GÖRE evrimleşir: DSL sinyal ağaçlarından bir popülasyon tutar,
turnuva seçimiyle iyi ebeveynleri seçer, CROSSOVER (alt-ağaç değişimi) ve
MUTASYON (pencere/alan/operatör değişimi, negate ekle/çıkar) ile yeni bireyler
üretir. Ekonomik gerekçe YOK, LLM YOK — saf evrimsel arama (AlphaGen/AlphaForge
ailesinin klasik yöntemi). Aynı kısıt yüzeyi, aynı pipeline, aynı bütçe.

Fitness geri bildirimi ResearchContext.prior_experiments'ten okunur (bu
üreticinin ürettiği hid'lerin Sharpe'ı); derlenemeyen/duplicate/gate'te düşen
bireyler ceza fitness'ı alır (GP o bölgeden kaçar).

    configs/models.yaml -> hypothesis_generator: {provider: gp, seed: 0}
"""
from __future__ import annotations

import random
import re

from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from contracts.research_context import ResearchContext
from baselines.random_search import (
    _COMBINERS, _DEFAULT_FIELDS, _DEFAULT_HORIZONS, _TS_OPS, _family_for,
)

_PENALTY = -9.0            # derlenemeyen/duplicate/geçersiz bireyin fitness'ı
_INIT_POP = 6             # bu kadar değerlendirilmiş birey olana dek rastgele başlat
_TOURNAMENT = 3          # turnuva boyutu (seçim baskısı)
_P_MUTATE = 0.5          # crossover sonrası mutasyon olasılığı
_SHARPE_RE = re.compile(r"Sharpe\s*(-?\d+(?:\.\d+)?)")


def _scan(expr: Expression, ops: set, fields: set) -> None:
    """Ağacı gez: kullanılan operatörleri ve alanları topla (aile etiketi için)."""
    if expr.op == "field":
        fields.add(expr.field)
        return
    ops.add(expr.op)
    for c in expr.inputs:
        if isinstance(c, Expression):
            _scan(c, ops, fields)


def _inner_slots(root: Expression) -> list[tuple[list, int]]:
    """(inputs_listesi, indeks) — köke kadar HER alt-düğüm için değiştirme yuvası."""
    out: list[tuple[list, int]] = []

    def rec(node: Expression) -> None:
        for i, child in enumerate(node.inputs):
            if isinstance(child, Expression):
                out.append((node.inputs, i))
                rec(child)

    rec(root)
    return out


class GPHypothesisProvider:
    """DSL sinyal ağaçları üstünde genetik-programlama baseline'ı (deterministik)."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._counter = 0
        self._trees: dict[str, Expression] = {}   # hid -> core (rank'in içi)
        self._fitness: dict[str, float] = {}      # hid -> araştırma Sharpe / ceza
        self.last_meta = {"model_name": f"genetic-programming(seed={seed})",
                          "temperature": None, "prompt_hash": None, "output_hash": None}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # ---- rastgele başlangıç bireyi (random baseline mantığı) -------------
    def _random_feature(self, fields, ts_ops, horizons) -> Expression:
        rng = self._rng
        expr = Expression(op=rng.choice(ts_ops), window=rng.choice(horizons),
                          inputs=[Expression(op="field", field=rng.choice(fields))])
        if rng.random() < 0.5:
            expr = Expression(op="negate", inputs=[expr])
        return expr

    def _random_core(self, fields, ts_ops, horizons) -> Expression:
        core = self._random_feature(fields, ts_ops, horizons)
        if self._rng.random() < 0.4:
            other = self._random_feature(fields, ts_ops, horizons)
            core = Expression(op=self._rng.choice(_COMBINERS), inputs=[core, other])
        return core

    # ---- evrimsel operatörler -------------------------------------------
    def _mutate(self, core: Expression, fields, ts_ops, horizons) -> Expression:
        """Ağacın rastgele bir düğümünü değiştir (pencere/alan/op/negate)."""
        rng = self._rng
        mut = core.model_copy(deep=True)
        # kök dahil tüm düğümler
        nodes = [mut] + [lst[i] for lst, i in _inner_slots(mut)]
        node = rng.choice(nodes)
        if node.op == "field":
            node.field = rng.choice(fields)
        elif node.window is not None:
            roll = rng.random()
            if roll < 0.5:
                node.window = rng.choice(horizons)          # pencere mutasyonu
            else:
                node.op = rng.choice(ts_ops)                # operatör mutasyonu
        else:
            # negate/combiner düğümü: negate ekle/çıkar
            if node.op == "negate" and node.inputs:
                inner = node.inputs[0]
                node.op, node.inputs = inner.op, inner.inputs
                node.field, node.window = inner.field, inner.window
        return mut

    def _crossover(self, a: Expression, b: Expression) -> Expression:
        """a'nın kopyasında rastgele bir alt-ağacı b'den bir alt-ağaçla değiştir."""
        rng = self._rng
        child = a.model_copy(deep=True)
        slots = _inner_slots(child)
        if not slots:
            return child
        donor_nodes = [b] + [lst[i] for lst, i in _inner_slots(b)]
        donor = rng.choice(donor_nodes).model_copy(deep=True)
        lst, idx = rng.choice(slots)
        lst[idx] = donor
        return child

    def _select(self, evaluated: list[tuple[str, Expression, float]]) -> Expression:
        """Turnuva seçimi: rastgele k birey, en iyi fitness'lıyı döndür."""
        k = min(_TOURNAMENT, len(evaluated))
        contenders = self._rng.sample(evaluated, k)
        best = max(contenders, key=lambda t: t[2])
        return best[1]

    def _update_fitness(self, context: ResearchContext) -> None:
        """prior_experiments'ten bu üreticinin bireylerinin Sharpe'ını oku."""
        for e in context.prior_experiments:
            hid = e.hypothesis_id
            if hid not in self._trees or hid in self._fitness:
                continue
            m = _SHARPE_RE.search(e.headline_metric or "")
            # Sharpe yoksa (derleme/duplicate/statik red) ceza fitness'ı
            self._fitness[hid] = float(m.group(1)) if m else _PENALTY

    def next(self, context: ResearchContext) -> HypothesisSpec:
        self._counter += 1
        hid = f"hyp_{self._counter:04d}"
        rng = self._rng
        self._update_fitness(context)

        fields = [f for f in (context.allowed_fields or _DEFAULT_FIELDS)
                  if f in _DEFAULT_FIELDS] or _DEFAULT_FIELDS
        ts_ops = [o for o in (context.allowed_operators or _TS_OPS)
                  if o in _TS_OPS] or ["return", "rolling_mean"]
        horizons = context.allowed_horizons or _DEFAULT_HORIZONS

        evaluated = [(h, self._trees[h], self._fitness[h])
                     for h in self._trees if h in self._fitness]

        if len(evaluated) < _INIT_POP:
            core = self._random_core(fields, ts_ops, horizons)     # popülasyonu tohumla
        else:
            p1, p2 = self._select(evaluated), self._select(evaluated)
            core = self._crossover(p1, p2)
            if rng.random() < _P_MUTATE:
                core = self._mutate(core, fields, ts_ops, horizons)

        signal = Expression(op="cross_sectional_rank", inputs=[core])
        self._trees[hid] = core

        ops_used: set = set()
        fields_used: set = set()
        _scan(core, ops_used, fields_used)
        combined = any(o in _COMBINERS for o in ops_used)
        fam = _family_for(ops_used, fields_used, combined)
        ptype = rng.choice(context.allowed_portfolio_types or ["cross_sectional_long_short"])
        q = rng.choice([0.1, 0.2, 0.3])
        return HypothesisSpec(
            hypothesis_id=hid, title=f"gp-{self._counter:03d} ({fam.value})",
            claim=("Genetik-programlama baseline'ı: fitness'a göre evrimleşen, "
                   "ekonomik gerekçesiz DSL ağacı."),
            family=fam,
            economic_mechanism=EconomicMechanism(
                type="genetic_baseline",
                description="Mekanizma yok — evrimsel arama (crossover+mutasyon)."),
            universe=Universe(source="sp500_point_in_time", minimum_price=5.0),
            features=[], signal=signal,
            portfolio=Portfolio(type=ptype, long_quantile=q, short_quantile=q),
            execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                                holding_period_days=rng.choice([1, 5, 10]),
                                rebalance=rng.choice(context.allowed_rebalance or ["daily"])),
            falsification=Falsification(minimum_oos_sharpe=0.5))
