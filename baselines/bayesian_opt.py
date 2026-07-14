"""
Bayesian-optimization baseline (Doküman 26 Deney A / MVP kriter 9 — 3. baseline).

Random kör örnekler, GP yapıyı evrimleştirir; BO ise FITNESS'İ MODELLEYİP sabit
strateji şablonlarının hiperparametrelerini (pencere, quantile, holding) akıllıca
arar — klasik "Bayesian optimization bir modelin hiperparametrelerini tune eder"
yaklaşımı. Ekonomik gerekçe YOK, LLM YOK.

Yöntem: Tree-structured Parzen Estimator (TPE, Hyperopt/Optuna'nın çekirdeği),
scipy'siz. Gözlemleri fitness'a göre 'iyi' (üst çeyrek) ve 'kötü' diye ayırır;
her hiperparametre için p(değer|iyi)/p(değer|kötü) oranını en çoklaştıran adayı
seçer. İlk turlar rastgele (keşif). Fitness prior_experiments'ten okunur.

    configs/models.yaml -> hypothesis_generator: {provider: bayesopt, seed: 0}
"""
from __future__ import annotations

import math
import random
import re

from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from contracts.research_context import ResearchContext

_SHARPE_RE = re.compile(r"Sharpe\s*(-?\d+(?:\.\d+)?)")
_N_INIT = 6            # bu kadar gözlem olana dek rastgele (keşif)
_N_CANDIDATES = 32     # TPE her turda bu kadar aday örnekler, en iyisini seçer
_GOOD_FRAC = 0.30      # üst bu oran 'iyi' havuz

# --- Sabit strateji şablonları (yapı sabit; BO yalnız parametreleri arar) -----
_TEMPLATES = ["momentum", "reversal", "low_vol", "volume_mom", "volume_reversal"]


def _field(name: str) -> Expression:
    return Expression(op="field", field=name)


def _build_signal(template: str, w: int, w2: int) -> tuple[Expression, HypothesisFamily]:
    ret = Expression(op="return", window=w, inputs=[_field("close")])
    if template == "momentum":
        return Expression(op="cross_sectional_rank", inputs=[ret]), HypothesisFamily.momentum
    if template == "reversal":
        neg = Expression(op="negate", inputs=[ret])
        return Expression(op="cross_sectional_rank", inputs=[neg]), HypothesisFamily.reversal
    if template == "low_vol":
        vol = Expression(op="volatility", window=w, inputs=[_field("close")])
        neg = Expression(op="negate", inputs=[vol])
        return Expression(op="cross_sectional_rank", inputs=[neg]), HypothesisFamily.volatility
    # hacimle etkileşim (composite)
    vz = Expression(op="zscore", window=w2, inputs=[_field("volume")])
    core = ret if template == "volume_mom" else Expression(op="negate", inputs=[ret])
    mul = Expression(op="multiply", inputs=[core, vz])
    return (Expression(op="cross_sectional_rank", inputs=[mul]),
            HypothesisFamily.composite)


class BayesianOptProvider:
    """TPE tabanlı Bayesian-optimization baseline (deterministik/seed)."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._counter = 0
        self._configs: dict[str, dict] = {}   # hid -> config
        self._fitness: dict[str, float] = {}
        self.last_meta = {"model_name": f"bayesian-opt-tpe(seed={seed})",
                          "temperature": None, "prompt_hash": None, "output_hash": None}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # ---- arama uzayı ----------------------------------------------------
    def _space(self, ctx: ResearchContext) -> dict:
        horizons = ctx.allowed_horizons or [5, 10, 20, 60]
        return {
            "template": _TEMPLATES,
            "w": horizons,
            "w2": horizons,
            "quantile": [0.1, 0.2, 0.3],
            "holding": [1, 5, 10],
            "rebalance": ctx.allowed_rebalance or ["daily"],
            "ptype": ctx.allowed_portfolio_types or ["cross_sectional_long_short"],
        }

    def _random_config(self, space: dict) -> dict:
        return {k: self._rng.choice(v) for k, v in space.items()}

    def _update_fitness(self, ctx: ResearchContext) -> None:
        for e in ctx.prior_experiments:
            hid = e.hypothesis_id
            if hid in self._configs and hid not in self._fitness:
                m = _SHARPE_RE.search(e.headline_metric or "")
                self._fitness[hid] = float(m.group(1)) if m else -9.0

    def _tpe_select(self, space: dict) -> dict:
        """TPE: iyi/kötü havuzlardan p(val|iyi)/p(val|kötü) oranını en çoklaştır."""
        obs = [(self._configs[h], self._fitness[h])
               for h in self._configs if h in self._fitness]
        obs.sort(key=lambda x: x[1], reverse=True)
        n_good = max(1, int(len(obs) * _GOOD_FRAC))
        good = [c for c, _ in obs[:n_good]]
        bad = [c for c, _ in obs[n_good:]] or good

        def _p(pool: list[dict], dim: str, val) -> float:
            # Laplace-düzeltilmiş oran (kategorik yoğunluk)
            hits = sum(1 for c in pool if c[dim] == val)
            return (hits + 1.0) / (len(pool) + len(space[dim]))

        best, best_score = None, -math.inf
        for _ in range(_N_CANDIDATES):
            cand = self._random_config(space)
            score = sum(math.log(_p(good, d, cand[d]) / _p(bad, d, cand[d]))
                        for d in space)
            if score > best_score:
                best, best_score = cand, score
        return best

    def next(self, context: ResearchContext) -> HypothesisSpec:
        self._counter += 1
        hid = f"hyp_{self._counter:04d}"
        self._update_fitness(context)
        space = self._space(context)

        evaluated = sum(1 for h in self._configs if h in self._fitness)
        cfg = (self._random_config(space) if evaluated < _N_INIT
               else self._tpe_select(space))
        self._configs[hid] = cfg

        signal, fam = _build_signal(cfg["template"], cfg["w"], cfg["w2"])
        return HypothesisSpec(
            hypothesis_id=hid, title=f"bo-{self._counter:03d} ({cfg['template']})",
            claim=("Bayesian-optimization baseline'ı: fitness modellenerek (TPE) "
                   "seçilen, ekonomik gerekçesiz parametrik strateji."),
            family=fam,
            economic_mechanism=EconomicMechanism(
                type="bayesopt_baseline",
                description="Mekanizma yok — TPE ile hiperparametre araması."),
            universe=Universe(source="sp500_point_in_time", minimum_price=5.0),
            features=[], signal=signal,
            portfolio=Portfolio(type=cfg["ptype"], long_quantile=cfg["quantile"],
                                short_quantile=cfg["quantile"]),
            execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                                holding_period_days=cfg["holding"],
                                rebalance=cfg["rebalance"]),
            falsification=Falsification(minimum_oos_sharpe=0.5))
