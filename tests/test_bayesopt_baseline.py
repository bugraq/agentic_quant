"""
Bayesian-optimization (TPE) baseline testi (Doküman 26 Deney A, MVP kriter 9).

- Deterministik (seed).
- Geçerli/derlenebilir, sızıntısız spec üretir.
- TPE fitness geri bildirimiyle İYİ konfigürasyonlara yönelir (iyi havuza benzeyen
  adayları seçme olasılığı artar).
"""
from contracts.research_context import (
    ExperimentSummary, GenerationMode, ResearchContext,
)
from baselines import BayesianOptProvider
from dsl import compile_hypothesis, validate


def _ctx(priors=None) -> ResearchContext:
    return ResearchContext(
        campaign_goal="t", universe_description="anon",
        allowed_fields=["close", "volume"],
        allowed_operators=["return", "negate", "volatility", "zscore", "multiply",
                           "cross_sectional_rank"],
        allowed_horizons=[5, 20, 60],
        allowed_rebalance=["daily"],
        allowed_portfolio_types=["cross_sectional_long_short"],
        prior_experiments=priors or [],
        generation_mode=GenerationMode.new)


def test_deterministic():
    a = BayesianOptProvider(seed=3)
    b = BayesianOptProvider(seed=3)
    ha = [a.next(_ctx()).signal.model_dump_json() for _ in range(5)]
    hb = [b.next(_ctx()).signal.model_dump_json() for _ in range(5)]
    assert ha == hb
    print("  [ok] deterministik (seed)")


def test_valid_compilable():
    bo = BayesianOptProvider(seed=1)
    for _ in range(12):
        spec = bo.next(_ctx())
        graph = compile_hypothesis(spec)
        dec = validate(graph, spec)
        assert all(i.type != "temporal_leakage" for i in dec.issues)
    print("  [ok] 12 birey derlendi, sızıntı yok")


def test_tpe_prefers_good_region():
    """momentum şablonu + w=60'a yüksek fitness verilince TPE onu tercih etmeli."""
    bo = BayesianOptProvider(seed=0)
    priors = []
    specs = []
    for _ in range(14):
        s = bo.next(_ctx(priors))
        specs.append(s)
        cfg = bo._configs[s.hypothesis_id]
        # momentum + uzun pencere = iyi; gerisi kötü (sentetik ödül)
        fit = 1.5 if (cfg["template"] == "momentum" and cfg["w"] == 60) else -0.5
        priors.append(ExperimentSummary(
            hypothesis_id=s.hypothesis_id, title=s.title, family=s.family.value,
            outcome="rejected", headline_metric=f"Sharpe {fit:.2f}"))
    # son 4 seçimde momentum oranı, ilk 4'e göre artmalı (TPE öğrendi)
    late = [bo._configs[s.hypothesis_id]["template"] for s in specs[-4:]]
    assert late.count("momentum") >= 2, f"TPE iyi bölgeye yönelmeli, son: {late}"
    print(f"  [ok] TPE iyi bölgeye yöneldi (son 4 şablon: {late})")


def main():
    test_deterministic()
    test_valid_compilable()
    test_tpe_prefers_good_region()
    print("OK — Bayesian-opt baseline testleri geçti.")


if __name__ == "__main__":
    main()
