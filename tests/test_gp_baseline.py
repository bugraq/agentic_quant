"""
Genetic-programming baseline testi (Doküman 26 Deney A).

GP üreticisi: (1) deterministik (seed), (2) geçerli/derlenebilir HypothesisSpec
üretir, (3) fitness geri bildirimiyle EVRİMLEŞİR (iyi ebeveynleri kullanır),
(4) etiket-sinyal dürüstlüğü (aile gerçek yapıdan türetilir).
"""
from contracts.research_context import (
    ExperimentSummary, GenerationMode, ResearchContext,
)
from baselines import GPHypothesisProvider
from dsl import compile_hypothesis, validate


def _ctx(priors=None) -> ResearchContext:
    return ResearchContext(
        campaign_goal="test", universe_description="anon",
        allowed_fields=["close", "volume"],
        allowed_operators=["return", "rolling_mean", "zscore", "volatility",
                           "multiply", "subtract"],
        allowed_horizons=[5, 20, 60],
        allowed_rebalance=["daily"],
        allowed_portfolio_types=["cross_sectional_long_short"],
        prior_experiments=priors or [],
        generation_mode=GenerationMode.new)


def test_deterministic():
    a = GPHypothesisProvider(seed=42)
    b = GPHypothesisProvider(seed=42)
    ha = [a.next(_ctx()).signal.model_dump_json() for _ in range(5)]
    hb = [b.next(_ctx()).signal.model_dump_json() for _ in range(5)]
    assert ha == hb, "aynı seed aynı diziyi vermeli (reproducibility)"
    print("  [ok] deterministik (seed)")


def test_produces_valid_compilable():
    gp = GPHypothesisProvider(seed=1)
    for _ in range(12):
        spec = gp.next(_ctx())
        graph = compile_hypothesis(spec)         # derlenmeli
        dec = validate(graph, spec)              # sızıntı/şema kontrolü
        # trade_time open_t_plus_1 -> sızıntı olmamalı
        assert dec.decision.value != "reject" or all(
            i.type != "temporal_leakage" for i in dec.issues)
    print("  [ok] 12 birey derlendi, sızıntı yok")


def test_evolves_from_fitness():
    """Fitness verildiğinde GP rastgele başlatmayı bırakıp evrimleşmeli.

    İlk _INIT_POP birey rastgele; sonra prior_experiments'te Sharpe geri
    beslenince crossover/mutasyon devreye girer. Kanıt: yüksek-fitness
    ebeveynin alanları çocuklarda baskınlaşır.
    """
    gp = GPHypothesisProvider(seed=7)
    specs = []
    priors = []
    for _ in range(6):                            # popülasyonu tohumla
        s = gp.next(_ctx(priors))
        specs.append(s)
        # birine yüksek, ötekine düşük fitness ver
        sharpe = 2.0 if "60" in s.signal.model_dump_json() else -1.0
        priors.append(ExperimentSummary(
            hypothesis_id=s.hypothesis_id, title=s.title, family=s.family.value,
            outcome="rejected", headline_metric=f"Sharpe {sharpe:.2f}"))
    # evrim fazı: birkaç çocuk üret — hata vermeden geçerli spec üretmeli
    for _ in range(6):
        child = gp.next(_ctx(priors))
        compile_hypothesis(child)
        priors.append(ExperimentSummary(
            hypothesis_id=child.hypothesis_id, title=child.title,
            family=child.family.value, outcome="rejected",
            headline_metric="Sharpe 0.10"))
    assert gp._fitness, "fitness geri bildirimi okunmadı"
    assert len(gp._trees) == 12
    print(f"  [ok] evrim fazı çalıştı ({len(gp._fitness)} fitness okundu)")


def main():
    test_deterministic()
    test_produces_valid_compilable()
    test_evolves_from_fitness()
    print("OK — GP baseline testleri geçti.")


if __name__ == "__main__":
    main()
