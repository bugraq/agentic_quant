"""
Random-search baseline testleri (Doküman 15 / 26 Deney A).

Baseline LLM ile aynı pipeline'dan geçmek zorunda: ürettiği her hipotez
derlenebilmeli, sızıntısız olmalı (validator accept), deterministik olmalı
(aynı seed = aynı dizi) ve tek tip olmamalı (çeşitlilik).
"""
from contracts.research_context import GenerationMode, ResearchContext
from baselines import RandomHypothesisProvider
from dsl import compile_hypothesis, validate

_CTX = ResearchContext(
    campaign_goal="test", universe_description="test evreni",
    allowed_fields=["close", "volume", "dollar_volume"],
    allowed_operators=["return", "rolling_mean", "rolling_std", "zscore",
                       "ewma", "volatility", "delta", "negate", "multiply",
                       "subtract", "cross_sectional_rank"],
    allowed_horizons=[5, 10, 20, 60],
    allowed_rebalance=["daily", "weekly"],
    allowed_portfolio_types=["cross_sectional_long_short", "long_only"],
    generation_mode=GenerationMode.new, experiments_remaining=30)

N = 30


def test_all_compile_and_validate():
    prov = RandomHypothesisProvider(seed=1)
    for _ in range(N):
        hyp = prov.next(_CTX)
        g = compile_hypothesis(hyp)                    # derlenmeli
        dec = validate(g, hyp,
                       allowed_fields=_CTX.allowed_fields,
                       allowed_rebalance=_CTX.allowed_rebalance,
                       allowed_portfolio_types=_CTX.allowed_portfolio_types)
        assert dec.decision.value == "accept", \
            f"{hyp.hypothesis_id} validator'dan geçemedi: {dec.issues}"
    print(f"  [ok] {N} rastgele hipotezin hepsi derlendi + sızıntısız")


def test_deterministic_and_diverse():
    a = RandomHypothesisProvider(seed=7)
    b = RandomHypothesisProvider(seed=7)
    sigs_a = [a.next(_CTX).signal.model_dump_json() for _ in range(N)]
    sigs_b = [b.next(_CTX).signal.model_dump_json() for _ in range(N)]
    assert sigs_a == sigs_b, "aynı seed farklı dizi üretti (reproducibility bozuk)"
    assert len(set(sigs_a)) > N // 2, "rastgele üreteç yeterince çeşitli değil"
    fams = {RandomHypothesisProvider(seed=7).next(_CTX).family for _ in range(1)}
    prov = RandomHypothesisProvider(seed=7)
    fams = {prov.next(_CTX).family.value for _ in range(N)}
    assert len(fams) >= 3, f"aile çeşitliliği düşük: {fams}"
    print(f"  [ok] deterministik (seed) + çeşitli ({len(set(sigs_a))}/{N} tekil, "
          f"aileler: {sorted(fams)})")


def main():
    test_all_compile_and_validate()
    test_deterministic_and_diverse()
    print("OK — random-search baseline hazır (Deney A).")


if __name__ == "__main__":
    main()
