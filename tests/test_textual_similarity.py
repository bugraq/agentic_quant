"""
Metinsel benzerlik testi (Doküman 14.1 — üç-seviyeli yeniliğin metinsel ayağı).

Near-verbatim aynı açıklama duplicate sayılmalı; ortak kelime paylaşan ama
FARKLI fikir olan hipotezler yanlış-pozitif YAPMAMALI (eşik yüksek).
"""
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from memory.similarity import NoveltyIndex, _cosine, _hyp_text


def _hyp(hid, title, claim, mech_desc, window=20) -> HypothesisSpec:
    sig = Expression(op="cross_sectional_rank",
                     inputs=[Expression(op="return", window=window,
                                        inputs=[Expression(op="field", field="close")])])
    return HypothesisSpec(
        hypothesis_id=hid, title=title, claim=claim, family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="momentum", description=mech_desc),
        universe=Universe(source="x"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.2, short_quantile=0.2),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def test_near_verbatim_flagged():
    idx = NoveltyIndex()
    a = _hyp("hyp_1", "Momentum winners persist",
             "Stocks with strong past returns continue to outperform next month.",
             "Investors underreact to momentum information gradually.")
    idx.add(a)
    # neredeyse aynı kelimeler, ufak değişiklik
    b = _hyp("hyp_2", "Momentum winners persist strongly",
             "Stocks with strong past returns continue to outperform the next month.",
             "Investors underreact to momentum information gradually.", window=60)
    assert idx.check_textual(b) == "hyp_1", "near-verbatim metin yakalanmalı"
    print("  [ok] near-verbatim açıklama duplicate işaretlendi")


def test_shared_vocab_not_flagged():
    idx = NoveltyIndex()
    a = _hyp("hyp_1", "Momentum",
             "Past winners keep winning due to gradual investor underreaction.",
             "Behavioral underreaction to price trends.")
    idx.add(a)
    # 'momentum/volume' ortak kelime ama FARKLI fikir
    b = _hyp("hyp_2", "Volume-driven reversal",
             "High abnormal trading volume precedes short-term price reversals as "
             "liquidity providers demand compensation.",
             "Temporary liquidity pressure from uninformed flow overshoots value.")
    assert idx.check_textual(b) is None, "farklı fikir yanlış-pozitif olmamalı"
    print("  [ok] ortak kelimeli ama farklı fikir yanlış-pozitif yapmadı")


def test_cosine_bounds():
    t = _hyp_text(_hyp("h", "momentum reversal", "a b c", "d e"))
    assert abs(_cosine(t, t) - 1.0) < 1e-9      # kendisiyle 1
    assert _cosine(t, __import__("collections").Counter()) == 0.0
    print("  [ok] cosine sınırları doğru")


def main():
    test_near_verbatim_flagged()
    test_shared_vocab_not_flagged()
    test_cosine_bounds()
    print("OK — metinsel benzerlik testleri geçti.")


if __name__ == "__main__":
    main()
