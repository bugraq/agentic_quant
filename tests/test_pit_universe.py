"""
Point-in-time evren testleri (survivorship düzeltmesi, Doküman 4/7).

İki grup — ikisi de OFFLINE (ağ yok):
  A) Rekonstrüksiyon: sentetik değişiklik tablosuyla üyelik matrisi doğru mu?
     (ekleme öncesi üye DEĞİL, çıkarma sonrası üye DEĞİL, aradaki günler üye)
  B) Motor maskesi: index_membership alanı varsa, hisse üye OLMADIĞI günlerde
     ağırlık alamamalı (bugünün listesi geçmişe uygulanamaz).
"""
import numpy as np
import pandas as pd

from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from data.pit_universe import normalize_changes, reconstruct_membership, yahoo_symbol
from data.synthetic import gen_cross_sectional_momentum
from dsl import compile_hypothesis
from backtest import evaluate_signal
from backtest.engine import _build_weights, _universe_mask


def test_reconstruction_semantics():
    # Bugün: A, B, C endekste. Olaylar:
    #   2020-06-01: D çıktı, C girdi   (öncesinde: D üye, C değil)
    #   2020-03-02: E çıktı, B girdi   (öncesinde: E üye, B değil)
    changes = pd.DataFrame({
        "date": pd.to_datetime(["2020-06-01", "2020-03-02"]),
        "added": ["C", "B"],
        "removed": ["D", "E"],
    })
    mem = reconstruct_membership(["A", "B", "C"], changes, "2020-01-01", "2020-12-31")

    def members(day):
        row = mem.loc[pd.Timestamp(day)]
        return set(mem.columns[row])

    assert members("2020-01-15") == {"A", "D", "E"}, members("2020-01-15")
    assert members("2020-04-15") == {"A", "B", "D"}, members("2020-04-15")
    assert members("2020-09-15") == {"A", "B", "C"}, members("2020-09-15")
    # Olay GÜNÜ yeni durum geçerli (o günden itibaren)
    assert members("2020-03-02") == {"A", "B", "D"}
    assert members("2020-06-01") == {"A", "B", "C"}
    print("  [ok] rekonstrüksiyon: ekleme/çıkarma sınırları doğru")


def test_normalize_changes_multiindex():
    raw = pd.DataFrame({
        ("Effective Date", "Effective Date"): ["June 1, 2020", "March 2, 2020"],
        ("Added", "Ticker"): ["C", None],
        ("Added", "Security"): ["C Corp", None],
        ("Removed", "Ticker"): ["D", "E"],
        ("Removed", "Security"): ["D Corp", "E Corp"],
        ("Reason", "Reason"): ["x", "y"],
    })
    ch = normalize_changes(raw)
    assert len(ch) == 2 and ch.iloc[0]["removed"] == "E"   # tarihe göre sıralı
    assert ch.iloc[1]["added"] == "C"
    print("  [ok] Wikipedia tablo normalizasyonu (multi-index, eksik hücre)")


def test_yahoo_symbol():
    assert yahoo_symbol("BRK.B") == "BRK-B" and yahoo_symbol(" BF.B ") == "BF-B"
    print("  [ok] Yahoo sembol dönüşümü (BRK.B -> BRK-B)")


def _hyp() -> HypothesisSpec:
    sig = Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=20, inputs=[Expression(op="field", field="close")])])
    return HypothesisSpec(
        hypothesis_id="hyp_pit", title="t", claim="t", family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="x", description="y"),
        universe=Universe(source="sp500_point_in_time"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def test_engine_respects_membership():
    data = gen_cross_sectional_momentum(seed=1, n_sec=10, n_days=200)
    tickers = list(data.get("close").columns)
    idx = data.dates

    # S00 ilk 100 gün üye DEĞİL; S01 son 50 gün üye DEĞİL; kalanlar hep üye.
    memb = pd.DataFrame(1.0, index=idx, columns=tickers)
    memb.iloc[:100, 0] = 0.0
    memb.iloc[-50:, 1] = 0.0
    data.fields["index_membership"] = memb

    h = _hyp()
    g = compile_hypothesis(h)
    sig = evaluate_signal(g, data)
    masked = _universe_mask(h, data, sig)
    w = _build_weights(masked, h.portfolio)

    assert (w.iloc[:100, 0].fillna(0) == 0).all(), "üye olmayan hisse ağırlık aldı!"
    assert (w.iloc[-50:, 1].fillna(0) == 0).all(), "endeksten çıkan hisse ağırlık aldı!"
    assert w.iloc[120:, 0].abs().sum() > 0, "üye olduğu dönemde hiç seçilmedi (şüpheli)"
    print("  [ok] motor point-in-time üyeliği uyguluyor (üye değilken ağırlık yok)")


def main():
    test_reconstruction_semantics()
    test_normalize_changes_multiindex()
    test_yahoo_symbol()
    test_engine_respects_membership()
    print("OK — point-in-time evren testleri geçti.")


if __name__ == "__main__":
    main()
