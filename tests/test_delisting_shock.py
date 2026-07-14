"""
Delisting şoku testi (Doküman 7 / survivorship kalıntısı).

Pencere sonundan önce verisi biten (delist olan) hisse, pozisyondan %0 yerine
şok getiriyle (-şok) çıkmalı. Sona kadar yaşayan hisseye dokunulmamalı.
"""
import numpy as np
import pandas as pd

from data.adapter import _inject_delisting_shock


def _mk(dates, vals):
    return pd.DataFrame(vals, index=dates)


def test_shock_injected_for_delisted():
    dates = pd.bdate_range("2020-01-01", periods=6)
    # A sona kadar yaşar; B 3. barda delist olur (sonrası NaN)
    close = _mk(dates, {"A": [10, 11, 12, 13, 14, 15],
                        "B": [20, 21, 22, np.nan, np.nan, np.nan]})
    fields = {"close": close.copy(), "adjusted_close": close.copy(),
              "open": close.copy(), "high": close.copy(), "low": close.copy()}
    n = _inject_delisting_shock(fields, ["open", "high", "low", "close",
                                         "adjusted_close"], shock=0.30)
    assert n == 1, "sadece B delist olmalı"
    # B'nin son geçerli fiyatı 22; şok barı (index 3) = 22*0.70 = 15.4
    shock_bar = dates[3]
    assert abs(fields["close"].at[shock_bar, "B"] - 15.4) < 1e-9
    # şok getirisi ~ -30%
    ret = fields["close"]["B"].pct_change(fill_method=None).loc[shock_bar]
    assert abs(ret - (-0.30)) < 1e-9
    # A'ya hiç dokunulmadı
    assert fields["close"]["A"].tolist() == [10, 11, 12, 13, 14, 15]
    print("  [ok] delist olan hisseye -%30 şok, yaşayana dokunulmadı")


def test_no_shock_when_disabled_semantics():
    """shock=0 çağrısı çağrılmasa da eski davranış (kayıpsız) korunur —
    adapter shock>0 iken çağırır; burada shock parametresinin etkisini test et."""
    dates = pd.bdate_range("2020-01-01", periods=4)
    close = _mk(dates, {"B": [20, 21, np.nan, np.nan]})
    fields = {"close": close.copy(), "adjusted_close": close.copy(),
              "open": close.copy(), "high": close.copy(), "low": close.copy()}
    n = _inject_delisting_shock(fields, ["close"], shock=0.0)
    # shock=0 -> şok barı = son fiyat*1.0 = 21 (getiri 0, eski davranış)
    assert abs(fields["close"].at[dates[2], "B"] - 21.0) < 1e-9
    assert n == 1
    print("  [ok] shock=0 eski kayıpsız davranışı verir")


def main():
    test_shock_injected_for_delisted()
    test_no_shock_when_disabled_semantics()
    print("OK — delisting şoku testleri geçti.")


if __name__ == "__main__":
    main()
