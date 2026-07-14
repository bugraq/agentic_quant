"""
Sektör-nötralizasyon testi (beyan=çalıştırılan, Doküman 7).

`neutralize_sector` operatörü ve `portfolio.sector_neutral` bayrağı sektör
haritası varken GERÇEKTEN sektör-bazlı çalışmalı; yoksa piyasa-nötre düşmeli.
"""
import numpy as np
import pandas as pd

from backtest.engine import _apply_sector_neutral
from backtest.evaluator import demean_by_sector
from contracts.hypothesis_spec import Portfolio
from data.synthetic import MarketData


def _panel(vals: dict[str, list], dates) -> pd.DataFrame:
    return pd.DataFrame(vals, index=dates)


def test_demean_by_sector_groups():
    dates = pd.bdate_range("2020-01-01", periods=2)
    # A,B tech; C,D finans. Her sektör içinde demean edilmeli.
    df = _panel({"A": [1.0, 2.0], "B": [3.0, 4.0],
                 "C": [10.0, 10.0], "D": [10.0, 30.0]}, dates)
    sectors = {"A": "Tech", "B": "Tech", "C": "Fin", "D": "Fin"}
    out = demean_by_sector(df, sectors)
    # Tech grubu: satır ortalaması (A+B)/2 çıkarılır -> A=-1, B=+1 (ilk gün)
    assert abs(out.loc[dates[0], "A"] + 1.0) < 1e-9
    assert abs(out.loc[dates[0], "B"] - 1.0) < 1e-9
    # Fin grubu ilk gün: (10+10)/2=10 -> C=0, D=0
    assert abs(out.loc[dates[0], "C"]) < 1e-9
    # Her sektör her satırda net-sıfır
    for sec_cols in (["A", "B"], ["C", "D"]):
        assert abs(out[sec_cols].sum(axis=1)).max() < 1e-9
    print("  [ok] neutralize_sector sektör-içi demean yapıyor")


def test_demean_by_sector_fallback_market():
    dates = pd.bdate_range("2020-01-01", periods=2)
    df = _panel({"A": [1.0, 2.0], "B": [3.0, 4.0]}, dates)
    # Harita yok -> piyasa-nötr (tüm evren tek grup)
    out = demean_by_sector(df, None)
    assert abs(out.sum(axis=1)).max() < 1e-9
    print("  [ok] harita yokken piyasa-nötre düşüyor")


def test_portfolio_sector_neutral_zeroes_each_sector():
    dates = pd.bdate_range("2020-01-01", periods=1)
    # Uzun A,B (tech); ağırlıklar sektör içinde net-sıfır olmalı
    w = _panel({"A": [0.4], "B": [0.1], "C": [-0.2], "D": [-0.3]}, dates)
    data = MarketData(fields={"close": w}, sectors={"A": "Tech", "B": "Tech",
                                                    "C": "Fin", "D": "Fin"})
    out = _apply_sector_neutral(w, data)
    for sec_cols in (["A", "B"], ["C", "D"]):
        assert abs(out[sec_cols].sum(axis=1)).max() < 1e-9, "sektör net-sıfır değil"
    # Gross ~1'e ölçeklendi
    assert abs(out.abs().sum(axis=1).iloc[0] - 1.0) < 1e-9
    print("  [ok] portfolio.sector_neutral her sektörü net-sıfır yapıyor")


def main():
    test_demean_by_sector_groups()
    test_demean_by_sector_fallback_market()
    test_portfolio_sector_neutral_zeroes_each_sector()
    print("OK — sektör-nötralizasyon testleri geçti.")


if __name__ == "__main__":
    main()
