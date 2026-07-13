"""
DataAdapter testleri — sentetik adaptör arayüzü (offline).
yfinance yolu ağ gerektirir; ayrı elle test edilir (scripts yok).
"""
from data import make_adapter
from data.adapter import _STD_FIELDS


def test_synthetic_adapter_via_config():
    cfg = {"source": "synthetic",
           "synthetic": {"kind": "momentum", "n_sec": 10, "n_days": 300, "seed": 1}}
    md = make_adapter(cfg).load()
    for f in _STD_FIELDS:
        assert f in md.fields, f"eksik alan: {f}"
    assert md.fields["close"].shape == (300, 10)
    print(f"  [ok] sentetik adaptör: {md.fields['close'].shape}, tüm alanlar var")


def test_unknown_source_raises():
    try:
        make_adapter({"source": "bloomberg"})
        raise AssertionError("bilinmeyen kaynak kabul edildi")
    except NotImplementedError:
        print("  [ok] bilinmeyen kaynak reddedildi")


def main():
    test_synthetic_adapter_via_config()
    test_unknown_source_raises()
    print("OK — adapter testleri geçti.")


if __name__ == "__main__":
    main()
