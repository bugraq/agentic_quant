"""
Procedural memory testi (Doküman 12.3).

Operasyon verimliliği (lineage'dan), aile doygunluğu ve eleme örüntüsü doğru
derslere dönüşmeli.
"""
from memory.procedural import (
    _operation_lessons, _saturation_lessons, _stage_lessons,
)


def test_operation_lessons():
    # revizyon: 3 deneme 2 kabul (verimli); inversion: 3 deneme 0 kabul (işe yaramadı)
    edges = [
        ("p1", "c1", "refinement", "accept"),
        ("p1", "c2", "refinement", "accept"),
        ("p1", "c3", "refinement", "reject"),
        ("f1", "c4", "inversion", "reject"),
        ("f2", "c5", "inversion", "reject"),
        ("f3", "c6", "inversion", "duplicate"),
        ("x", "c7", "parameter_variant", "reject"),   # optimizer -> atlanmalı
    ]
    out = _operation_lessons(edges)
    text = " ".join(out)
    assert "revizyon" in text and "verimli" in text
    assert "ters çevirme" in text and "hiç kabul getirmedi" in text
    assert "parametre varyantı" not in text, "optimizer hamlesi süreç dersi değil"
    print("  [ok] operasyon verimliliği doğru (revizyon verimli, inversion boş)")


def test_saturation_lessons():
    # (family, count, avg_sharpe, best_sharpe)
    stats = [("momentum", 6, 0.1, 0.25), ("volume", 2, 0.5, 0.9)]
    out = _saturation_lessons(stats)
    assert any("momentum" in l and "doygun" in l for l in out)
    assert not any("volume" in l for l in out), "az denenen/iyi aile doygun sayılmaz"
    print("  [ok] doygun aile (çok denenip zayıf) işaretleniyor")


def test_stage_lessons():
    counts = {"gate_rejected": 8, "robustness_rejected": 2, "accepted": 1,
              "duplicate": 3}
    out = _stage_lessons(counts)
    assert out and "performans kapısı" in out[0]
    print("  [ok] en çok eleme aşaması öne çıkıyor")


def main():
    test_operation_lessons()
    test_saturation_lessons()
    test_stage_lessons()
    print("OK — procedural memory testleri geçti.")


if __name__ == "__main__":
    main()
