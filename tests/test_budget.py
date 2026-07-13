"""
Bandit bütçe tahsisi testleri (Doküman 16.2 / Deney E).
Başarılı aile daha çok bütçe alır; denenmemiş aile keşfedilir.
"""
from collections import Counter

from orchestrator.budget import ThompsonBandit


def test_favors_successful_family():
    fams = ["momentum", "reversal", "volume"]
    counts = {"momentum": (5, 5), "reversal": (0, 5)}  # volume hiç denenmedi
    bandit = ThompsonBandit(fams, seed=0)
    picks = Counter(bandit.select(counts) for _ in range(500))
    assert picks["momentum"] > picks["reversal"] * 3, picks
    print(f"  [ok] başarılı aileye daha çok bütçe: {dict(picks)}")


def test_explores_untried_family():
    fams = ["momentum", "reversal", "volume"]
    counts = {"momentum": (5, 5), "reversal": (0, 5)}
    bandit = ThompsonBandit(fams, seed=0)
    picks = Counter(bandit.select(counts) for _ in range(500))
    assert picks["volume"] > 0, "denenmemiş aile hiç keşfedilmedi"
    print(f"  [ok] denenmemiş aile keşfediliyor: volume={picks['volume']}")


def test_empty_counts_uniform_ish():
    fams = ["a", "b", "c"]
    bandit = ThompsonBandit(fams, seed=1)
    picks = Counter(bandit.select({}) for _ in range(300))
    assert all(picks[f] > 0 for f in fams), f"bir kol hiç seçilmedi: {picks}"
    print(f"  [ok] boş geçmişte tüm kollar keşfediliyor: {dict(picks)}")


def main():
    test_favors_successful_family()
    test_explores_untried_family()
    test_empty_counts_uniform_ish()
    print("OK — bandit bütçe testleri geçti.")


if __name__ == "__main__":
    main()
