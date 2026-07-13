"""
Çok amaçlı Pareto sıralaması testleri (Doküman 11.2).
"""
from evaluation.pareto import conservative_score, evaluate_strategies

STRONG = [0.0015, 0.0005] * 100   # yüksek Sharpe (pozitif alt sınır)
WEAK = [0.0004, -0.0004] * 100    # Sharpe ~0


def test_domination():
    # A ve B aynı getiri (aynı sharpe_lb) ama A daha düşük DD ve turnover -> A domine eder
    recs = [
        ("A", "iyi", 1.0, 0.05, 50.0, STRONG),
        ("B", "kötü", 1.0, 0.30, 200.0, STRONG),
    ]
    evals = {e.hypothesis_id: e for e in evaluate_strategies(recs)}
    assert evals["A"].pareto_optimal and not evals["B"].pareto_optimal
    print("  [ok] tüm boyutlarda daha iyi olan domine ediyor (B Pareto-dışı)")


def test_tradeoff_both_pareto():
    # C: yüksek Sharpe ama yüksek DD; D: düşük Sharpe ama düşük DD -> ikisi de Pareto
    recs = [
        ("C", "yüksek getiri", 1.5, 0.35, 100.0, STRONG),
        ("D", "düşük risk", 0.2, 0.03, 100.0, WEAK),
    ]
    evals = {e.hypothesis_id: e for e in evaluate_strategies(recs)}
    assert evals["C"].pareto_optimal and evals["D"].pareto_optimal
    print("  [ok] takas (tradeoff) durumunda ikisi de Pareto-optimal")


def test_score_penalizes_risk():
    hi = conservative_score(1.0, 0.05, 50.0)
    lo = conservative_score(1.0, 0.40, 300.0)
    assert hi > lo
    print(f"  [ok] muhafazakâr skor riski cezalandırıyor: {hi:.2f} > {lo:.2f}")


def main():
    test_domination()
    test_tradeoff_both_pareto()
    test_score_penalizes_risk()
    print("OK — Pareto testleri geçti.")


if __name__ == "__main__":
    main()
