"""
İstatistiksel test motoru testleri (Doküman 10).
Gürültüde anlamlılık düşük, sinyalde yüksek; deneme sayısı DSR'yi düşürür.
"""
import numpy as np

from evaluation.statistics import (
    benjamini_hochberg, deflated_sharpe_ratio, norm_cdf, norm_ppf,
    probabilistic_sharpe_ratio, sharpe_moments,
)


def test_normal_helpers():
    assert abs(norm_cdf(0.0) - 0.5) < 1e-9
    assert abs(norm_ppf(0.975) - 1.959963985) < 1e-4
    print("  [ok] normal cdf/ppf doğru")


def test_noise_not_significant():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=750).tolist()   # sıfır ortalama gürültü
    psr = probabilistic_sharpe_ratio(sharpe_moments(r), 0.0)
    assert psr < 0.9, f"gürültü anlamlı çıktı, PSR={psr:.2f}"
    print(f"  [ok] gürültü anlamlı değil: PSR={psr:.2f}")


def test_signal_significant():
    rng = np.random.default_rng(1)
    r = (0.0015 + rng.normal(0.0, 0.01, size=750)).tolist()  # net pozitif drift
    psr = probabilistic_sharpe_ratio(sharpe_moments(r), 0.0)
    assert psr > 0.9, f"gerçek sinyal anlamsız çıktı, PSR={psr:.2f}"
    print(f"  [ok] gerçek sinyal anlamlı: PSR={psr:.2f}")


def test_more_trials_lowers_dsr():
    rng = np.random.default_rng(2)
    r = (0.0006 + rng.normal(0.0, 0.01, size=750)).tolist()
    m = sharpe_moments(r)
    dsr_few = deflated_sharpe_ratio(m, n_trials=2, var_sr=0.02)
    dsr_many = deflated_sharpe_ratio(m, n_trials=200, var_sr=0.02)
    assert dsr_many < dsr_few, f"deneme artınca DSR düşmedi ({dsr_few:.2f}->{dsr_many:.2f})"
    print(f"  [ok] deneme sayısı DSR'yi düşürüyor: {dsr_few:.2f} -> {dsr_many:.2f}")


def test_benjamini_hochberg():
    survive = benjamini_hochberg([0.001, 0.2, 0.9], alpha=0.10)
    assert survive[0] and not survive[2], f"BH beklenmedik: {survive}"
    print(f"  [ok] BH-FDR çalışıyor: {survive}")


def main():
    test_normal_helpers()
    test_noise_not_significant()
    test_signal_significant()
    test_more_trials_lowers_dsr()
    test_benjamini_hochberg()
    print("OK — istatistiksel testler geçti.")


if __name__ == "__main__":
    main()
