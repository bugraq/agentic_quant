"""
İstatistiksel testler (Doküman 10) — multiple testing farkındalığı.

Tek bir yüksek Sharpe, çok sayıda deneme sonrası ANLAMSIZDIR. Bu modül:
  - Probabilistic Sharpe Ratio (PSR): ham anlamlılık (H0: SR<=benchmark)
  - Deflated Sharpe Ratio (DSR): deneme sayısını hesaba katan seçilim düzeltmesi
  - Bootstrap: Sharpe güven aralığı (moving-block, otokorelasyona dayanıklı)
  - Benjamini-Hochberg: çoklu test için FDR kontrolü

scipy'ye bağımlı değiliz; normal CDF/PPF elle hesaplanır.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GAMMA = 0.5772156649015329  # Euler-Mascheroni
TRADING_DAYS = 252


# ---- Normal dağılım yardımcıları ----------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Ters normal CDF — Acklam rasyonel yaklaşımı."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---- Sharpe momentleri ---------------------------------------------------
@dataclass
class SharpeMoments:
    sr: float        # per-period (yıllıklaştırılmamış) Sharpe
    n: int           # gözlem sayısı
    skew: float
    kurt: float      # tam kurtosis (normal = 3)


def sharpe_moments(returns: list[float]) -> SharpeMoments:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return SharpeMoments(0.0, n, 0.0, 3.0)
    mean, std = r.mean(), r.std(ddof=1)
    sr = mean / std
    z = (r - mean) / std
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))   # tam kurtosis (normal=3)
    return SharpeMoments(float(sr), n, skew, kurt)


def _sr_std_factor(m: SharpeMoments) -> float:
    """SR tahmininin standart sapması için düzeltme (skew/kurt dahil)."""
    val = 1.0 - m.skew * m.sr + (m.kurt - 1.0) / 4.0 * m.sr**2
    return math.sqrt(max(val, 1e-9))


def probabilistic_sharpe_ratio(m: SharpeMoments, benchmark_sr: float = 0.0) -> float:
    """P(gerçek SR > benchmark). Non-normalliği hesaba katar."""
    if m.n < 3:
        return 0.5
    stat = (m.sr - benchmark_sr) * math.sqrt(m.n - 1) / _sr_std_factor(m)
    return norm_cdf(stat)


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """N bağımsız deneme altında beklenen MAKSİMUM Sharpe (per-period)."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    e = math.e
    return math.sqrt(var_sr) * (
        (1 - GAMMA) * norm_ppf(1 - 1.0 / n_trials)
        + GAMMA * norm_ppf(1 - 1.0 / (n_trials * e)))


def deflated_sharpe_ratio(m: SharpeMoments, n_trials: int, var_sr: float) -> float:
    """
    DSR = PSR, ama benchmark = deneme sayısından beklenen maksimum Sharpe.
    DSR > 0.95 => Sharpe, seçilim etkisi düzeltildikten sonra bile anlamlı.
    """
    sr0 = expected_max_sharpe(var_sr, n_trials)
    return probabilistic_sharpe_ratio(m, benchmark_sr=sr0)


def bootstrap_sharpe_ci(returns: list[float], n_boot: int = 1000,
                        alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Yıllıklaştırılmış Sharpe için moving-block bootstrap güven aralığı."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 30:
        return (float("nan"), float("nan"))
    block = max(1, int(round(n ** (1/3))))   # otokorelasyona dayanıklı blok
    n_blocks = int(math.ceil(n / block))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        sd = sample.std(ddof=1)
        stats[i] = (sample.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.10) -> list[bool]:
    """BH-FDR: hangi hipotezler yanlış-keşif oranı kontrolünde hayatta kalır."""
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    survive = [False] * n
    max_k = -1
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= alpha * rank / n:
            max_k = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= max_k:
            survive[idx] = True
    return survive
