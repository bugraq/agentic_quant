"""
Bütçe tahsisi — Thompson Sampling bandit (Doküman 16.2 / Deney E).

Araştırma bütçesi strateji aileleri arasında dağıtılır. Her aile bir "kol"dur;
başarısı Beta(1+kabul, 1+başarısızlık) ile modellenir. Her turda her aileden
bir örnek çekilir, en yüksek örneği veren aile seçilir:
  - Başarılı aile (momentum) -> yüksek örnekler -> daha çok bütçe (exploit)
  - Hiç denenmemiş aile -> Beta(1,1) uniform -> keşfedilme şansı (explore)
  - Sürekli başarısız aile -> düşük örnekler -> geçici olarak dondurulur

Bu, "keşfet sonra champion'ı revize et"in üstüne aile düzeyinde stratejik
bütçe kontrolü ekler.
"""
from __future__ import annotations

import numpy as np


class ThompsonBandit:
    def __init__(self, families: list[str], seed: int = 0) -> None:
        self.families = families
        self.rng = np.random.default_rng(seed)

    def select(self, counts: dict[str, tuple[int, int]]) -> str:
        """
        counts: family -> (kabul, toplam_backtest). Beta örneklemesiyle aile seç.
        """
        best_family, best_sample = self.families[0], -1.0
        for fam in self.families:
            accepts, total = counts.get(fam, (0, 0))
            fails = max(0, total - accepts)
            sample = self.rng.beta(1 + accepts, 1 + fails)
            if sample > best_sample:
                best_sample, best_family = sample, fam
        return best_family
