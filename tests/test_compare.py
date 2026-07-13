"""
Model karşılaştırma koşucusu testi (offline — LLM yok).

random-search ve dummy sağlayıcıları sentetik veride yarıştırır; metrik
tablosunun kurulduğunu ve alanların tutarlı olduğunu doğrular.
"""
import os
import tempfile

from agents.quant_critic import DummyCritic
from compare import print_table, run_contestant
from data.synthetic import gen_cross_sectional_momentum, split_by_fraction
from orchestrator import CampaignConfig


def _cfg() -> CampaignConfig:
    return CampaignConfig(
        allowed_fields=["close", "volume", "dollar_volume"],
        allowed_operators=["return", "rolling_mean", "rolling_std", "zscore",
                           "ewma", "volatility", "delta", "negate", "multiply",
                           "subtract", "cross_sectional_rank"],
        allowed_horizons=[5, 10, 20, 60],
        allowed_rebalance=["daily", "weekly"],
        portfolio_types=["cross_sectional_long_short", "long_only"],
        max_experiments=5, min_acceptance_sharpe=0.3)


def test_two_contestants_compared():
    data, _ = split_by_fraction(gen_cross_sectional_momentum(seed=1), 0.7)
    cfg = _cfg()
    tmp = tempfile.mkdtemp()
    results = []
    for contestant in ({"label": "random-search", "provider": "random", "seed": 3},
                       {"label": "dummy-katalog", "provider": "dummy"}):
        db = os.path.join(tmp, f"cmp_{contestant['label']}.sqlite")
        r = run_contestant(contestant, data, cfg, DummyCritic(), db)
        results.append(r)
        assert r.total_records > 0, f"{r.label}: hiç kayıt yok"
        assert r.backtested >= r.distinct >= 0
        assert r.tokens == 0, "offline sağlayıcıda token olmamalı"
        assert os.path.exists(db), "yarışmacı hafıza DB'si yazılmadı"

    labels = {r.label for r in results}
    assert labels == {"random-search", "dummy-katalog"}
    print_table(results, budget=cfg.max_experiments)
    print("  [ok] iki yarışmacı koştu, metrik tablosu kuruldu")


def main():
    test_two_contestants_compared()
    print("OK — model karşılaştırma koşucusu çalışıyor.")


if __name__ == "__main__":
    main()
