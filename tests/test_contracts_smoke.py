"""
Duman testi: contract'lar dokümandaki örnek hipotezi (4.4) temsil edebiliyor mu?
Henüz pytest kurmadık; bu dosya doğrudan `python -m tests.test_contracts_smoke`
ile çalışır ve hata yoksa "OK" basar.
"""
from contracts import (
    EconomicMechanism,
    Execution,
    Expression,
    Falsification,
    HypothesisFamily,
    HypothesisSpec,
    NamedFeature,
    Portfolio,
    ResearchContext,
    Universe,
)


def build_example_hypothesis() -> HypothesisSpec:
    """Doküman 4.4: 'Volume-confirmed short-term reversal'."""
    residual = NamedFeature(
        name="residual_return_20d",
        expression=Expression(
            op="residual_return", window=20, params={"benchmark": "market"}
        ),
    )
    abnormal_vol = NamedFeature(
        name="abnormal_volume_3d",
        expression=Expression(
            op="ratio",
            inputs=[
                Expression(op="rolling_mean", window=3,
                           inputs=[Expression(op="field", field="volume")]),
                Expression(op="rolling_mean", window=60,
                           inputs=[Expression(op="field", field="volume")]),
            ],
        ),
    )
    # signal = negate(residual_return_20d) * abnormal_volume_3d
    signal = Expression(
        op="multiply",
        inputs=[
            Expression(op="negate", inputs=[Expression(op="feature_ref",
                                                       name="residual_return_20d")]),
            Expression(op="feature_ref", name="abnormal_volume_3d"),
        ],
    )
    return HypothesisSpec(
        hypothesis_id="hyp_000123",
        title="Volume-confirmed short-term reversal",
        claim=("Stocks with strongly negative residual returns and abnormal volume "
               "are likely to reverse over the following five trading days."),
        family=HypothesisFamily.reversal,
        economic_mechanism=EconomicMechanism(
            type="behavioral_reversal",
            description="Temporary selling pressure causes overshoot.",
            expected_failure_conditions=["persistent selloffs", "crisis periods"],
        ),
        universe=Universe(source="sp500_point_in_time", minimum_price=5.0,
                          minimum_median_dollar_volume=10_000_000),
        features=[residual, abnormal_vol],
        signal=signal,
        portfolio=Portfolio(type="cross_sectional_long_short", long_quantile=0.1,
                            short_quantile=0.1, sector_neutral=True),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=5, rebalance="daily"),
        falsification=Falsification(minimum_oos_sharpe=0.5, maximum_turnover=20.0,
                                    maximum_drawdown=0.25,
                                    minimum_positive_walk_forward_folds=0.7),
    )


def main() -> None:
    hyp = build_example_hypothesis()

    # 1) round-trip: JSON'a serialize, geri parse — şema tutuyor mu?
    as_json = hyp.model_dump_json()
    parsed = HypothesisSpec.model_validate_json(as_json)
    assert parsed == hyp, "round-trip başarısız"

    # 2) bilinmeyen alan reddediliyor mu? (extra=forbid)
    try:
        Universe(source="x", bogus_field=1)  # type: ignore[call-arg]
        raise AssertionError("bilinmeyen alan kabul edildi — extra=forbid çalışmıyor")
    except Exception:
        pass

    # 3) holding_period_days > 0 kısıtı çalışıyor mu?
    try:
        Execution(signal_time="close_t", trade_time="open_t_plus_1",
                  holding_period_days=0)
        raise AssertionError("holding_period_days=0 kabul edildi")
    except Exception:
        pass

    # 4) ResearchContext kurulabiliyor mu?
    ctx = ResearchContext(
        campaign_goal="cross-sectional reversal ara",
        universe_description="500 ABD hissesi, point-in-time, günlük bar",
        allowed_operators=["return", "rolling_mean", "cross_sectional_rank", "negate"],
    )
    assert ctx.generation_mode.value == "new"

    print("OK — contract'lar dokümandaki örneği temsil ediyor, round-trip tutuyor.")


if __name__ == "__main__":
    main()
