from journal import Journal


def test_summary_of_wins_and_losses():
    j = Journal()
    a = j.open_trade(strategy="s", contract_id=1, stake=10, markup_pct=5.0)
    b = j.open_trade(strategy="s", contract_id=2, stake=10, markup_pct=7.0)
    c = j.open_trade(strategy="s", contract_id=3, stake=10, markup_pct=6.0)
    j.close_trade(a, 5.0, "won")
    j.close_trade(b, -2.0, "lost")
    s = j.summary()
    assert s["trades_closed"] == 2 and s["open"] == 1
    assert s["win_rate"] == 0.5 and s["total_profit"] == 3.0
    assert s["avg_win"] == 5.0 and s["avg_loss"] == -2.0 and s["expectancy"] == 1.5
    j.close_trade(c, None, "unknown")
    assert j.summary()["unknown"] == 1


def test_empty_summary_has_no_division_errors():
    s = Journal().summary()
    assert s["trades_closed"] == 0 and s["win_rate"] is None and s["expectancy"] is None


def test_skip_reasons_are_counted_most_common_first():
    j = Journal()
    for _ in range(3):
        j.log_skip("s", "CALL", "markup too high")
    j.log_skip("s", "PUT", "daily loss limit")
    reasons = j.skip_reasons()
    assert reasons[0] == ("markup too high", 3) and reasons[1] == ("daily loss limit", 1)


def test_signals_and_recent_trades_are_stored():
    j = Journal()
    j.log_signal("s", "CALL", "because", 1.5, 1000.0)
    assert j.db.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 1
    j.open_trade(strategy="s", contract_id=9, contract_type="VANILLALONGCALL", stake=2.0)
    assert j.recent_trades(5)[0][1] == "VANILLALONGCALL"


def test_performance_separates_markup_from_edge():
    from journal import verdict

    j = Journal()
    # 40 trades that pay exactly their fair value on average: win 0.5 of the time, paying
    # 2 * fair when winning. With 10% markup, stake 1.0 -> fair value 1/1.1 = 0.909.
    for i in range(40):
        t = j.open_trade(strategy="s", contract_id=i, stake=1.0, markup_pct=10.0)
        payoff = 2 * (1 / 1.1) if i % 2 == 0 else 0.0
        j.close_trade(t, payoff - 1.0, "won" if payoff > 1.0 else "lost")
    perf = j.performance()
    assert perf["trades"] == 40
    assert abs(perf["gross_roi"]) < 1e-9             # a fair game before the markup
    assert abs(perf["roi"] - (-1 / 11)) < 1e-9       # about -9.1% after the markup
    assert abs(perf["markup_cost"] - 40 * (1 - 1 / 1.1)) < 1e-9
    assert "No evidence of an edge" in verdict(perf)


def test_verdict_for_small_samples_and_empty_journal():
    from journal import verdict

    assert "No settled trades" in verdict(Journal().performance())
    j = Journal()
    t = j.open_trade(strategy="s", contract_id=1, stake=1.0, markup_pct=5.0)
    j.close_trade(t, 3.0, "won")
    assert "too few" in verdict(j.performance())


def test_open_trades_lists_only_unsettled_contracts():
    j = Journal()
    a = j.open_trade(strategy="s", contract_id=1, stake=1.0)
    b = j.open_trade(strategy="s", contract_id=2, stake=1.0)
    j.close_trade(a, 1.0, "won")
    assert j.open_trades() == [(b, "2", "s")]