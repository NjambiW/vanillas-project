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