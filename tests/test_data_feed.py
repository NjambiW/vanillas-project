import tempfile
from pathlib import Path

from data_feed import CandleBuilder, candles_to_frame, load_candles_csv, save_candles_csv


def test_ticks_in_one_bucket_build_ohlc():
    b = CandleBuilder(60)
    assert b.on_tick(120, 10.0) is None
    assert b.on_tick(130, 12.0) is None
    assert b.on_tick(140, 9.0) is None
    assert b.on_tick(150, 11.0) is None
    closed = b.on_tick(180, 11.5)  # first tick of the next minute closes the candle
    assert closed == {"epoch": 120, "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0}


def test_forming_candle_is_excluded_by_default():
    b = CandleBuilder(60)
    b.on_tick(120, 10.0)
    b.on_tick(180, 11.0)
    assert len(b.frame()) == 1
    assert len(b.frame(include_forming=True)) == 2


def test_late_tick_is_ignored():
    b = CandleBuilder(60)
    b.on_tick(180, 10.0)
    assert b.on_tick(100, 99.0) is None
    assert b.frame(include_forming=True)["high"].iloc[-1] == 10.0


def test_seed_treats_last_candle_as_forming():
    b = CandleBuilder(60)
    b.seed([
        {"epoch": 120, "open": 1, "high": 2, "low": 1, "close": 2},
        {"epoch": 180, "open": 2, "high": 3, "low": 2, "close": 3},
        {"epoch": 240, "open": 3, "high": 4, "low": 3, "close": 4},
    ])
    assert len(b.frame()) == 2
    closed = b.on_tick(250, 5.0)  # still the 240 candle, updates it
    assert closed is None
    closed = b.on_tick(300, 6.0)
    assert closed["epoch"] == 240 and closed["high"] == 5.0 and closed["close"] == 5.0


def test_candles_to_frame_sorts_dedupes_and_converts_text():
    rows = [
        {"epoch": 180, "open": "2", "high": "3", "low": "1", "close": "2.5"},
        {"epoch": 120, "open": "1", "high": "2", "low": "1", "close": "1.5"},
        {"epoch": 120, "open": "1", "high": "2", "low": "1", "close": "1.5"},
    ]
    df = candles_to_frame(rows)
    assert df["epoch"].tolist() == [120, 180]
    assert df["close"].tolist() == [1.5, 2.5]
    assert str(df.index.tz) == "UTC"


def test_empty_history_gives_empty_frame():
    assert len(candles_to_frame([])) == 0


def test_csv_roundtrip():
    rows = [{"epoch": 60 * i, "open": i, "high": i + 1, "low": i - 1, "close": i + 0.5} for i in range(1, 6)]
    df = candles_to_frame(rows)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data" / "x.csv"
        save_candles_csv(df, path)
        back = load_candles_csv(path)
    assert back["close"].tolist() == df["close"].tolist()
    assert back["epoch"].tolist() == df["epoch"].tolist()


def test_candles_from_ticks_groups_into_buckets():
    from data_feed import candles_from_ticks

    ticks = [{"epoch": 100 + i, "price": float(p)} for i, p in enumerate([5, 7, 4, 6, 8, 3])]
    out = candles_from_ticks(ticks, 3)
    assert out[0] == {"epoch": 99, "open": 5.0, "high": 7.0, "low": 5.0, "close": 7.0}
    assert out[1] == {"epoch": 102, "open": 4.0, "high": 8.0, "low": 4.0, "close": 8.0}
    assert out[2]["epoch"] == 105 and out[2]["close"] == 3.0
    assert candles_from_ticks([], 10) == []


def test_feed_builds_unusual_candle_size_from_ticks():
    import asyncio

    from data_feed import CandleFeed

    class FakeClient:
        async def get_tick_history(self, symbol, count, end="latest"):
            return [{"epoch": 100 + i, "price": float(i)} for i in range(60)]

        async def subscribe_ticks(self, symbol, cb):
            return {"tick": {"quote": 60.0, "epoch": 160}, "subscription": {"id": "s"}}

        async def get_candles(self, *a, **k):
            raise AssertionError("must not request a 10s candle from the API")

    async def go():
        feed = CandleFeed(FakeClient(), "X", granularity=10, history_count=5)
        await feed.start()
        return feed.frame()

    frame = asyncio.run(go())
    assert len(frame) >= 4
    assert (frame["epoch"] % 10 == 0).all()