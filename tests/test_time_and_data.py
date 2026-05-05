from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from app import (
    ensure_central_index,
    filter_active_chart_session,
    filter_extended_session,
    filter_rth_session,
    get_central_tz,
    default_session_date,
    get_latest_available_trading_day,
    get_live_signal_day,
    latest_price_for_session,
    next_session_after,
    next_session_date,
    get_prior_trading_day,
    resolve_session_clock,
    get_structure_projection_time,
    rth_session_window_label,
)


def _sample_day_frame(day: str = "2026-04-28") -> pd.DataFrame:
    idx = pd.date_range(f"{day} 02:00", f"{day} 20:00", freq="1min", tz=get_central_tz())
    return pd.DataFrame({"Close": range(len(idx))}, index=idx)


def test_timezone_conversion_to_central_from_aware_utc() -> None:
    utc_idx = pd.date_range("2026-04-28 13:00", periods=2, freq="60min", tz="UTC")
    df = pd.DataFrame({"Close": [1.0, 2.0]}, index=utc_idx)

    out = ensure_central_index(df)

    assert out.index.tz is not None
    assert str(out.index.tz) in {"US/Central", "America/Chicago"}
    assert out.index[0].hour == 8


def test_timezone_conversion_to_central_from_naive_assumed_utc() -> None:
    naive_idx = pd.date_range("2026-04-28 13:00", periods=1, freq="60min")
    df = pd.DataFrame({"Close": [1.0]}, index=naive_idx)

    out = ensure_central_index(df)

    assert out.index.tz is not None
    assert out.index[0].hour == 8


def test_rth_filter_boundaries() -> None:
    df = _sample_day_frame()
    day = date(2026, 4, 28)

    rth = filter_rth_session(df, day)

    assert not rth.empty
    assert datetime(2026, 4, 28, 8, 30, tzinfo=get_central_tz()) in rth.index
    assert datetime(2026, 4, 28, 15, 0, tzinfo=get_central_tz()) in rth.index
    assert datetime(2026, 4, 28, 8, 29, tzinfo=get_central_tz()) not in rth.index
    assert datetime(2026, 4, 28, 15, 1, tzinfo=get_central_tz()) not in rth.index


def test_rth_session_window_label_matches_strategy_window() -> None:
    assert rth_session_window_label() == "8:30-3:00 CT"


def test_structure_projection_time_uses_9am_decision_anchor() -> None:
    early = datetime(2026, 5, 1, 3, 15, tzinfo=get_central_tz())
    regular = datetime(2026, 5, 1, 9, 30, tzinfo=get_central_tz())
    after_close = datetime(2026, 5, 1, 18, 15, tzinfo=get_central_tz())

    assert get_structure_projection_time(early) == datetime(2026, 5, 1, 9, 0, tzinfo=get_central_tz())
    assert get_structure_projection_time(regular) == datetime(2026, 5, 1, 9, 0, tzinfo=get_central_tz())
    assert get_structure_projection_time(after_close) == datetime(2026, 5, 1, 9, 0, tzinfo=get_central_tz())


def test_rth_filter_excludes_after_close_hourly_bar() -> None:
    idx = pd.DatetimeIndex([
        datetime(2026, 4, 28, 8, 30, tzinfo=get_central_tz()),
        datetime(2026, 4, 28, 9, 30, tzinfo=get_central_tz()),
        datetime(2026, 4, 28, 14, 30, tzinfo=get_central_tz()),
        datetime(2026, 4, 28, 15, 0, tzinfo=get_central_tz()),
    ])
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0]}, index=idx)

    rth = filter_rth_session(df, date(2026, 4, 28))

    assert datetime(2026, 4, 28, 14, 30, tzinfo=get_central_tz()) in rth.index
    assert datetime(2026, 4, 28, 15, 0, tzinfo=get_central_tz()) not in rth.index


def test_extended_filter_boundaries() -> None:
    df = _sample_day_frame()
    day = date(2026, 4, 28)

    ext = filter_extended_session(df, day)

    assert not ext.empty
    assert datetime(2026, 4, 28, 3, 0, tzinfo=get_central_tz()) in ext.index
    assert datetime(2026, 4, 28, 19, 0, tzinfo=get_central_tz()) in ext.index
    assert datetime(2026, 4, 28, 2, 59, tzinfo=get_central_tz()) not in ext.index
    assert datetime(2026, 4, 28, 19, 1, tzinfo=get_central_tz()) not in ext.index


def test_active_chart_filter_matches_projection_window() -> None:
    df = _sample_day_frame()
    day = date(2026, 4, 28)

    active = filter_active_chart_session(df, day)

    assert not active.empty
    assert datetime(2026, 4, 28, 3, 0, tzinfo=get_central_tz()) in active.index
    assert datetime(2026, 4, 28, 18, 0, tzinfo=get_central_tz()) in active.index
    assert datetime(2026, 4, 28, 2, 59, tzinfo=get_central_tz()) not in active.index
    assert datetime(2026, 4, 28, 18, 1, tzinfo=get_central_tz()) not in active.index


def test_prior_trading_day_skips_missing_days() -> None:
    idx = pd.DatetimeIndex([
        datetime(2026, 4, 25, 9, 30, tzinfo=get_central_tz()),
        datetime(2026, 4, 29, 9, 30, tzinfo=get_central_tz()),
    ])
    df = pd.DataFrame({"Close": [1.0, 2.0]}, index=idx)

    cur = datetime(2026, 4, 30, 10, 0, tzinfo=get_central_tz())
    prior = get_prior_trading_day(df, cur)

    assert prior == date(2026, 4, 29)


def test_live_signal_day_uses_current_day_before_first_loaded_candle() -> None:
    idx = pd.DatetimeIndex([
        datetime(2026, 4, 29, 9, 30, tzinfo=get_central_tz()),
        datetime(2026, 4, 30, 9, 30, tzinfo=get_central_tz()),
    ])
    df = pd.DataFrame({"Close": [1.0, 2.0]}, index=idx)

    cur = datetime(2026, 5, 1, 7, 0, tzinfo=get_central_tz())

    assert get_latest_available_trading_day(df, cur) == date(2026, 4, 30)
    assert get_live_signal_day(df, cur) == date(2026, 5, 1)
    assert get_prior_trading_day(df, datetime(2026, 5, 1, tzinfo=get_central_tz())) == date(2026, 4, 30)


def test_session_date_defaults_to_monday_on_weekend() -> None:
    df = _sample_day_frame("2026-05-01")
    saturday = datetime(2026, 5, 2, 9, 0, tzinfo=get_central_tz())

    assert next_session_date(saturday.date()) == date(2026, 5, 4)
    assert next_session_after(date(2026, 5, 1)) == date(2026, 5, 4)
    assert default_session_date(df, saturday) == date(2026, 5, 4)


def test_future_session_clock_uses_nine_am_preview() -> None:
    df = _sample_day_frame("2026-05-01")
    monday = date(2026, 5, 4)

    session_clock = resolve_session_clock(df, monday, datetime(2026, 5, 2, 9, 0, tzinfo=get_central_tz()))

    assert session_clock == pd.Timestamp("2026-05-04 09:00", tz=get_central_tz())
    assert latest_price_for_session(df, monday, session_clock.to_pydatetime()) == float(df["Close"].iloc[-1])


def test_historical_session_uses_that_day_close() -> None:
    df = _sample_day_frame("2026-05-01")
    session_clock = resolve_session_clock(df, date(2026, 5, 1), datetime(2026, 5, 4, 9, 0, tzinfo=get_central_tz()))

    assert session_clock.date() == date(2026, 5, 1)
    assert session_clock.hour == 15
    assert session_clock.minute == 0
    assert latest_price_for_session(df, date(2026, 5, 1), session_clock.to_pydatetime()) == float(filter_rth_session(df, date(2026, 5, 1))["Close"].iloc[-1])


def test_empty_dataframe_handling() -> None:
    empty_df = pd.DataFrame()

    assert ensure_central_index(empty_df).empty
    assert filter_rth_session(empty_df, date(2026, 4, 28)).empty
    assert filter_extended_session(empty_df, date(2026, 4, 28)).empty
    assert get_prior_trading_day(empty_df, datetime(2026, 4, 30, tzinfo=get_central_tz())) is None
