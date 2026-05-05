from __future__ import annotations

from datetime import datetime

import pandas as pd

from app import (
    DEFAULT_SLOPE_PER_HOUR,
    DynamicLine,
    Pivot,
    SecondaryPivot,
    build_primary_lines,
    build_pivot_source_table,
    build_secondary_lines,
    build_structure_projection_table,
    calculate_slope_from_observed,
    get_central_tz,
    get_closest_primary_line,
    get_primary_anchor_summary,
    get_structure_calibration,
    market_hours_between,
    project_lines,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromisoformat(s), tz=get_central_tz())


def test_default_slope_constant() -> None:
    assert DEFAULT_SLOPE_PER_HOUR == 0.20


def test_structure_calibration_reads_env_without_ui(monkeypatch) -> None:
    monkeypatch.setenv("SPYPROPHET_STRUCTURE_CALIBRATION", "0.111")
    assert get_structure_calibration() == 0.111


def test_hours_since_uses_tradingview_active_chart_hours() -> None:
    line_desc = DynamicLine("X", 714.46, _ts("2026-04-28T14:00:00"), DEFAULT_SLOPE_PER_HOUR, "descending", "CALL_ZONE", "PRIMARY_HIGH", True, "")
    line_asc = DynamicLine("Y", 714.46, _ts("2026-04-28T14:00:00"), DEFAULT_SLOPE_PER_HOUR, "ascending", "PUT_ZONE", "PRIMARY_HIGH", True, "")
    now = _ts("2026-04-29T08:00:00")

    assert line_desc.hours_since(now) == 9
    assert abs(line_desc.raw_value_at(now) - 712.66) < 1e-9
    assert line_desc.tradable_value_at(now) == 712.66
    assert abs(line_asc.raw_value_at(now) - 716.26) < 1e-9
    assert line_asc.tradable_value_at(now) == 716.26


def test_two_pm_to_ten_am_matches_active_chart_calibration_pair() -> None:
    line = DynamicLine("UD", 714.46, _ts("2026-04-28T14:00:00"), DEFAULT_SLOPE_PER_HOUR, "descending", "CALL_ZONE", "PRIMARY_HIGH", True, "")
    next_ten = _ts("2026-04-29T10:00:00")

    assert line.hours_since(next_ten) == 11
    assert line.tradable_value_at(next_ten) == 712.26


def test_weekend_projection_skips_closed_chart_hours() -> None:
    friday_anchor = _ts("2026-05-01T09:00:00")
    monday_projection = _ts("2026-05-04T09:00:00")
    line = DynamicLine("UA", 724.87, friday_anchor, DEFAULT_SLOPE_PER_HOUR, "ascending", "PUT_ZONE", "PRIMARY_HIGH", True, "")

    assert market_hours_between(friday_anchor, monday_projection) == 15
    assert line.hours_since(monday_projection) == 15
    assert line.tradable_value_at(monday_projection) == 727.87


def test_friday_close_to_monday_morning_counts_chart_reopen_window() -> None:
    friday_close = _ts("2026-05-01T15:00:00")
    monday_projection = _ts("2026-05-04T09:00:00")
    line = DynamicLine("UA", 724.87, friday_close, DEFAULT_SLOPE_PER_HOUR, "ascending", "PUT_ZONE", "PRIMARY_HIGH", True, "")

    assert market_hours_between(friday_close, monday_projection) == 9
    assert line.tradable_value_at(monday_projection) == 726.67


def test_friday_two_pm_anchor_projects_to_monday_nine_am_tradingview_level() -> None:
    friday_two_pm = _ts("2026-05-01T14:00:00")
    monday_projection = _ts("2026-05-04T09:00:00")
    line = DynamicLine("LD", 720.47, friday_two_pm, DEFAULT_SLOPE_PER_HOUR, "descending", "CALL_ZONE", "PRIMARY_LOW", True, "")

    assert market_hours_between(friday_two_pm, monday_projection) == 10
    assert line.tradable_value_at(monday_projection) == 718.47


def test_calibration_helper() -> None:
    s_desc = calculate_slope_from_observed(714.46, 712.61, 18, "descending")
    assert abs(s_desc - 0.1027777778) < 1e-7
    s_asc = calculate_slope_from_observed(100, 101, 4, "ascending")
    assert s_asc == 0.25


def test_build_primary_lines_and_override() -> None:
    hp = Pivot("HIGH_PIVOT", 714.46, _ts("2026-04-28T14:00:00"), "x", "green", False)
    lp = Pivot("LOW_PIVOT", 700.12, _ts("2026-04-28T10:00:00"), "x", "red", False)
    lines = build_primary_lines(hp, lp)
    assert [l.name for l in lines] == ["UA", "UD", "LA", "LD"]
    assert [l.zone_type for l in lines] == ["PUT_ZONE", "CALL_ZONE", "PUT_ZONE", "CALL_ZONE"]
    assert [l.direction for l in lines] == ["ascending", "descending", "ascending", "descending"]
    assert all(l.is_primary for l in lines)
    assert all(l.slope_per_hour == DEFAULT_SLOPE_PER_HOUR for l in lines)

    lines2 = build_primary_lines(hp, lp, slope_per_hour=0.104)
    assert all(l.slope_per_hour == 0.104 for l in lines2)


def test_build_secondary_lines() -> None:
    pivs = [
        SecondaryPivot("A", 10, _ts("2026-04-28T09:00:00"), "ascending", "secondary_transition"),
        SecondaryPivot("B", 9, _ts("2026-04-28T10:00:00"), "descending", "secondary_transition"),
    ]
    lines = build_secondary_lines(pivs)
    assert lines[0].direction == "ascending" and lines[0].zone_type == "TARGET_ONLY"
    assert lines[1].direction == "descending" and lines[1].zone_type == "TARGET_ONLY"
    assert all(not l.is_primary and l.source == "SECONDARY" for l in lines)


def test_distance_and_percent() -> None:
    line = DynamicLine("X", 100, _ts("2026-04-28T14:00:00"), 0.103, "descending", "CALL_ZONE", "PRIMARY_HIGH", True, "")
    now = _ts("2026-04-28T15:00:00")
    tradable = line.tradable_value_at(now)
    dist = line.distance_from_price(tradable + 1, now)
    assert dist == 1
    assert line.distance_from_price(tradable - 1, now) == -1
    raw_dist = line.distance_from_price(line.raw_value_at(now) + 1, now, use_tradable_value=False)
    assert abs(raw_dist - 1) < 1e-9
    pct = line.percent_distance_from_price(tradable + 1, now)
    assert pct > 0
    assert pd.isna(line.percent_distance_from_price(0, now))


def test_project_lines_and_closest() -> None:
    now = _ts("2026-04-29T08:00:00")
    hp = Pivot("HIGH_PIVOT", 714.46, _ts("2026-04-28T14:00:00"), "x", "green", False)
    lp = Pivot("LOW_PIVOT", 700.00, _ts("2026-04-28T14:00:00"), "x", "red", False)
    prim = build_primary_lines(hp, lp)
    sec = build_secondary_lines([SecondaryPivot("A", 710, _ts("2026-04-28T12:00:00"), "ascending", "secondary_transition")])
    df = project_lines(prim + sec, now, 712.50)
    required = {"name","raw_projected_value","tradable_value","distance","abs_distance","percent_distance","direction","zone_type","source","is_primary","anchor_price","anchor_time","slope_per_hour","description"}
    assert required.issubset(set(df.columns))
    row = df[df["name"] == "UD"].iloc[0]
    assert row["distance"] == 712.50 - row["tradable_value"]

    closest = get_closest_primary_line(prim + sec, now, 712.50)
    assert closest is not None and closest.is_primary


def test_structure_tables_explain_source_and_projection() -> None:
    idx = pd.DatetimeIndex([
        _ts("2026-04-28T08:30:00"),
        _ts("2026-04-28T09:30:00"),
        _ts("2026-04-28T14:30:00"),
    ])
    candles = pd.DataFrame({
        "Open": [100.0, 102.0, 105.0],
        "High": [103.0, 104.0, 110.0],
        "Low": [99.0, 98.0, 104.0],
        "Close": [102.0, 103.0, 106.0],
    }, index=idx)

    source = build_pivot_source_table(candles)

    assert list(source["Pivot"]) == ["High Pivot", "Low Pivot"]
    assert source.iloc[0]["Source"] == "Yahoo SPY 60m RTH"
    assert source.iloc[0]["Pivot Price"] == 110.0
    assert source.iloc[1]["Pivot Price"] == 98.0

    hp = Pivot("HIGH_PIVOT", 110.0, _ts("2026-04-28T15:00:00"), "session_high", "green", False)
    lp = Pivot("LOW_PIVOT", 98.0, _ts("2026-04-28T10:30:00"), "session_low", "red", False)
    projection = build_structure_projection_table(build_primary_lines(hp, lp), _ts("2026-04-29T09:00:00"), 112.0, idx[0].date(), idx[0].date())

    assert "UA" not in set(projection["Trigger"])
    assert "Upper Ascending Trigger" in set(projection["Trigger"])
    assert "Formula" not in projection.columns
    assert "Slope / Hour" not in projection.columns
    assert "Projection Method" in projection.columns
    assert projection[projection["Trigger"] == "Upper Descending Trigger"].iloc[0]["Based On"] == "High Pivot"


def test_primary_anchor_summary() -> None:
    hp = Pivot("HIGH_PIVOT", 110.0, _ts("2026-04-28T15:00:00"), "session_high", "green", False)
    lp = Pivot("LOW_PIVOT", 98.0, _ts("2026-04-28T10:30:00"), "session_low", "red", False)
    summary = get_primary_anchor_summary(build_primary_lines(hp, lp))

    assert summary["high_time"] == _ts("2026-04-28T15:00:00")
    assert summary["high_price"] == 110.0
    assert summary["low_time"] == _ts("2026-04-28T10:30:00")
    assert summary["low_price"] == 98.0


def test_invalid_anchor_handling() -> None:
    now = _ts("2026-04-29T08:00:00")
    bad_time = DynamicLine("BT", 10, None, 0.103, "descending", "CALL_ZONE", "X", True, "")
    assert pd.isna(bad_time.hours_since(now))
    assert pd.isna(bad_time.raw_value_at(now))
    assert pd.isna(bad_time.value_at(now))
    assert pd.isna(bad_time.tradable_value_at(now))
    assert pd.isna(bad_time.distance_from_price(10, now))

    bad_price = DynamicLine("BP", float("nan"), _ts("2026-04-28T14:00:00"), 0.103, "descending", "CALL_ZONE", "X", True, "")
    assert pd.isna(bad_price.raw_value_at(now))
