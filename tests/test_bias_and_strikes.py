from __future__ import annotations

from datetime import datetime

import pandas as pd

from app import (
    DynamicLine,
    calculate_bias_strength,
    determine_preopen_bias,
    format_watch_contract,
    get_contract_watch_price,
    get_central_tz,
    get_line_by_name,
    OptionsIntelligence,
    SourceStatus,
    premium_flow_alignment,
    select_flow_aware_watch_contracts,
    select_watch_contracts,
    select_0dte_strikes,
    TradeSignal,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromisoformat(s), tz=get_central_tz())


def _lines() -> list[DynamicLine]:
    anc = _ts("2026-04-28T08:00:00")
    return [
        DynamicLine("UA", 100.005, anc, 0.0, "ascending", "PUT_ZONE", "PRIMARY_HIGH", True, ""),
        DynamicLine("UD", 99.995, anc, 0.0, "descending", "CALL_ZONE", "PRIMARY_HIGH", True, ""),
        DynamicLine("LA", 95.0, anc, 0.0, "ascending", "PUT_ZONE", "PRIMARY_LOW", True, ""),
        DynamicLine("LD", 94.0, anc, 0.0, "descending", "CALL_ZONE", "PRIMARY_LOW", True, ""),
    ]


def test_get_line_by_name() -> None:
    lines = _lines()
    assert get_line_by_name(lines, "UA") is not None
    assert get_line_by_name(lines, "ZZ") is None


def test_bullish_preopen() -> None:
    b = determine_preopen_bias(_lines(), 101.0, _ts("2026-04-29T08:30:00"))
    assert b.bias == "BULLISH"
    assert set(b.watched_call_lines) == {"UA", "UD", "LA", "LD"}
    assert b.watched_put_lines == []
    assert b.primary_line in {"UA", "UD"}
    s = select_0dte_strikes(101.0, _ts("2026-04-29T08:30:00"))
    assert format_watch_contract(s, bias_state=b) == "WATCH CALL 103"


def test_neutral_preopen() -> None:
    b = determine_preopen_bias(_lines(), 100.0, _ts("2026-04-29T08:30:00"))
    assert b.bias == "NEUTRAL"
    assert set(b.watched_call_lines) == {"LA", "LD"}
    assert b.watched_put_lines == []
    assert b.final_take_profit_line in {"LA", "LD"}
    s = select_0dte_strikes(100.0, _ts("2026-04-29T08:30:00"))
    assert format_watch_contract(s, bias_state=b) == "WATCH CALL 102"


def test_bearish_preopen() -> None:
    b = determine_preopen_bias(_lines(), 99.0, _ts("2026-04-29T08:30:00"))
    assert b.bias == "BEARISH"
    assert "LA" in b.watched_call_lines
    assert "UD" in b.watched_put_lines


def test_regular_session_mode() -> None:
    b = determine_preopen_bias(_lines(), 101.0, _ts("2026-04-29T09:00:00"))
    assert b.bias == "REGULAR_SESSION"
    assert "line-side confirmation" in b.explanation.lower()


def test_missing_ua_or_ud() -> None:
    lines = [l for l in _lines() if l.name != "UA"]
    b = determine_preopen_bias(lines, 100.0, _ts("2026-04-29T08:30:00"))
    assert b.bias == "UNKNOWN" and b.strength_score == 0


def test_bias_uses_tradable_values() -> None:
    b = determine_preopen_bias(_lines(), 100.00, _ts("2026-04-29T08:30:00"))
    assert b.bias == "NEUTRAL"


def test_bias_strength_bounds() -> None:
    assert 0 <= calculate_bias_strength(110, 100, 99, "BULLISH") <= 100
    assert 0 <= calculate_bias_strength(90, 100, 99, "BEARISH") <= 100
    assert calculate_bias_strength(float("nan"), 1, 2, "BULLISH") == 0


def test_strike_selection() -> None:
    s = select_0dte_strikes(712.61, _ts("2026-04-29T08:30:00"))
    assert s.call_strike == 715
    assert s.call_strike > s.underlying_price
    assert s.put_strike == 711
    assert s.put_strike < s.underlying_price


def test_strike_selection_whole_number() -> None:
    s = select_0dte_strikes(713.00, _ts("2026-04-29T08:30:00"))
    assert s.call_strike == 715
    assert s.call_strike > s.underlying_price
    assert s.put_strike == 711
    assert s.put_strike < s.underlying_price


def test_strike_selection_stays_near_two_points_otm() -> None:
    s = select_0dte_strikes(717.85, _ts("2026-04-29T08:30:00"))
    assert s.call_strike == 720
    assert s.put_strike == 716


def test_invalid_price() -> None:
    s = select_0dte_strikes(float("nan"), _ts("2026-04-29T08:30:00"))
    assert s.warning is not None


def test_watch_contracts_use_pending_signal_trigger_price() -> None:
    lines = _lines()
    sig = TradeSignal("p","PUT","PENDING_CONFIRMATION","UA",100,_ts("2026-04-29T10:00:00"),0,0,0,0,None,float("nan"),0,None,float("nan"),0,0,0,"","")
    s = select_watch_contracts(96.0, _ts("2026-04-29T10:00:00"), sig, lines)
    assert get_contract_watch_price(96.0, _ts("2026-04-29T10:00:00"), sig, lines) == 100.0
    assert s.put_strike == 98
    assert format_watch_contract(s, sig) == "WATCH PUT 98"


def test_watch_contracts_use_confirmed_signal_entry_price() -> None:
    lines = _lines()
    sig = TradeSignal("c","CALL","CONFIRMED","UD",100,_ts("2026-04-29T10:00:00"),0,0,0,0,_ts("2026-04-29T11:00:00"),101.2,0,None,float("nan"),0,0,0,"","")
    s = select_watch_contracts(96.0, _ts("2026-04-29T11:00:00"), sig, lines)
    assert s.underlying_price == 101.2
    assert s.call_strike == 103
    assert format_watch_contract(s, sig) == "WATCH CALL 103"


def test_flow_aware_contracts_use_nearby_otm_flow_not_far_chase() -> None:
    options = OptionsIntelligence(
        SourceStatus("Options intelligence", "connected", ""),
        1,
        1,
        718,
        724,
        715,
        [],
        unusual_whales={
            "flow_alerts": {
                "flow_bias": "Bullish flow",
                "largest_alerts": [
                    {"type": "CALL", "strike": 724, "premium": 900000},
                    {"type": "CALL", "strike": 720, "premium": 250000},
                    {"type": "PUT", "strike": 716, "premium": 220000},
                ],
                "key_strikes": [],
            }
        },
    )

    s = select_flow_aware_watch_contracts(717.85, _ts("2026-04-29T09:00:00"), options_intel=options)

    assert s.call_strike == 720
    assert s.put_strike == 716


def test_flow_alignment_warns_when_pressure_conflicts_with_watch_side() -> None:
    options = OptionsIntelligence(
        SourceStatus("Options intelligence", "connected", ""),
        1,
        1,
        718,
        724,
        715,
        [],
        unusual_whales={"flow_alerts": {"flow_bias": "Bearish flow"}, "market_tide": {"tone": "Risk-off options tide"}},
    )

    read = premium_flow_alignment(options, "CALL")

    assert read["state"] == "opposes"
    assert "Caution for call setup" in read["title"]
