from __future__ import annotations

from datetime import datetime

import pandas as pd

from app import (
    DynamicLine,
    calculate_signal_risk_reward,
    detect_rejection_signals,
    find_target_for_signal,
    get_central_tz,
    is_call_rejection,
    is_put_rejection,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromisoformat(s), tz=get_central_tz())


def _candles(rows):
    idx = pd.DatetimeIndex([_ts(r[0]) for r in rows])
    return pd.DataFrame({"Open":[r[1] for r in rows],"High":[r[2] for r in rows],"Low":[r[3] for r in rows],"Close":[r[4] for r in rows]}, index=idx)


def test_call_rejection_confirmed_and_pending_and_invalids() -> None:
    line = DynamicLine("UD", 100, _ts("2026-04-28T08:00:00"), 0, "descending", "CALL_ZONE", "PRIMARY_HIGH", True, "")
    df = _candles([("2026-04-28T10:00:00",101,101.5,99.9,100.2),("2026-04-28T11:00:00",100.4,101,100,100.6)])
    assert is_call_rejection(df.iloc[0], line, df.index[0])
    sigs = detect_rejection_signals(df,[line],[])
    assert sigs[0].status=="CONFIRMED" and sigs[0].entry_price==100.4 and sigs[0].stop_price==99.4

    df2 = _candles([("2026-04-28T10:00:00",101,101.5,99.9,100.2)])
    sigs2 = detect_rejection_signals(df2,[line],[])
    assert sigs2[0].status=="PENDING_CONFIRMATION" and pd.isna(sigs2[0].entry_price)

    bad_close = _candles([("2026-04-28T10:00:00",101,101.5,99.9,99.8)])
    assert not is_call_rejection(bad_close.iloc[0], line, bad_close.index[0])


def test_put_rejection_confirmed_pending_invalids() -> None:
    line = DynamicLine("UA", 100, _ts("2026-04-28T08:00:00"), 0, "ascending", "PUT_ZONE", "PRIMARY_HIGH", True, "")
    df = _candles([("2026-04-28T10:00:00",99,100.2,98.9,99.8),("2026-04-28T11:00:00",99.7,99.9,99.2,99.4)])
    assert is_put_rejection(df.iloc[0], line, df.index[0])
    sigs = detect_rejection_signals(df,[line],[])
    assert sigs[0].status=="CONFIRMED" and sigs[0].entry_price==99.7 and sigs[0].stop_price==100.7

    df2 = _candles([("2026-04-28T10:00:00",99,100.2,98.9,99.8)])
    assert detect_rejection_signals(df2,[line],[])[0].status=="PENDING_CONFIRMATION"

    bad = _candles([("2026-04-28T10:00:00",99,100.2,98.9,100.1)])
    assert not is_put_rejection(bad.iloc[0], line, bad.index[0])


def test_trade_side_comes_from_touch_side_not_line_direction() -> None:
    descending = DynamicLine("UD", 100, _ts("2026-04-28T08:00:00"), 0, "descending", "CALL_ZONE", "PRIMARY_HIGH", True, "")
    below_touch = _candles([
        ("2026-04-28T10:00:00", 99.0, 100.2, 98.8, 99.7),
        ("2026-04-28T11:00:00", 99.6, 99.8, 99.1, 99.2),
    ])
    put_signals = detect_rejection_signals(below_touch, [descending], [])

    assert put_signals[0].signal_type == "PUT"
    assert put_signals[0].line_name == "UD"

    ascending = DynamicLine("UA", 100, _ts("2026-04-28T08:00:00"), 0, "ascending", "PUT_ZONE", "PRIMARY_HIGH", True, "")
    above_touch = _candles([
        ("2026-04-28T10:00:00", 101.0, 101.3, 99.9, 100.2),
        ("2026-04-28T11:00:00", 100.4, 101.0, 100.1, 100.8),
    ])
    call_signals = detect_rejection_signals(above_touch, [ascending], [])

    assert call_signals[0].signal_type == "CALL"
    assert call_signals[0].line_name == "UA"


def test_secondary_no_entries_targets_rr_and_order() -> None:
    ud = DynamicLine("UD",100,_ts("2026-04-28T08:00:00"),0,"descending","CALL_ZONE","PRIMARY_HIGH",True,"")
    sd = DynamicLine("S_DESC_001",102,_ts("2026-04-28T08:00:00"),0,"descending","TARGET_ONLY","SECONDARY",False,"")
    su = DynamicLine("S_ASC_001",99,_ts("2026-04-28T08:00:00"),0,"ascending","TARGET_ONLY","SECONDARY",False,"")
    df = _candles([
        ("2026-04-28T09:00:00",101,101.2,99.9,100.2),
        ("2026-04-28T10:00:00",100.5,100.7,100.1,100.3),
    ])
    sigs = detect_rejection_signals(df,[ud],[sd,su])
    assert len(sigs)==1
    assert sigs[0].target_line_name=="S_DESC_001"
    assert sigs[0].signal_id==f"CALL_UD_{df.index[0].isoformat()}"

    tn,tp = find_target_for_signal("PUT","UA",100,_ts("2026-04-28T10:00:00"),[ud,sd,su])
    assert tn=="S_ASC_001" and tp==99

    r,rew,rr = calculate_signal_risk_reward("CALL",100,99,102)
    assert r==1 and rew==2 and rr==2
    r2,rew2,rr2 = calculate_signal_risk_reward("PUT",100,101,98)
    assert r2==1 and rew2==2 and rr2==2
    r3,_,_ = calculate_signal_risk_reward("CALL",float("nan"),99,102)
    assert pd.isna(r3)


def test_tradable_value_usage_and_sorted_oldest_first() -> None:
    line = DynamicLine("UD",100.004,_ts("2026-04-28T08:00:00"),0,"descending","CALL_ZONE","PRIMARY_HIGH",True,"")
    # tradable rounds to 100.00
    row = _candles([("2026-04-28T10:00:00",100.01,100.2,100.0,100.01)])
    assert is_call_rejection(row.iloc[0], line, row.index[0])
    df = _candles([
        ("2026-04-28T11:00:00",100.01,100.2,100.0,100.01),
        ("2026-04-28T10:00:00",100.01,100.2,100.0,100.01),
    ])
    sigs = detect_rejection_signals(df,[line],[])
    assert sigs[0].rejection_time < sigs[-1].rejection_time
