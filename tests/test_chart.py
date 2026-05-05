from __future__ import annotations
from datetime import datetime
import pandas as pd
from app import DynamicLine, Pivot, SecondaryPivot, TradeSignal, build_prophet_chart, build_structure_map_svg, build_structure_path_chart, get_central_tz, select_secondary_lines_for_chart


def _ts(s: str):
    return pd.Timestamp(datetime.fromisoformat(s), tz=get_central_tz())


def _candles():
    idx = pd.DatetimeIndex([_ts("2026-04-28T10:00:00"), _ts("2026-04-28T11:00:00")])
    return pd.DataFrame({"Open":[100,101],"High":[101,102],"Low":[99,100],"Close":[100.5,101.2]}, index=idx)


def _lines():
    t = _ts("2026-04-28T09:00:00")
    return [
        DynamicLine("UA",100.004,t,0,"ascending","PUT_ZONE","PRIMARY_HIGH",True,""),
        DynamicLine("UD",100.004,t,0,"descending","CALL_ZONE","PRIMARY_HIGH",True,""),
        DynamicLine("LA",99.5,t,0,"ascending","PUT_ZONE","PRIMARY_LOW",True,""),
        DynamicLine("LD",99.5,t,0,"descending","CALL_ZONE","PRIMARY_LOW",True,""),
    ]


def test_empty_chart():
    fig = build_prophet_chart(pd.DataFrame(), [], [], None, None, [], [], None, float("nan"), _ts("2026-04-28T12:00:00"))
    assert fig is not None
    assert any("No candle data" in a.text for a in fig.layout.annotations)


def test_basic_chart_and_tradable_values():
    df=_candles(); lines=_lines()
    fig=build_prophet_chart(df, lines, [], Pivot("H",101,_ts("2026-04-28T11:00:00"),"x","green",False), Pivot("L",99,_ts("2026-04-28T10:00:00"),"x","red",False), [], [], None, 101.2, _ts("2026-04-28T12:00:00"))
    names=[t.name for t in fig.data]
    assert any(n=="SPY" for n in names)
    assert all(x in names for x in ["Upper Ascending Trigger","Upper Descending Trigger","Lower Ascending Trigger","Lower Descending Trigger"])
    ua_trace = [t for t in fig.data if t.name=="Upper Ascending Trigger"][0]
    assert all(abs(y-100.0)<1e-9 for y in ua_trace.y)  # tradable rounds 100.004 -> 100.00
    assert fig.layout.margin.b >= 135
    assert fig.layout.legend.y < 0


def test_secondary_modes_and_signal_markers():
    df=_candles(); lines=_lines(); t=_ts("2026-04-28T09:00:00")
    second=[DynamicLine(f"S{i}",100+i*0.1,t,0,"ascending","TARGET_ONLY","SECONDARY",False,"") for i in range(20)]
    sel12 = select_secondary_lines_for_chart(second, 101, _ts("2026-04-28T12:00:00"), "nearest 12")
    sel6 = select_secondary_lines_for_chart(second, 101, _ts("2026-04-28T12:00:00"), "nearest 6")
    assert len(sel12)<=12 and len(sel6)<=6

    sig=TradeSignal("id","CALL","CONFIRMED","UD",100,_ts("2026-04-28T10:00:00"),100,101,99,100.2,_ts("2026-04-28T11:00:00"),101,98.5,"UA",102,2.5,1,0.4,"be","x")
    fig=build_prophet_chart(df,lines,second,None,None,[],[sig],None,101,_ts("2026-04-28T12:00:00"),show_trade_overlays=True)
    names=[t.name for t in fig.data]
    assert any("signal" in str(n).lower() for n in names)
    assert any(n=="ENTRY" for n in names)


def test_structure_path_chart_novice_view():
    df=_candles(); lines=_lines()
    fig=build_structure_path_chart(df, lines, [], [], None, 101.2, _ts("2026-04-28T12:00:00"))
    names=[t.name for t in fig.data]
    assert "SPY path" in names
    assert "Upper trade zone" in names
    assert "Upper Ascending Trigger" in names
    assert fig.layout.height == 700
    assert fig.layout.margin.b >= 120


def test_animated_structure_map_svg():
    df=_candles(); lines=_lines()
    html=build_structure_map_svg(df, lines, [], [], None, 101.2, _ts("2026-04-28T12:00:00"), title="Test Map")
    assert "<svg" in html
    assert "Animated structure map" in html
    assert "repeat(auto-fit,minmax(210px,1fr))" in html
    assert "Upper Ascending Trigger" in html
    assert "SPY 101.20" in html
    assert "Candlestick" not in html


def test_structure_map_marks_nine_am_decision_time():
    idx = pd.DatetimeIndex([
        _ts("2026-04-28T03:00:00"),
        _ts("2026-04-28T09:00:00"),
        _ts("2026-04-28T10:00:00"),
    ])
    df = pd.DataFrame({"Open":[100,101,102],"High":[101,102,103],"Low":[99,100,101],"Close":[100.5,101.5,102.5]}, index=idx)

    html = build_structure_map_svg(df, _lines(), [], [], None, 102.5, _ts("2026-04-28T10:00:00"), title="Test Map")

    assert "9 AM Decision" in html
    assert "decision-time-marker" in html
