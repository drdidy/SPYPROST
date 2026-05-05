from __future__ import annotations
from datetime import datetime
import pandas as pd
from app import TradeSignal, filter_replay_day, get_available_replay_dates, evaluate_signal_outcome, build_replay_state, get_central_tz, get_latest_active_signal, signal_target_milestones


def _ts(s): return pd.Timestamp(datetime.fromisoformat(s), tz=get_central_tz())


def _full_df():
    idx = pd.DatetimeIndex([
        _ts("2026-04-28T08:30:00"),_ts("2026-04-28T09:30:00"),_ts("2026-04-28T10:30:00"),
        _ts("2026-04-29T08:30:00"),_ts("2026-04-29T09:30:00"),_ts("2026-04-29T10:30:00"),_ts("2026-04-29T11:30:00")])
    return pd.DataFrame({"Open":[100,101,102,103,102,101,100],"High":[101,102,103,104,103,102,101],"Low":[99,100,101,102,101,100,99],"Close":[100.5,101.5,102.5,103.4,102.2,101.1,100.2]}, index=idx)


def test_filter_and_dates():
    df=_full_df(); d=filter_replay_day(df, datetime(2026,4,29).date())
    assert all(x.date()==datetime(2026,4,29).date() for x in d.index)
    assert get_available_replay_dates(df)==sorted(get_available_replay_dates(df))


def test_build_replay_state_step_limits_and_outcomes_hidden():
    df=_full_df(); rt=_ts("2026-04-29T09:30:00")
    rs=build_replay_state(df, datetime(2026,4,29).date(), replay_time=rt, include_future_outcomes=False)
    assert rs.replay_time==rt
    assert rs.outcomes=={} or "FUTURE_OUTCOMES_HIDDEN" in rs.warnings


def test_build_replay_state_full_day_and_prior_day():
    df=_full_df(); rs=build_replay_state(df, datetime(2026,4,29).date(), include_future_outcomes=True)
    assert rs.prior_trading_day==datetime(2026,4,28).date()
    assert rs.primary_lines is not None


def test_outcome_call_put_ambiguous_nohit_pending_unknown_and_bars_moves():
    fut = pd.DataFrame({"Open":[100,100],"High":[103,101],"Low":[99,98],"Close":[100.5,100]}, index=pd.DatetimeIndex([_ts("2026-04-29T10:30:00"),_ts("2026-04-29T11:30:00")]))
    call=TradeSignal("c","CALL","CONFIRMED","UD",100,_ts("2026-04-29T09:30:00"),101,102,99,100.2,_ts("2026-04-29T10:00:00"),100,98.5,"T",102,1,1,1,"be","")
    o=evaluate_signal_outcome(call,fut); assert o.outcome in {"TP1_FIRST","TP2_FIRST","TARGET_FIRST","AMBIGUOUS_SAME_CANDLE"}
    assert o.bars_to_outcome in {1,None}

    put=TradeSignal("p","PUT","CONFIRMED","UA",100,_ts("2026-04-29T09:30:00"),101,102,99,100.2,_ts("2026-04-29T10:00:00"),100,101.5,"T",98,1,1,1,"be","")
    op=evaluate_signal_outcome(put,fut); assert op.outcome in {"TP1_FIRST","TP2_FIRST","TARGET_FIRST","STOP_FIRST","AMBIGUOUS_SAME_CANDLE"}

    nohitf = pd.DataFrame({"Open":[100],"High":[100.4],"Low":[99.6],"Close":[100]}, index=pd.DatetimeIndex([_ts("2026-04-29T10:30:00")]))
    nh=evaluate_signal_outcome(call,nohitf); assert nh.outcome=="NO_HIT"

    pend=TradeSignal("x","CALL","PENDING_CONFIRMATION","UD",100,_ts("2026-04-29T09:30:00"),101,102,99,100.2,None,float('nan'),98.5,None,float('nan'),float('nan'),float('nan'),float('nan'),"be","")
    assert evaluate_signal_outcome(pend,fut).outcome=="PENDING"
    bad=TradeSignal("u","CALL","CONFIRMED","UD",100,_ts("2026-04-29T09:30:00"),101,102,99,100.2,None,float('nan'),98.5,None,float('nan'),float('nan'),float('nan'),float('nan'),"be","")
    assert evaluate_signal_outcome(bad,fut).outcome=="UNKNOWN"


def test_outcome_uses_tp1_and_tp2_milestones_before_full_target():
    fut = pd.DataFrame(
        {"Open":[100], "High":[101.2], "Low":[99.8], "Close":[101.0]},
        index=pd.DatetimeIndex([_ts("2026-04-29T10:30:00")]),
    )
    call=TradeSignal("tp1","CALL","CONFIRMED","UD",100,_ts("2026-04-29T09:30:00"),101,102,99,100.2,_ts("2026-04-29T10:00:00"),100,98.5,"T",102,1,2,2,"be","")
    tp1, tp2, full = signal_target_milestones(call)

    assert tp1 == 101
    assert tp2 == 101.5
    assert full == 102
    assert evaluate_signal_outcome(call, fut).outcome == "TP1_FIRST"


def test_latest_active_signal_skips_resolved_setups():
    candles = pd.DataFrame(
        {"Open":[100,100.5,102],"High":[101,103,103],"Low":[99.5,100,101],"Close":[100.2,102,102.5]},
        index=pd.DatetimeIndex([_ts("2026-04-29T09:30:00"), _ts("2026-04-29T10:30:00"), _ts("2026-04-29T11:30:00")]),
    )
    resolved=TradeSignal("resolved","CALL","CONFIRMED","UD",100,_ts("2026-04-29T09:30:00"),101,102,99,100.2,_ts("2026-04-29T10:30:00"),100.5,99.0,"T",101.0,1,1,1,"be","")
    active=TradeSignal("active","CALL","CONFIRMED","UD",100,_ts("2026-04-29T10:30:00"),101,102,99,100.2,_ts("2026-04-29T11:30:00"),102.0,99.0,"T",104.0,1,1,1,"be","")
    pending=TradeSignal("pending","PUT","PENDING_CONFIRMATION","UA",100,_ts("2026-04-29T11:30:00"),99,101,98,99.5,None,float("nan"),101.5,None,float("nan"),float("nan"),float("nan"),float("nan"),"be","")

    assert get_latest_active_signal([resolved], candles) is None
    assert get_latest_active_signal([resolved, active], candles).signal_id == "active"
    assert get_latest_active_signal([resolved, active, pending], candles).signal_id == "pending"
