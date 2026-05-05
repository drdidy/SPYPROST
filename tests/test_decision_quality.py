from __future__ import annotations

from datetime import datetime
import pandas as pd

from app import DynamicLine, SelectedStrikes, TradeSignal, build_decision_state, build_wait_discipline_items, calculate_wick_rejection_metrics, evaluate_daily_risk, evaluate_structure_status, get_central_tz, score_signal_quality, evaluate_chase_status


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromisoformat(s), tz=get_central_tz())


def _sig(sig_type="CALL", status="CONFIRMED", rr=2.1, close=100.1, line=100.0, target=102.0, entry=100.5):
    return TradeSignal("id",sig_type,status,"UD" if sig_type=="CALL" else "UA",line,_ts("2026-04-28T10:00:00"),101,101.5,99.8,close,_ts("2026-04-28T11:00:00") if status=="CONFIRMED" else None,entry if status=="CONFIRMED" else float('nan'),99.3 if sig_type=="CALL" else 101.7,"T" if not pd.isna(target) else None,target,1,1,rr,"be","x")


def test_wick_metrics_call_put() -> None:
    c = _sig("CALL")
    m = calculate_wick_rejection_metrics(c)
    assert m["candle_range"] == 1.7 and m["wick_penetration"] == 0.2
    p = _sig("PUT", close=101.2, line=101.0, target=99.0, entry=100.5)
    mp = calculate_wick_rejection_metrics(p)
    assert mp["candle_range"] == 1.7 and mp["wick_penetration"] == 0.5


def test_quality_warnings_and_strengths() -> None:
    q1 = score_signal_quality(_sig(close=102.2))
    assert "CLOSE_TOO_FAR_FROM_LINE" in q1.warnings
    q2 = score_signal_quality(_sig(close=99.81, rr=0.8))
    assert "VERY_WEAK_REJECTION" in q2.warnings and "POOR_RISK_REWARD" in q2.warnings
    q3 = score_signal_quality(_sig(rr=2.2, close=100.9))
    assert "GOOD_RISK_REWARD" in q3.strengths
    q4 = score_signal_quality(_sig(target=float('nan')))
    assert "NO_STRUCTURAL_TARGET" in q4.warnings and q4.target_quality == "NO_TARGET"


def test_target_too_close_and_pending() -> None:
    q = score_signal_quality(_sig(target=100.8, entry=100.5))
    assert "TARGET_TOO_CLOSE" in q.warnings
    qp = score_signal_quality(_sig(status="PENDING_CONFIRMATION"))
    assert "WAIT_FOR_NEXT_CANDLE_OPEN" in qp.warnings and qp.action_label == "WAIT_FOR_CONFIRMATION"


def test_structure_call_put_and_chase() -> None:
    line = DynamicLine("UD",100,_ts("2026-04-28T08:00:00"),0,"descending","CALL_ZONE","PRIMARY_HIGH",True,"")
    intact = evaluate_structure_status(_sig("CALL"), pd.Series({"Close":100.0}), line, _ts("2026-04-28T12:00:00"))
    broken = evaluate_structure_status(_sig("CALL"), pd.Series({"Close":99.9}), line, _ts("2026-04-28T12:00:00"))
    assert intact["structure_status"]=="INTACT"
    assert broken["structure_status"]=="BROKEN" and "support" in broken["structure_warning"].lower()

    linep = DynamicLine("UA",101,_ts("2026-04-28T08:00:00"),0,"ascending","PUT_ZONE","PRIMARY_HIGH",True,"")
    brokenp = evaluate_structure_status(_sig("PUT", close=101.2, line=101.0, target=99.0), pd.Series({"Close":101.2}), linep, _ts("2026-04-28T12:00:00"))
    assert brokenp["structure_status"]=="BROKEN" and "resistance" in brokenp["structure_warning"].lower()

    assert evaluate_chase_status(_sig("CALL"), 101.0)["chase_status"] == "MISSED_ENTRY"
    assert evaluate_chase_status(_sig("PUT", entry=100.0), 99.6)["chase_status"] == "MISSED_ENTRY"


def test_decision_priority_and_daily_guardrail() -> None:
    d0 = build_decision_state(None, [], float("nan"), _ts("2026-04-28T12:00:00"), None, [])
    assert d0.final_decision == "WAIT"
    line = DynamicLine("UD",100,_ts("2026-04-28T08:00:00"),0,"descending","CALL_ZONE","PRIMARY_HIGH",True,"")
    dp = build_decision_state(_sig(status="PENDING_CONFIRMATION"), [line], 100.1, _ts("2026-04-28T12:00:00"), pd.Series({"Close":100.2}), [])
    assert dp.final_decision == "WAIT_FOR_CONFIRMATION"

    stop = evaluate_daily_risk([_sig(), _sig(), _sig()], [score_signal_quality(_sig())], max_signals_per_day=3)
    assert stop["daily_action"] == "STOP_TRADING"


def test_wait_discipline_items_fill_wait_card_with_strategy_gates() -> None:
    line = DynamicLine("UD",100,_ts("2026-04-28T08:00:00"),0,"descending","CALL_ZONE","PRIMARY_HIGH",True,"")
    pending = _sig(status="PENDING_CONFIRMATION")
    decision = build_decision_state(pending, [line], 100.1, _ts("2026-04-28T10:30:00"), pd.Series({"Close":100.2}), [])
    strikes = SelectedStrikes(100.0, 102, 98, _ts("2026-04-28").date(), "0DTE", None)

    items = build_wait_discipline_items(decision, pending, line, strikes, _ts("2026-04-28T10:30:00"))

    assert [item["label"] for item in items] == ["Candle Gate", "Chase Guard", "Contract Guard"]
    assert items[0]["value"] == "Open 11:00 AM CDT"
    assert items[1]["value"] == "No early entry"
    assert items[2]["value"] == "Call OTM"
