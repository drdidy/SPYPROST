from __future__ import annotations
from datetime import date, datetime
import json
import pathlib
import pandas as pd
from app import JournalEntry, ensure_data_dir, load_signal_journal, save_signal_journal, make_journal_id, get_journal_signal_key, upsert_journal_entry, entry_is_more_complete, compute_journal_analytics, auto_journal_live_signals, build_journal_entry_from_live_state, save_replay_signals_to_journal, signal_journal_to_json, TradeSignal, ReplayState, SignalQuality, get_central_tz


def _ts(s): return pd.Timestamp(datetime.fromisoformat(s), tz=get_central_tz())

def _entry(signal_id='s1', outcome=None, source='LIVE_MANUAL'):
    e=JournalEntry('',_ts('2026-04-29T10:00:00'),None,_ts('2026-04-29T10:00:00').date(),source,signal_id,'CALL','CONFIRMED','UD','CALL_ZONE','BULLISH','A',90,'TRADE_ALLOWED','TRADE_ALLOWED',_ts('2026-04-29T09:00:00'),_ts('2026-04-29T10:00:00'),100,99,'UA',101,1.5,outcome,None,1.2,-0.4,2,'CALL',709,2.1,3.1,100,'TASTYTRADE_TEST',None,[])
    return e.__class__(make_journal_id(e), *list(e.__dict__.values())[1:])


def test_dir_load_save_and_atomic(tmp_path):
    p=tmp_path/'data'; ensure_data_dir(p); assert p.exists()
    f=p/'signal_journal.json'; assert load_signal_journal(str(f))==[]
    e=_entry(); save_signal_journal([e], str(f)); back=load_signal_journal(str(f)); assert len(back)==1 and back[0].signal_id=='s1'


def test_journal_export_serializes_dates_and_nan_values():
    e = _entry()
    e = e.__class__(**{**e.__dict__, "trade_date": date(2026, 4, 29), "estimated_entry_mark": float("nan")})
    payload = signal_journal_to_json([e])
    parsed = json.loads(payload)
    assert parsed[0]["trade_date"] == "2026-04-29"
    assert parsed[0]["estimated_entry_mark"] is None
    json.dumps(parsed, allow_nan=False)


def test_malformed_backup(tmp_path):
    f=tmp_path/'signal_journal.json'; f.write_text('{bad')
    out=load_signal_journal(str(f)); assert out==[]
    assert any(x.name.startswith('signal_journal.corrupt') for x in tmp_path.iterdir())


def test_ids_keys_upsert_completeness():
    e1=_entry('a'); e2=_entry('a'); e3=_entry('b')
    assert make_journal_id(e1)==make_journal_id(e2) and make_journal_id(e1)!=make_journal_id(e3)
    assert get_journal_signal_key(e1).startswith('sig:')
    arr,act=upsert_journal_entry([],e1); assert act=='inserted'
    arr,act=upsert_journal_entry(arr,e2); assert act=='skipped'
    e2b=_entry('a', outcome='TARGET_FIRST'); assert entry_is_more_complete(e2b,e1)
    arr,act=upsert_journal_entry(arr,e2b); assert act=='updated'


def test_build_live_and_auto_journal_and_analytics(tmp_path):
    sig=TradeSignal('sig','CALL','CONFIRMED','UD',100,_ts('2026-04-29T09:00:00'),101,102,99,100.2,_ts('2026-04-29T10:00:00'),100,99,'UA',101,1,1,1,'be','x')
    ent=build_journal_entry_from_live_state(sig,None,None,None,source='LIVE_MANUAL')
    assert ent is not None
    path=str(tmp_path/'journal.json')
    entries,status=auto_journal_live_signals([sig],None,None,None,[],path,enabled=False)
    assert status.enabled is False
    entries,status=auto_journal_live_signals([sig],None,None,None,[],path,enabled=True)
    assert status.saved_count==1
    entries2,status2=auto_journal_live_signals([sig],None,None,None,entries,path,enabled=True)
    assert status2.skipped_duplicate_count>=1
    entries[0]=entries[0].__class__(**{**entries[0].__dict__,'outcome':'TARGET_FIRST'})
    a=compute_journal_analytics(entries)
    assert a.total_entries==1 and a.target_first_count==1
    entries[0]=entries[0].__class__(**{**entries[0].__dict__,'outcome':'TP1_FIRST'})
    a=compute_journal_analytics(entries)
    assert a.target_first_count==1


def test_save_replay_signals_to_journal_handles_export_safe_payload(tmp_path):
    sig=TradeSignal('replay-sig','PUT','CONFIRMED','LA',100,_ts('2026-04-29T09:00:00'),101,102,99,100.2,_ts('2026-04-29T10:00:00'),100,101,'LD',98,1,2,2,'be','x')
    quality=SignalQuality('replay-sig','A',90,0.1,0.2,0.3,0.4,20,20,'CLEAN',[],[],'TRADE_ALLOWED','Good setup.')
    rs=ReplayState(date(2026,4,29),None,date(2026,4,28),None,None,[],[],None,[sig],{'replay-sig':quality},{},None,'Replay built.',[])
    path=str(tmp_path/'signal_journal.json')

    entries,status=save_replay_signals_to_journal(rs,[],path)

    assert status == {'total': 1, 'inserted': 1, 'updated': 0, 'skipped': 0}
    assert len(entries) == 1
    saved = json.loads(pathlib.Path(path).read_text())
    assert saved[0]['trade_date'] == '2026-04-29'
    assert saved[0]['estimated_entry_mark'] is None
