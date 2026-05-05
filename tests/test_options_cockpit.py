from __future__ import annotations
from datetime import datetime
import pandas as pd
from app import (DynamicLine, SecondaryPivot, TradeSignal, build_options_cockpit_state, build_options_cockpit_state_with_fallback, get_central_tz, get_default_projection_time, option_mark_from_bid_ask_or_last, option_provider_label, option_quote_card_html, project_option_entry_to_target, quote_has_live_market_data, resolve_entry_target_lines, simulate_option_scenarios)


def _ts(s): return pd.Timestamp(datetime.fromisoformat(s), tz=get_central_tz())

class Strikes:
    def __init__(self,p,c,pu,e): self.underlying_price=p; self.call_strike=c; self.put_strike=pu; self.expiration_date=e


class FakeOptionProvider:
    provider_name = "TASTYTRADE_TEST"

    def get_selected_quotes(self, underlying_price, expiration_date, call_strike, put_strike):
        base = {
            "underlying": "SPY",
            "expiration": expiration_date,
            "bid": 1.0,
            "ask": 1.2,
            "mark": 1.1,
            "spread": 0.2,
            "gamma": 0.05,
            "theta": -0.1,
            "vega": 0.03,
            "iv": 0.25,
            "provider": self.provider_name,
            "timestamp": _ts("2026-04-29T10:00:00"),
            "warning": None,
        }
        return {
            "CALL": {"symbol": "SPY_CALL", "strike": call_strike, "option_type": "CALL", "delta": 0.5, **base},
            "PUT": {"symbol": "SPY_PUT", "strike": put_strike, "option_type": "PUT", "delta": -0.5, **base},
            "warning": None,
        }


def _lines():
    t=_ts("2026-04-29T09:00:00")
    return [DynamicLine("UA",714.2,t,0,"ascending","PUT_ZONE","PRIMARY_HIGH",True,""),DynamicLine("UD",711.8,t,0,"descending","CALL_ZONE","PRIMARY_HIGH",True,""),DynamicLine("LA",713.4,t,0,"ascending","PUT_ZONE","PRIMARY_LOW",True,""),DynamicLine("LD",711.2,t,0,"descending","CALL_ZONE","PRIMARY_LOW",True,"")]


def test_provider_and_quote_math():
    p=FakeOptionProvider(); q=p.get_selected_quotes(712.61, _ts("2026-04-29").date(), 717,708)
    state=build_options_cockpit_state(Strikes(712.61,717,708,_ts("2026-04-29").date()), provider=p)
    assert state.call_quote.provider=='TASTYTRADE_TEST' and state.put_quote.provider=='TASTYTRADE_TEST'
    assert state.call_quote.bid < state.call_quote.ask and state.call_quote.spread == round(state.call_quote.ask-state.call_quote.bid,2)
    assert q["CALL"]["mark"]>0 and q["PUT"]["mark"]>0


def test_mock_provider_quotes_are_rejected():
    class MockProvider(FakeOptionProvider):
        provider_name = "MOCK"

        def get_selected_quotes(self, underlying_price, expiration_date, call_strike, put_strike):
            q = super().get_selected_quotes(underlying_price, expiration_date, call_strike, put_strike)
            q["CALL"]["provider"] = "MOCK"
            q["PUT"]["provider"] = "MOCK"
            return q

    state=build_options_cockpit_state(Strikes(712.61,717,708,_ts("2026-04-29").date()), provider=MockProvider())
    assert state.call_quote is None and state.put_quote is None
    assert "Demo option quotes are disabled" in state.warning


def test_truncated_mock_provider_quotes_are_rejected():
    class MockProvider(FakeOptionProvider):
        provider_name = "MOC"

        def get_selected_quotes(self, underlying_price, expiration_date, call_strike, put_strike):
            q = super().get_selected_quotes(underlying_price, expiration_date, call_strike, put_strike)
            q["CALL"]["provider"] = "MOC"
            q["PUT"]["provider"] = "MOC"
            return q

    state=build_options_cockpit_state(Strikes(712.61,717,708,_ts("2026-04-29").date()), provider=MockProvider())
    assert state.call_quote is None and state.put_quote is None
    assert "Demo option quotes are disabled" in state.warning


def test_state_selection_signal_types_and_no_signal():
    st=Strikes(712.61,717,708,_ts("2026-04-29").date())
    s=build_options_cockpit_state(st)
    assert s.selected_trade_quote is None and 'Live quote feed unavailable' in s.warning
    cs=TradeSignal('1','CALL','CONFIRMED','UD',0,_ts("2026-04-29T10:00:00"),0,0,0,0,_ts("2026-04-29T11:00:00"),0,0,None,float('nan'),0,0,0,'','')
    ps=TradeSignal('2','PUT','CONFIRMED','UA',0,_ts("2026-04-29T10:00:00"),0,0,0,0,_ts("2026-04-29T11:00:00"),0,0,None,float('nan'),0,0,0,'','')
    assert build_options_cockpit_state(st, latest_signal=cs, provider=FakeOptionProvider()).selected_trade_quote.option_type=='CALL'
    assert build_options_cockpit_state(st, latest_signal=ps, provider=FakeOptionProvider()).selected_trade_quote.option_type=='PUT'


def test_live_provider_state_label_and_warning():
    class LiveProvider:
        provider_name = "TASTYTRADE"

        def get_selected_quotes(self, underlying_price, expiration_date, call_strike, put_strike):
            base = {
                "underlying": "SPY",
                "expiration": expiration_date,
                "bid": 1.0,
                "ask": 1.2,
                "mark": 1.1,
                "spread": 0.2,
                "gamma": 0.05,
                "theta": -0.1,
                "vega": 0.03,
                "iv": 0.25,
                "provider": "TASTYTRADE_LIVE",
                "timestamp": _ts("2026-04-29T10:00:00"),
                "warning": None,
            }
            return {
                "CALL": {"symbol": "SPY_CALL", "strike": call_strike, "option_type": "CALL", "delta": 0.5, **base},
                "PUT": {"symbol": "SPY_PUT", "strike": put_strike, "option_type": "PUT", "delta": -0.5, **base},
                "warning": None,
            }

    st=Strikes(712.61,717,708,_ts("2026-04-29").date())
    sig=TradeSignal('1','CALL','CONFIRMED','UD',0,_ts("2026-04-29T10:00:00"),0,0,0,0,_ts("2026-04-29T11:00:00"),0,0,None,float('nan'),0,0,0,'','')
    state=build_options_cockpit_state(st, latest_signal=sig, provider=LiveProvider())
    assert state.provider == "TASTYTRADE_LIVE"
    assert state.selected_trade_quote.provider == "TASTYTRADE_LIVE"
    assert state.warning is None


def test_chain_only_contracts_do_not_count_as_live_quotes():
    class ChainOnlyProvider:
        provider_name = "TASTYTRADE"

        def get_selected_quotes(self, underlying_price, expiration_date, call_strike, put_strike):
            base = {
                "underlying": "SPY",
                "expiration": expiration_date,
                "bid": float("nan"),
                "ask": float("nan"),
                "mark": float("nan"),
                "spread": float("nan"),
                "delta": float("nan"),
                "gamma": float("nan"),
                "theta": float("nan"),
                "vega": float("nan"),
                "iv": float("nan"),
                "provider": "TASTYTRADE_LIVE",
                "timestamp": _ts("2026-04-29T10:00:00"),
                "warning": None,
            }
            return {
                "CALL": {"symbol": "SPY_CALL", "strike": call_strike, "option_type": "CALL", **base},
                "PUT": {"symbol": "SPY_PUT", "strike": put_strike, "option_type": "PUT", **base},
                "warning": None,
            }

    state = build_options_cockpit_state(Strikes(712.61,717,708,_ts("2026-04-29").date()), provider=ChainOnlyProvider())
    assert state.call_quote is not None and quote_has_live_market_data(state.call_quote) is False
    assert state.scenarios == [] and state.entry_target_projection is None
    assert "Live quote streaming" in state.warning
    assert "Contract found" in option_quote_card_html(state.call_quote, 717, state.warning)


def test_yfinance_delayed_quotes_are_allowed_for_marks():
    class YFinanceProvider:
        provider_name = "YFINANCE_DELAYED"

        def get_selected_quotes(self, underlying_price, expiration_date, call_strike, put_strike):
            base = {
                "underlying": "SPY",
                "expiration": expiration_date,
                "bid": float("nan"),
                "ask": float("nan"),
                "mark": 1.35,
                "spread": float("nan"),
                "delta": float("nan"),
                "gamma": float("nan"),
                "theta": float("nan"),
                "vega": float("nan"),
                "iv": 0.22,
                "provider": "YFINANCE_DELAYED",
                "timestamp": _ts("2026-04-29T10:00:00"),
                "warning": "Delayed quote.",
            }
            return {
                "CALL": {"symbol": "SPY_CALL", "strike": call_strike, "option_type": "CALL", **base},
                "PUT": {"symbol": "SPY_PUT", "strike": put_strike, "option_type": "PUT", **base},
                "warning": "Using delayed option data.",
            }

    state = build_options_cockpit_state(Strikes(712.61,717,708,_ts("2026-04-29").date()), provider=YFinanceProvider())

    assert state.provider == "YFINANCE_DELAYED"
    assert state.call_quote.mark == 1.35
    assert state.scenarios == []
    assert option_provider_label(state) == "Delayed quotes"
    assert "Delayed price" in option_quote_card_html(state.call_quote, 717, state.warning)


def test_option_state_fallback_centralizes_yfinance_path(monkeypatch):
    class EmptyLiveProvider:
        provider_name = "TASTYTRADE"

        def get_selected_quotes(self, underlying_price, expiration_date, call_strike, put_strike):
            base = {
                "underlying": "SPY",
                "expiration": expiration_date,
                "bid": float("nan"),
                "ask": float("nan"),
                "mark": float("nan"),
                "spread": float("nan"),
                "delta": float("nan"),
                "gamma": float("nan"),
                "theta": float("nan"),
                "vega": float("nan"),
                "iv": float("nan"),
                "provider": "TASTYTRADE_CHAIN",
                "timestamp": _ts("2026-04-29T10:00:00"),
                "warning": None,
            }
            return {
                "CALL": {"symbol": "SPY_CALL", "strike": call_strike, "option_type": "CALL", **base},
                "PUT": {"symbol": "SPY_PUT", "strike": put_strike, "option_type": "PUT", **base},
                "warning": "No live stream data.",
            }

    def delayed_quotes(expiration_date, call_strike, put_strike):
        base = {
            "underlying": "SPY",
            "expiration": expiration_date,
            "bid": float("nan"),
            "ask": float("nan"),
            "mark": 0.75,
            "spread": float("nan"),
            "delta": float("nan"),
            "gamma": float("nan"),
            "theta": float("nan"),
            "vega": float("nan"),
            "iv": 0.21,
            "provider": "YFINANCE_DELAYED",
            "timestamp": _ts("2026-04-29T10:00:00"),
            "warning": "Delayed quote.",
        }
        return {
            "CALL": {"symbol": "SPY_CALL", "strike": call_strike, "option_type": "CALL", **base},
            "PUT": {"symbol": "SPY_PUT", "strike": put_strike, "option_type": "PUT", **base},
            "warning": "Using delayed option data.",
        }

    monkeypatch.setattr("app.fetch_yfinance_option_quotes", delayed_quotes)
    state = build_options_cockpit_state_with_fallback(
        Strikes(712.61,717,708,_ts("2026-04-29").date()),
        provider=EmptyLiveProvider(),
        current_dt=_ts("2026-04-29T10:00:00"),
    )

    assert state.provider == "YFINANCE_DELAYED"
    assert state.call_quote.mark == 0.75


def test_option_mark_prefers_midpoint_then_last_price():
    assert option_mark_from_bid_ask_or_last(1.0, 1.2, 0.9) == 1.1
    assert option_mark_from_bid_ask_or_last(float("nan"), float("nan"), 0.85) == 0.85


def test_scenarios_and_projection_behaviors():
    q=build_options_cockpit_state(Strikes(712.61,717,708,_ts("2026-04-29").date()), provider=FakeOptionProvider()).call_quote
    sc=simulate_option_scenarios(q)
    assert sc[0].estimated_pnl_per_contract == round((sc[0].estimated_mark-sc[0].current_mark)*100,2)

    lines=_lines(); entry=lines[1]; target=lines[0]
    proj=project_option_entry_to_target(q,712.61,entry,target,_ts("2026-04-29T09:00:00"),_ts("2026-04-29T09:00:00"))
    assert proj.estimated_entry_mark < q.mark
    assert proj.estimated_target_mark > proj.estimated_entry_mark


def test_put_entry_target_and_floor_and_missing_target_and_helpers():
    q=build_options_cockpit_state(Strikes(712.61,717,708,_ts("2026-04-29").date()), provider=FakeOptionProvider()).put_quote
    lines=_lines(); entry=lines[2]; target=lines[3]
    proj=project_option_entry_to_target(q,712.61,entry,target,_ts("2026-04-29T09:00:00"),_ts("2026-04-29T09:00:00"))
    assert proj.estimated_entry_mark < q.mark and proj.estimated_target_mark > proj.estimated_entry_mark
    tiny=project_option_entry_to_target(q,800,entry,target,_ts("2026-04-29T09:00:00"),_ts("2026-04-29T09:00:00"))
    assert tiny.estimated_entry_mark >= 0.01
    miss=project_option_entry_to_target(q,712.61,entry,None,_ts("2026-04-29T09:00:00"),None)
    assert miss.warning is not None
    assert get_default_projection_time(_ts("2026-04-29T12:00:00")).hour==9
    e,t=resolve_entry_target_lines(lines, latest_signal=TradeSignal('x','CALL','CONFIRMED','UD',0,_ts("2026-04-29T10:00:00"),0,0,0,0,_ts("2026-04-29T11:00:00"),0,0,'UA',0,0,0,0,'',''), option_type='CALL', current_dt=_ts("2026-04-29T12:00:00"))
    assert e.name=='UD' and t.name=='UA'
