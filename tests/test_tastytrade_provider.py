from __future__ import annotations

import math
from tastytrade_provider import TastytradeProvider, TastytradeProviderStatus


def test_status_defaults():
    st = TastytradeProviderStatus()
    assert st.connected is False and st.auth_ok is False


def test_auth_failure(monkeypatch):
    p = TastytradeProvider("id","secret","refresh")
    class R: status_code=401
    monkeypatch.setattr("requests.post", lambda *a, **k: R())
    assert p.authenticate() is False
    assert p.get_status()["auth_ok"] is False


def test_chain_selection_exact_and_nearest(monkeypatch):
    p = TastytradeProvider("id","secret","refresh")
    monkeypatch.setattr(p, "get_nested_option_chain", lambda *a, **k: {"data":{"items":[{"underlying-symbol":"SPY","expirations":[{"expiration-date":"2026-04-29","strikes":[{"strike-price":"709.0","call":"C709","put":"P709","call-streamer-symbol":"CS709","put-streamer-symbol":"PS709"},{"strike-price":"716.0","call":"C716","put":"P716","call-streamer-symbol":"CS716","put-streamer-symbol":"PS716"}]}]}]}})
    c,pu,w = p.select_contracts("SPY","2026-04-29",709,716)
    assert c["symbol"]=="C709" and pu["symbol"]=="P716" and w is None
    assert c["streamer-symbol"] == "CS709" and pu["streamer-symbol"] == "PS716"

    c2,pu2,w2 = p.select_contracts("SPY","2026-04-29",710,715)
    assert w2 is not None and c2 and pu2


def test_chain_selection_legacy_shape(monkeypatch):
    p = TastytradeProvider("id","secret","refresh")
    monkeypatch.setattr(p, "get_nested_option_chain", lambda *a, **k: {"data":{"items":[{"expiration-date":"2026-04-29","calls":[{"strike-price":709,"symbol":"C709"}],"puts":[{"strike-price":716,"symbol":"P716"}]}]}})
    c,pu,w = p.select_contracts("SPY","2026-04-29",709,716)
    assert c["symbol"]=="C709" and pu["symbol"]=="P716" and w is None


def test_chain_endpoint(monkeypatch):
    p = TastytradeProvider("id","secret","refresh")
    p.access_token = "token"
    class R:
        status_code = 200
        def json(self):
            return {"data":{"items":[]}}
    calls = []
    monkeypatch.setattr("requests.get", lambda url, **kwargs: calls.append((url, kwargs)) or R())
    assert p.get_nested_option_chain("SPY","2026-04-29") == {"data":{"items":[]}}
    assert calls[0][0].endswith("/option-chains/SPY/nested")
    assert calls[0][1]["headers"]["Authorization"] == "Bearer token"
    assert calls[0][1]["headers"]["Accept-Version"] == "20251101"


def test_expiration_unavailable(monkeypatch):
    p = TastytradeProvider("id","secret","refresh")
    monkeypatch.setattr(p, "get_nested_option_chain", lambda *a, **k: {"data":{"items":[]}})
    c,pu,w = p.select_contracts("SPY","2026-04-29",709,716)
    assert c is None and pu is None and "Expiration" in w


def test_quote_conversion_and_missing_greeks():
    p = TastytradeProvider("id","secret","refresh")
    q = p._to_quote_dict({"strike-price":709,"symbol":"C","bid":1.0,"ask":1.2}, "CALL", 712)
    assert q["mark"] == 1.1 and q["spread"] == 0.19999999999999996
    assert math.isnan(q["delta"]) and math.isnan(q["iv"])
    assert q["provider"] == "TASTYTRADE_CHAIN"


def test_selected_quotes_use_live_streamer_data(monkeypatch):
    p = TastytradeProvider("id","secret","refresh")
    monkeypatch.setattr(
        p,
        "select_contracts",
        lambda *a, **k: (
            {"strike-price":720,"symbol":"C720","streamer-symbol":".SPY260429C720","expiration-date":"2026-04-29"},
            {"strike-price":718,"symbol":"P718","streamer-symbol":".SPY260429P718","expiration-date":"2026-04-29"},
            None,
        ),
    )
    monkeypatch.setattr(
        p,
        "_fetch_live_market_data",
        lambda symbols: {
            ".SPY260429C720": {"bid":1.1,"ask":1.3,"mark":1.2,"spread":0.2,"delta":0.42,"gamma":0.1,"theta":-0.3,"vega":0.02,"iv":0.23},
            ".SPY260429P718": {"bid":1.0,"ask":1.2,"mark":1.1,"spread":0.2,"delta":-0.39,"gamma":0.1,"theta":-0.31,"vega":0.02,"iv":0.24},
        },
    )

    q = p.get_selected_quotes(719.0, "2026-04-29", 720, 718)

    assert q["CALL"]["provider"] == "TASTYTRADE_LIVE"
    assert q["CALL"]["bid"] == 1.1 and q["CALL"]["ask"] == 1.3
    assert q["CALL"]["delta"] == 0.42
    assert q["PUT"]["provider"] == "TASTYTRADE_LIVE"
    assert q["status"]["quotes_ok"] is True


def test_selected_quotes_do_not_claim_live_without_stream_data(monkeypatch):
    p = TastytradeProvider("id","secret","refresh")
    monkeypatch.setattr(
        p,
        "select_contracts",
        lambda *a, **k: (
            {"strike-price":720,"symbol":"C720","streamer-symbol":".SPY260429C720","expiration-date":"2026-04-29","bid":1.1,"ask":1.3},
            {"strike-price":718,"symbol":"P718","streamer-symbol":".SPY260429P718","expiration-date":"2026-04-29","bid":1.0,"ask":1.2},
            None,
        ),
    )
    monkeypatch.setattr(p, "_fetch_live_market_data", lambda symbols: {})

    q = p.get_selected_quotes(719.0, "2026-04-29", 720, 718)

    assert q["CALL"]["provider"] == "TASTYTRADE_CHAIN"
    assert q["PUT"]["provider"] == "TASTYTRADE_CHAIN"
    assert q["CALL"]["mark"] == 1.2000000000000002
    assert q["status"]["quotes_ok"] is False
    assert "live bid/ask/Greek streaming" in q["warning"]


def test_no_order_methods():
    banned=["submit_order","place_order","dry_run_order","cancel_order","replace_order"]
    for b in banned:
        assert not hasattr(TastytradeProvider, b)
