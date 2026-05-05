from __future__ import annotations

import json

import pandas as pd

from app import (
    EconomicEvent,
    GammaExposureInsight,
    MarketMove,
    MorningBriefingBundle,
    MorningBriefingResult,
    OptionsIntelligence,
    SentimentContext,
    SourceStatus,
    StructureLearningProfile,
    TechnicalContext,
    best_structure_scenario,
    build_daily_brief_explainer,
    build_openai_calendar_prompt,
    build_openai_request_payload,
    build_daily_brief_context,
    build_morning_briefing_prompt,
    build_app_decision_context,
    build_foresight_audit_record,
    build_foresight_desk_reviews,
    economic_event_from_ai_calendar_dict,
    darkpool_entry_read,
    darkpool_ranked_levels,
    external_context_verdicts,
    extract_json_payload_from_text,
    fallback_morning_decision,
    order_flow_board_cards,
    order_flow_plain_english,
    summarize_unusual_whales_flow_alerts,
    summarize_unusual_whales_gex,
    summarize_unusual_whales_greeks,
    summarize_unusual_whales_net_premium_ticks,
    summarize_unusual_whales_recent_flow,
    unusual_whales_card_data,
    merge_citations,
    morning_decision_from_result,
    normalize_morning_decision,
    calculate_max_pain,
    filter_near_spy_strikes,
    render_daily_brief_pdf_bytes,
    render_daily_brief_png_bytes,
    render_daily_brief_svg,
    rule_based_morning_briefing,
    save_foresight_decision_audit,
    structure_external_scenarios,
    support_refute_scorecard,
)


def test_darkpool_levels_rank_by_largest_premium() -> None:
    darkpool = {
        "key_levels": [
            {"price": 718.0, "premium": 4_000_000},
            {"price": 722.0, "premium": 12_000_000},
            {"price": 720.0, "premium": 7_000_000},
        ]
    }

    levels = darkpool_ranked_levels(darkpool, 3)

    assert [row["price"] for row in levels] == [722.0, 720.0, 718.0]


def test_darkpool_entry_read_supports_near_trigger() -> None:
    darkpool = {
        "key_levels": [
            {"price": 724.0, "premium": 11_000_000},
            {"price": 722.2, "premium": 9_000_000},
        ]
    }

    read = darkpool_entry_read(darkpool, entry_price=722.42, watch_side="PUT", entry_label="Upper Ascending Trigger")

    assert read["state"] == "aligned"
    assert "Upper Ascending Trigger" in read["copy"]


def test_external_context_verdicts_support_and_refute_entry() -> None:
    bundle = _bundle()
    options = OptionsIntelligence(
        bundle.options_intelligence.status,
        bundle.options_intelligence.put_call_open_interest_ratio,
        bundle.options_intelligence.put_call_volume_ratio,
        bundle.options_intelligence.max_pain,
        bundle.options_intelligence.call_wall,
        bundle.options_intelligence.put_wall,
        bundle.options_intelligence.selected_quotes,
        unusual_whales={
            "flow_alerts": {"flow_bias": "Bullish flow", "alert_count": 3, "net_premium_pressure": 500000},
            "market_tide": {"tone": "Risk-on options tide"},
            "net_premium_ticks": {"tone": "Call premium building"},
            "options_volume": {"put_call_volume_ratio": 0.7},
            "darkpool": {"key_levels": [{"price": 587.5, "premium": 10_000_000}]},
            "gex": {"net_gex": -1000000},
        },
    )
    bundle = MorningBriefingBundle(
        bundle.generated_at,
        bundle.lines,
        bundle.economic_events,
        bundle.global_context,
        [MarketMove("Dollar Index", "DX-Y.NYB", 100.0, 0.5, 0.3, bundle.generated_at, "Yahoo Finance")],
        bundle.sector_context,
        options,
        bundle.gamma_insight,
        SentimentContext(SourceStatus("Headline sentiment", "connected", "Headline score."), -2, "Bearish headlines", 1, 3),
        bundle.technical_context,
        bundle.news_items,
        bundle.learning_profile,
        bundle.source_statuses,
        bundle.session_date,
        bundle.latest_price,
    )

    verdicts = external_context_verdicts(bundle, "CALL", 587.42, "Upper Descending Trigger", 587.42)
    by_source = {row["source"]: row for row in verdicts}

    assert by_source["Option Flow"]["state"] == "aligned"
    assert by_source["Dark Pool"]["state"] == "aligned"
    assert by_source["Headlines"]["state"] == "opposes"
    assert by_source["Catalyst Clock"]["state"] == "risk"


def test_external_verdict_opposing_title_names_active_setup_side() -> None:
    verdicts = external_context_verdicts(_bundle(), "PUT", 587.42, "Upper Ascending Trigger", 587.42)
    by_source = {row["source"]: row for row in verdicts}

    assert by_source["Technicals"]["state"] == "opposes"
    assert by_source["Technicals"]["title"] == "Caution for put setup"
    assert "SPY above" in by_source["Technicals"]["copy"]


def test_support_refute_scorecard_weights_actionable_context() -> None:
    scorecard = support_refute_scorecard([
        {"source": "Option Flow", "state": "aligned", "title": "Calls supported"},
        {"source": "Catalyst Clock", "state": "risk", "title": "CPI timing"},
        {"source": "Headlines", "state": "opposes", "title": "Risk-off headlines"},
        {"source": "Global Tape", "state": "neutral", "title": "Mixed"},
    ])

    assert scorecard["support"] == 1
    assert scorecard["risk"] == 1
    assert scorecard["caution"] == 1
    assert scorecard["neutral"] == 1
    assert scorecard["net_score"] < 0


def test_foresight_desk_reviews_cover_execution_stack() -> None:
    decision = fallback_morning_decision(_bundle())
    reviews = build_foresight_desk_reviews(_bundle(), decision)

    assert [row["desk"] for row in reviews] == ["Structure", "Order Flow", "Catalyst", "Risk", "Execution"]
    assert any(row["state"] == "risk" for row in reviews)


def test_calculate_max_pain_uses_open_interest() -> None:
    calls = pd.DataFrame({"strike": [100, 101, 102], "openInterest": [10, 100, 10]})
    puts = pd.DataFrame({"strike": [100, 101, 102], "openInterest": [10, 100, 10]})

    assert calculate_max_pain(calls, puts) == 101.0


def test_filter_near_spy_strikes_removes_far_open_interest() -> None:
    calls = pd.DataFrame({"strike": [500, 720, 725], "openInterest": [99999, 100, 200]})
    puts = pd.DataFrame({"strike": [510, 718, 716], "openInterest": [99999, 300, 200]})

    near_calls, near_puts = filter_near_spy_strikes(calls, puts, 721.0, width=10.0)

    assert 500 not in set(near_calls["strike"])
    assert 510 not in set(near_puts["strike"])


def _bundle() -> MorningBriefingBundle:
    now = pd.Timestamp("2026-05-01 06:30", tz="America/Chicago")
    options = OptionsIntelligence(
        SourceStatus("Options intelligence", "connected", "Delayed option-chain OI/volume proxy."),
        1.2,
        0.9,
        588.0,
        590.0,
        585.0,
        [{"type": "CALL", "strike": 590.0, "open_interest": 1000}],
    )
    gamma = GammaExposureInsight(
        SourceStatus("Gamma exposure", "unavailable", "GEX_API_URL is not configured; using OI magnet proxy."),
        None,
        "Proxy only",
        [585.0, 588.0, 590.0],
        "True dealer GEX is not available without GEX_API_URL.",
    )
    technical = TechnicalContext(
        SourceStatus("Yahoo Finance daily SPY", "connected", "Daily SPY history available."),
        589.0,
        584.0,
        587.0,
        580.0,
        550.0,
        590.0,
        582.0,
        595.0,
        570.0,
        1.1,
    )
    learning = StructureLearningProfile(20, 12, "Call-side watch", "Developing sample", 0.58, 0.08, 0.25, 1.2, 2.3, -0.8, None, "Historical tendency only.")
    return MorningBriefingBundle(
        now,
        [{"code": "UD", "name": "Upper Descending Trigger", "role": "Descending Trigger", "value": 587.42, "anchor_price": 589.0, "anchor_time": "2026-04-30 15:00 CDT"}],
        [EconomicEvent(now.date(), "8:30 AM ET / 7:30 AM CT", "CPI", "High", "Local calendar", "Inflation release")],
        [MarketMove("ES futures", "ES=F", 5900.0, 20.0, 0.34, now, "Yahoo Finance")],
        [],
        [MarketMove("Technology", "XLK", 240.0, 1.5, 0.63, now, "Yahoo Finance")],
        options,
        gamma,
        SentimentContext(SourceStatus("Headline sentiment", "connected", "Headline score."), 2, "Bullish headlines", 3, 1),
        technical,
        [],
        learning,
        [options.status, gamma.status, technical.status],
        now.date(),
        590.0,
    )


def test_prompt_carries_truthful_source_statuses() -> None:
    prompt = build_morning_briefing_prompt(_bundle())

    assert "Do not invent unavailable options flow" in prompt
    assert "Return ONLY valid JSON" in prompt
    assert '"primary_trade"' in prompt
    assert "APP_DECISION_CONTEXT_JSON" in prompt
    assert '"support_refute"' in prompt
    assert '"desk_reviews"' in prompt
    assert "SCOUT_LIST_JSON" in prompt
    assert "Tradytics" in prompt
    assert "GEX_API_URL is not configured" in prompt
    assert "CPI" in prompt
    assert "Upper Descending Trigger" in prompt


def test_rule_based_briefing_marks_unavailable_premium_feeds() -> None:
    result = rule_based_morning_briefing(_bundle(), ai_warning="Live synthesis connection is not configured.")

    assert result.provider == "Verified internal assessment"
    assert any("Live synthesis connection is not configured" in warning for warning in result.warnings)
    assert "True dealer GEX is not available" not in result.text
    assert any("GEX_API_URL is not configured" in warning for warning in result.warnings)
    assert "CPI at 8:30 AM ET / 7:30 AM CT" in result.text


def test_daily_brief_renderer_exports_visual_files() -> None:
    bundle = _bundle()
    result = rule_based_morning_briefing(bundle)

    context = build_daily_brief_context(bundle, result)
    guide = build_daily_brief_explainer(bundle, result)
    svg = render_daily_brief_svg(bundle, result)
    png = render_daily_brief_png_bytes(bundle, result)
    pdf = render_daily_brief_pdf_bytes(png)

    assert context["primary_value"] == "587.42"
    assert "SPY Foresight" in guide["plain_read"]
    assert "touch it from above" in guide["confirmation"]
    assert any(item["label"] == "Confirm" for item in guide["checklist"])
    assert "SPY PROPHET" in svg
    assert "DAILY ONE-PAGE TRADING BRIEF" in svg
    assert "fake" not in svg.lower()
    assert png and png.startswith(b"\x89PNG")
    assert pdf and pdf.startswith(b"%PDF")


def test_daily_brief_put_plan_separates_opening_pivot_from_put_entry() -> None:
    bundle = _bundle()
    lines = [
        {"code": "UA", "name": "Upper Ascending Trigger", "role": "Ascending Trigger", "value": 727.87, "anchor_price": 724.87, "anchor_time": "2026-05-01 09:00 CDT"},
        {"code": "LA", "name": "Lower Ascending Trigger", "role": "Ascending Trigger", "value": 722.47, "anchor_price": 720.47, "anchor_time": "2026-05-01 14:00 CDT"},
        {"code": "UD", "name": "Upper Descending Trigger", "role": "Descending Trigger", "value": 721.87, "anchor_price": 720.47, "anchor_time": "2026-05-01 14:00 CDT"},
        {"code": "LD", "name": "Lower Descending Trigger", "role": "Descending Trigger", "value": 720.60, "anchor_price": 720.47, "anchor_time": "2026-05-01 14:00 CDT"},
    ]
    bundle = MorningBriefingBundle(
        bundle.generated_at,
        lines,
        bundle.economic_events,
        bundle.global_context,
        bundle.macro_context,
        bundle.sector_context,
        bundle.options_intelligence,
        bundle.gamma_insight,
        bundle.sentiment,
        bundle.technical_context,
        bundle.news_items,
        bundle.learning_profile,
        bundle.source_statuses,
        bundle.session_date,
        722.0,
    )
    result = MorningBriefingResult(
        bundle.generated_at,
        "OpenAI",
        "gpt-5.2",
        json.dumps({
            "stance": "WATCH_PUT",
            "headline": "Wait for a put rejection.",
            "primary_trade": {
                "trigger_line": "Upper Ascending Trigger",
                "trigger_price": "727.87",
                "contract": "PUT 725",
                "confidence": 66,
            },
            "why": ["Opening pivot decides whether price can reach the upper ascending trigger."],
            "risk_flags": [],
        }),
        66,
        [],
        [],
        [],
    )

    context = build_daily_brief_context(bundle, result)
    svg = render_daily_brief_svg(bundle, result)

    assert context["primary_value"] == "722.47"
    assert context["entry_value"] == "727.87"
    assert context["upper_value"] == "727.87"
    assert context["contract_value"] == "PUT 725 / PUT 720"
    assert context["entry_a_contract"] == "PUT 725"
    assert context["entry_b_contract"] == "PUT 720"
    assert "722.47 -&gt; 727.87" in svg
    key_values = [value for _, value in context["key_levels"]]
    assert key_values == list(dict.fromkeys(key_values))
    assert ("Opening Pivot / Lower Ascending Trigger", "722.47") in context["key_levels"]


def test_external_scenarios_rank_gex_confluence_at_lower_put() -> None:
    bundle = _bundle()
    lines = [
        {"code": "UA", "name": "Upper Ascending Trigger", "role": "Ascending Trigger", "value": 727.87, "anchor_price": 724.87, "anchor_time": "2026-05-01 09:00 CDT"},
        {"code": "LA", "name": "Lower Ascending Trigger", "role": "Ascending Trigger", "value": 722.47, "anchor_price": 720.47, "anchor_time": "2026-05-01 14:00 CDT"},
        {"code": "UD", "name": "Upper Descending Trigger", "role": "Descending Trigger", "value": 721.87, "anchor_price": 720.47, "anchor_time": "2026-05-01 14:00 CDT"},
        {"code": "LD", "name": "Lower Descending Trigger", "role": "Descending Trigger", "value": 720.60, "anchor_price": 720.47, "anchor_time": "2026-05-01 14:00 CDT"},
    ]
    options = OptionsIntelligence(
        SourceStatus("Options intelligence", "connected", "Premium context active."),
        1.1,
        1.2,
        722.50,
        728.0,
        722.0,
        [],
        [],
        {
            "flow_alerts": {"flow_bias": "Bearish flow", "alert_count": 4, "net_premium_pressure": -250000},
            "gex": {"levels": [{"strike": 722.47, "total_gex": -4_200_000}, {"strike": 727.87, "total_gex": -250_000}], "net_gex": -4_450_000},
            "darkpool": {"key_levels": [{"price": 722.40, "premium": 8_000_000}]},
        },
    )
    bundle = MorningBriefingBundle(
        bundle.generated_at,
        lines,
        bundle.economic_events,
        bundle.global_context,
        bundle.macro_context,
        bundle.sector_context,
        options,
        bundle.gamma_insight,
        bundle.sentiment,
        bundle.technical_context,
        bundle.news_items,
        bundle.learning_profile,
        bundle.source_statuses,
        bundle.session_date,
        722.0,
    )

    scenarios = structure_external_scenarios(bundle)
    best = best_structure_scenario(bundle, "PUT")

    assert best["name"] == "Lower Ascending Trigger"
    assert best["state"] == "aligned"
    assert any("Dealer GEX cluster" in item for item in best["support"])
    assert scenarios[0]["name"] == "Lower Ascending Trigger"


def test_daily_brief_uses_verified_line_value_instead_of_ai_price() -> None:
    bundle = _bundle()
    lines = [
        {"code": "LA", "name": "Lower Ascending Trigger", "role": "Ascending Trigger", "value": 722.47, "anchor_price": 720.47, "anchor_time": "2026-05-01 14:00 CDT"},
        {"code": "UD", "name": "Upper Descending Trigger", "role": "Descending Trigger", "value": 721.87, "anchor_price": 720.47, "anchor_time": "2026-05-01 14:00 CDT"},
    ]
    bundle = MorningBriefingBundle(
        bundle.generated_at,
        lines,
        bundle.economic_events,
        bundle.global_context,
        bundle.macro_context,
        bundle.sector_context,
        bundle.options_intelligence,
        bundle.gamma_insight,
        bundle.sentiment,
        bundle.technical_context,
        bundle.news_items,
        bundle.learning_profile,
        bundle.source_statuses,
        bundle.session_date,
        722.0,
    )
    result = MorningBriefingResult(
        bundle.generated_at,
        "OpenAI",
        "gpt-5.2",
        json.dumps({
            "stance": "WATCH_PUT",
            "headline": "Watch the lower ascending trigger.",
            "primary_trade": {
                "trigger_line": "Lower Ascending Trigger",
                "trigger_price": "999.99",
                "contract": "PUT 998",
                "confidence": 60,
            },
        }),
        60,
        [],
        [],
        [],
    )

    context = build_daily_brief_context(bundle, result)

    assert context["entry_value"] == "722.47"
    assert context["entry_label"] == "Lower Ascending Trigger"
    assert context["contract_value"] != "PUT 998"
    assert "999.99" not in render_daily_brief_svg(bundle, result)


def test_openai_request_payload_enables_web_search() -> None:
    payload = build_openai_request_payload("brief me", "gpt-4.1-mini", enable_web_search=True)

    assert payload["tools"][0]["type"] == "web_search"
    assert payload["tools"][0]["user_location"]["timezone"] == "America/Chicago"
    assert payload["include"] == ["web_search_call.action.sources"]


def test_calendar_prompt_requires_verified_json_rows() -> None:
    prompt = build_openai_calendar_prompt(pd.Timestamp("2026-05-01 06:30", tz="America/Chicago"))

    assert "Return ONLY a JSON object" in prompt
    assert "Investing.com Economic Calendar" in prompt
    assert "ForexFactory Calendar" in prompt
    assert "2026-05-01" in prompt
    assert '"events"' in prompt


def test_extract_json_payload_from_text_handles_markdown_fences() -> None:
    text = '```json\n{"events":[{"event":"NFP","impact":"High"}]}\n```'

    assert extract_json_payload_from_text(text) == {"events": [{"event": "NFP", "impact": "High"}]}


def test_morning_decision_parser_reads_structured_ai_output() -> None:
    result = MorningBriefingResult(
        pd.Timestamp("2026-05-01 06:45", tz="America/Chicago"),
        "OpenAI",
        "gpt-5.2",
        '{"stance":"WATCH_PUT","headline":"Wait for Upper Ascending Trigger rejection.","primary_trade":{"trigger_line":"Upper Ascending Trigger","trigger_price":"722.42","contract":"PUT 718","confidence":61},"why":["ISM at 9:00 AM CT"],"avoid":[{"label":"Chase","reason":"Between lines"}],"risk_flags":["same-day spread risk"],"source_notes":[],"novice_summary":"Wait for confirmation."}',
        61,
        [],
        [],
        [],
    )

    decision = morning_decision_from_result(result)

    assert decision is not None
    assert decision["stance"] == "WATCH_PUT"
    assert decision["schema_version"] == "spy_foresight_v2"
    assert decision["primary_trade"]["contract"] == "PUT 718"
    assert decision["primary_trade"]["entry_rule"]
    assert decision["avoid"][0]["label"] == "Chase"


def test_normalize_morning_decision_enforces_required_schema() -> None:
    decision = normalize_morning_decision({"stance": "BUY_NOW", "primary_trade": {"confidence": 101}}, 44)

    assert decision["stance"] == "WAIT"
    assert decision["schema_version"] == "spy_foresight_v2"
    assert decision["primary_trade"]["confidence"] == 100
    assert decision["primary_trade"]["contract"] == "No contract until confirmation"


def test_app_decision_context_includes_weighted_verdicts() -> None:
    context = build_app_decision_context(_bundle())

    assert context["support_refute"]["risk"] >= 1
    assert context["desk_reviews"]
    assert all("weight" in row for row in context["external_verdicts"])


def test_merge_citations_dedupes_normalized_urls() -> None:
    citations = merge_citations(
        [{"url": "https://example.com/a/", "title": "A"}, {"url": "https://example.com/a#section", "title": "A again"}],
        [{"url": "https://example.com/b", "title": "B"}],
    )

    assert [row["url"] for row in citations] == ["https://example.com/a", "https://example.com/b"]


def test_fallback_decision_hides_contract_until_confirmation() -> None:
    bundle = _bundle()
    bundle = MorningBriefingBundle(
        bundle.generated_at,
        [{"code": "UA", "name": "Upper Ascending Trigger", "role": "Ascending Trigger", "value": 722.42, "anchor_price": 719.79, "anchor_time": "2026-04-30 15:00 CDT"}],
        bundle.economic_events,
        bundle.global_context,
        bundle.macro_context,
        bundle.sector_context,
        OptionsIntelligence(
            bundle.options_intelligence.status,
            bundle.options_intelligence.put_call_open_interest_ratio,
            bundle.options_intelligence.put_call_volume_ratio,
            bundle.options_intelligence.max_pain,
            bundle.options_intelligence.call_wall,
            bundle.options_intelligence.put_wall,
            [],
            [{"type": "CALL", "strike": 724.0}, {"type": "PUT", "strike": 720.0}],
        ),
        bundle.gamma_insight,
        bundle.sentiment,
        bundle.technical_context,
        bundle.news_items,
        bundle.learning_profile,
        bundle.source_statuses,
        bundle.session_date,
        721.0,
    )

    decision = fallback_morning_decision(bundle)

    assert decision["primary_trade"]["contract"] == "No contract until confirmation"
    assert decision["support_refute"]["risk"] >= 1
    assert decision["desk_reviews"]


def test_foresight_audit_file_records_daily_decision(tmp_path) -> None:
    bundle = _bundle()
    result = rule_based_morning_briefing(bundle)

    path = save_foresight_decision_audit(bundle, result, directory=str(tmp_path))
    rows = json.loads(path.read_text())

    assert path.name == "2026-05-01.json"
    assert rows[0]["schema_version"] == "spy_foresight_v2"
    assert rows[0]["decision"]["support_refute"]
    assert rows[0]["source_statuses"]


def test_foresight_audit_record_contains_provider_matrix() -> None:
    bundle = _bundle()
    result = rule_based_morning_briefing(bundle)
    record = build_foresight_audit_record(bundle, result)

    assert record["app_decision_context"]["external_verdicts"]
    assert any(row["name"] == "SPY Foresight Synthesis" for row in record["source_statuses"])


def test_ai_calendar_event_requires_exact_date_time_and_source() -> None:
    event = economic_event_from_ai_calendar_dict(
        {
            "event_date": "2026-05-01",
            "time_label": "8:30 AM ET / 7:30 AM CT",
            "event": "ISM Manufacturing PMI",
            "impact": "High",
            "source": "Investing.com",
            "notes": "Forecast 49.0; previous 49.0",
        }
    )

    assert event is not None
    assert event.event_date == pd.Timestamp("2026-05-01").date()
    assert event.time_label == "8:30 AM ET / 7:30 AM CT"
    assert event.source == "Investing.com"
    assert event.impact == "High"

    assert economic_event_from_ai_calendar_dict({"event": "CPI", "event_date": "2026-05-01"}) is None


def test_unusual_whales_flow_alerts_reduce_to_actionable_pressure() -> None:
    now = pd.Timestamp("2026-05-01 09:45", tz="America/Chicago")
    summary = summarize_unusual_whales_flow_alerts(
        {
            "data": [
                {
                    "ticker": "SPY",
                    "type": "call",
                    "strike": "720",
                    "created_at": "2026-05-01T14:40:00Z",
                    "total_premium": "250000",
                    "total_ask_side_prem": "240000",
                    "total_bid_side_prem": "10000",
                    "has_sweep": True,
                    "alert_rule": "RepeatedHits",
                },
                {
                    "ticker": "SPY",
                    "type": "put",
                    "strike": "715",
                    "created_at": "2026-04-20T14:40:00Z",
                    "total_premium": "999999",
                    "total_ask_side_prem": "999999",
                    "total_bid_side_prem": "0",
                },
            ]
        },
        now,
    )

    assert summary["alert_count"] == 1
    assert summary["flow_bias"] == "Bullish flow"
    assert summary["key_strikes"][0]["strike"] == 720.0
    assert summary["largest_alerts"][0]["sweep"] is True


def test_unusual_whales_gex_finds_flip_and_levels() -> None:
    summary = summarize_unusual_whales_gex(
        {
            "data": [
                {"strike": "718", "call_gamma_oi": "100", "put_gamma_oi": "-300"},
                {"strike": "720", "call_gamma_oi": "400", "put_gamma_oi": "-100"},
            ]
        },
        latest_price=719,
    )

    assert summary["gamma_flip"] == 719.0
    assert summary["levels"][0]["strike"] in {718.0, 720.0}


def test_unusual_whales_recent_flow_summarizes_current_tape() -> None:
    now = pd.Timestamp("2026-05-01 09:45", tz="America/Chicago")
    summary = summarize_unusual_whales_recent_flow(
        {
            "data": [
                {
                    "ticker": "SPY",
                    "option_type": "call",
                    "strike": "720",
                    "executed_at": "2026-05-01T14:44:00Z",
                    "premium": "180000",
                    "side": "ask",
                },
                {
                    "ticker": "SPY",
                    "option_type": "put",
                    "strike": "715",
                    "executed_at": "2026-04-20T14:44:00Z",
                    "premium": "999999",
                    "side": "ask",
                },
            ]
        },
        now,
        latest_price=718.0,
    )

    assert summary is not None
    assert summary["trade_count"] == 1
    assert summary["tone"] == "Recent call buying"
    assert summary["top_strikes"][0]["strike"] == 720.0


def test_unusual_whales_net_premium_ticks_reads_direction() -> None:
    now = pd.Timestamp("2026-05-01 09:45", tz="America/Chicago")
    summary = summarize_unusual_whales_net_premium_ticks(
        {
            "data": [
                {"timestamp": "2026-05-01T14:30:00Z", "net_call_premium": "200000", "net_put_premium": "-100000"},
                {"timestamp": "2026-05-01T14:45:00Z", "net_call_premium": "1200000", "net_put_premium": "-100000"},
            ]
        },
        now,
    )

    assert summary is not None
    assert summary["tone"] == "Call premium building"
    assert summary["net_premium"] == 1100000.0


def test_unusual_whales_greeks_keeps_nearby_strikes() -> None:
    summary = summarize_unusual_whales_greeks(
        {
            "data": [
                {"strike": "720", "call_delta": "0.38", "put_delta": "-0.42", "call_gamma": "0.05"},
                {"strike": "740", "call_delta": "0.05", "put_delta": "-0.95"},
            ]
        },
        latest_price=719.0,
    )

    assert summary is not None
    assert summary["nearest"]["strike"] == 720.0
    assert len(summary["levels"]) == 1


def test_unusual_whales_card_only_appears_when_paid_data_loaded() -> None:
    empty = OptionsIntelligence(SourceStatus("Options intelligence", "connected", ""), 1, 1, 710, 712, 708, [])
    assert unusual_whales_card_data(empty)[0] == ""

    loaded = OptionsIntelligence(
        SourceStatus("Options intelligence", "connected", ""),
        1,
        1,
        710,
        712,
        708,
        [],
        unusual_whales={
            "flow_alerts": {
                "flow_bias": "Bearish flow",
                "alert_count": 3,
                "net_premium_pressure": -350000,
                "key_strikes": [{"strike": 718, "call_premium": 0, "put_premium": 400000}],
            },
            "market_tide": {"tone": "Risk-off options tide"},
        },
    )

    value, copy, chips, tone = unusual_whales_card_data(loaded)
    assert "Bearish flow" in value
    assert "718" in copy
    assert "3 flow alerts" in chips
    assert tone == "red"


def test_order_flow_board_exposes_flow_and_darkpool_levels() -> None:
    options = OptionsIntelligence(
        SourceStatus("Options intelligence", "connected", ""),
        1,
        1,
        710,
        712,
        708,
        [],
        unusual_whales={
            "flow_alerts": {
                "flow_bias": "Bullish flow",
                "alert_count": 4,
                "net_premium_pressure": 450000,
                "key_strikes": [{"strike": 720, "net_pressure": 300000}],
            },
            "recent_flow": {
                "tone": "Recent call buying",
                "trade_count": 8,
                "net_pressure": 600000,
                "top_strikes": [{"strike": 721, "net_pressure": 250000}],
            },
            "market_tide": {"tone": "Risk-on options tide", "net_call_premium": 1200000, "net_put_premium": -300000},
            "net_premium_ticks": {"tone": "Call premium building", "net_premium": 900000, "net_call_premium": 1200000, "net_put_premium": -300000},
            "options_volume": {"put_call_volume_ratio": 0.8},
            "darkpool": {
                "print_count": 12,
                "total_premium": 88_000_000,
                "key_levels": [{"price": 719.5, "premium": 32_000_000}],
            },
        },
    )

    cards = order_flow_board_cards(options)
    read = order_flow_plain_english(options)

    assert {card["title"] for card in cards} >= {"Same-Day Flow Alerts", "Recent Tape", "Market Tide", "Dark Pool Levels"}
    flow_card = next(card for card in cards if card["title"] == "Same-Day Flow Alerts")
    assert "supports call setups" in flow_card["means"]
    darkpool = next(card for card in cards if card["title"] == "Dark Pool Levels")
    assert darkpool["levels"][0]["label"] == "719.50"
    assert "support or caution against an entry" in darkpool["means"]
    assert read["label"] == "Flow supports calls"
    assert read["tone"] == "call"
