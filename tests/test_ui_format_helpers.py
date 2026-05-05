from app import DynamicLine, display_anchor_source, display_line_description, display_line_list, display_line_name, display_state_label, fmt_price, fmt_nan, safe_to_dict, render_badge, render_brand_logo, ui_icon
import pandas as pd


def test_fmt_helpers():
    assert fmt_price(1.234) == '1.23'
    assert fmt_nan(float('nan')) == '-'
    assert fmt_nan(pd.NA) == '-'


def test_safe_to_dict_redacts():
    d = safe_to_dict({'access_token':'abc','x':1})
    assert d['access_token'] == '[REDACTED]'


def test_render_badge_markup():
    m = render_badge('CALL','call')
    assert 'badge-call' in m and 'CALL' in m


def test_ui_icon_markup():
    m = ui_icon('target', 'amber', 'lg')
    assert "ui-icon amber lg" in m
    assert "<svg" in m and "aria-hidden='true'" in m


def test_brand_logo_is_animated_spy_mark():
    m = render_brand_logo()
    assert "brand-logo" in m
    assert "brand-path" in m and ">SPY<" in m


def test_display_anchor_source_uses_pivot_price():
    line = DynamicLine("UA", 719.78, None, 0.103, "ascending", "PUT_ZONE", "PRIMARY_HIGH", True, "")
    assert display_anchor_source(line) == "High pivot 719.78"


def test_display_line_names_are_product_facing():
    assert display_line_name("UA") == "Upper Ascending Trigger"
    assert display_line_name("UD") == "Upper Descending Trigger"
    assert display_line_name("S DESC 002") == "Upper Target"
    assert display_line_description("LA") == "Ascending structure from the low pivot"
    assert display_line_list(["UD", "UA"]) == "Upper Descending Trigger, Upper Ascending Trigger"


def test_display_state_labels_hide_internal_enums():
    assert display_state_label("REGULAR_SESSION") == "Session watch"
    assert display_state_label("WAIT_FOR_CONFIRMATION") == "Wait for confirmation"
    assert display_state_label("YFINANCE delayed") == "Delayed quotes"
    assert display_state_label("YFINANCE_FALLBACK") == "Delayed quotes"
    assert display_state_label("unavailable") == "Needs data"
    assert display_state_label("connected") == "Connected"
    assert display_state_label("not used") == "Not used"
