import importlib


def test_app_import_safe():
    m = importlib.import_module('app')
    assert hasattr(m, 'main')


def test_tastytrade_provider_import_safe():
    m = importlib.import_module('tastytrade_provider')
    assert hasattr(m, 'TastytradeProvider')
