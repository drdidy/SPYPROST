# SPY Prophet Streamlit

This repository contains the production Streamlit version of SPY Prophet. It is separate from the Next/Vercel build so the Streamlit app can be deployed directly to Streamlit Cloud without dragging in the web-app stack.

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

## Streamlit Cloud Deploy
1. Create a new Streamlit app from this repository.
2. Set the entrypoint to `app.py`.
3. Add secrets in Streamlit Cloud using `.streamlit/secrets.example.toml` as the template.
4. Deploy.

## What it includes
- SPY hourly data ingest (`yfinance`) with US/Central normalization.
- Prior-day pivot and dynamic line projection engines.
- Bias, signal, decision quality, risk guardrails.
- Prophet Chart, Replay Lab, Options Cockpit, Journal Analytics.
- Options Cockpit uses live Tastytrade quotes when credentials are configured.
- SPY Foresight, Daily Brief, order-flow context, GEX, dark-pool, and flow cards when configured market feeds are available.

## Secrets
Create `.streamlit/secrets.toml` locally or add the same values in Streamlit Cloud secrets. Keep real values out of Git.

```toml
TASTYTRADE_CLIENT_ID = "your_client_id"
TASTYTRADE_CLIENT_SECRET = "your_client_secret"
TASTYTRADE_REFRESH_TOKEN = "your_refresh_token"
TASTYTRADE_ENVIRONMENT = "production"

OPENAI_API_KEY = "your_openai_api_key"
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_ENABLE_WEB_SEARCH = "true"

UNUSUAL_WHALES_API_KEY = "your_unusual_whales_token"
```

- Missing secrets => Options Cockpit stays unavailable until live Tastytrade credentials are configured.
- Missing OpenAI or Unusual Whales secrets => the app omits those external intelligence layers instead of inventing data.
- Never commit secrets.

## Data Sources
- **SPY candles**: loaded from `yfinance`.
- **Options quotes**: live Tastytrade when configured, with delayed yfinance fallback for mark-only review.
- **Order flow, GEX, dark-pool, and options pressure**: Unusual Whales when configured.
- **Daily Brief synthesis and calendar scout**: OpenAI when configured.
- No mock quote mode is used in the product app.

## Replay bias safety
- **Step Replay** hides future candles/signals by default.
- **Full Day Review** enables hindsight/outcome review.

## Auto-Journal
- Sidebar toggle: **Auto-journal live signals** (default OFF).
- Journal path: `data/signal_journal.json`.
- Duplicate-safe upsert prevents rerun spam.
- Malformed JSON is backed up as `data/signal_journal.corrupt.YYYYMMDD_HHMMSS.json`.

## Troubleshooting
- `ModuleNotFoundError: pandas` (or others): reinstall requirements in active venv.
- Empty SPY data: retry and verify network/data availability.
- Missing secrets: app reports missing key names and leaves Options Cockpit unavailable.
- Tastytrade failures: app reports the provider error and does not substitute fake quotes.

## Known limitations
- yfinance may have delays/gaps.
- Hourly candles cannot resolve intrabar target-vs-stop sequence.
- Option projection is delta-only (ignores gamma/IV/theta/liquidity/spread).
- Same-day option Greeks can change rapidly.

## Safety
**No order execution is implemented.**
No submit/cancel/replace/dry-run trading functions are provided.
