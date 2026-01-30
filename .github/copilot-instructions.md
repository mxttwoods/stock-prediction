# Copilot Instructions - Stock Hedge Analysis Tool

## Project Overview

Single-file Python CLI tool (`put_hedge_v5_improved.py`) for analyzing put option hedging strategies. Uses Yahoo Finance API for market data, Bollinger Bands for technical analysis, and log-normal probability models for option pricing.

## Architecture & Data Flow

```
CLI Args / JSON Config → Data Loading (yfinance + parquet cache) → Technical Analysis
  → Options Chain Filtering → Hedge Recommendations → PDF Report Generation
```

**Key components in execution order:**

1. `parse_cli_args()` + `load_config()` - Configuration with JSON file override
2. `load_price_history()` - Cached price data in `{ticker}_data_file.parquet`
3. `fetch_live_options_chain()` - Live options with strict liquidity filtering
4. Volatility blending: EWMA historical + implied vol (weighted by `IV_WEIGHT`)
5. `recommend_optimal_hedge()` / `recommend_full_protection_hedge()` - Two distinct strategies
6. PDF generation with matplotlib multi-page output

## Critical Conventions

### Probability Calculations

- Uses log-normal GBM model with **configurable drift** (`DRIFT_RATE`, default 5%)
- `calculate_downside_probability()` returns P(S_T ≤ K), not P(S_T ≥ K)
- Time scaling: σ_t = σ × √t (from current date, not BB window)

### Liquidity Filtering (DTE-adaptive)

```python
# Near-expiry requires HIGHER thresholds (see get_liquidity_requirements())
if dte <= 7:  return MIN_VOLUME * 2, MIN_OPEN_INTEREST * 1.5
if dte <= 14: return MIN_VOLUME * 1.5, MIN_OPEN_INTEREST * 1.25
```

Options must also have `bid >= $0.05` (exit liquidity) and `spread_ratio <= MAX_SPREAD_RATIO`.

### Two Hedge Strategies

- **`recommend_optimal_hedge()`**: Maximize efficiency score within budget
- **`recommend_full_protection_hedge()`**: Achieve target coverage, then ladder longer DTEs

## Running the Tool

```bash
# Basic usage
python put_hedge_v5_improved.py --ticker AAPL --shares 100 --budget 500

# With config file
python put_hedge_v5_improved.py --config hedge_config.json

# Key flags
--use-ewma         # EWMA volatility instead of equal-weighted
--drift 0.08       # Set expected annual return (default: 0.05)
--iv-weight 0.8    # Weight implied vol higher (default: 0.7)
--protection-percentile 26  # Lower = more extreme protection
```

## Output Files

- `{ticker}_put_hedge_analysis.pdf` - Multi-page visual report
- `{ticker}_data_file.parquet` - Cached price history (auto-updated)
- `put_hedge_analysis_review.md` - Methodology documentation

## Adding New Analysis Features

When extending analysis functions, follow this pattern from `analyze_put_option()`:

1. Return `None` for invalid/illiquid options (caller filters these)
2. Include `liquidity_score` (0-100 composite metric)
3. Add `exit_strategy` dict via `generate_exit_strategy()`
4. Use `DRIFT_RATE` global for probability calculations

## Dependencies & Environment

Uses virtual environment at `.venv/`. Key packages: `yfinance`, `scipy.stats`, `matplotlib`, `pandas`, `numpy`. See `requirements.txt`.
