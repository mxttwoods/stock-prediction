#!/usr/bin/env python3
"""
Put Hedge V5 - Improved Statistical Accuracy
==============================================

Key Improvements over V4:
1. ✅ Configurable drift parameter (no more zero-drift bias)
2. ✅ Uses implied volatility when available (falls back to historical)
3. ✅ Proper data filtering (no fillna(0) for option prices)
4. ✅ Fixed volatility time-scaling methodology
5. ✅ Probability-weighted protection metrics
6. ✅ EWMA volatility calculation option
7. ✅ Clearer probability calculations with better naming
8. ✅ DTE-normalized efficiency scores
9. ✅ Statistical significance testing for trends

Statistical Accuracy Improvements:
- Eliminates bearish bias from zero-drift assumption
- Uses market's forward-looking IV expectations
- Filters illiquid/missing-data options properly
- Correct time-scaling for volatility projections
- Probability-weighted scenario analysis
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings('ignore')

# =================== CONFIGURATION ===================


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Hedge analysis tool (V5 - Statistically Improved)")
    parser.add_argument(
        "--config", type=Path, help="Path to JSON config file", default=None
    )
    parser.add_argument(
        "--ticker", type=str, default="EOSE", help="Ticker symbol to analyze"
    )
    parser.add_argument("--shares", type=int, default=600, help="Number of shares held")
    parser.add_argument(
        "--budget", type=float, default=1000, help="Target budget for hedge (USD)"
    )
    parser.add_argument(
        "--max-dte",
        type=int,
        default=365,
        help="Max days to expiration to include in options chain",
    )
    parser.add_argument(
        "--lookback", type=int, default=365, help="Lookback window for price history"
    )
    parser.add_argument(
        "--drop-scenarios",
        type=str,
        default="10,25,50,75,90",
        help="Comma-separated drop percentages for scenario analysis (e.g., 10,25,50)",
    )
    parser.add_argument(
        "--bb-window", type=int, default=20, help="Lookback window for Bollinger Bands"
    )
    parser.add_argument(
        "--bb-std",
        type=float,
        default=2.0,
        help="Standard deviations for Bollinger Bands",
    )
    parser.add_argument(
        "--forward-days",
        type=int,
        default=30,
        help="Number of business days for forward projection",
    )
    parser.add_argument(
        "--trend-lookback",
        type=int,
        default=20,
        help="Lookback window in days for calculating recent trend",
    )
    parser.add_argument(
        "--min-volume", type=int, default=5, help="Minimum option volume filter"
    )
    parser.add_argument(
        "--min-oi", type=int, default=25, help="Minimum open interest filter"
    )
    parser.add_argument(
        "--max-spread",
        type=float,
        default=0.8,
        help="Maximum bid/ask spread as a fraction of mid price",
    )
    parser.add_argument(
        "--drift",
        type=float,
        default=0.05,
        help="Expected annual drift/return rate (default 5%%, use 0 for risk-neutral)",
    )
    parser.add_argument(
        "--use-ewma",
        action="store_true",
        help="Use exponentially weighted moving average for volatility (default: equal weight)",
    )
    parser.add_argument(
        "--ewma-span",
        type=int,
        default=60,
        help="Span (half-life) for EWMA volatility calculation in days",
    )
    parser.add_argument(
        "--use-iv",
        action="store_true",
        default=True,
        help="Use implied volatility from options market (default: True)",
    )
    parser.add_argument(
        "--iv-weight",
        type=float,
        default=0.7,
        help="Weight for implied volatility when blending with historical (0.0-1.0, default 0.7)",
    )
    parser.add_argument(
        "--protection-percentile",
        type=float,
        default=26.0,
        help="Target percentile for protection (lower = more extreme protection, default: 26 for realistic severe drops)",
    )
    return parser.parse_args()


def load_config(config_path: Optional[Path]) -> Dict:
    if not config_path:
        return {}

    try:
        with config_path.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


cli_args = parse_cli_args()
file_config = load_config(cli_args.config)

TICKER = file_config.get("ticker", cli_args.ticker)
SHARES_HELD = int(file_config.get("shares", cli_args.shares))
INSURANCE_BUDGET = float(file_config.get("budget", cli_args.budget))
MAX_DAYS_TO_EXPIRATION = int(file_config.get("max_dte", cli_args.max_dte))
LOOKBACK_DAYS = int(file_config.get("lookback", cli_args.lookback))
DROP_SCENARIOS = [
    float(x)
    for x in str(file_config.get("drop_scenarios", cli_args.drop_scenarios))
    .replace(" ", "")
    .split(",")
    if x
]

DATA_FILE = Path(file_config.get("data_file", "data_file.parquet"))

# Bollinger Band parameters
BB_WINDOW = int(file_config.get("bb_window", cli_args.bb_window))
BB_STD = float(file_config.get("bb_std", cli_args.bb_std))

# Forward projection parameters
FORWARD_PROJECTION_DAYS = int(
    file_config.get("forward_projection_days", cli_args.forward_days)
)

TREND_LOOKBACK_WINDOW = int(file_config.get("trend_lookback", cli_args.trend_lookback))

# Liquidity filters
MIN_VOLUME = int(file_config.get("min_volume", cli_args.min_volume))
MIN_OPEN_INTEREST = int(file_config.get("min_oi", cli_args.min_oi))
MAX_SPREAD_RATIO = float(file_config.get("max_spread", cli_args.max_spread))

# NEW: Statistical parameters
DRIFT_RATE = float(file_config.get("drift", cli_args.drift))
USE_EWMA = file_config.get("use_ewma", cli_args.use_ewma)
EWMA_SPAN = int(file_config.get("ewma_span", cli_args.ewma_span))
USE_IMPLIED_VOL = file_config.get("use_iv", cli_args.use_iv)
IV_WEIGHT = float(file_config.get("iv_weight", cli_args.iv_weight))
PROTECTION_PERCENTILE = float(file_config.get("protection_percentile", cli_args.protection_percentile))

# =================== DATA LOADING ===================


def load_price_history(
    ticker: str, lookback_days: int, cache_file: Path
) -> pd.DataFrame:
    """Load daily price data with CSV caching."""
    today = pd.Timestamp.today().normalize()
    start_date = today - pd.Timedelta(days=lookback_days + 5)
    cache_file = ticker.lower() + "_" + str(cache_file)
    cache_file = Path(cache_file)
    df = None

    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        except Exception:
            try:
                cache_file.unlink()
            except FileNotFoundError:
                pass
            df = None

    if df is None or df.empty:
        df = yf.download(
            ticker,
            start=start_date,
            end=today + pd.Timedelta(days=1),
            auto_adjust=False,
            progress=False,
        )
        df.to_parquet(cache_file)
    else:
        last_date = df.index[-1].normalize()
        if last_date < today:
            new = yf.download(
                ticker,
                start=last_date + pd.Timedelta(days=1),
                end=today + pd.Timedelta(days=1),
                auto_adjust=False,
                progress=False,
            )
            if not new.empty:
                new = new[~new.index.isin(df.index)]
                df = pd.concat([df, new])
                df.to_parquet(cache_file)

    cutoff = today - pd.Timedelta(days=lookback_days)
    df = df[df.index >= cutoff]
    return df


def fetch_live_options_chain(ticker: str, max_dte: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch live options chain data from Yahoo Finance.
    Returns DataFrame with all put options within max_dte days.

    IMPROVEMENT: No longer fills missing prices with 0, filters them out instead.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        expirations = ticker_obj.options

        if not expirations:
            return None

        all_puts = []
        today = pd.Timestamp.today().normalize()

        for exp_date_str in expirations:
            try:
                exp_date = pd.to_datetime(exp_date_str)
                dte = (exp_date - today).days

                if 0 < dte <= max_dte:
                    opt_chain = ticker_obj.option_chain(exp_date_str)
                    puts = opt_chain.puts.copy()

                    # Add metadata
                    puts["expiration"] = exp_date_str
                    puts["exp_date"] = exp_date
                    puts["DTE"] = dte
                    puts["strike"] = puts["strike"].astype(float)

                    # IMPROVEMENT: Don't fill with 0, keep as NaN for filtering
                    puts["lastPrice"] = puts["lastPrice"].astype(float)
                    puts["bid"] = puts["bid"].astype(float)
                    puts["ask"] = puts["ask"].astype(float)
                    puts["volume"] = puts["volume"].fillna(0).astype(int)
                    puts["openInterest"] = puts["openInterest"].fillna(0).astype(int)

                    # IMPROVEMENT: Filter out options with no pricing data
                    # Keep only options with valid bid/ask or lastPrice
                    valid_pricing = (
                        (puts["bid"] > 0) & (puts["ask"] > 0)
                    ) | (puts["lastPrice"] > 0)
                    puts = puts[valid_pricing]

                    if not puts.empty:
                        all_puts.append(puts)

            except Exception as e:
                continue

        if all_puts:
            combined = pd.concat(all_puts, ignore_index=True)
            return combined
        else:
            return None

    except Exception as e:
        return None


# =================== MAIN DATA LOAD ===================

df = load_price_history(TICKER, LOOKBACK_DAYS, DATA_FILE)

if df.empty:
    raise SystemExit("❌ No data downloaded. Check ticker or internet connection.")

# Handle MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    cols = df.columns.get_level_values(0)
    if "Adj Close" in set(cols):
        close = df.xs("Adj Close", axis=1, level=0).iloc[:, 0]
    else:
        close = df.xs("Close", axis=1, level=0).iloc[:, 0]
else:
    close = df["Adj Close"].copy() if "Adj Close" in df.columns else df["Close"].copy()

current_price = float(close.iloc[-1])
today = pd.Timestamp.today().normalize()

# Fetch live options
options_df = fetch_live_options_chain(TICKER, MAX_DAYS_TO_EXPIRATION)

# =================== TECHNICAL ANALYSIS ===================

df["SMA20"] = close.rolling(BB_WINDOW).mean()
df["BB_up"] = df["SMA20"] + BB_STD * close.rolling(BB_WINDOW).std()
df["BB_dn"] = df["SMA20"] - BB_STD * close.rolling(BB_WINDOW).std()

# IMPROVEMENT: Calculate volatility using EWMA if requested
returns = close.pct_change().dropna()

if USE_EWMA:
    # Exponentially weighted volatility (more weight on recent data)
    ewma_var = returns.ewm(span=EWMA_SPAN, adjust=False).var()
    daily_vol = np.sqrt(ewma_var.iloc[-1])
    print(f"📊 Using EWMA volatility (span={EWMA_SPAN} days): {daily_vol*100:.2f}% daily")
else:
    # Equal-weighted historical volatility
    daily_vol = returns.std()
    print(f"📊 Using equal-weighted historical volatility: {daily_vol*100:.2f}% daily")

annual_vol_hist = daily_vol * np.sqrt(252)

# IMPROVEMENT: Get average implied volatility from options chain
annual_vol_iv = np.nan
if options_df is not None and not options_df.empty and USE_IMPLIED_VOL:
    valid_iv = options_df["impliedVolatility"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_iv) > 0:
        # Use ATM implied vol (closest to current price)
        options_df["strike_dist"] = np.abs(options_df["strike"] - current_price)
        atm_options = options_df.nsmallest(10, "strike_dist")
        annual_vol_iv = atm_options["impliedVolatility"].median()
        print(f"📊 Market implied volatility (IV): {annual_vol_iv*100:.2f}%")

# IMPROVEMENT: Blend historical and implied volatility
if not np.isnan(annual_vol_iv) and USE_IMPLIED_VOL:
    annual_vol = IV_WEIGHT * annual_vol_iv + (1 - IV_WEIGHT) * annual_vol_hist
    print(f"📊 Using blended volatility ({IV_WEIGHT*100:.0f}% IV, {(1-IV_WEIGHT)*100:.0f}% hist): {annual_vol*100:.2f}%")
    daily_vol = annual_vol / np.sqrt(252)
else:
    annual_vol = annual_vol_hist
    print(f"📊 Using historical volatility only: {annual_vol*100:.2f}%")

# Current BB values
current_sma = (
    df["SMA20"].iloc[-1] if not pd.isna(df["SMA20"].iloc[-1]) else current_price
)
current_std = (
    close.rolling(BB_WINDOW).std().iloc[-1]
    if not pd.isna(close.rolling(BB_WINDOW).std().iloc[-1])
    else daily_vol * current_price
)
bb_upper = current_sma + BB_STD * current_std
bb_lower = current_sma - BB_STD * current_std

# =================== FORWARD PROJECTION ===================


def calculate_trend_with_significance(
    prices: pd.Series, lookback: int, alpha: float = 0.05
) -> Tuple[float, bool, float]:
    """
    Calculate linear trend with statistical significance testing.

    IMPROVEMENT: Tests if trend is statistically significant before using it.

    Returns:
        (trend_slope, is_significant, p_value)
    """
    lookback_size = min(lookback, len(prices))
    if lookback_size < 3:
        return 0.0, False, 1.0

    recent_prices = prices.iloc[-lookback_size:].values
    x = np.arange(len(recent_prices))

    # Fit linear regression
    coeffs = np.polyfit(x, recent_prices, 1)
    trend_slope = coeffs[0]

    # Calculate residuals and standard error
    y_pred = coeffs[0] * x + coeffs[1]
    residuals = recent_prices - y_pred
    residual_std = np.std(residuals, ddof=2)

    # Standard error of slope
    x_mean = np.mean(x)
    se_slope = residual_std / np.sqrt(np.sum((x - x_mean) ** 2))

    # T-statistic and p-value
    t_stat = trend_slope / se_slope if se_slope > 0 else 0
    df_resid = len(x) - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_resid))

    is_significant = p_value < alpha

    return trend_slope, is_significant, p_value


def project_bollinger_bands_forward(current_price, sma, std, days_ahead, daily_vol):
    """
    Project Bollinger Bands forward using proper time-scaling methodology.

    IMPROVEMENTS:
    1. Correct time-scaling: σ_t = σ * √t (from NOW, not from BB window)
    2. Trend only used if statistically significant
    3. Drift parameter incorporated for unbiased projection
    """
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq="B"
    )

    # IMPROVEMENT: Start projection from CURRENT PRICE, not SMA
    # This creates visual continuity - projection emanates from where we are now
    projected_center = current_price
    base_std = std.iloc[-1] if not pd.isna(std.iloc[-1]) else daily_vol * current_price

    # IMPROVEMENT: Calculate trend with significance testing
    lookback_size = min(TREND_LOOKBACK_WINDOW, len(close))
    daily_trend, is_significant, p_value = calculate_trend_with_significance(
        close, lookback_size, alpha=0.05
    )

    if is_significant:
        print(f"📈 Detected significant trend: ${daily_trend:.4f}/day (p={p_value:.4f})")
    else:
        print(f"📊 No significant trend detected (p={p_value:.3f}), using drift only")
        daily_trend = 0.0

    projected_sma = []
    projected_bb_up = []
    projected_bb_dn = []

    # IMPROVEMENT: Use drift for expected move calculation
    daily_drift = (DRIFT_RATE / 252) * current_price

    for i in range(days_ahead):
        # Project center with trend + drift FROM CURRENT PRICE
        days_forward = i + 1
        projected_center += (daily_trend + daily_drift)
        projected_center = max(projected_center, current_price * 0.01)

        # IMPROVEMENT: Proper time-scaling for volatility
        # σ_t = σ * √t where t is days from NOW
        time_scaling = np.sqrt(days_forward / BB_WINDOW)
        projected_std = base_std * time_scaling

        # Bollinger Bands: projected_center ± BB_STD × projected_std
        projected_bb_up_val = projected_center + BB_STD * projected_std
        projected_bb_dn_val = projected_center - BB_STD * projected_std

        projected_sma.append(projected_center)
        projected_bb_up.append(max(projected_bb_up_val, 0))
        projected_bb_dn.append(max(projected_bb_dn_val, 0))

    return (
        future_dates,
        np.array(projected_sma),
        np.array(projected_bb_up),
        np.array(projected_bb_dn),
    )


future_dates, proj_sma, proj_bb_up, proj_bb_dn = project_bollinger_bands_forward(
    current_price,
    df["SMA20"],
    close.rolling(BB_WINDOW).std(),
    FORWARD_PROJECTION_DAYS,
    daily_vol,
)

# Expected move window (projected BB range)
lower_bound_95pct = max(proj_bb_dn[-1], 0)  # IMPROVEMENT: Renamed for clarity
upper_bound_95pct = max(proj_bb_up[-1], 0)

# =================== PROBABILITY CALCULATIONS ===================


def calculate_downside_probability(
    strike_price: float,
    current_price: float,
    days: int,
    daily_vol: float,
    annual_vol: float,
    drift_rate: float = 0.0,
) -> float:
    """
    Calculate probability that price will be AT OR BELOW strike at expiration.

    IMPROVEMENTS:
    1. Renamed for clarity (was ambiguous "price_probability")
    2. Uses configurable drift parameter instead of hardcoded mu=0
    3. Returns P(S_T <= K) which is what matters for puts

    Uses log-normal distribution assumption:
    log(S_T / S_0) ~ N((μ - 0.5σ²)T, σ²T)

    Args:
        strike_price: Target strike price
        current_price: Current underlying price
        days: Days to expiration
        daily_vol: Daily volatility (not used, kept for compatibility)
        annual_vol: Annualized volatility
        drift_rate: Expected annual return/drift (e.g., 0.05 for 5%)

    Returns:
        Probability between 0 and 1 that price <= strike
    """
    if days <= 0:
        return 1.0 if current_price <= strike_price else 0.0

    t = days / 252.0
    mu = drift_rate  # IMPROVEMENT: Configurable drift
    sigma = annual_vol

    log_ratio = np.log(strike_price / current_price)
    mean_log = (mu - 0.5 * sigma**2) * t
    var_log = sigma**2 * t

    if var_log <= 0:
        return 1.0 if current_price <= strike_price else 0.0

    z_score = (log_ratio - mean_log) / np.sqrt(var_log)

    # IMPROVEMENT: This is P(S_T <= K), not P(S_T >= K)
    prob = stats.norm.cdf(z_score)

    return prob


# =================== OPTIONS ANALYSIS & RECOMMENDATIONS ===================


def analyze_put_option(
    put_row: pd.Series,
    current_price: float,
    shares_held: int,
    daily_vol: float,
    annual_vol: float,
    drift_rate: float = 0.0,
) -> Dict:
    """
    Analyze a single put option for hedge effectiveness.

    IMPROVEMENTS:
    1. Uses downside probability (not upside)
    2. Probability-weighted protection metrics
    3. DTE-normalized efficiency score
    """
    strike = float(put_row["strike"])
    bid = float(put_row.get("bid", 0) or 0)
    ask = float(put_row.get("ask", 0) or 0)
    last_price = float(put_row.get("lastPrice", np.nan))

    # IMPROVEMENT: Require valid bid/ask for mid calculation
    if bid <= 0 or ask <= 0:
        mid_price = last_price if not pd.isna(last_price) and last_price > 0 else np.nan
    else:
        mid_price = (bid + ask) / 2

    premium = last_price if not pd.isna(last_price) and last_price > 0 else mid_price

    if pd.isna(premium) or premium <= 0:
        return None  # Skip options with no valid pricing

    spread = abs(ask - bid)
    spread_ratio = spread / mid_price if mid_price and mid_price > 0 else np.inf
    dte = int(put_row["DTE"])
    exp_date = put_row["exp_date"]

    # IMPROVEMENT: Calculate downside probability (P(S_T <= K))
    prob_itm = calculate_downside_probability(
        strike, current_price, dte, daily_vol, annual_vol, drift_rate
    )

    # IMPROVEMENT: Probability-weighted protection scenarios
    scenarios = [0.5, 0.6, 0.7, 0.8, 0.9]
    scenario_prices = [current_price * s for s in scenarios]

    # Calculate probability of each scenario using normal distribution
    scenario_probs = []
    for scenario_price in scenario_prices:
        prob_below = calculate_downside_probability(
            scenario_price, current_price, dte, daily_vol, annual_vol, drift_rate
        )
        scenario_probs.append(prob_below)

    # Normalize probabilities
    total_prob = sum(scenario_probs)
    if total_prob > 0:
        scenario_weights = [p / total_prob for p in scenario_probs]
    else:
        scenario_weights = [1.0 / len(scenarios)] * len(scenarios)

    weighted_protection = 0.0
    for scenario_price, weight in zip(scenario_prices, scenario_weights):
        intrinsic = max(strike - scenario_price, 0.0)
        weighted_protection += intrinsic * weight

    # Cost per contract
    cost_per_contract = premium * 100

    # Protection per dollar (efficiency metric)
    protection_per_dollar = weighted_protection / premium if premium > 0 else 0

    # Expected value (probability weighted)
    expected_value = prob_itm * weighted_protection - premium

    # IMPROVEMENT: DTE-normalized efficiency score
    # Normalize to 30-day equivalent for comparison
    dte_factor = np.sqrt(30.0 / max(dte, 1))
    efficiency_score = protection_per_dollar * prob_itm * dte_factor * 100

    # Distance from current price
    distance_pct = ((strike - current_price) / current_price) * 100

    return {
        "strike": strike,
        "premium": premium,
        "dte": dte,
        "exp_date": exp_date,
        "expiration": put_row["expiration"],
        "prob_itm": prob_itm,
        "protection_per_dollar": protection_per_dollar,
        "expected_value": expected_value,
        "efficiency_score": efficiency_score,
        "distance_pct": distance_pct,
        "cost_per_contract": cost_per_contract,
        "volume": int(put_row["volume"]) if "volume" in put_row else 0,
        "open_interest": int(put_row["openInterest"])
        if "openInterest" in put_row
        else 0,
        "bid": bid,
        "ask": ask,
        "mid": mid_price,
        "spread": spread,
        "spread_ratio": spread_ratio,
        "implied_vol": float(put_row.get("impliedVolatility", np.nan)),
    }


def recommend_optimal_hedge(
    options_df: pd.DataFrame,
    current_price: float,
    shares_held: int,
    budget: float,
    daily_vol: float,
    annual_vol: float,
    lower_bound: float,
    drift_rate: float = 0.0,
) -> List[Dict]:
    """
    Recommend optimal put positions based on:
    - Expected move window (Bollinger Bands)
    - Probability analysis
    - Cost efficiency
    - Budget constraints

    IMPROVEMENTS:
    1. Filters out None results from analyze_put_option
    2. Uses drift rate in analysis
    """
    if options_df is None or options_df.empty:
        return []

    # Analyze all puts
    analyzed_puts = []
    for idx, row in options_df.iterrows():
        try:
            analysis = analyze_put_option(
                row, current_price, shares_held, daily_vol, annual_vol, drift_rate
            )
            if analysis is not None:  # IMPROVEMENT: Skip invalid options
                analyzed_puts.append(analysis)
        except Exception as e:
            continue

    if not analyzed_puts:
        return []

    analyzed_df = pd.DataFrame(analyzed_puts)

    # Filter criteria:
    # 1. Strikes below lower bound (likely to provide protection)
    # 2. Reasonable liquidity (volume or open interest)
    # 3. Within budget constraints

    # Focus on strikes that would be in-the-money if price drops to lower bound
    protective_puts = analyzed_df[
        (analyzed_df["strike"] >= lower_bound * 0.9)  # Not too far OTM
        & (analyzed_df["strike"] <= current_price * 0.95)  # Below current price
        & (analyzed_df["premium"] > 0)
        & (analyzed_df["efficiency_score"] > 0)
        & (analyzed_df["volume"] >= MIN_VOLUME)
        & (analyzed_df["open_interest"] >= MIN_OPEN_INTEREST)
        & (analyzed_df["spread_ratio"] <= MAX_SPREAD_RATIO)
    ].copy()

    if protective_puts.empty:
        # Fallback: use all puts sorted by efficiency
        protective_puts = analyzed_df[
            (analyzed_df["strike"] <= current_price)
            & (analyzed_df["premium"] > 0)
            & (analyzed_df["volume"] >= MIN_VOLUME)
            & (analyzed_df["open_interest"] >= MIN_OPEN_INTEREST)
            & (analyzed_df["spread_ratio"] <= MAX_SPREAD_RATIO)
        ].copy()

    # Sort by efficiency score
    protective_puts = protective_puts.sort_values("efficiency_score", ascending=False)

    # Recommend positions within budget
    recommendations = []
    total_cost = 0.0

    # Strategy: Recommend a ladder of puts at different strikes/expirations
    # This provides protection across time and price levels

    # Group by expiration to get variety
    for exp_date in protective_puts["exp_date"].unique():
        exp_puts = protective_puts[protective_puts["exp_date"] == exp_date]

        # Get best strike for this expiration
        best_put = exp_puts.iloc[0]

        # Calculate how many contracts we can afford
        cost_per_contract = best_put["cost_per_contract"]
        remaining_budget = budget - total_cost

        if cost_per_contract > 0 and remaining_budget >= cost_per_contract:
            contracts = min(
                int(remaining_budget / cost_per_contract), int(shares_held / 100)
            )  # Don't over-hedge

            if contracts > 0:
                position_cost = cost_per_contract * contracts
                total_cost += position_cost

                recommendations.append(
                    {
                        "strike": best_put["strike"],
                        "premium": best_put["premium"],
                        "contracts": contracts,
                        "dte": best_put["dte"],
                        "exp_date": best_put["exp_date"],
                        "expiration": best_put["expiration"],
                        "cost": position_cost,
                        "efficiency_score": best_put["efficiency_score"],
                        "prob_itm": best_put["prob_itm"],
                        "protection_per_dollar": best_put["protection_per_dollar"],
                    }
                )

    return recommendations


def recommend_full_protection_hedge(
    options_df: pd.DataFrame,
    current_price: float,
    shares_held: int,
    budget: float,
    daily_vol: float,
    annual_vol: float,
    target_drop_pct: float,
    protection_level: float = 1.0,
    drift_rate: float = 0.0,
    max_budget_multiplier: float = 3.0,
) -> Tuple[List[Dict], Dict]:
    """
    Find hedge that achieves TARGET PROTECTION LEVEL at specified drop.

    CRITICAL DIFFERENCE from recommend_optimal_hedge:
    - OLD: Maximize efficiency within budget
    - NEW: Achieve protection target, minimize cost

    Args:
        target_drop_pct: Target drop to protect against (e.g., 50 for 50%)
        protection_level: 1.0 = 100% coverage (breakeven), 0.5 = 50% coverage
        max_budget_multiplier: Allow up to N× budget if needed

    Returns:
        (recommendations, diagnostics_dict)

    Algorithm:
    1. Calculate required protection at target drop
    2. Find strikes that will be ITM at that price
    3. Use greedy algorithm to build hedge achieving target
    4. Prioritize: ATM/ITM puts > efficiency > cost
    """
    if options_df is None or options_df.empty:
        return [], {"error": "No options data"}

    # Calculate protection needed
    scenario_price = current_price * (1 - target_drop_pct / 100)
    stock_loss = (scenario_price - current_price) * shares_held
    required_protection = abs(stock_loss) * protection_level

    print(f"\n🎯 FULL PROTECTION MODE")
    print(f"Target drop: {target_drop_pct}% → Price: ${scenario_price:.2f}")
    print(f"Stock loss at drop: ${stock_loss:,.0f}")
    print(f"Required protection ({protection_level*100:.0f}%): ${required_protection:,.0f}")

    # Analyze all options
    analyzed_puts = []
    for idx, row in options_df.iterrows():
        try:
            analysis = analyze_put_option(
                row, current_price, shares_held, daily_vol, annual_vol, drift_rate
            )
            if analysis is not None:
                # Calculate protection at target drop scenario
                strike = analysis["strike"]
                intrinsic_at_drop = max(strike - scenario_price, 0.0)
                analysis["protection_at_target"] = intrinsic_at_drop
                analyzed_puts.append(analysis)
        except Exception:
            continue

    if not analyzed_puts:
        return [], {"error": "No valid options found"}

    analyzed_df = pd.DataFrame(analyzed_puts)

    # Filter for liquid, tradeable options
    valid_puts = analyzed_df[
        (analyzed_df["premium"] > 0)
        & (analyzed_df["volume"] >= MIN_VOLUME)
        & (analyzed_df["open_interest"] >= MIN_OPEN_INTEREST)
        & (analyzed_df["spread_ratio"] <= MAX_SPREAD_RATIO)
    ].copy()

    if valid_puts.empty:
        return [], {"error": "No liquid options available"}

    # Focus on puts that will provide protection at target drop
    # Prioritize strikes near or above the drop price
    protective_puts = valid_puts[
        valid_puts["strike"] >= scenario_price * 0.95
    ].copy()

    if protective_puts.empty:
        # Fallback: use all puts
        protective_puts = valid_puts.copy()

    # GREEDY ALGORITHM: Build hedge to achieve target protection
    # Sort by "bang for buck" at the target drop scenario
    protective_puts["protection_per_cost"] = (
        protective_puts["protection_at_target"] /
        protective_puts["cost_per_contract"].replace(0, np.inf)
    )

    # Secondary sort: prefer higher strikes (more ITM at drop)
    protective_puts = protective_puts.sort_values(
        ["protection_per_cost", "strike"],
        ascending=[False, False]
    )

    # IMPROVEMENT: Build TIME-DIVERSIFIED LADDER instead of greedy short-term selection
    # Strategy: Allocate budget across time horizons (7-14d, 14-30d, 30-60d, 60-90d)
    # This provides protection across different timeframes

    recommendations = []
    total_cost = 0.0
    total_protection = 0.0
    max_allowed_cost = budget * max_budget_multiplier

    print(f"Max allowed cost: ${max_allowed_cost:,.0f} ({max_budget_multiplier}× budget)")

    # PROGRESSIVE LADDER STRATEGY:
    # 1. First: Achieve 100% coverage at target drop using short-term (5-21d)
    # 2. Then: Use surplus budget to progressively add longer DTEs (14-30d, 30-60d, 60-90d)

    # Phase 1: Short-term protection to hit 100% coverage
    recommendations = []
    total_cost = 0.0
    total_protection = 0.0

    print(f"\n🎯 PHASE 1: Achieving 100% coverage with short-term options...")

    short_term_puts = protective_puts[protective_puts["dte"] <= 21].copy()
    short_term_puts = short_term_puts.sort_values("protection_per_cost", ascending=False)

    for idx, put_option in short_term_puts.iterrows():
        if total_protection >= required_protection:
            break  # Coverage achieved!

        if total_cost >= max_allowed_cost:
            break

        protection_gap = required_protection - total_protection
        cost_per_contract = put_option["cost_per_contract"]
        protection_per_contract = put_option["protection_at_target"] * 100

        if protection_per_contract <= 0:
            continue

        contracts_for_gap = int(np.ceil(protection_gap / protection_per_contract))
        max_contracts = int(shares_held / 100)
        remaining_budget = max_allowed_cost - total_cost
        contracts_affordable = int(remaining_budget / cost_per_contract)
        contracts = min(contracts_for_gap, max_contracts, contracts_affordable, 10)

        if contracts <= 0:
            continue

        position_cost = cost_per_contract * contracts
        position_protection = protection_per_contract * contracts

        recommendations.append({
            "strike": put_option["strike"],
            "premium": put_option["premium"],
            "contracts": contracts,
            "dte": put_option["dte"],
            "exp_date": put_option["exp_date"],
            "expiration": put_option["expiration"],
            "cost": position_cost,
            "efficiency_score": put_option["efficiency_score"],
            "prob_itm": put_option["prob_itm"],
            "protection_per_dollar": put_option["protection_per_dollar"],
            "protection_at_target": position_protection,
            "phase": "Coverage",
        })

        total_cost += position_cost
        total_protection += position_protection

    coverage_achieved = total_protection >= required_protection
    surplus_budget = max_allowed_cost - total_cost

    print(f"   Coverage: {(total_protection/required_protection)*100:.1f}% | Cost: ${total_cost:.0f} | Surplus: ${surplus_budget:.0f}")

    # Phase 2: If coverage achieved and surplus budget exists, add longer DTEs
    if coverage_achieved and surplus_budget > 20:  # At least $20 surplus
        print(f"\n🎯 PHASE 2: Adding longer-term options with ${surplus_budget:.0f} surplus budget...")

        # Define DTE tiers for surplus budget
        dte_tiers = [
            (21, 45, "21-45d", 0.50),   # 50% to mid-term
            (45, 75, "45-75d", 0.35),   # 35% to longer-term
            (75, 90, "75-90d", 0.15),   # 15% to longest-term
        ]

        for (min_dte, max_dte, label, weight) in dte_tiers:
            tier_budget = surplus_budget * weight

            if tier_budget < 10:  # Skip if less than $10
                continue

            tier_puts = protective_puts[
                (protective_puts["dte"] >= min_dte) &
                (protective_puts["dte"] <= max_dte)
            ].copy()

            if tier_puts.empty:
                continue

            tier_puts = tier_puts.sort_values("protection_per_cost", ascending=False)
            tier_cost = 0.0

            for idx, put_option in tier_puts.iterrows():
                if tier_cost >= tier_budget or total_cost >= max_allowed_cost:
                    break

                cost_per_contract = put_option["cost_per_contract"]
                protection_per_contract = put_option["protection_at_target"] * 100

                # For surplus phase, buy 1-2 contracts per strike for rollability
                remaining = min(tier_budget - tier_cost, max_allowed_cost - total_cost)
                contracts_affordable = int(remaining / cost_per_contract)
                contracts = min(contracts_affordable, 3, int(shares_held / 100))  # Cap at 3 for diversity

                if contracts <= 0:
                    continue

                position_cost = cost_per_contract * contracts
                position_protection = protection_per_contract * contracts

                recommendations.append({
                    "strike": put_option["strike"],
                    "premium": put_option["premium"],
                    "contracts": contracts,
                    "dte": put_option["dte"],
                    "exp_date": put_option["exp_date"],
                    "expiration": put_option["expiration"],
                    "cost": position_cost,
                    "efficiency_score": put_option["efficiency_score"],
                    "prob_itm": put_option["prob_itm"],
                    "protection_per_dollar": put_option["protection_per_dollar"],
                    "protection_at_target": position_protection,
                    "phase": f"Rollability-{label}",
                })

                total_cost += position_cost
                total_protection += position_protection
                tier_cost += position_cost

            if tier_cost > 0:
                print(f"   {label}: ${tier_cost:.0f} spent ({tier_cost/tier_budget*100:.0f}% of tier budget)")

    # Calculate final metrics
    net_pl_at_drop = stock_loss + total_protection - total_cost
    protection_pct = (total_protection / abs(stock_loss)) * 100 if stock_loss != 0 else 0

    diagnostics = {
        "target_drop_pct": target_drop_pct,
        "scenario_price": scenario_price,
        "stock_loss": stock_loss,
        "required_protection": required_protection,
        "total_cost": total_cost,
        "total_protection": total_protection,
        "net_pl_at_drop": net_pl_at_drop,
        "protection_pct": protection_pct,
        "budget": budget,
        "over_budget": total_cost > budget,
        "achieved_target": total_protection >= required_protection,
    }

    print(f"\n📊 HEDGE RESULTS:")
    print(f"Total cost: ${total_cost:,.0f} ({total_cost/budget:.1f}× budget)")
    print(f"Total protection: ${total_protection:,.0f}")
    print(f"Net P/L at {target_drop_pct}% drop: ${net_pl_at_drop:,.0f}")
    print(f"Protection level: {protection_pct:.1f}%")

    if net_pl_at_drop >= 0:
        print(f"✅ TARGET ACHIEVED! You'll break even or profit at {target_drop_pct}% drop")
    elif protection_pct >= 80:
        print(f"⚠️  Close! {protection_pct:.0f}% protected (${abs(net_pl_at_drop):,.0f} loss)")
    else:
        print(f"❌ Target not achieved. Need more budget or adjust filters.")

    return recommendations, diagnostics


# =================== GENERATE RECOMMENDATIONS ===================

# IMPROVEMENT: Calculate target drop from STATISTICAL MODEL (10th percentile)
# Instead of arbitrary 50%, use the actual expected distribution
# This adapts to: volatility, drift, and time horizon

if options_df is not None and not options_df.empty:
    # Get maximum DTE from available options to set time horizon
    max_dte_available = int(options_df['DTE'].max()) if len(options_df) > 0 else 30
    max_dte_available = min(max_dte_available, 90)  # Cap at 90 days for practicality

    # Calculate target percentile price (configurable - default 25th percentile)
    # Using log-normal distribution: log(S_T/S_0) ~ N((μ - 0.5σ²)T, σ²T)
    t = max_dte_available / 252.0
    z_target = norm.ppf(PROTECTION_PERCENTILE / 100.0)
    log_return_target = (DRIFT_RATE - 0.5 * annual_vol**2) * t + z_target * annual_vol * np.sqrt(t)
    price_at_target_percentile = current_price * np.exp(log_return_target)

    # Convert to drop percentage
    TARGET_DROP_FOR_PROTECTION = ((current_price - price_at_target_percentile) / current_price) * 100

    print(f"\n📊 Statistical Target Calculation:")
    print(f"Time horizon: {max_dte_available} days")
    print(f"{PROTECTION_PERCENTILE:.0f}th percentile price: ${price_at_target_percentile:.2f}")
    print(f"Target drop for protection: {TARGET_DROP_FOR_PROTECTION:.1f}%")
    confidence_level = 100 - PROTECTION_PERCENTILE
    print(f"(Protecting against {confidence_level:.0f}% worst-case scenarios)")

    # USER REQUIREMENT: Target 40% drop for guaranteed profit
    # Override statistical calculation with user's specific requirement
    TARGET_DROP_FOR_PROTECTION = 40.0  # User wants profit at 40% drop
    price_at_40pct_drop = current_price * 0.6

    print(f"\n📊 User-Specified Target:")
    print(f"Target drop: 40% → Price: ${price_at_40pct_drop:.2f}")
    print(f"(User requirement: PROFIT at this level within $100 budget)")

    # Use FULL PROTECTION optimizer with STRICT budget constraint
    recommendations, protection_diagnostics = recommend_full_protection_hedge(
        options_df,
        current_price,
        SHARES_HELD,
        INSURANCE_BUDGET,
        daily_vol,
        annual_vol,
        target_drop_pct=TARGET_DROP_FOR_PROTECTION,
        protection_level=1.0,  # 100% protection (breakeven/profit)
        drift_rate=DRIFT_RATE,
        max_budget_multiplier=1.0,  # STRICT $100 cap - no exceeding
    )
else:
    recommendations = []
    protection_diagnostics = {}
    TARGET_DROP_FOR_PROTECTION = 50.0  # Fallback

# =================== CONSOLE SUMMARY ===================

print("\n" + "="*60)
print(f"🎯 PUT HEDGE ANALYSIS v5 - {TICKER}")
print("="*60)
print(f"Current Price: ${current_price:.2f}")
print(f"Shares Held: {SHARES_HELD:,}")
print(f"Hedge Budget: ${INSURANCE_BUDGET:,.2f}")
print(f"\n📊 Volatility Analysis:")
print(f"  Historical Vol: {annual_vol_hist*100:.2f}%")
if not np.isnan(annual_vol_iv):
    print(f"  Implied Vol (IV): {annual_vol_iv*100:.2f}%")
print(f"  Using: {annual_vol*100:.2f}% annualized")
print(f"\n📈 Forward Projection ({FORWARD_PROJECTION_DAYS} days):")
print(f"  Lower bound (95%): ${lower_bound_95pct:.2f}")
print(f"  Upper bound (95%): ${upper_bound_95pct:.2f}")
print(f"  Drift assumption: {DRIFT_RATE*100:.1f}% annually")

if recommendations:
    print(f"\n✅ Recommended Hedge Positions ({len(recommendations)}):")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['dte']}d Put @ ${rec['strike']:.2f} × {rec['contracts']} contracts")
        print(f"     Cost: ${rec['cost']:.0f} | ITM Prob: {rec['prob_itm']*100:.1f}% | Eff: {rec['efficiency_score']:.1f}")
else:
    print("\n⚠️  No eligible put options found for the configured filters.")

print("="*60 + "\n")

# =================== P/L CALCULATIONS & SCENARIO ANALYSIS ===================


def portfolio_pl_at_price(
    price: float, positions: List[Dict], target_date: Optional[pd.Timestamp] = None
) -> Dict:
    """Calculate portfolio P/L at given price with specified positions."""
    stock_pl = (price - current_price) * SHARES_HELD

    opts_pl = 0.0
    prem_paid = 0.0

    for pos in positions:
        exp_date = pos.get(
            "exp_date", today + pd.Timedelta(days=pos.get("days_to_exp", 0))
        )

        if target_date is not None and exp_date < target_date:
            continue

        strike = pos["strike"]
        premium = pos.get("premium", 0)
        contracts = pos.get("contracts", 0)

        intrinsic = max(strike - price, 0.0) * 100 * contracts
        cost = premium * 100 * contracts

        prem_paid += cost
        opts_pl += intrinsic - cost

    return {
        "total": stock_pl + opts_pl,
        "stock": stock_pl,
        "options": opts_pl,
        "premium_paid": prem_paid,
    }


# Calculate scenario analysis using the SAME target as protection optimizer
worst_drop_pct = TARGET_DROP_FOR_PROTECTION  # Use 50% drop target
scenario_price_down = current_price * (1 - worst_drop_pct / 100)
scenario_hedged = portfolio_pl_at_price(scenario_price_down, recommendations)
scenario_unhedged = portfolio_pl_at_price(scenario_price_down, [])

hedge_benefit = scenario_hedged["total"] - scenario_unhedged["total"]
protection_pct = (
    (hedge_benefit / abs(scenario_unhedged["total"])) * 100
    if scenario_unhedged["total"] < 0
    else 0
)

# Calculate multiple drop scenarios for reporting & exports
scenario_results = []
for drop_pct in sorted(DROP_SCENARIOS):
    scenario_price = current_price * (1 - drop_pct / 100)
    unhedged = portfolio_pl_at_price(scenario_price, [])
    hedged = portfolio_pl_at_price(scenario_price, recommendations)
    scenario_results.append(
        {
            "drop_pct": drop_pct,
            "scenario_price": scenario_price,
            "unhedged_total": unhedged["total"],
            "hedged_total": hedged["total"],
            "benefit": hedged["total"] - unhedged["total"],
            "protection_pct": (
                (hedged["total"] - unhedged["total"]) / abs(unhedged["total"]) * 100
                if unhedged["total"] < 0
                else 0
            ),
        }
    )


def backtest_drawdowns(
    price_series: pd.Series,
    positions: List[Dict],
    lookback_days: int = 180,
    top_n: int = 8,
    drop_threshold: float = 0.05,
) -> pd.DataFrame:
    """Evaluate hedge benefit across the worst daily drawdowns in recent history."""

    if price_series.empty:
        return pd.DataFrame()

    recent = price_series.tail(lookback_days)
    returns = recent.pct_change().dropna()
    worst_returns = returns[returns <= -drop_threshold].nsmallest(
        min(top_n, len(returns))
    )

    records = []
    for drop_date, pct_change in worst_returns.items():
        price_at_drop = recent.loc[drop_date]
        unhedged = portfolio_pl_at_price(price_at_drop, [], target_date=drop_date)
        hedged = portfolio_pl_at_price(price_at_drop, positions, target_date=drop_date)
        records.append(
            {
                "date": drop_date.strftime("%Y-%m-%d"),
                "pct_change": pct_change * 100,
                "price": price_at_drop,
                "unhedged_total": unhedged["total"],
                "hedged_total": hedged["total"],
                "benefit": hedged["total"] - unhedged["total"],
                "protection_pct": (
                    (hedged["total"] - unhedged["total"]) / abs(unhedged["total"]) * 100
                    if unhedged["total"] < 0
                    else 0
                ),
            }
        )

    return pd.DataFrame(records)


backtest_df = backtest_drawdowns(close, recommendations)

# =================== VISUALIZATIONS ===================

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3, height_ratios=[2.5, 2, 2, 1.5])

# ===== PLOT 1: Price with Forward Projection & Options Overlay =====
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df.index, close, label="Adj Close", linewidth=2, color="#2E86AB")
ax1.plot(
    df.index, df["SMA20"], label="SMA20", linestyle="--", alpha=0.7, color="#A23B72"
)
ax1.fill_between(
    df.index, df["BB_dn"], df["BB_up"], alpha=0.2, color="gray", label="Historical BB"
)

# Forward projections - Clean and simple
# Plot projected Bollinger Bands (traditional style, but starting from current price)
ax1.plot(
    future_dates,
    proj_sma,
    linestyle="--",
    alpha=0.6,
    color="#A23B72",
    linewidth=2,
    label=f"Projected Center: ${proj_sma[-1]:.2f}",
)

ax1.fill_between(
    future_dates,
    proj_bb_dn,
    proj_bb_up,
    alpha=0.15,
    color="orange",
    label=f"Projected Range (±2σ)",
)

ax1.plot(
    future_dates, proj_bb_up, linestyle=":", alpha=0.7, color="orange", linewidth=1.5
)
ax1.plot(
    future_dates, proj_bb_dn, linestyle=":", alpha=0.7, color="orange", linewidth=1.5
)

# Add ONLY key percentile reference lines (not cluttering with bands)
t = FORWARD_PROJECTION_DAYS / 252.0
sigma = annual_vol

# Calculate important percentiles
key_percentiles = {
    10: ("#D32F2F", "10th %ile (Severe Drop)"),  # Red - severe drop
    25: ("#FF6F00", "25th %ile (Bad Scenario)"),  # Orange - bad scenario
    50: ("#FFC107", "50th %ile (Median)"),  # Yellow - median
    75: ("#8BC34A", "75th %ile"),  # Light green
    90: ("#4CAF50", "90th %ile (Upside)"),  # Green - upside
}

for percentile, (color, label) in key_percentiles.items():
    z_score = norm.ppf(percentile / 100.0)
    log_return = (DRIFT_RATE - 0.5 * sigma**2) * t + z_score * sigma * np.sqrt(t)
    price_at_percentile = current_price * np.exp(log_return)

    # Plot as horizontal reference line
    ax1.axhline(
        price_at_percentile,
        linestyle=":",
        linewidth=1.2,
        color=color,
        alpha=0.6,
        label=f"{label}: ${price_at_percentile:.2f}",
    )

# Mark expected move window
ax1.axhline(
    lower_bound_95pct,
    linestyle="--",
    linewidth=2,
    color="red",
    alpha=0.7,
    label=f"Lower Bound (95%): ${lower_bound_95pct:.2f}",
)
ax1.axhline(
    upper_bound_95pct,
    linestyle="--",
    linewidth=2,
    color="green",
    alpha=0.7,
    label=f"Upper Bound (95%): ${upper_bound_95pct:.2f}",
)

# Overlay recommended put strikes
strike_colors = ["#F18F01", "#C73E1D", "#6A994E", "#8B5CF6", "#06A77D"]
for i, pos in enumerate(recommendations):
    strike = pos["strike"]
    exp_date = pos.get("exp_date", today + pd.Timedelta(days=pos.get("dte", 0)))
    label = f"{pos.get('dte', 'N/A')}d Put @ ${strike:.1f}"
    ax1.axhline(
        strike,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        color=strike_colors[i % len(strike_colors)],
        label=label,
    )
    if exp_date <= future_dates[-1]:
        ax1.axvline(
            exp_date,
            linestyle=":",
            alpha=0.4,
            color=strike_colors[i % len(strike_colors)],
            linewidth=1,
        )

ax1.axhline(
    current_price,
    color="green",
    linestyle="-",
    linewidth=1.5,
    alpha=0.5,
    label=f"Current: {current_price:.2f}",
)

ax1.set_title(
    f"{TICKER} Price History with Forward Projections & Put Strikes (V5 - Statistically Improved)",
    fontsize=14,
    fontweight="bold",
)
ax1.set_ylabel("Price ($)", fontsize=11)
ax1.legend(loc="best", fontsize=8, ncol=3)
ax1.grid(True, alpha=0.3)

# ===== PLOT 2: P/L Comparison =====
ax2 = fig.add_subplot(gs[1, 0])
prices = np.linspace(current_price * 0.1, current_price * 1.6, 500)
pl_stock_only = [portfolio_pl_at_price(p, [])["total"] for p in prices]
pl_with_hedge = [portfolio_pl_at_price(p, recommendations)["total"] for p in prices]

ax2.plot(
    prices,
    pl_stock_only,
    label="Unhedged",
    linewidth=2.5,
    color="#D62828",
    linestyle="--",
    alpha=0.8,
)
ax2.plot(
    prices,
    pl_with_hedge,
    label="Hedged",
    linewidth=2.5,
    color="#06A77D",
    linestyle="-",
)

ax2.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
ax2.axvline(current_price, color="green", linewidth=1, linestyle=":", alpha=0.5)
ax2.axvline(
    lower_bound_95pct,
    color="red",
    linewidth=1.5,
    linestyle="--",
    alpha=0.7,
    label="Lower Bound",
)

ax2.set_title("P/L Comparison: Hedged vs Unhedged", fontsize=12, fontweight="bold")
ax2.set_xlabel(f"{TICKER} Price ($)", fontsize=10)
ax2.set_ylabel("Net P/L ($)", fontsize=10)
ax2.xaxis.set_major_locator(ticker.AutoLocator())
ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.yaxis.set_major_locator(ticker.AutoLocator())
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.legend(loc="best", fontsize=9)
ax2.grid(True, alpha=0.3)

# ===== PLOT 3: Options Efficiency Analysis =====
ax3 = fig.add_subplot(gs[1, 1])
if options_df is not None and not options_df.empty:
    strikes_available = options_df["strike"].values
    premiums_available = (
        options_df["lastPrice"]
        .fillna((options_df["bid"] + options_df["ask"]) / 2)
        .values
    )

    efficiencies = []
    for idx, row in options_df.iterrows():
        try:
            analysis = analyze_put_option(
                row, current_price, SHARES_HELD, daily_vol, annual_vol, DRIFT_RATE
            )
            if analysis:
                efficiencies.append(analysis["efficiency_score"])
            else:
                efficiencies.append(0)
        except:
            efficiencies.append(0)

    scatter = ax3.scatter(
        strikes_available,
        efficiencies,
        alpha=0.6,
        s=50,
        c=premiums_available,
        cmap="viridis",
        label="Available Puts",
    )
    fig.colorbar(scatter, ax=ax3, label="Premium ($)")

    if recommendations:
        rec_strikes = [r["strike"] for r in recommendations]
        rec_effs = [r["efficiency_score"] for r in recommendations]
        ax3.scatter(
            rec_strikes,
            rec_effs,
            s=200,
            marker="*",
            color="red",
            edgecolors="black",
            linewidths=2,
            label="Recommended",
            zorder=5,
        )

    ax3.axvline(
        lower_bound_95pct,
        linestyle="--",
        linewidth=2,
        color="red",
        alpha=0.7,
        label="Lower Bound",
    )
    ax3.axvline(
        current_price,
        linestyle=":",
        linewidth=1,
        color="green",
        alpha=0.5,
        label="Current Price",
    )

    ax3.set_title("Put Options Efficiency Analysis", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Strike Price ($)", fontsize=10)
    ax3.set_ylabel("Efficiency Score (DTE-Normalized)", fontsize=10)
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(
        0.5,
        0.5,
        "No live options data available",
        ha="center",
        va="center",
        transform=ax3.transAxes,
        fontsize=12,
    )
    ax3.set_title("Options Efficiency Analysis", fontsize=12, fontweight="bold")

# ===== PLOT 4: Probability Distribution =====
ax4 = fig.add_subplot(gs[2, :])
price_range = np.linspace(current_price * 0.5, current_price * 1.5, 200)
max_dte = max([p.get("dte", 30) for p in recommendations]) if recommendations else 30

# IMPROVEMENT: Use calculate_downside_probability with drift
probabilities = []
for p in price_range:
    prob_below = calculate_downside_probability(
        p, current_price, max_dte, daily_vol, annual_vol, DRIFT_RATE
    )
    probabilities.append(prob_below * 100)

ax4.plot(
    price_range,
    probabilities,
    linewidth=2.5,
    color="#8B5CF6",
    label=f"Downside Probability (next {max_dte} days)",
)
ax4.fill_between(price_range, 0, probabilities, alpha=0.3, color="#8B5CF6")

ax4.axvline(current_price, color="green", linewidth=1.5, linestyle="-", alpha=0.7)
ax4.axvline(
    lower_bound_95pct,
    color="red",
    linewidth=2,
    linestyle="--",
    alpha=0.7,
    label="Lower Bound",
)
ax4.axvspan(
    proj_bb_dn[-1],
    proj_bb_up[-1],
    alpha=0.2,
    color="orange",
    label="Projected BB Range",
)

for pos in recommendations:
    strike = pos["strike"]
    dte = pos.get("dte", max_dte)
    prob = (
        calculate_downside_probability(
            strike, current_price, dte, daily_vol, annual_vol, DRIFT_RATE
        )
        * 100
    )
    ax4.axvline(strike, linestyle="--", alpha=0.6, linewidth=1)
    ax4.text(
        strike,
        prob + 2,
        f"${strike:.1f}\n({prob:.1f}%)",
        ha="center",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

ax4.set_title(
    "Probability Distribution & Put Strike Analysis", fontsize=12, fontweight="bold"
)
ax4.set_xlabel(f"{TICKER} Price ($)", fontsize=10)
ax4.set_ylabel("Probability Below Price (%)", fontsize=10)
ax4.legend(loc="best", fontsize=8)
ax4.grid(True, alpha=0.3)

# ===== PLOT 5: Position Summary =====
ax5 = fig.add_subplot(gs[3, :])
ax5.axis("off")

y_pos = 0.7
bar_height = 0.12
spacing = 0.22

for i, pos in enumerate(recommendations):
    strike = pos["strike"]
    premium = pos.get("premium", 0)
    contracts = pos.get("contracts", 0)
    dte = pos.get("dte", 0)
    exp_date = pos.get("exp_date", today + pd.Timedelta(days=dte))
    cost = premium * 100 * contracts

    intrinsic_worst = max(strike - scenario_price_down, 0) * 100 * contracts
    net_value_worst = intrinsic_worst - cost
    prob_itm = (
        calculate_downside_probability(
            strike, current_price, dte, daily_vol, annual_vol, DRIFT_RATE
        )
        * 100
    )

    x_start = 0.05
    x_width = 0.9
    y_start = y_pos - bar_height / 2
    color = "#90EE90" if strike > scenario_price_down else "#FFB6C1"

    rect = Rectangle(
        (x_start, y_start),
        x_width,
        bar_height,
        facecolor=color,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.6,
    )
    ax5.add_patch(rect)

    strike_text = f"{dte}d Put @ ${strike:.1f} × {contracts} contracts | Exp: {exp_date.strftime('%Y-%m-%d')}"
    ax5.text(0.1, y_pos, strike_text, fontsize=9, fontweight="bold", va="center")
    cost_text = (
        f"Cost: USD {cost:,.0f} | At {worst_drop_pct:.0f}% drop: USD {net_value_worst:,.0f} | "
        f"ITM Prob: {prob_itm:.1f}%"
    )
    ax5.text(0.1, y_pos - 0.06, cost_text, fontsize=8, va="center")

    y_pos -= spacing

summary_text = f"Positions | Total Cost: USD {scenario_hedged['premium_paid']:,.0f} | "
summary_text += f"Hedge Benefit at {worst_drop_pct:.0f}% Drop: USD {hedge_benefit:,.0f} | Annual Vol: {annual_vol * 100:.1f}%"
ax5.text(
    0.5,
    0.95,
    "PUT POSITIONS",
    fontsize=14,
    fontweight="bold",
    ha="center",
    transform=ax5.transAxes,
)
ax5.text(0.5, 0.88, summary_text, fontsize=10, ha="center", transform=ax5.transAxes)

ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)

plt.suptitle(
    f"{TICKER} V5 Hedge Analysis: Statistically Improved (Drift={DRIFT_RATE*100:.0f}%, IV Weight={IV_WEIGHT*100:.0f}%)",
    fontsize=16,
    fontweight="bold",
    y=0.998,
)
try:
    plt.tight_layout()
except:
    pass

# =================== EXPORT TO PDF ===================

pdf_filename = f"{TICKER}_hedge_analysis_v5_{today.strftime('%Y%m%d')}.pdf"

print(f"\n📄 Generating PDF report: {pdf_filename}")

with PdfPages(pdf_filename) as pdf:
    # Save the main dashboard
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # =================== PAGE 2: TECHNICAL INDICATORS ===================

    print("📊 Generating technical indicators page...")

    fig_tech = plt.figure(figsize=(20, 12))
    gs_tech = fig_tech.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

    # Calculate technical indicators
    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # MACD
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    rsi = calculate_rsi(close)
    macd, signal_line, histogram = calculate_macd(close)

    # Plot 1: RSI
    ax_rsi = fig_tech.add_subplot(gs_tech[0, 0])
    ax_rsi.plot(df.index, rsi, linewidth=2, color="#2E86AB", label="RSI(14)")
    ax_rsi.axhline(70, color="red", linestyle="--", alpha=0.5, label="Overbought (70)")
    ax_rsi.axhline(30, color="green", linestyle="--", alpha=0.5, label="Oversold (30)")
    ax_rsi.axhline(50, color="gray", linestyle=":", alpha=0.3)
    ax_rsi.fill_between(df.index, 30, 70, alpha=0.1, color="gray")
    ax_rsi.set_title("Relative Strength Index (RSI)", fontsize=12, fontweight="bold")
    ax_rsi.set_ylabel("RSI", fontsize=10)
    ax_rsi.legend(loc="best", fontsize=8)
    ax_rsi.grid(True, alpha=0.3)
    ax_rsi.set_ylim(0, 100)

    # Plot 2: MACD
    ax_macd = fig_tech.add_subplot(gs_tech[0, 1])
    ax_macd.plot(df.index, macd, linewidth=2, color="#2E86AB", label="MACD")
    ax_macd.plot(df.index, signal_line, linewidth=2, color="#FF6F00", label="Signal")
    ax_macd.bar(df.index, histogram, alpha=0.3, color="gray", label="Histogram")
    ax_macd.axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=1)
    ax_macd.set_title("MACD (12, 26, 9)", fontsize=12, fontweight="bold")
    ax_macd.set_ylabel("MACD Value", fontsize=10)
    ax_macd.legend(loc="best", fontsize=8)
    ax_macd.grid(True, alpha=0.3)

    # Plot 3: Rolling Volatility
    ax_vol = fig_tech.add_subplot(gs_tech[1, 0])
    rolling_vol_30 = returns.rolling(30).std() * np.sqrt(252) * 100
    rolling_vol_60 = returns.rolling(60).std() * np.sqrt(252) * 100
    # Use returns.index to match the data length
    ax_vol.plot(returns.index, rolling_vol_30, linewidth=2, color="#D62828", label="30-day Vol", alpha=0.8)
    ax_vol.plot(returns.index, rolling_vol_60, linewidth=2, color="#2E86AB", label="60-day Vol", alpha=0.8)
    ax_vol.axhline(annual_vol * 100, color="green", linestyle="--", alpha=0.7,
                   label=f"Current: {annual_vol*100:.1f}%")
    ax_vol.fill_between(returns.index, rolling_vol_30, rolling_vol_60, alpha=0.1, color="gray")
    ax_vol.set_title("Rolling Annualized Volatility", fontsize=12, fontweight="bold")
    ax_vol.set_ylabel("Volatility (%)", fontsize=10)
    ax_vol.legend(loc="best", fontsize=8)
    ax_vol.grid(True, alpha=0.3)

    # Plot 4: Volume Analysis
    ax_volume = fig_tech.add_subplot(gs_tech[1, 1])
    if isinstance(df.columns, pd.MultiIndex):
        volume = df.xs("Volume", axis=1, level=0).iloc[:, 0]
    else:
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)

    volume_ma = volume.rolling(20).mean()
    colors = ['green' if close.iloc[i] >= close.iloc[i-1] else 'red'
              for i in range(len(close))]
    ax_volume.bar(df.index, volume, color=colors, alpha=0.5, width=0.8)
    ax_volume.plot(df.index, volume_ma, linewidth=2, color="blue", label="20-day MA")
    ax_volume.set_title("Volume Analysis", fontsize=12, fontweight="bold")
    ax_volume.set_ylabel("Volume", fontsize=10)
    ax_volume.legend(loc="best", fontsize=8)
    ax_volume.grid(True, alpha=0.3, axis='y')

    # Plot 5: Drawdown Chart
    ax_dd = fig_tech.add_subplot(gs_tech[2, :])
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    # Use cumulative.index to match the data
    ax_dd.fill_between(cumulative.index, drawdown, 0, alpha=0.3, color="red", label="Drawdown")
    ax_dd.plot(cumulative.index, drawdown, linewidth=1.5, color="darkred")
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax_dd.axhline(max_dd, color="red", linestyle="--", alpha=0.7,
                  label=f"Max DD: {max_dd:.1f}% on {max_dd_date.strftime('%Y-%m-%d')}")
    ax_dd.set_title("Historical Drawdown Analysis", fontsize=12, fontweight="bold")
    ax_dd.set_ylabel("Drawdown (%)", fontsize=10)
    ax_dd.set_xlabel("Date", fontsize=10)
    ax_dd.legend(loc="best", fontsize=9)
    ax_dd.grid(True, alpha=0.3)

    # Plot 6: Price Returns Distribution
    ax_dist = fig_tech.add_subplot(gs_tech[3, 0])
    returns_pct = returns * 100
    ax_dist.hist(returns_pct.dropna(), bins=50, alpha=0.7, color="#2E86AB", edgecolor="black")
    ax_dist.axvline(returns_pct.mean(), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {returns_pct.mean():.2f}%")
    ax_dist.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax_dist.set_title("Daily Returns Distribution", fontsize=12, fontweight="bold")
    ax_dist.set_xlabel("Daily Return (%)", fontsize=10)
    ax_dist.set_ylabel("Frequency", fontsize=10)
    ax_dist.legend(loc="best", fontsize=8)
    ax_dist.grid(True, alpha=0.3, axis='y')

    # Plot 7: ATR (Average True Range)
    ax_atr = fig_tech.add_subplot(gs_tech[3, 1])
    if isinstance(df.columns, pd.MultiIndex):
        high = df.xs("High", axis=1, level=0).iloc[:, 0]
        low = df.xs("Low", axis=1, level=0).iloc[:, 0]
    else:
        high = df["High"] if "High" in df.columns else close
        low = df["Low"] if "Low" in df.columns else close

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    atr_pct = (atr / close) * 100

    ax_atr.plot(df.index, atr_pct, linewidth=2, color="#8B5CF6", label="ATR (14)")
    ax_atr.fill_between(df.index, 0, atr_pct, alpha=0.3, color="#8B5CF6")
    ax_atr.set_title("Average True Range (% of Price)", fontsize=12, fontweight="bold")
    ax_atr.set_ylabel("ATR (%)", fontsize=10)
    ax_atr.set_xlabel("Date", fontsize=10)
    ax_atr.legend(loc="best", fontsize=8)
    ax_atr.grid(True, alpha=0.3)

    fig_tech.suptitle(f"{TICKER} - Technical Indicators Dashboard",
                      fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    pdf.savefig(fig_tech, bbox_inches="tight")
    plt.close(fig_tech)

    # =================== PAGE 3: OPTIONS GREEKS & ANALYTICS ===================

    if recommendations:
        print("📊 Generating Greeks & options analytics page...")

        fig_greeks = plt.figure(figsize=(20, 12))
        gs_greeks = fig_greeks.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

        # Calculate approximate Greeks for each recommendation
        def black_scholes_greeks(S, K, T, r, sigma, option_type='put'):
            """Approximate Black-Scholes Greeks"""
            from scipy.stats import norm

            if T <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type == 'put':
                delta = norm.cdf(d1) - 1
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                        - r * K * np.exp(-r * T) * norm.cdf(-d2))
            else:
                delta = norm.cdf(d1)
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                        + r * K * np.exp(-r * T) * norm.cdf(d2))

            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change

            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta / 365,  # Daily theta
                'vega': vega
            }

        greeks_data = []
        for rec in recommendations:
            T = rec['dte'] / 365.0
            greeks = black_scholes_greeks(
                current_price, rec['strike'], T,
                DRIFT_RATE, annual_vol, 'put'
            )
            greeks_data.append({
                'strike': rec['strike'],
                'dte': rec['dte'],
                'contracts': rec['contracts'],
                'delta': greeks['delta'] * rec['contracts'] * 100,
                'gamma': greeks['gamma'] * rec['contracts'] * 100,
                'theta': greeks['theta'] * rec['contracts'] * 100,
                'vega': greeks['vega'] * rec['contracts'] * 100,
            })

        greeks_df = pd.DataFrame(greeks_data)

        # Plot 1: Delta by Position
        ax_delta = fig_greeks.add_subplot(gs_greeks[0, 0])
        strikes = greeks_df['strike'].astype(str)
        ax_delta.bar(range(len(strikes)), greeks_df['delta'], color='#2E86AB', alpha=0.7)
        ax_delta.set_xticks(range(len(strikes)))
        ax_delta.set_xticklabels([f"${s}" for s in greeks_df['strike']], rotation=45)
        ax_delta.set_title("Portfolio Delta by Strike", fontsize=12, fontweight="bold")
        ax_delta.set_ylabel("Delta (per $1 move)", fontsize=10)
        ax_delta.axhline(0, color='black', linewidth=0.5)
        ax_delta.grid(True, alpha=0.3, axis='y')
        total_delta = greeks_df['delta'].sum()
        ax_delta.text(0.02, 0.98, f"Total Delta: {total_delta:.1f}",
                     transform=ax_delta.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 2: Theta Decay
        ax_theta = fig_greeks.add_subplot(gs_greeks[0, 1])
        ax_theta.bar(range(len(strikes)), greeks_df['theta'], color='#D62828', alpha=0.7)
        ax_theta.set_xticks(range(len(strikes)))
        ax_theta.set_xticklabels([f"${s}" for s in greeks_df['strike']], rotation=45)
        ax_theta.set_title("Daily Theta Decay by Strike", fontsize=12, fontweight="bold")
        ax_theta.set_ylabel("Theta (daily decay)", fontsize=10)
        ax_theta.axhline(0, color='black', linewidth=0.5)
        ax_theta.grid(True, alpha=0.3, axis='y')
        total_theta = greeks_df['theta'].sum()
        ax_theta.text(0.02, 0.98, f"Total Daily Decay: ${total_theta:.2f}",
                     transform=ax_theta.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 3: Vega Exposure
        ax_vega = fig_greeks.add_subplot(gs_greeks[1, 0])
        ax_vega.bar(range(len(strikes)), greeks_df['vega'], color='#8B5CF6', alpha=0.7)
        ax_vega.set_xticks(range(len(strikes)))
        ax_vega.set_xticklabels([f"${s}" for s in greeks_df['strike']], rotation=45)
        ax_vega.set_title("Vega (Volatility Sensitivity) by Strike", fontsize=12, fontweight="bold")
        ax_vega.set_ylabel("Vega (per 1% vol change)", fontsize=10)
        ax_vega.axhline(0, color='black', linewidth=0.5)
        ax_vega.grid(True, alpha=0.3, axis='y')
        total_vega = greeks_df['vega'].sum()
        ax_vega.text(0.02, 0.98, f"Total Vega: {total_vega:.1f}",
                    transform=ax_vega.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 4: Gamma Exposure
        ax_gamma = fig_greeks.add_subplot(gs_greeks[1, 1])
        ax_gamma.bar(range(len(strikes)), greeks_df['gamma'], color='#06A77D', alpha=0.7)
        ax_gamma.set_xticks(range(len(strikes)))
        ax_gamma.set_xticklabels([f"${s}" for s in greeks_df['strike']], rotation=45)
        ax_gamma.set_title("Gamma (Delta Sensitivity) by Strike", fontsize=12, fontweight="bold")
        ax_gamma.set_ylabel("Gamma", fontsize=10)
        ax_gamma.axhline(0, color='black', linewidth=0.5)
        ax_gamma.grid(True, alpha=0.3, axis='y')
        total_gamma = greeks_df['gamma'].sum()
        ax_gamma.text(0.02, 0.98, f"Total Gamma: {total_gamma:.4f}",
                     transform=ax_gamma.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 5: Strike Selection Heatmap (Probability × Protection)
        ax_heatmap = fig_greeks.add_subplot(gs_greeks[2, :])

        if options_df is not None and not options_df.empty:
            # Create heatmap data
            strikes_range = np.linspace(current_price * 0.7, current_price * 0.95, 20)
            dte_range = [7, 14, 30, 60, 90, 180, 365]

            heatmap_data = np.zeros((len(dte_range), len(strikes_range)))

            for i, dte in enumerate(dte_range):
                for j, strike in enumerate(strikes_range):
                    prob = calculate_downside_probability(
                        strike, current_price, dte, daily_vol, annual_vol, DRIFT_RATE
                    )
                    protection = max(strike - current_price * 0.8, 0)  # Protection at 20% drop
                    score = prob * protection  # Expected protection
                    heatmap_data[i, j] = score

            im = ax_heatmap.imshow(heatmap_data, cmap='RdYlGn', aspect='auto',
                                   interpolation='bilinear')
            ax_heatmap.set_xticks(np.arange(0, len(strikes_range), 2))
            ax_heatmap.set_xticklabels([f"${s:.1f}" for s in strikes_range[::2]], rotation=45)
            ax_heatmap.set_yticks(range(len(dte_range)))
            ax_heatmap.set_yticklabels([f"{d}d" for d in dte_range])
            ax_heatmap.set_title("Strike Selection Heatmap: Expected Protection Value",
                                fontsize=12, fontweight="bold")
            ax_heatmap.set_xlabel("Strike Price", fontsize=10)
            ax_heatmap.set_ylabel("Days to Expiration", fontsize=10)

            # Add colorbar
            cbar = fig_greeks.colorbar(im, ax=ax_heatmap)
            cbar.set_label('Expected Protection Score', rotation=270, labelpad=20)

            # Mark recommended positions
            for rec in recommendations:
                # Find closest strike and dte in the heatmap
                strike_idx = np.argmin(np.abs(strikes_range - rec['strike']))
                dte_idx = np.argmin(np.abs(np.array(dte_range) - rec['dte']))
                ax_heatmap.scatter(strike_idx, dte_idx, s=200, marker='*',
                                  color='red', edgecolors='black', linewidths=2,
                                  zorder=10)

        fig_greeks.suptitle(f"{TICKER} - Options Greeks & Risk Analytics",
                           fontsize=16, fontweight="bold", y=0.995)
        plt.tight_layout()
        pdf.savefig(fig_greeks, bbox_inches="tight")
        plt.close(fig_greeks)

    # =================== PAGE 4: SCENARIO ANALYSIS MATRIX ===================

    print("📊 Generating scenario analysis matrix...")

    fig_scenario = plt.figure(figsize=(20, 12))
    gs_scenario = fig_scenario.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Scenario Analysis Table with visualization
    ax_scenarios = fig_scenario.add_subplot(gs_scenario[0, :])
    ax_scenarios.axis('off')

    scenario_table_data = []
    for scenario in scenario_results:
        scenario_table_data.append([
            f"{scenario['drop_pct']:.0f}%",
            f"${scenario['scenario_price']:.2f}",
            f"${scenario['unhedged_total']:,.0f}",
            f"${scenario['hedged_total']:,.0f}",
            f"${scenario['benefit']:,.0f}",
            f"{scenario['protection_pct']:.1f}%"
        ])

    table = ax_scenarios.table(
        cellText=scenario_table_data,
        colLabels=['Drop', 'Price', 'Unhedged P/L', 'Hedged P/L', 'Benefit', 'Protection %'],
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.3, 0.8, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style table
    for i in range(6):
        table[(0, i)].set_facecolor('#404040')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

    for i in range(1, len(scenario_table_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f9f9f9')
            table[(i, j)].set_text_props(fontsize=10, family='monospace')

    ax_scenarios.set_title("Comprehensive Scenario Analysis",
                          fontsize=14, fontweight='bold', pad=20)

    # Plot 2: Hedge Effectiveness by Scenario
    ax_eff = fig_scenario.add_subplot(gs_scenario[1, 0])
    drops = [s['drop_pct'] for s in scenario_results]
    protections = [s['protection_pct'] for s in scenario_results]
    ax_eff.bar(range(len(drops)), protections, color='#06A77D', alpha=0.7)
    ax_eff.set_xticks(range(len(drops)))
    ax_eff.set_xticklabels([f"{d:.0f}%" for d in drops])
    ax_eff.set_title("Hedge Effectiveness by Drop Scenario", fontsize=12, fontweight='bold')
    ax_eff.set_xlabel("Price Drop", fontsize=10)
    ax_eff.set_ylabel("Protection %", fontsize=10)
    ax_eff.grid(True, alpha=0.3, axis='y')

    # Plot 3: Absolute Benefit by Scenario
    ax_benefit = fig_scenario.add_subplot(gs_scenario[1, 1])
    benefits = [s['benefit'] for s in scenario_results]
    colors_benefit = ['#06A77D' if b > 0 else '#D62828' for b in benefits]
    ax_benefit.bar(range(len(drops)), benefits, color=colors_benefit, alpha=0.7)
    ax_benefit.set_xticks(range(len(drops)))
    ax_benefit.set_xticklabels([f"{d:.0f}%" for d in drops])
    ax_benefit.set_title("Absolute Hedge Benefit ($) by Scenario", fontsize=12, fontweight='bold')
    ax_benefit.set_xlabel("Price Drop", fontsize=10)
    ax_benefit.set_ylabel("Hedge Benefit ($)", fontsize=10)
    ax_benefit.axhline(0, color='black', linewidth=0.5)
    ax_benefit.grid(True, alpha=0.3, axis='y')

    fig_scenario.suptitle(f"{TICKER} - Detailed Scenario Analysis",
                         fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    pdf.savefig(fig_scenario, bbox_inches='tight')
    plt.close(fig_scenario)

    print(f"✅ PDF created successfully: {pdf_filename}")
print("📊 Visualization complete with statistically improved calculations.")
print("📄 Review 'put_hedge_analysis_review.md' for detailed methodology.")
