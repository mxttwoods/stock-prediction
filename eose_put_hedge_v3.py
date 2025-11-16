#!/usr/bin/env python3

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import stats
from scipy.stats import norm

# =================== CONFIGURATION ===================

TICKER = "EOSE"
SHARES_HELD = 400
INSURANCE_BUDGET = 100  # Target budget for hedge
MAX_DAYS_TO_EXPIRATION = 180  # Look at options expiring within this many days
DROP_SCENARIO_PCT = 0.50
LOOKBACK_DAYS = 365
DATA_FILE = Path("eose_daily.parquet")

# Bollinger Band parameters
BB_WINDOW = 20
BB_STD = 2.0

# Forward projection parameters
FORWARD_PROJECTION_DAYS = 30

# =================== DATA LOADING ===================


def load_price_history(
    ticker: str, lookback_days: int, cache_file: Path
) -> pd.DataFrame:
    """Load daily price data with CSV caching."""
    today = pd.Timestamp.today().normalize()
    start_date = today - pd.Timedelta(days=lookback_days + 5)

    df = None

    if cache_file.exists():
        try:
            # df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # df.index = pd.to_datetime(df.index)
            df = pd.read_parquet(cache_file)
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        except Exception as e:
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
                # df.to_csv(cache_file)
                df.to_parquet(cache_file)

    cutoff = today - pd.Timedelta(days=lookback_days)
    df = df[df.index >= cutoff]
    return df


def fetch_live_options_chain(ticker: str, max_dte: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch live options chain data from Yahoo Finance.
    Returns DataFrame with all put options within max_dte days.
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
                    puts["lastPrice"] = puts["lastPrice"].fillna(0).astype(float)
                    puts["bid"] = puts["bid"].fillna(0).astype(float)
                    puts["ask"] = puts["ask"].fillna(0).astype(float)
                    puts["volume"] = puts["volume"].fillna(0).astype(int)
                    puts["openInterest"] = puts["openInterest"].fillna(0).astype(int)

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

# Calculate volatility
returns = close.pct_change().dropna()
daily_vol = returns.std()
annual_vol = daily_vol * np.sqrt(252)

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


def project_bollinger_bands_forward(current_price, sma, std, days_ahead, daily_vol):
    """Project Bollinger Bands forward using current values and volatility."""
    recent_trend = (close.iloc[-1] - close.iloc[-20]) / 20 if len(close) >= 20 else 0
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq="B"
    )

    projected_sma = []
    projected_bb_up = []
    projected_bb_dn = []

    current_sma_val = sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else current_price
    current_std_val = (
        std.iloc[-1] if not pd.isna(std.iloc[-1]) else daily_vol * current_price
    )

    for i in range(days_ahead):
        current_sma_val += recent_trend
        time_factor = 1 + (i / days_ahead) * 0.1
        current_std_val = current_std_val * time_factor

        projected_sma.append(current_sma_val)
        projected_bb_up.append(current_sma_val + BB_STD * current_std_val)
        projected_bb_dn.append(current_sma_val - BB_STD * current_std_val)

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
expected_low = proj_bb_dn[-1]
expected_high = proj_bb_up[-1]

# =================== PROBABILITY CALCULATIONS ===================


def calculate_price_probability(
    target_price: float,
    current_price: float,
    days: int,
    daily_vol: float,
    annual_vol: float,
) -> float:
    """Calculate probability of reaching target price within given days."""
    if days <= 0:
        return 0.0

    t = days / 252.0
    mu = 0  # No drift assumption
    sigma = annual_vol

    log_ratio = np.log(target_price / current_price)
    mean_log = (mu - 0.5 * sigma**2) * t
    var_log = sigma**2 * t

    if var_log <= 0:
        return 0.5 if abs(target_price - current_price) < 0.01 else 0.0

    z_score = (log_ratio - mean_log) / np.sqrt(var_log)
    prob = 1 - stats.norm.cdf(z_score)

    return prob


# =================== OPTIONS ANALYSIS & RECOMMENDATIONS ===================


def analyze_put_option(
    put_row: pd.Series,
    current_price: float,
    shares_held: int,
    daily_vol: float,
    annual_vol: float,
) -> Dict:
    """
    Analyze a single put option for hedge effectiveness.
    Returns metrics including cost efficiency, expected value, etc.
    """
    strike = float(put_row["strike"])
    premium = (
        float(put_row["lastPrice"])
        if not pd.isna(put_row["lastPrice"])
        else (float(put_row["bid"]) + float(put_row["ask"])) / 2
    )
    dte = int(put_row["DTE"])
    exp_date = put_row["exp_date"]

    # Calculate metrics
    prob_itm = calculate_price_probability(
        strike, current_price, dte, daily_vol, annual_vol
    )

    # Protection scenarios
    scenarios = [0.5, 0.6, 0.7, 0.8, 0.9]
    scenario_prices = [current_price * s for s in scenarios]

    total_protection_value = 0.0
    for scenario_price in scenario_prices:
        intrinsic = max(strike - scenario_price, 0.0)
        total_protection_value += intrinsic

    avg_protection = total_protection_value / len(scenarios)

    # Cost per contract
    cost_per_contract = premium * 100

    # Protection per dollar (efficiency metric)
    protection_per_dollar = avg_protection / premium if premium > 0 else 0

    # Expected value (probability weighted)
    # Simplified: prob_itm * avg_intrinsic - premium
    expected_value = prob_itm * avg_protection - premium

    # Cost efficiency score (higher is better)
    efficiency_score = protection_per_dollar * prob_itm * 100

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
    }


def recommend_optimal_hedge(
    options_df: pd.DataFrame,
    current_price: float,
    shares_held: int,
    budget: float,
    daily_vol: float,
    annual_vol: float,
    expected_low: float,
) -> List[Dict]:
    """
    Recommend optimal put positions based on:
    - Expected move window (Bollinger Bands)
    - Probability analysis
    - Cost efficiency
    - Budget constraints
    """
    if options_df is None or options_df.empty:
        return []

    # Analyze all puts
    analyzed_puts = []
    for idx, row in options_df.iterrows():
        try:
            analysis = analyze_put_option(
                row, current_price, shares_held, daily_vol, annual_vol
            )
            analyzed_puts.append(analysis)
        except Exception:
            continue

    if not analyzed_puts:
        return []

    analyzed_df = pd.DataFrame(analyzed_puts)

    # Filter criteria:
    # 1. Strikes below expected low (likely to provide protection)
    # 2. Reasonable liquidity (volume or open interest)
    # 3. Within budget constraints

    # Focus on strikes that would be in-the-money if price drops to expected low
    protective_puts = analyzed_df[
        (analyzed_df["strike"] >= expected_low * 0.9)  # Not too far OTM
        & (analyzed_df["strike"] <= current_price * 0.95)  # Below current price
        & (analyzed_df["premium"] > 0)
        & (analyzed_df["efficiency_score"] > 0)
    ].copy()

    if protective_puts.empty:
        # Fallback: use all puts sorted by efficiency
        protective_puts = analyzed_df[
            (analyzed_df["strike"] <= current_price) & (analyzed_df["premium"] > 0)
        ].copy()

    # Sort by efficiency score
    protective_puts = protective_puts.sort_values("efficiency_score", ascending=False)

    # Recommend positions within budget
    recommendations = []
    total_cost = 0.0

    # Strategy: Recommend a ladder of puts at different strikes/expirations
    # This provides protection across time and price levels

    # Group by expiration to get variety
    for exp_date in protective_puts["exp_date"].unique()[:4]:  # Top 4 expirations
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


# =================== GENERATE RECOMMENDATIONS ===================

if options_df is not None and not options_df.empty:
    recommendations = recommend_optimal_hedge(
        options_df,
        current_price,
        SHARES_HELD,
        INSURANCE_BUDGET,
        daily_vol,
        annual_vol,
        expected_low,
    )
else:
    recommendations = []

# =================== PORTFOLIO P/L CALCULATIONS ===================


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


# =================== PRINT ANALYSIS ===================

# Calculate scenario analysis
scenario_price_50_down = 0.5 * current_price
scenario_hedged = portfolio_pl_at_price(scenario_price_50_down, recommendations)
scenario_unhedged = portfolio_pl_at_price(scenario_price_50_down, [])

hedge_benefit = scenario_hedged["total"] - scenario_unhedged["total"]
protection_pct = (
    (hedge_benefit / abs(scenario_unhedged["total"])) * 100
    if scenario_unhedged["total"] < 0
    else 0
)


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

# Forward projections
ax1.plot(
    future_dates,
    proj_sma,
    linestyle="--",
    alpha=0.5,
    color="#A23B72",
    linewidth=1.5,
    label="Projected SMA20",
)
ax1.fill_between(
    future_dates,
    proj_bb_dn,
    proj_bb_up,
    alpha=0.15,
    color="orange",
    label="Projected BB Range",
)
ax1.plot(
    future_dates, proj_bb_up, linestyle=":", alpha=0.6, color="orange", linewidth=1
)
ax1.plot(
    future_dates, proj_bb_dn, linestyle=":", alpha=0.6, color="orange", linewidth=1
)

# Calculate percentiles for forward projections
prob_expected_low = (
    calculate_price_probability(
        expected_low, current_price, FORWARD_PROJECTION_DAYS, daily_vol, annual_vol
    )
    * 100
)
prob_expected_high = (
    100
    - calculate_price_probability(
        expected_high, current_price, FORWARD_PROJECTION_DAYS, daily_vol, annual_vol
    )
    * 100
)

# Calculate and plot percentile lines within the orange expected move area
# Using normal distribution to calculate price levels at different percentiles

t = FORWARD_PROJECTION_DAYS / 252.0
sigma = annual_vol
percentiles = [10, 25, 50, 75, 90]
percentile_prices = []

# Create color gradient from red (low/loss) to green (high/profit)
percentile_colors = [
    "#D32F2F",
    "#FF6F00",
    "#FFC107",
    "#8BC34A",
    "#4CAF50",
]  # Red to Green gradient

for i, percentile in enumerate(percentiles):
    # Convert percentile to z-score
    z_score = norm.ppf(percentile / 100.0)
    # Calculate price at this percentile
    # Using log-normal distribution: log(S_t/S_0) ~ N((mu - 0.5*sigma^2)*t, sigma^2*t)
    log_return = (0 - 0.5 * sigma**2) * t + z_score * sigma * np.sqrt(t)
    price_at_percentile = current_price * np.exp(log_return)
    percentile_prices.append((percentile, price_at_percentile))

    # Plot percentile line across the forward projection period with color gradient
    ax1.plot(
        future_dates,
        [price_at_percentile] * len(future_dates),
        linestyle=":",
        linewidth=1.5,
        color=percentile_colors[i],
        alpha=0.7,
        label=f"{percentile}th %ile: ${price_at_percentile:.2f}",
    )

# Mark expected move window with percentiles
ax1.axhline(
    expected_low,
    linestyle="--",
    linewidth=2,
    color="red",
    alpha=0.7,
    label=f"Expected Low: ${expected_low:.2f} ({100 - prob_expected_low:.0f}th %ile)",
)
ax1.axhline(
    expected_high,
    linestyle="--",
    linewidth=2,
    color="green",
    alpha=0.7,
    label=f"Expected High: ${expected_high:.2f} ({prob_expected_high:.0f}th %ile)",
)

# Overlay recommended/current put strikes
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
    f"{TICKER} Price History with Forward Projections & Put Strikes",
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
    label=f"Hedged",
    linewidth=2.5,
    color="#06A77D",
    linestyle="-",
)

ax2.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
ax2.axvline(current_price, color="green", linewidth=1, linestyle=":", alpha=0.5)
ax2.axvline(
    expected_low,
    color="red",
    linewidth=1.5,
    linestyle="--",
    alpha=0.7,
    label="Expected Low",
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
    # Plot available strikes vs efficiency
    strikes_available = options_df["strike"].values
    premiums_available = (
        options_df["lastPrice"]
        .fillna((options_df["bid"] + options_df["ask"]) / 2)
        .values
    )

    # Calculate efficiency for all strikes
    efficiencies = []
    for idx, row in options_df.iterrows():
        try:
            analysis = analyze_put_option(
                row, current_price, SHARES_HELD, daily_vol, annual_vol
            )
            efficiencies.append(analysis["efficiency_score"])
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

    # Mark recommended strikes
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

    # Mark expected low
    ax3.axvline(
        expected_low,
        linestyle="--",
        linewidth=2,
        color="red",
        alpha=0.7,
        label="Expected Low",
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
    ax3.set_ylabel("Efficiency Score", fontsize=10)
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
probabilities = [
    calculate_price_probability(p, current_price, max_dte, daily_vol, annual_vol) * 100
    for p in price_range
]

ax4.plot(
    price_range,
    probabilities,
    linewidth=2.5,
    color="#8B5CF6",
    label=f"Probability (next {max_dte} days)",
)
ax4.fill_between(price_range, 0, probabilities, alpha=0.3, color="#8B5CF6")

ax4.axvline(current_price, color="green", linewidth=1.5, linestyle="-", alpha=0.7)
ax4.axvline(
    expected_low,
    color="red",
    linewidth=2,
    linestyle="--",
    alpha=0.7,
    label="Expected Low",
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
        calculate_price_probability(strike, current_price, dte, daily_vol, annual_vol)
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
ax4.set_ylabel("Probability (%)", fontsize=10)
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

    intrinsic_50 = max(strike - scenario_price_50_down, 0) * 100 * contracts
    net_value_50 = intrinsic_50 - cost
    prob_itm = (
        calculate_price_probability(strike, current_price, dte, daily_vol, annual_vol)
        * 100
    )

    x_start = 0.05
    x_width = 0.9
    y_start = y_pos - bar_height / 2
    color = "#90EE90" if strike > scenario_price_50_down else "#FFB6C1"

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
    cost_text = f"Cost: USD {cost:,.0f} | At 50% drop: USD {net_value_50:,.0f} | ITM Prob: {prob_itm:.1f}%"
    ax5.text(0.1, y_pos - 0.06, cost_text, fontsize=8, va="center")

    y_pos -= spacing

summary_text = f"Positions | Total Cost: USD {scenario_hedged['premium_paid']:,.0f} | "
summary_text += f"Hedge Benefit at 50% Drop: USD {hedge_benefit:,.0f} | Annual Vol: {annual_vol * 100:.1f}%"
ax5.text(
    0.5,
    0.95,
    f"PUT POSITIONS",
    fontsize=14,
    fontweight="bold",
    ha="center",
    transform=ax5.transAxes,
)
ax5.text(0.5, 0.88, summary_text, fontsize=10, ha="center", transform=ax5.transAxes)

ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)

plt.suptitle(
    f"{TICKER} AI-Powered Hedge Analysis: Expected Moves vs Available Options",
    fontsize=16,
    fontweight="bold",
    y=0.998,
)
try:
    plt.tight_layout()
except:
    pass

# =================== EXPORT TO PDF ===================

pdf_filename = f"{TICKER}_hedge_analysis_{today.strftime('%Y%m%d')}.pdf"

with PdfPages(pdf_filename) as pdf:
    # Page 1: Executive Summary - APA Style (Black & White)
    fig_summary = plt.figure(figsize=(11, 8.5))
    ax_summary = fig_summary.add_subplot(111)
    ax_summary.axis("off")

    # Title (APA style)
    ax_summary.text(
        0.5,
        0.96,
        f"Protective Put Hedge Analysis: {TICKER}",
        ha="center",
        fontsize=16,
        fontweight="bold",
        transform=fig_summary.transFigure,
        color="black",
    )
    ax_summary.text(
        0.5,
        0.92,
        f"Analysis Date: {today.strftime('%B %d, %Y')}",
        ha="center",
        fontsize=11,
        transform=fig_summary.transFigure,
        color="black",
    )

    # Draw separator line
    line = plt.Line2D(
        [0.1, 0.9],
        [0.88, 0.88],
        transform=fig_summary.transFigure,
        color="black",
        linewidth=0.5,
    )
    ax_summary.add_line(line)

    # Portfolio Overview Section
    y_pos = 0.82
    ax_summary.text(
        0.12,
        y_pos,
        "Portfolio Overview",
        fontsize=12,
        fontweight="bold",
        transform=fig_summary.transFigure,
        color="black",
    )

    # Metrics in two columns
    y_pos -= 0.06
    left_col = [
        ("Current Price", f"\\${current_price:.2f}"),
        ("Shares Held", f"{SHARES_HELD:,}"),
        ("Budget Target", f"\\${INSURANCE_BUDGET:,.0f}"),
    ]
    right_col = [
        ("Expected Low", f"\\${expected_low:.2f}"),
        ("Expected High", f"\\${expected_high:.2f}"),
        ("Annual Volatility", f"{annual_vol * 100:.1f}%"),
    ]

    for i, (label, value) in enumerate(left_col):
        ax_summary.text(
            0.12,
            y_pos,
            f"{label}:",
            fontsize=10,
            transform=fig_summary.transFigure,
            color="black",
        )
        ax_summary.text(
            0.35,
            y_pos,
            value,
            fontsize=10,
            fontweight="bold",
            transform=fig_summary.transFigure,
            color="black",
        )
        y_pos -= 0.045

    y_pos = 0.82 - 0.06
    for i, (label, value) in enumerate(right_col):
        ax_summary.text(
            0.55,
            y_pos,
            f"{label}:",
            fontsize=10,
            transform=fig_summary.transFigure,
            color="black",
        )
        ax_summary.text(
            0.78,
            y_pos,
            value,
            fontsize=10,
            fontweight="bold",
            transform=fig_summary.transFigure,
            color="black",
        )
        y_pos -= 0.045

    # Scenario Analysis Section
    y_pos -= 0.06
    ax_summary.text(
        0.12,
        y_pos,
        "50% Price Decline Scenario Analysis",
        fontsize=12,
        fontweight="bold",
        transform=fig_summary.transFigure,
        color="black",
    )

    y_pos -= 0.05
    ax_summary.text(
        0.12,
        y_pos,
        "Unhedged Portfolio P/L:",
        fontsize=10,
        transform=fig_summary.transFigure,
        color="black",
    )
    ax_summary.text(
        0.50,
        y_pos,
        f"\\${scenario_unhedged['total']:>12,.0f}",
        fontsize=11,
        fontweight="bold",
        transform=fig_summary.transFigure,
        color="black",
        family="monospace",
    )

    y_pos -= 0.04
    ax_summary.text(
        0.12,
        y_pos,
        "Hedged Portfolio P/L:",
        fontsize=10,
        transform=fig_summary.transFigure,
        color="black",
    )
    ax_summary.text(
        0.50,
        y_pos,
        f"\\${scenario_hedged['total']:>12,.0f}",
        fontsize=11,
        fontweight="bold",
        transform=fig_summary.transFigure,
        color="black",
        family="monospace",
    )

    y_pos -= 0.04
    ax_summary.text(
        0.12,
        y_pos,
        "Hedge Benefit:",
        fontsize=10,
        transform=fig_summary.transFigure,
        color="black",
    )
    ax_summary.text(
        0.50,
        y_pos,
        f"\\${hedge_benefit:>12,.0f} ({protection_pct:.1f}% protection)",
        fontsize=11,
        fontweight="bold",
        transform=fig_summary.transFigure,
        color="black",
        family="monospace",
    )

    pdf.savefig(fig_summary, bbox_inches="tight", facecolor="white")
    plt.close(fig_summary)

    # Page 2: Recommended Hedge Positions
    if recommendations:
        fig_rec = plt.figure(figsize=(11, 8.5))
        ax_rec = fig_rec.add_subplot(111)
        ax_rec.axis("off")

        # Title
        ax_rec.text(
            0.5,
            0.93,
            "Recommended Hedge Positions",
            ha="center",
            fontsize=16,
            fontweight="bold",
            transform=fig_rec.transFigure,
            color="black",
        )
        ax_rec.text(
            0.5,
            0.89,
            f"Optimal Put Options Strategy for {TICKER}",
            ha="center",
            fontsize=12,
            transform=fig_rec.transFigure,
            color="#555555",
        )

        # Separator line
        line = plt.Line2D(
            [0.1, 0.9],
            [0.86, 0.86],
            transform=fig_rec.transFigure,
            color="black",
            linewidth=0.5,
        )
        ax_rec.add_line(line)

        # Create recommendations table
        rec_table_data = []
        for rec in recommendations:
            rec_table_data.append(
                [
                    f"\\${rec['strike']:.2f}",
                    f"\\${rec['premium']:.2f}",
                    f"{rec['contracts']}",
                    f"{rec['dte']}d",
                    rec["expiration"],
                    f"\\${rec['cost']:,.0f}",
                    f"{rec['prob_itm'] * 100:.1f}%",
                    f"{rec['efficiency_score']:.1f}",
                ]
            )

        # Add total row
        total_cost = sum(r["cost"] for r in recommendations)
        rec_table_data.append(
            ["TOTAL", "-", "-", "-", "-", f"\\${total_cost:,.0f}", "-", "-"]
        )

        rec_table = ax_rec.table(
            cellText=rec_table_data,
            colLabels=[
                "Strike",
                "Premium",
                "Contracts",
                "DTE",
                "Expiration",
                "Cost",
                "ITM Prob",
                "Efficiency",
            ],
            cellLoc="center",
            loc="center",
            bbox=[0.10, 0.35, 0.80, 0.45],
        )
        rec_table.auto_set_font_size(False)
        rec_table.set_fontsize(9)
        rec_table.scale(1, 2.5)

        # Style recommendation table headers
        for i in range(8):
            cell = rec_table[(0, i)]
            cell.set_facecolor("#404040")
            cell.set_text_props(weight="bold", color="white", fontsize=9)
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)

        # Style recommendation table data rows
        for i in range(1, len(rec_table_data) + 1):
            for j in range(8):
                cell = rec_table[(i, j)]
                cell.set_edgecolor("#cccccc")
                cell.set_linewidth(0.8)
                if i == len(rec_table_data):  # Total row
                    cell.set_facecolor("#e8e8e8")
                    cell.set_text_props(weight="bold", color="black", fontsize=10)
                    cell.set_edgecolor("black")
                    cell.set_linewidth(1.5)
                else:
                    if i % 2 == 0:
                        cell.set_facecolor("#f9f9f9")
                    else:
                        cell.set_facecolor("white")
                    cell.set_text_props(color="black", fontsize=9, family="monospace")

        # Summary box
        summary_box = FancyBboxPatch(
            (0.1, 0.15),
            0.80,
            0.15,
            boxstyle="round,pad=0.01",
            edgecolor="#404040",
            facecolor="#f0f8ff",
            linewidth=1.5,
            transform=fig_rec.transFigure,
        )
        ax_rec.add_patch(summary_box)

        ax_rec.text(
            0.5,
            0.27,
            "Strategy Summary",
            fontsize=12,
            fontweight="bold",
            ha="center",
            transform=fig_rec.transFigure,
            color="black",
        )

        # Summary metrics
        avg_dte = sum(r["dte"] for r in recommendations) / len(recommendations)
        avg_efficiency = sum(r["efficiency_score"] for r in recommendations) / len(
            recommendations
        )
        total_contracts = sum(r["contracts"] for r in recommendations)

        summary_text = f"""
Total Investment: \\${total_cost:,.0f}
Total Contracts: {total_contracts}
Average DTE: {avg_dte:.0f} days
Average Efficiency Score: {avg_efficiency:.1f}
Portfolio Coverage: {(total_contracts * 100 / SHARES_HELD) * 100:.0f}%
        """

        ax_rec.text(
            0.5,
            0.15,
            summary_text,
            fontsize=10,
            ha="center",
            transform=fig_rec.transFigure,
            color="black",
            family="monospace",
        )

        ax_rec.text(
            0.5,
            0.08,
            "Note: These recommendations are based on current market conditions and volatility analysis.\\n"
            "Options strategies involve risk and may not be suitable for all investors.",
            fontsize=8,
            ha="center",
            transform=fig_rec.transFigure,
            color="#666666",
            style="italic",
        )

        pdf.savefig(fig_rec, bbox_inches="tight", facecolor="white")
        plt.close(fig_rec)

    # Page 3: Volatility & Price Statistics
    fig_vol = plt.figure(figsize=(11, 8.5))
    ax_vol = fig_vol.add_subplot(111)
    ax_vol.axis("off")

    # Title
    ax_vol.text(
        0.5,
        0.90,
        "Volatility & Price Statistics",
        ha="center",
        fontsize=16,
        fontweight="bold",
        transform=fig_vol.transFigure,
        color="black",
    )
    ax_vol.text(
        0.5,
        0.86,
        "Historical Price Analysis & Risk Metrics",
        ha="center",
        fontsize=12,
        transform=fig_vol.transFigure,
        color="#555555",
    )

    # Separator line
    line = plt.Line2D(
        [0.15, 0.85],
        [0.83, 0.83],
        transform=fig_vol.transFigure,
        color="black",
        linewidth=0.5,
    )
    ax_vol.add_line(line)

    vol_table_data = [
        ["Annual Volatility", f"{annual_vol * 100:.2f}%"],
        ["Daily Volatility", f"{daily_vol * 100:.2f}%"],
        ["Current Price", f"\\${current_price:.2f}"],
        [
            "52-Week High",
            f"\\${close.max():.2f}" if len(close) >= 252 else f"\\${close.max():.2f}",
        ],
        [
            "52-Week Low",
            f"\\${close.min():.2f}" if len(close) >= 252 else f"\\${close.min():.2f}",
        ],
        ["Price Range", f"\\${close.min():.2f} - \\${close.max():.2f}"],
    ]

    vol_table = ax_vol.table(
        cellText=vol_table_data,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        bbox=[0.1, 0.64, 0.35, 0.18],
    )
    vol_table.auto_set_font_size(False)
    vol_table.set_fontsize(9)
    # vol_table.scale(1, 2)
    # auto scale the table
    vol_table.auto_set_column_width(col=list(range(len(vol_table_data))))

    # Style vol table
    for i in range(2):
        vol_table[(0, i)].set_facecolor("#404040")
        vol_table[(0, i)].set_text_props(weight="bold", color="white")
        vol_table[(0, i)].set_edgecolor("black")
        vol_table[(0, i)].set_linewidth(1.5)

    for i in range(1, len(vol_table_data) + 1):
        for j in range(2):
            cell = vol_table[(i, j)]
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.8)
            if i % 2 == 0:
                cell.set_facecolor("#f9f9f9")
            else:
                cell.set_facecolor("white")
            if j == 1:  # Value column
                cell.set_text_props(weight="bold", family="monospace")

    # Page 4: Portfolio Risk Assessment
    fig_risk = plt.figure(figsize=(11, 8.5))
    ax_risk = fig_risk.add_subplot(111)
    ax_risk.axis("off")

    # Title
    ax_risk.text(
        0.5,
        0.90,
        "Portfolio Risk Assessment",
        ha="center",
        fontsize=16,
        fontweight="bold",
        transform=fig_risk.transFigure,
        color="black",
    )
    ax_risk.text(
        0.5,
        0.86,
        "Value at Risk & Position Analysis",
        ha="center",
        fontsize=12,
        transform=fig_risk.transFigure,
        color="#555555",
    )

    # Separator line
    line = plt.Line2D(
        [0.15, 0.85],
        [0.83, 0.83],
        transform=fig_risk.transFigure,
        color="black",
        linewidth=0.5,
    )
    ax_risk.add_line(line)

    portfolio_value = current_price * SHARES_HELD
    daily_var = portfolio_value * daily_vol  # 1-day VaR at 1 std dev

    risk_table_data = [
        ["Portfolio Value", f"\\${portfolio_value:,.0f}"],
        ["Position Size", f"{SHARES_HELD} shares"],
        ["Daily Value at Risk (1σ)", f"\\${daily_var:,.0f}"],
        ["Expected Loss (50% drop)", f"\\${abs(scenario_unhedged['total']):,.0f}"],
        ["Hedge Budget", f"\\${INSURANCE_BUDGET:,.0f}"],
        [
            "Budget as % of Portfolio",
            f"{(INSURANCE_BUDGET / portfolio_value) * 100:.2f}%",
        ],
    ]

    risk_table = ax_risk.table(
        cellText=risk_table_data,
        colLabels=["Risk Metric", "Value"],
        cellLoc="left",
        loc="center",
        bbox=[0.20, 0.42, 0.60, 0.35],
    )
    risk_table.auto_set_font_size(False)
    risk_table.set_fontsize(11)
    risk_table.scale(1, 2.5)

    # Style risk table
    for i in range(2):
        risk_table[(0, i)].set_facecolor("#404040")
        risk_table[(0, i)].set_text_props(weight="bold", color="white")
        risk_table[(0, i)].set_edgecolor("black")
        risk_table[(0, i)].set_linewidth(1.5)

    for i in range(1, len(risk_table_data) + 1):
        for j in range(2):
            cell = risk_table[(i, j)]
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.8)
            if i % 2 == 0:
                cell.set_facecolor("#f9f9f9")
            else:
                cell.set_facecolor("white")
            if j == 1:
                cell.set_text_props(weight="bold", family="monospace")

    pdf.savefig(fig_risk, bbox_inches="tight", facecolor="white")
    plt.close(fig_risk)

    # Page 5: Downside Scenario Analysis
    fig_scenario = plt.figure(figsize=(11, 8.5))
    ax_scenario = fig_scenario.add_subplot(111)
    ax_scenario.axis("off")

    # Title
    ax_scenario.text(
        0.5,
        0.90,
        "Downside Scenario Analysis",
        ha="center",
        fontsize=16,
        fontweight="bold",
        transform=fig_scenario.transFigure,
        color="black",
    )
    ax_scenario.text(
        0.5,
        0.86,
        "Hedge Performance Across Multiple Price Declines",
        ha="center",
        fontsize=12,
        transform=fig_scenario.transFigure,
        color="#555555",
    )

    # Separator line
    line = plt.Line2D(
        [0.15, 0.85],
        [0.83, 0.83],
        transform=fig_scenario.transFigure,
        color="black",
        linewidth=0.5,
    )
    ax_scenario.add_line(line)

    # Calculate multiple scenarios
    scenarios = [
        ("10% Decline", 0.90),
        ("25% Decline", 0.75),
        ("50% Decline", 0.50),
        ("75% Decline", 0.25),
    ]

    scenario_table_data = []
    for scenario_name, price_mult in scenarios:
        scenario_price = current_price * price_mult
        unhedged_pl = portfolio_pl_at_price(scenario_price, [])
        hedged_pl = portfolio_pl_at_price(scenario_price, recommendations)
        benefit = hedged_pl["total"] - unhedged_pl["total"]
        protection = (
            (benefit / abs(unhedged_pl["total"])) * 100
            if unhedged_pl["total"] < 0
            else 0
        )

        scenario_table_data.append(
            [
                scenario_name,
                f"\\${scenario_price:.2f}",
                f"\\${unhedged_pl['total']:,.0f}",
                f"\\${hedged_pl['total']:,.0f}",
                f"\\${benefit:,.0f}",
                f"{protection:.1f}%",
            ]
        )

    scenario_table = ax_scenario.table(
        cellText=scenario_table_data,
        colLabels=[
            "Scenario",
            "Price",
            "Unhedged P/L",
            "Hedged P/L",
            "Benefit",
            "Protection",
        ],
        cellLoc="center",
        loc="center",
        bbox=[0.10, 0.42, 0.80, 0.32],
    )
    scenario_table.auto_set_font_size(False)
    scenario_table.set_fontsize(10)
    scenario_table.scale(1, 2.8)

    # Style scenario table
    for i in range(6):
        scenario_table[(0, i)].set_facecolor("#404040")
        scenario_table[(0, i)].set_text_props(weight="bold", color="white", fontsize=8)
        scenario_table[(0, i)].set_edgecolor("black")
        scenario_table[(0, i)].set_linewidth(1.5)

    for i in range(1, len(scenario_table_data) + 1):
        for j in range(6):
            cell = scenario_table[(i, j)]
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.8)
            if i % 2 == 0:
                cell.set_facecolor("#f9f9f9")
            else:
                cell.set_facecolor("white")
            cell.set_text_props(fontsize=8.5, family="monospace")

    pdf.savefig(fig_scenario, bbox_inches="tight", facecolor="white")
    plt.close(fig_scenario)

    # Page 6: Price Chart with Forward Projections
    fig_price = plt.figure(figsize=(11, 8.5))
    ax_price = fig_price.add_subplot(111)

    ax_price.plot(df.index, close, label="Adj Close", linewidth=2, color="#2E86AB")
    ax_price.plot(
        df.index, df["SMA20"], label="SMA20", linestyle="--", alpha=0.7, color="#A23B72"
    )
    ax_price.fill_between(
        df.index,
        df["BB_dn"],
        df["BB_up"],
        alpha=0.2,
        color="gray",
        label="Historical BB",
    )

    ax_price.plot(
        future_dates,
        proj_sma,
        linestyle="--",
        alpha=0.5,
        color="#A23B72",
        linewidth=1.5,
        label="Projected SMA20",
    )
    ax_price.fill_between(
        future_dates,
        proj_bb_dn,
        proj_bb_up,
        alpha=0.15,
        color="orange",
        label="Projected BB Range",
    )
    ax_price.plot(
        future_dates, proj_bb_up, linestyle=":", alpha=0.6, color="orange", linewidth=1
    )
    ax_price.plot(
        future_dates, proj_bb_dn, linestyle=":", alpha=0.6, color="orange", linewidth=1
    )

    # Add percentile lines to the orange expected move area with color gradient
    for i, (percentile, price_at_percentile) in enumerate(percentile_prices):
        ax_price.plot(
            future_dates,
            [price_at_percentile] * len(future_dates),
            linestyle=":",
            linewidth=1.5,
            color=percentile_colors[i],
            alpha=0.7,
            label=f"{percentile}th %ile: ${price_at_percentile:.2f}",
        )

    ax_price.axhline(
        expected_low,
        linestyle="--",
        linewidth=2,
        color="red",
        alpha=0.7,
        label=f"Expected Low: ${expected_low:.2f}",
    )
    ax_price.axhline(
        expected_high,
        linestyle="--",
        linewidth=2,
        color="green",
        alpha=0.7,
        label=f"Expected High: ${expected_high:.2f}",
    )

    strike_colors = ["#F18F01", "#C73E1D", "#6A994E", "#8B5CF6", "#06A77D"]
    for i, pos in enumerate(recommendations):
        strike = pos["strike"]
        exp_date = pos.get("exp_date", today + pd.Timedelta(days=pos.get("dte", 0)))
        label = f"{pos.get('dte', 'N/A')}d Put @ ${strike:.1f}"
        ax_price.axhline(
            strike,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            color=strike_colors[i % len(strike_colors)],
            label=label,
        )
        if exp_date <= future_dates[-1]:
            ax_price.axvline(
                exp_date,
                linestyle=":",
                alpha=0.4,
                color=strike_colors[i % len(strike_colors)],
                linewidth=1,
            )

    ax_price.axhline(
        current_price,
        color="green",
        linestyle="-",
        linewidth=1.5,
        alpha=0.5,
        label=f"Current: {current_price:.2f}",
    )

    ax_price.set_title(
        f"Figure 1: {TICKER} Price History with Forward Projections & Put Strikes",
        fontsize=12,
        fontweight="bold",
    )
    # add auto ticket increments for price y axis
    ax_price.yaxis.set_major_locator(ticker.AutoLocator())
    ax_price.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_price.set_ylabel("Price ($)", fontsize=11)
    ax_price.set_xlabel("Date", fontsize=11)
    ax_price.xaxis.set_major_locator(ticker.AutoLocator())
    ax_price.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_price.legend(loc="best", fontsize=8, ncol=2)
    ax_price.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig_price, bbox_inches="tight")
    plt.close(fig_price)

    # Page 4: P/L Comparison
    fig_pl = plt.figure(figsize=(11, 8.5))
    ax_pl = fig_pl.add_subplot(111)

    prices = np.linspace(current_price * 0.1, current_price * 1.6, 500)
    pl_stock_only = [portfolio_pl_at_price(p, [])["total"] for p in prices]
    pl_with_hedge = [portfolio_pl_at_price(p, recommendations)["total"] for p in prices]

    ax_pl.plot(
        prices,
        pl_stock_only,
        label="Unhedged",
        linewidth=2.5,
        color="#D62828",
        linestyle="--",
        alpha=0.8,
    )
    ax_pl.plot(
        prices,
        pl_with_hedge,
        label=f"Hedged",
        linewidth=2.5,
        color="#06A77D",
        linestyle="-",
    )

    ax_pl.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
    ax_pl.axvline(
        current_price,
        color="green",
        linewidth=1,
        linestyle=":",
        alpha=0.5,
        label="Current Price",
    )
    ax_pl.axvline(
        expected_low,
        color="red",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        label="Expected Low",
    )

    ax_pl.set_title(
        "Figure 2: P/L Comparison: Hedged vs Unhedged", fontsize=12, fontweight="bold"
    )
    ax_pl.set_xlabel(f"{TICKER} Price ($)", fontsize=12)
    ax_pl.set_ylabel("Net P/L ($)", fontsize=12)
    ax_pl.xaxis.set_major_locator(ticker.AutoLocator())
    ax_pl.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_pl.yaxis.set_major_locator(ticker.AutoLocator())
    ax_pl.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_pl.legend(loc="best", fontsize=10)
    ax_pl.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig_pl, bbox_inches="tight")
    plt.close(fig_pl)

    # Page 5: Options Efficiency Analysis
    if options_df is not None and not options_df.empty:
        fig_eff = plt.figure(figsize=(11, 8.5))
        ax_eff = fig_eff.add_subplot(111)

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
                    row, current_price, SHARES_HELD, daily_vol, annual_vol
                )
                efficiencies.append(analysis["efficiency_score"])
            except:
                efficiencies.append(0)

        scatter = ax_eff.scatter(
            strikes_available,
            efficiencies,
            alpha=0.6,
            s=50,
            c=premiums_available,
            cmap="viridis",
            label="Available Puts",
        )
        fig_eff.colorbar(scatter, ax=ax_eff, label="Premium ($)")

        if recommendations:
            rec_strikes = [r["strike"] for r in recommendations]
            rec_effs = [r["efficiency_score"] for r in recommendations]
            ax_eff.scatter(
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

        ax_eff.axvline(
            expected_low,
            linestyle="--",
            linewidth=2,
            color="red",
            alpha=0.7,
            label="Expected Low",
        )
        ax_eff.axvline(
            current_price,
            linestyle=":",
            linewidth=1,
            color="green",
            alpha=0.5,
            label="Current Price",
        )

        ax_eff.set_title(
            "Figure 4: Put Options Efficiency Analysis", fontsize=12, fontweight="bold"
        )
        ax_eff.set_xlabel("Strike Price ($)", fontsize=11)
        ax_eff.set_ylabel("Efficiency Score", fontsize=11)
        ax_eff.xaxis.set_major_locator(ticker.AutoLocator())
        ax_eff.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_eff.yaxis.set_major_locator(ticker.AutoLocator())
        ax_eff.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_eff.legend(loc="best", fontsize=9)
        ax_eff.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig_eff, bbox_inches="tight")
        plt.close(fig_eff)

    # Page 6: Probability Distribution
    fig_prob = plt.figure(figsize=(11, 8.5))
    ax_prob = fig_prob.add_subplot(111)

    price_range = np.linspace(current_price * 0.5, current_price * 1.5, 200)
    max_dte = (
        max([p.get("dte", 30) for p in recommendations]) if recommendations else 30
    )
    probabilities = [
        calculate_price_probability(p, current_price, max_dte, daily_vol, annual_vol)
        * 100
        for p in price_range
    ]

    ax_prob.plot(
        price_range,
        probabilities,
        linewidth=2.5,
        color="#8B5CF6",
        label=f"Probability (next {max_dte} days)",
    )
    ax_prob.fill_between(price_range, 0, probabilities, alpha=0.3, color="#8B5CF6")

    ax_prob.axvline(
        current_price, color="green", linewidth=1.5, linestyle="-", alpha=0.7
    )
    ax_prob.axvline(
        expected_low,
        color="red",
        linewidth=2,
        linestyle="--",
        alpha=0.7,
        label="Expected Low",
    )
    ax_prob.axvspan(
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
            calculate_price_probability(
                strike, current_price, dte, daily_vol, annual_vol
            )
            * 100
        )
        ax_prob.axvline(strike, linestyle="--", alpha=0.6, linewidth=1)
        ax_prob.text(
            strike,
            prob + 2,
            f"${strike:.1f}\n({prob:.1f}%)",
            ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax_prob.set_title(
        "Figure 5: Probability Distribution & Put Strike Analysis",
        fontsize=12,
        fontweight="bold",
    )
    ax_prob.set_xlabel(f"{TICKER} Price ($)", fontsize=11)
    ax_prob.set_ylabel("Probability (%)", fontsize=11)
    ax_prob.xaxis.set_major_locator(ticker.AutoLocator())
    ax_prob.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_prob.yaxis.set_major_locator(ticker.AutoLocator())
    ax_prob.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_prob.legend(loc="best", fontsize=9)
    ax_prob.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig_prob, bbox_inches="tight")
    plt.close(fig_prob)

    # Page 7: Position Summary - Detailed Breakdown
    fig_pos = plt.figure(figsize=(11, 8.5))
    ax_pos = fig_pos.add_subplot(111)
    ax_pos.axis("off")

    # Header
    ax_pos.text(
        0.5,
        0.96,
        f"Hedge Positions: Detailed Analysis",
        fontsize=16,
        fontweight="bold",
        ha="center",
        transform=fig_pos.transFigure,
        color="black",
    )
    ax_pos.text(
        0.5,
        0.92,
        "Position-by-Position Breakdown & Performance Metrics",
        fontsize=11,
        ha="center",
        transform=fig_pos.transFigure,
        color="#555555",
    )

    # Separator line
    line = plt.Line2D(
        [0.05, 0.95],
        [0.89, 0.89],
        transform=fig_pos.transFigure,
        color="black",
        linewidth=0.5,
    )
    ax_pos.add_line(line)

    # Prepare table data - split into two tables
    table1_data = []  # Position details
    table2_data = []  # Financial metrics
    headers1 = ["Pos", "Strike", "Expiration", "DTE", "Contracts"]
    headers2 = ["Pos", "Premium", "Cost", "ITM%", "Value@50%", "ROI%"]

    for i, pos in enumerate(recommendations):
        strike = pos["strike"]
        premium = pos.get("premium", 0)
        contracts = pos.get("contracts", 0)
        dte = pos.get("dte", 0)
        exp_date = pos.get("exp_date", today + pd.Timedelta(days=dte))
        cost = premium * 100 * contracts

        intrinsic_50 = max(strike - scenario_price_50_down, 0) * 100 * contracts
        net_value_50 = intrinsic_50 - cost
        prob_itm = (
            calculate_price_probability(
                strike, current_price, dte, daily_vol, annual_vol
            )
            * 100
        )
        protection_ratio = f"{(net_value_50 / cost) * 100:.1f}" if cost > 0 else "N/A"

        table1_data.append(
            [
                f"P{i + 1}",
                f"\\${strike:.2f}",
                exp_date.strftime("%m/%d/%y"),
                f"{dte}d",
                f"{contracts}",
            ]
        )

        table2_data.append(
            [
                f"P{i + 1}",
                f"\\${premium:.2f}",
                f"\\${cost:,.0f}",
                f"{prob_itm:.1f}",
                f"\\${net_value_50:,.0f}",
                protection_ratio,
            ]
        )

    # Add total rows
    total_cost = sum(
        p.get("premium", 0) * 100 * p.get("contracts", 0) for p in recommendations
    )
    total_value_50 = (
        sum(
            max(p.get("strike", 0) - scenario_price_50_down, 0)
            * 100
            * p.get("contracts", 0)
            for p in recommendations
        )
        - total_cost
    )
    total_ratio = (
        f"{(total_value_50 / total_cost) * 100:.1f}" if total_cost > 0 else "N/A"
    )
    total_contracts = sum(p.get("contracts", 0) for p in recommendations)

    table1_data.append(["TOT", "-", "-", "-", f"{total_contracts}"])

    table2_data.append(
        [
            "TOT",
            "-",
            f"\\${total_cost:,.0f}",
            "-",
            f"\\${total_value_50:,.0f}",
            total_ratio,
        ]
    )

    # Add table section labels
    ax_pos.text(
        0.5,
        0.87,
        "Position Details",
        fontsize=12,
        fontweight="bold",
        ha="center",
        transform=fig_pos.transFigure,
        color="#404040",
    )

    # Create Table 1: Position Details
    table1 = ax_pos.table(
        cellText=table1_data,
        colLabels=headers1,
        cellLoc="center",
        loc="center",
        bbox=[0.10, 0.70, 0.80, 0.22],
    )

    # Style table1
    table1.auto_set_font_size(False)
    table1.set_fontsize(8)
    # table1.scale(1, 2.2)
    # auto scale the table
    table1.auto_set_column_width(col=list(range(len(headers1))))

    # Style header row table1
    for i in range(len(headers1)):
        cell = table1[(0, i)]
        cell.set_facecolor("#404040")
        cell.set_text_props(weight="bold", color="white", fontsize=9)
        cell.set_edgecolor("black")
        cell.set_linewidth(1.5)

    # Style data rows table1
    for i in range(1, len(table1_data) + 1):
        for j in range(len(headers1)):
            cell = table1[(i, j)]
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.8)
            if i == len(table1_data):  # Total row
                cell.set_facecolor("#e8e8e8")
                cell.set_text_props(weight="bold", color="black", fontsize=9)
                cell.set_edgecolor("black")
                cell.set_linewidth(1.5)
            else:
                if i % 2 == 0:
                    cell.set_facecolor("#f9f9f9")
                else:
                    cell.set_facecolor("white")
                cell.set_text_props(color="black", fontsize=8)

    # Add second table label
    ax_pos.text(
        0.5,
        0.58,
        "Financial Metrics",
        fontsize=12,
        fontweight="bold",
        ha="center",
        transform=fig_pos.transFigure,
        color="#404040",
    )

    # Create Table 2: Financial Metrics
    table2 = ax_pos.table(
        cellText=table2_data,
        colLabels=headers2,
        cellLoc="center",
        loc="center",
        bbox=[0.10, 0.33, 0.80, 0.22],
    )

    # Style table2
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    # auto scale the table
    table2.auto_set_column_width(col=list(range(len(headers2))))

    # Style header row table2
    for i in range(len(headers2)):
        cell = table2[(0, i)]
        cell.set_facecolor("#404040")
        cell.set_text_props(weight="bold", color="white", fontsize=9)
        cell.set_edgecolor("black")
        cell.set_linewidth(1.5)

    # Style data rows table2
    for i in range(1, len(table2_data) + 1):
        for j in range(len(headers2)):
            cell = table2[(i, j)]
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.8)
            if i == len(table2_data):  # Total row
                cell.set_facecolor("#e8e8e8")
                cell.set_text_props(weight="bold", color="black", fontsize=9)
                cell.set_edgecolor("black")
                cell.set_linewidth(1.5)
            else:
                if i % 2 == 0:
                    cell.set_facecolor("#f9f9f9")
                else:
                    cell.set_facecolor("white")
                cell.set_text_props(color="black", fontsize=8, family="monospace")

    # Summary section below tables
    # Draw section box
    summary_box = FancyBboxPatch(
        (0.10, 0.12),
        0.80,
        0.15,
        boxstyle="round,pad=0.01",
        edgecolor="#404040",
        facecolor="#f5f5f5",
        linewidth=1.5,
        transform=fig_pos.transFigure,
    )
    ax_pos.add_patch(summary_box)

    ax_pos.text(
        0.5,
        0.25,
        "Portfolio Summary Statistics",
        fontsize=12,
        fontweight="bold",
        ha="center",
        transform=fig_pos.transFigure,
        color="black",
    )

    # Create summary metrics in columns
    summary_y = 0.20
    col1_x = 0.15
    col2_x = 0.40
    col3_x = 0.68

    ax_pos.text(
        col1_x,
        summary_y,
        "Total Premium:",
        fontsize=9,
        ha="left",
        transform=fig_pos.transFigure,
        color="#333333",
    )
    ax_pos.text(
        col1_x,
        summary_y - 0.03,
        f"\\${scenario_hedged['premium_paid']:,.0f}",
        fontsize=11,
        fontweight="bold",
        ha="left",
        transform=fig_pos.transFigure,
        color="black",
        family="monospace",
    )

    ax_pos.text(
        col2_x,
        summary_y,
        "Hedge Benefit (50% drop):",
        fontsize=9,
        ha="left",
        transform=fig_pos.transFigure,
        color="#333333",
    )
    ax_pos.text(
        col2_x,
        summary_y - 0.03,
        f"\\${hedge_benefit:,.0f}",
        fontsize=11,
        fontweight="bold",
        ha="left",
        transform=fig_pos.transFigure,
        color="#006600",
        family="monospace",
    )

    ax_pos.text(
        col3_x,
        summary_y,
        "Protection Level:",
        fontsize=9,
        ha="left",
        transform=fig_pos.transFigure,
        color="#333333",
    )
    ax_pos.text(
        col3_x,
        summary_y - 0.03,
        f"{protection_pct:.1f}%",
        fontsize=11,
        fontweight="bold",
        ha="left",
        transform=fig_pos.transFigure,
        color="#006600",
        family="monospace",
    )

    ax_pos.text(
        col1_x,
        0.14,
        f"Annual Volatility: {annual_vol * 100:.1f}% | "
        f"Analysis based on {LOOKBACK_DAYS}-day historical data",
        fontsize=8,
        ha="left",
        transform=fig_pos.transFigure,
        color="#666666",
        style="italic",
    )

    pdf.savefig(fig_pos, bbox_inches="tight", facecolor="white")
    plt.close(fig_pos)


# Optionally show interactive plots (comment out if you only want PDF)
# plt.show()
print("PDF created successfully")
exit()