#!/usr/bin/env python3
"""
EOSE Call Strategy Analyzer v1 - AI-Powered Options Recommendation Engine
======================================================================

This script uses live options data to recommend optimal call option positions
to enhance returns, based on:
- Forward-projected Bollinger Bands (expected move window)
- Probability analysis (likelihood of different price scenarios)
- Return on investment metrics (potential return per dollar spent)
- Expected value calculations
- Real-time options chain data from Yahoo Finance

KEY FEATURES:
1. Fetches LIVE options chain data
2. Analyzes expected moves vs available strikes
3. Recommends optimal call positions
4. Calculates cost-benefit ratios
5. Shows correlation between expected moves and available options
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# =================== CONFIGURATION ===================

TICKER = "EOSE"
SHARES_HELD = 400
# Allocate 3-8% of holdings to long call options
CALL_INVESTMENT_PERCENTAGE = 0.05  # 5%
MAX_DAYS_TO_EXPIRATION = 365  # Look at options expiring within this many days

# Optional: Your current positions (will be compared against recommendations)
CURRENT_POSITIONS = []

LOOKBACK_DAYS = 365
DATA_FILE = Path("eose_daily.parquet")

# Bollinger Band parameters
BB_WINDOW = 20
BB_STD = 2.0

# Forward projection parameters
FORWARD_PROJECTION_DAYS = 90

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
            df = pd.read_parquet(cache_file)
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        except Exception as e:
            print(f"⚠️  Cache read failed ({e}), redownloading...")
            try:
                cache_file.unlink()
            except FileNotFoundError:
                pass
            df = None

    if df is None or df.empty:
        print(f"📥 Downloading {ticker} price data...")
        df = yf.download(
            ticker,
            start=start_date,
            end=today + pd.Timedelta(days=1),
            auto_adjust=False,
            progress=False,
        )
        df.to_parquet(cache_file)
        print(f"✅ Downloaded {len(df)} days of data")
    else:
        last_date = df.index[-1].normalize()
        if last_date < today:
            print(f"🔄 Updating cache (last date: {last_date.date()})...")
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
                print(f"✅ Added {len(new)} new days")

    cutoff = today - pd.Timedelta(days=lookback_days)
    df = df[df.index >= cutoff]
    return df


def fetch_live_call_options_chain(
    ticker: str, max_dte: int = 30
) -> Optional[pd.DataFrame]:
    """
    Fetch live options chain data from Yahoo Finance.
    Returns DataFrame with all call options within max_dte days.
    """
    try:
        print(f"📡 Fetching live options chain for {ticker}...")
        ticker_obj = yf.Ticker(ticker)
        expirations = ticker_obj.options

        if not expirations:
            print("❌ No options data available for this ticker")
            return None

        print(f"   Found {len(expirations)} expiration dates")

        all_calls = []
        today = pd.Timestamp.today().normalize()

        for exp_date_str in expirations:
            try:
                exp_date = pd.to_datetime(exp_date_str)
                dte = (exp_date - today).days

                if 0 < dte <= max_dte:
                    opt_chain = ticker_obj.option_chain(exp_date_str)
                    calls = opt_chain.calls.copy()

                    # Add metadata
                    calls["expiration"] = exp_date_str
                    calls["exp_date"] = exp_date
                    calls["DTE"] = dte
                    calls["strike"] = calls["strike"].astype(float)
                    calls["lastPrice"] = calls["lastPrice"].fillna(0).astype(float)
                    calls["bid"] = calls["bid"].fillna(0).astype(float)
                    calls["ask"] = calls["ask"].fillna(0).astype(float)
                    calls["volume"] = calls["volume"].fillna(0).astype(int)
                    calls["openInterest"] = calls["openInterest"].fillna(0).astype(int)

                    all_calls.append(calls)

            except Exception as e:
                print(f"   ⚠️  Error fetching {exp_date_str}: {e}")
                continue

        if all_calls:
            combined = pd.concat(all_calls, ignore_index=True)
            print(f"✅ Fetched {len(combined)} call options within {max_dte} days")
            return combined
        else:
            print("❌ No call options found within specified date range")
            return None

    except Exception as e:
        print(f"❌ Error fetching options chain: {e}")
        return None


# =================== MAIN DATA LOAD ===================

print(f"\n{'=' * 80}")
print("🚀 EOSE CALL STRATEGY ANALYZER - AI-POWERED OPTIONS RECOMMENDATION ENGINE")
print(f"{'=' * 80}\n")

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

# Calculate the investment budget for calls
total_holding_value = current_price * SHARES_HELD
CALL_INVESTMENT_BUDGET = total_holding_value * CALL_INVESTMENT_PERCENTAGE

# Fetch live options
options_df = fetch_live_call_options_chain(TICKER, MAX_DAYS_TO_EXPIRATION)

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
    """
    Project Bollinger Bands forward using current values and volatility.

    Volatility scales with sqrt(time) according to standard financial theory.
    """
    recent_trend = (close.iloc[-1] - close.iloc[-20]) / 20 if len(close) >= 20 else 0
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq="B"
    )

    projected_sma = []
    projected_bb_up = []
    projected_bb_dn = []

    current_sma_val = sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else current_price
    base_std_val = (
        std.iloc[-1] if not pd.isna(std.iloc[-1]) else daily_vol * current_price
    )

    for i in range(days_ahead):
        current_sma_val = sma.iloc[-1] + recent_trend * i
        time_scaling = np.sqrt(1 + (i / BB_WINDOW))
        current_std_val = base_std_val * time_scaling

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

expected_low = proj_bb_dn[-1]
expected_high = proj_bb_up[-1]

# =================== PROBABILITY CALCULATIONS ===================


def calculate_call_itm_probability(
    target_price: float,
    current_price: float,
    days: int,
    annual_vol: float,
) -> float:
    """
    Calculate probability of price rising above target price within given days.
    """
    if days <= 0:
        return 0.0
    if current_price <= 0 or target_price <= 0:
        return 0.0
    if annual_vol <= 0:
        return 0.5 if abs(target_price - current_price) < 0.01 else 0.0

    t = days / 252.0
    mu = 0
    sigma = annual_vol

    log_ratio = np.log(target_price / current_price)
    mean_log = (mu - 0.5 * sigma**2) * t
    var_log = sigma**2 * t

    if var_log <= 0:
        return 0.5 if abs(target_price - current_price) < 0.01 else 0.0

    z_score = (log_ratio - mean_log) / np.sqrt(var_log)
    prob = 1 - stats.norm.cdf(z_score)

    return max(0.0, min(1.0, prob))


# =================== OPTIONS ANALYSIS & RECOMMENDATIONS ===================


def analyze_call_option(
    call_row: pd.Series,
    current_price: float,
    annual_vol: float,
) -> Dict:
    """
    Analyze a single call option for return enhancement.
    """
    strike = float(call_row["strike"])
    premium = (
        float(call_row["lastPrice"])
        if not pd.isna(call_row["lastPrice"])
        else (float(call_row["bid"]) + float(call_row["ask"])) / 2
    )
    dte = int(call_row["DTE"])
    exp_date = call_row["exp_date"]

    prob_itm = calculate_call_itm_probability(strike, current_price, dte, annual_vol)

    scenarios = [1.1, 1.2, 1.3, 1.4, 1.5]
    scenario_prices = [current_price * s for s in scenarios]

    total_potential_return = 0.0
    for scenario_price in scenario_prices:
        intrinsic = max(scenario_price - strike, 0.0)
        total_potential_return += intrinsic

    avg_potential_return = total_potential_return / len(scenarios)
    cost_per_contract = premium * 100
    return_per_dollar = avg_potential_return / premium if premium > 0 else 0
    expected_value = prob_itm * avg_potential_return - premium
    efficiency_score = return_per_dollar * prob_itm * 100
    distance_pct = ((strike - current_price) / current_price) * 100

    return {
        "strike": strike,
        "premium": premium,
        "dte": dte,
        "exp_date": exp_date,
        "expiration": call_row["expiration"],
        "prob_itm": prob_itm,
        "return_per_dollar": return_per_dollar,
        "expected_value": expected_value,
        "efficiency_score": efficiency_score,
        "distance_pct": distance_pct,
        "cost_per_contract": cost_per_contract,
        "volume": int(call_row.get("volume", 0)),
        "open_interest": int(call_row.get("openInterest", 0)),
    }


def recommend_optimal_calls(
    options_df: pd.DataFrame,
    current_price: float,
    budget: float,
    annual_vol: float,
    expected_high: float,
) -> List[Dict]:
    """
    Recommend optimal call positions.
    """
    if options_df is None or options_df.empty:
        return []

    print(f"\n{'=' * 80}")
    print(f"🔍 ANALYZING {len(options_df)} CALL OPTIONS...")
    print(f"{'=' * 80}")

    analyzed_calls = [
        analyze_call_option(row, current_price, annual_vol)
        for _, row in options_df.iterrows()
    ]

    if not analyzed_calls:
        return []

    analyzed_df = pd.DataFrame(analyzed_calls)

    potential_calls = analyzed_df[
        (analyzed_df["strike"] >= current_price)
        & (analyzed_df["strike"] <= expected_high * 1.1)
        & (analyzed_df["premium"] > 0)
        & (analyzed_df["efficiency_score"] > 0)
    ].copy()

    if potential_calls.empty:
        potential_calls = analyzed_df[
            (analyzed_df["strike"] >= current_price) & (analyzed_df["premium"] > 0)
        ].copy()

    potential_calls = potential_calls.sort_values("efficiency_score", ascending=False)

    recommendations = []
    total_cost = 0.0
    sorted_expirations = sorted(potential_calls["exp_date"].unique())[:4]

    for exp_date in sorted_expirations:
        exp_calls = potential_calls[potential_calls["exp_date"] == exp_date]
        best_call = exp_calls.iloc[0]
        cost_per_contract = best_call["cost_per_contract"]
        remaining_budget = budget - total_cost

        if cost_per_contract > 0 and remaining_budget >= cost_per_contract:
            contracts = int(remaining_budget / cost_per_contract)
            if contracts > 0:
                position_cost = cost_per_contract * contracts
                total_cost += position_cost
                recommendations.append(
                    {
                        "strike": best_call["strike"],
                        "premium": best_call["premium"],
                        "contracts": contracts,
                        "dte": best_call["dte"],
                        "exp_date": best_call["exp_date"],
                        "expiration": best_call["expiration"],
                        "cost": position_cost,
                        "efficiency_score": best_call["efficiency_score"],
                        "prob_itm": best_call["prob_itm"],
                        "return_per_dollar": best_call["return_per_dollar"],
                    }
                )

    return recommendations


# =================== GENERATE RECOMMENDATIONS ===================

if options_df is not None and not options_df.empty:
    recommendations = recommend_optimal_calls(
        options_df,
        current_price,
        CALL_INVESTMENT_BUDGET,
        annual_vol,
        expected_high,
    )
else:
    recommendations = []
    print("⚠️  No live options data, using current positions for analysis.")

# =================== PORTFOLIO P/L CALCULATIONS ===================


def portfolio_pl_at_price(
    price: float,
    stock_shares: int,
    positions: List[Dict],
) -> Dict:
    """Calculate portfolio P/L at a given price."""
    stock_pl = (price - current_price) * stock_shares
    opts_pl = 0.0
    prem_paid = 0.0

    for pos in positions:
        strike = pos["strike"]
        premium = pos.get("premium", 0)
        contracts = pos.get("contracts", 0)
        intrinsic = max(price - strike, 0.0) * 100 * contracts
        cost = premium * 100 * contracts
        prem_paid += cost
        opts_pl += intrinsic - cost

    return {
        "total": stock_pl + opts_pl,
        "stock": stock_pl,
        "options": opts_pl,
        "premium_paid": prem_paid,
    }


calls_to_analyze = recommendations if recommendations else CURRENT_POSITIONS
calls_label = "RECOMMENDED" if recommendations else "CURRENT"

# =================== PRINT ANALYSIS ===================

print(f"\n{'=' * 80}")
print(f"📊 {TICKER} CALL STRATEGY ANALYSIS - {calls_label} POSITIONS")
print(f"{'=' * 80}")
print(f"Current price:        ${current_price:.2f}")
print(f"Shares held:          {SHARES_HELD:,}")
print(f"Expected move range:  ${expected_low:.2f} - ${expected_high:.2f}")
print(f"Annual volatility:    {annual_vol * 100:.1f}%")
print(f"Budget target:        ${CALL_INVESTMENT_BUDGET:,.0f}")

if recommendations:
    total_rec_cost = sum(r["cost"] for r in recommendations)
    print("\n💰 RECOMMENDED CALL POSITIONS:")
    print(f"{'─' * 80}")
    print(
        f"{'Strike':<8} {'Premium':<10} {'Contracts':<10} {'DTE':<6} {'Cost':<12} {'Efficiency':<12} {'ITM Prob':<10}"
    )
    print(f"{'─' * 80}")
    for rec in recommendations:
        print(
            f"${rec['strike']:>6.2f}  ${rec['premium']:>8.2f}  {rec['contracts']:>9}  "
            f"{rec['dte']:>4}d  ${rec['cost']:>10,.0f}  {rec['efficiency_score']:>10.1f}  {rec['prob_itm'] * 100:>7.1f}%"
        )
    print(f"{'─' * 80}")
    print(f"{'TOTAL':<50} ${total_rec_cost:>10,.0f}")
    print(f"{'=' * 80}\n")

# Scenario analysis
scenario_price_50_up = 1.5 * current_price
scenario_with_calls = portfolio_pl_at_price(
    scenario_price_50_up, SHARES_HELD, calls_to_analyze
)
scenario_stock_only = portfolio_pl_at_price(scenario_price_50_up, SHARES_HELD, [])
return_enhancement = scenario_with_calls["total"] - scenario_stock_only["total"]
enhancement_pct = (
    (return_enhancement / abs(scenario_stock_only["total"])) * 100
    if scenario_stock_only["total"] != 0
    else 0
)

print(f"\n📈 50% RALLY SCENARIO (price → ${scenario_price_50_up:.2f}):")
print(f"{'─' * 80}")
print(f"  Stock-only P/L:     ${scenario_stock_only['total']:>12,.0f}")
print(f"  With Calls P/L:     ${scenario_with_calls['total']:>12,.0f}")
print(f"  └─ Stock gain:      ${scenario_with_calls['stock']:>12,.0f}")
print(f"  └─ Options gain:    ${scenario_with_calls['options']:>12,.0f}")
print(f"  └─ Premium paid:    ${scenario_with_calls['premium_paid']:>12,.0f}")
print(f"\n  💡 Return enhancement: ${return_enhancement:>12,.0f}")
print(f"  💡 Enhancement:      {enhancement_pct:.1f}% over stock-only profit")
print(f"{'=' * 80}\n")

# =================== VISUALIZATIONS ===================

pdf_filename = f"{TICKER}_call_strategy_analysis_{today.strftime('%Y%m%d')}.pdf"
print(f"\n📄 Generating PDF report: {pdf_filename}...")

with PdfPages(pdf_filename) as pdf:
    # Page 1: Main Chart
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3, height_ratios=[2.5, 2, 2, 1.5])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, close, label="Adj Close", linewidth=2, color="#2E86AB")
    ax1.plot(
        df.index, df["SMA20"], label="SMA20", linestyle="--", alpha=0.7, color="#A23B72"
    )
    ax1.fill_between(
        df.index,
        df["BB_dn"],
        df["BB_up"],
        alpha=0.2,
        color="gray",
        label="Historical BB",
    )
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
    ax1.axhline(
        expected_high,
        linestyle="--",
        linewidth=2,
        color="green",
        alpha=0.7,
        label=f"Expected High: ${expected_high:.2f}",
    )

    strike_colors = ["#F18F01", "#C73E1D", "#6A994E", "#8B5CF6", "#06A77D"]
    for i, pos in enumerate(calls_to_analyze):
        strike = pos["strike"]
        label = f"{pos.get('dte', 'N/A')}d Call @ ${strike:.1f}"
        ax1.axhline(
            strike,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            color=strike_colors[i % len(strike_colors)],
            label=label,
        )

    ax1.axhline(
        current_price,
        color="blue",
        linestyle="-",
        linewidth=1.5,
        alpha=0.5,
        label=f"Current: {current_price:.2f}",
    )
    ax1.set_title(
        f"{TICKER} Price History with Forward Projections & {calls_label} Call Strikes",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylabel("Price ($)", fontsize=11)
    ax1.legend(loc="best", fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    # P/L Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    prices = np.linspace(current_price * 0.8, current_price * 1.8, 500)
    pl_stock_only = [portfolio_pl_at_price(p, SHARES_HELD, [])["total"] for p in prices]
    pl_with_calls = [
        portfolio_pl_at_price(p, SHARES_HELD, calls_to_analyze)["total"] for p in prices
    ]
    ax2.plot(
        prices,
        pl_stock_only,
        label="Stock Only",
        linewidth=2.5,
        color="#D62828",
        linestyle="--",
        alpha=0.8,
    )
    ax2.plot(
        prices,
        pl_with_calls,
        label=f"With Calls ({calls_label})",
        linewidth=2.5,
        color="#06A77D",
    )
    ax2.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax2.axvline(current_price, color="blue", linestyle=":", alpha=0.5)
    ax2.set_title(
        "P/L Comparison: With Calls vs. Stock Only", fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel(f"{TICKER} Price ($)", fontsize=10)
    ax2.set_ylabel("Net P/L ($)", fontsize=10)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Options Efficiency
    ax3 = fig.add_subplot(gs[1, 1])
    if options_df is not None and not options_df.empty:
        analyzed_df = pd.DataFrame(
            [
                analyze_call_option(row, current_price, annual_vol)
                for _, row in options_df.iterrows()
            ]
        )
        scatter = ax3.scatter(
            analyzed_df["strike"],
            analyzed_df["efficiency_score"],
            alpha=0.6,
            s=50,
            c=analyzed_df["premium"],
            cmap="viridis",
            label="Available Calls",
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
        ax3.axvline(current_price, linestyle=":", color="blue", label="Current Price")
        ax3.set_title(
            "Call Options Efficiency Analysis", fontsize=12, fontweight="bold"
        )
        ax3.set_xlabel("Strike Price ($)", fontsize=10)
        ax3.set_ylabel("Efficiency Score", fontsize=10)
        ax3.legend(loc="best", fontsize=8)
        ax3.grid(True, alpha=0.3)

    # Probability Distribution
    ax4 = fig.add_subplot(gs[2, :])
    price_range = np.linspace(current_price * 0.8, current_price * 1.8, 200)
    max_dte = (
        max([p.get("dte", 30) for p in calls_to_analyze]) if calls_to_analyze else 30
    )
    probabilities = [
        calculate_call_itm_probability(p, current_price, max_dte, annual_vol) * 100
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
    ax4.axvline(current_price, color="blue", linestyle="-", alpha=0.7)
    ax4.axvspan(
        proj_bb_dn[-1],
        proj_bb_up[-1],
        alpha=0.2,
        color="orange",
        label="Projected BB Range",
    )
    for pos in calls_to_analyze:
        strike = pos["strike"]
        prob = (
            calculate_call_itm_probability(
                strike, current_price, pos.get("dte", max_dte), annual_vol
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
        "Probability Distribution & Call Strike Analysis",
        fontsize=12,
        fontweight="bold",
    )
    ax4.set_xlabel(f"{TICKER} Price ($)", fontsize=10)
    ax4.set_ylabel("Probability of Price > Strike (%)", fontsize=10)
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(
        f"{TICKER} AI-Powered Call Strategy Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print("✅ PDF report generated successfully.")
