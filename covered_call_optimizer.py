#!/usr/bin/env python3
"""
Covered Call Optimizer
======================

Optimizes covered call selling for income generation using a laddered strategy.
- Filters for Strike >= Cost Basis (Safety)
- Ladders contracts across 30-180 DTE (Diversification)
- Generates professional PDF report
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from scipy import stats

warnings.filterwarnings('ignore')

# =================== CONFIGURATION ===================

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Covered Call Optimizer")
    parser.add_argument("--ticker", type=str, default="EOSE", help="Ticker symbol")
    parser.add_argument("--shares", type=int, default=600, help="Total shares held")
    parser.add_argument("--cost-basis", type=float, default=15.00, help="Average cost per share")
    parser.add_argument("--min-dte", type=int, default=30, help="Minimum DTE")
    parser.add_argument("--max-dte", type=int, default=180, help="Maximum DTE")
    parser.add_argument("--max-per-date", type=int, default=2, help="Max contracts per expiration")
    return parser.parse_args()

args = parse_cli_args()

TICKER = args.ticker
SHARES_HELD = args.shares
COST_BASIS = args.cost_basis
MIN_DTE = args.min_dte
MAX_DTE = args.max_dte
MAX_PER_DATE = args.max_per_date

# =================== DATA LOADING ===================

def fetch_live_options_chain(ticker: str, min_dte: int, max_dte: int) -> Optional[pd.DataFrame]:
    """Fetch live call options chain."""
    try:
        ticker_obj = yf.Ticker(ticker)
        expirations = ticker_obj.options

        if not expirations:
            return None

        all_calls = []
        today = pd.Timestamp.today().normalize()

        for exp_date_str in expirations:
            try:
                exp_date = pd.to_datetime(exp_date_str)
                dte = (exp_date - today).days

                if min_dte <= dte <= max_dte:
                    opt_chain = ticker_obj.option_chain(exp_date_str)
                    calls = opt_chain.calls.copy()

                    calls["expiration"] = exp_date_str
                    calls["exp_date"] = exp_date
                    calls["dte"] = dte
                    calls["strike"] = calls["strike"].astype(float)

                    # Filter valid pricing
                    valid_pricing = (calls["bid"] > 0) | (calls["lastPrice"] > 0)
                    calls = calls[valid_pricing]

                    if not calls.empty:
                        all_calls.append(calls)

            except Exception:
                continue

        if all_calls:
            return pd.concat(all_calls, ignore_index=True)
        return None

    except Exception:
        return None

def get_current_price(ticker: str) -> float:
    """Get current stock price."""
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        return 0.0
    except Exception:
        return 0.0

# =================== STRATEGY LOGIC ===================

def optimize_ladder(calls_df: pd.DataFrame, current_price: float) -> List[Dict]:
    """
    Implement laddering strategy:
    1. Filter Strike >= Cost Basis
    2. Group by DTE buckets (30-60, 60-90, etc.)
    3. Select best premium options
    4. Allocate contracts
    """
    if calls_df is None or calls_df.empty:
        return []

    # 1. Safety Filter: Strike >= Cost Basis
    safe_calls = calls_df[calls_df["strike"] >= COST_BASIS].copy()

    if safe_calls.empty:
        print(f"⚠️ No calls found above cost basis ${COST_BASIS:.2f}")
        return []

    # Calculate metrics
    safe_calls["mid_price"] = (safe_calls["bid"] + safe_calls["ask"]) / 2
    safe_calls["premium"] = np.where(safe_calls["mid_price"] > 0, safe_calls["mid_price"], safe_calls["lastPrice"])

    # Return calculation (Annualized)
    safe_calls["return_pct"] = (safe_calls["premium"] / current_price) * 100
    safe_calls["annualized_return"] = safe_calls["return_pct"] * (365 / safe_calls["dte"])

    # Sort by premium (income maximization)
    safe_calls = safe_calls.sort_values("premium", ascending=False)

    recommendations = []
    contracts_allocated = 0
    max_contracts = SHARES_HELD // 100

    # Define time buckets for laddering
    buckets = [
        (30, 60),
        (60, 90),
        (90, 120),
        (120, 150),
        (150, 180)
    ]

    print(f"🎯 Optimizing Ladder Strategy (Max {max_contracts} contracts)...")

    for min_d, max_d in buckets:
        if contracts_allocated >= max_contracts:
            break

        # Find best option in this bucket
        bucket_opts = safe_calls[
            (safe_calls["dte"] >= min_d) &
            (safe_calls["dte"] < max_d)
        ]

        if not bucket_opts.empty:
            # Take top performer
            best_opt = bucket_opts.iloc[0]

            # Allocate contracts (up to max_per_date, but not exceeding total shares)
            remaining_capacity = max_contracts - contracts_allocated
            to_allocate = min(MAX_PER_DATE, remaining_capacity)

            if to_allocate > 0:
                rec = {
                    "strike": best_opt["strike"],
                    "dte": best_opt["dte"],
                    "expiration": best_opt["expiration"],
                    "premium": best_opt["premium"],
                    "contracts": to_allocate,
                    "income": best_opt["premium"] * 100 * to_allocate,
                    "return_pct": best_opt["return_pct"],
                    "annualized_return": best_opt["annualized_return"],
                    "bucket": f"{min_d}-{max_d}d"
                }
                recommendations.append(rec)
                contracts_allocated += to_allocate
                print(f"✅ Added: {rec['dte']}d Call @ ${rec['strike']:.2f} ({to_allocate} contracts)")

    return recommendations

# =================== MAIN EXECUTION ===================

print(f"🚀 Starting Covered Call Optimizer for {TICKER}...")
print(f"💰 Cost Basis: ${COST_BASIS:.2f} | Shares: {SHARES_HELD}")

current_price = get_current_price(TICKER)
print(f"📉 Current Price: ${current_price:.2f}")

calls_df = fetch_live_options_chain(TICKER, MIN_DTE, MAX_DTE)
recommendations = optimize_ladder(calls_df, current_price)

# =================== PDF GENERATION ===================

if recommendations:
    pdf_filename = f"{TICKER}_income_strategy_{pd.Timestamp.today().strftime('%Y%m%d')}.pdf"
    print(f"\n📄 Generating PDF report: {pdf_filename}")

    with PdfPages(pdf_filename) as pdf:
        # PAGE 1: SUMMARY
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(111)
        ax.axis("off")

        # Title
        ax.text(0.5, 0.95, f"Covered Call Income Strategy: {TICKER}",
               fontsize=24, fontweight="bold", ha="center", color="#2E86AB")

        # Metrics
        total_income = sum(r["income"] for r in recommendations)
        avg_return = np.mean([r["annualized_return"] for r in recommendations])
        contracts_sold = sum(r["contracts"] for r in recommendations)
        coverage_pct = (contracts_sold * 100) / SHARES_HELD

        metrics = [
            f"Total Income: ${total_income:,.2f}",
            f"Avg Annualized Return: {avg_return:.1f}%",
            f"Shares Covered: {coverage_pct:.1f}% ({contracts_sold*100}/{SHARES_HELD})",
            f"Safety Buffer: ${(COST_BASIS - current_price):.2f} below strike"
        ]

        y_metrics = 0.8
        for m in metrics:
            ax.text(0.5, y_metrics, m, fontsize=16, ha="center",
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F0F0", alpha=0.5))
            y_metrics -= 0.08

        # Strategy Note
        ax.text(0.5, 0.4,
               f"Strategy: Laddered selling of calls with Strike >= ${COST_BASIS:.2f} (Cost Basis).\n"
               f"Contracts distributed across {MIN_DTE}-{MAX_DTE} days to maximize income stability.",
               fontsize=12, ha="center", style="italic")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PAGE 2: DETAILED LADDER (Professional Table)
        fig_table = plt.figure(figsize=(20, 12))
        ax_table = fig_table.add_subplot(111)
        ax_table.axis("off")

        # Header Bar
        table_width = 0.8
        table_left = (1.0 - table_width) / 2

        ax_table.add_patch(Rectangle((table_left, 0.85), table_width, 0.08, facecolor="#2E86AB", alpha=1.0))
        ax_table.text(0.5, 0.89, "Income Ladder Positions",
                     fontsize=24, fontweight="bold", ha="center", va="center", color="white")

        # Headers
        headers = ["Bucket", "DTE", "Strike", "Contracts", "Premium", "Income", "Ann. Return"]
        col_width = table_width / len(headers)
        col_positions = [table_left + (i * col_width) + (col_width/2) for i in range(len(headers))]

        header_y = 0.78
        ax_table.add_patch(Rectangle((table_left, header_y - 0.025), table_width, 0.05, facecolor="#E0E0E0"))

        for i, h in enumerate(headers):
            ax_table.text(col_positions[i], header_y, h, fontsize=14, fontweight="bold", ha="center")

        # Rows
        y_pos = 0.70
        row_height = 0.05

        for idx, rec in enumerate(recommendations):
            # Zebra striping
            bg_color = "#F8F9FA" if idx % 2 == 0 else "#FFFFFF"
            ax_table.add_patch(Rectangle((table_left, y_pos - 0.02), table_width, row_height, facecolor=bg_color))

            row_data = [
                rec["bucket"],
                f"{rec['dte']}d",
                f"${rec['strike']:.2f}",
                str(rec["contracts"]),
                f"${rec['premium']:.2f}",
                f"${rec['income']:.2f}",
                f"{rec['annualized_return']:.1f}%"
            ]

            for i, d in enumerate(row_data):
                ax_table.text(col_positions[i], y_pos, d, fontsize=12, ha="center")

            y_pos -= row_height

        pdf.savefig(fig_table, bbox_inches="tight")
        plt.close(fig_table)

        # =================== ADDITIONAL GRAPHS ===================
        # 1. Stock Price History Chart
        try:
            hist_df = yf.Ticker(TICKER).history(period="1y")
            if not hist_df.empty:
                fig_price = plt.figure(figsize=(20, 12))
                ax_price = fig_price.add_subplot(111)
                ax_price.plot(hist_df.index, hist_df["Close"], label="Close Price", color="#2E86AB")
                ax_price.set_title(f"{TICKER} Price History (1 Year)", fontsize=20, fontweight="bold")
                ax_price.set_xlabel("Date")
                ax_price.set_ylabel("Price ($)")
                ax_price.legend()
                ax_price.grid(True, alpha=0.3)
                pdf.savefig(fig_price, bbox_inches="tight")
                plt.close(fig_price)
        except Exception as e:
            print(f"⚠️ Failed to generate price history chart: {e}")

        # 2. Income Allocation Bar Chart
        try:
            buckets = [rec["bucket"] for rec in recommendations]
            incomes = [rec["income"] for rec in recommendations]
            fig_income = plt.figure(figsize=(20, 12))
            ax_income = fig_income.add_subplot(111)
            ax_income.bar(buckets, incomes, color="#06A77D")
            ax_income.set_title("Income Allocation by DTE Bucket", fontsize=20, fontweight="bold")
            ax_income.set_xlabel("DTE Bucket")
            ax_income.set_ylabel("Income ($)")
            for i, v in enumerate(incomes):
                ax_income.text(i, v + max(incomes) * 0.02, f"${v:,.0f}", ha="center", fontsize=12)
            pdf.savefig(fig_income, bbox_inches="tight")
            plt.close(fig_income)
        except Exception as e:
            print(f"⚠️ Failed to generate income allocation chart: {e}")

        # 3. Payoff Diagram (combined for all contracts)
        try:
            price_range = np.linspace(0, current_price * 2, 500)
            total_payoff = np.zeros_like(price_range)
            for rec in recommendations:
                # Covered call payoff per contract (100 shares)
                strike = rec["strike"]
                premium = rec["premium"]
                contracts = rec["contracts"]
                # Payoff: if price <= strike, keep premium + (price - cost_basis)*100; if price > strike, premium + (strike - cost_basis)*100
                # Simplified: assume cost basis is COST_BASIS
                payoff_per_share = np.where(
                    price_range <= strike,
                    (price_range - COST_BASIS) + premium,
                    (strike - COST_BASIS) + premium
                )
                total_payoff += payoff_per_share * 100 * contracts
            fig_payoff = plt.figure(figsize=(20, 12))
            ax_payoff = fig_payoff.add_subplot(111)
            ax_payoff.plot(price_range, total_payoff, color="#D62828", linewidth=2)
            ax_payoff.axhline(0, color="black", linewidth=1, linestyle="--")
            ax_payoff.set_title("Combined Covered Call Payoff Diagram", fontsize=20, fontweight="bold")
            ax_payoff.set_xlabel("Underlying Price at Expiration ($)")
            ax_payoff.set_ylabel("Total Profit ($)")
            ax_payoff.grid(True, alpha=0.3)
            pdf.savefig(fig_payoff, bbox_inches="tight")
            plt.close(fig_payoff)
        except Exception as e:
            print(f"⚠️ Failed to generate payoff diagram: {e}")


    print("✅ Analysis Complete.")
else:
    print("❌ No valid recommendations found. Check cost basis or market data.")
