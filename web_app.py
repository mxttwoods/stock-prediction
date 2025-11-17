"""Lightweight Flask UI for screening call and put ideas.

The server keeps calculations simple so that users can enter a ticker, budget,
and other constraints via a form and instantly see ranked candidates. Both
calls and puts are supported and results are filtered by budget, days to
expiration, and a basic efficiency score.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template_string, request
from scipy import stats

app = Flask(__name__)


def load_price_history(ticker: str, lookback_days: int, cache_file: Path) -> pd.DataFrame:
    """Load daily price data with simple parquet caching."""
    today = pd.Timestamp.today().normalize()
    start_date = today - pd.Timedelta(days=lookback_days + 5)

    df: Optional[pd.DataFrame] = None

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
    """Fetch live call and put chains within the requested DTE window."""
    try:
        ticker_obj = yf.Ticker(ticker)
        expirations = ticker_obj.options

        if not expirations:
            return None

        today = pd.Timestamp.today().normalize()
        contracts: List[pd.DataFrame] = []

        for exp_date_str in expirations:
            exp_date = pd.to_datetime(exp_date_str)
            dte = (exp_date - today).days

            if 0 < dte <= max_dte:
                opt_chain = ticker_obj.option_chain(exp_date_str)
                for option_type, frame in ("call", opt_chain.calls), ("put", opt_chain.puts):
                    subset = frame.copy()
                    subset["optionType"] = option_type
                    subset["expiration"] = exp_date_str
                    subset["exp_date"] = exp_date
                    subset["DTE"] = dte
                    subset["strike"] = subset["strike"].astype(float)
                    subset["lastPrice"] = subset["lastPrice"].fillna(0).astype(float)
                    subset["bid"] = subset["bid"].fillna(0).astype(float)
                    subset["ask"] = subset["ask"].fillna(0).astype(float)
                    subset["volume"] = subset["volume"].fillna(0).astype(int)
                    subset["openInterest"] = subset["openInterest"].fillna(0).astype(int)
                    contracts.append(subset)

        if contracts:
            combined = pd.concat(contracts, ignore_index=True)
            return combined
        return None
    except Exception:
        return None


def calculate_price_probability(
    target_price: float, current_price: float, days: int, daily_vol: float, annual_vol: float
) -> float:
    """Probability the asset closes at or above the target price within the window."""
    if days <= 0:
        return 0.0

    t = days / 252.0
    mu = 0  # Driftless assumption
    sigma = annual_vol

    log_ratio = np.log(target_price / current_price)
    mean_log = (mu - 0.5 * sigma**2) * t
    var_log = sigma**2 * t

    if var_log <= 0:
        return 0.5 if abs(target_price - current_price) < 0.01 else 0.0

    z_score = (log_ratio - mean_log) / np.sqrt(var_log)
    return 1 - stats.norm.cdf(z_score)


def _mid_price(row: pd.Series) -> float:
    if not pd.isna(row.get("lastPrice")) and row.get("lastPrice"):
        return float(row["lastPrice"])
    bid = float(row.get("bid", 0) or 0)
    ask = float(row.get("ask", 0) or 0)
    if bid and ask:
        return (bid + ask) / 2
    return bid or ask or 0.0


def analyze_option(
    option_row: pd.Series,
    option_type: str,
    current_price: float,
    shares_held: int,
    daily_vol: float,
    annual_vol: float,
) -> Dict:
    strike = float(option_row["strike"])
    premium = _mid_price(option_row)
    dte = int(option_row["DTE"])
    exp_date = option_row["exp_date"]

    # Probability of finishing in the desired direction
    prob_upside = calculate_price_probability(strike, current_price, dte, daily_vol, annual_vol)
    if option_type == "call":
        prob_itm = prob_upside
        scenario_multipliers = [1.05, 1.1, 1.2, 1.3]
        intrinsic_fn = lambda price: max(price - strike, 0.0)
    else:
        prob_itm = 1 - prob_upside
        scenario_multipliers = [0.5, 0.6, 0.7, 0.8, 0.9]
        intrinsic_fn = lambda price: max(strike - price, 0.0)

    scenario_prices = [current_price * m for m in scenario_multipliers]
    avg_intrinsic = float(np.mean([intrinsic_fn(p) for p in scenario_prices]))

    cost_per_contract = premium * 100
    efficiency_score = (avg_intrinsic / premium * prob_itm * 100) if premium > 0 else 0.0

    return {
        "option_type": option_type,
        "strike": strike,
        "premium": premium,
        "dte": dte,
        "exp_date": exp_date,
        "expiration": option_row["expiration"],
        "prob_itm": prob_itm,
        "average_payoff": avg_intrinsic,
        "efficiency_score": efficiency_score,
        "distance_pct": ((strike - current_price) / current_price) * 100,
        "cost_per_contract": cost_per_contract,
        "volume": int(option_row.get("volume", 0)),
        "open_interest": int(option_row.get("openInterest", 0)),
        "breakeven": strike + premium if option_type == "call" else strike - premium,
        "contracts_for_100_shares": max(int(shares_held / 100), 1),
    }


def recommend_options(
    options_df: pd.DataFrame,
    option_type: str,
    current_price: float,
    shares_held: int,
    budget: float,
    daily_vol: float,
    annual_vol: float,
) -> List[Dict]:
    if options_df is None or options_df.empty:
        return []

    filtered = options_df[options_df["optionType"] == option_type].copy()
    if filtered.empty:
        return []

    analyzed: List[Dict] = []
    for _, row in filtered.iterrows():
        try:
            analyzed.append(
                analyze_option(row, option_type, current_price, shares_held, daily_vol, annual_vol)
            )
        except Exception:
            continue

    if not analyzed:
        return []

    analyzed_df = pd.DataFrame(analyzed)

    if option_type == "put":
        candidates = analyzed_df[(analyzed_df["strike"] <= current_price) & (analyzed_df["premium"] > 0)]
    else:
        candidates = analyzed_df[(analyzed_df["strike"] >= current_price * 0.9) & (analyzed_df["premium"] > 0)]

    if budget > 0:
        candidates = candidates[candidates["cost_per_contract"] <= budget]

    if candidates.empty:
        return []

    candidates = candidates.sort_values(
        by=["efficiency_score", "prob_itm"], ascending=False
    ).head(12)

    return candidates.to_dict(orient="records")


def format_currency(value: float) -> str:
    return f"${value:,.2f}" if value or value == 0 else "-"


def format_pct(value: float) -> str:
    return f"{value:.1f}%" if value or value == 0 else "-"


def format_prob(value: float) -> str:
    return f"{value*100:.1f}%" if value or value == 0 else "-"


TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Options Screener</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #f8f9fa; color: #222; }
    h1 { margin-bottom: 0.25rem; }
    form { background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    label { display: block; margin-top: 0.6rem; font-weight: 600; }
    input, select { padding: 0.4rem; width: 100%; border: 1px solid #ccc; border-radius: 4px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; }
    .submit { margin-top: 1rem; }
    table { width: 100%; border-collapse: collapse; margin-top: 1rem; background: white; }
    th, td { padding: 0.6rem; border-bottom: 1px solid #e2e6ea; text-align: left; }
    th { background: #f1f3f5; }
    .pill { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 12px; font-size: 0.85rem; }
    .pill.call { background: #e5f4ff; color: #0969da; }
    .pill.put { background: #ffe8e5; color: #c0392b; }
    .muted { color: #6c757d; }
    .card { background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-top: 1rem; }
  </style>
</head>
<body>
  <h1>Options Screener</h1>
  <p class="muted">Screen calls or puts quickly with a lightweight Flask UI.</p>

  <form method="post">
    <div class="grid">
      <div>
        <label for="ticker">Ticker</label>
        <input type="text" id="ticker" name="ticker" value="{{ form_values.ticker }}" required />
      </div>
      <div>
        <label for="option_type">Option Type</label>
        <select id="option_type" name="option_type">
          <option value="call" {% if form_values.option_type == 'call' %}selected{% endif %}>Calls</option>
          <option value="put" {% if form_values.option_type == 'put' %}selected{% endif %}>Puts</option>
        </select>
      </div>
      <div>
        <label for="shares_held">Shares Held</label>
        <input type="number" id="shares_held" name="shares_held" min="0" value="{{ form_values.shares_held }}" />
      </div>
      <div>
        <label for="budget">Budget (per position)</label>
        <input type="number" id="budget" name="budget" step="0.01" min="0" value="{{ form_values.budget }}" />
      </div>
      <div>
        <label for="max_dte">Max Days to Expiration</label>
        <input type="number" id="max_dte" name="max_dte" min="1" value="{{ form_values.max_dte }}" />
      </div>
      <div>
        <label for="lookback">Price History Lookback (days)</label>
        <input type="number" id="lookback" name="lookback" min="30" value="{{ form_values.lookback }}" />
      </div>
    </div>
    <div class="submit">
      <button type="submit">Run Screen</button>
    </div>
  </form>

  {% if error %}
    <div class="card">
      <strong>Issue:</strong> {{ error }}
    </div>
  {% endif %}

  {% if summary %}
    <div class="card">
      <strong>{{ summary.ticker }}</strong> last close: {{ summary.price }} · Annual vol: {{ summary.vol }} · Options checked: {{ summary.count }}
    </div>
  {% endif %}

  {% if recommendations %}
    <table>
      <thead>
        <tr>
          <th>Type</th>
          <th>Strike</th>
          <th>Premium</th>
          <th>Breakeven</th>
          <th>DTE</th>
          <th>Prob ITM</th>
          <th>Avg Payoff</th>
          <th>Efficiency</th>
          <th>Volume / OI</th>
          <th>Cost</th>
        </tr>
      </thead>
      <tbody>
        {% for rec in recommendations %}
          <tr>
            <td><span class="pill {{ rec.option_type }}">{{ rec.option_type|title }}</span></td>
            <td>{{ rec.strike | round(2) }}</td>
            <td>{{ format_currency(rec.premium) }}</td>
            <td>{{ format_currency(rec.breakeven) }}</td>
            <td>{{ rec.dte }} days</td>
            <td>{{ format_prob(rec.prob_itm) }}</td>
            <td>{{ format_currency(rec.average_payoff) }}</td>
            <td>{{ format_pct(rec.efficiency_score) }}</td>
            <td>{{ rec.volume }} / {{ rec.open_interest }}</td>
            <td>{{ format_currency(rec.cost_per_contract) }}</td>
          </tr>
        {% endfor %}
      </tbody>
    </table>
  {% elif summary %}
    <div class="card">No matching options found under the current filters.</div>
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    error = None
    recommendations: List[Dict] = []
    summary = None

    form_values = {
        "ticker": request.form.get("ticker", "EOSE").strip().upper(),
        "option_type": request.form.get("option_type", "put"),
        "shares_held": int(request.form.get("shares_held", 100) or 0),
        "budget": float(request.form.get("budget", 150) or 0),
        "max_dte": int(request.form.get("max_dte", 45) or 30),
        "lookback": int(request.form.get("lookback", 365) or 365),
    }

    if request.method == "POST":
        ticker = form_values["ticker"]
        cache_file = Path(f"{ticker.lower()}_daily.parquet")

        try:
            df = load_price_history(ticker, form_values["lookback"], cache_file)
            if df.empty:
                raise ValueError("No price history returned.")

            if isinstance(df.columns, pd.MultiIndex):
                cols = df.columns.get_level_values(0)
                close = df.xs("Adj Close", axis=1, level=0).iloc[:, 0] if "Adj Close" in set(cols) else df.xs("Close", axis=1, level=0).iloc[:, 0]
            else:
                close = df["Adj Close"].copy() if "Adj Close" in df.columns else df["Close"].copy()

            current_price = float(close.iloc[-1])
            returns = close.pct_change().dropna()
            daily_vol = returns.std()
            annual_vol = float(daily_vol * np.sqrt(252))

            options_df = fetch_live_options_chain(ticker, form_values["max_dte"])
            if options_df is None or options_df.empty:
                raise ValueError("No live option chain data available for this request.")

            recommendations = recommend_options(
                options_df,
                form_values["option_type"],
                current_price,
                form_values["shares_held"],
                form_values["budget"],
                daily_vol,
                annual_vol,
            )

            summary = {
                "ticker": ticker,
                "price": format_currency(current_price),
                "vol": format_pct(annual_vol * 100),
                "count": len(options_df),
            }
        except Exception as exc:
            error = str(exc)

    return render_template_string(
        TEMPLATE,
        form_values=form_values,
        recommendations=recommendations,
        summary=summary,
        error=error,
        format_currency=format_currency,
        format_pct=format_pct,
        format_prob=format_prob,
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
