# Hedge Fund Review of `eose_put_hedge_v3.py`

## Correctness Findings
1. **Put probability direction is reversed.** `calculate_price_probability` computes `1 - norm.cdf(z)` which returns the probability that the terminal price **exceeds** the target. In the hedge context we need `P(S_T \leq strike)` for ITM put probability, so the current implementation overstates downside risk and inflates `prob_itm`, `expected_value`, and the efficiency ranking for every contract. Swap to `norm.cdf(z)` when evaluating puts. 【F:eose_put_hedge_v3.py†L237-L262】
2. **Option pricing uses historical drift/vol without risk-neutral adjustment.** The probability model fixes `mu = 0` and `sigma = annual_vol` derived from historical closes, ignoring carry (risk-free minus dividend) and forward-vol skew. This misprices ITM probability vs. what the market-implied volatility surface suggests, potentially biasing hedge selection away from real costs. 【F:eose_put_hedge_v3.py†L244-L262】【F:eose_put_hedge_v3.py†L168-L233】
3. **Forward band projection grows volatility deterministically.** The projection inflates the rolling std by `time_factor` irrespective of realized variance or term structure, which can create unrealistically wide expected lows/highs that feed the strike filter. A model tied to annualized vol and square-root-of-time scaling would align closer to market conventions. 【F:eose_put_hedge_v3.py†L188-L233】
4. **Hedge budget ignores liquidity and execution costs.** Premium uses `lastPrice` when available and otherwise mid, without checking bid/ask width or minimum volume/OI thresholds. Recommendations can target illiquid strikes that cannot be executed at quoted marks. 【F:eose_put_hedge_v3.py†L279-L336】【F:eose_put_hedge_v3.py†L374-L438】
5. **Scenario protection metric averages deep downside outcomes equally.** `avg_protection` weights 50–90% price scenarios evenly; this over-emphasizes tail moves relative to a risk-budgeted hedge and overstates value for far OTM strikes. Consider weighting by probability or using expected shortfall conditional on VaR. 【F:eose_put_hedge_v3.py†L293-L336】

## Assumption Checks
- **Zero drift / historical vol proxy:** Assumes geometric Brownian motion with zero drift and constant historical volatility; ignores jumps, earnings events, and IV term structure. 【F:eose_put_hedge_v3.py†L244-L262】【F:eose_put_hedge_v3.py†L168-L233】
- **Bollinger-derived expected low:** Uses Bollinger Band extrapolation as a proxy for one-month downside, effectively replacing implied distribution with a technical overlay; this could under/overstate true risk versus options market signals. 【F:eose_put_hedge_v3.py†L188-L233】
- **Budget laddering:** Caps notional at `shares_held/100` contracts per expiry and allocates sequentially until the budget is spent, assuming fills at marks and no partial fills/fees. 【F:eose_put_hedge_v3.py†L400-L438】

## Recommendations
- Flip the ITM probability to `norm.cdf(z)` for put payouts and validate against Black-Scholes or option-implied probabilities.
- Replace historical-vol projection with implied-vol (IV) by expiry and use risk-neutral drift (`r - q`) when estimating strike hit probabilities.
- Use square-root-of-time scaling for volatility and avoid deterministic volatility growth; consider Monte Carlo or closed-form lognormal projections.
- Enforce liquidity screens (min OI/volume, bid/ask spread thresholds) and price at executable marks (e.g., bid for buys).
- Reweight downside scenarios using probability-weighted loss metrics (e.g., ES/VaR) so hedge efficiency aligns with risk appetite.
