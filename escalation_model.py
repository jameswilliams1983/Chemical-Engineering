import numpy as np

def cagr(x0: float, xT: float, years: int) -> float:
    if x0 <= 0 or xT <= 0:
        return np.nan
    return (xT / x0) ** (1.0 / years) - 1.0

def simulate_escalation(
    years: int = 10,
    n_sims: int = 20000,
    seed: int = 42,
    # Base year (USD)
    revenue0: float = 162_116_958.67,
    cogs0: float = 100_458_789.21,
    opex0: float = 9_812_335.22,
    # Optional: split out energy share inside opex if you want
    opex_energy_share: float = 0.75,  # 75% of opex behaves like energy, rest like general inflation

    # Long run drifts (annual)
    mu_general_infl: float = 0.025,      # general inflation
    mu_demand: float = 0.015,            # demand growth affecting selling prices
    mu_product_real: float = 0.000,      # product real price trend beyond inflation
    mu_feed_real: float = 0.005,         # feed real price trend beyond inflation
    mu_energy_real: float = 0.010,       # energy real price trend beyond inflation

    # Volatilities (annual). Keep simple and transparent.
    sigma_demand: float = 0.030,
    sigma_product: float = 0.080,
    sigma_feed: float = 0.100,
    sigma_energy: float = 0.150,
    sigma_infl: float = 0.010,

    # Carbon (optional)
    include_carbon: bool = False,
    carbon0: float = 0.0,                # base year carbon cost in USD
    mu_carbon: float = 0.070,            # annual carbon price drift
    sigma_carbon: float = 0.200,         # annual carbon price volatility

    # FX (optional). If some costs are actually GBP but reported in USD.
    include_fx: bool = False,
    fx_usd_per_gbp_0: float = 1.25,
    mu_fx: float = 0.00,                 # drift in USD/GBP
    sigma_fx: float = 0.08,              # volatility in USD/GBP
    gbp_cost_share_of_opex: float = 0.50 # fraction of opex that is really GBP denominated
):
    """
    Returns:
      revenue_cagrs, cost_cagrs arrays of length n_sims
      plus a small dict of summary stats
    """
    rng = np.random.default_rng(seed)

    # Pre-allocate end values
    rev_end = np.zeros(n_sims)
    cost_end = np.zeros(n_sims)

    # Start levels
    rev = np.full(n_sims, revenue0, dtype=float)
    cogs = np.full(n_sims, cogs0, dtype=float)
    opex = np.full(n_sims, opex0, dtype=float)
    carbon = np.full(n_sims, carbon0, dtype=float)

    fx = np.full(n_sims, fx_usd_per_gbp_0, dtype=float)

    # Build yearly simulation
    for _ in range(years):
        # Random shocks
        eps_infl = rng.normal(0.0, sigma_infl, n_sims)
        eps_dem = rng.normal(0.0, sigma_demand, n_sims)
        eps_prod = rng.normal(0.0, sigma_product, n_sims)
        eps_feed = rng.normal(0.0, sigma_feed, n_sims)
        eps_energy = rng.normal(0.0, sigma_energy, n_sims)

        # Inflation process
        infl = np.clip(mu_general_infl + eps_infl, -0.10, 0.20)

        # Demand growth process
        demand_g = np.clip(mu_demand + eps_dem, -0.20, 0.25)

        # Price changes: inflation + real component + shock
        # Using simple arithmetic rates, applied multiplicatively
        prod_price_g = np.clip(infl + mu_product_real + eps_prod + 0.5 * demand_g, -0.40, 0.50)
        feed_price_g = np.clip(infl + mu_feed_real + eps_feed, -0.40, 0.60)
        energy_price_g = np.clip(infl + mu_energy_real + eps_energy, -0.60, 0.80)

        # Update revenue: assume volumes flat and price responds to demand
        # If you want, you can also multiply by (1 + demand_g) to include volume growth.
        rev *= (1.0 + prod_price_g)

        # Update COGS: driven by feed price
        cogs *= (1.0 + feed_price_g)

        # Update Opex: split into energy-like and general inflation-like
        opex_energy = opex * opex_energy_share
        opex_other = opex * (1.0 - opex_energy_share)

        opex_energy *= (1.0 + energy_price_g)
        opex_other *= (1.0 + infl)

        opex = opex_energy + opex_other

        # Carbon (optional)
        if include_carbon:
            eps_c = rng.normal(0.0, sigma_carbon, n_sims)
            carbon_g = np.clip(mu_carbon + eps_c, -0.50, 1.00)
            carbon *= (1.0 + carbon_g)

        # FX (optional): apply FX translation to GBP denominated share of opex
        if include_fx:
            eps_fx = rng.normal(0.0, sigma_fx, n_sims)
            fx_g = np.clip(mu_fx + eps_fx, -0.30, 0.30)
            fx *= (1.0 + fx_g)

            # If part of opex is GBP denominated, its USD value moves with FX
            # Interpreting opex as already in USD base year, so adjust only the GBP share by FX ratio
            fx_ratio = fx / fx_usd_per_gbp_0
            opex = opex * (1.0 - gbp_cost_share_of_opex) + (opex * gbp_cost_share_of_opex) * fx_ratio

    rev_end[:] = rev
    total_cost_end = cogs + opex + (carbon if include_carbon else 0.0)
    total_cost0 = cogs0 + opex0 + (carbon0 if include_carbon else 0.0)

    # Convert to single escalation percentages via CAGR
    revenue_cagrs = np.array([cagr(revenue0, rT, years) for rT in rev_end])
    cost_cagrs = np.array([cagr(total_cost0, cT, years) for cT in total_cost_end])

    def summarise(arr):
        arr = arr[~np.isnan(arr)]
        return {
            "median": float(np.quantile(arr, 0.50)),
            "p10": float(np.quantile(arr, 0.10)),
            "p90": float(np.quantile(arr, 0.90)),
            "mean": float(np.mean(arr))
        }

    summary = {
        "revenue_escalation_cagr": summarise(revenue_cagrs),
        "cost_escalation_cagr": summarise(cost_cagrs),
    }

    return revenue_cagrs, cost_cagrs, summary

if __name__ == "__main__":
    rev_cagrs, cost_cagrs, summary = simulate_escalation(
        years=10,
        n_sims=20000,
        # Toggle these on if you want them in the model
        include_carbon=False,
        include_fx=False
    )

    # Print results as percentages
    r = summary["revenue_escalation_cagr"]
    c = summary["cost_escalation_cagr"]

    print("Revenue escalation CAGR (annual):")
    print(f"  Median: {100*r['median']:.2f}%  P10: {100*r['p10']:.2f}%  P90: {100*r['p90']:.2f}%  Mean: {100*r['mean']:.2f}%")

    print("\nCost escalation CAGR (annual):")
    print(f"  Median: {100*c['median']:.2f}%  P10: {100*c['p10']:.2f}%  P90: {100*c['p90']:.2f}%  Mean: {100*c['mean']:.2f}%")