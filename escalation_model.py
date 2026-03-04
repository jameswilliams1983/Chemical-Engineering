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
    # Expected annual trends
    mu_inflation: float = 0.025,
    mu_product: float = 0.0065,
    mu_feed: float = 0.0035,
    mu_energy: float = 0.0040,
    # Volatility (annual)
    sigma_product: float = 0.060,
    sigma_feed: float = 0.070,
    sigma_energy: float = 0.080,
    sigma_inflation: float = 0.010,
):
    rng = np.random.default_rng(seed)

    rev = np.full(n_sims, revenue0, dtype=float)
    cogs = np.full(n_sims, cogs0, dtype=float)
    opex = np.full(n_sims, opex0, dtype=float)

    for _ in range(years):
        infl = mu_inflation + rng.normal(0.0, sigma_inflation, n_sims)
        prod = mu_product + rng.normal(0.0, sigma_product, n_sims)
        feed = mu_feed + rng.normal(0.0, sigma_feed, n_sims)
        energy = mu_energy + rng.normal(0.0, sigma_energy, n_sims)

        rev *= (1.0 + infl + prod)        # selling price moves with inflation + product real
        cogs *= (1.0 + infl + feed)       # feedstock driven
        opex *= (1.0 + infl + energy)     # energy and utilities driven

    total_cost0 = cogs0 + opex0
    total_costT = cogs + opex

    revenue_cagrs = (rev / revenue0) ** (1.0 / years) - 1.0
    cost_cagrs = (total_costT / total_cost0) ** (1.0 / years) - 1.0

    def summarise(arr: np.ndarray):
        arr = arr[~np.isnan(arr)]
        return {
            "median": float(np.quantile(arr, 0.50)),
            "p10": float(np.quantile(arr, 0.10)),
            "p90": float(np.quantile(arr, 0.90)),
            "mean": float(np.mean(arr)),
        }

    summary = {
        "revenue_escalation_cagr": summarise(revenue_cagrs),
        "cost_escalation_cagr": summarise(cost_cagrs),
    }

    return revenue_cagrs, cost_cagrs, summary


if __name__ == "__main__":
    rev_cagrs, cost_cagrs, summary = simulate_escalation(years=10, n_sims=20000, seed=42)

    r = summary["revenue_escalation_cagr"]
    c = summary["cost_escalation_cagr"]

    print("Revenue escalation CAGR (annual):")
    print(f"  Median: {100*r['median']:.2f}%  P10: {100*r['p10']:.2f}%  P90: {100*r['p90']:.2f}%  Mean: {100*r['mean']:.2f}%")

    print("\nCost escalation CAGR (annual):")
    print(f"  Median: {100*c['median']:.2f}%  P10: {100*c['p10']:.2f}%  P90: {100*c['p90']:.2f}%  Mean: {100*c['mean']:.2f}%")