import numpy as np
import matplotlib.pyplot as plt


def cagr(x0: float, xT: float, years: int) -> float:
    if x0 <= 0 or xT <= 0:
        return np.nan
    return (xT / x0) ** (1.0 / years) - 1.0


def simulate_escalation(
    years: int = 10,
    n_sims: int = 20000,
    seed: int = 42,

    #initial financials
    revenue0: float = 162_116_958.67,
    cogs0: float = 100_458_789.21,
    opex0: float = 9_812_335.22,

    #mean values
    mu_inflation: float = 0.025,
    mu_product: float = 0.0065,
    mu_feed: float = 0.0035,
    mu_energy: float = 0.0040,

    #volatility values
    sigma_product: float = 0.060,
    sigma_feed: float = 0.070,
    sigma_energy: float = 0.080,
    sigma_inflation: float = 0.010,

    #shock probabilities
    prob_product_up: float = 0.03,
    prob_product_down: float = 0.03,
    prob_feed_up: float = 0.04,
    prob_feed_down: float = 0.03,
    prob_energy_up: float = 0.04,
    prob_energy_down: float = 0.02,

    #shock sizes
    product_up_jump: float = 0.12,
    product_down_jump: float = -0.12,
    feed_up_jump: float = 0.15,
    feed_down_jump: float = -0.10,
    energy_up_jump: float = 0.20,
    energy_down_jump: float = -0.10,
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

        #petrochemical shocks
        #product shocks
        prod[rng.random(n_sims) < prob_product_up] += product_up_jump
        prod[rng.random(n_sims) < prob_product_down] += product_down_jump

        #feed shocks
        feed[rng.random(n_sims) < prob_feed_up] += feed_up_jump
        feed[rng.random(n_sims) < prob_feed_down] += feed_down_jump

        # energy shocks
        energy[rng.random(n_sims) < prob_energy_up] += energy_up_jump
        energy[rng.random(n_sims) < prob_energy_down] += energy_down_jump

        #clip extreme annual changes 
        infl = np.clip(infl, -0.05, 0.10)
        prod = np.clip(prod, -0.25, 0.25)
        feed = np.clip(feed, -0.25, 0.30)
        energy = np.clip(energy, -0.20, 0.35)

        rev *= (1.0 + infl + prod)
        cogs *= (1.0 + infl + feed)
        opex *= (1.0 + infl + energy)

    total_cost0 = cogs0 + opex0
    total_costT = cogs + opex

    revenue_cagrs = (rev / revenue0) ** (1.0 / years) - 1.0
    cost_cagrs = (total_costT / total_cost0) ** (1.0 / years) - 1.0

    def summarise(arr):
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
   
    rev_cagrs, cost_cagrs, summary = simulate_escalation()

    r = summary["revenue_escalation_cagr"]
    c = summary["cost_escalation_cagr"]

    print("Revenue escalation CAGR (annual):")
    print(
        f"  Median: {100*r['median']:.2f}%  "
        f"P10: {100*r['p10']:.2f}%  "
        f"P90: {100*r['p90']:.2f}%  "
        f"Mean: {100*r['mean']:.2f}%"
    )

    print("\nCost escalation CAGR (annual):")
    print(
        f"  Median: {100*c['median']:.2f}%  "
        f"P10: {100*c['p10']:.2f}%  "
        f"P90: {100*c['p90']:.2f}%  "
        f"Mean: {100*c['mean']:.2f}%"
    )

    rev_pct = 100 * rev_cagrs
    cost_pct = 100 * cost_cagrs

    #Revenue histogram
    plt.figure()
    plt.hist(rev_pct, bins=60, density=True)
    plt.axvline(100 * r["median"], linewidth=2, color="forestgreen", label="Median")
    plt.axvline(100 * r["p10"], linestyle="--", linewidth=2, color="red", label="P10")
    plt.axvline(100 * r["p90"], linestyle="--", linewidth=2, color="red", label="P90")
    plt.title("Revenue escalation CAGR distribution")
    plt.xlabel("CAGR (% per year)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    #Cost histogram
    plt.figure()
    plt.hist(cost_pct, bins=60, density=True)
    plt.axvline(100 * c["median"], linewidth=2, color="forestgreen", label="Median")
    plt.axvline(100 * c["p10"], linestyle="--", linewidth=2, color="red", label="P10")
    plt.axvline(100 * c["p90"], linestyle="--", linewidth=2, color="red", label="P90")
    plt.title("Cost escalation CAGR distribution")
    plt.xlabel("CAGR (% per year)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    #Monte Carlo path plots
    years = 10
    paths_to_plot = 200
    rng = np.random.default_rng(42)

    revenue0 = 162_116_958.67
    cogs0 = 100_458_789.21
    opex0 = 9_812_335.22

    mu_inflation = 0.025
    mu_product = 0.0065
    mu_feed = 0.0035
    mu_energy = 0.0040

    sigma_product = 0.060
    sigma_feed = 0.070
    sigma_energy = 0.080
    sigma_inflation = 0.010

    prob_product_up = 0.03
    prob_product_down = 0.03
    prob_feed_up = 0.04
    prob_feed_down = 0.03
    prob_energy_up = 0.04
    prob_energy_down = 0.02

    product_up_jump = 0.12
    product_down_jump = -0.12
    feed_up_jump = 0.15
    feed_down_jump = -0.10
    energy_up_jump = 0.20
    energy_down_jump = -0.10

    rev = np.full(paths_to_plot, revenue0)
    cogs = np.full(paths_to_plot, cogs0)
    opex = np.full(paths_to_plot, opex0)

    rev_paths = np.zeros((years + 1, paths_to_plot))
    cost_paths = np.zeros((years + 1, paths_to_plot))

    rev_paths[0] = revenue0
    cost_paths[0] = cogs0 + opex0

    for t in range(1, years + 1):
        infl = mu_inflation + rng.normal(0.0, sigma_inflation, paths_to_plot)
        prod = mu_product + rng.normal(0.0, sigma_product, paths_to_plot)
        feed = mu_feed + rng.normal(0.0, sigma_feed, paths_to_plot)
        energy = mu_energy + rng.normal(0.0, sigma_energy, paths_to_plot)

        # same shocks as main simulation
        prod[rng.random(paths_to_plot) < prob_product_up] += product_up_jump
        prod[rng.random(paths_to_plot) < prob_product_down] += product_down_jump

        feed[rng.random(paths_to_plot) < prob_feed_up] += feed_up_jump
        feed[rng.random(paths_to_plot) < prob_feed_down] += feed_down_jump

        energy[rng.random(paths_to_plot) < prob_energy_up] += energy_up_jump
        energy[rng.random(paths_to_plot) < prob_energy_down] += energy_down_jump

        infl = np.clip(infl, -0.05, 0.10)
        prod = np.clip(prod, -0.25, 0.25)
        feed = np.clip(feed, -0.25, 0.30)
        energy = np.clip(energy, -0.20, 0.35)

        rev *= (1 + infl + prod)
        cogs *= (1 + infl + feed)
        opex *= (1 + infl + energy)

        rev_paths[t] = rev
        cost_paths[t] = cogs + opex

    years_axis = np.arange(0, years + 1)

    plt.figure()
    plt.plot(
        years_axis,
        100 * ((rev_paths / revenue0) ** (1 / np.maximum(years_axis, 1).reshape(-1, 1)) - 1)
    )
    plt.title("Monte Carlo Revenue CAGR Paths")
    plt.xlabel("Year")
    plt.ylabel("Running CAGR (%)")
    plt.grid(True, alpha=0.3)

    plt.figure()
    plt.plot(
        years_axis,
        100 * ((cost_paths / (cogs0 + opex0)) ** (1 / np.maximum(years_axis, 1).reshape(-1, 1)) - 1)
    )
    plt.title("Monte Carlo Cost CAGR Paths")
    plt.xlabel("Year")
    plt.ylabel("Running CAGR (%)")
    plt.grid(True, alpha=0.3)

    plt.show()