#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alex Power

OPTION B: SHARED FUNDAMENTAL PROCESS — ENDOGENOUS ORDER FLOW
=============================================================

Architecture
------------
The unit of comparison shifts from order flow to the fundamental value
process.  Both models receive the same sequence of:

    (I_t, C_t, sigma_t)   — informational flags, shock sizes, volatility

but each generates its own order flow independently.

Santa Fe LOB
    A full endogenous agent loop at each time step:
      1. Fundamental update — fundamental_price += delta_F[t].
      2. Limit providers  — Poisson(lam * depth_levels) new limit orders
                            placed with geometric depth distribution.
      3. Cancellations    — each resting unit cancels with prob delta.
      4. Informed agent   — if I_t=1, submits one market order in the
                            direction of the fundamental signal.
      5. Gap-trader       — if |fundamental - mid| > gap_threshold,
                            submits a market order of size
                            gap_intensity * |gap| in the correcting
                            direction (Kyle 1985 style endogenous
                            price discovery).
      6. Uninformed agents — Poisson(mu_u) buys + Poisson(mu_u) sells
                            arrive as market orders each step.
      7. Mid-price update — driven by order flow only; no exogenous
                            recentring term.

Manual model
    Synthesises a statistically equivalent order flow from the same
    Santa Fe arrival parameters, then applies the power-law impact
    kernel and cumulative informational drift as before:

        P(t) = P(0)  +  p_mech(t)  +  p_info(t)  +  noise(t)

    The noise-trader signed impacts are drawn from the same Poisson
    arrival process used by the LOB (mu_u per side), so both models
    are parametrically coupled to the same market microstructure even
    though the LOB is fully endogenous.

Comparison question
    "Given the same fundamental value process, which market mechanisms
    are responsible for the shape of the recovered propagator kernel?"

Parameters shared across both models
    mu_u    uninformed market-order arrival rate (per side, per step)
    lam     limit-order placement rate (per level, per step)
    delta   cancellation probability (per resting unit, per step)
    p_info  probability a time step has an informed trader present
    C, sigma_C  lognormal parameters for informational shock size
    alpha_vol, beta_vol, omega  GARCH volatility parameters

Changes from Simulation07
    - Meta-order channel removed for baseline diagnostics (will be
      reintroduced in a later iteration once baseline behaviour is
      understood).
    - Docstrings updated to reflect iteration 08 diagnostic focus.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import random
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# FUNDAMENTAL PROCESS
# ─────────────────────────────────────────────────────────────────────────────

def generate_fundamental_process(n_minutes, trades_per_minute,
                                 p_info, C, sigma_C,
                                 alpha_vol, beta_vol, omega,
                                 sigma_noise, rho,
                                 seed=None):
    """
    Generate the shared fundamental value process that both models receive.

    This is the minimal shared state under Option B: which time steps
    carry informational content, the direction and size of those shocks,
    and what the prevailing volatility is.  Order flow is NOT generated here.

    The informed direction sequence is produced here once (AR(1) with
    persistence rho) so that both models consume the identical direction
    on every informed step, guaranteeing they track the same fundamental
    price path.

    Parameters
    ----------
    n_minutes, trades_per_minute : simulation length and granularity
    p_info    : float  probability each step has an informed trader
    C         : float  mean informational shock size (lognormal)
    sigma_C   : float  std of log(shock size)
    alpha_vol, beta_vol, omega : GARCH(1,1) parameters for hourly vol
    sigma_noise : float  std of microstructure noise (added post-hoc)
    rho       : float  AR(1) persistence for informed sign direction
    seed      : int or None

    Returns
    -------
    dict with keys:
        T                   total number of time steps
        I                   (T,) int  — informational flag per step
        C_t                 (T,) float — informational shock size per step
        sigma               (T,) float — stochastic volatility per step
        noise               (T,) float — microstructure noise per step
        informed_direction  (T,) int  — AR(1) sign for each step (+/-1)
        delta_F             (T,) float — canonical fundamental shock:
                                         I_t * C_t * direction_t * sigma_t
    """
    rng = np.random.default_rng(seed)

    T               = n_minutes * trades_per_minute
    trades_per_hour = 60 * trades_per_minute
    n_hours         = int(np.ceil(T / trades_per_hour))

    I = rng.binomial(1, p_info, size=T)

    sigma2_hourly    = np.zeros(n_hours)
    sigma2_hourly[0] = omega / (1 - alpha_vol - beta_vol)
    z_hourly         = rng.normal(size=n_hours)

    for h in range(1, n_hours):
        sigma2_hourly[h] = (
            omega
            + alpha_vol * z_hourly[h-1]**2
            + beta_vol  * sigma2_hourly[h-1]
        )

    sigma2       = np.repeat(sigma2_hourly, trades_per_hour)[:T]
    sigma_series = np.sqrt(sigma2)

    mu_C = np.log(C) - 0.5 * sigma_C**2
    C_t  = rng.lognormal(mean=mu_C, sigma=sigma_C, size=T)

    x        = np.zeros(T)
    ar_noise = rng.normal(0, 1, size=T)
    for t in range(1, T):
        if I[t] == 1:
            x[t] = rho * x[t-1] + ar_noise[t]
    informed_direction = np.sign(x).astype(int)
    informed_direction[informed_direction == 0] = 1

    delta_F = I * C_t * informed_direction * sigma_series
    noise   = rng.normal(0, sigma_noise, size=T)

    return {
        "T":                  T,
        "I":                  I,
        "C_t":                C_t,
        "sigma":              sigma_series,
        "noise":              noise,
        "informed_direction": informed_direction,
        "delta_F":            delta_F,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def simulate_market_manual(fundamental, initial_price,
                           L, A, beta,
                           mu_u,
                           mean_trade_size, size_dispersion,
                           seed=None):
    """
    Manual (mechanical + informational) price impact model.

    The informed direction and canonical fundamental shock (delta_F) are
    consumed directly from the fundamental dict.  Only uninformed order
    flow is generated independently using the Santa Fe arrival rate mu_u.

    Price:
        P(t) = P(0) + p_mech(t) + p_info(t) + noise(t)

        p_mech(t) = Σ_{k=1}^{L}  A * k^{-beta} * epsilon_noise(t-k) * size(t-k)
        p_info(t) = cumsum( delta_F )

    Parameters
    ----------
    fundamental       : dict from generate_fundamental_process()
    initial_price     : float
    L                 : int    look-back window for mechanical kernel
    A                 : float  mechanical impact magnitude
    beta              : float  power-law decay exponent
    mu_u              : float  uninformed arrival rate (per side, per step)
    mean_trade_size   : float  lognormal mean trade size
    size_dispersion   : float  lognormal sigma for trade size
    seed              : int or None  (for uninformed flow only)
    """
    rng = np.random.default_rng(seed)

    T                  = fundamental["T"]
    I                  = fundamental["I"]
    sigma_series       = fundamental["sigma"]
    noise              = fundamental["noise"]
    informed_direction = fundamental["informed_direction"]
    delta_F            = fundamental["delta_F"]

    n_buy  = rng.poisson(mu_u, size=T)
    n_sell = rng.poisson(mu_u, size=T)
    epsilon_noise = np.sign(n_buy - n_sell).astype(int)
    epsilon       = np.where(I == 1, informed_direction, epsilon_noise)

    trade_size     = rng.lognormal(
        mean=np.log(mean_trade_size), sigma=size_dispersion, size=T
    )
    effective_size = trade_size * sigma_series

    signed_impact = epsilon_noise * effective_size
    lags = np.arange(T)
    G    = A * (lags + 1) ** (-beta)

    p_mech = np.zeros(T)
    for t in tqdm(range(T), desc="Manual model — mechanical kernel"):
        start     = max(0, t - L)
        past      = signed_impact[start:t][::-1]
        p_mech[t] = np.sum(past * G[:len(past)])

    p_info            = np.cumsum(delta_F)
    fundamental_price = initial_price + p_info
    price             = initial_price + p_mech + p_info + noise

    return pd.DataFrame({
        "time":              np.arange(T),
        "trade_sign":        epsilon,
        "trade_size":        effective_size,
        "price":             price,
        "p_mech_true":       p_mech,
        "p_info_true":       p_info,
        "fundamental_price": fundamental_price,
        "volatility":        sigma_series,
    })


# ─────────────────────────────────────────────────────────────────────────────
# SANTA FE LOB
# ─────────────────────────────────────────────────────────────────────────────

def santa_fe_profile(depth_levels, mu_u, lam, delta):
    """
    Stationary expected depth at each price level under Santa Fe dynamics:

        D(n) = (lam / delta) * [1 - exp(-delta * n / mu_u)]

    Index runs from n=1 to depth_levels (resting orders cannot sit at the
    exact mid-price), giving D(1) > 0 at the best quote.
    """
    n       = np.arange(1, depth_levels + 1, dtype=float)
    profile = (lam / delta) * (1.0 - np.exp(-delta * n / mu_u))
    return np.maximum(profile, 1e-3)


def simulate_market_LOB(fundamental, initial_price,
                        mu_u, lam, delta,
                        mean_trade_size, size_dispersion,
                        depth_levels=50, tick_size=0.01,
                        lam_placement_decay=0.3,
                        gap_threshold=0.0,
                        gap_intensity=10.0,
                        seed=None):
    """
    Full Santa Fe agent-based LOB simulation (baseline, no meta-orders).

    The informed direction and canonical fundamental shock (delta_F) are
    consumed from the fundamental dict.  Uninformed flow and limit order
    activity are drawn independently.

    Price discovery is endogenous: a gap-trader submits market orders
    proportional to (fundamental_price - mid_price) whenever the gap
    exceeds gap_threshold.  This replaces the exogenous discovery_speed
    term and follows the spirit of Kyle (1985) informed demand.

    At each time step:

    1. FUNDAMENTAL UPDATE
       fundamental_price += delta_F[t]

    2. LIMIT ORDER PLACEMENT
       N_lim ~ Poisson(lam * depth_levels) orders, geometric level dist.

    3. CANCELLATIONS
       Binomial(depth, delta) cancellations per level.

    4. INFORMED MARKET ORDER
       If I_t = 1, one market order in fundamental["informed_direction"][t].

    5. GAP-TRADING
       If |fundamental_price - mid_price| > gap_threshold, submit a
       market order of size gap_intensity * |gap| in the correcting
       direction.  Anchors the LOB price to the fundamental through
       endogenous order flow rather than exogenous recentring.

    6. UNINFORMED MARKET ORDERS
       N_buy ~ Poisson(mu_u) and N_sell ~ Poisson(mu_u).

    7. MID-PRICE UPDATE (order-flow driven only).

    8. NOISE added post-hoc.

    Parameters
    ----------
    fundamental         : dict from generate_fundamental_process()
    initial_price       : float
    mu_u                : float  uninformed arrival rate (per side, per step)
    lam                 : float  limit-order placement rate (per level, per step)
    delta               : float  cancellation probability per resting unit
    mean_trade_size     : float  lognormal mean for market order sizes
    size_dispersion     : float  lognormal sigma for market order sizes
    depth_levels        : int    number of price levels each side
    tick_size           : float  price increment per level
    lam_placement_decay : float  geometric decay for limit order placement
    gap_threshold       : float  minimum |fundamental - mid| to trigger gap trade
    gap_intensity       : float  gap trade size = gap_intensity * |gap|;
                                 set so that gap_intensity * typical_gap ≈ mean_trade_size
    seed                : int or None
    """
    rng = np.random.default_rng(seed)

    T                  = fundamental["T"]
    I                  = fundamental["I"]
    sigma_series       = fundamental["sigma"]
    noise              = fundamental["noise"]
    informed_direction = fundamental["informed_direction"]
    delta_F            = fundamental["delta_F"]

    sf_profile = santa_fe_profile(depth_levels, mu_u, lam, delta)
    bid_book   = sf_profile.copy()
    ask_book   = sf_profile.copy()

    mid_price         = initial_price
    fundamental_price = initial_price

    mid_prices         = np.zeros(T)
    fundamental_prices = np.zeros(T)
    n_ub_rec           = np.zeros(T, dtype=int)
    n_us_rec           = np.zeros(T, dtype=int)
    tick_shifts        = np.zeros(T)

    max_size = 5.0 * (lam / delta)

    def execute_order(size, side, bid_book, ask_book):
        remaining   = min(size, max_size)
        price_shift = 0.0
        if side == 1:
            level = 0
            while remaining > 0 and level < depth_levels:
                if remaining >= ask_book[level]:
                    remaining       -= ask_book[level]
                    ask_book[level]  = 0.0
                    level           += 1
                    price_shift     += tick_size
                else:
                    ask_book[level] -= remaining
                    remaining        = 0.0
        else:
            level = 0
            while remaining > 0 and level < depth_levels:
                if remaining >= bid_book[level]:
                    remaining       -= bid_book[level]
                    bid_book[level]  = 0.0
                    level           += 1
                    price_shift     += tick_size
                else:
                    bid_book[level] -= remaining
                    remaining        = 0.0
        return price_shift, bid_book, ask_book

    for t in tqdm(range(T), desc="Santa Fe LOB — agent simulation"):

        sigma_t = sigma_series[t]

        # 1. Fundamental update
        fundamental_price += delta_F[t]

        # 2. Limit order placement
        n_lim = rng.poisson(lam * depth_levels)
        if n_lim > 0:
            sides  = rng.integers(0, 2, size=n_lim)
            levels = rng.geometric(lam_placement_decay, size=n_lim) - 1
            levels = np.clip(levels, 0, depth_levels - 1)
            for i in range(n_lim):
                if sides[i] == 0:
                    bid_book[levels[i]] += 1.0
                else:
                    ask_book[levels[i]] += 1.0

        # 3. Cancellations
        for level in range(depth_levels):
            if bid_book[level] > 0:
                cancelled = rng.binomial(int(bid_book[level]), delta)
                bid_book[level] = max(0.0, bid_book[level] - cancelled)
            if ask_book[level] > 0:
                cancelled = rng.binomial(int(ask_book[level]), delta)
                ask_book[level] = max(0.0, ask_book[level] - cancelled)

        # 4. Informed market order
        total_price_shift = 0.0
        if I[t] == 1:
            inf_size = rng.lognormal(
                mean=np.log(mean_trade_size), sigma=size_dispersion
            ) * sigma_t
            ps, bid_book, ask_book = execute_order(
                inf_size, informed_direction[t], bid_book, ask_book
            )
            total_price_shift += informed_direction[t] * ps

        # 5. Gap-trading: endogenous price discovery
        gap = fundamental_price - mid_price
        if abs(gap) > gap_threshold:
            gap_dir  = 1 if gap > 0 else -1
            gap_size = gap_intensity * abs(gap)
            ps, bid_book, ask_book = execute_order(gap_size, gap_dir, bid_book, ask_book)
            total_price_shift += gap_dir * ps

        # 6. Uninformed market orders
        n_buy  = rng.poisson(mu_u)
        n_sell = rng.poisson(mu_u)
        n_ub_rec[t] = n_buy
        n_us_rec[t] = n_sell

        uninformed_orders = (
            [(1,  rng.lognormal(mean=np.log(mean_trade_size),
                                sigma=size_dispersion) * sigma_t)
             for _ in range(n_buy)]
            +
            [(-1, rng.lognormal(mean=np.log(mean_trade_size),
                                sigma=size_dispersion) * sigma_t)
             for _ in range(n_sell)]
        )
        rng.shuffle(uninformed_orders)

        for side, size in uninformed_orders:
            ps, bid_book, ask_book = execute_order(size, side, bid_book, ask_book)
            total_price_shift += side * ps

        # 7. Mid-price update (order-flow driven only)
        tick_shifts[t] = total_price_shift
        mid_price += total_price_shift / 2.0

        mid_prices[t]         = mid_price
        fundamental_prices[t] = fundamental_price

    price = mid_prices + noise

    return pd.DataFrame({
        "time":               np.arange(T),
        "price":              price,
        "mid_price":          mid_prices,
        "fundamental_price":  fundamental_prices,
        "is_informational":   I,
        "informed_direction": informed_direction * I,
        "n_uninformed_buy":   n_ub_rec,
        "n_uninformed_sell":  n_us_rec,
        "tick_shift":         tick_shifts,
    })


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_returns_panel(returns_mech, returns_LOB, title_prefix, xlabel, bins=60):
    """Shared time-series + histogram panel for minute or daily returns."""
    fig = plt.figure(figsize=(12, 6))
    gs  = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
    ax_ret      = fig.add_subplot(gs[0, 0])
    ax_hist_ret = fig.add_subplot(gs[0, 1], sharey=ax_ret)

    ax_ret.plot(returns_mech, linewidth=1,           label="Manual (synthetic flow)")
    ax_ret.plot(returns_LOB,  linewidth=1, alpha=0.8, label="Santa Fe LOB")
    ax_ret.set_title(f"{title_prefix} Returns (Option B)")
    ax_ret.set_xlabel(xlabel)
    ax_ret.set_ylabel("Log Return")
    ax_ret.set_xlim(0, len(returns_mech))
    ylim = max(np.max(np.abs(returns_mech)), np.max(np.abs(returns_LOB)))
    ax_ret.set_ylim(-ylim, ylim)
    n = len(returns_mech)
    ax_ret.xaxis.set_major_locator(MultipleLocator(max(1, n // 5)))
    ax_ret.xaxis.set_minor_locator(MultipleLocator(max(1, n // 20)))
    ax_ret.grid(which="major", axis="y", linewidth=1.0, alpha=0.6)
    ax_ret.grid(which="minor", axis="y", linewidth=0.4, alpha=0.4)
    ax_ret.grid(which="major", axis="x", linewidth=0.6, alpha=0.5)
    ax_ret.grid(which="minor", axis="x", linewidth=0.4, alpha=0.4)
    ax_ret.legend()

    ax_hist_ret.hist(returns_mech, bins=bins, orientation="horizontal",
                     density=True, alpha=0.7)
    ax_hist_ret.hist(returns_LOB,  bins=bins, orientation="horizontal",
                     density=True, alpha=0.7)
    ax_hist_ret.set_title("Return\nDistribution")
    ax_hist_ret.set_xlabel("Density")
    ax_hist_ret.set_ylim(ax_ret.get_ylim())
    ax_hist_ret.tick_params(axis="y", labelleft=False)
    plt.tight_layout()
    plt.show()


def _plot_distribution(returns_mech, returns_LOB, title, bins=100):
    """Distribution plot with Gaussian overlay and stats box."""
    mu_m  = np.mean(returns_mech); sig_m = np.std(returns_mech)
    mu_l  = np.mean(returns_LOB);  sig_l = np.std(returns_LOB)
    kurt_m = ((returns_mech - mu_m)**4).mean() / sig_m**4
    kurt_l = ((returns_LOB  - mu_l)**4).mean() / sig_l**4

    x_m = np.linspace(mu_m - 5*sig_m, mu_m + 5*sig_m, 1000)
    x_l = np.linspace(mu_l - 5*sig_l, mu_l + 5*sig_l, 1000)
    g_m = np.exp(-(x_m - mu_m)**2 / (2*sig_m**2)) / (sig_m * np.sqrt(2*np.pi))
    g_l = np.exp(-(x_l - mu_l)**2 / (2*sig_l**2)) / (sig_l * np.sqrt(2*np.pi))

    fig, ax = plt.subplots(figsize=(8, 5))
    low  = min(np.percentile(returns_mech, 0.05), np.percentile(returns_LOB, 0.05))
    high = max(np.percentile(returns_mech, 99.95), np.percentile(returns_LOB, 99.95))
    ax.set_xlim(low, high)
    ax.hist(returns_mech, bins=bins, density=True, alpha=0.6,
            label="Manual (synthetic flow)")
    ax.hist(returns_LOB,  bins=bins, density=True, alpha=0.6,
            label="Santa Fe LOB")
    ax.plot(x_m, g_m, linewidth=2, label="Gaussian fit — Manual")
    ax.plot(x_l, g_l, linewidth=2, label="Gaussian fit — LOB")
    ax.set_title(title)
    ax.set_xlabel("Log Return"); ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.text(0.98, 0.98,
            f"μ_m={mu_m:.2e}\nσ_m={sig_m:.2e}\nKurt_m={kurt_m:.2f}\n"
            f"μ_L={mu_l:.2e}\nσ_L={sig_l:.2e}\nKurt_L={kurt_l:.2f}",
            transform=ax.transAxes, fontsize=10,
            va="top", ha="right", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

# PARAMETERS ──────────────────────────────────────────────────────────────────
    minutes_per_day   = 390
    num_days          = 252
    trades_per_minute = 10

    n_minutes      = minutes_per_day * num_days
    trades_per_day = minutes_per_day * trades_per_minute

    p_info      = 0.1
    C           = 0.05
    sigma_C     = 0.05
    alpha_vol   = 0.19
    beta_vol    = 0.80
    omega       = 0.01
    sigma_noise = 0.01

    initial_price = 100.0

    L    = 5000
    A    = 0.025
    beta = 0.6
    rho  = 0.7
    mean_trade_size = 1.0
    size_dispersion = 0.5

    mu_u  = 2.0
    lam = 0.4
    delta = 0.05

    depth_levels        = 50
    tick_size           = 0.01
    lam_placement_decay = 0.1
    gap_threshold       = 0.0
    gap_intensity       = 0.3

    seed = random.randint(0, 100)
    print(f"Seed: {seed}")

# SIMULATION ──────────────────────────────────────────────────────────────────
    print("\nGenerating fundamental process...")
    fundamental = generate_fundamental_process(
        n_minutes=n_minutes, trades_per_minute=trades_per_minute,
        p_info=p_info, C=C, sigma_C=sigma_C,
        alpha_vol=alpha_vol, beta_vol=beta_vol, omega=omega,
        sigma_noise=sigma_noise, rho=rho, seed=seed
    )

    df_manual = simulate_market_manual(
        fundamental=fundamental, initial_price=initial_price,
        L=L, A=A, beta=beta, mu_u=mu_u,
        mean_trade_size=mean_trade_size, size_dispersion=size_dispersion,
        seed=seed + 1
    )

    df_LOB = simulate_market_LOB(
        fundamental=fundamental, initial_price=initial_price,
        mu_u=mu_u, lam=lam, delta=delta,
        mean_trade_size=mean_trade_size, size_dispersion=size_dispersion,
        depth_levels=depth_levels, tick_size=tick_size,
        lam_placement_decay=lam_placement_decay,
        gap_threshold=gap_threshold,
        gap_intensity=gap_intensity,
        seed=seed + 2
    )

# PRICE PATH ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    t_day = df_manual["time"].values / trades_per_day

    ax.plot(t_day, df_manual["price"],             label="Manual — observed price",    linewidth=1)
    ax.plot(t_day, df_LOB["price"],                label="LOB — observed price",       linewidth=1, alpha=0.8)
    ax.plot(t_day, df_manual["fundamental_price"], label="Fundamental (shared)",       linewidth=1.5,
            linestyle="--", color="black")
    ax.plot(t_day, df_LOB["fundamental_price"],    label="LOB fundamental (internal)", linewidth=1,
            linestyle=":", color="grey")

    ax.set_title("Simulated Asset Price — Shared Fundamental Process")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(which="major", linestyle="-",  linewidth=0.6, alpha=0.7)
    ax.grid(which="minor", linestyle="--", linewidth=0.4, alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()

# PRICE DISCOVERY: DEVIATION FROM FUNDAMENTAL ─────────────────────────────────
    dev_manual = df_manual["price"].values - df_manual["fundamental_price"].values
    dev_LOB    = df_LOB["price"].values    - df_LOB["fundamental_price"].values

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t_day, dev_manual, linewidth=1, color="tab:blue")
    axes[0].axhline(0, linestyle="--", linewidth=0.8, color="black", alpha=0.5)
    axes[0].set_ylabel("Price − Fundamental")
    axes[0].set_title("Price Discovery: Deviation from Fundamental — Manual Model")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_day, dev_LOB, linewidth=1, color="tab:orange")
    axes[1].axhline(0, linestyle="--", linewidth=0.8, color="black", alpha=0.5)
    axes[1].set_ylabel("Price − Fundamental")
    axes[1].set_xlabel("Trading Day")
    axes[1].set_title("Price Discovery: Deviation from Fundamental — Santa Fe LOB")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nPrice Discovery Statistics (deviation from fundamental)")
    print(f"  Manual — mean |dev|: {np.mean(np.abs(dev_manual)):.4f}  "
          f"std: {np.std(dev_manual):.4f}")
    print(f"  LOB    — mean |dev|: {np.mean(np.abs(dev_LOB)):.4f}  "
          f"std: {np.std(dev_LOB):.4f}")

# HIDDEN COMPONENTS — MANUAL ──────────────────────────────────────────────────
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_price.plot(t_day, df_manual["p_mech_true"], label="Mechanical (temporary)")
    ax_price.plot(t_day, df_manual["p_info_true"], label="Informational (permanent)")
    ax_price.set_ylabel("Contribution to Price")
    ax_price.set_xlabel("Trading Day")
    ax_price.set_title("Hidden Price Components — Manual Model (Option B)")
    ax_price.grid(alpha=0.3)
    ax_price.legend()

    ax_vol.plot(t_day, df_manual["volatility"], color="tab:red", linewidth=1.5)
    ax_vol.set_ylabel("Volatility (σ_t)")
    ax_vol.set_title("Stochastic Volatility (shared)")
    ax_vol.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# MINUTE RETURNS ──────────────────────────────────────────────────────────────
    lp_m = np.log(df_manual["price"].values)
    lp_l = np.log(df_LOB["price"].values)

    n_min   = len(lp_m) // trades_per_minute
    lp_m_r  = lp_m[:n_min * trades_per_minute].reshape(n_min, trades_per_minute)
    lp_l_r  = lp_l[:n_min * trades_per_minute].reshape(n_min, trades_per_minute)
    mr_mech = lp_m_r[:, -1] - lp_m_r[:, 0]
    mr_LOB  = lp_l_r[:, -1] - lp_l_r[:, 0]

    _plot_returns_panel(mr_mech, mr_LOB, "Minute", "Time (mins)")
    _plot_distribution(mr_mech, mr_LOB, "Distribution of Minute Returns (Option B)")

# DAILY RETURNS ───────────────────────────────────────────────────────────────
    ret_m = np.diff(lp_m)
    ret_l = np.diff(lp_l)

    n_days = len(ret_m) // trades_per_day
    dr_m   = ret_m[:n_days * trades_per_day].reshape(n_days, trades_per_day).sum(axis=1)
    dr_l   = ret_l[:n_days * trades_per_day].reshape(n_days, trades_per_day).sum(axis=1)

    _plot_returns_panel(dr_m, dr_l, "Daily", "Time (days)", bins=25)
    _plot_distribution(dr_m, dr_l, "Distribution of Daily Returns (Option B)", bins=40)

# SUMMARY AND DATA EXPORT ─────────────────────────────────────────────────────
    df_manual.to_csv("synthetic_data08_manual.csv", index=False)
    df_LOB.to_csv("synthetic_data08_LOB_SantaFe.csv", index=False)

    print("\nManual Model Statistics")
    print(f"  Daily mean : {dr_m.mean():.4e}")
    print(f"  Daily std  : {dr_m.std():.4e}")
    print(f"  Max daily  : {dr_m.max():.4e}")
    print(f"  Min daily  : {dr_m.min():.4e}")

    print("\nSanta Fe LOB Statistics")
    print(f"  Daily mean : {dr_l.mean():.4e}")
    print(f"  Daily std  : {dr_l.std():.4e}")
    print(f"  Max daily  : {dr_l.max():.4e}")
    print(f"  Min daily  : {dr_l.min():.4e}")
