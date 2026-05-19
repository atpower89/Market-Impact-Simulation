#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 2026

@author: Alex Power

PARAMETER RECOVERY FROM SYNTHETIC DATA — OPTION B (Simulation08)
=================================================================

Goal
----
Fit a linear regression / power-law decay model to both the manual and
LOB outputs of Simulation08, recovering estimates of the mechanical and
informational impact parameters (C, A, beta).

The key research question is which market mechanisms are responsible for
the shape of the recovered propagator kernel G_hat — specifically whether
the approximately linear decay and non-monotonicity produced by the
zero-intelligence LOB (versus the convex, monotonically decaying kernel of
the TIM) can be attributed to the absence of long-memory order flow
autocorrelation and strategic market intelligence.

    Δp_t = C * ε_t  +  Σ_{k=1}^{L} K_k * ε_{t-k}  +  η_t

    G(τ) = cumsum(K)  ≈  A * (τ+1)^{-beta}

Manual model
------------
`trade_sign` is already a single signed flow per step (composite of
informed and uninformed), so `prepare_regression` is applied directly.

LOB model
---------
The LOB does not produce a single trade sign per step.  Each step may
contain an informed market order, gap-trader order, and uninformed
buy/sell arrivals.  We construct a net signed flow per step:

    epsilon_net_t = informed_direction_t * is_informational_t
                    + n_uninformed_buy_t
                    - n_uninformed_sell_t

Inputs
------
    synthetic_data08_manual.csv   — from Simulation08.py
    synthetic_data08_LOB_SantaFe.csv

Outputs
-------
    Printed estimates of C_hat, A_hat, beta_hat for both models
    Plots: recovered kernel G_hat vs power-law fit, for both models
    recovered_params08.csv — table of true vs estimated parameters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# NET SIGNED FLOW FOR LOB OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def build_lob_epsilon(df_lob, method="sign"):
    """
    Construct a per-step signed flow regressor from LOB output columns.

    The LOB records separate buy/sell arrivals rather than a single trade
    sign, so we aggregate them here before passing to the regression.

    Parameters
    ----------
    df_lob  : pd.DataFrame from synthetic_data08_LOB_SantaFe.csv
    method  : str
        "sign" — collapse net (informed + uninformed) imbalance to
                 +1 / 0 / -1  (default).  Keeps regressor on the same
                 scale as the manual trade_sign.
        "raw"  — use the integer net (informed + uninformed) imbalance
                 directly.  Mixes order-flow scale into coefficients.

    Returns
    -------
    epsilon : np.ndarray, shape (T,)
    """
    informed = (df_lob["informed_direction"].values
                * df_lob["is_informational"].values)
    net = (informed
           + df_lob["n_uninformed_buy"].values
           - df_lob["n_uninformed_sell"].values)

    if method == "sign":
        return np.sign(net).astype(int)
    elif method == "raw":
        return net.astype(float)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'sign' or 'raw'.")


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_regression(price, epsilon, L):
    """
    Build the design matrix and response vector for the impact regression:

        Δp_t = C * ε_t  +  Σ_{k=1}^{L} K_k * ε_{t-k}  +  η_t

    The first column of X is ε_t (contemporaneous — informational impact).
    Columns 1..L are ε_{t-1}..ε_{t-L} (lagged — mechanical kernel).
    y is Δp_t = p_t - p_{t-1}.

    Parameters
    ----------
    price   : np.ndarray (T+1,)
    epsilon : np.ndarray (T+1,)  — aligned with price
    L       : int  lag window

    Returns
    -------
    X : np.ndarray (T-L, L+1)
    y : np.ndarray (T-L,)
    """
    returns = np.diff(price)
    eps     = epsilon[1:]
    T       = len(returns)

    X = np.zeros((T - L, L + 1))
    y = returns[L:]

    for t in tqdm(range(L, T), desc="  Building design matrix"):
        X[t - L, 0]  = eps[t]
        X[t - L, 1:] = eps[t - L:t][::-1]

    return X, y


def estimate_kernel(price, epsilon, L=100):
    """
    Fit the regression and return C_hat and the cumulative kernel G_hat.

    Returns
    -------
    C_hat  : float        — estimated contemporaneous (informational) impact
    G_hat  : np.ndarray   — estimated cumulative mechanical kernel, length L
    K_hat  : np.ndarray   — raw lag coefficients before cumsum, length L
    r2     : float        — in-sample R²
    """
    X, y = prepare_regression(price, epsilon, L)

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)

    coef  = reg.coef_
    C_hat = coef[0]
    K_hat = coef[1:]
    G_hat = np.cumsum(K_hat)
    r2    = reg.score(X, y)

    return C_hat, G_hat, K_hat, r2


# ─────────────────────────────────────────────────────────────────────────────
# POWER-LAW FIT
# ─────────────────────────────────────────────────────────────────────────────

def powerlaw(tau, A, beta):
    return A * (tau + 1) ** (-beta)


def fit_powerlaw(G_hat):
    """
    Fit G_hat ≈ A * (τ+1)^{-beta} by nonlinear least squares.

    Returns A_hat, beta_hat, and the fitted curve for plotting.
    """
    tau  = np.arange(1, len(G_hat) + 1)
    popt, _ = curve_fit(
        powerlaw, tau, G_hat,
        p0=[0.05, 0.5],
        bounds=(0, np.inf),
        maxfev=10000
    )
    A_hat, beta_hat = popt
    G_fit = powerlaw(tau, A_hat, beta_hat)
    return A_hat, beta_hat, G_fit


# ─────────────────────────────────────────────────────────────────────────────
# PRICE RESPONSE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def price_response_function(price, epsilon, max_lag=100):
    """
    Non-parametric price response function:

        R(l) = < (p_{t+l} - p_t) * epsilon_t >

    For each lag l this is the average price change l steps after a signed
    trade, averaged over all t.  No model is assumed.

    Relationship to the regression cumulative kernel G_hat:

        i.i.d. flow:            R(l) ≈ <ε²> * G(l)
        Autocorrelated flow:    R(l) > <ε²> * G(l)

    For the baseline zero-intelligence LOB (no meta-orders), order flow is
    approximately i.i.d. and R(l) should track G_hat closely.  A persistent
    gap between the two curves at long lags would indicate residual order
    flow autocorrelation beyond what the regression captures.

    Parameters
    ----------
    price   : np.ndarray (T,)
    epsilon : np.ndarray (T,)  — signed flow, aligned with price
    max_lag : int  maximum lag (should match L used in regression)

    Returns
    -------
    R : np.ndarray (max_lag,)
    """
    T = len(price)
    R = np.zeros(max_lag)
    for l in tqdm(range(1, max_lag + 1), desc="  Computing response function"):
        R[l - 1] = np.mean((price[l:] - price[:T - l]) * epsilon[:T - l])
    return R


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_kernel(G_hat, G_fit, K_hat, label, C_hat, A_hat, beta_hat, r2,
                true_params=None, R=None):
    """
    Two-panel plot:
      Left  — raw lag coefficients K_hat
      Right — cumulative kernel G_hat with power-law overlay and optional R(l)
    """
    tau = np.arange(1, len(G_hat) + 1)

    fig, (ax_k, ax_g) = plt.subplots(1, 2, figsize=(12, 4))

    ax_k.bar(tau, K_hat, width=1.0, alpha=0.6, color="tab:blue")
    ax_k.axhline(0, linewidth=0.8, color="black", alpha=0.5)
    ax_k.set_title(f"Lag Coefficients K̂  —  {label}")
    ax_k.set_xlabel("Lag τ")
    ax_k.set_ylabel("K̂(τ)")
    ax_k.grid(alpha=0.3)

    ax_g.plot(tau, G_hat, linewidth=1.5, alpha=0.8, label="G_hat (regression)")
    ax_g.plot(tau, G_fit, linewidth=2,   linestyle="--", color="tab:red",
              label=f"Power-law fit: A={A_hat:.4f}, β={beta_hat:.4f}")

    if true_params is not None:
        A_true, beta_true = true_params
        G_true = powerlaw(tau, A_true, beta_true)
        ax_g.plot(tau, G_true, linewidth=1.5, linestyle=":", color="tab:green",
                  label=f"True kernel: A={A_true}, β={beta_true}")

    if R is not None:
        ax_g.plot(tau, R[:len(G_hat)], linewidth=1.5, linestyle=(0, (3, 1, 1, 1)),
                  color="tab:purple", alpha=0.9,
                  label="R(l)  [non-parametric response]")

    ax_g.set_title(f"Cumulative Kernel G_hat  —  {label}")
    ax_g.set_xlabel("Lag τ")
    ax_g.set_ylabel("G_hat(τ)")
    ax_g.grid(alpha=0.3)
    ax_g.legend(fontsize=9)

    textstr = f"Ĉ = {C_hat:.4f}\nÂ = {A_hat:.4f}\nβ̂ = {beta_hat:.4f}\nR² = {r2:.4f}"
    ax_g.text(0.97, 0.97, textstr, transform=ax_g.transAxes,
              fontsize=10, va="top", ha="right",
              bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    plt.show()


def plot_response_comparison(G_hat_m, R_m, G_hat_l, R_l):
    """
    Side-by-side comparison of G_hat (parametric regression) and R(l)
    (non-parametric response function) for both models.

    For i.i.d. order flow the two curves should overlap closely (scaled by
    <ε²>).  A gap between R(l) and G_hat at long lags indicates residual
    order flow autocorrelation contributing to observed price impact beyond
    what the mechanical kernel alone accounts for.
    """
    tau = np.arange(1, len(G_hat_m) + 1)

    fig, (ax_m, ax_l) = plt.subplots(1, 2, figsize=(12, 4))

    for ax, G_hat, R, title in [
        (ax_m, G_hat_m, R_m, "Manual model"),
        (ax_l, G_hat_l, R_l, "Santa Fe LOB"),
    ]:
        ax.plot(tau, G_hat, linewidth=1.5,
                label="G_hat  (parametric regression)")
        ax.plot(tau, R[:len(G_hat)], linewidth=1.5, linestyle=(0, (3, 1, 1, 1)),
                color="tab:purple", alpha=0.9,
                label="R(l)  (non-parametric response)")
        ax.axhline(0, linewidth=0.8, color="black", alpha=0.4)
        ax.set_title(f"{title} — G_hat vs R(l)")
        ax.set_xlabel("Lag τ")
        ax.set_ylabel("Cumulative impact")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle(
        "Parametric G_hat vs Non-Parametric R(l)\n"
        "Gap at long lags = residual order flow autocorrelation",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    L = 100

    A_true    = 0.025
    beta_true = 0.6
    C_true    = 0.05

    print("Loading data...")
    df_manual = pd.read_csv("synthetic_data08_manual.csv")
    df_lob    = pd.read_csv("synthetic_data08_LOB_SantaFe.csv")

    # ── Manual model recovery ─────────────────────────────────────────────
    print("\nManual model — estimating kernel...")
    price_m   = df_manual["price"].values
    epsilon_m = df_manual["trade_sign"].values

    C_hat_m, G_hat_m, K_hat_m, r2_m = estimate_kernel(price_m, epsilon_m, L)
    A_hat_m, beta_hat_m, G_fit_m     = fit_powerlaw(G_hat_m)

    print(f"  C_hat : {C_hat_m:.4f}   (true mean C ≈ {C_true})")
    print(f"  A_hat : {A_hat_m:.4f}   (true A = {A_true})")
    print(f"  beta  : {beta_hat_m:.4f}   (true beta = {beta_true})")
    print(f"  R²    : {r2_m:.4f}")

    print("\nManual model — computing response function...")
    R_m = price_response_function(price_m, epsilon_m, max_lag=L)

    plot_kernel(G_hat_m, G_fit_m, K_hat_m,
                label="Manual model",
                C_hat=C_hat_m, A_hat=A_hat_m, beta_hat=beta_hat_m, r2=r2_m,
                true_params=(A_true, beta_true), R=R_m)

    # ── LOB model recovery ────────────────────────────────────────────────
    print("\nLOB model — estimating kernel...")
    price_lob   = df_lob["price"].values
    epsilon_lob = build_lob_epsilon(df_lob, method="sign")

    print(f"  Fraction of zero steps : {(epsilon_lob == 0).mean():.3f}")
    print(f"  Mean epsilon (≈ 0)     : {epsilon_lob.mean():.4f}")

    C_hat_l, G_hat_l, K_hat_l, r2_l = estimate_kernel(price_lob, epsilon_lob, L)
    A_hat_l, beta_hat_l, G_fit_l     = fit_powerlaw(G_hat_l)

    print(f"  C_hat : {C_hat_l:.4f}")
    print(f"  A_hat : {A_hat_l:.4f}")
    print(f"  beta  : {beta_hat_l:.4f}")
    print(f"  R²    : {r2_l:.4f}")

    print("\nLOB model — computing response function...")
    R_l = price_response_function(price_lob, epsilon_lob, max_lag=L)

    plot_kernel(G_hat_l, G_fit_l, K_hat_l,
                label="Santa Fe LOB (baseline)",
                C_hat=C_hat_l, A_hat=A_hat_l, beta_hat=beta_hat_l, r2=r2_l,
                true_params=None, R=R_l)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"{'':20s}  {'C_hat':>8s}  {'A_hat':>8s}  {'beta_hat':>8s}  {'R²':>8s}")
    print(f"{'True (manual)':20s}  {C_true:>8.4f}  {A_true:>8.4f}  {beta_true:>8.4f}  {'—':>8s}")
    print(f"{'Manual recovered':20s}  {C_hat_m:>8.4f}  {A_hat_m:>8.4f}  {beta_hat_m:>8.4f}  {r2_m:>8.4f}")
    print(f"{'LOB recovered':20s}  {C_hat_l:>8.4f}  {A_hat_l:>8.4f}  {beta_hat_l:>8.4f}  {r2_l:>8.4f}")

    results = pd.DataFrame({
        "model":    ["true_manual", "manual_recovered", "LOB_recovered"],
        "C":        [C_true,    C_hat_m,    C_hat_l],
        "A":        [A_true,    A_hat_m,    A_hat_l],
        "beta":     [beta_true, beta_hat_m, beta_hat_l],
        "R2":       [None,      r2_m,       r2_l],
    })
    results.to_csv("recovered_params08.csv", index=False)
    print("\nSaved to recovered_params08.csv")

    # ── Response function comparison ──────────────────────────────────────
    plot_response_comparison(G_hat_m, R_m, G_hat_l, R_l)
