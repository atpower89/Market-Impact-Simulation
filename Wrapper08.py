#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 2026

@author: Alex Power

WRAPPER SCRIPT TO GENERATE AND EVALUATE SYNTHETIC DATA
Compatible with Simulation08.py and Recovery08.py

Runs the Option B simulation (shared fundamental process, baseline LOB)
and immediately applies parameter recovery, appending results to a CSV
without producing any plots.

model parameter controls which simulation output is recovered:
    "TIM"  — manual (Transient Impact Model) output
    "LOB"  — Santa Fe LOB output
    "both" — run recovery on both and save a row for each
"""

import numpy as np
import pandas as pd
import random
import os

from Simulation08 import (
    generate_fundamental_process,
    simulate_market_manual,
    simulate_market_LOB,
)
from Recovery08 import (
    build_lob_epsilon,
    estimate_kernel,
    fit_powerlaw,
)


def run_experiment(
    # Simulation length
    n_minutes, trades_per_minute,
    # TIM parameters
    L, A, beta,
    # Fundamental process parameters
    p_info, C, sigma_C,
    alpha_vol, beta_vol, omega,
    sigma_noise,
    # Shared microstructure parameters
    mu_u, lam, delta, rho,
    # Trade size parameters
    mean_trade_size, size_dispersion,
    # Initial price
    initial_price=100.0,
    # LOB-specific parameters
    depth_levels=50, tick_size=0.01,
    lam_placement_decay=0.3,
    gap_threshold=0.0,
    gap_intensity=10.0,
    # Recovery lag window
    recovery_L=100,
    # Which model(s) to recover
    model="both",
    # Random seeds
    seed=None,
):
    """
    Run the Option B simulation (baseline LOB, no meta-orders) and recover
    parameters, without plots.

    Parameters
    ----------
    model : str
        "TIM"  — simulate and recover manual model only
        "LOB"  — simulate and recover Santa Fe LOB only
        "both" — simulate both, recover both, return two result dicts

    Returns
    -------
    results : list of dicts
        One dict per recovered model, each containing true and estimated
        parameters plus recovery diagnostics.
    """

    print(f"\nGenerating fundamental process (seed={seed})...")
    fundamental = generate_fundamental_process(
        n_minutes=n_minutes, trades_per_minute=trades_per_minute,
        p_info=p_info, C=C, sigma_C=sigma_C,
        alpha_vol=alpha_vol, beta_vol=beta_vol, omega=omega,
        sigma_noise=sigma_noise, rho=rho,
        seed=seed,
    )

    rows = []

    # ── TIM / manual model ────────────────────────────────────────────────
    if model in ("TIM", "both"):
        print("Simulating manual (TIM) model...")
        df_manual = simulate_market_manual(
            fundamental=fundamental, initial_price=initial_price,
            L=L, A=A, beta=beta,
            mu_u=mu_u,
            mean_trade_size=mean_trade_size, size_dispersion=size_dispersion,
            seed=seed + 1 if seed is not None else None,
        )

        print("Recovering TIM parameters...")
        C_hat_m, G_hat_m, _, r2_m = estimate_kernel(
            df_manual["price"].values,
            df_manual["trade_sign"].values,
            L=recovery_L,
        )
        A_hat_m, beta_hat_m, _ = fit_powerlaw(G_hat_m)

        rows.append({
            "model":     "TIM",
            "seed":      seed,
            "A_true":    A,
            "beta_true": beta,
            "C_true":    C,
            "A_est":     A_hat_m,
            "beta_est":  beta_hat_m,
            "C_est":     C_hat_m,
            "R2":        r2_m,
            "mu_u":      mu_u,
            "lam":       lam,
            "delta":     delta,
            "p_info":    p_info,
        })

    # ── Santa Fe LOB ──────────────────────────────────────────────────────
    if model in ("LOB", "both"):
        print("Simulating Santa Fe LOB...")
        df_lob = simulate_market_LOB(
            fundamental=fundamental, initial_price=initial_price,
            mu_u=mu_u, lam=lam, delta=delta,
            mean_trade_size=mean_trade_size, size_dispersion=size_dispersion,
            depth_levels=depth_levels, tick_size=tick_size,
            lam_placement_decay=lam_placement_decay,
            gap_threshold=gap_threshold,
            gap_intensity=gap_intensity,
            seed=seed + 2 if seed is not None else None,
        )

        print("Recovering LOB parameters...")
        epsilon_lob = build_lob_epsilon(df_lob, method="sign")
        C_hat_l, G_hat_l, _, r2_l = estimate_kernel(
            df_lob["price"].values,
            epsilon_lob,
            L=recovery_L,
        )
        A_hat_l, beta_hat_l, _ = fit_powerlaw(G_hat_l)

        rows.append({
            "model":     "LOB",
            "seed":      seed,
            "A_true":    None,
            "beta_true": None,
            "C_true":    C,
            "A_est":     A_hat_l,
            "beta_est":  beta_hat_l,
            "C_est":     C_hat_l,
            "R2":        r2_l,
            "mu_u":      mu_u,
            "lam":       lam,
            "delta":     delta,
            "p_info":    p_info,
        })

    if model not in ("TIM", "LOB", "both"):
        raise ValueError("model must be 'TIM', 'LOB', or 'both'")

    return rows


if __name__ == "__main__":

# PARAMETERS ──────────────────────────────────────────────────────────────────
    minutes_per_day   = 390
    num_days          = 252
    trades_per_minute = 10
    n_minutes         = minutes_per_day * num_days

    L    = 5000
    A    = 0.025
    beta = 0.6

    p_info      = 0.1
    C           = 0.05
    sigma_C     = 0.05
    alpha_vol   = 0.19
    beta_vol    = 0.80
    omega       = 0.01
    sigma_noise = 0.01

    mu_u  = 2.0
    lam   = 0.5
    delta = 0.05
    rho   = 0.7

    mean_trade_size = 1.0
    size_dispersion = 0.5

    initial_price = 100.0

    depth_levels        = 50
    tick_size           = 0.01
    lam_placement_decay = 0.3
    gap_threshold       = 0.0
    gap_intensity       = 10.0

    recovery_L = 100

    seed = random.randint(0, 100)

    output_file = "Results08.csv"
    file_exists = os.path.isfile(output_file)

# RUN ─────────────────────────────────────────────────────────────────────────
    rows = run_experiment(
        n_minutes=n_minutes, trades_per_minute=trades_per_minute,
        L=L, A=A, beta=beta,
        p_info=p_info, C=C, sigma_C=sigma_C,
        alpha_vol=alpha_vol, beta_vol=beta_vol, omega=omega,
        sigma_noise=sigma_noise,
        mu_u=mu_u, lam=lam, delta=delta, rho=rho,
        mean_trade_size=mean_trade_size, size_dispersion=size_dispersion,
        initial_price=initial_price,
        depth_levels=depth_levels, tick_size=tick_size,
        lam_placement_decay=lam_placement_decay,
        gap_threshold=gap_threshold,
        gap_intensity=gap_intensity,
        recovery_L=recovery_L,
        model="both",
        seed=seed,
    )

# PRINT AND SAVE ──────────────────────────────────────────────────────────────
    for row in rows:
        print(f"\nResults ({row['model']})")
        for k, v in row.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(
        output_file,
        mode="a",
        header=not file_exists,
        index=False,
    )
    print(f"\nAppended {len(rows)} row(s) to {output_file}")
