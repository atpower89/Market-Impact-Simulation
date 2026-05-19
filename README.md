The Sources of Price Impact Kernel Shape: A Controlled Simulation
Author: Alex Power — Cornell University, Department of Chemical Engineering
Supervisor: Dr. Jeffrey Varner
Python version: 3.8

Overview
This repository accompanies the paper The Sources of Price Impact Kernel Shape: A Controlled Simulation (see Market_Impact_Simulation.pdf). The project investigates which market mechanisms are responsible for the shape of the price impact propagator kernel, using two controlled simulation environments:

Transient Impact Model (TIM) — a manual model in which the kernel is imposed directly by construction as a power-law decay.
Santa Fe Limit Order Book (LOB) — a zero-intelligence agent-based model in which price dynamics emerge endogenously from order book mechanics.
Both models are driven by a shared fundamental value process, so differences in their recovered kernels can be attributed to the price formation mechanism rather than to differences in information flow.

The key finding is that the zero-intelligence LOB produces a qualitatively different kernel from the TIM: a brief initial non-monotonicity at short lags (book momentum) followed by an approximately linear decay to zero. This violates two of Gatheral's (2010) no-dynamic-arbitrage conditions — non-increasing and convexity — both of which are attributed to the absence of market intelligence in the zero-intelligence framework.

Repository Structure
├── Simulation08.py                  # Core simulation: TIM and Santa Fe LOB
├── Recovery08.py                    # Parameter recovery: kernel estimation and power-law fitting
├── Wrapper08.py                     # Batch runner: runs simulation + recovery, appends results to CSV
├── Market_Impact_Simulation.pdf     # Full paper
└── Results08.csv                    # Output from Wrapper08.py (appended across runs)
File Descriptions
Simulation08.py
Implements both models under the shared fundamental process:

generate_fundamental_process() — generates the GARCH-driven fundamental value series, informational flags, and signed shock directions shared by both models.
simulate_market_manual() — manual TIM: applies a power-law kernel to synthetic uninformed order flow, adds cumulative informational drift.
simulate_market_LOB() — Santa Fe LOB: full agent-based simulation with Poisson limit order placement, binomial cancellations, informed market orders, and Kyle-style gap-trading for endogenous price discovery.
Running this file directly produces price path, price discovery, return distribution, and hidden component plots.

Recovery08.py
Fits the linear impact regression to both model outputs:

build_lob_epsilon() — constructs a net signed flow series from the LOB's per-step buy/sell records.
estimate_kernel() — fits the lag regression and returns the estimated cumulative kernel.
fit_powerlaw() — fits the cumulative kernel to a power-law decay by nonlinear least squares.
price_response_function() — computes the non-parametric price response function for comparison with the regression kernel.
Running this file directly loads CSVs produced by Simulation08.py, recovers parameters for both models, and plots the results.

Wrapper08.py
Convenience script for repeated experiments. Runs the full simulation and recovery pipeline in one call, without producing plots, and appends results to Results08.csv. Useful for building a distribution of estimates across seeds.

Quickstart
Install dependencies:

pip install numpy pandas matplotlib scipy scikit-learn tqdm
Run a single simulation with plots:

python Simulation08.py
python Recovery08.py
Run the batch wrapper (appends one row per model to Results08.csv):

python Wrapper08.py
Key Parameters
Parameter |	Description	| Default
mu_u | Uninformed market order arrival rate (per side, per step) |	2.0
lam	| Limit order placement rate (per level, per step) | 0.5
delta | Cancellation probability per resting unit |	0.05
A |	TIM kernel magnitude | 0.025
beta | TIM kernel power-law decay exponent |	0.6
p_info |	Probability of an informed trader at each step | 0.1
C	| Mean informational shock size (lognormal)	| 0.05
gap_intensity	| Gap-trader order size scaling (Kyle-style price discovery) |	10.0
Dependencies
All code is written for Python 3.8.

Package	Purpose
numpy	Numerical computation
pandas	Data handling and CSV output
matplotlib	Plotting
scipy	Power-law curve fitting
scikit-learn	Linear regression
tqdm	Progress bars
References
Bouchaud, J.-P., Gefen, Y., Potters, M., & Wyart, M. (2004). Fluctuations and response in financial markets.
Bouchaud, J.-P., Farmer, J. D., & Lillo, F. (2009). How markets slowly digest changes in supply and demand.
Gatheral, J. (2010). No-dynamic-arbitrage and market impact.
Kyle, A. S. (1985). Continuous auctions and insider trading.
Lillo, F., & Farmer, J. D. (2004). The long memory of the efficient market.
