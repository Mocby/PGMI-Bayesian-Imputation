PGMI–Bayesian Imputation

Code and simulations supporting the paper:

“Phase-Guided Bayesian Model-Based Imputation for Missing Sensor Data in Mass–Spring–Damper Systems.”

This repository contains the full implementation of the PGMI–Bayesian method, including Monte Carlo experiments, entropy analysis, CRLB benchmarks, and comparisons with classical imputation methods.

⭐ Overview

PGMI–Bayesian is a physics-guided, two-stage imputation method designed to improve phase estimation when sensor data is missing.

It works by:

Stage 1 — Observed-Only Phase Estimation:
Estimate the phase using only available samples.

Stage 2 — Model-Based Reconstruction:
Reconstruct missing samples using the mass–spring–damper model.

Bayesian Shrinkage Update:
Apply a weak prior to stabilize the likelihood and reduce estimation variance.

This hybrid strategy provides lower variance, lower MAE/MSE, and lower entropy than classical imputation methods such as linear interpolation, Kalman filtering, and standard Bayesian imputation.

📂 Repository Structure
PGMI-Bayesian-Imputation/
│
├── MAE ans MSE PGMI_bayes.py      # Computes MAE & MSE across missingness levels
├── MAE of PGMI.py                 # MAE evaluation for observed-only estimator
├── Variance PGMI_bayes.py         # Variance computation script
├── phi_mse_results.csv            # Example output CSV for MSE aggregation
│
└── README.md

📊 Features

Monte Carlo simulation framework (600 trials per missingness level)

CRLB (ideal and MCAR) computation for theoretical benchmarking

Entropy-based analysis to quantify uncertainty in reconstructions

Full comparison with:

Observed-only estimation

Linear interpolation

Kalman filtering

Classical Bayesian imputation

PGMI–Bayesian consistently outperforms all baselines across all levels of missingness.

🧠 Key Results

Lowest variance, MAE, and MSE among all reconstruction methods

Below the MCAR CRLB up to 30% missingness, indicating improved effective Fisher information

Lowest entropy, meaning reconstructions preserve the model’s dynamics without injecting randomness

Strong stability even under severe missingness (70–90%)

🛠️ Dependencies

Install required Python packages:

pip install numpy scipy matplotlib seaborn

🚀 Running the Experiments

To reproduce the results from the paper:

python "Variance PGMI_bayes.py"
python "MAE ans MSE PGMI_bayes.py"
python "MAE of PGMI.py"


These scripts generate variance curves, MAE/MSE values, and CSV summaries.

📝 Citation

If you use this code, please cite:

Omanda Bouraima, M., Zawodniok, M.
Phase-Guided Bayesian Model-Based Imputation for Missing Sensor Data
in Mass–Spring–Damper Systems. I2MTC 2026 (submitted).

📧 Contact

For questions, collaboration, or further development:

mocby@mst.edu
