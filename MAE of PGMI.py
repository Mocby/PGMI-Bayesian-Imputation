import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import expm
import seaborn as sns

# ===========================
# Config
# ===========================
SEED_BASE = 123
T = 5.0
N = 100
sigma = 0.20
A = 1.0
zeta = 0.2
omega_n = 2 * np.pi * 1
omega_d = omega_n * np.sqrt(1 - zeta**2)
phi_true = np.pi / 4

missing_fractions = np.linspace(0.1, 0.9, 9)
missing_mode = "MCAR"
n_iterations = 600

# Prior for shrinkage (PGMI–Bayes)
phi0_prior = phi_true   # "correctly" centered prior (best-case scenario)
tau_prior = 0.5         # prior std dev in radians (tune to change shrinkage strength)

# ===========================
# Time and clean signal
# ===========================
t = np.linspace(0, T, N, endpoint=False)
dt = t[1] - t[0]
y_true = A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t + phi_true)

# ===========================
# CRLB (kept, but not plotted now)
# ===========================
def crlb_phi(t, A, zeta, omega_n, omega_d, phi, sigma):
    g = A * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t + phi)
    info = np.sum(g**2) / (sigma**2)
    return 1.0 / info

def crlb_phi_mcar(t, A, zeta, omega_n, omega_d, phi, sigma, beta):
    g = A * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t + phi)
    info_full = np.sum(g**2) / (sigma**2)
    info_mcar = (1.0 - beta) * info_full
    return 1.0 / info_mcar

ideal_crlb_list = [crlb_phi(t, A, zeta, omega_n, omega_d, phi_true, sigma)
                   for _ in missing_fractions]
mcar_crlb_list = [crlb_phi_mcar(t, A, zeta, omega_n, omega_d, phi_true, sigma, b)
                  for b in missing_fractions]

# ===========================
# Missingness generators
# ===========================
def remove_data_mcar(y, beta, rng):
    mask = rng.random(len(y)) > beta
    y_missing = y.copy()
    y_missing[~mask] = np.nan
    return y_missing, mask

def remove_data_block(y, beta, rng):
    L = len(y)
    block_len = max(1, int(np.round(beta * L)))
    start = rng.integers(0, L - block_len + 1)
    mask = np.ones(L, dtype=bool)
    mask[start:start+block_len] = False
    y_missing = y.copy()
    y_missing[~mask] = np.nan
    return y_missing, mask

# ===========================
# Simple imputers
# ===========================
def linear_interpolation_impute(y):
    y = y.copy()
    nans = np.isnan(y)
    if np.all(~nans):
        return y
    idx = np.arange(len(y))
    y[nans] = np.interp(idx[nans], idx[~nans], y[~nans])
    return y

def simulation_impute(y_missing, mask, y_true, noise_std=0.1, rng=None):
    y_sim = y_missing.copy()
    noise = rng.normal(0, 2*noise_std, size=np.sum(~mask))
    y_sim[~mask] = y_true[~mask] + noise
    return y_sim

# ===========================
# Kalman filter + RTS smoother
# ===========================
def _ss_matrices(zeta, omega_n, dt):
    Ac = np.array([[0.0, 1.0],
                   [-(omega_n**2), -2.0*zeta*omega_n]])
    F = expm(Ac * dt)
    H = np.array([[1.0, 0.0]])
    return F, H

def kalman_filter_impute(y_obs, sigma, zeta, omega_n, dt, q=1e-4):
    F, H = _ss_matrices(zeta, omega_n, dt)
    Q = q * np.eye(2)
    R = np.array([[sigma**2]])
    Nn = len(y_obs)

    x_f = np.zeros((Nn, 2))
    P_f = np.zeros((Nn, 2, 2))

    first_idx = np.where(~np.isnan(y_obs))[0]
    if len(first_idx) == 0:
        return np.zeros_like(y_obs)
    x0 = np.array([y_obs[first_idx[0]], 0.0])
    P0 = np.diag([1,1]) * 1e3

    for k in range(Nn):
        if k == 0:
            x_pred = F @ x0
            P_pred = F @ P0 @ F.T + Q
        else:
            x_pred = F @ x_f[k-1]
            P_pred = F @ P_f[k-1] @ F.T + Q

        if np.isnan(y_obs[k]):
            x_f[k] = x_pred
            P_f[k] = P_pred
        else:
            yk = np.array([[y_obs[k]]])
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            innov = yk - H @ x_pred
            x_f[k] = x_pred + (K @ innov).ravel()
            P_f[k] = (np.eye(2) - K @ H) @ P_pred

    y_hat = (H @ x_f.transpose(1,0)).squeeze()
    y_imp = y_obs.copy()
    y_imp[np.isnan(y_imp)] = y_hat[np.isnan(y_imp)]
    return y_imp

def kalman_smoother_impute(y_obs, sigma, zeta, omega_n, dt, q=1e-4):
    F, H = _ss_matrices(zeta, omega_n, dt)
    Q = q * np.eye(2)
    R = np.array([[sigma**2]])
    Nn = len(y_obs)

    x_f = np.zeros((Nn, 2))
    P_f = np.zeros((Nn, 2, 2))

    first_idx = np.where(~np.isnan(y_obs))[0]
    if len(first_idx) == 0:
        return np.zeros_like(y_obs)
    x0 = np.array([y_obs[first_idx[0]], 0.0])
    P0 = np.diag([1,1]) * 1e3

    for k in range(Nn):
        if k == 0:
            x_pred = F @ x0
            P_pred = F @ P0 @ F.T + Q
        else:
            x_pred = F @ x_f[k-1]
            P_pred = F @ P_f[k-1] @ F.T + Q

        if np.isnan(y_obs[k]):
            x_f[k] = x_pred
            P_f[k] = P_pred
        else:
            yk = np.array([[y_obs[k]]])
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            innov = yk - H @ x_pred
            x_f[k] = x_pred + (K @ innov).ravel()
            P_f[k] = (np.eye(2) - K @ H) @ P_pred

    # RTS smoother
    x_s = np.zeros_like(x_f)
    P_s = np.zeros_like(P_f)
    x_s[-1], P_s[-1] = x_f[-1], P_f[-1]

    for k in range(Nn-2, -1, -1):
        P_pred = F @ P_f[k] @ F.T + Q
        Ck = P_f[k] @ F.T @ np.linalg.inv(P_pred)
        x_s[k] = x_f[k] + (Ck @ (x_s[k+1] - (F @ x_f[k]))).ravel()
        P_s[k] = P_f[k] + Ck @ (P_s[k+1] - P_pred) @ Ck.T

    y_hat_s = (H @ x_s.transpose(1,0)).squeeze()
    y_imp = y_obs.copy()
    y_imp[np.isnan(y_imp)] = y_hat_s[np.isnan(y_imp)]
    return y_imp

# ===========================
# Bayesian (MH over φ) for imputation method
# ===========================
def _wrap_to_pi(phi):
    return (phi + np.pi) % (2*np.pi) - np.pi

def bayesian_impute_phi_mh(y_missing, t, sigma, A, zeta, omega_n, omega_d,
                           n_samples=600, burn=200, proposal_sd=0.06, seed=None):
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(y_missing)
    if not np.any(mask):
        mu0 = A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t)
        return np.where(np.isnan(y_missing), mu0, y_missing)

    y_obs = y_missing[mask]
    t_obs = t[mask]

    def mu_phi(phi, tt):
        return A * np.exp(-zeta * omega_n * tt) * np.sin(omega_d * tt + phi)

    def loglik(phi):
        resid = y_obs - mu_phi(phi, t_obs)
        return -0.5 * np.sum(resid**2) / sigma**2

    coarse = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    ll_vals = np.array([loglik(ph) for ph in coarse])
    phi_curr = coarse[np.argmax(ll_vals)]
    ll_curr = ll_vals.max()

    draws = []
    for _ in range(burn + n_samples):
        phi_prop = _wrap_to_pi(phi_curr + rng.normal(0, proposal_sd))
        ll_prop = loglik(phi_prop)
        if np.log(rng.uniform()) < (ll_prop - ll_curr):
            phi_curr, ll_curr = phi_prop, ll_prop
        draws.append(phi_curr)

    phi_samples = np.array(draws[burn:])
    mu_all = np.stack([mu_phi(ph, t) for ph in phi_samples])
    y_pp_mean = mu_all.mean(axis=0)
    y_imp = y_missing.copy()
    y_imp[np.isnan(y_imp)] = y_pp_mean[np.isnan(y_imp)]
    return y_imp

# ===========================
# MLE for phase
# ===========================
def mle_phi(y, t):
    best_phi = None
    best_loss = np.inf
    phi_grid = np.linspace(-np.pi, np.pi, 80)

    for phi_init in phi_grid:
        def nll(phi):
            y_model = A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t + phi)
            return np.sum((y - y_model)**2)
        res = minimize(nll, x0=np.array([phi_init]), bounds=[(-np.pi, np.pi)])
        if res.fun < best_loss:
            best_loss = res.fun
            best_phi = res.x[0]

    return best_phi

# ===========================
# PGMI–Bayes shrinkage estimator for φ
# ===========================
def bayes_phi_shrinkage(y, t, sigma, A, zeta, omega_n, omega_d,
                        phi0, tau, n_grid=361):
    """
    Bayesian shrinkage estimator for φ using a prior φ ~ N(phi0, tau^2).

    Uses the FULL (PGMI-completed) data y (no NaNs).
    Returns the posterior mean E[φ | y] via grid integration.
    """
    phi_grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)

    def model(phi):
        return A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t + phi)

    SSR = np.empty_like(phi_grid)
    for i, phi in enumerate(phi_grid):
        y_model = model(phi)
        SSR[i] = np.sum((y - y_model)**2)

    # log-likelihood (up to a constant)
    loglik = -0.5 * SSR / (sigma**2)
    # log-prior (Gaussian)
    logprior = -0.5 * ((phi_grid - phi0)**2) / (tau**2)
    # log-posterior (unnormalized)
    logpost = loglik + logprior
    # numerical stability
    logpost -= np.max(logpost)
    w = np.exp(logpost)
    w_sum = np.sum(w)

    if w_sum == 0:
        # fallback to MAP on the grid
        phi_map = phi_grid[np.argmax(logpost)]
        return _wrap_to_pi(phi_map)

    phi_mean = np.sum(w * phi_grid) / w_sum
    return _wrap_to_pi(phi_mean)

# ===========================
# Storage (MAE instead of variance)
# ===========================
phi_mae = {
    "missing_data": [], "linear": [], "sim": [],
    "kalman_filt": [], "kalman_smooth": [],
    "bayes": [], "pgmi_bayes": []
}

# ===========================
# Main MC loop
# ===========================
for i_beta, beta in enumerate(missing_fractions):
    v_missing = []
    v_lin = []
    v_sim = []
    v_kf = []
    v_rts = []
    v_bayes = []
    v_pgmi_bayes = []

    for mc in range(n_iterations):
        rng = np.random.default_rng(SEED_BASE + mc)
        y_noisy = y_true + rng.normal(0, sigma, N)

        # Generate missingness
        if missing_mode.upper() == "MCAR":
            y_missing, mask = remove_data_mcar(y_noisy, beta, rng)
        else:
            y_missing, mask = remove_data_block(y_noisy, beta, rng)

        # 1) Observed-only MLE (Stage 1 of PGMI)
        phi_missing = mle_phi(y_missing[mask], t[mask])
        v_missing.append(phi_missing)

        # 2) Linear interpolation
        y_lin = linear_interpolation_impute(y_missing)
        v_lin.append(mle_phi(y_lin, t))

        # 3) Simulated imputation
        y_sim = simulation_impute(y_missing, mask, y_true, noise_std=sigma, rng=rng)
        v_sim.append(mle_phi(y_sim, t))

        # 4) Kalman filter
        y_kf = kalman_filter_impute(y_missing, sigma, zeta, omega_n, dt)
        v_kf.append(mle_phi(y_kf, t))

        # 5) RTS smoother
        y_rts = kalman_smoother_impute(y_missing, sigma, zeta, omega_n, dt)
        v_rts.append(mle_phi(y_rts, t))

        # 6) Bayesian imputation (posterior predictive mean, then MLE on that)
        y_bayes = bayesian_impute_phi_mh(
            y_missing, t, sigma, A, zeta, omega_n, omega_d,
            n_samples=600, burn=200, proposal_sd=0.06,
            seed=SEED_BASE + mc
        )
        v_bayes.append(mle_phi(y_bayes, t))

        # 7) PGMI–Bayes (shrinkage on PGMI-completed data)
        y_pgmi = y_missing.copy()
        y_model = A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t + phi_missing)
        y_pgmi[~mask] = y_model[~mask]

        phi_pgmi_bayes = bayes_phi_shrinkage(
            y_pgmi, t, sigma, A, zeta, omega_n, omega_d,
            phi0=phi0_prior, tau=tau_prior, n_grid=361
        )
        v_pgmi_bayes.append(phi_pgmi_bayes)

    # Store MAE for each method at this missing fraction
    phi_mae["missing_data"].append(np.mean(np.abs(np.array(v_missing) - phi_true)))
    phi_mae["linear"].append(np.mean(np.abs(np.array(v_lin) - phi_true)))
    phi_mae["sim"].append(np.mean(np.abs(np.array(v_sim) - phi_true)))
    phi_mae["kalman_filt"].append(np.mean(np.abs(np.array(v_kf) - phi_true)))
    phi_mae["kalman_smooth"].append(np.mean(np.abs(np.array(v_rts) - phi_true)))
    phi_mae["bayes"].append(np.mean(np.abs(np.array(v_bayes) - phi_true)))
    phi_mae["pgmi_bayes"].append(np.mean(np.abs(np.array(v_pgmi_bayes) - phi_true)))

# Convert lists to arrays
for k in phi_mae:
    phi_mae[k] = np.asarray(phi_mae[k])

# ===========================
# Plot MAE
# ===========================
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 18
})

plt.figure(figsize=(16, 7))
plt.plot(missing_fractions, phi_mae["missing_data"], label="Observed Only", marker='o')
plt.plot(missing_fractions, phi_mae["linear"], label="Linear", marker='^')
plt.plot(missing_fractions, phi_mae["sim"], label="Simulated", marker='d')
plt.plot(missing_fractions, phi_mae["kalman_filt"], label="Kalman", marker='x')
plt.plot(missing_fractions, phi_mae["kalman_smooth"], label="RTS Smoother", marker='P')
plt.plot(missing_fractions, phi_mae["bayes"], label="Bayesian Imputation", marker='h')
plt.plot(missing_fractions, phi_mae["pgmi_bayes"], label="PGMI–Bayesian", marker='v')

plt.xlabel(r"Missing Data Fraction")
plt.ylabel(r"Mean Absolute Error of $\hat{\phi}$ (rad)")
plt.grid(True, alpha=0.4)
sns.set_style("darkgrid")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig("phi_mae_plot_with_pgmi_bayes.pdf", format="pdf", bbox_inches="tight")
plt.show()
