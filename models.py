"""The six tuning models, least-squares fitting, goodness-of-fit metrics and the train/val/test split."""
import numpy as np
from scipy.optimize import curve_fit

ORI = np.arange(8) * 22.5  # stimulus orientations (deg)


def gaussian(theta, B, A, mu, sigma):
    return B + A * np.exp(-(theta - mu) ** 2 / (2 * sigma ** 2))


def von_mises(theta, B, A, mu, kappa):
    return B + A * np.exp(kappa * np.cos(2 * np.deg2rad(theta - mu)))


DEGREE = {'Linear': 1, '3rd-order': 3, '4th-order': 4, '7th-order': 7}
CURVES = {'Gaussian': gaussian, 'von Mises': von_mises}
MODELS = list(DEGREE) + list(CURVES)
N_PARAMS = {**{m: d + 1 for m, d in DEGREE.items()}, 'Gaussian': 4, 'von Mises': 4}


def fit(model, theta, r):
    """Least-squares fit to mean responses r at orientations theta -> (fitted curve, parameters)."""
    if model in DEGREE:
        poly = np.polynomial.Polynomial.fit(theta, r, DEGREE[model])
        return poly, poly.convert().coef
    f, best = CURVES[model], None
    for mu0 in ORI[::2]:  # start from 0, 45, 90 and 135 deg and keep the best fit: mu has local minima
        if model == 'Gaussian':
            p0, bounds = [r.min(), np.ptp(r), mu0, 30], ([-np.inf, 0, -90, 1], [np.inf, np.inf, 270, 180])
        else:  # start at kappa = 1, where the curve spans A * (e - 1/e) = 2.35 A
            p0, bounds = [r.min(), np.ptp(r) / 2.35, mu0, 1], ([-np.inf, 0, -np.inf, 0], [np.inf, np.inf, np.inf, 50])
        try:
            p = curve_fit(f, theta, r, p0, bounds=bounds, maxfev=10000)[0]
        except RuntimeError:
            continue
        sse = np.sum((f(theta, *p) - r) ** 2)
        if best is None or sse < best[0]:
            best = sse, p
    p = best[1]
    return (lambda t: f(t, *p)), p


def metrics(r, pred):
    """Goodness of fit of predictions to observed mean responses r."""
    err = pred - r
    return {'MSE': np.mean(err ** 2), 'MAE': np.mean(np.abs(err)),
            'MAPE': 100 * np.mean(np.abs(err / r)),
            'R2': 1 - np.sum(err ** 2) / np.sum((r - np.mean(r)) ** 2)}


def split_trials(stim, seed=0):
    """Assign the trials of each orientation to train/val/test (60/20/20 %) at random."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(stim), dtype=object)
    for s in np.unique(stim):
        idx = rng.permutation(np.flatnonzero(stim == s))
        a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
        split[idx[:a]], split[idx[a:b]], split[idx[b:]] = 'train', 'val', 'test'
    return split


def neuron_trials(counts, session, neuron):
    """Trials of one neuron in presentation order, labelled train/val/test (same split for every neuron)."""
    trials = counts[(counts.session == session) & (counts.neuron == neuron)].sort_values('trial').copy()
    trials['split'] = split_trials(trials.stim.values)
    return trials


def trial_mse(f, trials):
    """MSE of the fitted curve against every single trial, instead of against the 8 means."""
    return np.mean((f(trials['orientation'].values) - trials['count'].values) ** 2)


def split_means(trials):
    """Mean count per orientation (rows) in each split (columns), and each split's noise floor:
    the MSE that even the true tuning curve would have, i.e. the mean squared SEM."""
    g = trials.groupby(['split', 'orientation'])['count']
    return g.mean().unstack('split'), (g.var() / g.size()).groupby('split').mean()
