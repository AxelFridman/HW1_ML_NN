"""Parts 2-4: polynomial, Gaussian and von Mises fits to the selected neuron."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import COLORS, NEURON, OUT, SESSION, SHORT, save, save_tex, stack
from models import CURVES, DEGREE, MODELS, N_PARAMS, ORI, fit, metrics, neuron_trials, split_means, trial_mse

counts = pd.read_csv(f'{OUT}/counts.csv')
trials = neuron_trials(counts, SESSION, NEURON)
means, floor = split_means(trials)
train, val, test = means['train'].values, means['val'].values, means['test'].values
sem = trials[trials.split == 'train'].groupby('orientation')['count'].sem().values

# the data the models see: one row per trial (the first 8 shown) and the 8 training means they are fitted to
trials.head(8)[['trial', 'stim', 'orientation', 'count', 'split']].to_latex(
    f'{OUT}/tab_trials.tex', index=False, float_format='%g',
    header=['Trial', 'Code', 'Orientation (deg)', 'Spike count', 'Split'])
by_ori = trials[trials.split == 'train'].groupby('orientation')['count']
pd.DataFrame({'Training trials': by_ori.size().map('{:d}'.format), 'Mean count': by_ori.mean().map('{:.2f}'.format),
              'SD': by_ori.std().map('{:.2f}'.format), 'SEM': by_ori.sem().map('{:.2f}'.format)}).T.rename(
    columns='{:g}'.format).rename_axis(columns='Orientation (deg)').to_latex(f'{OUT}/tab_means.tex', column_format='lrrrrrrrr')

# fit each model to the 8 training means (same solution as fitting the 1920 training trials)
train_trials = trials[trials.split == 'train']
fits = {m: fit(m, ORI, train) for m in MODELS}
table = pd.DataFrame([{'Model': m, '$k$': N_PARAMS[m], **metrics(train, f(ORI)),
                       'Trial MSE': trial_mse(f, train_trials), 'Val. MSE': metrics(val, f(ORI))['MSE']}
                      for m, (f, p) in fits.items()])
table.to_csv(f'{OUT}/fits.csv', index=False)
scores = {'MSE': '{:.2f}'.format, 'MAE': '{:.2f}'.format, 'MAPE (\\%)': '{:.1f}'.format,
          '$R^2$': '{:.3f}'.format, 'Trial MSE': '{:.1f}'.format, 'Val. MSE': '{:.2f}'.format}
fit_table = table.rename(columns={'MAPE': 'MAPE (\\%)', 'R2': '$R^2$'}).set_index('Model')
fit_table.loc[list(DEGREE)].reset_index().to_latex(  # the polynomials of part 2
    f'{OUT}/tab_fits.tex', index=False, column_format='lrrrrrrr', formatters=scores)


def sci(v):
    """Readable number: plain for ordinary sizes, scientific for the tiny high-order coefficients."""
    if v == 0 or abs(v) >= 1e-3:
        return f'{v:.3g}'
    mantissa, exponent = f'{v:.1e}'.split('e')
    return f'${mantissa}\\times 10^{{{int(exponent)}}}$'


# the fitted polynomial coefficients (part 2)
pd.DataFrame([[m] + [sci(v) for v in fits[m][1]] for m in DEGREE],
             columns=['Model'] + [f'$\\beta_{i}$' for i in range(8)]).fillna('').to_latex(
    f'{OUT}/tab_coef.tex', index=False, na_rep='', column_format='l' + 'r' * 8)

# the two nonlinear models: their parameters next to the same scores as the polynomials (part 4)
WIDTH = {'Gaussian': '$\\sigma$', 'von Mises': '$\\kappa$'}
nonlinear = pd.DataFrame([{'Model': m, '$B$': f'{p[0]:.1f}', '$A$': f'{p[1]:.1f}',
                           '$\\mu$ (deg)': f'{p[2] % 180:.1f}', 'Width': f'{WIDTH[m]} = {p[3]:.2f}'}
                          for m in CURVES for p in [fits[m][1]]]).set_index('Model')
nonlinear.join(fit_table.loc[list(CURVES), ['MSE', 'Trial MSE', 'Val. MSE']]).reset_index().to_latex(
    f'{OUT}/tab_nonlinear.tex', index=False, column_format='lrrrlrrr', formatters=scores)

# does an L1 (lasso) penalty, with its strength chosen on the validation means, beat the plain fit?
powers = lambda theta, d: np.vander(theta, d + 1, increasing=True)[:, 1:]
LAMBDAS = np.r_[0, np.logspace(-6, 2.5, 120)]  # down to 1e-6, so that "best lambda = 0" is not a coarse grid
rows, chosen_lambdas = [], []
for m, d in DEGREE.items():
    scored = []
    for lam in LAMBDAS:
        if lam == 0:  # no penalty: the ordinary least-squares fit of part 2
            predict, kept = fits[m][0](ORI), d
        else:
            model = make_pipeline(StandardScaler(), Lasso(alpha=lam, max_iter=100000))
            model.fit(powers(ORI, d), train)
            predict, kept = model.predict(powers(ORI, d)), int(np.sum(model[-1].coef_ != 0))
        scored.append((metrics(val, predict)['MSE'], lam, predict, kept))
    val_mse, lam, predict, kept = min(scored, key=lambda s: s[0])
    chosen_lambdas.append(lam)
    rows.append({'Model': m, '$\\lambda$': f'{lam:.3g}', 'Terms kept': f'{kept}/{d}',
                 'Val. MSE (lasso)': val_mse, 'Val. MSE (plain)': metrics(val, fits[m][0](ORI))['MSE'],
                 'Test MSE (lasso)': metrics(test, predict)['MSE'],
                 'Test MSE (plain)': metrics(test, fits[m][0](ORI))['MSE']})
lasso = pd.DataFrame(rows)
lasso.columns = ['Model', '$\\lambda$', stack('Terms', 'kept'), stack('Val. MSE', '(lasso)'),
                 stack('Val. MSE', '(plain)'), stack('Test MSE', '(lasso)'), stack('Test MSE', '(plain)')]
lasso.to_latex(f'{OUT}/tab_lasso.tex', index=False, na_rep='--', float_format='%.2f', column_format='lrrrrrr')

ylim = (train.min() - 0.25 * np.ptp(train), train.max() + 0.4 * np.ptp(train))  # room for the legend


def plot_data(ax):
    ax.errorbar(ORI, train, sem, fmt='o', color='k', ms=7, zorder=3, label='Data')
    ax.set(xticks=[0, 45, 90, 135, 180], xlabel='Orientation (deg)', ylabel='Spike count', ylim=ylim)
    ax.legend(frameon=False, ncol=3, loc='upper center')


x = np.linspace(-11.25, 168.75, 400)

# part 2: polynomial fits
fig, ax = plt.subplots(figsize=(7, 4.5))
for m in DEGREE:
    ax.plot(x, fits[m][0](x), color=COLORS[m], label=m)
plot_data(ax)
save(fig, 'poly_fits')

# part 2: training and validation error against polynomial degree
sweep = pd.DataFrame([{s: metrics(means[s].values, np.polynomial.Polynomial.fit(ORI, train, d)(ORI))['MSE']
                       for s in ('train', 'val')} for d in range(8)])
fig, ax = plt.subplots(figsize=(6, 4.5))
for s, c, label in [('train', 'k', 'Train'), ('val', '0.55', 'Validation')]:
    ax.plot(sweep.index, sweep[s], 'o-', color=c, label=label)
ax.set_yscale('symlog', linthresh=0.1)  # linear below 0.1 so that MSE = 0 can be shown
ax.set(ylim=(0, None), xlabel='Polynomial degree', ylabel='MSE (spikes$^2$)')
ax.legend(frameon=False)
save(fig, 'poly_degree')

# part 3: Gaussian fit
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(x, fits['Gaussian'][0](x), color=COLORS['Gaussian'], label='Gaussian')
plot_data(ax)
save(fig, 'gaussian_fit')

# part 4: the same stimuli repeat every 180 deg; only the von Mises curve repeats with them
xx = np.linspace(-90, 270, 721)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axvspan(0, 180, color='0.93', zorder=0)
for m in ['7th-order', 'Gaussian', 'von Mises']:
    ax.plot(xx, fits[m][0](xx), color=COLORS[m], label=m)
for shift in (-180, 0, 180):
    ax.plot(ORI + shift, train, 'o', ms=7, mec='k', mfc='k' if shift == 0 else 'w', zorder=3,
            label='Data' if shift == 0 else None)
ax.set(xlim=(-90, 270), xticks=np.arange(-90, 271, 45), xlabel='Orientation (deg)', ylabel='Spike count', ylim=ylim)
ax.legend(frameon=False, ncol=4, loc='upper center')
save(fig, 'circular')

# part 4: refit after moving the arbitrary start of the orientation axis
cut = pd.DataFrame([{m: metrics(train, fit(m, (ORI - c) % 180, train)[0]((ORI - c) % 180))['MSE']
                     for m in ['4th-order', 'Gaussian', 'von Mises']} for c in ORI], index=ORI)
fig, ax = plt.subplots(figsize=(6, 4.5))
for m in ['Gaussian', 'von Mises']:
    ax.plot(cut.index, cut[m], 'o-', color=COLORS[m], label=m)
ax.set(yscale='log', xticks=[0, 45, 90, 135], xlabel='Start of the orientation axis (deg)', ylabel='MSE (spikes$^2$)')
ax.legend(frameon=False)
save(fig, 'cut')

# numbers quoted in the report
B, A, mu, sigma = fits['Gaussian'][1]
Bv, Av, muv, kappa = fits['von Mises'][1]
save_tex('fit',
         **{f'mse{SHORT[m]}': f'{r.MSE:.2f}' for m, r in table.set_index('Model').iterrows()},
         **{f'val{SHORT[m]}': f'{r["Val. MSE"]:.2f}' for m, r in table.set_index('Model').iterrows()},
         **{f'wrap{SHORT[m]}': f'{abs(f(180.0) - f(0.0)):.1f}' for m, (f, p) in fits.items()},
         **{f'betaOne{SHORT[m]}': f'{fits[m][1][1]:.3g}' for m in DEGREE},
         floorTrain=f'{floor["train"]:.2f}', floorVal=f'{floor["val"]:.2f}', semMax=f'{sem.max():.1f}',
         trainDegSix=f'{sweep.train[6]:.3f}', valDegSix=f'{sweep.val[6]:.2f}', respZero=f'{train[0]:.0f}',
         BGauss=f'{B:.1f}', muGauss=f'{mu % 180:.1f}', sigmaGauss=f'{sigma:.1f}',
         hwhmGauss=f'{sigma * np.sqrt(2 * np.log(2)):.1f}',
         BVM=f'{Bv:.1f}', muVM=f'{muv % 180:.1f}', kappaVM=f'{kappa:.2f}',
         hwhmVM=f'{np.rad2deg(0.5 * np.arccos(np.log(np.cosh(kappa)) / kappa)):.1f}',
         peakVM=f'{Bv + Av * np.exp(kappa):.1f}', troughVM=f'{Bv + Av * np.exp(-kappa):.1f}',
         cutGaussMin=f'{cut["Gaussian"].min():.1f}', cutGaussBest=f'{cut["Gaussian"].idxmin():g}',
         cutVM=f'{cut["von Mises"].mean():.2f}',
         cutQuartMin=f'{cut["4th-order"].min():.1f}', cutQuartMax=f'{cut["4th-order"].max():.1f}',
         lamMax=f'{max(chosen_lambdas):.3g}',
         betaSmall=sci(min(fits['7th-order'][1], key=abs)),
         trialFloor=f'{train_trials.groupby("orientation")["count"].var().mean():.0f}',
         **{f'trial{SHORT[m]}': f'{r["Trial MSE"]:.0f}' for m, r in table.set_index('Model').iterrows()})
print(table.round(3).to_string(index=False))
print(cut.round(2))
