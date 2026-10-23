"""Part 5: how well the models generalize to new trials, new orientations, noisier neurons
and other neurons and sessions."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COLORS, NEURON, NOISY, OUT, SESSION, SHORT, SKIP, save, save_tex, stack
from models import MODELS, N_PARAMS, ORI, fit, metrics, neuron_trials, split_means, trial_mse

THREE = ['7th-order', 'Gaussian', 'von Mises']  # the three models compared across neurons
CHOICE = 'von Mises'  # the model we end up choosing, highlighted in the tables
counts = pd.read_csv(f'{OUT}/counts.csv')
neurons = pd.read_csv(f'{OUT}/neurons.csv').set_index(['session', 'neuron'])
snr = neurons.loc[SESSION, 'snr']


def loo_rmse(model, r):
    """Leave one orientation out: fit the other 7 means, predict the held-out one; RMSE over the 8 folds."""
    if N_PARAMS[model] > 7:
        return np.nan  # 8 coefficients cannot be determined from 7 points
    err = [fit(model, np.delete(ORI, k), np.delete(r, k))[0](ORI[k]) - r[k] for k in range(8)]
    return np.sqrt(np.mean(np.square(err)))


# fit on the training means; score on the training and test means, on held-out orientations
# (all 400 trials of the 7 remaining orientations) and on the 0/180 deg wrap
rows = []
for n, models in [(NEURON, MODELS)] + [(n, THREE) for n in NOISY]:
    trials = neuron_trials(counts, SESSION, n)
    means, floor = split_means(trials)
    all_means = trials.groupby('orientation')['count'].mean().values
    for m in models:
        f, p = fit(m, ORI, means['train'].values)
        test = metrics(means['test'].values, f(ORI))
        rows.append({'neuron': n, 'Model': m, '$k$': N_PARAMS[m],
                     'Trial MSE': trial_mse(f, trials[trials.split == 'train']),
                     'Train MSE': metrics(means['train'].values, f(ORI))['MSE'], 'Test MSE': test['MSE'],
                     'Test MAE': test['MAE'], 'Test MAPE (\\%)': test['MAPE'], 'Test $R^2$': test['R2'],
                     'LOO RMSE': loo_rmse(m, all_means), 'Wrap': abs(f(180.0) - f(0.0)),
                     'floor': floor['test'], 'min_mean': means['train'].min(), 'kappa': p[3] if m == 'von Mises' else np.nan})
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/generalize.csv', index=False)

fmt = {'Train MSE': '{:.2f}', 'Test MSE': '{:.2f}', 'Test MAE': '{:.2f}', 'Test MAPE (\\%)': '{:.1f}',
       'Test $R^2$': '{:.3f}', 'LOO RMSE': '{:.1f}', 'Wrap': '{:.1f}'}
fmt = {c: f.format for c, f in fmt.items()}

# table 6.1 of the assignment: the six models on the selected neuron, chosen model in bold
main = res[res.neuron == NEURON]
summary = pd.DataFrame({
    'Model': main.Model, 'Number of parameters': main['$k$'].astype(str),
    'MSE': main['Train MSE'].map('{:.2f}'.format),
    'MSE (single trials)': main['Trial MSE'].map('{:.0f}'.format),
    'Biologically interpretable?': np.where(main.Model.isin(['Gaussian', 'von Mises']), 'yes', 'no'),
    'Accounts for circularity?': np.where(main.Model == 'von Mises', 'yes', 'no')})
chosen = summary.Model == CHOICE
summary.loc[chosen] = summary.loc[chosen].map('\\textbf{{{}}}'.format)
summary.columns = ['Model', stack('Number of', 'parameters'), 'MSE', stack('Trial', 'MSE'),
                   stack('Biologically', 'interpretable?'), stack('Accounts for', 'circularity?')]
summary.to_latex(f'{OUT}/tab_compare.tex', index=False, column_format='lrrrcc')

# the three models on the three neurons
noise = res[res.Model.isin(THREE)].copy()
noise['Neuron'] = [f'\\#{n} (SNR {snr[n]:.2f})' for n in noise.neuron]
noise['Model'] = noise.Model.map(lambda m: f'\\textbf{{{m}}}' if m == CHOICE else m)
columns = ['Train MSE', 'Test MSE', 'Test MAE', 'Test MAPE (\\%)', 'Test $R^2$', 'LOO RMSE']
noise.set_index(['Neuron', 'Model'])[columns].to_latex(
    f'{OUT}/tab_noise.tex', na_rep='--', formatters=fmt, multirow=True, index_names=False, column_format='llrrrrrr')

# fits of the three models to the three neurons; grey band = trial-to-trial SD
x = np.linspace(-11.25, 168.75, 400)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, n in zip(axs, [NEURON] + NOISY):
    by_ori = neuron_trials(counts, SESSION, n).query('split == "train"').groupby('orientation')['count']
    lo, hi = by_ori.mean() - by_ori.std(), by_ori.mean() + by_ori.std()
    ax.fill_between(ORI, lo, hi, color='0.88')
    for m in THREE:
        ax.plot(x, fit(m, ORI, by_ori.mean().values)[0](x), color=COLORS[m], label=m)
    ax.errorbar(ORI, by_ori.mean(), by_ori.sem(), fmt='o', color='k', ms=6, zorder=3)
    ax.set_title(f'#{n}', loc='left')
    ax.set(xticks=[0, 45, 90, 135, 180], xlabel='Orientation (deg)',
           ylim=(lo.min() - 0.1 * (hi.max() - lo.min()), hi.max() + 0.1 * (hi.max() - lo.min())))
axs[0].set_ylabel('Spike count')
axs[0].legend(frameon=False, loc='upper center')
save(fig, 'noise_fits')

# learning curves: split the shuffled training trials of each orientation into disjoint groups of n trials,
# fit each group's 8 means and score them on the test means; average over groups
NS = [3, 5, 10, 20, 40, 80, 240]
rng = np.random.default_rng(1)
curves = []
for n in [NEURON] + NOISY:
    trials = neuron_trials(counts, SESSION, n)
    test_means = split_means(trials)[0]['test'].values
    train_counts = np.stack([c.values for _, c in trials[trials.split == 'train'].groupby('orientation')['count']])
    train_counts = rng.permuted(train_counts, axis=1)  # shuffle the trials of each orientation
    for size in NS:
        for g in range(train_counts.shape[1] // size):
            r = train_counts[:, g * size:(g + 1) * size].mean(axis=1)
            curves += [{'neuron': n, 'model': m, 'n': size, 'MSE': metrics(test_means, fit(m, ORI, r)[0](ORI))['MSE']}
                       for m in THREE]
curves = pd.DataFrame(curves).groupby(['neuron', 'model', 'n'])['MSE'].mean()
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, n in zip(axs, [NEURON] + NOISY):
    for m in THREE:
        ax.plot(NS, curves[n, m], 'o-', color=COLORS[m], label=m)
    ax.axhline(res[res.neuron == n].floor.iloc[0], color='k', ls='--', lw=1)
    ax.set(xscale='log', yscale='log', xlabel='Training trials per orientation')
    ax.set_xticks(NS, [str(s) for s in NS])
    ax.minorticks_off()
    ax.set_title(f'#{n}', loc='left')
axs[0].set_ylabel('Test MSE (spikes$^2$)')
axs[0].legend(frameon=False, loc='lower left')
save(fig, 'learning_curves')
crossover = {n: next((s for s in NS if curves[n, '7th-order', s] < curves[n, 'von Mises', s]), 'none') for n in [NEURON] + NOISY}

# other neurons and sessions: does von Mises predict the test trials better than the Gaussian?
pop = []
for s, n in neurons[neurons.snr > 1].drop(SKIP, level='session').index:
    means = split_means(neuron_trials(counts, s, n))[0]
    (fg, _), (fv, pv) = [fit(m, ORI, means['train'].values) for m in ('Gaussian', 'von Mises')]
    pop.append({'session': s, 'neuron': n, 'pref': pv[2] % 180,
                'ratio': metrics(means['test'].values, fg(ORI))['MSE'] / metrics(means['test'].values, fv(ORI))['MSE']})
pop = pd.DataFrame(pop)
pop['near_cut'] = np.minimum(pop.pref, 180 - pop.pref) < 22.5
pop.to_csv(f'{OUT}/population.csv', index=False)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(pop.pref, pop.ratio, s=25, color='0.6')
chosen = pop[(pop.session == SESSION) & (pop.neuron == NEURON)].iloc[0]
ax.scatter(chosen.pref, chosen.ratio, s=60, color='k')
ax.annotate(f'#{NEURON}', (chosen.pref, chosen.ratio), xytext=(6, 0), textcoords='offset points', va='center')
ax.axhline(1, color='k', lw=1)
ax.set(yscale='log', xticks=[0, 45, 90, 135, 180], xlabel='Preferred orientation (deg)',
       ylabel='Test MSE, Gaussian / von Mises')
save(fig, 'population')

# numbers quoted in the report, rounded as in the tables
tag = {NEURON: 'Main', NOISY[0]: 'A', NOISY[1]: 'B'}
metric = {'testMAE': 'Test MAE', 'testMAPE': 'Test MAPE (\\%)', 'testRtwo': 'Test $R^2$', 'loo': 'LOO RMSE'}
vm = res[res.Model == 'von Mises'].set_index('neuron')
save_tex('generalize',
         **{f'{k}{SHORT[r.Model]}{tag[r.neuron]}': fmt[c](r[c]) for _, r in res.iterrows() for k, c in metric.items()},
         **{f'floor{tag[n]}': f'{vm.floor[n]:.2f}' for n in tag},
         **{f'minResp{tag[n]}': f'{vm.min_mean[n]:.1f}' for n in tag},
         **{f'kappa{tag[n]}': f'{vm.kappa[n]:.2g}' for n in tag},
         **{f'cross{tag[n]}': crossover[n] for n in tag},
         popN=len(pop), popSessions=pop.session.nunique(), popFrac=f'{100 * (pop.ratio > 1).mean():.0f}',
         popNear=int(pop.near_cut.sum()), popMedNear=f'{pop.ratio[pop.near_cut].median():.1f}',
         popMedFar=f'{pop.ratio[~pop.near_cut].median():.2f}', popWorse=int((pop.ratio < 0.5).sum()))
print(res.drop(columns='floor').round(3).to_string(index=False))
print(curves.unstack('n').round(2))
print('crossover', crossover)
print(pop.groupby('near_cut').ratio.describe().round(2))
