"""Mean response per neuron and orientation, and one summary row per neuron
-> out/tuning.csv, out/neurons.csv and the part 1 figures."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import NEURON, NOISY, OUT, SESSION, SKIP, save, save_tex

counts = pd.read_csv(f'{OUT}/counts.csv')

# tuning curve: mean, SD and SEM of the spike count over the trials of each orientation
tuning = (counts.groupby(['session', 'neuron', 'orientation'])['count']
          .agg(n='size', mean='mean', sd='std').reset_index())
tuning['sem'] = tuning['sd'] / np.sqrt(tuning['n'])
tuning.to_csv(f'{OUT}/tuning.csv', index=False)

# per neuron: rate, preferred orientation, selectivity (OTI), signal-to-noise, Fano factor and drift
keys = [tuning.session, tuning.neuron]
var = tuning['sd'] ** 2
g = tuning.groupby(keys)['mean']
counts['resid'] = counts['count'] - counts.groupby(['session', 'neuron', 'orientation'])['count'].transform('mean')
neurons = pd.DataFrame({
    'rate_hz': g.mean() / 1.28,
    'pref': tuning.loc[g.idxmax()].set_index(['session', 'neuron'])['orientation'],
    'oti': (g.max() - g.min()) / (g.max() + g.min()),
    'snr': g.var(ddof=0) / var.groupby(keys).mean(),  # variance across orientations / across trials
    'fano': (var / tuning['mean']).groupby(keys).mean(),
    'drift': counts.groupby(['session', 'neuron'])[['trial', 'resid']].corr().xs('trial', level=2)['resid'],
}).reset_index()
neurons.to_csv(f'{OUT}/neurons.csv', index=False)

# choosing a session: its size, its well-tuned neurons (SNR > 1), and whether the stimulus codes behave like
# consecutive orientations, i.e. whether neighbouring codes (22.5 deg apart) evoke similar population responses
sessions = pd.read_csv(f'{OUT}/sessions.csv').set_index('session')
sessions['tuned'] = neurons[neurons.snr > 1].groupby('session').size()
neighbour = {}
for s, d in tuning.merge(neurons[neurons.snr > 1][['session', 'neuron']]).groupby('session'):
    m = d.pivot(index='neuron', columns='orientation', values='mean')
    z = m.sub(m.mean(axis=1), axis=0).div(m.std(axis=1), axis=0)  # every tuning curve on the same scale
    corr = np.corrcoef(z.values.T)
    neighbour[s] = [corr[k, (k + 1) % 8] for k in range(8)]  # codes 1-2, 2-3, ..., 8-1
sessions['lowest'] = pd.Series({s: min(c) for s, c in neighbour.items()})
sessions['note'] = ['chosen' if s == SESSION else 'codes out of order' if s in SKIP else '' for s in sessions.index]
sessions.reset_index()[['session', 'v1', 'v2', 'trials', 'tuned', 'lowest', 'note']].to_latex(
    f'{OUT}/tab_sessions.tex', index=False, formatters={'lowest': '${:.2f}$'.format}, column_format='lrrrrrl',
    header=['Session', 'V1 neurons', 'V2 neurons', 'Trials', 'Well-tuned V1', 'Min. neighbour corr.', ''])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [3, 2]})
for s, c in neighbour.items():
    ax1.plot(range(8), c, 'o-', color='#eb6834' if s in SKIP else '0.6', label=s if s in SKIP else 'other sessions')
ax1.axhline(0, color='k', lw=1)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[:2], labels[:2], frameon=False)
ax1.set(xticks=range(8), xticklabels=[f'{k + 1}-{(k + 1) % 8 + 1}' for k in range(8)],
        xlabel='Neighbouring stimulus codes', ylabel='Correlation of responses')
ax1.set_title('(a)', loc='left')
odd = tuning[(tuning.session == SKIP[0]) & tuning.neuron.isin([55, 88, 94, 103])]  # examples shown in EDA.ipynb
for n, d in odd.groupby('neuron'):
    ax2.plot(np.r_[d.orientation, 180], np.r_[d['mean'], d['mean'].iloc[0]] / d['mean'].max(), 'o-', color='#eb6834')
ax2.set(xticks=[0, 45, 90, 135, 180], xlabel='Orientation (deg)', ylabel='Response / maximum')
ax2.set_title(f'(b)  four neurons of {SKIP[0]}', loc='left')
save(fig, 'sessions')

# figures for the selected session
tun = tuning[tuning.session == SESSION]
nrn = neurons[neurons.session == SESSION].set_index('neuron')

# from trials to a tuning curve: the count of every trial of the selected neuron, and the mean of each orientation
trials = counts[(counts.session == SESSION) & (counts.neuron == NEURON)]
curve = tun[tun.neuron == NEURON]
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(trials.orientation + np.random.default_rng(0).uniform(-7, 7, len(trials)), trials['count'], s=5, color='0.75')
ax.errorbar(curve.orientation, curve['mean'], curve.sd, fmt='o-', color='k', ms=7, capsize=5)
ax.set(xticks=curve.orientation, xticklabels=[f'{o:g}' for o in curve.orientation], xlabel='Orientation (deg)',
       ylabel='Spike count per trial')
save(fig, 'trial_counts')

# choosing a neuron: signal-to-noise ratio of every neuron in the session, highest first
ranked = nrn.snr.sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
rank = np.arange(1, len(ranked) + 1)
ax.plot(rank, ranked.values, 'o', color='0.6', ms=4)
chosen = ranked.index.get_loc(NEURON)
ax.plot(rank[chosen], ranked[NEURON], 'o', color='k', ms=8)
ax.annotate(f'#{NEURON}', (rank[chosen], ranked[NEURON]), xytext=(8, 0), textcoords='offset points', va='center')
ax.axhline(1, color='k', lw=1, ls='--')
ax.set(yscale='log', xlabel='Neuron, ranked by SNR', ylabel='Signal-to-noise ratio')
save(fig, 'snr_rank')

# example tuning curves: the highest-SNR neuron for each preferred orientation
examples = nrn.sort_values('snr').groupby('pref').tail(1).sort_values('pref').index
fig, axs = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
for ax, n in zip(axs.flat, examples):
    d = tun[tun.neuron == n]
    ax.fill_between(d.orientation, d['mean'] - d.sd, d['mean'] + d.sd, color='0.85')
    ax.errorbar(d.orientation, d['mean'], d['sem'], fmt='o-', color='k', ms=5)
    ax.set_title(f'#{n}', loc='left')
    ax.set_xticks([0, 45, 90, 135])
for ax in axs[1]:
    ax.set_xlabel('Orientation (deg)')
for ax in axs[:, 0]:
    ax.set_ylabel('Spike count')
fig.tight_layout()
save(fig, 'tuning_examples')

# all neurons of the session: SNR against preferred orientation, chosen neurons labelled
fig, ax = plt.subplots(figsize=(7, 4.5))
x = nrn.pref + np.random.default_rng(0).uniform(-4, 4, len(nrn))  # jittered preferred orientation
ax.scatter(x, nrn.snr, s=25, color='0.6')
for n in [NEURON] + NOISY:
    ax.scatter(x[n], nrn.snr[n], s=60, color='k')
    ax.annotate(f'#{n}', (x[n], nrn.snr[n]), xytext=(6, 0), textcoords='offset points', va='center')
ax.set(yscale='log', xticks=[0, 45, 90, 135], xlabel='Preferred orientation (deg)', ylabel='Signal-to-noise ratio')
save(fig, 'neuron_map')

peak, low = curve.loc[curve['mean'].idxmax()], curve.loc[curve['mean'].idxmin()]
save_tex('tuning',
         nbrSkip=f'{sessions.lowest[SKIP[0]]:.2f}', nbrOthers=f'{sessions.lowest.drop(SKIP).min():.2f}',
         tunedChosen=int(sessions.tuned[SESSION]), nVtwo=int(sessions.v2[SESSION]),
         nSkipVOne=int(sessions.v1[SKIP[0]]), vOneRange=f'{sessions.v1.min()}--{sessions.v1.max()}',
         meanPeak=f'{peak["mean"]:.0f}', sdPeak=f'{peak.sd:.0f}', semPeak=f'{peak["sem"]:.1f}',
         meanLow=f'{low["mean"]:.0f}', sdLow=f'{low.sd:.0f}', oriLow=f'{low.orientation:g}',
         nSessions=counts.session.nunique(),
         nNeurons=len(nrn),
         nTrials=counts[counts.session == SESSION].trial.nunique(),
         nPerOri=int(tun.n.iloc[0]),
         neuron=NEURON, noisyA=NOISY[0], noisyB=NOISY[1],
         snrMain=f'{nrn.snr[NEURON]:.2f}', snrA=f'{nrn.snr[NOISY[0]]:.2f}', snrB=f'{nrn.snr[NOISY[1]]:.2f}',
         rateMain=f'{nrn.rate_hz[NEURON]:.1f}', rateA=f'{nrn.rate_hz[NOISY[0]]:.1f}', rateB=f'{nrn.rate_hz[NOISY[1]]:.1f}',
         prefMain=f'{nrn.pref[NEURON]:g}', otiMain=f'{nrn.oti[NEURON]:.2f}',
         fanoMain=f'{nrn.fano[NEURON]:.1f}', driftMain=f'{nrn.drift[NEURON]:.2f}',
         ratioA=f'{nrn.snr[NEURON] / nrn.snr[NOISY[0]]:.0f}', ratioB=f'{nrn.snr[NEURON] / nrn.snr[NOISY[1]]:.0f}')
print(nrn.loc[[NEURON] + NOISY].round(3))
