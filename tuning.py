"""Mean response per neuron and orientation, and one summary row per neuron
-> out/tuning.csv, out/neurons.csv and the part 1 figures."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import NEURON, NOISY, OUT, SESSION, save, save_tex

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

# figures for the selected session
tun = tuning[tuning.session == SESSION]
nrn = neurons[neurons.session == SESSION].set_index('neuron')

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

save_tex('tuning',
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
