"""Raster and PSTH of the selected neuron from its raw 1 ms spike trains (part 1 figure)."""
import matplotlib.pyplot as plt
import numpy as np

from common import NEURON, SESSION, save
from counts import load_session

spikes, stim = load_session(SESSION)
spikes = spikes[:, :, NEURON]  # trials x 1280 ms
theta = (stim - 1) * 22.5
ori = np.unique(theta)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# raster: the first 25 trials of each orientation
rows = np.concatenate([np.flatnonzero(theta == t)[:25] for t in ori])
ax1.eventplot([np.flatnonzero(spikes[i]) for i in rows], color='k', linewidths=0.6)
for y in np.arange(1, 8) * 25 - 0.5:
    ax1.axhline(y, color='0.8', lw=0.5)
ax1.set(xlim=(0, 1280), ylim=(-0.5, 199.5), yticks=np.arange(8) * 25 + 12, yticklabels=[f'{t:g}' for t in ori],
        xlabel='Time from stimulus onset (ms)', ylabel='Orientation (deg)')

# PSTH: firing rate in 20 ms bins, averaged over the 400 trials of the preferred and orthogonal orientations
pref = ori[np.argmax([spikes[theta == t].sum() for t in ori])]
for t, color in [(pref, 'k'), ((pref + 90) % 180, '0.6')]:
    rate = spikes[theta == t].reshape(-1, 64, 20).mean(axis=(0, 2)) * 1000
    ax2.plot(np.arange(64) * 20 + 10, rate, color=color, label=f'{t:g} deg')
ax2.set(xlim=(0, 1280), ylim=(0, None), xlabel='Time from stimulus onset (ms)', ylabel='Firing rate (Hz)')
ax2.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1))
save(fig, 'raster')
