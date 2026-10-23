"""Part 1 figures from the raw 1 ms spike trains: how the data are organised, how one trial becomes one
spike count, and the raster and PSTH of the selected neuron."""
import matplotlib.pyplot as plt
import numpy as np

from common import NEURON, SESSION, save, save_tex
from counts import load_session

spikes, stim = load_session(SESSION)  # spikes[trial, ms, neuron] = 0 or 1
theta = (stim - 1) * 22.5
ori = np.unique(theta)
s = spikes[:, :, NEURON]  # trials x 1280 ms of the selected neuron
pref = ori[np.argmax([s[theta == t].sum() for t in ori])]
HIGHLIGHT = '#2a78d6'

# 1. what the data look like: the stimulus of each trial, all neurons in one trial, the 0/1 values of one neuron
trial = np.flatnonzero(theta == pref)[0]  # first trial showing the neuron's preferred orientation
t0 = 80                                   # start of the 40 ms close-up
fig = plt.figure(figsize=(15, 6.5))
grid = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.6, wspace=0.35)
ax = fig.add_subplot(grid[0, 0])
ax.plot(np.arange(32), theta[:32], 'o', color='k')
for b in (7.5, 15.5, 23.5):
    ax.axvline(b, color='0.8', lw=1)
ax.set(xlabel='Trial', ylabel='Orientation (deg)', yticks=ori[::2])
ax.set_title('(a)', loc='left')
ax = fig.add_subplot(grid[0, 1:])
ax.eventplot([np.flatnonzero(spikes[trial, :, n]) for n in range(spikes.shape[2])], color='k', linewidths=0.6)
ax.eventplot([np.flatnonzero(s[trial])], lineoffsets=NEURON, colors=HIGHLIGHT, linewidths=2)
ax.axvspan(t0, t0 + 40, color=HIGHLIGHT, alpha=0.15)
ax.set(xlim=(0, 1280), ylim=(-1, spikes.shape[2]), xlabel='Time from stimulus onset (ms)', ylabel='Neuron')
ax.set_title(f'(b)  trial {trial}', loc='left')
ax = fig.add_subplot(grid[1, :])
window = s[trial, t0:t0 + 40]
ax.imshow(window[None, :], cmap='Blues', vmin=0, vmax=2, aspect='auto')
for i, v in enumerate(window):
    ax.text(i, 0, v, ha='center', va='center', fontsize=13)
ax.set(yticks=[], xticks=np.arange(0, 40, 5), xticklabels=np.arange(t0, t0 + 40, 5), xlabel='Time (ms)')
ax.set_title(f'(c)  neuron {NEURON}, trial {trial}', loc='left')
save(fig, 'data_structure')

# 2. from spike train to spike count: the first block of 8 trials, which shows every orientation once
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.eventplot([np.flatnonzero(s[t]) for t in range(8)], color='k', linewidths=1)
for t in range(8):
    ax.text(1300, t, f'{s[t].sum()} spikes', va='center')
ax.set(xlim=(0, 1280), ylim=(7.5, -0.5), yticks=range(8), yticklabels=[f'trial {t}: {theta[t]:g} deg' for t in range(8)],
       xlabel='Time from stimulus onset (ms)')
save(fig, 'count_example')

# 3. raster of the first 25 trials of each orientation, and the PSTH at the preferred and orthogonal orientations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
rows = np.concatenate([np.flatnonzero(theta == t)[:25] for t in ori])
ax1.eventplot([np.flatnonzero(s[i]) for i in rows], color='k', linewidths=0.6)
for y in np.arange(1, 8) * 25 - 0.5:
    ax1.axhline(y, color='0.8', lw=0.5)
ax1.set(xlim=(0, 1280), ylim=(-0.5, 199.5), yticks=np.arange(8) * 25 + 12, yticklabels=[f'{t:g}' for t in ori],
        xlabel='Time from stimulus onset (ms)', ylabel='Orientation (deg)')
for t, color in [(pref, 'k'), ((pref + 90) % 180, '0.6')]:
    rate = s[theta == t].reshape(-1, 64, 20).mean(axis=(0, 2)) * 1000  # Hz in 20 ms bins
    ax2.plot(np.arange(64) * 20 + 10, rate, color=color, label=f'{t:g} deg')
ax2.set(xlim=(0, 1280), ylim=(0, None), xlabel='Time from stimulus onset (ms)', ylabel='Firing rate (Hz)')
ax2.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1))
save(fig, 'raster')

save_tex('spikes',
         arrayShape=' \\times '.join(map(str, spikes.shape)),
         spikeShare=f'{100 * spikes.mean():.1f}',
         exTrial=trial, exOri=f'{pref:g}', exCount=int(s[trial].sum()),
         exActive=int((spikes[trial].sum(axis=0) > 0).sum()),
         exStart=t0, exEnd=t0 + 40, exWindow=int(window.sum()),
         blockMin=int(s[:8].sum(axis=1).min()), blockMax=int(s[:8].sum(axis=1).max()))
