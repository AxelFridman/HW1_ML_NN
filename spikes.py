"""Part 1 figures from the raw 1 ms spike trains: how the data are organised and how one trial becomes
one spike count."""
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

save_tex('spikes',
         arrayShape=' \\times '.join(map(str, spikes.shape)),
         spikeShare=f'{100 * spikes.mean():.1f}',
         exTrial=trial, exOri=f'{pref:g}', exCount=int(s[trial].sum()),
         exActive=int((spikes[trial].sum(axis=0) > 0).sum()),
         exStart=t0, exEnd=t0 + 40, exWindow=int(window.sum()),
         blockMin=int(s[:8].sum(axis=1).min()), blockMax=int(s[:8].sum(axis=1).max()))
