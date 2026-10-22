"""Spike count of every V1 neuron on every trial of every session -> out/counts.csv."""
import glob
import os

import h5py
import numpy as np
import pandas as pd

from common import DATA, OUT


def load_session(session, area=0):
    """Spikes (trials x 1280 ms x neurons, 0/1) and stimulus codes 1-8; area 0 = V1, 1 = V2."""
    with h5py.File(os.path.join(DATA, session + '_spikes.mat'), 'r') as f:
        stim = f['stim'][0].astype(int)
        data = f[f['spikes'][area, 0]]
        spikes = np.empty(data.shape, np.uint8)
        for i in range(0, len(stim), 200):  # read in blocks: a session is ~4 GB as float64
            spikes[i:i + 200] = data[i:i + 200]
    return spikes, stim


if __name__ == '__main__':
    tables = []
    for path in sorted(glob.glob(os.path.join(DATA, '*_spikes.mat'))):
        session = os.path.basename(path).replace('_spikes.mat', '')
        spikes, stim = load_session(session)
        counts = spikes.sum(axis=1)  # spikes during the 1280 ms stimulus
        n_trials, n_neurons = counts.shape
        tables.append(pd.DataFrame({
            'session': session,
            'neuron': np.tile(np.arange(n_neurons), n_trials),
            'trial': np.repeat(np.arange(n_trials), n_neurons),
            'stim': np.repeat(stim, n_neurons),
            'orientation': np.repeat((stim - 1) * 22.5, n_neurons),
            'count': counts.ravel()}))
        print(session, spikes.shape)
    pd.concat(tables).to_csv(os.path.join(OUT, 'counts.csv'), index=False)
