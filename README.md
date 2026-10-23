# HW1: orientation tuning in macaque V1


## Data

Download `dataset/dataset1/*.mat` from Dryad ([doi:10.5061/dryad.h9w0vt4s0](https://doi.org/10.5061/dryad.h9w0vt4s0),
Xia et al. 2024) into `dataset/dataset1/`. Only dataset 1 is used.

## Run

```
pip install -r requirements.txt
python make_all.py
```

Every table, figure and number goes to `out/`.

| File | What it does |
|---|---|
| `EDA.ipynb` | Exploratory checks, run after `make_all.py` (reads `out/`): file audit, rasters, PSTHs, spike-count noise, drift, stimulus-code order, all tuning curves |
| `counts.py` | Spike count of every V1 neuron on every trial (1280 ms window) → `out/counts.csv`; size of every session → `out/sessions.csv` |
| `tuning.py` | Mean, SD and SEM per neuron and orientation → `out/tuning.csv`; rate, preferred orientation, selectivity, SNR, Fano factor and drift per neuron → `out/neurons.csv`; the choice of session and neuron |
| `spikes.py` | Figures from the raw spike trains: how the data are organised, from spikes to counts, raster and PSTH |
| `models.py` | The six models, least-squares fitting, goodness-of-fit metrics and the train/validation/test split |
| `fit.py` | Parts 2-4: polynomial, Gaussian and von Mises fits |
| `generalize.py` | Part 5: held-out trials, held-out orientations, noisier neurons, other neurons and sessions |
| `common.py` | Paths, the chosen session and neurons, plot style |
| `make_all.py` | Runs the scripts in order |
