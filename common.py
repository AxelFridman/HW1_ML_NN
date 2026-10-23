"""Paths, the chosen session and neurons, and helpers shared by all scripts."""
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

DATA = 'dataset/dataset1'
OUT = 'out'
FIGS = os.path.join(OUT, 'figs')
os.makedirs(FIGS, exist_ok=True)

SESSION = '107l003p143'  # its stimulus codes follow orientation order (EDA.ipynb)
SKIP = ['105l001p16']    # its codes 6-8 do not behave like consecutive orientations (EDA.ipynb)
NEURON = 81              # neuron analysed in parts 2-5
NOISY = [4, 91]          # noisier neurons for part 5

COLORS = {'Linear': '#eda100', '3rd-order': '#008300', '4th-order': '#e87ba4',
          '7th-order': '#2a78d6', 'Gaussian': '#eb6834', 'von Mises': '#1baf7a'}
SHORT = {'Linear': 'Lin', '3rd-order': 'Cub', '4th-order': 'Quart',  # model names inside LaTeX macro names
         '7th-order': 'Sept', 'Gaussian': 'Gauss', 'von Mises': 'VM'}

plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 14,
                     'ytick.labelsize': 14, 'legend.fontsize': 12, 'lines.linewidth': 2,
                     'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def stack(top, bottom):
    """A two-line table header, so that wide tables stay inside the page."""
    return f'\\shortstack{{{top}\\\\{bottom}}}'


def save_tex(name, **numbers):
    """Write numbers quoted in the report as LaTeX macros, e.g. \\muVM."""
    with open(os.path.join(OUT, name + '.tex'), 'w') as f:
        f.writelines(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in numbers.items())
