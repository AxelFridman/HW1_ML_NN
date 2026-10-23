"""Run the whole analysis in order; every output goes to out/."""
import subprocess
import sys

for script in ['counts.py', 'tuning.py', 'spikes.py', 'fit.py', 'generalize.py']:
    print('running', script, flush=True)
    subprocess.run([sys.executable, script], check=True)
