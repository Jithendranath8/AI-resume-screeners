"""Execute notebooks with this interpreter; retain executed M2 as an audit artifact.

    .venv/bin/python report/verify_notebooks.py                       # M2, saved under results/m2_reproduction/
    .venv/bin/python report/verify_notebooks.py M3_Research_Gap.ipynb # any notebook at the repository root, in place

Notebooks are run with the repository root as the working directory.
"""
import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]  # repository root
os.environ['MPLCONFIGDIR'] = str(root / '.cache/matplotlib')
os.environ['IPYTHONDIR'] = str(root / '.cache/ipython')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['IPYTHONDIR']).mkdir(parents=True, exist_ok=True)

class LocalKernel(KernelManager):
    def format_kernel_cmd(self, extra_arguments=None):
        return [sys.executable, '-m', 'ipykernel_launcher', '-f', self.connection_file]

for name in sys.argv[1:] or ['M2_Reproduction.ipynb']:
    notebook = nbformat.read(root / name, as_version=4)
    if name == 'M2_Reproduction.ipynb':
        notebook.cells.append(nbformat.v4.new_code_cell('''
import json
from pathlib import Path
assert results['corpus'] == {'total': 104, 'injected': 72, 'clean': 32}
assert results['detection']['confusion'] == {'tp': 68, 'fp': 0, 'fn': 4, 'tn': 32}
assert [v['successes'] for v in results['asr'].values()] == [72, 44, 40, 20, 0, 0]
Path('results/m2_reproduction/M2_validation.json').write_text(json.dumps(results, indent=2))
print('M2 benchmark counts and historical values independently asserted.')
'''))
    notebook = NotebookClient(notebook, timeout=180, km=LocalKernel(),
                              resources={'metadata': {'path': str(root)}}).execute()
    target = root / 'results/m2_reproduction/M2_verified.ipynb' if name == 'M2_Reproduction.ipynb' else root / name
    nbformat.write(notebook, target)
    print(f'PASS {name}: all {sum(c.cell_type == "code" for c in notebook.cells)} code cells executed')
