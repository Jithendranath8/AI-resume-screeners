"""Run the Colab real-PDF workflow locally, cell by cell, without Colab.

The experiment logic lives in colab/real_pdf_workflow.py and is executed unchanged.
This script only replaces the Colab-specific plumbing:

* ``google.colab.files.upload``  -> reads PDFs and labels.csv from --inputs
* ``google.colab.files.download`` -> copies the file into --out
* ``google.colab.userdata.get``   -> environment variables (optionally from .env)
* ``IPython.display.display``     -> print
* ``/content``                    -> --out
* the apt/pip install cell and the ipywidgets demo cell are skipped

Usage:
    .venv/bin/python experiment/run_real_pdf_local.py --inputs data/real_pdfs --out results/real_pdf
    .venv/bin/python experiment/run_real_pdf_local.py --stop-after labels   # only write the label template

Stops cleanly with instructions when labels.csv is missing. Everything else
(variants, OCR, classifier, LLM calls, metrics, export) runs end to end.
"""
import argparse
import os
import re
import shutil
import sys
import types
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # repository root
WORKFLOW = PROJECT / 'colab' / 'real_pdf_workflow.py'


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def install_shims(inputs_dir, out_dir):
    class _Files:
        @staticmethod
        def upload():
            payload = {}
            for p in sorted(Path(inputs_dir).iterdir()):
                if p.suffix.lower() == '.pdf' or p.name == 'labels.csv':
                    payload[p.name] = p.read_bytes()
            print(f'[local] upload shim supplied {len(payload)} file(s) from {inputs_dir}')
            return payload

        @staticmethod
        def download(path):
            path = Path(path)
            target = Path(out_dir) / path.name
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
            print(f'[local] download shim saved {target}')

    class _Userdata:
        @staticmethod
        def get(name):
            value = os.environ.get(name)
            if not value:
                raise KeyError(f'{name} is not set in the environment or .env')
            return value

    from importlib.machinery import ModuleSpec
    colab = types.ModuleType('google.colab')
    colab.files, colab.userdata = _Files, _Userdata
    colab.__spec__ = ModuleSpec('google.colab', None)  # importlib.util.find_spec probes this
    google = sys.modules.get('google') or types.ModuleType('google')
    if getattr(google, '__spec__', None) is None:
        google.__spec__ = ModuleSpec('google', None)
    google.colab = colab
    sys.modules['google'] = google
    sys.modules['google.colab'] = colab

    def display(*objects):
        for obj in objects:
            try:
                import pandas as pd
                if isinstance(obj, pd.DataFrame):
                    with pd.option_context('display.max_rows', 200, 'display.max_columns', 30, 'display.width', 160):
                        print(obj.to_string())
                    continue
            except Exception:
                pass
            print(obj)

    try:
        import IPython.display as ipd  # real package: patch display so DataFrames print as text
        ipd.display = display
    except ImportError:
        ipd = types.ModuleType('IPython.display')
        ipd.display = display
        ipd.Markdown = lambda text: text
        ipython = types.ModuleType('IPython')
        ipython.display = ipd
        ipython.get_ipython = lambda: None  # matplotlib probes this
        sys.modules['IPython'] = ipython
        sys.modules['IPython.display'] = ipd


def split_cells(source):
    parts = re.split(r'^# %%.*$', source, flags=re.M)[1:]
    headers = re.findall(r'^# %%.*$', source, flags=re.M)
    cells = []
    for header, body in zip(headers, parts):
        if '[markdown]' in header:
            continue
        cells.append(body.strip('\n'))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default=str(PROJECT / 'data' / 'real_pdfs'))
    ap.add_argument('--out', default=str(PROJECT / 'results' / 'real_pdf'))
    ap.add_argument('--stop-after', choices=['labels', 'prepare', 'preflight', 'all'], default='all',
                    help='labels: write the template and stop; prepare: stop before any LLM call; '
                         'preflight: stop after the call-count preflight')
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                    help="override a configuration constant after the config cell, e.g. "
                         "--set BACKEND=compatible --set RUN_ID=pdf_new_01 --set COMPAT_MODEL=llama-3.3-70b-versatile. "
                         "Values are parsed as Python literals when possible, otherwise kept as strings.")
    args = ap.parse_args()
    overrides = {}
    for item in args.set:
        name, _, value = item.partition('=')
        try:
            import ast
            value = ast.literal_eval(value)
        except Exception:
            pass
        overrides[name.strip()] = value

    os.environ.setdefault('MPLBACKEND', 'Agg')
    load_env(PROJECT / '.env')
    inputs_dir = Path(args.inputs).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    install_shims(inputs_dir, out_dir)

    source = WORKFLOW.read_text().replace('/content', str(out_dir))
    if overrides:
        # Insert overrides right after the constants, before the asserts and derived values.
        marker = "assert BACKEND in {"
        assert source.count(marker) == 1, 'config marker not found'
        lines = ''.join(f'{k} = {v!r}\n' for k, v in overrides.items())
        derived = ("API_PAUSE_SECONDS = 4 if BACKEND in {'gemini', 'compatible'} else 0\n"
                   "API_MAX_OUTPUT_TOKENS = 2048 if BACKEND == 'gemini' else 250\n"
                   "HOSTED = BACKEND in {'gemini', 'openai', 'compatible'}\n")
        for k in overrides:
            derived = '\n'.join(l for l in derived.splitlines() if not l.startswith(k + ' =')) + '\n'
        source = source.replace(marker, f"# [local overrides]\n{lines}{derived}{marker}")
        print('[local] overrides:', overrides)
    cells = split_cells(source)
    namespace = {'__name__': '__workflow__'}
    os.chdir(out_dir)

    for index, cell in enumerate(cells):
        first = cell.splitlines()[0] if cell else ''
        if 'apt-get' in cell and 'pip' in cell:
            print(f'[local] skip cell {index}: Colab package installation')
            namespace['subprocess'] = __import__('subprocess')
            namespace['sys'] = sys
            continue
        if 'import ipywidgets' in cell:
            print(f'[local] skip cell {index}: ipywidgets demo (Colab only)')
            continue
        if "label_upload = files.upload()" in cell and not (inputs_dir / 'labels.csv').exists() \
                and not (out_dir / namespace.get('RUN_ID', '') / 'labels.csv').exists():
            print('\n[local] labels.csv not found. Fill the template written above as '
                  f'{inputs_dir / "labels.csv"} (expected_decision = SELECT or REJECT for every row, '
                  'source filled, permission_confirmed = yes) and run again.')
            return 2
        print(f'\n[local] ---- cell {index}: {first[:70]}')
        exec(compile(cell, f'{WORKFLOW.name}:cell{index}', 'exec'), namespace)
        if args.stop_after == 'labels' and 'labels_template.csv' in cell:
            print('[local] stop-after labels reached')
            return 0
        if args.stop_after == 'prepare' and "print('Prepared'" in cell:
            print('[local] stop-after prepare reached (no LLM calls made)')
            return 0
        if args.stop_after == 'preflight' and 'planned_calls =' in cell:
            print('[local] stop-after preflight reached (no LLM calls made)')
            return 0
    print('\n[local] workflow finished; results are under', out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
