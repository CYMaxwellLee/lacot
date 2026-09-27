import json
import os
from pathlib import Path
import subprocess
import tempfile

PYTHON = '/home/cymaxwelllee/Projects/lacot/.venv/bin/python3'
ROOT = Path(__file__).resolve().parents[1]


def workspace():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name) / 'Projects' / 'lacot'
    root.mkdir(parents=True)
    return tmp, root


def dump(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value), encoding='utf-8')


def run(script, tmp, *args):
    return subprocess.run([PYTHON, str(ROOT / script), *args], env={**os.environ, 'HOME': tmp.name},
                          cwd=ROOT, text=True, capture_output=True)
