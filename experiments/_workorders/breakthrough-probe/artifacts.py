"""Exclusive per-rollout checkpoints and compressed RNG evidence, with hash checks."""
import copy
import gzip
import json
from pathlib import Path
from common import canonical, file_sha, sha, write_json


def save_row(outdir, row, index):
    path = Path(outdir)/'shards'/f'{index:04d}.json'
    compact = copy.deepcopy(row)
    rng = {k: compact.pop(k) for k in ('rng_before', 'rng_after')}
    data = gzip.compress(canonical(rng), mtime=0)
    rng_path = path.with_suffix('.rng.json.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    with rng_path.open('xb') as f:
        f.write(data)
        f.flush()
    compact['rng_sidecar'] = dict(path=rng_path.name, sha256=sha(data))
    write_json(path, compact)
    return dict(path=str(path.relative_to(outdir)), sha256=file_sha(path))


def checked_path(base, ref):
    path = (base/ref['path']).resolve()
    if not path.is_relative_to(base.resolve()) or file_sha(path) != ref['sha256']:
        raise ValueError('artifact sidecar path/hash mismatch')
    return path


def load_artifact(path):
    path = Path(path)
    if file_sha(path) != Path(str(path)+'.sha256').read_text().strip():
        raise ValueError('artifact file SHA mismatch')
    artifact = json.loads(path.read_text())
    if artifact.get('storage') == 'shards-v1':
        for name in ('rows', 'conformance_rows'):
            rows = []
            for ref in artifact[name]:
                row_path = checked_path(path.parent, ref)
                row = json.loads(row_path.read_text())
                rng_path = checked_path(row_path.parent, row.pop('rng_sidecar'))
                row.update(json.loads(gzip.decompress(rng_path.read_bytes())))
                rows.append(row)
            artifact[name] = rows
    return artifact


def resume_rows(directory):
    """Read completed shards after interruption; dangling partial shards fail closed."""
    directory = Path(directory)
    header = load_artifact(directory/'checkpoint.json')
    rows = []
    for path in sorted((directory/'shards').glob('*.json')):
        row = load_artifact(path)
        rng_path = checked_path(path.parent, row.pop('rng_sidecar'))
        row.update(json.loads(gzip.decompress(rng_path.read_bytes())))
        rows.append(row)
    return header, rows
