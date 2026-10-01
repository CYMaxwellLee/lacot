"""Pinned provenance and serialization; safe to import without torch/ogbench."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = REPO / 'experiments/_workorders/ucontrast1/smoke/mutants/M9/scratch_lacot_rollout.py'
PIN = '276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc'
COLLECTOR = REPO / 'experiments/_workorders/ucontrast1/collector.py'
BACKGROUND = Path('/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u')
GATE = BACKGROUND / 'gate0b-result.json'
CARD = HERE / 'CONTRACT-FROZEN.md'
CARD_PIN = '7525254c83a62031ad761f6993112b14df59595fadfdfd743a45e9313bc91cf4'
CODE_FILES = ('common.py', 'builder.py', 'rules.py', 'runtime.py', 'harvest.py',
              'harness_move1.py', 'harness_move3.py', 'smoke.py', 'calibrate.py', 'artifacts.py')
ENV = 'pointmaze-large-stitch-v0'
H, RHO, SIGMA = 1000, 1.875, .05


class Blocked(RuntimeError):
    pass


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def verify_source():
    actual = file_sha(SOURCE)
    if actual != PIN:
        raise Blocked(f'BLOCKED: frozen source SHA256 expected={PIN} actual={actual} path={SOURCE}')
    return actual


def digest_array(value):
    # Exactly collector.py's dtype + shape + tobytes algorithm.
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def write_json(path, value):
    """Exclusive creation: never silently overwrite a previous experiment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    with path.open('xb') as f:
        f.write(data)
    with Path(str(path) + '.sha256').open('x') as f:
        f.write(sha(data) + '\n')
    return sha(data)


def seeds(task, episode, draw, injected=False):
    if draw not in range(64, 80):
        raise ValueError('draw outside 64..79')
    stream = 7 * task + episode + 1_000_003 * draw
    return dict(env_seed=1000 * task + episode, numpy_seed=1000 * task + episode,
                action_space_seed=1000 * task + episode, base_seed=7 * task + episode,
                stream_seed=stream, flow_seed=None if injected else stream,
                noise_seed=stream, python_seed=stream, torch_seed=stream)


def reset_env(env, task, episode):
    import numpy as np
    seed = 1000 * task + episode
    np.random.seed(seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    return env.reset(seed=seed, options={'task_id': task, 'render_goal': False})


def receipt(builder):
    verify_contract()
    verify_source()  # A receipt may never launder a mismatched source.
    return dict(source_path=str(SOURCE), source_sha256=PIN, collector_sha256=file_sha(COLLECTOR),
                card_sha256=file_sha(CARD), gate_sha256=file_sha(GATE),
                builder_sha256=sha(canonical(builder)),
                question_sha256=sha(canonical(builder['questions'])),
                code_sha256=code_hashes(),
                H=H, rho=RHO, env=ENV, offline_only=True, training_use_prohibited=True)


def verify_contract():
    if file_sha(CARD) != CARD_PIN:
        raise Blocked('BLOCKED: frozen contract SHA256 mismatch')
    return CARD_PIN


def code_hashes():
    return {name: file_sha(HERE/name) for name in CODE_FILES}


def read_flags():
    flags = json.loads((HERE/'flags.json').read_text())
    if set(flags) != {'PRODUCTION_READY', 'SMOKE_ENABLED'} or any(type(v) is not bool for v in flags.values()):
        raise Blocked('BLOCKED: flags.json requires exactly two boolean release flags')
    return flags


def release_check(smoke):
    flags = read_flags()
    if not smoke and not flags['PRODUCTION_READY']:
        raise Blocked('BLOCKED: PRODUCTION_READY=False; lead release requires GPU smoke')
    if smoke and not flags['SMOKE_ENABLED']:
        raise Blocked('BLOCKED: SMOKE_ENABLED=False')
