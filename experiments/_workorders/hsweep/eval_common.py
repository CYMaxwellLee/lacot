"""Parameterized copy of PB runtime.load_frozen; frozen sources remain untouched."""
import ast
import importlib.util
import os
from pathlib import Path
import random
import sys
sys.dont_write_bytecode = True
PB = Path(__file__).resolve().parents[1] / 'breakthrough-probe'
sys.path.insert(0, str(PB))
import numpy as np
from common import (Blocked, ENV, H, PIN, REPO, SOURCE, digest_array, file_sha,
                    reset_env, seeds, verify_source)
from runtime import import_boundary, DATASET_PIN


def load_ckpt(dataset_dir, ckpt_path, ckpt_sha256):
    """Execute original M9 bytes through its s33/EMA loader, never legacy eval.

    A trace exception at the module boundary leaves the imported definitions and
    loaded weights intact. No sed, AST transformation, source-copy or flat-guard
    modification is used. Call once in a fresh process (determinism setup).
    """
    # wallpen r2b: read the caller's seed override; retain s33 by default.
    seed_value = os.environ.get('WALLPEN_EXPECTED_SEED', '33')
    try:
        expected_seed = int(seed_value)
    except ValueError:
        raise Blocked(f'BLOCKED: WALLPEN_EXPECTED_SEED must be an integer: {seed_value!r}') from None
    if expected_seed != 33:
        print(f'⚠️ eval_common: WALLPEN_EXPECTED_SEED={seed_value}（非 s33 契約）', file=sys.stderr)
    ckpt_path = Path(ckpt_path).resolve()
    boundary = import_boundary()  # before torch, dataset, checkpoint, imports
    if not ckpt_path.is_file() or file_sha(ckpt_path) != ckpt_sha256:
        raise Blocked('BLOCKED: s33 checkpoint SHA256 mismatch/unavailable')
    dataset = Path(dataset_dir) / (ENV + '.npz')
    if not dataset.is_file():
        raise Blocked(f'BLOCKED: normalization dataset unavailable: {dataset}')
    if file_sha(dataset) != DATASET_PIN:
        raise Blocked('BLOCKED: normalization dataset SHA256 mismatch')
    import torch
    from harness_move1 import configure_determinism
    settings = configure_determinism(torch)
    config = dict(ENV=ENV, CONS='self', K=8, COND=256, CHUNK=4, TCAP=128,
                  ENC_OBJ='recon_ictr', LEARNED_REFINE=0, COND_DROP=0.1,
                  BC_INDEP=1, TEACHER_MIX=0.5, WARMUP=500, DEC_START='soft',
                  LOAD_EMA=1, CONT_TRAIN=0, STEPS1=0, STEPS2=0, DEV_EVAL=0,
                  PREREQ=0, GRAD_REFINE=0, SUBGOAL='', FINISH_R=0, INTENT='',
                  U_SOURCE='flow', BON_N=0, LOAD_CKPT=str(ckpt_path), SEED=33,
                  BOOT_DATA='', S1_FROM='', FLOW_PROBE=0, DIAG_DUMP=0)
    saved_env, saved_path, saved_trace = dict(os.environ), list(sys.path), sys.gettrace()
    saved_bytecode = sys.dont_write_bytecode
    class ModelLoaded(Exception):
        pass
    spec = importlib.util.spec_from_file_location('breakthrough_frozen_m9', SOURCE)
    module = importlib.util.module_from_spec(spec)
    def stop_before_legacy_eval(frame, event, arg):
        if frame.f_code.co_filename == str(SOURCE) and frame.f_code.co_name == '<module>':
            if event == 'line' and frame.f_lineno == boundary:
                raise ModelLoaded()
            return stop_before_legacy_eval
        return None
    try:
        for key in list(os.environ):
            if key.startswith('LACOT_'):
                del os.environ[key]
        os.environ.update({'LACOT_'+k: str(v) for k, v in config.items()})
        os.environ['OGBENCH_DATA_DIR'] = str(dataset_dir)
        sys.path.insert(0, str(REPO))
        sys.dont_write_bytecode = True
        sys.settrace(stop_before_legacy_eval)
        try:
            spec.loader.exec_module(module)
        except ModelLoaded:
            pass
        else:
            raise Blocked('BLOCKED: frozen module crossed legacy evaluation boundary')
    finally:
        sys.settrace(saved_trace)
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_bytecode
        os.environ.clear()
        os.environ.update(saved_env)
    # wallpen r2b: only the expected checkpoint seed changes in this contract.
    if not (module.LOAD_EMA == 1 and module.TAG_SEED == expected_seed and module.T_CAP == 128
            and module.STEPS2 == module._S1 == 0 and module.CONT_TRAIN == 0):
        raise Blocked('BLOCKED: s33/EMA/no-training contract mismatch')
    # Source loads EMA for the consumer chain, raw checkpoint encoder/decoder.
    names = ('cond_enc', 'cond_head', 'flow', 'ahead', 'bc_head',
             'traj_enc', 'e_pooler', 'u_dec')
    for name in names:
        model = getattr(module, name)
        model.eval()
        model.requires_grad_(False)
    for name, state in module._ck['ema'].items():
        actual = getattr(module, name).state_dict()
        if any(not torch.equal(actual[k], v.to(module.device)) for k, v in state.items()):
            raise Blocked('BLOCKED: EMA weights not delivered')
    module.probe_provenance = dict(source_path=str(SOURCE), source_sha256=verify_source(),
        ckpt_path=str(ckpt_path), ckpt_sha256=ckpt_sha256, dataset_path=str(dataset),
        dataset_sha256=file_sha(dataset), normalization_mu=module.MU_XY.tolist(),
        normalization_sd=module.SD_XY.tolist(), import_stop_line=boundary,
        config=config, deterministic_settings=settings)
    return module


class StopProbe(RuntimeError):
    pass


def smooth(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (128, 2) or not np.isfinite(raw).all():
        raise StopProbe(f'Decoder API requires finite [128,2], got {raw.shape}')
    return np.stack([np.convolve(raw[:, k], np.ones(9)/9, mode='valid') for k in range(2)], axis=1)


def smooth_anchor(raw, xy):
    points = smooth(raw)
    points += np.asarray(xy) - points[0]
    return points
