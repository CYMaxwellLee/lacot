"""Real s33 CPU interface and runtime tests; no GPU/production evidence."""
import json
import os
import numpy as np
from builder import build
from common import digest_array
import harness_move1 as m1
import harness_move3 as m3
import harvest
from runtime import load_frozen, interface_check, run_draw



def protocol_equivalence(module, env):
    """S2 p16: independent frozen policy_chunk oracle A, 3 x 160 real steps."""
    import runtime
    from unittest.mock import patch
    from common import reset_env, REPO
    import sys
    sys.path.insert(0, str(REPO))
    from experiments._workorders.ucontrast1.collector import Collector
    module._bon_plan = lambda n, cond, anc, s, g: module.sample_plan(n, cond, anc, s, g)
    original = env.step
    try:
        for task, episode, draw in [(4, 4, 64), (1, 10, 64), (5, 5, 71)]:
            actions = []
            def record(action):
                actions.append(action.copy())
                return original(action)
            env.step = record
            with patch.object(runtime, 'H', 160):
                row = run_draw(module, env, dict(task=task, episode=episode, draw=draw, arm='N'), move=1)
            env.step = original
            obs, info = reset_env(env, task, episode)
            goal = info['goal']
            module.torch.manual_seed(7*task+episode)
            col = Collector('A', 0., 16)
            module.ORACLE_COLLECTOR = col
            col.begin(task, episode, draw, obs, goal)
            module._reset_grad_cache()
            module._reseed_shuf(1000*task+episode)
            points, oracle_actions = [obs[:2].copy()], []
            stop = False
            while len(oracle_actions) < 160 and not stop:
                for action in module.policy_chunk(obs, goal, 0, True):
                    action = col.action(action)
                    oracle_actions.append(action.copy())
                    obs, _, term, trunc, info = env.step(action)
                    points.append(obs[:2].copy())
                    stop = bool(info.get('success') or term or trunc or len(oracle_actions) >= 160)
                    if stop:
                        break
            assert np.asarray(actions).tobytes() == np.asarray(oracle_actions).tobytes(), 'oracle A actions differ'
            assert np.asarray(row['xy'], np.float32).tobytes() == np.asarray(points, np.float32).tobytes(), 'oracle A XY differ'
    finally:
        module.ORACLE_COLLECTOR = None
        env.step = original
    print('PROTOCOL_EQUIVALENCE_PASS: 3 x 160 real maze steps, frozen oracle A, actions + XY byte-identical')

def main():
    dataset_dir = os.environ.get('PROBE_TEST_DATASET_DIR', '/home/cymaxwelllee/data/ogbench')
    module = load_frozen(dataset_dir)
    assert module.device == 'cpu', 'CPU selftest must not claim GPU evidence'
    assert not hasattr(module, 'env'), 'legacy environment loop must not run on import'
    env = module.ogbench.make_env_and_datasets(module.ENV_NAME, env_only=True)
    builder = build()
    interface = None
    try:
        # Genuine encode, false mask, EMA head, decode on both geometries.
        for task in (4, 5):
            geom = builder['geometries'][str(task)]
            for arm in 'AB':
                raw = [env.unwrapped.ij_to_xy(c) for c in geom['routes'][arm]]
                interface = interface_check(module, raw)
                assert interface['encoder_shape'] == [1, 8, 256]
                assert interface['head_shape'] == [1, 4, 2]
                u = m1.encode_trajectory(module, raw)
                with module.torch.no_grad():
                    decoded = module._dec(u, module.normstate(raw[0]))
                assert tuple(decoded.shape) == (1, 128, 2)
                assert bool(module.torch.isfinite(decoded).all())
        # Use a small explicit synthetic env for actual loaded-model P/Q reruns.
        # Five observed steps include a partial final chunk; never official evidence.
        class Space:
            def seed(self, seed):
                self.np_random = np.random.default_rng(seed)
        class TinyEnv:
            action_space = Space()
            def __init__(self):
                self.unwrapped = self
            def reset(self, seed, options):
                self.np_random = np.random.default_rng(seed)
                self.xy = np.array([0., 0.], np.float32)
                self.t = 0
                return self.xy.copy(), {'goal': np.array([8., 8.], np.float32)}
            def step(self, action):
                self.t += 1
                self.xy = (self.xy + action*.1).astype(np.float32)
                return self.xy.copy(), 0., False, self.t == 5, {'success': False}
        tiny = TinyEnv()
        for task in (4, 5):
            for arm, route in [('P', 'A'), ('Q', 'B')]:
                p = next(p for p in m1.rollout_plan(builder) if p['task'] == task and
                         p['arm'] == arm and p['kind'] == 'rollout')
                points = [env.unwrapped.ij_to_xy(c) for c in builder['geometries'][str(task)]['routes'][route]]
                u = m1.encode_trajectory(module, points)
                first = run_draw(module, tiny, p, move=1, u=u)
                module._GRAD_CACHE['u'] = module.torch.ones_like(u)
                second = run_draw(module, tiny, p, move=1, u=u)
                assert first == second, 'actual P/Q path including RNG/cache reset must reproduce'
                assert first['observed_steps'] == 5 and len(first['xy']) == 6
                assert first['head_u_hashes'] == [digest_array(u.cpu().numpy())]*2
                assert not first['flow_calls']
                first['synthetic'] = True
                fp = {(task, p['episode']): (first['initial_sha256'], first['goal_sha256'])}
                harvest.validate_rows([first], {harvest.key(first)}, fp, move=1, fixture=True)
        # Real env, real policy, legal waypoint: terminate on explicit short cap.
        p = m3.rollout_plan(builder, smoke=True)[0]
        waypoint = env.unwrapped.ij_to_xy(builder['geometries'][str(p['task'])]['w'][p['arm']])
        row = run_draw(module, env, p, move=3, waypoint=waypoint, cap_steps=1)
        assert row['termination_reason'] == 'cap' and row['observed_steps'] == 1
        assert len(row['xy']) == 2 and row['flow_calls'] == [{'step': 0, 'target': 'w'}]
        assert row['seeds']['flow_seed'] == 7*p['task']+p['episode']+1_000_003*64
        report = harvest.recompute_move3(row, waypoint, geometry=builder['geometries'][str(p['task'])],
                                         xy_to_cell=env.unwrapped.xy_to_ij)
        assert report['category'] == 'd2' and not report['invalid']
        # CPU evidence remains explicitly synthetic when entering fixture validator.
        row['synthetic'] = True
        fp = {(row['task'], row['episode']): (row['initial_sha256'], row['goal_sha256'])}
        harvest.validate_rows([row], {harvest.key(row)}, fp, move=3, fixture=True)
        protocol_equivalence(module, env)
        print('REAL_CPU_PASS: M9 import, s33/EMA, encode_u false mask, head, decode, '
              'four P/Q same-u reruns, fresh-flow seed, legal-w short cap; GPU evidence=false')
        print(json.dumps(interface, sort_keys=True))
    finally:
        env.close()


if __name__ == '__main__':
    main()
