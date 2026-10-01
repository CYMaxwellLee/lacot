"""Independent evidence validation/recomputation; scientific release is separate.

No harness classification, success aggregate, compliance label, or claimed gate
pass is accepted as evidence. The CLI validates receipts and raw runtime rows
before recomputing trace-based readout; GPU smoke remains a lead release gate.
"""
import argparse
from collections import defaultdict
import json
from math import ceil, dist
from pathlib import Path
import numpy as np
from common import (H, RHO, PIN, SOURCE, COLLECTOR, HERE, CARD, Blocked, canonical, digest_array,
                    file_sha, seeds, sha, verify_source, write_json, code_hashes, verify_contract)
import rules


class Rejected(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Rejected(message)


def key(row):
    return row['task'], row['episode'], row['arm'], row['draw']


def historical_fingerprints(gate):
    """Verify the *original* n64 artifacts; gate0b does not contain reset hashes."""
    sources = []
    for arm in ('A', 'B'):
        source = gate['receipt']['src_'+arm]
        path = Path(source['path'])
        if not path.is_file():
            raise Blocked(f'BLOCKED: historical reset reference unavailable: {path}')
        require(file_sha(path) == source['sha256'], 'historical artifact hash mismatch')
        artifact = json.loads(path.read_text())
        fingerprints = {}
        for task in artifact['tasks']:
            k = task['task'], task['episode']
            require(k not in fingerprints, 'duplicate historical question')
            draws = task['draws']
            require(sorted(d['draw'] for d in draws) == list(range(64)), 'historical draw coverage')
            pairs = {(d['initial_sha256'], d['goal_sha256']) for d in draws}
            require(len(pairs) == 1, 'historical reset pairing')
            fingerprints[k] = next(iter(pairs))
        sources.append(fingerprints)
    require(sources[0] == sources[1], 'historical A/B fingerprints disagree')
    require(set(sources[0]) == {(r['task'], r['episode']) for r in gate['rows']},
            'historical question coverage')
    return sources[0]


def validate_rows(rows, expected_keys, fingerprints, *, move, fixture=False):
    """Structural rejection is distinct from valid protocol draws over call budget.

    Fixture acceptance is explicit and cannot enter production. Expected keys
    and fingerprints come from trusted builder/n64 inputs, never the rows.
    """
    seen = set()
    for row in rows:
        k = key(row)
        require(k not in seen, 'duplicate row')
        require(k in expected_keys, 'unexpected/contaminated row')
        seen.add(k)
        require(row.get('synthetic') is fixture, 'synthetic/production contamination')
        require(row.get('polluted') is False, 'polluted row')
        require(row['source_sha256'] == PIN, 'wrong frozen source fingerprint')
        require(row['seeds'] == seeds(k[0], k[1], k[3], move == 1 and k[2] != 'N'), 'wrong seed')
        expected_fp = fingerprints[(k[0], k[1])]
        require((row['initial_sha256'], row['goal_sha256']) == expected_fp, 'wrong reset fingerprint')
        initial = np.asarray(row['initial'], dtype=row['observation_dtype'])
        goal = np.asarray(row['goal'], dtype=row['goal_dtype'])
        require(digest_array(initial) == expected_fp[0] and digest_array(goal) == expected_fp[1],
                'raw reset does not match fingerprint')
        xy = np.asarray(row['xy'], dtype=np.float32)
        steps = row.get('observed_steps')
        require(type(steps) is int and 0 < steps <= H, 'invalid observed steps')
        require(xy.shape == (steps+1, 2) and np.isfinite(xy).all(), 'missing/nonfinite full XY trace')
        require(np.array_equal(xy[0], initial[:2].astype(np.float32)), 'initial XY mismatch')
        reason = row.get('termination_reason')
        require(reason in ('success', 'terminated', 'truncated', 'horizon', 'cap', 'stuck'),
                'missing/unknown termination reason')
        require(reason != 'horizon' or steps == H, 'short horizon trace')
        require(not row.get('padded', False) and not row.get('spliced', False), 'padding/splicing forbidden')
        require(row['trace_sha256'] == digest_array(xy), 'trace hash mismatch')
        require(len(row['goal_success']) == steps+1 and all(type(v) is bool for v in row['goal_success']),
                'missing per-step success evidence')
        require(not any(row['goal_success'][:-1]), 'observations after first success')
        require((reason == 'success') == row['goal_success'][-1], 'success/reason mismatch')
        require(row.get('success') is row['goal_success'][-1], 'success summary mismatch')
        validate_rng(row)
        calls = row['flow_calls']
        require(all(set(c) == {'step', 'target'} and type(c['step']) is int and
                    0 <= c['step'] < steps and c['target'] in ('w', 'g') for c in calls), 'bad flow ledger')
        require(type(row['chunk_steps']) is int and row['chunk_steps'] == 4, 'bad chunk size')
        if move == 1 and k[2] == 'N':
            require(bool(calls) and calls == [dict(step=t, target='g') for t in range(0, steps, 4)],
                    'N fresh-flow chunk ledger')
        if move == 1 and k[2] != 'N':
            require(not calls, 'injection overwritten by flow')
            require(len(row['head_u_hashes']) == ceil(steps/row['chunk_steps']), 'missing head chunks')
            require(all(h == row['encoded_u_sha256'] for h in row['head_u_hashes']), 'E0 head delivery')
    require(seen == set(expected_keys), 'missing row')


def validate_receipt(receipt, builder):
    require(receipt['source_sha256'] == PIN, 'receipt source pin')
    require(receipt['source_path'] == str(SOURCE), 'receipt M9 source path')
    require(receipt['collector_sha256'] == file_sha(COLLECTOR), 'receipt collector hash')
    require(receipt['builder_sha256'] == sha(canonical(builder)), 'receipt builder hash')
    require(receipt['question_sha256'] == sha(canonical(builder['questions'])), 'receipt question hash')
    require(receipt['gate_sha256'] == builder['gate_sha256'], 'receipt gate hash')
    require(receipt['card_sha256'] == verify_contract(), 'receipt frozen contract hash')
    require(receipt['code_sha256'] == code_hashes(),
            'receipt harness code hash')
    interface = receipt.get('interface')
    require(isinstance(interface, dict) and interface.get('encoder_shape') == [1, 8, 256]
            and interface.get('head_shape') == [1, 4, 2] and interface.get('mask_all_false') is True
            and interface.get('source_sha256') == PIN
            and interface.get('head_u_hashes') == [interface.get('encoded_u_sha256')]
            and isinstance(interface.get('encoded_u_sha256'), str)
            and len(interface['encoded_u_sha256']) == 64, 'E0 receipt interface')
    require(receipt['H'] == H and receipt['rho'] == RHO, 'receipt protocol')
    require(receipt['env'] == builder['env'] and receipt['offline_only'] is True
            and receipt['training_use_prohibited'] is True, 'receipt scope')



def validate_rng(row):
    import torch
    before, after = row.get('rng_before'), row.get('rng_after')
    keys = {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'environment', 'action_space'}
    require(isinstance(before, dict) and isinstance(after, dict) and
            keys <= before.keys() and keys <= after.keys(), 'missing archived RNG states')
    expected = torch.Generator(device='cpu').manual_seed(row['seeds']['torch_seed']).get_state().tolist()
    require(before['torch_cpu'] == expected, 'rng_before torch seed mismatch')
    if row['seeds']['flow_seed'] is not None:
        noise = torch.Generator(device='cpu').manual_seed(row['seeds']['noise_seed']).get_state().tolist()
        require(before.get('noise') == noise, 'rng_before noise seed mismatch')


def injection_points(plan, builder, by_key, cell_to_xy):
    task, arm = plan['task'], plan['arm']
    if plan['group'] == 'gate' and arm == 'P':
        return by_key[(task, plan['episode'], 'N', 64)]['xy']
    if arm == 'R':
        donor = builder['R_donor'][str(task)]
        task, route = donor['task'], donor['route']
    else:
        route = 'A' if arm == 'P' else 'B'
    return [cell_to_xy(c) for c in builder['geometries'][str(task)]['routes'][route]]


def hard_prerequisites(artifact, task, repeats, conformance):
    receipt = artifact['receipt']
    declared = receipt.get('hard_prerequisites', {})
    if not isinstance(declared, dict):
        declared = {}
    hard = {k: declared.get(k) is True for k in rules.HARD_PREREQUISITES}
    settings = receipt.get('runtime', {}).get('deterministic_settings', {})
    required = dict(deterministic=True, warn_only=False, cudnn_benchmark=False,
                    cudnn_deterministic=True, tf32=False, sdpa='math',
                    fill_uninitialized_memory=True, threads=1, dataloader_workers=0)
    hard['torch_deterministic'] &= all(type(settings.get(k)) is type(v) and settings.get(k) == v
                                         for k, v in required.items())
    hard['dtype_batch_backend_frozen'] &= (settings.get('dtype') == 'float32' and
        settings.get('batch') == 1 and settings.get('cublas_workspace_config') in (':4096:8', ':16:8'))
    selected = [r for r in repeats if r['task'] == task]
    hard['same_u_rerun'] = ({r['arm'] for r in selected} == {'P', 'Q'} and len(selected) == 2
                           and all(conformance[str(key(r))] for r in selected))
    return hard


def combine(move1, move3):
    require(move1.get('move') == 1 and move3.get('move') == 3, 'combine requires both moves')
    require(move1['smoke'] == move3['smoke'] and move1['synthetic'] == move3['synthetic'],
            'combine scope mismatch')
    layers = {}
    delivery_ok = all(layer['E0']['passed'] for layer in move1['layers'].values())
    for task in (4, 5):
        one, three = move1['layers'][str(task)], move3['layers'][str(task)]
        if not delivery_ok:
            result = dict(exit='E0', action='全停：修編碼-注入管線')
        elif not one['representation']['passed']:
            result = dict(exit='E1', action='該層除名：表示層不可判')
        elif move1['gate']['qualified'] < 5 and not move1['smoke']:
            result = dict(exit='gate 樣本不足', action='探針改期')
        elif not one['complete'] or len(three['grids']) != {4: 14, 5: 6}[task]:
            result = dict(exit='INCOMPLETE', action='smoke/partial inventory; no full-layer decision')
        else:
            result = rules.exits(delivery_ok=one['E0']['passed'],
                representation_ok=one['representation']['passed'], failures=one['failures'],
                n_entries=one['n_entries'], task=task, grids=three['grids'],
                forward=three['discordant_forward'], reverse=three['discordant_reverse'],
                move1=one['interpretation'])
        layers[str(task)] = dict(result=result, move1=one, move3=three,
            gate_status=move1['gate']['status'], E2_lower_bound=one['E2']['lower_bound'],
            axis1_deprioritized=three['axis1_deprioritized'],
            E0=one['E0'], E1=one['E1'], E2=one['E2'], E3=one['E3'], E4=result.get('readout'))
    return dict(layers=layers, smoke=move1['smoke'], synthetic=move1['synthetic'], scientific_release=False)

def exclusive_crossing(row, geometry, xy_to_cell):
    """v6.1: at least one own exclusive cell, zero opposite exclusive cells.

    Single-route task 2 verifies entry into its own route interior; no other arm.
    """
    arm = row['arm']
    require(arm in ('A', 'B'), 'crossing requires waypoint arm')
    visited = {tuple(xy_to_cell(p)) for p in row['xy']}
    if geometry.get('n_routes') == 1:
        require(arm == 'A', 'single route has no B arm')
        return bool(visited & set(map(tuple, geometry['routes']['A'][1:-1])))
    own = set(map(tuple, geometry['exclusive'][arm]))
    other = set(map(tuple, geometry['exclusive']['B' if arm == 'A' else 'A']))
    return bool(visited & own) and not bool(visited & other)


def recompute_move3(row, waypoint, crossing_verifier=None, *, geometry=None, xy_to_cell=None):
    """Recompute d1-d5 from trace and raw events; never read row.classification.

    Production callers supply builder geometry and the environment XY mapping;
    explicit verifier injection is reserved for synthetic rule fixtures.
    """
    arm, xy = row['arm'], row['xy']
    w = None if waypoint is None else next((t for t, p in enumerate(xy)
                                          if dist(p, waypoint) < RHO), None)
    g = next((t for t, success in enumerate(row['goal_success']) if success), None)
    calls, chunk = len(row['flow_calls']), row['chunk_steps']
    invalid = []
    if calls > ceil(H/chunk)+2:
        invalid.append('budget-exceeded')
    # Independent expected schedule: discard remaining w actions on a hit.
    expected = []
    t = 0
    while t < row['observed_steps']:
        target = 'w' if waypoint is not None and (w is None or t < w) else 'g'
        expected.append(dict(step=t, target=target))
        t = w if target == 'w' and w is not None and t < w < t+chunk else t+chunk
    if row['flow_calls'] != expected:
        invalid.append('mid-chunk-switch-or-call-ledger-anomaly')
    crossing = None
    if w is not None and g is not None and g >= w:
        if geometry is not None and xy_to_cell is not None:
            crossing = exclusive_crossing(row, geometry, xy_to_cell)
        else:
            require(row.get('synthetic') is True and crossing_verifier is not None,
                    'production crossing requires geometry and XY mapping')
            crossing = bool(crossing_verifier(row))
    if invalid:
        category = None
    elif arm == 'C':
        category = 'success' if g is not None else 'failure'
    elif g is not None and (w is None or g < w):
        category = 'd1'
    elif w is None:
        category = 'd2'
    elif g is None:
        category = 'd3'
    else:
        category = 'd4' if crossing else 'd5'
    return dict(category=category, invalid=invalid, w_step=w, g_step=g,
                crossing=crossing, bypass_g=category == 'd1')


def first_entry(xy, exclusive, xy_to_cell):
    a, b = set(map(tuple, exclusive['A'])), set(map(tuple, exclusive['B']))
    require(not a.intersection(b), 'nonexclusive route segments')
    for t, p in enumerate(xy):
        cell = tuple(xy_to_cell(p))
        if cell in a or cell in b:
            return dict(route='A' if cell in a else 'B', step=t, cell=[int(c) for c in cell])
    return dict(route='O', step=None, cell=None)


def recompute_move1(p, q, geometry, xy_to_cell):
    pa, qa = np.asarray(p['xy'], np.float32), np.asarray(q['xy'], np.float32)
    pe = first_entry(pa, geometry['exclusive'], xy_to_cell)
    qe = first_entry(qa, geometry['exclusive'], xy_to_cell)
    return dict(P=pe, Q=qe, different=pa.tobytes() != qa.tobytes())


def summarize_move3(rows, builder, waypoint_xy, xy_to_cell):
    grouped, details = defaultdict(dict), []
    for row in rows:
        k = row['task'], row['episode']
        geometry = builder['geometries'][str(row['task'])]
        waypoint = None if row['arm'] == 'C' else waypoint_xy(geometry['w'][row['arm']])
        result = recompute_move3(row, waypoint, geometry=geometry, xy_to_cell=xy_to_cell)
        grouped[k].setdefault(row['arm'], []).append(result['category'])
        details.append(dict(task=k[0], episode=k[1], arm=row['arm'], draw=row['draw'], **result))
    questions = []
    for (task, episode), arms in sorted(grouped.items()):
        result = (dict(control=True, arms={a: rules.arm_stats(v) for a, v in arms.items()}) if task == 2
                  else rules.question_grid(arms['A'], arms['B'], arms['C']))
        questions.append(dict(task=task, episode=episode, **result))
    layers = {}
    for task in (4, 5):
        qs = [q for q in questions if q['task'] == task]
        eligible = [q for q in qs if all(not q['arms'][a]['insufficient'] for a in ('B', 'C'))]
        forward = sum(q['arms']['B']['counts'].get('d4', 0) > 0 and
                      q['arms']['C']['counts'].get('success', 0) == 0 for q in eligible)
        reverse = sum(q['arms']['B']['counts'].get('d4', 0) == 0 and
                      q['arms']['C']['counts'].get('success', 0) > 0 for q in eligible)
        layers[str(task)] = dict(grids=[q['grid'] for q in qs], discordant_forward=forward,
            discordant_reverse=reverse, exact_p=rules.exact_one_sided(forward, reverse),
            axis1_deprioritized=forward < 3, paired_questions=len(eligible),
            excluded_questions=[q['episode'] for q in qs if q not in eligible])
    return dict(draws=details, questions=questions, layers=layers)



def validate_artifact(artifact, builder, fingerprints, cell_to_xy, xy_to_cell, *, fixture=False):
    """Independently rederive conditional coverage and trace-based readout."""
    import harness_move1 as m1
    import harness_move3 as m3
    require(artifact['builder'] == builder, 'artifact builder differs from frozen inputs')
    validate_receipt(artifact['receipt'], builder)
    move, smoke = artifact['move'], artifact['receipt']['smoke']
    require(move in (1, 3) and type(smoke) is bool, 'invalid artifact protocol')
    plan = (m1 if move == 1 else m3).rollout_plan(builder, smoke)
    rows, repeats = artifact['rows'], artifact['conformance_rows']
    possible = {key(p) for p in plan if p.get('kind', 'rollout') == 'rollout'}
    actual = {key(row) for row in rows}
    require(actual <= possible, 'unexpected row')
    validate_rows(rows, actual, fingerprints, move=move, fixture=fixture)
    if move == 3:
        require(actual == possible and not repeats and not artifact['skipped'], 'Move 3 coverage')
        for row in rows:
            geom = builder['geometries'][str(row['task'])]
            expected_w = None if row['arm'] == 'C' else cell_to_xy(geom['w'][row['arm']])
            require('waypoint_xy' in row and (row['waypoint_xy'] is None if expected_w is None else
                    np.array_equal(row['waypoint_xy'], expected_w)), 'calibration waypoint content')
        return dict(summarize_move3(rows, builder, cell_to_xy, xy_to_cell), move=3, smoke=smoke,
                    synthetic=fixture, scientific_release=False)
    by_key = {key(row): row for row in rows}
    gate = [p for p in plan if p['group'] == 'gate' and p['arm'] == 'N']
    require(all(key(p) in actual for p in gate), 'missing gate N prerequisite')
    qualified = [p for p in gate if any(by_key[key(p)]['goal_success'])]
    representation = {}
    for task in ('4', '5'):
        evidence = artifact['representation_evidence'][task]
        ua, ub = (np.asarray(evidence['u_'+arm], np.float32) for arm in 'AB')
        require(ua.shape == ub.shape == (1, 8, 256) and
                np.isfinite(ua).all() and np.isfinite(ub).all(), 'invalid encoded representation')
        noise = artifact['receipt']['calibration']['noise_band'][task]
        require(np.isfinite(noise) and noise >= 0, 'invalid calibrated noise band')
        identities = []
        for arm in 'AB':
            points = np.asarray(evidence['decoded_'+arm], np.float32)
            require(points.shape == (128, 2) and np.isfinite(points).all(), 'invalid decoded trace')
            identities.append(first_entry(points, builder['geometries'][task]['exclusive'], xy_to_cell)['route'])
        distance = float(np.linalg.norm(ua.astype(np.float64)-ub.astype(np.float64)))
        representation[task] = dict(passed=distance > noise and identities == ['A', 'B'],
                                    distance=distance, noise_band=noise, roundtrip=identities)
    qualified_keys = {(p['task'], p['episode']) for p in qualified}
    expected, expected_repeats, expected_skips = set(), set(), set()
    for p in plan:
        skip = ((p['group'] == 'main' and not smoke and len(qualified) < 5) or
                (p['conditional_on_N_success'] and (p['task'], p['episode']) not in qualified_keys) or
                (p['conditional_on_representation'] and not representation[str(p['task'])]['passed']))
        if skip:
            expected_skips.add((*key(p), p['kind']))
        elif p['kind'] == 'determinism-verification':
            expected_repeats.add(key(p))
        else:
            expected.add(key(p))
    require(actual == expected, 'missing/extra conditional rollout')
    for p in plan:
        if key(p) not in actual:
            continue
        row = by_key[key(p)]
        require(row['injection_source'] == p['source'], 'wrong injection source')
        if p['arm'] == 'R':
            require(row['donor'] == builder['R_donor'][str(p['task'])], 'wrong donor')
        if p['group'] == 'gate' and p['arm'] == 'P':
            n = by_key[(p['task'], p['episode'], 'N', 64)]
            require(row['source_trace_sha256'] == n['trace_sha256'], 'gate P is not self-rollout-u')
        if p['arm'] != 'N':
            points = injection_points(p, builder, by_key, cell_to_xy)
            require(row.get('injection_traj_sha256') == digest_array(np.asarray(points, np.float64)),
                    'wrong injection trajectory content')
        if p['group'] == 'main' and p['arm'] in 'PQ':
            route = 'A' if p['arm'] == 'P' else 'B'
            u = np.asarray(artifact['representation_evidence'][str(p['task'])]['u_'+route], np.float32)
            require(row['encoded_u_sha256'] == digest_array(u), 'wrong main route injection')
    skipped = artifact['skipped']
    require(len(skipped) == len(expected_skips) and
            {(*key(p), p['kind']) for p in skipped} == expected_skips, 'incorrect skipped plan')
    validate_rows(repeats, expected_repeats, fingerprints, move=1, fixture=fixture)
    entries = {k: first_entry(row['xy'], builder['geometries'][str(row['task'])]['exclusive'], xy_to_cell)
               for k, row in by_key.items()}
    reproduced = {'P': 0, 'R': 0}
    for task, episode in qualified_keys:
        n = entries[(task, episode, 'N', 64)]['route']
        for arm in 'PR':
            k = task, episode, arm, 64
            reproduced[arm] += bool(n != 'O' and entries[k]['route'] == n and any(by_key[k]['goal_success']))
    conformance = {}
    for row in repeats:
        k = key(row)
        require(row.get('injection_traj_sha256') == by_key[k].get('injection_traj_sha256'),
                'rerun injection trajectory differs')
        require(row['kind'] == 'determinism-verification', 'rerun is not conformance')
        require(row['encoded_u_sha256'] == by_key[k]['encoded_u_sha256'], 'rerun is not same u')
        a, b = np.asarray(by_key[k]['xy'], np.float32), np.asarray(row['xy'], np.float32)
        conformance[str(k)] = a.shape == b.shape and a.tobytes() == b.tobytes()
    layers = {}
    for task in (4, 5):
        details = []
        for q in builder['questions']['main']:
            if q['task'] != task:
                continue
            pk, qk = (task, q['episode'], 'P', 64), (task, q['episode'], 'Q', 64)
            if pk in actual and qk in actual:
                details.append(dict(episode=q['episode'], **recompute_move1(
                    by_key[pk], by_key[qk], builder['geometries'][str(task)], xy_to_cell)))
        layers[str(task)] = dict(representation=representation[str(task)], pairs=details,
            n_diff=sum(d['different'] for d in details),
            response=rules.paired_response([(d['P']['route'], d['Q']['route']) for d in details]))
        layer = layers[str(task)]
        pairs = [(d['P']['route'], d['Q']['route']) for d in details]
        failures, n_entries = [], []
        for d in details:
            episode = d['episode']
            failures.append(tuple(entries[(task, episode, a, 64)]['route'] == 'O' or
                by_key[(task, episode, a, 64)]['termination_reason'] == 'stuck' or
                bool(by_key[(task, episode, a, 64)].get('runtime_errors')) for a in 'PQ'))
            n_entries.append(entries[(task, episode, 'N', 64)]['route'])
        complete = len(details) == {4: 14, 5: 6}[task]
        failure = rules.e2(failures, task) if complete else None
        n_ceiling = rules.ceiling(n_entries, task) if complete else None
        hard = hard_prerequisites(artifact, task, repeats, conformance)
        absorbed = rules.adsorption(pairs, builder['geometries'][str(task)]['demo_label'],
                                    n_ceiling is not False)
        interpretation = ('吸附' if absorbed['established'] else
                          '內容不敏感' if layer['n_diff'] == 0 and complete else layer['response']['action'])
        layer.update(failures=failures, n_entries=n_entries, complete=complete,
            E0=dict(passed=True), E1=dict(passed=representation[str(task)]['passed']),
            E2=dict(result=failure, lower_bound=not artifact['receipt']['calibration'].get('calibrated', False)
                    or not all(r.get('stuck_check_enabled') for r in rows if r['task'] == task)),
            E3=dict(n_ceiling=n_ceiling), adsorption=absorbed, interpretation=interpretation,
            first_layer=dict(hard_prerequisites=hard, exact_zero_qualified=rules.exact_zero_qualified(hard),
                             language=rules.content_sensitivity(layer['n_diff'], hard)))
    return dict(move=1, gate=rules.behavior_gate(len(qualified), reproduced['P'], reproduced['R']),
                layers=layers, conformance=conformance, smoke=smoke,
                synthetic=fixture, scientific_release=False)


def fixture_report():
    """Reviewable CPU decision table; emphatically not an experiment result."""
    c = ['failure']*16
    grid_examples = {
        'C-first': rules.question_grid([None]*16, [None]*16, ['success']+[None]*15),
        'both-rescue': rules.question_grid(['d4']*16, ['d4']*16, c),
        'A-only': rules.question_grid(['d4']*16, ['d3']*16, c),
        'B-only': rules.question_grid(['d3']*16, ['d4']*16, c),
        'deep': rules.question_grid(['d3']*16, ['d3']*16, c),
        'd5-unverified': rules.question_grid(['d5']*16, ['d5']*16, c),
        'v5-tie': rules.question_grid(['d3']*8+['d5']*8, ['d3']*8+['d5']*8, c),
        'invalid-flow': rules.question_grid(['d3']*7+[None]*9, ['d3']*16, c),
    }
    args = dict(delivery_ok=True, representation_ok=True, failures=[(False, False)]*14,
                n_entries=['O']*14, task=4, grids=[rules.BOTH]*14,
                forward=8, reverse=0, move1='跟隨者')
    exits = {name: rules.exits(**dict(args, **change)) for name, change in [
        ('E0', {'delivery_ok': False}), ('E1', {'representation_ok': False}),
        ('E2', {'failures': [(True, False)]*8+[(False, False)]*6}),
        ('E3', {'n_entries': ['A']*11+['O']*3}), ('E4', {})]}
    return dict(synthetic=True, production_evidence=False, scope='CPU rule fixtures only',
                grids=grid_examples, exits=exits,
                nine_cells=rules.paired_response([(a, b) for a in 'ABO' for b in 'ABO']),
                ignored_u_gate=rules.behavior_gate(8, 8, 8),
                operational_status='CPU_FIXTURES_ONLY; GPU smoke and scientific release pending')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--input', type=Path, help='runtime result.json to independently validate')
    p.add_argument('--move1', type=Path, help='combine: Move 1 artifact')
    p.add_argument('--move3', type=Path, help='combine: Move 3 artifact')
    p.add_argument('--fixtures', action='store_true', help='CPU rule fixtures; never scientific readout')
    args = p.parse_args()
    if args.fixtures:
        write_json(args.out, fixture_report())
        return
    try:
        verify_source()
        if args.input is None and not (args.move1 and args.move3):
            raise Blocked('BLOCKED: --input runtime result.json is required')
        from builder import build
        from common import GATE, ENV
        import ogbench
        from artifacts import load_artifact
        builder = build()
        fingerprints = historical_fingerprints(json.loads(GATE.read_text()))
        env = ogbench.make_env_and_datasets(ENV, env_only=True)
        try:
            def read(path):
                artifact = load_artifact(path)
                return artifact, validate_artifact(artifact, builder, fingerprints,
                                                  env.unwrapped.ij_to_xy, env.unwrapped.xy_to_ij)
            if args.move1 and args.move3:
                a, one = read(args.move1)
                b, three = read(args.move3)
                require(a['receipt']['calibration'] == b['receipt']['calibration'] and
                        a['receipt'].get('runtime') == b['receipt'].get('runtime'),
                        'combine calibration/runtime mismatch')
                result = combine(one, three)
                result['inputs'] = {str(path): file_sha(path) for path in (args.move1, args.move3)}
            else:
                _, result = read(args.input)
            write_json(args.out, result)
        finally:
            env.close()
    except (Blocked, ValueError, KeyError) as e:
        p.exit(2, str(e)+'\n')


if __name__ == '__main__':
    main()
