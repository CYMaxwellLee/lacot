"""Literal card decision rules, independent of neural/environment code.

Inputs to these pure rules are observations, not harness classifications.
Runtime evidence extraction whose metric is not frozen is deliberately separate.
"""
from collections import Counter
from math import ceil, comb

C_RESCUE = 'C 對照救回'
BOTH = '時程形＋岔路可執行'
A_ONLY = '時程形、alt 介入未見效'
B_ONLY = 'demo 段局部洞指紋'
DEEP = '深執行洞候選'
INSUFFICIENT = '介入觀測不足'
AMBIGUOUS = '介入觀測歧義'
UNVERIFIED = '介入疑似有效、穿越未核實'


def classify(arm, *, calls, chunk_steps, w_step, g_step, crossing, H=1000):
    if calls < 0 or chunk_steps < 1:
        raise ValueError('invalid call ledger')
    if calls > ceil(H/chunk_steps)+2:
        return None  # invalid, excluded from every classification/denominator
    for step in (w_step, g_step):
        if step is not None and (type(step) is not int or not 0 <= step <= H):
            raise ValueError('event outside episode')
    if arm == 'C':
        if w_step is not None:
            raise ValueError('C has no waypoint')
        return 'success' if g_step is not None else 'failure'
    if arm not in ('A', 'B'):
        raise ValueError('unknown Move 3 arm')
    if g_step is not None and (w_step is None or g_step < w_step):
        return 'd1'
    if w_step is None:
        return 'd2'
    if g_step is None:
        return 'd3'
    return 'd4' if crossing is True else 'd5'


def arm_stats(classes):
    counts = Counter(x for x in classes if x is not None)
    valid = sum(counts.values())
    return dict(counts=dict(counts), valid=valid, invalid=len(classes)-valid,
                w_miss_rate=counts['d2']/valid if valid else None,
                insufficient=valid < 8,
                w_unreliable=counts['d2'] > valid/2 if valid else False)


def question_grid(a, b, c):
    """Main-set only. Task 2 reports A/C stats, not a fictitious missing B."""
    stats = {name: arm_stats(values) for name, values in [('A', a), ('B', b), ('C', c)]}
    # C success is first. Even few valid C draws cannot erase observed success.
    if stats['C']['counts'].get('success', 0):
        grid = C_RESCUE
    elif any(v['insufficient'] or v['w_unreliable'] for v in stats.values()):
        grid = INSUFFICIENT
    else:
        ar = stats['A']['counts'].get('d4', 0) > 0
        br = stats['B']['counts'].get('d4', 0) > 0
        if ar or br:
            grid = BOTH if ar and br else A_ONLY if ar else B_ONLY
        elif sum(stats[a]['counts'].get('d1', 0) for a in ('A', 'B')) >= 8:
            grid = C_RESCUE
        else:
            combined = {k: sum(stats[arm]['counts'].get(k, 0) for arm in ('A', 'B'))
                        for k in ('d2', 'd3', 'd5')}
            winners = [k for k, v in combined.items() if v == max(combined.values())]
            grid = ({'d2': INSUFFICIENT, 'd3': DEEP, 'd5': UNVERIFIED}[winners[0]]
                    if len(winners) == 1 else AMBIGUOUS)
    return dict(grid=grid, bypass_version=grid == C_RESCUE and not stats['C']['counts'].get('success', 0),
                world_counted=grid not in (INSUFFICIENT, AMBIGUOUS), arms=stats)


def exact_one_sided(forward, reverse):
    """Conditional discordant-pair binomial tail, separately for each geometry."""
    n = forward + reverse
    return sum(comb(n, k) for k in range(forward, n+1)) / 2**n


def paired_response(pairs):
    table = {p+q: 0 for p in 'ABO' for q in 'ABO'}
    for p, q in pairs:
        if p not in 'ABO' or q not in 'ABO' or len(p) != 1 or len(q) != 1:
            raise ValueError('compliance must be A/B/O')
        table[p+q] += 1
    switch, anti = table['AB'], table['BA']
    action = ('跟隨者' if switch >= 5 and anti == 0 else
              '方向性訊號' if 1 <= switch < 5 or anti > 0 else '部分跟隨/不可判')
    return dict(table=table, n_switch=switch, n_anti=anti, action=action,
                calibrated_p=None, unit='question', descriptive=True)


def behavior_gate(qualified, p_reproduced, r_reproduced):
    if not 0 <= p_reproduced <= qualified or not 0 <= r_reproduced <= qualified:
        raise ValueError('gate count outside qualified set')
    if qualified < 5:
        status = 'gate 樣本不足'
    elif p_reproduced < ceil(qualified*3/4):
        status = '行為資格不足'
    elif r_reproduced > 2:
        status = 'gate-content-insensitive'
    else:
        status = 'passed'
    return dict(status=status, qualified=qualified, p_reproduced=p_reproduced,
                r_reproduced=r_reproduced, p_threshold=ceil(qualified*3/4),
                run_main=qualified >= 5)


def e2(failures, task):
    expected = {4: 14, 5: 6}[task]
    if len(failures) != expected:
        raise ValueError('E2 requires the entire frozen geometry question set')
    count = sum(bool(p or q) for p, q in failures)
    return dict(numerator=count, denominator=expected, stopped=count > expected/2)


def ceiling(n_entries, task):
    if len(n_entries) != {4: 14, 5: 6}[task]:
        raise ValueError('N geometry count')
    counts = Counter(n_entries)
    return max(counts['A'], counts['B']) >= {4: 11, 5: 5}[task]


def adsorption(pairs, demo_label, n_ceiling):
    if demo_label not in ('A', 'B') or n_ceiling:
        return dict(enabled=False, established=False)
    threshold = ceil(len(pairs)/2)
    positive = (sum(p == demo_label for p, q in pairs) >= threshold and
                sum(q == demo_label for p, q in pairs) >= threshold and
                sum(p == 'A' and q == 'B' for p, q in pairs) <= 1)
    return dict(enabled=True, established=positive, threshold=threshold)


HARD_PREREQUISITES = ('torch_deterministic', 'environment_policy_reset', 'rng_archived',
                      'dtype_batch_backend_frozen', 'head_deterministic',
                      'paired_protocol', 'same_u_rerun')


def exact_zero_qualified(hard_prerequisites):
    return isinstance(hard_prerequisites, dict) and all(
        hard_prerequisites.get(k) is True for k in HARD_PREREQUISITES)


def content_sensitivity(n_diff, hard_prerequisites):
    if not exact_zero_qualified(hard_prerequisites):
        return '行為差異觀測（不可歸因）'
    return '行為對注入張量有因果敏感性' if n_diff else '未觀察到內容敏感性'


def e4(grids, forward, reverse, move1):
    counts = Counter(g for g in grids if g not in (INSUFFICIENT, AMBIGUOUS))
    ordered = counts.most_common()
    dominant = (ordered[0][0] if ordered and ordered[0][1] -
                (ordered[1][1] if len(ordered) > 1 else 0) >= 2 else None)
    p = exact_one_sided(forward, reverse)
    insensitive = move1 in ('內容不敏感', '吸附')
    # First matching row only. A p-value cannot invent a second dominant row.
    if not counts:
        row, action = '無合格題', '探針失效：工程細讀報告呈裁'
    elif dominant == UNVERIFIED:
        row, action = UNVERIFIED, '修核實口徑後重判'
    elif dominant in (BOTH, B_ONLY) and p < .05:
        row = '岔路可執行'
        action = ('軸一劑量反應三組對照' if move1 == '跟隨者' else
                  '修讀出端（Astra 分段介面）' if insensitive else '加題補證後重判')
    elif dominant == A_ONLY:
        row = '時程形主導'
        action = '階層線＋讀出端並行診斷' if insensitive else '階層線另立案'
    elif dominant == DEEP:
        row, action = '深執行洞候選/alt 介入未見效', 'consumer 診斷/字典線'
    elif dominant == C_RESCUE:
        row, action = 'C 對照救回主導', 'support 假說重審、回 BoN 框內重讀'
    elif dominant in (BOTH, B_ONLY):
        row, action = 'B d4 主導但未站穩', '方向性訊號：補 episode 擴證工單呈裁'
    else:
        row, action = '混合格/幾何間分歧', '暫不分支：分層明細呈主人裁'
    return dict(row=row, action=action, dominant=dominant, counts=dict(counts),
                discordant_forward=forward, discordant_reverse=reverse, exact_p=p,
                axis1_deprioritized=forward < 3)


def exits(*, delivery_ok, representation_ok, failures, n_entries, task,
          grids, forward, reverse, move1):
    if not delivery_ok:
        return dict(exit='E0', action='全停：修編碼-注入管線')
    if not representation_ok:
        return dict(exit='E1', action='該層除名：表示層不可判')
    failure = e2(failures, task)
    if failure['stopped']:
        return dict(exit='E2', action='該層 consumer 判定停用：注入工程細讀', failure=failure)
    n_ceiling = ceiling(n_entries, task)
    return dict(exit='E3' if n_ceiling else 'E4', adsorption_disabled=n_ceiling,
                failure=failure, readout=e4(grids, forward, reverse, move1))
