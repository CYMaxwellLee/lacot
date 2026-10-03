"""Read machine question inventories only; never manually enumerate episodes."""
import argparse
from collections import Counter
from pathlib import Path
from common import write_json
import harness_detour as d

MAIN = Path('/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/readout-2026-10-02/harvest-m3.json')
MAIN_PIN = '46de59ed8d46f7990e514c4c7fd9e52a4e1c0ecc50c9c31cf07504678fe421ce'


def build_plan(main_path=MAIN):
    main_data = d.checked_json(main_path, MAIN_PIN)
    traps = d.load_trapset()
    builder = d.checked_json(d.HERE/'builder.json',traps['inputs']['builder.json'])
    main = [dict(task=q['task'],episode=q['episode']) for q in main_data['questions']]
    gate = builder['questions']['gate']
    if Counter(q['task'] for q in main) != {4:14,5:6,2:10} or Counter(q['task'] for q in gate) != {1:7,3:1}:
        raise ValueError('machine question inventory mismatch')
    if len({(q['task'],q['episode']) for q in main+gate}) != 38:
        raise ValueError('duplicate machine questions')
    def expand(qs, group, arms, draws):
        return [dict(**q,protocol=d.PROTOCOL,group=group,arm=a,draw=k,
                     seeds=d.seeds_detour(q['task'],q['episode'],k,a not in d.FLOW_ARMS))
                for q in qs for a in arms for k in draws]
    formal = expand(main,'main',('O3','F3','OF'),range(80,84)) + expand(gate,'gate',('Q-C','Q-O3','Q-F3'),range(80,82))
    first = lambda qs,t: min((q for q in qs if q['task']==t),key=lambda q:q['episode'])
    smoke = expand([first(main,t) for t in (4,5,2)],'main',('O3','F3','OF'),[80])
    smoke += expand([first(gate,t) for t in (1,3)],'gate',('Q-C','Q-O3','Q-F3'),[80])
    return dict(protocol=d.PROTOCOL,card_sha256=d.CARD_PIN,trapset_sha256=d.TRAPSET_PIN,
                main_sha256=MAIN_PIN,builder_sha256=traps['inputs']['builder.json'],
                questions=dict(main=main,gate=gate),formal=formal,smoke=smoke)


def validate_plan(plan):
    # Regenerate from the frozen local plan's question inventory and pinned input hashes.
    from common import canonical
    frozen = d.checked_json(d.HERE/'detour-plan.json')
    if canonical(plan) != canonical(frozen):
        raise ValueError('plan differs from frozen detour-plan.json')
    if (plan['protocol'],plan['card_sha256'],plan['trapset_sha256'],plan['main_sha256']) != (
            d.PROTOCOL,d.CARD_PIN,d.TRAPSET_PIN,MAIN_PIN):
        raise ValueError('plan provenance mismatch')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main',type=Path,default=MAIN)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    print(write_json(args.out,build_plan(args.main)))
