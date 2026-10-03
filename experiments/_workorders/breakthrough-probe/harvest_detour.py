"""Fail-closed detour collector: frozen table authority and unconditional noise replay."""
import argparse
import json
from pathlib import Path
import numpy as np
import common
from common import canonical, digest_array, file_sha, sha, write_json
import harness_detour as d
from detour_plan import validate_plan


class Rejected(ValueError):
    pass


def require(test, message):
    if not test:
        raise Rejected(message)


def key(row):
    return tuple(row[k] for k in ('protocol','task','episode','arm','draw'))


def validate_table(table, runtime=None, fixture=False):
    require(table['protocol']==d.PROTOCOL and table['card_sha256']==d.CARD_PIN and
            table['source_sha256']==d.M9_PIN and table['trapset_sha256']==d.TRAPSET_PIN,'E1 provenance')
    require(fixture or table['synthetic'] is False,'mock E1 is not production evidence')
    require(table['code_sha256']==d.code_hashes(),'E1 generator code differs from frozen harness')
    if runtime is not None:
        require(table['runtime']==runtime,'E1/runtime deterministic settings or model mismatch')
    require(table['ruler_sha256']==d.RULER_PIN and table['cpu_e1_sha256']==d.CPU_E1_PIN,'E1 ruler provenance')
    require(table['threshold']==d.direction_threshold() and table['h']==3,'E1 threshold/h')
    traps=d.load_trapset()
    require(set(table['cells'])==set(traps['tasks'])==set(table['of_cells']),'E1 task inventory')
    for task,t in traps['tasks'].items():
        expected={k for k,v in t['oracle_table'].items() if len(v['route'])>4}
        require(set(table['cells'][task])==expected,'E1 missing/extra static cells (w=g excluded)')
        require(set(table['of_cells'][task])==set(t['oracle_table'])-{d.cell_key(t['g'])},'OF inventory')
        for c in table['cells'][task].values():
            validate_direction(c)
            require(c['w']!=t['g'] and c['h']==3 and c['tau_endpoint']==c['target_xy'],'E1 static target')
            require(np.isfinite([c['clearance'],c['frechet']]).all(),'nonfinite descriptive metrics')
            for field in ('u_sha256','delivered_sha256','tau_sha256'):
                require(isinstance(c[field],str) and len(c[field])==64,'E1 missing hash')


def validate_direction(record):
    expected=d.direction_metrics(record['direction_D'],record['direction_P'],record['direction_G'],d.direction_threshold())
    for k,v in expected.items(): require(record[k]==v,'E1 direction '+k)


def validate_runtime_direction(record, points, goal):
    validate_direction(record)
    tau=np.repeat(points[:1],128,axis=0) if np.linalg.norm(np.diff(points,axis=0),axis=1).sum()<1e-6 else d.resample(points)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(tau,axis=0),axis=1))]
    require(record['arc4_index']==min(127,int(np.searchsorted(arc,4.))),'runtime arc4 index')
    require(np.array_equal(record['direction_P'],points[1]-points[0]) and
            np.array_equal(record['direction_G'],goal-points[0]),'runtime direction reference')


def validate_receipt(receipt, plan, table_path, fixture=False):
    d.verify_pins()
    table = d.checked_json(table_path)
    validate_plan(plan)
    expected = dict(protocol=d.PROTOCOL,card_sha256=d.CARD_PIN,source_sha256=d.M9_PIN,
                    collector_sha256=d.COLLECTOR_PIN,trapset_sha256=d.TRAPSET_PIN,
                    code_sha256=d.code_hashes(),plan_sha256=sha(canonical(plan)),
                    e1_sha256=file_sha(table_path),H=d.H,sigma=common.SIGMA,env=common.ENV)
    for k,v in expected.items():
        require(receipt.get(k)==v,'receipt '+k)
    validate_table(table,receipt['runtime'],fixture)
    props = receipt['env_properties']
    require(props['success_timing']=='post' and props['add_noise_to_goal'] is True and props['goal_tol']==1.,
            'environment timing/goal contract')
    require(receipt['offline_only'] is True and receipt['training_use_prohibited'] is True,'receipt scope')
    return table


def validate_noise(row):
    """Never conditional on flow_seed. Replay exactly one randn per actual env step."""
    import torch
    before, after = row['rng_before'], row['rng_after']
    required = {'python','numpy','torch_cpu','torch_cuda','environment','action_space','noise'}
    require(required <= before.keys() and required <= after.keys(),'missing RNG states')
    gen = torch.Generator(device='cpu').manual_seed(row['seeds']['noise_seed'])
    require(before['noise']==gen.get_state().tolist(),'noise initial state mismatch')
    require(before['torch_cpu']==torch.Generator(device='cpu').manual_seed(row['seeds']['torch_seed']).get_state().tolist(),'policy initial RNG state')
    heads = np.asarray(row['head_actions'],np.float32)
    sent = np.asarray(row['actions'],np.float32)
    require(heads.shape==sent.shape==(row['observed_steps'],2),'noise action evidence shape')
    require(np.isfinite(heads).all() and np.isfinite(sent).all(),'nonfinite actions')
    replay = []
    for head in heads:
        eps = torch.randn(head.shape,generator=gen).numpy()
        replay.append(np.clip(np.clip(head,-1.,1.)+common.SIGMA*eps,-1.,1.).astype(np.float32))
    require(np.asarray(replay,np.float32).tobytes()==sent.tobytes(),'noise action sequence mismatch')
    require(after['noise']==gen.get_state().tolist(),'noise final state mismatch')


def validate_row(row, plan, table, xy_to_cell, cell_to_xy, *, fixture=False):
    try:
        _validate_row(row,plan,table,xy_to_cell,cell_to_xy,fixture=fixture)
    except (KeyError,TypeError,IndexError) as error:
        raise Rejected('missing/malformed row contract: '+str(error)) from error


def _validate_row(row, plan, table, xy_to_cell, cell_to_xy, *, fixture=False):
    require(key(row)==key(plan),'row identity')
    require(row['protocol']==d.PROTOCOL and plan['protocol']==d.PROTOCOL,'unknown protocol')
    arm = row['arm']
    require(arm in d.ARMS,'unknown arm')
    require(row['seeds']==d.seeds_detour(row['task'],row['episode'],row['draw'],arm not in d.FLOW_ARMS),'seed mismatch')
    require(row['source_sha256']==d.M9_PIN and (fixture or not row['synthetic']),'row source/synthetic')
    require(row['group']==plan['group'],'row group')
    require(row['env_properties']==dict(success_timing='post',add_noise_to_goal=True,goal_tol=1.),'row environment properties')
    n = row['observed_steps']
    require(type(n) is int and 0<n<=d.H,'H/steps')
    trace = np.asarray(row['xy'],np.float64)
    require(trace.shape==(n+1,2) and np.isfinite(trace).all(),'trace shape/nonfinite')
    require(digest_array(trace)==row['trace_sha256'],'trace hash')
    for field,dtype in (('initial','observation_dtype'),('goal','goal_dtype')):
        value = np.asarray(row[field],dtype=row[dtype])
        require(value.shape==(2,) and np.isfinite(value).all(),'reset field shape')
        require(digest_array(value)==row[field+'_sha256'],'reset fingerprint')
    require(np.array_equal(trace[0],row['initial'][:2]),'initial trace')
    successes = row['goal_success']
    require(len(successes)==n+1 and all(type(s) is bool for s in successes) and not successes[0], 'success trace')
    require(row['success']==any(successes) and not any(successes[1:-1]),'success termination timing')
    reason = ('success' if successes[-1] else 'terminated' if row['terminated'] else
              'truncated' if row['truncated'] else 'horizon' if n==d.H else None)
    require(reason is not None and row['termination_reason']==reason,'early stop / reason')
    require(row['chunk_steps']==4,'pinned CHUNK=4')
    chunks = row['chunks']
    require([c['step'] for c in chunks]==list(range(0,n,row['chunk_steps'])),'missing/duplicate chunk')
    require(row['flow_calls']==([dict(step=c['step'],target_xy=c['cond_target_xy']) for c in chunks] if arm in d.FLOW_ARMS else []),'E0 actual flow.sample calls')
    traps = d.load_trapset(); task = row['task']; free = d.free_cells(traps,task)
    previous = None
    for chunk in chunks:
        step = chunk['step']; end = min(n,step+4)
        require(chunk['end_step']==end and np.array_equal(chunk['end_xy'],trace[end]),'chunk displacement')
        require(np.array_equal(chunk['xy'],trace[step]),'chunk position')
        raw = tuple(int(x) for x in xy_to_cell(trace[step]))
        eff = raw if raw in free else min(free,key=lambda c:(float(np.sum((np.asarray(cell_to_xy(c))-trace[step])**2)),c))
        require(tuple(chunk['c_raw'])==raw and tuple(chunk['c_eff'])==eff,'raw/effective cell mismatch')
        goal=np.asarray(row['goal'],dtype=row['goal_dtype'])
        position=np.asarray(trace[step],dtype=row['observation_dtype'])
        g=tuple(traps['tasks'][str(task)]['g'])
        short=arm in d.SHORT_ARMS
        w,target,points=d.short_points(traps,task,eff,cell_to_xy,position,goal)
        if not short:
            w,target=list(g),goal
            points=np.asarray([position,goal]) if eff==g else d.oracle_points(traps,task,eff,cell_to_xy)
        require(chunk['w']==list(w) and np.array_equal(chunk['target_xy'],target) and
                np.array_equal(chunk['cond_target_xy'],target),'E0 supplied w/cond target')
        require(chunk['h']==(3 if short else None),'chunk h')
        if arm in d.FLOW_ARMS:
            require(chunk['mode']=='flow' and chunk['tau_sha256'] is None,'flow mode')
            if short:
                require(chunk['qualification_source']=='runtime-flow','F3 runtime qualification')
                validate_runtime_direction(chunk,points,goal)
                # F3 decoder endpoint distance is descriptive only, never an acceptance gate.
                require(np.isfinite(chunk['goal_distance']),'F3 decoded endpoint description')
            else:
                require(chunk['qualification']=='N/A' and chunk['qualification_source']=='flow-reference','Q-C reference')
            expected=chunk['expected_delivered_sha256']
        elif chunk['mode']=='零長度沿用':
            require(previous is not None and eff==g and np.linalg.norm(position-goal)<1e-6,'zero-length inheritance rule')
            require(chunk['qualification_source']=='runtime','zero-length source')
            for k in ('u_sha256','tau_sha256','tau_dtype','tau_endpoint','qualification'):
                require(chunk[k]==previous[k],'zero-length lost qualification/input')
            expected=previous['delivered_sha256']
        else:
            static=tuple(w)!=g if short else eff!=g
            require(chunk['mode']==('static' if static else 'goal' if eff==g else 'tail'),'qualification mode')
            require(chunk['source_cell']==list(eff),'source cell')
            require(chunk['tau_dtype']==str(points.dtype) and chunk['tau_sha256']==digest_array(points),'runtime/static tau hash')
            require(np.array_equal(chunk['tau_endpoint'],points[-1]),'encoding input endpoint')
            if short: require(np.array_equal(chunk['tau_endpoint'],target),'O3 encoding endpoint = target')
            if static:
                frozen=table['cells' if short else 'of_cells'][str(task)][d.cell_key(eff)]
                require(chunk['qualification_source']=='靜態表','static qualification source')
                for k in ('u_sha256','tau_sha256','qualification','clearance','frechet'):
                    require(chunk[k]==frozen[k],'E0 frozen table '+k)
                if short:
                    for k in ('direction_D','direction_P','direction_G','cos_plan','cos_greedy','cos_plan_greedy','arc4_index'):
                        require(chunk[k]==frozen[k],'E1 frozen direction '+k)
                expected=frozen['delivered_sha256']
            else:
                require(chunk['qualification_source']=='runtime','tail qualification source')
                require(np.linalg.norm(points[0]-points[-1])>=1e-6 or eff!=g,'nonzero goal rule')
                if short: validate_runtime_direction(chunk,points,goal)
                else:
                    require(chunk['qualification']==('PASS' if chunk['clearance']>=.7 and chunk['frechet']<=2 and chunk['goal_distance']<=1 else 'FAIL'),'OF runtime description')
                expected=chunk['expected_delivered_sha256']
        require(chunk['delivered_sha256']==expected and chunk['expected_delivered_sha256']==expected,'E0 delivered u hash')
        previous = chunk
    require(row['failed_qualification_chunks']==sum(c['qualification']=='FAIL' for c in chunks),'failed qualification count')
    validate_noise(row)  # No flow_seed conditional, including O and S.


def validate_rows(rows, plans, table, xy_to_cell, cell_to_xy, fixture=False):
    by_key = {key(p):p for p in plans}
    require(len(by_key)==len(plans),'duplicate planned identity')
    seen = set(); fingerprints = {}
    for row in rows:
        k = key(row)
        require(k not in seen,'duplicate row'); seen.add(k)
        require(k in by_key,'extra row/draw outside inventory')
        validate_row(row,by_key[k],table,xy_to_cell,cell_to_xy,fixture=fixture)
        q = row['task'],row['episode']
        fp = row['initial_sha256'],row['goal_sha256']
        require(q not in fingerprints or fingerprints[q]==fp,'unpaired environment resets')
        fingerprints[q] = fp
    require(seen==set(by_key),'missing rows')


def readout(rows, plan, table, xy_to_cell, cell_to_xy, smoke=False):
    traps=d.load_trapset()
    gate=[r for r in rows if r['group']=='gate']
    totals={a:sum(r['success'] for r in gate if r['arm']==a) for a in ('Q-C','Q-O3','Q-F3')}
    q=dict(label='Q 未跑',annotations=[]) if smoke else d.q_result(totals['Q-O3'],totals['Q-F3'],totals['Q-C'])
    layers={}
    for task in (4,5,2):
        main=[r for r in rows if r['group']=='main' and r['task']==task]
        arms={}
        for arm in ('O3','F3'):
            questions=[]
            for question in plan['questions']['main']:
                if question['task']!=task: continue
                rs=sorted((r for r in main if r['arm']==arm and r['episode']==question['episode']),key=lambda r:r['draw'])
                bits=[r['success'] for r in rs]
                questions.append(dict(**question,bits=bits,success=None if smoke else d.question_success(bits)))
            n=len(questions)
            arms[arm]=dict(n=n,label='觀測不足' if smoke else d.layer_result(n,sum(q['success'] for q in questions)),questions=questions)
        oracle=[r for r in main if r['arm']=='O3']
        failures=d.failure_types(oracle,traps['tasks'][str(task)]['traps_g'],xy_to_cell)
        manhattan=d.failure_types(oracle,traps['tasks'][str(task)]['traps_g'],xy_to_cell,'manhattan')
        obedience=[]
        for row in oracle:
            for chunk in row['chunks']:
                route=traps['tasks'][str(task)]['oracle_table'][d.cell_key(chunk['c_eff'])]['route']
                if len(route)<2: continue
                xy=np.asarray(chunk['xy']); P=np.asarray(cell_to_xy(route[1]))-xy; G=np.asarray(row['goal'])-xy
                if 0 not in traps['tasks'][str(task)]['oracle_table'][d.cell_key(chunk['c_eff'])]['turn_away']: continue
                obedience.append(dict(episode=row['episode'],draw=row['draw'],step=chunk['step'],
                    **d.compliance(np.asarray(chunk['end_xy'])-xy,P,G)))
        observed=[x for x in obedience if x['preferred'] is not None]
        by={(r['episode'],r['draw']):{} for r in main}
        for r in main: by[(r['episode'],r['draw'])][r['arm']]=r['success']
        pairs=[(v['O3'],v['OF']) for v in by.values() if 'O3' in v and 'OF' in v]
        of=[r for r in main if r['arm']=='OF']
        layers[str(task)]=dict(arms=arms,**d.conclusion(arms['O3']['label'],arms['F3']['label'],q),
            failure=failures,manhattan=manhattan,norm_sensitive=failures['label']!=manhattan['label'],
            compliance=dict(details=obedience,denominator=len(observed),proportion=None if not observed else sum(x['preferred'] for x in observed)/len(observed)),
            critical_cells=table.get('critical_cells',{}).get(str(task),[]),
            runtime_qualifications=[dict(episode=r['episode'],draw=r['draw'],step=c['step'],cell=c['c_eff'],qualification=c['qualification'])
                for r in oracle for c in r['chunks'] if c['qualification_source']=='runtime'],
            OF=dict(successes=sum(r['success'] for r in of),rollouts=len(of),success_rate=None if not of else sum(r['success'] for r in of)/len(of),
                d_h=sum(int(o and not f)-int(f and not o) for o,f in pairs),n_pairs=len(pairs),
                description='短 u＋換近目標整組介入；描述性'))
    return dict(protocol=d.PROTOCOL,smoke=smoke,layers=layers,Q=q,descriptive_thresholds=True,
                statistical_test=False,scientific_release=False)


if __name__ == '__main__':
    from artifacts import load_artifact
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--e1-table',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    import socket
    require(socket.gethostname().split('.')[0]=='jasmine' and
            args.out.resolve().is_relative_to(d.OUTPUT_ROOT.resolve()),
            'harvest output must be jasmine local /archive detour-u')
    artifact = load_artifact(args.artifact)
    plan = artifact['plan']
    table = validate_receipt(artifact['receipt'],plan,args.e1_table)
    # Coordinates frozen in receipt from actual environment methods at production startup.
    mapping = artifact['receipt']['coordinate_system']
    cell_to_xy = lambda c: np.asarray(mapping['centers'][d.cell_key(c)])
    def xy_to_cell(xy):
        return (int((xy[1]+mapping['offset_y']+.5*mapping['unit'])/mapping['unit']),
                int((xy[0]+mapping['offset_x']+.5*mapping['unit'])/mapping['unit']))
    validate_rows(artifact['rows'],plan['smoke' if artifact['smoke'] else 'formal'],table,xy_to_cell,cell_to_xy)
    write_json(args.out,readout(artifact['rows'],plan,table,xy_to_cell,cell_to_xy,artifact['smoke']))
