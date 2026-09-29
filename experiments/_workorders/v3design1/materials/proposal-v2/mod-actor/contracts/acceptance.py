"""u 編碼想像軌跡。Observable contract; scientific measurement is owned by M3."""
def assess_run(stats):
    q=stats['quality']; m=stats['mode_error']
    checks=dict(off_manifold=q[3]<.15*q[0] and q[3]<.12,
                scaling=q[1]<q[0] and q[3]<=1.05*q[1],
                modes=m[3]<.05,
                exposure=stats['exposure']['exposure/valid'] and stats['exposure']['exposure/r3_mse']<.08)
    return all(checks.values()),checks

def suite(assess):
    good=dict(quality=[1.,.1,.06,.04],mode_error=[0.,0.,0.,0.],
              exposure={'exposure/valid':True,'exposure/r3_mse':.01})
    assert assess(good)[0]
    import copy
    for key,val in [('quality',[1.,1.,1.,1.]),('quality',[1.,.01,.1,.2]),
                    ('mode_error',[0.,0.,0.,.5]),('exposure',{'exposure/valid':True,'exposure/r3_mse':.3})]:
        bad=copy.deepcopy(good);bad[key]=val;assert not assess(bad)[0]
    print('PASS acceptance schema: identity, bad scaling, mode collapse, exposure failure rejected')
