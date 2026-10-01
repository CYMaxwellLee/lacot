"""S2 independent recompute: question sets + geometry, no import of builder.py."""
import json, hashlib, itertools, sys
from collections import Counter, deque
import numpy as np
BG='/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/gate0b-result.json'
B='/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/builder.json'
g=json.load(open(BG)); b=json.load(open(B))
rows=g['rows']
print('rows',len(rows), 'keys', sorted(rows[0].keys()))
# question sets from raw rule text in card
main=[(r['task'],r['episode']) for r in rows if r['a_bits']==0 and r['b_bits']==0 and r['n_shortest']==2]
ctrl=[(r['task'],r['episode']) for r in rows if r['a_bits']==0 and r['b_bits']==0 and r['n_shortest']==1]
gate=[(r['task'],r['episode']) for r in rows if r['a_bits']==64 and r['b_bits']==64]
print('main',Counter(t for t,_ in main), 'ctrl',Counter(t for t,_ in ctrl),'gate',gate)
bm=[(q['task'],q['episode']) for q in b['questions']['main']]
bc=[(q['task'],q['episode']) for q in b['questions']['control']]
bg=[(q['task'],q['episode']) for q in b['questions']['gate']]
print('builder==mine main/ctrl/gate:', sorted(main)==bm, sorted(ctrl)==bc, sorted(gate)==bg)
# Are there other rows with n_shortest==2 and partial failures that might matter? list bits dist
print('bits dist by task:', {t: sorted(Counter((r['a_bits'],r['b_bits']) for r in rows if r['task']==t).items()) for t in range(1,6)})
# D / n_shortest by task from gate0b
print({t:(set((r['shortest_len'],r['n_shortest']) for r in rows if r['task']==t)) for t in range(1,6)})
# maze from ogbench source via ast, pick by sha
import ast
src=open('/home/cymaxwelllee/Projects/ogbench/ogbench/locomaze/maze.py').read()
tree=ast.parse(src)
cands=[]
for n in ast.walk(tree):
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='maze_map' for t in n.targets):
        try: v=ast.literal_eval(n.value)
        except Exception: continue
        arr=np.asarray(v,dtype='<i8')
        cands.append((hashlib.sha256(arr.tobytes()).hexdigest(), arr))
print('maze candidates',[(h[:10],a.shape) for h,a in cands], 'target', g['maze_sha256'][:10])
maze=[a for h,a in cands if h==g['maze_sha256']][0]
print(maze)
free={(i,j) for i in range(maze.shape[0]) for j in range(maze.shape[1]) if maze[i,j]==0}
def bfs(s):
    d={s:0}; q=deque([s])
    while q:
        c=q.popleft()
        for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
            n=(c[0]+dx,c[1]+dy)
            if n in free and n not in d: d[n]=d[c]+1; q.append(n)
    return d
# also independent: DP count of shortest paths (not enumeration), then enumeration via itertools-free DFS
for task in range(1,6):
    r=[x for x in rows if x['task']==task][0]
    s=tuple(r['s']); gg=tuple(r['g'])
    ds=bfs(s); dg=bfs(gg); D=ds[gg]
    cnt={s:1}
    order=sorted(ds,key=lambda c:ds[c])
    for c in order:
        if c==s: continue
        cnt[c]=sum(cnt.get(p,0) for p in [(c[0]+dx,c[1]+dy) for dx,dy in((1,0),(-1,0),(0,1),(0,-1))] if p in ds and ds[p]==ds[c]-1)
    nroutes=cnt[gg]
    # enumerate
    paths=[]
    def dfs(c,path):
        if c==gg: paths.append(path[:]); return
        for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
            n=(c[0]+dx,c[1]+dy)
            if n in free and ds[n]==ds[c]+1 and ds[n]+dg[n]<=D:
                if ds[n]+dg[n]==D:
                    path.append(n); dfs(n,path); path.pop()
    dfs(s,[s])
    paths=sorted(paths)
    geo=b['geometries'][str(task)]
    ok=[ (len(paths)==nroutes==geo['n_routes']), D==geo['D'], [list(map(list,p)) for p in paths]==[geo['routes'][k] for k in sorted(geo['routes'])] ]
    info=dict(task=task,D=D,nroutes=nroutes,builder_nroutes=geo['n_routes'])
    if len(paths)==2:
        a,bb=paths
        diff=[i for i in range(D+1) if a[i]!=bb[i]]
        lo,hi=min(diff),max(diff)
        contiguous = diff==list(range(lo,hi+1))
        mid=(lo+hi)/2; d=int(np.floor(mid))
        info.update(lo=lo,hi=hi,contig=contiguous,mid=mid,d_star=d,wA=a[d],wB=bb[d],builder_w=geo['w'],builder_depth=geo['w_depth'],
                    excl_ok=([list(c) for c in a[lo:hi+1]]==geo['exclusive']['A'] and [list(c) for c in bb[lo:hi+1]]==geo['exclusive']['B']),
                    branch=a[lo-1],merge=a[hi+1],bb=geo['branch'],bm=geo['merge'])
        # sanity: exclusive sets disjoint, and equal length
        assert not set(a[lo:hi+1])&set(bb[lo:hi+1])
        # also check: are there cells in common depth where a[i]==b[i] inside (lo,hi)? (would be non-contiguous)
    else:
        d=D//2
        info.update(d_star=d,w=paths[0][d],builder_w=geo['w'],builder_depth=geo['w_depth'])
    print(ok, info)
# R donor
print('R_donor',b['R_donor'])
for t,o,grp in [(4,5,'main'),(5,4,'main'),(1,3,'gate'),(3,1,'gate')]:
    q={'main':main,'gate':gate}[grp]
    ep=min(e for tt,e in q if tt==o)
    print(t,'<-',o,'min ep',ep, b['R_donor'][str(t)])
