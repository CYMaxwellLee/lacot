import json, ast, hashlib
import numpy as np
from collections import deque
B=json.load(open('/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/builder.json'))
maze=np.array(B['maze']); free={(i,j) for i in range(maze.shape[0]) for j in range(maze.shape[1]) if maze[i,j]==0}
def bfs(s):
    d={s:0}; q=deque([s])
    while q:
        c=q.popleft()
        for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
            n=(c[0]+dx,c[1]+dy)
            if n in free and n not in d: d[n]=d[c]+1; q.append(n)
    return d
allok=True
for t,g in B['geometries'].items():
    s,gg=tuple(g['s']),tuple(g['g']); ds,dg=bfs(s),bfs(gg); D=ds[gg]
    nodes={c for c in ds if c in dg and ds[c]+dg[c]==D}
    mine={c:sorted(n for n in [(c[0]+dx,c[1]+dy) for dx,dy in((1,0),(-1,0),(0,1),(0,-1))] if n in nodes and ds[n]==ds[c]+1) for c in nodes}
    theirs={tuple(e['cell']):sorted(map(tuple,e['next'])) for e in g['dag']}
    depth_ok=all(e['depth']==ds[tuple(e['cell'])] for e in g['dag'])
    ok= mine==theirs and depth_ok
    allok&=ok; print('task',t,'DAG nodes',len(nodes),'match',ok)
# maze literal hash vs gate
import hashlib
arr=np.asarray(B['maze'],dtype='<i8'); print('maze sha ok', hashlib.sha256(arr.tobytes()).hexdigest()==B['maze_sha256'])
print('ALL DAG OK',allok)
# cross-check that every free cell's cell-center maps back via env conventions (xy_to_ij o ij_to_xy == id) for routes
