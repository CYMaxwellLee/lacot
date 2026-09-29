"""u 編碼想像軌跡。Execute the ACTUAL AST branch without importing the training script."""
import ast
from pathlib import Path
import torch

def load_inline(root):
    path=Path(root)/'experiments/scratch_lacot_rollout.py'
    tree=ast.parse(path.read_text())
    loop=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_stage2_loop')
    branch=next(n for n in ast.walk(loop) if isinstance(n,ast.If)
                and isinstance(n.test,ast.Name) and n.test.id=='LEARNED_REFINE')
    total_node=next(n for n in ast.walk(loop) if isinstance(n,ast.Assign)
                    and any(isinstance(x,ast.Name) and x.id=='total' for x in n.targets))
    code=compile(ast.Module(body=[branch,total_node],type_ignores=[]),str(path),'exec')
    from toy import ReplayFlow
    def call(refine,teacher,head,c,clean,labels,sampled):
        ns=dict(torch=torch,LEARNED_REFINE=1,CONS='ema',refine=refine,refine_ema=teacher,
            flow=ReplayFlow(sampled),B=len(c),cond=c,anc=None,flow_cond=lambda c,a:c,
            ahead=lambda c,u:head(u),mse=lambda p,a:(p-a).square().mean(),act=labels,
            l_nf=clean.square().mean(),l_anchor=(head(clean)-labels).square().mean(),
            l_bc=clean.new_zeros(()),BC_INDEP=True)
        exec(code,ns)
        return ns
    call.source=path; call.lines=(branch.lineno,branch.end_lineno);call.total_line=total_node.lineno
    return call
