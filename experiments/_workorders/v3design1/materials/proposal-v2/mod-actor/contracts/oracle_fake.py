"""u 編碼想像軌跡。Two-corridor analytic geometry, not a production action teacher."""
import torch
from torch import nn
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('matrix',torch.eye(2))
    def forward(self,u):
        return u @ self.matrix
class Quality:
    def __init__(self,sampled,cond):
        self.route=sampled.detach()[...,0].sign()
        self.goal=cond.detach()[:,:1]
        self.decoder=Decoder().to(sampled)
    def __call__(self,u):
        p=self.decoder(u)
        # Valid target from this plan's branch and this condition, never logged action.
        return (p[...,0]-self.route).square().mean(1)+(p[...,1]-self.goal).square().mean(1)
    def action(self,u):
        # Toy inverse dynamics exactly defined: first coordinate is steering action.
        return self.decoder(u)[...,:1]
    def feedback(self,u):
        return self.decoder(u)
@torch.no_grad()
def gauge(head, sampled_states, clean, clean_actions, oracle):
    anchor=(head(clean)-clean_actions).square().mean().item()
    result={'exposure/anchor_mse':anchor,'exposure/valid':True}
    for r,u in enumerate(sampled_states):
        err=(head(u)-oracle.action(u)).square().mean().item()
        result[f'exposure/r{r}_mse']=err
        result[f'exposure/r{r}_gap']=err-anchor
    return result
