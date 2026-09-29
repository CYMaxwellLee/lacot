"""u 編碼想像軌跡。v2 proposal interfaces, awaiting decision."""
from dataclasses import dataclass
from typing import Callable
import torch
Tensor = torch.Tensor
@dataclass
class Terms:
    quality: Tensor
    consistency: Tensor
    states: list[Tensor]
@dataclass
class Adapter:
    # Closures preserve clean semantics: density input may differ from head input.
    density: Callable
    anchor: Callable
    sample: Callable
    exposure: Callable
    features: Callable

def refinement_terms(refine, teacher, quality, cond, clean, sampled, *, rounds, noise):
    raise NotImplementedError('M1')
def compose(adapter, refine, teacher, quality_factory, cond, clean, *, rounds, noise, lam_cons=.1):
    raise NotImplementedError('M2')
