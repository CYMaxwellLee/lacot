"""u 編碼一段想像軌跡。C9 external-route acceptance oracle."""
import torch

def assess_routes(pred, route):
    if (pred.shape != route.shape or pred.ndim != 3 or pred.shape[1:] != (1, 1)
        or len(pred) == 0 or len(pred) % 2 or pred.device.type != 'cpu'
        or route.device.type != 'cpu' or pred.dtype != torch.float64
        or route.dtype != torch.float64 or not torch.isfinite(pred).all()
        or not torch.isfinite(route).all() or not ((route == 1) | (route == -1)).all()
        or (route == 1).sum() != (route == -1).sum()):
        raise ValueError('C9: finite CPU float64 [even N,1,1], balanced +/-1 routes required')
    x, y = pred.flatten(), route.flatten()
    cx, cy = x-x.mean(), y-y.mean()
    denominator = cx.norm() * cy.norm()
    cc = float(cx.dot(cy) / denominator) if denominator > 1e-12 else 0.0
    error, average = float((x-y).abs().mean()), float(x.abs().mean())
    plus = float((x[y > 0]-1).abs().mean())
    minus = float((x[y < 0]+1).abs().mean())
    gap = float((x+y).abs().mean()) - error
    passed = error < .1 and error < .25*average and cc >= .95 and plus < .1 and minus < .1 and gap > .75
    return passed, dict(route_mae=error, mean_mae=average, corr=cc,
                       plus_mae=plus, minus_mae=minus, swap_gap=gap)
