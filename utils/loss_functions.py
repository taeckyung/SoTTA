import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510
# but there is a bug in the original code: it sums up the entropy over a batch. so I take mean instead of sum
class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(HLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):
        softmax = F.softmax(x / self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax + 1e-6)
        b = entropy.mean()

        return b


@torch.jit.script
def calc_energy(x: torch.Tensor, temp_factor: float = 1.0) -> torch.Tensor:
    return temp_factor * torch.logsumexp(x / temp_factor, dim=1)


class EnergyLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(EnergyLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):
        e = calc_energy(x, self.temp_factor)
        # energy = 1.0 / torch.linalg.vector_norm(6.0 - energy, 2)
        e = 1.0 / e.mean()
        return e


class JSDLoss(nn.Module):
    def __init__(self):
        super(JSDLoss, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    softmax = x.softmax(1)
    return -(softmax * torch.log(softmax + 1e-6)).sum(1)


@torch.jit.script
def softmax_entropy_rotta(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)
