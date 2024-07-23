import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from models.blocks import MLP, ActNorm, LULinear, AdditiveCoupling, Gaussianize


class FlowStep(nn.Module):

    def __init__(self, mask, dim, hidden_dim, num_blocks, c_dim, dropout_probability):
        super().__init__()

        if c_dim is not None:
            self.condition_embedding = MLP(c_dim, hidden_dim, dim * 2)
        else:
            self.condition_embedding = None

        self.actnorm = ActNorm(dim)
        self.linear = LULinear(dim)
        self.coupling = AdditiveCoupling(
            mask, dim, hidden_dim, num_blocks, c_dim, dropout_probability)


    def forward(self, x, condition=None):
        if self.condition_embedding is not None and condition is not None:
            c_hidden = self.condition_embedding(condition)
            scale_drift, bias_drift = c_hidden[:, 0::2], c_hidden[:, 1::2]
        else:
            scale_drift = bias_drift = None 

        x, logdet1 = self.actnorm(x, scale_drift, bias_drift)
        x, logdet2 = self.linear(x)
        x, logdet3 = self.coupling(x, condition)

        logdet = logdet1 + logdet2 + logdet3
        return x, logdet


    def inverse(self, z, condition=None):
        if self.condition_embedding is not None and condition is not None:
            c_hidden = self.condition_embedding(condition)
            scale_drift, bias_drift = c_hidden[:, 0::2], c_hidden[:, 1::2]
        else:
            scale_drift = bias_drift = None

        z, logdet3 = self.coupling.inverse(z, condition)
        z, logdet2 = self.linear.inverse(z)
        z, logdet1 = self.actnorm.inverse(z, scale_drift, bias_drift)

        logdet = logdet1 + logdet2 + logdet3
        return z, logdet



class ConditionFlow(nn.Module):

    def __init__(self, dim, hidden_dim, c_dim=None, num_blocks_per_layer=2, num_layers=4, dropout_probability=0.0):
        super().__init__()
        self.dim = dim
        flows = []
        mask = torch.ones(dim)
        mask[::2] = -1
        for _ in range(num_layers):
            flows.append(FlowStep(
                mask, dim, hidden_dim, num_blocks_per_layer, c_dim, dropout_probability))
            mask *= -1
        self.flows = nn.ModuleList(flows)
        self.gaussianize = Gaussianize(dim)
        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))


    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)


    def forward(self, x, condition=None, gaussianize=True):
        # x: [b, n]
        sum_logdets = 0
        for flow in self.flows:
            x, logdet = flow(x, condition)
            sum_logdets += logdet

        if gaussianize:
            x, logdet = self.gaussianize(torch.zeros_like(x), x)
            sum_logdets = sum_logdets + logdet

        return x, sum_logdets


    def inverse(self, z, condition=None, gaussianize=True):
        sum_logdets = 0
        if gaussianize:
            z, logdet = self.gaussianize.inverse(torch.zeros_like(z), z)
            sum_logdets = sum_logdets + logdet

        for flow in reversed(self.flows):
            z, logdet = flow.inverse(z, condition)
            sum_logdets += logdet

        return z, sum_logdets


    def sampling(self, n_samples, condition=None, z_std=1.):
        z = z_std * self.base_dist.sample((n_samples, self.dim)).squeeze(-1)
        x, sum_logdets = self.inverse(z, condition)
        return x, sum_logdets


    def log_prob(self, x, condition=None):
        z, logdet = self.forward(x, condition)
        log_prob = self.base_dist.log_prob(z).sum(-1) + logdet
        log_prob = log_prob / z.shape[1]
        return log_prob
