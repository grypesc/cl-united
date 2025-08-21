import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.distributions.multivariate_normal import MultivariateNormal


class BaselineAdapter(torch.nn.Module):
    def __init__(self,
                 num_experts,
                 sz_embedding,
                 multiplier,
                 network_type="mlp",
                 ):
        super().__init__()
        self.num_experts = num_experts
        network_fun = lambda x, d: nn.Linear(x, x)
        if network_type == "mlp":
            network_fun = lambda x, d: nn.Sequential(nn.Linear(x, d * x),
                                                     nn.GELU(),
                                                     nn.Linear(d * x, x)
                                                     )
        self.adapters = nn.ModuleList([network_fun(sz_embedding, multiplier) for _ in range(self.num_experts)])

    def forward(self, features, target_features):
        total_loss = 0
        for e, network in enumerate(self.adapters):
            total_loss += F.mse_loss(network(features[:, e]), target_features[:, e])
        return total_loss / self.num_experts

    @torch.no_grad()
    def adapt(self, means, covs, shrink=0.):
        self.adapters.eval()
        new_means = torch.zeros_like(means)
        new_covs = torch.zeros_like(covs)

        for expert_num, adapter in enumerate(self.adapters):
            for c in range(means.shape[0]):
                distribution = MultivariateNormal(means[c, expert_num], covs[c, expert_num])
                samples = distribution.sample((10000,))
                if torch.isnan(samples).any():
                    raise RuntimeError(f"Nan in features sampled for class {c}")
                adapted_samples = adapter(samples)
                new_means[c, expert_num] = adapted_samples.mean(0)
                # print(f"Rank pre-adapt {c}: {torch.linalg.matrix_rank(self.covs[c])}")
                new_covs[c, expert_num] = torch.cov(adapted_samples.T)
                new_covs[c, expert_num] = shrink_cov(new_covs[c, expert_num], shrink)
        return new_means, new_covs


class ConcatenatedAdapter(torch.nn.Module):
    def __init__(self,
                 num_experts,
                 sz_embedding,
                 multiplier,
                 network_type="mlp",
                 ):
        super().__init__()
        self.num_experts = num_experts
        network_fun = lambda x, d, k: nn.Linear(k * x, x)
        if network_type == "mlp":
            network_fun = lambda x, d, k: nn.Sequential(nn.Linear(k * x, d * x),
                                                     nn.GELU(),
                                                     nn.Linear(d * x, x)
                                                     )
        self.adapters = nn.ModuleList([network_fun(sz_embedding, multiplier, num_experts) for _ in range(self.num_experts)])

    def forward(self, features, target_features):
        total_loss = 0
        for e, network in enumerate(self.adapters):
            features = features.flatten(1)
            total_loss += F.mse_loss(network(features[:, e]), target_features[:, e])
        return total_loss / self.num_experts

    @torch.no_grad()
    def adapt(self, means, covs, shrink=0.):
        self.adapters.eval()
        new_means = torch.zeros_like(means)
        new_covs = torch.zeros_like(covs)

        for expert_num, adapter in enumerate(self.adapters):
            for c in range(means.shape[0]):
                distribution = MultivariateNormal(means[c, expert_num], covs[c, expert_num])
                samples = distribution.sample((10000,))
                if torch.isnan(samples).any():
                    raise RuntimeError(f"Nan in features sampled for class {c}")
                adapted_samples = adapter(samples)
                new_means[c, expert_num] = adapted_samples.mean(0)
                # print(f"Rank pre-adapt {c}: {torch.linalg.matrix_rank(self.covs[c])}")
                new_covs[c, expert_num] = torch.cov(adapted_samples.T)
                new_covs[c, expert_num] = shrink_cov(new_covs[c, expert_num], shrink)
        return new_means, new_covs


@torch.no_grad()
def shrink_cov(cov, alpha1=1., alpha2=0.):
    if alpha2 == -1.:
        return cov + alpha1 * torch.eye(cov.shape[0], device=cov.device)  # ordinary epsilon
    diag_mean = torch.mean(torch.diagonal(cov))
    iden = torch.eye(cov.shape[0], device=cov.device)
    mask = iden == 0.0
    off_diag_mean = torch.mean(cov[mask])
    return cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))


@torch.no_grad()
def norm_cov(cov):
    diag = torch.diagonal(cov, dim1=1, dim2=2)
    std = torch.sqrt(diag)
    cov = cov / (std.unsqueeze(2) @ std.unsqueeze(1))
    return cov