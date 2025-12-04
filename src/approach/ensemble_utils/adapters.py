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
                is_ok = False
                additional_shrink_factor = 1.0
                while not is_ok:
                    distribution = MultivariateNormal(means[c, expert_num], covs[c, expert_num])
                    samples = distribution.sample((10000,))
                    if not torch.isnan(samples).any():
                        is_ok = True
                    else:
                        print("Applying additional shrinking factor.")
                        additional_shrink_factor *= 10
                        covs[c, expert_num] = shrink_cov(covs[c, expert_num], shrink * additional_shrink_factor)

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
        self.N = 10000

    def forward(self, features, target_features):
        total_loss = 0
        features = features.flatten(1)
        for e, network in enumerate(self.adapters):
            total_loss += F.mse_loss(network(features), target_features[:, e])
        return total_loss / self.num_experts

    @torch.no_grad()
    def adapt(self, means, covs, shrink=0.):
        self.adapters.eval()
        new_means = torch.zeros_like(means)
        new_covs = torch.zeros_like(covs)

        for expert_adapted, adapter in enumerate(self.adapters):
            for c in range(means.shape[0]):
                samples = torch.zeros((10000, self.num_experts,  means.shape[2]), device=means.device)
                for e in range(self.num_experts):
                    torch.manual_seed(0)  # Make sure that sampled points are polarized
                    distribution = MultivariateNormal(means[c, e], covs[c, e])
                    samples[:, e, :] = distribution.sample((self.N,))
                samples = samples.flatten(1)
                if torch.isnan(samples).any():
                    raise RuntimeError(f"Nan in features sampled for class {c}")
                adapted_samples = adapter(samples)
                new_means[c, expert_adapted] = adapted_samples.mean(0)
                # print(f"Rank pre-adapt {c}: {torch.linalg.matrix_rank(self.covs[c])}")
                new_covs[c, expert_adapted] = torch.cov(adapted_samples.T)
                new_covs[c, expert_adapted] = shrink_cov(new_covs[c, expert_adapted], shrink)
        return new_means, new_covs


class AveragedAdapter(torch.nn.Module):
    def __init__(self,
                 num_experts,
                 sz_embedding,
                 multiplier,
                 network_type="mlp",
                 ):
        super().__init__()
        self.num_experts = num_experts
        self.sz_embedding = sz_embedding
        network_fun = lambda x, d: nn.Linear(x, x)
        if network_type == "mlp":
            network_fun = lambda x, d: nn.Sequential(nn.Linear(x, d * x),
                                                     nn.GELU(),
                                                     nn.Linear(d * x, x)
                                                     )
        self.adapters = nn.ModuleList([
            nn.ModuleList([network_fun(sz_embedding, multiplier) for _ in range(self.num_experts)])
            for i in range(self.num_experts)])
        self.N = 10000

    def forward(self, features, target_features):
        total_loss = 0
        for expert_adapted, networks_list in enumerate(self.adapters):
            for e, network in enumerate(networks_list):
                total_loss += F.mse_loss(network(features[:, expert_adapted]), target_features[:, expert_adapted])
        return total_loss / self.num_experts

    @torch.no_grad()
    def adapt(self, means, covs, shrink=0.):
        self.adapters.eval()
        new_means = torch.zeros(means.shape[0], self.num_experts, self.num_experts, self.sz_embedding, device=means.device)
        new_covs = torch.zeros(means.shape[0], self.num_experts, self.num_experts, self.sz_embedding, self.sz_embedding, device=means.device)

        for expert_adapted, networks_list in enumerate(self.adapters):
            for c in range(means.shape[0]):
                for e in range(self.num_experts):
                    distribution = MultivariateNormal(means[c, e], covs[c, e])
                    samples = distribution.sample((self.N,))
                    if torch.isnan(samples).any():
                        raise RuntimeError(f"Nan in features sampled for class {c}")
                    adapted_samples = networks_list[e](samples)
                    new_means[c, expert_adapted, e] = adapted_samples.mean(0)

                    new_covs[c, expert_adapted, e] = torch.cov(adapted_samples.T)
                    new_covs[c, expert_adapted, e] = shrink_cov(new_covs[c, expert_adapted, e], shrink)

        # Average distribution over all experts
        new_means = new_means.mean(2)
        new_covs = new_covs.mean(2)
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