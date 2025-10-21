import torch
import torch.nn.functional as F
import torch.nn as nn


class BaselineDistiller(torch.nn.Module):
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
        self.distillers = nn.ModuleList([network_fun(sz_embedding, multiplier) for _ in range(self.num_experts)])

    def forward(self, features, target_features):
        total_loss = 0
        for e, network in enumerate(self.distillers):
            total_loss += F.mse_loss(network(features[:, e]), target_features[:, e])
        return total_loss / self.num_experts


class AveragedDistiller(torch.nn.Module):
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
        self.distillers = nn.ModuleList([
            nn.ModuleList([network_fun(sz_embedding, multiplier) for _ in range(self.num_experts)])
            for i in range(self.num_experts)])

    def forward(self, features, target_features):
        total_loss = 0
        for distilled_expert, network_list in enumerate(self.distillers):
            for e, network in enumerate(network_list):
                total_loss += F.mse_loss(network(features[:, distilled_expert]), target_features[:, e])
        return total_loss / self.num_experts


class ConcatenatedDistiller(torch.nn.Module):
    def __init__(self,
                 num_experts,
                 sz_embedding,
                 multiplier,
                 network_type="mlp",
                 ):
        super().__init__()
        self.num_experts = num_experts
        network_fun = lambda x, d, k: nn.Linear(x, k * x)
        if network_type == "mlp":
            network_fun = lambda x, d, k: nn.Sequential(nn.Linear(x, d * x),
                                                     nn.GELU(),
                                                     nn.Linear(d * x, k * x)
                                                     )
        self.distillers = nn.ModuleList([network_fun(sz_embedding, multiplier, num_experts) for _ in range(self.num_experts)])

    def forward(self, features, target_features):
        total_loss = 0
        target_features = target_features.flatten(1)
        for e, network in enumerate(self.distillers):
            total_loss += F.mse_loss(network(features[:, e]), target_features)
        return total_loss / self.num_experts
