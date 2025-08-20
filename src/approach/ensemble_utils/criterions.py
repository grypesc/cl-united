import torch
import torch.nn.functional as F


class EnsembledCE(torch.nn.Module):
    def __init__(self,
                 num_experts,
                 nb_classes,
                 sz_embedding,
                 device,
                 smoothing=0.0,
                 ):
        super().__init__()
        self.num_experts = num_experts
        self.heads = torch.nn.ModuleList([torch.nn.Linear(sz_embedding, nb_classes, device=device) for _ in range(self.num_experts)])
        self.smoothing = smoothing

    def forward(self, features, T):
        total_loss = 0
        for e in range(self.num_experts):
            logits = self.heads[e](features[:, e])
            total_loss += F.cross_entropy(logits, T, label_smoothing=self.smoothing)
        return total_loss / self.num_experts


