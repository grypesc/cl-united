import random

import torch
import torch.nn.functional as F


class CE(torch.nn.Module):
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


class SCE(torch.nn.Module):
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
        self.nb_classes = nb_classes
        # smoothing_matrices is a symmetrical matrix of probabilities for each class. Its columns sum to 1
        self.smooth_matrices = torch.zeros((num_experts, nb_classes, nb_classes), device=device)
        for e in range(num_experts):
            indices_to_smooth = torch.randint(0, nb_classes, (10, 2), device=device)
            indices_to_smooth = indices_to_smooth[indices_to_smooth[:, 0] != indices_to_smooth[:, 1]]
            symmetrical_indices = indices_to_smooth.flip(1)
            indices_to_smooth = torch.cat((indices_to_smooth, symmetrical_indices), dim=0)
            self.smooth_matrices[e, indices_to_smooth[:, 0], indices_to_smooth[:, 1]] = smoothing
            col_sum = self.smooth_matrices[e].sum(1)
            torch.diagonal(self.smooth_matrices[e])[:] = 1 - col_sum

    def forward(self, features, T):
        total_loss = 0
        bsz = features.shape[0]

        for e in range(self.num_experts):
            sm = self.smooth_matrices[e].expand(bsz, self.nb_classes, self.nb_classes)
            T = T.unsqueeze(1).unsqueeze(1).repeat(1, 1, self.nb_classes)
            target = torch.gather(sm, 1, T).squeeze(1)
            logits = self.heads[e](features[:, e])
            total_loss += F.cross_entropy(logits, target)
        return total_loss / self.num_experts
