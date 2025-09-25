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
        self.smooth_matrices = torch.diagonal((num_experts, nb_classes, nb_classes), device=device)
        for e in range(num_experts):
            for class_ in range(nb_classes):
                indices = [0, 3, 5]
                for index in indices:
                    if self.smooth_matrices[num_experts, ]


    def forward(self, features, T):
        total_loss = 0
        for e in range(self.num_experts):
            target = torch.nn.functional.one_hot(T, num_classes=self.nb_classes)
            logits = self.heads[e](features[:, e])
            total_loss += F.cross_entropy(logits, target)
        return total_loss / self.num_experts