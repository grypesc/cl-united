import copy
import random
import torch
import torch.nn.functional as F
import numpy as np

from argparse import ArgumentParser
from itertools import compress
from torch import nn
from torch.utils.data import Dataset
from torchmetrics import Accuracy

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .models.resnet18 import resnet18
from .incremental_learning import Inc_Learning_Appr
from .criterions.proxy_nca import ProxyNCA
from .criterions.proxy_yolo import ProxyYolo
from .criterions.ce import CE

from torch.distributions.multivariate_normal import MultivariateNormal


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=1,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, N=10000, alpha=0.01, distillation="projected", K=3, use_224=False, S=64, dump=False, rotation=False, distiller="linear", adapter="linear", criterion="proxy-nca", lamb=10, tau=2, smoothing=0., sval_fraction=0.95,
                 adaptation_strategy="mean-only", normalize=False, shrink1=1., shrink2=1., multiplier=8, mahalanobis=False, nnet="resnet18"):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)

        self.N = N
        self.K = K
        self.S = S
        self.dump = dump
        self.lamb = lamb
        self.alpha = alpha
        self.tau = tau
        self.multiplier = multiplier
        self.shrink1, self.shrink2 = shrink1, shrink2
        self.default_shrink = 0.01
        self.smoothing = smoothing
        self.adaptation_strategy = adaptation_strategy
        self.old_model = None
        self.model = resnet18(num_features=S, is_224=use_224)
        self.model.to(device, non_blocking=True)
        self.train_data_loaders, self.val_data_loaders = [], []
        self.means = torch.empty((0, self.S), device=self.device)
        self.covs = torch.empty((0, self.S, self.S), device=self.device)
        self.covs_inverted = None
        self.is_mahalanobis = mahalanobis
        self.is_normalization = normalize
        self.is_rotation = rotation
        self.task_offset = [0]
        self.classes_in_tasks = []
        self.criterion_type = criterion
        self.criterion = {"proxy-yolo": ProxyYolo,
                          "proxy-nca": ProxyNCA,
                          "ce": CE}[criterion]
        self.heads = torch.nn.ModuleList()
        self.sval_fraction = sval_fraction
        self.svals_explained_by = []
        self.distiller_type = distiller
        self.distillation = distillation
        self.adapter_type = adapter

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--N',
                            help='Number of samples to adapt cov',
                            type=int,
                            default=10000)
        parser.add_argument('--K',
                            help='number of learners sampled for task',
                            type=int,
                            default=3)
        parser.add_argument('--S',
                            help='latent space size',
                            type=int,
                            default=64)
        parser.add_argument('--alpha',
                            help='Weight of pizdolos loss',
                            type=float,
                            default=1e-4)
        parser.add_argument('--lamb',
                            help='Weight of kd loss',
                            type=float,
                            default=10)
        parser.add_argument('--multiplier',
                            help='mlp muliplier',
                            type=int,
                            default=8)
        parser.add_argument('--tau',
                            help='temperature',
                            type=float,
                            default=2)
        parser.add_argument('--shrink1',
                            help='Weight of kd loss',
                            type=float,
                            default=1)
        parser.add_argument('--shrink2',
                            help='Weight of kd loss',
                            type=float,
                            default=1)
        parser.add_argument('--sval-fraction',
                            help='Fraction of eigenvalues sum that is explained',
                            type=float,
                            default=0.95)
        parser.add_argument('--adaptation-strategy',
                            help='Activation functions in resnet',
                            type=str,
                            choices=["no-adapt", "mean-only", "diag", "full"],
                            default="mean-only")
        parser.add_argument('--distiller',
                            help='Distiller',
                            type=str,
                            choices=["linear", "mlp"],
                            default="mlp")
        parser.add_argument('--adapter',
                            help='Adapter',
                            type=str,
                            choices=["linear", "mlp"],
                            default="linear")
        parser.add_argument('--criterion',
                            help='Loss function',
                            type=str,
                            choices=["ce", "proxy-nca", "proxy-yolo"],
                            default="proxy-yolo")
        parser.add_argument('--distillation',
                            help='Loss function',
                            type=str,
                            choices=["projected", "logit", "feature"],
                            default="projected")
        parser.add_argument('--smoothing',
                            help='label smoothing',
                            type=float,
                            default=0.0)
        parser.add_argument('--use-224',
                            help='xxx',
                            action='store_true',
                            default=False)
        parser.add_argument('--mahalanobis',
                            help='xxx',
                            action='store_true',
                            default=False)
        parser.add_argument('--normalize',
                            help='xxx',
                            action='store_true',
                            default=False)
        parser.add_argument('--dump',
                            help='xxx',
                            action='store_true',
                            default=False)
        parser.add_argument('--rotation',
                            help='xxx',
                            action='store_true',
                            default=False)
        parser.add_argument('--nnet',
                            type=str,
                            choices=["resnet8", "resnet14", "resnet20", "resnet32", "resnet18"],
                            default="resnet18")
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        num_classes_in_t = len(np.unique(trn_loader.dataset.labels))
        self.classes_in_tasks.append(num_classes_in_t)
        self.train_data_loaders.extend([trn_loader])
        self.val_data_loaders.extend([val_loader])
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.task_offset.append(num_classes_in_t + self.task_offset[-1])
        print("### Training backbone ###")
        # state_dict = torch.load(f"../ckpts/model_{t}.pth")
        # self.model.load_state_dict(state_dict, strict=True)
        self.train_backbone(t, trn_loader, val_loader, num_classes_in_t)
        if t > 0 and self.adaptation_strategy != "no-adapt":
            print("### Adapting prototypes ###")
            self.adapt_distributions(t, trn_loader, val_loader)
        if self.dump:
            torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model_{t}.pth")
        print("### Creating new prototypes ###\n")
        self.create_distributions(t, trn_loader, val_loader, num_classes_in_t)

        # Calculate inverted covariances for evaluation with mahalanobis
        covs = self.covs.clone()
        print(f"Cov matrix det: {torch.linalg.det(covs)}")
        for i in range(covs.shape[0]):
            print(f"Rank for class {i}: {torch.linalg.matrix_rank(covs[i], tol=0.001)}")
            covs[i] = self.shrink_cov(covs[i], self.shrink1, self.shrink2)
        if self.is_normalization:
            covs = self.norm_cov(covs)
        self.covs_inverted = torch.inverse(covs)

        self.check_singular_values(t, val_loader)
        self.print_singular_values()
        self.print_covs(trn_loader, val_loader)

    def train_backbone(self, t, trn_loader, val_loader, num_classes_in_t):
        print(f'The model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,} shared parameters\n')
        distiller = nn.Linear(self.S, self.S)
        if self.distiller_type == "mlp":
            distiller = nn.Sequential(nn.Linear(self.S, self.multiplier * self.S),
                                      nn.GELU(),
                                      nn.Linear(self.multiplier * self.S, self.S)
                                      )

        distiller.to(self.device, non_blocking=True)
        criterion = self.criterion(num_classes_in_t, self.S, self.device, smoothing=self.smoothing)
        if t == 0 and self.is_rotation:
            criterion = self.criterion(4*num_classes_in_t, self.S, self.device, smoothing=self.smoothing)
        self.heads.eval()
        old_heads = copy.deepcopy(self.heads)
        parameters = list(self.model.parameters()) + list(criterion.parameters()) + list(distiller.parameters()) + list(self.heads.parameters())
        optimizer, lr_scheduler = self.get_optimizer(parameters, t, self.wd)

        for epoch in range(self.nepochs):
            train_loss, train_kd_loss, valid_loss, valid_kd_loss = [], [], [], []
            train_pizdolos = []
            train_hits, val_hits = 0, 0
            self.model.train()
            criterion.train()
            distiller.train()
            for images, targets in trn_loader:
                if t == 0 and self.is_rotation:
                    images, targets = compute_rotations(images, targets, num_classes_in_t)
                targets -= self.task_offset[t]
                bsz = images.shape[0]
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                features = self.model(images)
                if epoch < int(self.nepochs * 0.01) and t > 0:
                    features = features.detach()
                loss, logits = criterion(features, targets)

                if self.distillation == "logit":
                    total_loss, kd_loss = self.distill_logits(t, loss, features, images, old_heads)
                elif self.distillation == "projected":
                    total_loss, kd_loss = self.distill_projected(t, loss, features, distiller, images)
                else:  # feature
                    total_loss, kd_loss = self.distill_features(t, loss, features, images)

                pizdolos = loss_pizdolos(features)
                total_loss += self.alpha * pizdolos
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                optimizer.step()
                if logits is not None:
                    train_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
                train_pizdolos.append(float(pizdolos))
                train_kd_loss.append(float(bsz * kd_loss))
            lr_scheduler.step()

            self.model.eval()
            criterion.eval()
            distiller.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    if t == 0 and self.is_rotation:
                        images, targets = compute_rotations(images, targets, num_classes_in_t)
                    targets -= self.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                    features = self.model(images)
                    loss, logits = criterion(features, targets)
                    if self.distillation == "logit":
                        _, kd_loss = self.distill_logits(t, loss, features, images, old_heads)
                    elif self.distillation == "projected":
                        _, kd_loss = self.distill_projected(t, loss, features, distiller, images)
                    else:  # feature
                        _, kd_loss = self.distill_features(t, loss, features, images)

                    if logits is not None:
                        val_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                    valid_loss.append(float(bsz * loss))
                    valid_kd_loss.append(float(bsz * kd_loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            train_kd_loss = sum(train_kd_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            valid_kd_loss = sum(valid_kd_loss) / len(val_loader.dataset)
            train_pizdolos = sum(train_pizdolos) / len(train_pizdolos)
            train_acc = train_hits / len(trn_loader.dataset)
            val_acc = val_hits / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train: {train_loss:.2f} KD: {train_kd_loss:.3f} Acc: {100 * train_acc:.2f} Singularity: {train_pizdolos:.3f} "
                  f"Val: {valid_loss:.2f} KD: {valid_kd_loss:.3f} Acc: {100 * val_acc:.2f}")

        if self.distillation == "logit":
            self.heads.append(criterion.head)

    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader, num_classes_in_t):
        """ Creating distributions for task t"""
        self.model.eval()
        transforms = val_loader.dataset.transform
        new_means = torch.zeros((num_classes_in_t, self.S), device=self.device)
        new_covs = torch.zeros((num_classes_in_t, self.S, self.S), device=self.device)
        for c in range(num_classes_in_t):
            train_indices = torch.tensor(trn_loader.dataset.labels) == c + self.task_offset[t]
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = trn_loader.dataset.images[train_indices]
                ds = ClassMemoryDataset(ds, transforms)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((2 * len(ds), self.S), fill_value=0., device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                class_features[from_: from_+bsz] = features
                features = self.model(torch.flip(images, dims=(3,)))
                class_features[from_+bsz: from_+2*bsz] = features
                from_ += 2*bsz

            # Calculate  mean and cov
            new_means[c] = class_features.mean(dim=0)
            new_covs[c] = self.shrink_cov(torch.cov(class_features.T), self.default_shrink)
            if self.adaptation_strategy == "diag":
                new_covs[c] = torch.diag(torch.diag(new_covs[c]))

            # print(f"Rank {c + self.task_offset[t]}: {torch.linalg.matrix_rank(new_covs[c])}")

            if torch.isnan(new_covs[c]).any():
                raise RuntimeError(f"Nan in covariance matrix of class {c}")

        self.means = torch.cat((self.means, new_means), dim=0)
        self.covs = torch.cat((self.covs, new_covs), dim=0)

    def adapt_distributions(self, t, trn_loader, val_loader):
        # Train the adapter
        self.model.eval()
        adapter = nn.Linear(self.S, self.S)
        if self.adapter_type == "mlp":
            adapter = nn.Sequential(nn.Linear(self.S, self.multiplier * self.S),
                                    nn.GELU(),
                                    nn.Linear(self.multiplier * self.S, self.S)
                                    )
        adapter.to(self.device, non_blocking=True)
        # state_dict = torch.load(f"../ckpts/adapter_{t}.pth")
        # adapter.load_state_dict(state_dict, strict=True)
        optimizer, lr_scheduler = self.get_adapter_optimizer(adapter.parameters())
        old_means = copy.deepcopy(self.means)
        old_covs = copy.deepcopy(self.covs)
        for epoch in range(self.nepochs // 2):
            adapter.train()
            train_loss, valid_loss = [], []
            for images, _ in trn_loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                with torch.no_grad():
                    target = self.model(images)
                    old_features = self.old_model(images)
                adapted_features = adapter(old_features)
                loss = torch.nn.functional.mse_loss(adapted_features, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1)
                optimizer.step()
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            adapter.eval()
            with torch.no_grad():
                for images, _ in val_loader:
                    bsz = images.shape[0]
                    images = images.to(self.device, non_blocking=True)
                    target = self.model(images)
                    old_features = self.old_model(images)
                    adapted_features = adapter(old_features)
                    loss = torch.nn.functional.mse_loss(adapted_features, target)
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} ")

        if self.dump:
            torch.save(adapter.state_dict(), f"{self.logger.exp_path}/adapter_{t}.pth")

        # Adapt
        with torch.no_grad():
            adapter.eval()
            if self.adaptation_strategy == "mean-only":
                self.means = adapter(self.means)

            if self.adaptation_strategy == "full" or self.adaptation_strategy == "diag":
                for c in range(self.means.shape[0]):
                    cov = self.covs[c].clone()
                    distribution = MultivariateNormal(self.means[c], cov)
                    samples = distribution.sample((self.N,))
                    if torch.isnan(samples).any():
                        raise RuntimeError(f"Nan in features sampled for class {c}")
                    adapted_samples = adapter(samples)
                    self.means[c] = adapted_samples.mean(0)
                    # print(f"Rank pre-adapt {c}: {torch.linalg.matrix_rank(self.covs[c])}")
                    self.covs[c] = torch.cov(adapted_samples.T)
                    if self.adaptation_strategy == "diag":
                        self.covs[c] = torch.diag(torch.diag(self.covs[c]))

            print("### Adaptation evaluation ###")
            for (subset, loaders) in [("train", self.train_data_loaders), ("val", self.val_data_loaders)]:
                old_mean_diff, new_mean_diff = [], []
                old_kld, new_kld = [], []
                old_cov_diff, old_cov_norm_diff, new_cov_diff, new_cov_norm_diff = [], [], [], []
                class_images = np.concatenate([dl.dataset.images for dl in loaders[-2:-1]])
                labels = np.concatenate([dl.dataset.labels for dl in loaders[-2:-1]])

                for c in list(np.unique(labels)):
                    train_indices = torch.tensor(labels) == c

                    if isinstance(trn_loader.dataset.images, list):
                        train_images = list(compress(trn_loader.dataset.images, train_indices))
                        ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
                    else:
                        ds = ClassMemoryDataset(class_images[train_indices], val_loader.dataset.transform)
                    loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
                    from_ = 0
                    class_features = torch.full((2 * len(ds), self.S), fill_value=0., device=self.device)
                    for images in loader:
                        bsz = images.shape[0]
                        images = images.to(self.device, non_blocking=True)
                        features = self.model(images)
                        class_features[from_: from_+bsz] = features
                        features = self.model(torch.flip(images, dims=(3,)))
                        class_features[from_+bsz: from_+2*bsz] = features
                        from_ += 2*bsz

                    gt_mean = class_features.mean(0)
                    gt_cov = torch.cov(class_features.T)
                    gt_cov = self.shrink_cov(gt_cov, self.default_shrink)
                    gt_gauss = torch.distributions.MultivariateNormal(gt_mean, gt_cov)
                    if self.adaptation_strategy == "diag":
                        gt_cov = torch.diag(torch.diag(gt_cov))

                    # Calculate old diffs
                    old_mean_diff.append((gt_mean - old_means[c]).norm())
                    old_cov_diff.append(torch.norm(gt_cov - old_covs[c]))
                    old_cov_norm_diff.append(torch.norm(self.norm_cov(gt_cov.unsqueeze(0)) - self.norm_cov(old_covs[c].unsqueeze(0))))
                    old_gauss = torch.distributions.MultivariateNormal(old_means[c], old_covs[c])
                    old_kld.append(torch.distributions.kl_divergence(old_gauss, gt_gauss) + torch.distributions.kl_divergence(gt_gauss, old_gauss))
                    # Calculate new diffs
                    new_mean_diff.append((gt_mean - self.means[c]).norm())
                    new_cov_diff.append(torch.norm(gt_cov - self.covs[c]))
                    new_cov_norm_diff.append(torch.norm(self.norm_cov(gt_cov.unsqueeze(0)) - self.norm_cov(self.covs[c].unsqueeze(0))))
                    new_gauss = torch.distributions.MultivariateNormal(self.means[c], self.covs[c])
                    new_kld.append(torch.distributions.kl_divergence(new_gauss, gt_gauss) + torch.distributions.kl_divergence(gt_gauss, new_gauss))

                old_mean_diff = torch.stack(old_mean_diff)
                old_cov_diff = torch.stack(old_cov_diff)
                old_cov_norm_diff = torch.stack(old_cov_norm_diff)
                old_kld = torch.stack(old_kld)

                new_mean_diff = torch.stack(new_mean_diff)
                new_cov_diff = torch.stack(new_cov_diff)
                new_cov_norm_diff = torch.stack(new_cov_norm_diff)
                new_kld = torch.stack(new_kld)
                print(f"Old {subset} mean diff: {old_mean_diff.mean():.2f} ± {old_mean_diff.std():.2f}")
                print(f"New {subset} mean diff: {new_mean_diff.mean():.2f} ± {new_mean_diff.std():.2f}")
                print(f"Old {subset} cov diff: {old_cov_diff.mean():.2f} ± {old_cov_diff.std():.2f}")
                print(f"New {subset} cov diff: {new_cov_diff.mean():.2f} ± {new_cov_diff.std():.2f}")
                print(f"Old {subset} norm-cov diff: {old_cov_norm_diff.mean():.2f} ± {old_cov_norm_diff.std():.2f}")
                print(f"New {subset} norm-cov diff: {new_cov_norm_diff.mean():.2f} ± {new_cov_norm_diff.std():.2f}")
                print(f"Old {subset} KLD: {old_kld.mean():.2f} ± {old_kld.std():.2f}")
                print(f"New {subset} KLD: {new_kld.mean():.2f} ± {new_kld.std():.2f}")
                print("")

    def distill_projected(self, t, loss, features, distiller, images):
        """ Projected distillation through the distiller"""
        if t == 0:
            return loss, 0
        with torch.no_grad():
            old_features = self.old_model(images)
        kd_loss = F.mse_loss(distiller(features), old_features)
        total_loss = loss + self.lamb * kd_loss
        return total_loss, kd_loss

    def distill_features(self, t, loss, features, images):
        """ Projected distillation through the distiller"""
        if t == 0:
            return loss, 0
        with torch.no_grad():
            old_features = self.old_model(images)
        kd_loss = F.mse_loss(features, old_features)
        total_loss = loss + self.lamb * kd_loss
        return total_loss, kd_loss

    def distill_logits(self, t, loss, features, images, old_heads):
        """ Projected distillation through the distiller"""
        if t == 0:
            return loss, 0
        with torch.no_grad():
            old_features = self.old_model(images)
            old_logits = torch.cat([head(old_features) for head in old_heads], dim=1)
        new_logits = torch.cat([head(features) for head in self.heads], dim=1)
        kd_loss = self.cross_entropy(new_logits, old_logits, exp=1 / self.tau)
        total_loss = loss + self.lamb * kd_loss
        return total_loss, kd_loss

    def get_optimizer(self, parameters, t, wd):
        """Returns the optimizer"""
        milestones = (int(0.3*self.nepochs), int(0.6*self.nepochs), int(0.9*self.nepochs))
        optimizer = torch.optim.SGD(parameters, lr=self.lr if t == 0 else 0.1*self.lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_adapter_optimizer(self, parameters, milestones=(30, 60, 90)):
        """Returns the optimizer"""
        optimizer = torch.optim.SGD(parameters, lr=0.01, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    @torch.no_grad()
    def eval(self, t, val_loader):
        """ Perform nearest centroids classification """
        self.model.eval()
        tag_acc = Accuracy("multiclass", num_classes=self.means.shape[0])
        taw_acc = Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t]
        for images, target in val_loader:
            images = images.to(self.device, non_blocking=True)
            features = self.model(images)
            if self.is_mahalanobis:
                if self.is_normalization:
                    diff = F.normalize(features.unsqueeze(1), p=2, dim=-1) - F.normalize(self.means.unsqueeze(0), p=2, dim=-1)
                else:
                    diff = features.unsqueeze(1) - self.means.unsqueeze(0)
                res = diff.unsqueeze(2) @ self.covs_inverted.unsqueeze(0)
                res = res @ diff.unsqueeze(3)
                dist = res.squeeze(2).squeeze(2)
            else:  # Euclidean
                dist = torch.cdist(features, self.means)
            tag_preds = torch.argmin(dist, dim=1)
            taw_preds = torch.argmin(dist[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset

            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)

        return 0, float(taw_acc.compute()), float(tag_acc.compute())

    @torch.no_grad()
    def check_singular_values(self, t, val_loader):
        self.model.eval()
        self.svals_explained_by.append([])
        for i, _ in enumerate(self.train_data_loaders):
            if isinstance(self.train_data_loaders[i].dataset.images, list):
                train_images = self.train_data_loaders[i].dataset.images
                ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
            else:
                ds = ClassMemoryDataset(self.train_data_loaders[i].dataset.images, val_loader.dataset.transform)

            loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=val_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((len(ds), self.S), fill_value=-999999999.0, device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                class_features[from_: from_ + bsz] = features
                from_ += bsz

            cov = torch.cov(class_features.T)
            svals = torch.linalg.svdvals(cov)
            xd = torch.cumsum(svals, 0)
            xd = xd[xd < self.sval_fraction * torch.sum(svals)]
            explain = xd.shape[0]
            self.svals_explained_by[t].append(explain)

    @torch.no_grad()
    def print_singular_values(self):
        print(f"### {self.sval_fraction} of eigenvalues sum is explained by: ###")
        for t, explained_by in enumerate(self.svals_explained_by):
            print(f"Task {t}: {explained_by}")

    @torch.no_grad()
    def shrink_cov(self, cov, alpha1=1., alpha2=0.):
        if alpha2 == -1.:
            return cov + alpha1 * torch.eye(cov.shape[0], device=self.device)  # ordinary epsilon
        diag_mean = torch.mean(torch.diagonal(cov))
        iden = torch.eye(cov.shape[0], device=self.device)
        mask = iden == 0.0
        off_diag_mean = torch.mean(cov[mask])
        return cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))

    @torch.no_grad()
    def norm_cov(self, cov):
        diag = torch.diagonal(cov, dim1=1, dim2=2)
        std = torch.sqrt(diag)
        cov = cov / (std.unsqueeze(2) @ std.unsqueeze(1))
        return cov

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    @torch.no_grad()
    def print_covs(self, trn_loader, val_loader):
        self.model.eval()
        print("### Norms per task: ###")
        gt_means, gt_covs, gt_inverted_covs = [], [], []
        class_images = np.concatenate([dl.dataset.images for dl in self.train_data_loaders])
        labels = np.concatenate([dl.dataset.labels for dl in self.train_data_loaders])

        # Calculate ground truth
        for c in list(np.unique(labels)):
            train_indices = torch.tensor(labels) == c

            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
            else:
                ds = ClassMemoryDataset(class_images[train_indices], val_loader.dataset.transform)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((2 * len(ds), self.S), fill_value=0., device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                class_features[from_: from_ + bsz] = features
                features = self.model(torch.flip(images, dims=(3,)))
                class_features[from_ + bsz: from_ + 2 * bsz] = features
                from_ += 2 * bsz

            gt_means.append(class_features.mean(0))
            cov = torch.cov(class_features.T)
            gt_covs.append(cov)
            gt_inverted_covs.append(torch.inverse(self.shrink_cov(cov, self.default_shrink)))

        gt_means = torch.stack(gt_means)
        gt_covs = torch.stack(gt_covs)
        gt_inverted_covs = torch.stack(gt_inverted_covs)

        # Calculate statistics per task
        mean_norms, cov_norms = [], []
        gt_mean_norms, gt_cov_norms = [], []
        inverted_cov_norms, gt_inverted_cov_norms = [], []
        for task in range(len(self.task_offset[1:])):
            from_ = self.task_offset[task]
            to_ = self.task_offset[task + 1]
            mean_norms.append(round(float(torch.norm(self.means[from_:to_], dim=1).mean()), 2))
            cov_norms.append(round(float(torch.linalg.matrix_norm(self.covs[from_:to_]).mean()), 2))
            inverted_cov_norms.append(round(float(torch.linalg.matrix_norm(torch.inverse(self.covs[from_:to_])).mean()), 2))  # no shrink, no norm!
            gt_mean_norms.append(round(float(torch.norm(gt_means[from_:to_], dim=1).mean()), 2))
            gt_cov_norms.append(round(float(torch.linalg.matrix_norm(gt_covs[from_:to_]).mean()), 2))
            gt_inverted_cov_norms.append(round(float(torch.linalg.matrix_norm(gt_inverted_covs[from_:to_]).mean()), 2))
        print(f"Means: {mean_norms}")
        print(f"GT Means: {gt_mean_norms}")
        print(f"Covs: {cov_norms}")
        print(f"GT Covs: {gt_cov_norms}")
        print(f"Inverted Covs: {inverted_cov_norms}")
        print(f"GT Inverted Covs: {gt_inverted_cov_norms}")


def compute_rotations(images, targets, total_classes):
    # compute self-rotation for the first task following PASS https://github.com/Impression2805/CVPR21_PASS
    images_rot = torch.cat([torch.rot90(images, k, (2, 3)) for k in range(1, 4)])
    images = torch.cat((images, images_rot))
    target_rot = torch.cat([(targets + total_classes * k) for k in range(1, 4)])
    targets = torch.cat((targets, target_rot))
    return images, targets


def loss_pizdolos(features):
    # Idea 1 - determinant
    cov = torch.cov(features.T)
    det = torch.det(cov)
    loss = - torch.log(torch.clamp(torch.abs(det), max=1))
    return loss
