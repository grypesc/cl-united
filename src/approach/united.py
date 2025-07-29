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


class SampledDataset(torch.utils.data.Dataset):
    """ Dataset that samples pseudo prototypes from memorized distributions to train pseudo head """
    def __init__(self, distributions, samples, task_offset):
        self.distributions = distributions
        self.samples = samples
        self.total_classes = task_offset[-1]

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        target = random.randint(0, self.total_classes-1)
        val = self.distributions[target].sample()
        return val, target


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=1,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, N=10000, K=5, alpha=1., lr_backbone=0.01, lr_adapter=0.01, beta=1., distillation="projected", use_224=False, S=64, dump=False, rotation=False, distiller="linear", adapter="linear", criterion="proxy-nca", lamb=10, tau=2, smoothing=0., sval_fraction=0.95,
                 adaptation_strategy="full", pretrained_net=False, normalize=False, shrink=0., multiplier=32, classifier="bayes"):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)

        self.N = N
        self.K = K
        self.S = S
        self.dump = dump
        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        self.tau = tau
        self.lr_backbone = lr_backbone
        self.lr_adapter = lr_adapter
        self.multiplier = multiplier
        self.shrink = shrink
        self.smoothing = smoothing
        self.adaptation_strategy = adaptation_strategy
        self.old_model = None
        self.model = None
        self.models = nn.ModuleList([resnet18(num_features=S, is_224=use_224) for _ in range(self.K)])
        self.pretrained = pretrained_net
        if pretrained_net:
            # wget https://download.pytorch.org/models/resnet18-f37072fd.pth
            state_dict = torch.load("../resnet18-f37072fd.pth")
            del state_dict["fc.weight"]
            del state_dict["fc.bias"]
            for i in range(self.K):
                self.models[i].load_state_dict(state_dict, strict=False)
        for i in range(self.K):
            self.models[i].to(device, non_blocking=True)
        self.train_data_loaders, self.val_data_loaders = [], []
        self.means = [torch.empty((self.K, 0, self.S), device=self.device) for _ in range(self.K)]
        self.covs = [torch.empty((self.K, 0, self.S, self.S), device=self.device) for _ in range(self.K)]
        self.covs_inverted = [torch.empty((self.K, 0, self.S, self.S), device=self.device) for _ in range(self.K)]
        self.classifier = classifier
        self.is_normalization = normalize
        self.is_rotation = rotation
        self.task_offset = [0]
        self.classes_in_tasks = []
        self.criterion_type = criterion
        self.criterion = {"proxy-yolo": ProxyYolo,
                          "proxy-nca": ProxyNCA,
                          "ce": CE}[criterion]
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
                            help='Number of experts',
                            type=int,
                            default=5)
        parser.add_argument('--S',
                            help='latent space size',
                            type=int,
                            default=64)
        parser.add_argument('--alpha',
                            help='Weight of anti-collapse loss',
                            type=float,
                            default=1.0)
        parser.add_argument('--beta',
                            help='Anti-collapse loss clamp',
                            type=float,
                            default=1.0)
        parser.add_argument('--lamb',
                            help='Weight of kd loss',
                            type=float,
                            default=10)
        parser.add_argument('--lr-backbone',
                            help='lr for backbone of the pretrained model',
                            type=float,
                            default=0.01)
        parser.add_argument('--lr-adapter',
                            help='lr for backbone of the adapter',
                            type=float,
                            default=0.01)
        parser.add_argument('--multiplier',
                            help='mlp multiplier',
                            type=int,
                            default=32)
        parser.add_argument('--tau',
                            help='temperature for logit distill',
                            type=float,
                            default=2)
        parser.add_argument('--shrink',
                            help='shrink during inference',
                            type=float,
                            default=1.0)
        parser.add_argument('--sval-fraction',
                            help='Fraction of eigenvalues sum that is explained',
                            type=float,
                            default=0.95)
        parser.add_argument('--adaptation-strategy',
                            help='Activation functions in resnet',
                            type=str,
                            choices=["none", "mean", "diag", "full"],
                            default="full")
        parser.add_argument('--distiller',
                            help='Distiller',
                            type=str,
                            choices=["linear", "mlp"],
                            default="mlp")
        parser.add_argument('--adapter',
                            help='Adapter',
                            type=str,
                            choices=["linear", "mlp"],
                            default="mlp")
        parser.add_argument('--criterion',
                            help='Loss function',
                            type=str,
                            choices=["ce", "proxy-nca", "proxy-yolo"],
                            default="ce")
        parser.add_argument('--classifier',
                            help='Classifier type',
                            type=str,
                            choices=["bayes", "nmc"],
                            default="bayes")
        parser.add_argument('--distillation',
                            help='Loss function',
                            type=str,
                            choices=["projected", "logit", "feature", "none"],
                            default="projected")
        parser.add_argument('--smoothing',
                            help='label smoothing',
                            type=float,
                            default=0.0)
        parser.add_argument('--use-224',
                            help='Additional max pool and different conv1 in Resnet18',
                            action='store_true',
                            default=False)
        parser.add_argument('--pretrained-net',
                            help='Load pretrained weights',
                            action='store_true',
                            default=False)
        parser.add_argument('--normalize',
                            help='xxx',
                            action='store_true',
                            default=True)
        parser.add_argument('--dump',
                            help='save checkpoints',
                            action='store_true',
                            default=False)
        parser.add_argument('--rotation',
                            help='Rotate images in the first task to enhance feature extractor',
                            action='store_true',
                            default=False)
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

        # In the first task train all experts
        expert_to_train = t % self.K
        if t == 0:
            for m in self.models:
                self.train_expert(t, expert_to_train, trn_loader, val_loader, num_classes_in_t)
        if self.dump:
            torch.save(self.models.state_dict(), f"{self.logger.exp_path}/models_{t}.pth")
        if t > 0 and self.adaptation_strategy != "none":
            print("### Adapting prototypes ###")
            self.adapt_distributions_vanilla(t, trn_loader, val_loader)
        print("### Creating new prototypes ###\n")
        self.create_distributions(t, trn_loader, val_loader, num_classes_in_t)

        # Calculate inverted covariances for evaluation with mahalanobis
        covs = self.covs.clone()
        print(f"Cov matrix det: {torch.linalg.det(covs)}")
        for i in range(covs.shape[0]):
            print(f"Rank for class {i}: {torch.linalg.matrix_rank(self.covs_raw[i], tol=0.01)}, {torch.linalg.matrix_rank(self.covs[i], tol=0.01)}")
            covs[i] = self.shrink_cov(covs[i], 3)
        if self.is_normalization:
            covs = self.norm_cov(covs)
        self.covs_inverted = torch.inverse(covs)

    def train_expert(self, t, model, trn_loader, val_loader, num_classes_in_t):
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        print(f'The expert has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} shared parameters\n')
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
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size // 4, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size // 4, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)

        parameters = list(model.parameters()) + list(criterion.parameters()) + list(distiller.parameters())
        parameters_dict = [
            {"params": list(model.parameters())[:-1], "lr": self.lr_backbone},
            {"params": list(criterion.parameters()) + list(model.parameters())[-1:]},
            {"params": list(distiller.parameters())},
        ]
        optimizer, lr_scheduler = self.get_optimizer(parameters_dict if self.pretrained else parameters, t, self.wd)

        for epoch in range(self.nepochs):
            train_loss, train_kd_loss, valid_loss, valid_kd_loss = [], [], [], []
            train_hits, val_hits, train_total, val_total = 0, 0, 0, 0
            model.train()
            criterion.train()
            distiller.train()
            for images, targets in trn_loader:
                if t == 0 and self.is_rotation:
                    images, targets = compute_rotations(images, targets, num_classes_in_t)
                targets -= self.task_offset[t]
                bsz = images.shape[0]
                train_total += bsz
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                features = model(images)
                if epoch < int(self.nepochs * 0.01):
                    features = features.detach()
                loss, logits = criterion(features, targets)

                if self.distillation == "logit":
                    raise NotImplementedError("shiit")
                elif self.distillation == "projected":
                    total_loss, kd_loss = self.distill_projected(t, loss, features, distiller, images)
                elif self.distillation == "feature":
                    total_loss, kd_loss = self.distill_features(t, loss, features, images)
                else:  # no distillation
                    total_loss, kd_loss = loss, 0.

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                optimizer.step()
                if logits is not None:
                    train_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
                train_kd_loss.append(float(bsz * kd_loss))
            lr_scheduler.step()

            val_total = 1e-8
            if epoch % 10 == 9:
                model.eval()
                criterion.eval()
                distiller.eval()
                with torch.no_grad():
                    for images, targets in val_loader:
                        if t == 0 and self.is_rotation:
                            images, targets = compute_rotations(images, targets, num_classes_in_t)
                        targets -= self.task_offset[t]
                        bsz = images.shape[0]
                        val_total += bsz
                        images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                        features = model(images)
                        loss, logits = criterion(features, targets)
                        if self.distillation == "logit":
                            raise NotImplementedError("life is a bitch")
                        elif self.distillation == "projected":
                            _, kd_loss = self.distill_projected(t, loss, features, distiller, images)
                        elif self.distillation == "feature":
                            _, kd_loss = self.distill_features(t, loss, features, images)
                        else:  # no distillation
                            kd_loss = 0.

                        if logits is not None:
                            val_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                        valid_loss.append(float(bsz * loss))
                        valid_kd_loss.append(float(bsz * kd_loss))

            train_loss = sum(train_loss) / train_total
            train_kd_loss = sum(train_kd_loss) / train_total
            valid_loss = sum(valid_loss) / val_total
            valid_kd_loss = sum(valid_kd_loss) / val_total
            train_acc = train_hits / train_total
            val_acc = val_hits / val_total

            print(f"Epoch: {epoch} Train: {train_loss:.2f} KD: {train_kd_loss:.3f} Acc: {100 * train_acc:.2f}"
                  f"Val: {valid_loss:.2f} KD: {valid_kd_loss:.3f} Acc: {100 * val_acc:.2f}")

    def adapt_distributions_vanilla(self, t, trn_loader, val_loader):
        expert_to_train = t % self.K
        model = self.model[expert_to_train]
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        # Train the adapter
        model.eval()
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
                    target = model(images)
                    old_features = self.old_model(images)
                adapted_features = adapter(old_features)
                loss = torch.nn.functional.mse_loss(adapted_features, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1)
                optimizer.step()
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            if epoch % 10 == 9:
                adapter.eval()
                with torch.no_grad():
                    for images, _ in val_loader:
                        bsz = images.shape[0]
                        images = images.to(self.device, non_blocking=True)
                        target = model(images)
                        old_features = self.old_model(images)
                        adapted_features = adapter(old_features)
                        total_loss = torch.nn.functional.mse_loss(adapted_features, target)
                        valid_loss.append(float(bsz * total_loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f}")

        if self.dump:
            torch.save(adapter.state_dict(), f"{self.logger.exp_path}/adapter_{t}.pth")

        # Adapt
        with torch.no_grad():
            adapter.eval()
            if self.adaptation_strategy == "mean":
                self.means[expert_to_train] = adapter(self.means[expert_to_train])

            if self.adaptation_strategy == "full" or self.adaptation_strategy == "diag":
                for c in range(self.means.shape[1]):
                    cov = self.covs[expert_to_train, c].clone()
                    distribution = MultivariateNormal(self.means[c], cov)
                    samples = distribution.sample((self.N,))
                    if torch.isnan(samples).any():
                        raise RuntimeError(f"Nan in features sampled for class {c}")
                    adapted_samples = adapter(samples)
                    self.means[c] = adapted_samples.mean(0)
                    # print(f"Rank pre-adapt {c}: {torch.linalg.matrix_rank(self.covs[c])}")
                    self.covs[expert_to_train][c] = torch.cov(adapted_samples.T)
                    self.covs[expert_to_train][c] = self.shrink_cov(self.covs[c], self.shrink)
                    if self.adaptation_strategy == "diag":
                        self.covs[expert_to_train][c] = torch.diag(torch.diag(self.covs[expert_to_train, c]))

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
                        train_images = list(compress(class_images, train_indices))
                        ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
                    else:
                        ds = ClassMemoryDataset(class_images[train_indices], val_loader.dataset.transform)
                    loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
                    from_ = 0
                    class_features = torch.full((2 * len(ds), self.S), fill_value=0., device=self.device)
                    for images in loader:
                        bsz = images.shape[0]
                        images = images.to(self.device, non_blocking=True)
                        features = model(images)
                        class_features[from_: from_+bsz] = features
                        features = model(torch.flip(images, dims=(3,)))
                        class_features[from_+bsz: from_+2*bsz] = features
                        from_ += 2*bsz

                    gt_mean = class_features.mean(0)
                    gt_cov = torch.cov(class_features.T)
                    gt_cov = self.shrink_cov(gt_cov, self.shrink)
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

    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader, num_classes_in_t):
        """ Creating distributions for task t"""
        self.models.eval()
        transforms = val_loader.dataset.transform
        new_means = torch.zeros((self.K, num_classes_in_t, self.S), device=self.device)
        new_covs = torch.zeros((self.K, num_classes_in_t, self.S, self.S), device=self.device)
        new_covs_not_shrinked = torch.zeros((self.K, num_classes_in_t, self.S, self.S), device=self.device)
        # svals_task = torch.full((10, self.S), fill_value=0., device=self.device)
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
            class_features = torch.full((self.K, 2 * len(ds), self.S), fill_value=0., device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                for expert_num, model in enumerate(self.models):
                    features = model(images)
                    class_features[expert_num, from_: from_+bsz] = features
                    features = model(torch.flip(images, dims=(3,)))
                    class_features[expert_num, from_+bsz: from_+2*bsz] = features
                from_ += 2*bsz

            # svals = torch.linalg.svdvals(class_features)
            # torch.sort(svals, descending=True)
            # svals_task[c] = svals

            # Calculate  mean and cov
            new_means[:, c] = class_features.mean(dim=1)
            new_covs[:, c] = self.shrink_cov(torch.cov(class_features.T), self.shrink)
            if self.adaptation_strategy == "diag":
                new_covs[c] = torch.diag(torch.diag(new_covs[c]))

            if torch.isnan(new_covs[c]).any():
                raise RuntimeError(f"Nan in covariance matrix of class {c}")

        # np.savetxt("svals_collapse.txt", np.array(svals_task.mean(0).cpu()))
        self.means = torch.cat((self.means, new_means), dim=1)
        self.covs = torch.cat((self.covs, new_covs), dim=1)

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

    def get_optimizer(self, parameters, t, wd):
        """Returns the optimizer"""
        milestones = (int(0.3*self.nepochs), int(0.6*self.nepochs), int(0.9*self.nepochs))
        lr = self.lr
        if t > 0 and not self.pretrained:
            lr *= 0.1
        optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_adapter_optimizer(self, parameters, milestones=(30, 60, 90)):
        """Returns the optimizer"""
        optimizer = torch.optim.SGD(parameters, lr=self.lr_adapter, weight_decay=5e-4, momentum=0.9)
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
            if self.classifier == "linear":
                logits = self.pseudo_head(features)
                tag_preds = torch.argmax(logits, dim=1)
                taw_preds = torch.argmax(logits[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset
            else:
                if self.classifier == "bayes":  # Calcualte mahalanobis distances
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


def compute_rotations(images, targets, total_classes):
    # compute self-rotation for the first task following PASS https://github.com/Impression2805/CVPR21_PASS
    images_rot = torch.cat([torch.rot90(images, k, (2, 3)) for k in range(1, 4)])
    images = torch.cat((images, images_rot))
    target_rot = torch.cat([(targets + total_classes * k) for k in range(1, 4)])
    targets = torch.cat((targets, target_rot))
    return images, targets

