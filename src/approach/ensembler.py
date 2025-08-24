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
from .ensemble_utils.criterions import EnsembledCE
from .ensemble_utils.distillers import BaselineDistiller, ConcatenatedDistiller, AveragedDistiller
from .ensemble_utils.adapters import BaselineAdapter, ConcatenatedAdapter, AveragedAdapter, shrink_cov, norm_cov


class SampledDataset(torch.utils.data.Dataset):
    """ Dataset that samples pseudo prototypes from memorized distributions to train pseudo head """

    def __init__(self, distributions, samples, task_offset):
        self.distributions = distributions
        self.samples = samples
        self.total_classes = task_offset[-1]

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        target = random.randint(0, self.total_classes - 1)
        val = self.distributions[target].sample()
        return val, target


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=1,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, N=10000, K=5, alpha=1., lr_backbone=0.01, lr_adapter=0.01, beta=1., use_224=False, S=64, dump=False, rotation=False, distiller="linear", adapter="linear", criterion="proxy-nca", lamb=10, tau=2, smoothing=0.,
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
        self.old_models = None
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
        self.means = torch.empty((0, self.K, self.S), device=self.device)
        self.covs = torch.empty((0, self.K, self.S, self.S), device=self.device)
        self.covs_inverted = None
        self.old_means = None
        self.old_covs = None
        self.classifier = classifier
        self.is_normalization = normalize
        self.is_rotation = rotation
        self.task_offset = [0]
        self.classes_in_tasks = []
        self.criterion = {"ce": EnsembledCE}[criterion]
        self.adapter = {"baseline": BaselineAdapter, "concatenated": ConcatenatedAdapter,
                        "averaged": AveragedAdapter, "none": None}[adapter]
        self.distiller = {"baseline": BaselineDistiller, "concatenated": ConcatenatedDistiller,
                          "averaged": AveragedDistiller, "none": None}[distiller]

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
                            default=0.0)
        parser.add_argument('--adaptation-strategy',
                            help='Activation functions in resnet',
                            type=str,
                            choices=["none", "mean", "diag", "full"],
                            default="full")
        parser.add_argument('--adapter',
                            help='Adapter',
                            type=str,
                            choices=["baseline", "concatenated", "averaged", "none"],
                            default="baseline")
        parser.add_argument('--distiller',
                            help='Distiller',
                            type=str,
                            choices=["baseline", "concatenated", "averaged", "none"],
                            default="baseline")
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
        self.old_models = copy.deepcopy(self.models)
        self.old_models.eval()
        self.old_means = copy.deepcopy(self.means)
        self.old_covs = copy.deepcopy(self.covs)
        self.task_offset.append(num_classes_in_t + self.task_offset[-1])
        print("### Training backbone ###")
        # state_dict = torch.load(f"ckpts-5/models_{t}.pth")
        # self.models.load_state_dict(state_dict, strict=True)

        self.train_experts(t, trn_loader, val_loader, num_classes_in_t)
        if t > 0 and self.adaptation_strategy != "none":
            print("### Adapting gausses ###")
            self.adapt_distributions(t, trn_loader, val_loader)
            self.evaluate_adaptation(0, trn_loader, val_loader)
        if self.dump:
            torch.save(self.models.state_dict(), f"{self.logger.exp_path}/models_{t}.pth")
        print("### Creating new gausses ###\n")
        self.create_distributions(t, trn_loader, val_loader, num_classes_in_t)

        # Calculate inverted covariances for evaluation with mahalanobis
        covs = self.covs.clone()
        for expert_num in range(self.K):

            print(f"Cov matrix det: {torch.linalg.det(covs[expert_num])}")
            for i in range(covs.shape[1]):
                print(f"Rank for expert: {expert_num}, class {i}: {torch.linalg.matrix_rank(self.covs[expert_num, i], tol=0.01)}")
                covs[expert_num, i] = shrink_cov(covs[expert_num, i], 3)
            covs[expert_num] = norm_cov(covs[expert_num])

        self.covs_inverted = torch.linalg.inv(covs)

    def train_experts(self, t, trn_loader, val_loader, num_classes_in_t):
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        print(f'The expert has {sum(p.numel() for p in self.models.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in self.models.parameters() if not p.requires_grad):,} shared parameters\n')
        distiller = self.distiller(self.K, self.S, self.multiplier, "mlp")
        distiller.to(self.device, non_blocking=True)
        criterion = self.criterion(self.K, num_classes_in_t, self.S, self.device, smoothing=self.smoothing)
        if t == 0 and self.is_rotation:
            criterion = self.criterion(self.K, 4 * num_classes_in_t, self.S, self.device, smoothing=self.smoothing)
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size // 4, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size // 4, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)

        parameters = list(self.models.parameters()) + list(criterion.parameters()) + list(distiller.parameters())
        optimizer, lr_scheduler = self.get_optimizer(parameters if self.pretrained else parameters, t, self.wd)

        for epoch in range(self.nepochs):
            train_loss, train_kd_loss, valid_loss, valid_kd_loss = [], [], [], []
            train_hits, val_hits, train_total, val_total = 0, 0, 0, 0
            self.models.train()
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

                features = torch.zeros((bsz, self.K, self.S), device=self.device)
                for expert_num, model in enumerate(self.models):
                    features[:, expert_num] = model(images)
                if epoch < int(self.nepochs * 0.01):
                    features = features.detach()
                discriminative_loss = criterion(features, targets)

                # Perform knowledge distillation
                kd_loss = 0
                if t > 0 and self.distiller is not None:
                    old_features = torch.zeros((bsz, self.K, self.S), device=self.device)
                    with torch.no_grad():
                        for expert_num, old_model in enumerate(self.old_models):
                            old_features[:, expert_num] = old_model(images)
                    kd_loss = distiller(features, old_features)

                total_loss = discriminative_loss + self.lamb * kd_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                optimizer.step()
                train_loss.append(float(bsz * discriminative_loss))
                train_kd_loss.append(float(bsz * kd_loss))
            lr_scheduler.step()

            val_total = 1e-8
            if epoch % 10 == 0:
                self.models.eval()
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

                        features = torch.zeros((bsz, self.K, self.S), device=self.device)
                        for expert_num, model in enumerate(self.models):
                            features[:, expert_num] = model(images)
                        discriminative_loss = criterion(features, targets)

                        # Perform knowledge distillation
                        kd_loss = 0
                        if t > 0 and self.distiller is not None:
                            old_features = torch.zeros((bsz, self.K, self.S), device=self.device)
                            for expert_num, old_model in enumerate(self.old_models):
                                old_features[:, expert_num] = old_model(images)
                            kd_loss = distiller(features, old_features)

                        valid_loss.append(float(bsz * discriminative_loss))
                        valid_kd_loss.append(float(bsz * kd_loss))

            train_loss = sum(train_loss) / train_total
            train_kd_loss = sum(train_kd_loss) / train_total
            valid_loss = sum(valid_loss) / val_total
            valid_kd_loss = sum(valid_kd_loss) / val_total

            print(f"Epoch: {epoch} Train: {train_loss:.2f} KD: {train_kd_loss:.3f} "
                  f"Val: {valid_loss:.2f} KD: {valid_kd_loss:.3f}")

    def adapt_distributions(self, t, trn_loader, val_loader):
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        # Train the adapter
        self.models.eval()

        adapter = self.adapter(self.K, self.S, self.multiplier)
        adapter.to(self.device, non_blocking=True)
        # state_dict = torch.load(f"ckpts-5/adapter_{t}.pth")
        # adapter.load_state_dict(state_dict, strict=True)
        optimizer, lr_scheduler = self.get_adapter_optimizer(adapter.parameters())
        for epoch in range(self.nepochs // 2):
            adapter.train()
            train_loss, valid_loss = [], []
            for images, _ in trn_loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                new_features = torch.zeros((bsz, self.K, self.S), device=self.device)
                old_features = torch.zeros((bsz, self.K, self.S), device=self.device)
                with torch.no_grad():
                    for expert_num in range(self.K):
                        new_features[:, expert_num] = self.models[expert_num](images)
                        old_features[:, expert_num] = self.old_models[expert_num](images)
                loss = adapter(old_features, new_features)
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
                        new_features = torch.zeros((bsz, self.K, self.S), device=self.device)
                        old_features = torch.zeros((bsz, self.K, self.S), device=self.device)
                        for expert_num in range(self.K):
                            new_features[:, expert_num] = self.models[expert_num](images)
                            old_features[:, expert_num] = self.old_models[expert_num](images)
                        loss = adapter(old_features, new_features)
                        valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f}")

        if self.dump:
            torch.save(adapter.state_dict(), f"{self.logger.exp_path}/adapter_{t}.pth")

        # Adaptation
        adapter.eval()
        self.means, self.covs = adapter.adapt(self.means, self.covs)

    @torch.no_grad()
    def evaluate_adaptation(self, expert_trained, trn_loader, val_loader):
        print("### Evaluating adaptation ###")
        for (subset, loaders) in [("train", self.train_data_loaders), ("val", self.val_data_loaders)]:
            model = self.models[expert_trained]
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
                    class_features[from_: from_ + bsz] = features
                    features = model(torch.flip(images, dims=(3,)))
                    class_features[from_ + bsz: from_ + 2 * bsz] = features
                    from_ += 2 * bsz

                gt_mean = class_features.mean(0)
                gt_cov = torch.cov(class_features.T)
                gt_cov = shrink_cov(gt_cov, self.shrink)
                gt_gauss = torch.distributions.MultivariateNormal(gt_mean, gt_cov)

                # Calculate old diffs
                old_mean_diff.append((gt_mean - self.old_means[c, expert_trained]).norm())
                old_cov_diff.append(torch.norm(gt_cov - self.old_covs[c, expert_trained]))
                old_cov_norm_diff.append(torch.norm(norm_cov(gt_cov.unsqueeze(0)) - norm_cov(self.old_covs[c, expert_trained].unsqueeze(0))))
                old_gauss = torch.distributions.MultivariateNormal(self.old_means[c, expert_trained], self.old_covs[c, expert_trained])
                old_kld.append(torch.distributions.kl_divergence(old_gauss, gt_gauss) + torch.distributions.kl_divergence(gt_gauss, old_gauss))
                # Calculate new diffs
                new_mean_diff.append((gt_mean - self.means[c, expert_trained]).norm())
                new_cov_diff.append(torch.norm(gt_cov - self.covs[c, expert_trained]))
                new_cov_norm_diff.append(torch.norm(norm_cov(gt_cov.unsqueeze(0)) - norm_cov(self.covs[c, expert_trained].unsqueeze(0))))
                new_gauss = torch.distributions.MultivariateNormal(self.means[c, expert_trained], self.covs[c, expert_trained])
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
        new_means = torch.zeros((num_classes_in_t, self.K,  self.S), device=self.device)
        new_covs = torch.zeros((num_classes_in_t, self.K, self.S, self.S), device=self.device)

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
            class_features = torch.zeros((2 * len(ds), self.K, self.S), device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                for expert_num, model in enumerate(self.models):
                    features = model(images)
                    class_features[from_: from_ + bsz, expert_num] = features
                    features = model(torch.flip(images, dims=(3,)))
                    class_features[from_ + bsz: from_ + 2 * bsz, expert_num] = features
                from_ += 2 * bsz

            # svals = torch.linalg.svdvals(class_features)
            # torch.sort(svals, descending=True)
            # svals_task[c] = svals

            # Calculate  mean and cov
            new_means[c, :] = class_features.mean(dim=0)
            for expert_num, _ in enumerate(self.models):
                new_covs[c, expert_num] = shrink_cov(torch.cov(class_features[:, expert_num].T), self.shrink)
                if self.adaptation_strategy == "diag":
                    new_covs[c, expert_num] = torch.diag(torch.diag(new_covs[c, expert_num]))

            if torch.isnan(new_covs[c, :]).any():
                raise RuntimeError(f"Nan in covariance matrix of class {c}")

        # np.savetxt("svals_collapse.txt", np.array(svals_task.mean(0).cpu()))
        self.means = torch.cat((self.means, new_means), dim=0)
        self.covs = torch.cat((self.covs, new_covs), dim=0)

    def get_optimizer(self, parameters, t, wd):
        """Returns the optimizer"""
        milestones = (int(0.3 * self.nepochs), int(0.6 * self.nepochs), int(0.9 * self.nepochs))
        lr = self.lr
        if t > 0 and not self.pretrained:
            lr *= 0.33
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
        self.models.eval()
        tag_acc = Accuracy("multiclass", num_classes=self.means.shape[0])
        taw_acc = Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t]
        for images, target in val_loader:
            images = images.to(self.device, non_blocking=True)
            features = torch.zeros((images.shape[0], self.K, self.S), device=self.device)
            for expert_num, model in enumerate(self.models):
                features[:, expert_num] = model(images)

            if self.classifier == "bayes":  # Calculate mahalanobis distances
                dist = torch.zeros((images.shape[0], self.K, self.means.shape[0]), device=self.device)
                for expert_num in range(self.K):
                    diff = F.normalize(features[:, expert_num].unsqueeze(1), p=2, dim=-1) - F.normalize(self.means[:, expert_num].unsqueeze(0), p=2, dim=-1)
                    res = diff.unsqueeze(2) @ self.covs_inverted[:, expert_num].unsqueeze(0)
                    res = res @ diff.unsqueeze(3)
                    dist[:, expert_num] = res.squeeze(2).squeeze(2)
            else:  # Euclidean
                for expert_num in range(self.K):
                    dist[:, expert_num] = torch.cdist(features[:, expert_num], self.means[:, expert_num])

            dist = dist.mean(1)
            tag_preds = torch.argmin(dist, dim=1)
            taw_preds = torch.argmin(dist[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset

            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)

        return 0, float(taw_acc.compute()), float(tag_acc.compute())


def compute_rotations(images, targets, total_classes):
    # compute self-rotation for the first task following PASS https://github.com/Impression2805/CVPR21_PASS
    images_rot = torch.cat([torch.rot90(images, k, (2, 3)) for k in range(1, 4)])
    images = torch.cat((images, images_rot))
    target_rot = torch.cat([(targets + total_classes * k) for k in range(1, 4)])
    targets = torch.cat((targets, target_rot))
    return images, targets
