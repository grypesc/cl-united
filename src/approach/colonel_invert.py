import copy
import random
import torch
import numpy as np

from argparse import ArgumentParser
from itertools import compress
from torch import nn
from torch.utils.data import Dataset
from torchmetrics import Accuracy

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .models.resnet32 import resnet8, resnet14, resnet20, resnet32
from .models.resnet18 import resnet18
from .incremental_learning import Inc_Learning_Appr
from .criterions.proxy_yolo import ProxyYolo

torch.backends.cuda.matmul.allow_tf32 = False


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=1, cross_batch_distill=False, cross_batch_adapt=False,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, N=10, K=11, S=64, beta=100, distiller="linear", head="linear", alpha=0.5, smoothing=0., sval_fraction=0.95, use_gt_features=False, nnet="resnet32"):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)

        self.N = N
        self.K = K
        if K > N:
            raise RuntimeError("K cannot be grater than N")
        self.S = S
        self.cross_batch_distill = cross_batch_distill
        self.cross_batch_adapt = cross_batch_adapt
        self.beta = beta
        self.use_gt_features = use_gt_features
        self.alpha = alpha
        self.smoothing = smoothing
        self.patience = patience
        self.old_model = None
        self.model = {"resnet8": resnet8(num_features=S),
                      "resnet14": resnet14(num_features=S),
                      "resnet20": resnet20(num_features=S),
                      "resnet32": resnet32(num_features=S),
                      "resnet18": resnet18(num_features=S, is_32=True)}[nnet]
        # if nnet == "resnet18":
        #     self.S = 512
        self.model.fc = nn.Identity()
        self.model.to(device, non_blocking=True)
        self.train_data_loaders, self.val_data_loaders = [], []
        self.prototypes = torch.empty((0, self.S), device=self.device)
        self.prototypes_class = torch.empty((0, ), dtype=torch.int, device=self.device)
        self.task_offset = [0]
        self.classes_in_tasks = []
        self.head_type = head
        self.sval_fraction = sval_fraction
        self.svals_explained_by = []
        self.distiller_type = distiller



    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--N',
                            help='Number of learners',
                            type=int,
                            default=10)
        parser.add_argument('--K',
                            help='number of learners sampled for task',
                            type=int,
                            default=21)
        parser.add_argument('--S',
                            help='latent space size',
                            type=int,
                            default=64)
        parser.add_argument('--beta',
                            help='latent space size',
                            type=int,
                            default=100)
        parser.add_argument('--alpha',
                            help='relative weight of kd loss',
                            type=float,
                            default=0.5)
        parser.add_argument('--sval-fraction',
                            help='Fraction of eigenvalues sum that is explained',
                            type=float,
                            default=0.95)
        parser.add_argument('--use-gt-features',
                            help='Use GT features instead of adapting',
                            action='store_true',
                            default=False)
        parser.add_argument('--cross-batch-distill',
                            help='xxx',
                            action='store_true',
                            default=False)
        parser.add_argument('--cross-batch-adapt',
                            help='xxx',
                            action='store_true',
                            default=False)
        parser.add_argument('--distiller',
                            help='Distiller',
                            type=str,
                            choices=["linear", "mlp"],
                            default="linear")
        parser.add_argument('--head',
                            help='head',
                            type=str,
                            choices=["linear", "mlp"],
                            default="linear")
        parser.add_argument('--smoothing',
                            help='label smoothing',
                            type=float,
                            default=0.0)
        parser.add_argument('--nnet',
                            type=str,
                            choices=["resnet8", "resnet14", "resnet20", "resnet32", "resnet18"],
                            default="resnet32")
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
        self.train_backbone(t, trn_loader, val_loader, num_classes_in_t)
        # torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model_{t}.pth")
        if t > 0 and self.use_gt_features:
            self.adapt_prototypes_gt(t, val_loader)
        print("### Creating new prototypes ###")
        self.create_prototypes(t, trn_loader, val_loader, num_classes_in_t)
        self.check_singular_values(t, val_loader)
        self.print_singular_values()

        print("Proto norm statistics:")
        norms = torch.norm(self.prototypes, dim=1)
        print(f"Mean: {norms.mean():.2f}, median: {norms.median():.2f}")
        print(f"Range: [{norms.min():.2f}; {norms.max():.2f}]")

    def train_backbone(self, t, trn_loader, val_loader, num_classes_in_t):
        print(f'The model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,} shared parameters\n')

        distiller = nn.Linear(self.S, self.S)
        if self.distiller_type == "mlp":
            distiller = nn.Sequential(nn.Linear(self.S, 4 * self.S),
                                      nn.LeakyReLU(),
                                      nn.Linear(4 * self.S, self.S)
                                      )

        distiller.to(self.device, non_blocking=True)
        criterion = ProxyYolo(num_classes_in_t, self.S, self.device, smoothing=0)
        parameters = list(self.model.parameters()) + list(distiller.parameters()) + list(criterion.parameters())
        optimizer, lr_scheduler = self.get_optimizer(parameters, self.wd * (t == 0), self.nepochs)

        for epoch in range(self.nepochs):
            train_loss, train_kd_loss, valid_loss, valid_kd_loss = [], [], [], []
            self.model.train()
            distiller.train()
            criterion.train()
            for images, targets in trn_loader:
                targets -= self.task_offset[t]
                bsz = images.shape[0]
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                features = self.model(images)
                if epoch < 20 and t > 0:
                    features = features.detach()
                nca_loss, _, _ = criterion(features, targets)
                if self.cross_batch_distill and t > 0:
                    images = self.interpolate_cross_batch(images)
                    features = self.model(images)
                with torch.no_grad():
                    old_features = self.old_model(images) if t > 0 else None
                adapted_features = distiller(features) if t > 0 else None

                total_loss, kd_loss = self.distill_knowledge(nca_loss, adapted_features, old_features)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                optimizer.step()
                train_loss.append(float(bsz * nca_loss))
                train_kd_loss.append(float(bsz * kd_loss))
            lr_scheduler.step()

            self.model.eval()
            distiller.eval()
            criterion.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    targets -= self.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                    features = self.model(images)
                    nca_loss, _, _ = criterion(features, targets)

                    if self.cross_batch_distill and t > 0:
                        images = self.interpolate_cross_batch(images)
                        features = self.model(images)

                    old_features = self.old_model(images) if t > 0 else None
                    adapted_features = distiller(features) if t > 0 else None

                    _, kd_loss = self.distill_knowledge(nca_loss, adapted_features, old_features)

                    valid_loss.append(float(bsz * nca_loss))
                    valid_kd_loss.append(float(bsz * kd_loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            train_kd_loss = sum(train_kd_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            valid_kd_loss = sum(valid_kd_loss) / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train: {train_loss:.2f} KD: {train_kd_loss:.3f} "
                  f"Val: {valid_loss:.2f} KD: {valid_kd_loss:.3f}")

        print("### Adapting protos via inverting distiller ###")
        if t > 0:
            distiller.eval()
            self.prototypes = self.adapt_protos_from_distiller(distiller)

    def interpolate_cross_batch(self, images):
        bsz = images.shape[0]
        idx = torch.randint(0, high=bsz - 1, size=(bsz, 2), device=self.device)
        weights = torch.rand((bsz, 1, 1, 1), device=self.device)
        weights = torch.stack((weights, 1 - weights), dim=1)
        images_interpolated = images[idx[:, 0]] * weights[:, 0] + images[idx[:, 1]] * weights[:, 1]
        return images_interpolated

    @torch.no_grad()
    def adapt_protos_from_distiller(self, distiller):
        W = copy.deepcopy(distiller.weight.data.detach())
        b = copy.deepcopy(distiller.bias.data.detach())
        rank = torch.linalg.matrix_rank(W)
        print(f"Rank: {rank}")
        adapted_protos = torch.linalg.solve(W.T, self.prototypes - b.unsqueeze(0), left=False)
        return adapted_protos

    @torch.no_grad()
    def create_prototypes(self, t, trn_loader, val_loader, num_classes_in_t):
        """ Create distributions for task t"""
        self.model.eval()
        transforms = val_loader.dataset.transform
        for c in range(num_classes_in_t):
            c = c + self.task_offset[t]
            train_indices = torch.tensor(trn_loader.dataset.labels) == c
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = trn_loader.dataset.images[train_indices]
                ds = ClassMemoryDataset(ds, transforms)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=True)
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

            # Calculate centroid
            if self.N == 1:
                new_protos = torch.mean(class_features, dim=0).unsqueeze(0)
            else:
                new_protos = class_features[::2][:self.N]
            self.prototypes = torch.cat((self.prototypes, new_protos), dim=0)
            self.prototypes_class = torch.cat((self.prototypes_class, torch.full((self.N,), fill_value=c, device=self.device)), dim=0)


    @torch.no_grad()
    def eval(self, t, val_loader):
        """ Perform nearest centroids classification """
        self.model.eval()
        tag_acc = Accuracy("multiclass", num_classes=sum(self.classes_in_tasks))
        taw_acc = Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t] * self.N
        for images, target in val_loader:
            images = images.to(self.device, non_blocking=True)
            features = self.model(images)
            dist = torch.cdist(features, self.prototypes)
            _, nearest_n = torch.topk(dist, self.K, 1, largest=False)
            nearest_n = self.prototypes_class[nearest_n]
            tag_preds = torch.mode(nearest_n, 1)[0]

            _, nearest_n = torch.topk(dist[:, offset: offset + self.N * self.classes_in_tasks[t]], self.K, 1, largest=False)
            nearest_n += offset
            nearest_n = self.prototypes_class[nearest_n]
            taw_preds = torch.mode(nearest_n, 1)[0]

            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)

        return 0, float(taw_acc.compute()), float(tag_acc.compute())

    def distill_knowledge(self, loss, adapted_features, old_features=None):
        if old_features is None:
            return loss, 0
        kd_loss = nn.functional.mse_loss(adapted_features, old_features)
        total_loss = loss + self.alpha * kd_loss
        return total_loss, kd_loss

    def get_optimizer(self, parameters, wd, epochs):
        """Returns the optimizer"""
        milestones = (int(0.3*epochs), int(0.6*epochs), int(0.8*epochs))
        optimizer = torch.optim.SGD(parameters, lr=self.lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_adapter_optimizer(self, parameters, milestones=(40, 80)):
        """Returns the optimizer"""
        optimizer = torch.optim.SGD(parameters, lr=0.01, weight_decay=1e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    @torch.no_grad()
    def adapt_prototypes_gt(self, t, val_loader):
        """ Use GT data loaders to calculate features"""
        self.model.eval()
        self.prototypes = torch.empty((0, self.S), device=self.device)
        self.prototypes_class = torch.empty((0, ), dtype=torch.int, device=self.device)
        transforms = val_loader.dataset.transform
        for trn_loader in self.train_data_loaders[:-1]:
            for c in list(np.unique(trn_loader.dataset.labels)):
                train_indices = torch.tensor(trn_loader.dataset.labels) == c
                if isinstance(trn_loader.dataset.images, list):
                    train_images = list(compress(trn_loader.dataset.images, train_indices))
                    ds = ClassDirectoryDataset(train_images, transforms)
                else:
                    ds = trn_loader.dataset.images[train_indices]
                    ds = ClassMemoryDataset(ds, transforms)
                loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=True)
                from_ = 0
                class_features = torch.zeros((len(ds), self.S), device=self.device)
                for images in loader:
                    bsz = images.shape[0]
                    images = images.to(self.device, non_blocking=True)
                    features = self.model(images)
                    class_features[from_: from_+bsz] = features
                    from_ += bsz

                # Calculate centroid
                if self.N == 1:
                    new_protos = torch.mean(class_features, dim=0).unsqueeze(0)
                else:
                    new_protos = class_features[:self.N]
                self.prototypes = torch.cat((self.prototypes, new_protos), dim=0)
                self.prototypes_class = torch.cat((self.prototypes_class, torch.full((self.N,), fill_value=c, device=self.device)), dim=0)

    @torch.no_grad()
    def check_singular_values(self, t, val_loader):
        self.model.eval()
        self.svals_explained_by.append([])
        for i, _ in enumerate(self.train_data_loaders):
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

    def print_singular_values(self):
        print(f"{self.sval_fraction} of eigenvalues sum is explained by:")
        for t, explained_by in enumerate(self.svals_explained_by):
            print(f"Task {t}: {explained_by}")
