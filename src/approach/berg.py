import copy
import numpy as np
import torch

from argparse import ArgumentParser
from itertools import compress
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .gmm import GaussianMixture
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, max_experts=999, gmms=1, alpha=1.0, tau=3.0, use_multivariate=True, use_z_score=False, use_head=False, remove_outliers=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)
        self.max_experts = max_experts
        self.use_z_score = use_z_score
        self.gmms = gmms
        self.alpha = alpha
        self.tau = tau
        self.patience = patience
        self.use_multivariate = use_multivariate
        self.use_head = use_head
        self.remove_outliers = remove_outliers
        self.model.to(device)
        self.experts_distributions = []

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--max-experts',
                            help='Maximum number of experts',
                            type=int,
                            default=999)
        parser.add_argument('--gmms',
                            help='Number of gaussian models in the mixture',
                            type=int,
                            default=1)
        parser.add_argument('--use-multivariate',
                            help='Use multivariate distribution',
                            action='store_true',
                            default=False)
        parser.add_argument('--use-z-score',
                            help='Replace gumbel softmax with z-score normalized softmax',
                            action='store_true',
                            default=False)
        parser.add_argument('--alpha',
                            help='relative weight of kd loss',
                            type=float,
                            default=1.0)
        parser.add_argument('--tau',
                            help='gumbel softmax temperature',
                            type=float,
                            default=3.0)
        parser.add_argument('--remove-outliers',
                            help='Remove class outliers before creating distribution',
                            action='store_true',
                            default=False)
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        # Train backbone
        if t < self.max_experts:
            print(f"Training backbone on task {t}:")
            self.train_backbone(t, trn_loader, val_loader)
            self.experts_distributions.append([])
        else:
            print(f"Finetuning backbone on task {t}:")
            self.finetune_backbone(t, trn_loader, val_loader)

        # Create distributions
        print(f"Creating distributions for task {t}:")
        self.create_distributions(t, trn_loader, val_loader)

    def train_backbone(self, t, trn_loader, val_loader):
        if t == 0:
            model = self.model.bbs[0]
            for param in model.parameters():
                param.requires_grad = True
        else:
            self.model.bbs.append(copy.deepcopy(self.model.bbs[-1]))
            model = self.model.bbs[t]
            for name, param in model.named_parameters():
                param.requires_grad = True
                # if "layer2" in name or "layer3" in name or "layer4" in name:
                #     param.requires_grad = True
            model.fc = nn.Linear(self.model.num_features, self.model.taskcla[t][1])
        print(f'The expert has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} frozen parameters\n')

        model.to(self.device)
        optimizer, lr_scheduler = self._get_optimizer(t, self.wd)
        for epoch in range(self.nepochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for images, targets in trn_loader:
                targets -= self.model.task_offset[t]
                bsz = images.shape[0]
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                out = model(images)
                loss = self.criterion(t, out, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    targets -= self.model.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device), targets.to(self.device)
                    out = model(images)
                    loss = self.criterion(t, out, targets)

                    val_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            train_acc = train_hits / len(trn_loader.dataset)
            val_acc = val_hits / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                  f"Train acc: {100 * train_acc:.2f} Val acc: {100 * val_acc:.2f}")

        model.fc = nn.Identity()
        self.model.bbs[t] = model
        torch.save(self.model.state_dict(), "best.pth")

    def finetune_backbone(self, t, trn_loader, val_loader):
        """ This time use knowledge distillation to not let old distributions to drift too much """
        bb_to_finetune = t % self.max_experts
        old_model = copy.deepcopy(self.model.bbs[bb_to_finetune])
        for name, param in old_model.named_parameters():
            param.requires_grad = False
        old_model.eval()

        model = self.model.bbs[bb_to_finetune]
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "layer2" in name or "layer3" in name or "layer4" in name:
                param.requires_grad = True
        model.fc = nn.Linear(self.model.num_features, self.model.taskcla[t][1])
        model.to(self.device)

        optimizer, lr_scheduler = self._get_optimizer(bb_to_finetune, 0)
        for epoch in range(self.nepochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            for images, targets in trn_loader:
                targets -= self.model.task_offset[t]
                bsz = images.shape[0]
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    old_features = old_model(images)  # resnet with fc as identity returns features by default
                out, features = model(images, return_features=True)
                loss = self.criterion(t, out, targets, features, old_features)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                train_loss.append(float(bsz * loss))

            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    targets -= self.model.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device), targets.to(self.device)
                    with torch.no_grad():
                        old_features = old_model(images)  # resnet with fc as identity returns features by default
                    out, features = model(images, return_features=True)
                    loss = self.criterion(t, out, targets, features, old_features)

                    val_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            train_acc = train_hits / len(trn_loader.dataset)
            val_acc = val_hits / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                  f"Train acc: {100 * train_acc:.2f} Val acc: {100 * val_acc:.2f}")

        model.fc = nn.Identity()
        self.model.bbs[bb_to_finetune] = model
        torch.save(self.model.state_dict(), "best_ft.pth")

    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader):
        """ Create distributions for task t"""
        eps = 1e-8
        self.model.eval()
        classes = self.model.taskcla[t][1]
        self.model.task_offset.append(self.model.task_offset[-1] + classes)
        transforms = Compose([tr for tr in val_loader.dataset.transform.transforms
                              if "Resize" in tr.__class__.__name__
                              or "CenterCrop" in tr.__class__.__name__
                              or "ToTensor" in tr.__class__.__name__
                              or "Normalize" in tr.__class__.__name__])
        for bb_num in range(min(self.max_experts, t+1)):
            model = self.model.bbs[bb_num]
            for c in range(classes):
                c = c + self.model.task_offset[t]
                train_indices = torch.tensor(trn_loader.dataset.labels) == c
                # Uncomment to add valid set to distributions
                # val_indices = torch.tensor(val_loader.dataset.labels) == c
                if isinstance(trn_loader.dataset.images, list):
                    train_images = list(compress(trn_loader.dataset.images, train_indices))
                    ds = ClassDirectoryDataset(train_images, transforms)
                    # val_images = list(compress(val_loader.dataset.images, val_indices))
                    # ds = ClassDirectoryDataset(train_images + val_images, transforms)
                else:
                    ds = trn_loader.dataset.images[train_indices]
                    # ds = np.concatenate((trn_loader.dataset.images[train_indices], val_loader.dataset.images[val_indices]), axis=0)
                    ds = ClassMemoryDataset(ds, transforms)
                loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=0, shuffle=False)
                from_ = 0
                class_features = torch.full((2 * len(ds), self.model.num_features), fill_value=-999999999.0, device=self.model.device)
                for images in loader:
                    bsz = images.shape[0]
                    images = images.to(self.device)
                    features = model(images)
                    class_features[from_: from_+bsz] = features
                    features = model(torch.flip(images, dims=(3,)))
                    class_features[from_+bsz: from_+2*bsz] = features
                    from_ += 2*bsz

                if self.remove_outliers:
                    median = torch.median(class_features, dim=0)[0]
                    dist = torch.cdist(class_features, median.unsqueeze(0), p=2).squeeze(1)
                    not_outliers = torch.topk(dist, int(0.99*class_features.shape[0]), largest=False, sorted=False)[1]
                    class_features = class_features[not_outliers]

                # Calculate distributions
                cov_type = "full" if self.use_multivariate else "diag"
                is_ok = False
                while not is_ok:
                    try:
                        gmm = GaussianMixture(self.gmms, class_features.shape[1], covariance_type=cov_type, eps=eps).to(self.device)
                        gmm.fit(class_features, delta=1e-3, n_iter=100)
                    except RuntimeError:
                        eps = 10 * eps
                        print(f"WARNING: Covariance matrix is singular. Increasing eps to: {eps:.7f} but this may hurt results")
                    else:
                        is_ok = True

                self.experts_distributions[bb_num].append(gmm)

    @torch.no_grad()
    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
        self.model.eval()
        for images, targets in val_loader:
            targets = targets.to(self.device)
            # Forward current model
            features = self.model(images.to(self.device))
            hits_taw, hits_tag = self.calculate_metrics(features, targets, t)
            # Log
            total_loss = 0
            total_acc_taw += hits_taw.sum().item()
            total_acc_tag += hits_tag.sum().item()
            total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets, features=None, old_features=None):
        """Returns the loss value"""
        ce_loss = nn.functional.cross_entropy(outputs, targets, label_smoothing=0.0)
        if old_features is not None:  # Knowledge distillation loss on features
            kd_loss = nn.functional.mse_loss(features, old_features)
            total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            return total_loss
        return ce_loss

    @torch.no_grad()
    def calculate_metrics(self, features, targets, t):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        # Task-Aware
        classes = self.model.task_offset[t+1] - self.model.task_offset[t]
        log_probs = torch.zeros((features.shape[0], classes), device=features.device)
        bb_num = t % self.max_experts
        from_ = 0 if t < self.max_experts else self.model.task_offset[t] - self.model.task_offset[bb_num]
        for c, class_gmm in enumerate(self.experts_distributions[bb_num][from_:from_ + classes]):
            log_probs[:, c] = class_gmm.score_samples(features[:, bb_num])
        class_id = torch.argmax(log_probs, dim=1) + self.model.task_offset[t]
        hits_taw = (class_id == targets).float()

        # Task-Agnostic
        pred = self.predict_class_bayes(features)
        hits_tag = (pred == targets).float()
        return hits_taw, hits_tag

    @torch.no_grad()
    def predict_class_bayes(self, features):
        log_probs = torch.full((features.shape[0], len(self.experts_distributions), len(self.experts_distributions[0])), fill_value=-1e12, device=features.device)
        mask = torch.full_like(log_probs, fill_value=False, dtype=torch.bool)
        for bb_num, _ in enumerate(self.experts_distributions):
            for c, class_gmm in enumerate(self.experts_distributions[bb_num]):
                c += self.model.task_offset[bb_num]
                log_probs[:, bb_num, c] = class_gmm.score_samples(features[:, bb_num])
                mask[:, bb_num, c] = True

        if self.use_z_score:
            for i in range(len(self.experts_distributions)):
                mean = torch.mean(log_probs[:, i][mask[:, i]].reshape(mask.shape[0], -1), dim=1)
                std = torch.std(log_probs[:, i][mask[:, i]].reshape(mask.shape[0], -1), dim=1)
                log_probs[:, i] = (log_probs[:, i] - mean.unsqueeze(1)) / std.unsqueeze(1)
            log_probs[~mask] = -1e12
            log_probs = torch.softmax(10*log_probs, dim=2)
        else:
            if len(self.experts_distributions) > 1:
                log_probs = torch.nn.functional.gumbel_softmax(log_probs, dim=2, tau=self.tau)

        confidences = torch.sum(log_probs, dim=1) / torch.sum(mask, dim=1)
        class_id = torch.argmax(confidences, dim=1)
        return class_id

    def _get_optimizer(self, num, wd):
        """Returns the optimizer"""
        optimizer = torch.optim.SGD(self.model.bbs[num].parameters(), lr=self.lr, weight_decay=wd, momentum=self.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[60, 120, 160], gamma=0.1)
        return optimizer, scheduler
