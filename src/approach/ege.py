import copy
import pickle
import random
import numpy as np
import torch

from argparse import ArgumentParser
from itertools import compress
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .mvgb import WarmUpScheduler, ClassMemoryDataset, ClassDirectoryDataset
from .gmm import GaussianMixture
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, gmms=1, use_multivariate=True, use_head=False, remove_outliers=False, load_distributions=False, save_distributions=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)
        self.gmms = gmms
        self.patience = patience
        self.use_multivariate = use_multivariate
        self.use_head = use_head
        self.remove_outliers = remove_outliers
        self.load_distributions = load_distributions
        self.save_distributions = save_distributions
        self.model.to(device)
        self.task_distributions = []

        if load_distributions:
            with open(f"distributions.pickle", 'rb') as f:
                data_file = pickle.load(f)
                self.task_distributions = data_file["distributions"]

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--gmms',
                            help='Number of gaussian models in the mixture',
                            type=int,
                            default=1)
        parser.add_argument('--patience',
                            help='Early stopping',
                            type=int,
                            default=5)
        parser.add_argument('--use-multivariate',
                            help='Use multivariate distribution',
                            action='store_true',
                            default=False)
        parser.add_argument('--use-head',
                            help='Use trainable head instead of Bayesian inference',
                            action='store_true',
                            default=False)
        parser.add_argument('--remove-outliers',
                            help='Remove class outliers before creating distribution',
                            action='store_true',
                            default=False)
        parser.add_argument('--load-distributions',
                            help='Load distributions from a pickle file',
                            action='store_true',
                            default=False)
        parser.add_argument('--save-distributions',
                            help='Save distributions to a pickle file',
                            action='store_true',
                            default=False)
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        # Train backbone
        print(f"Training backbone on task {t}:")
        self.train_backbone(t, trn_loader, val_loader)

        # Create distributions
        print(f"Creating distributions for task {t}:")
        self.task_distributions.append([])
        self.create_distributions(t, trn_loader, val_loader)

        # Dump distributions
        if self.save_distributions:
            with open(f"distributions.pickle", 'wb') as f:
                pickle.dump({"distributions": self.task_distributions}, f)

    def train_backbone(self, t, trn_loader, val_loader):
        if t == 0:
            model = self.model.bbs[0]
            for param in model.parameters():
                param.requires_grad = True
        else:
            self.model.bbs.append(copy.deepcopy(self.model.bbs[-1]))
            model = self.model.bbs[t]
            for name, param in model.named_parameters(): #TODO: LOLZ
                param.requires_grad = False
                if "layer2" in name or "layer3" in name or "layer4" in name:
                    param.requires_grad = True
                model.fc = nn.Linear(self.model.num_features, self.model.taskcla[t][1])

        model.to(self.device)
        optimizer, lr_scheduler = self._get_optimizer()
        best_loss, best_epoch, best_model = 1e8, 0, None
        for epoch in range(self.nepochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for images, targets in trn_loader:
                bsz = images.shape[0]
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                out = model(images)
                loss = self.criterion(t, out, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                lr_scheduler.step_iter()
                train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))

                train_loss.append(float(bsz * loss))
            lr_scheduler.step_epoch()

            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
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

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_model = copy.deepcopy(model)

            if epoch - best_epoch >= self.patience:
                break

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                  f"Train acc: {100 * train_acc:.2f} Val acc: {100 * val_acc:.2f}")

        print(f"Best epoch: {epoch}")
        best_model.fc = nn.Identity()
        self.model.bbs[t] = best_model
        torch.save(self.model.state_dict(), "best.pth")

    def create_distributions(self, t, trn_loader, val_loader):
        """ Create distributions for task t"""
        self.model.eval()
        with torch.no_grad():
            classes = self.model.taskcla[t][1]
            self.model.task_offset.append(self.model.task_offset[-1] + classes)
            transforms = Compose([t for t in val_loader.dataset.transform.transforms
                                  if "CenterCrop" in t.__class__.__name__
                                  or "ToTensor" in t.__class__.__name__
                                  or "Normalize" in t.__class__.__name__])
            for task_num, model in enumerate(self.model.bbs):
                for c in range(classes):
                    c = c + self.model.task_offset[t]
                    train_indices = torch.tensor(trn_loader.dataset.labels) == c
                    val_indices = torch.tensor(val_loader.dataset.labels) == c
                    if isinstance(trn_loader.dataset.images, list):
                        train_images = list(compress(trn_loader.dataset.images, train_indices))
                        val_images = list(compress(val_loader.dataset.images, val_indices))
                        ds = ClassDirectoryDataset(train_images + val_images, transforms)
                    else:
                        ds = np.concatenate((trn_loader.dataset.images[train_indices], val_loader.dataset.images[val_indices]), axis=0)
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
                    gmm = GaussianMixture(self.gmms, class_features.shape[1], covariance_type=cov_type, eps=1e-8).to(self.device)
                    gmm.fit(class_features, delta=1e-3, n_iter=100)
                    self.task_distributions[task_num].append(gmm)

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
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

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return nn.functional.cross_entropy(outputs, targets, label_smoothing=0.0)

    def calculate_metrics(self, features, targets, t):
        """Contains the main Task-Aware and Task-Agnostic metrics"""

        # Task-Aware Multi-Head
        # for m in range(len(pred)):
        #     this_task = t
        #     pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (targets == targets).float()

        pred = self.predict_class(features)
        hits_tag = (pred == targets).float()
        return hits_taw, hits_tag

    def predict_class(self, features):
        return self.predict_class_bayes(features)

    def predict_class_bayes(self, features):
        with torch.no_grad():
            confidences = torch.zeros((features.shape[0], len(self.task_distributions), len(self.task_distributions[0])), device=features.device)
            mask = torch.full_like(confidences, fill_value=False)
            for t, _ in enumerate(self.task_distributions):
                for c, class_gmm in enumerate(self.task_distributions[t]):
                    confidences[:, t, c] = class_gmm.score_samples(features)
                    mask[:, t, c] = True

            log_probs = torch.sum(confidences, dim=1) / torch.sum(mask, dim=1)
            class_id = torch.argmax(log_probs, dim=1)
        return class_id

    def _get_optimizer(self):
        """Returns the optimizer"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = WarmUpScheduler(optimizer, 100, 0.96)
        return optimizer, scheduler

