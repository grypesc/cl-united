import pickle
import random
import time

import numpy as np
import torch

from argparse import ArgumentParser

from torch import nn
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from .incremental_learning import Inc_Learning_Appr


class ClassDataset(torch.utils.data.Dataset):
    """ Dataset consisting of samples of the same class """
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = self.transforms(self.images[index].copy())
        return image


class DreamingDataset(torch.utils.data.Dataset):
    """ Dataset that samples from learned distributions to train head """
    def __init__(self, distributions, samples):
        self.distributions = distributions
        self.samples = samples

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        target = random.randint(0, len(self.distributions)-1)
        val = self.distributions[target].sample(torch.Size([]))
        return val, target


class WarmUpScheduler(nn.Module):
    """Warm-up and exponential decay chain scheduler. If warm_up_iters > 0 than warm-ups linearly for warm_up_iters iterations.
    Then it decays the learning rate every epoch. It is a good idea to set warm_up_iters as total number of samples in epoch / batch size"""

    def __init__(self, optimizer, warm_up_iters=0, lr_decay=0.97):
        super().__init__()
        self.total_steps, self.warm_up_iters = 0, warm_up_iters
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-6, total_iters=warm_up_iters) if warm_up_iters else None
        self.decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay, last_epoch=-1)

    def step_iter(self):
        self.total_steps += 1
        if self.warmup_scheduler:
            self.warmup_scheduler.step()

    def step_epoch(self):
        if self.total_steps > self.warm_up_iters:
            self.decay_scheduler.step()


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, use_multivariate=False, remove_outliers=False, load_distributions=False, save_distributions=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)
        self.use_multivariate = use_multivariate
        self.remove_outliers = remove_outliers
        self.load_distributions = load_distributions
        self.save_distributions = save_distributions

        if load_distributions:
            with open(f"distributions.pickle", 'rb') as f:
                data_file = pickle.load(f)
                self.model.task_distributions = data_file["distributions"]

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--use-multivariate',
                            help='Use multivariate distribution',
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
        """Contains the epochs loop"""
        if self.load_distributions:
            return

        # Train backbone
        if t == 0:
            self.train_backbone(t, trn_loader, val_loader)

        # Create distributions
        self.create_distributions(t, trn_loader, val_loader)

        # Dump distributions
        if self.save_distributions:
            with open(f"distributions.pickle", 'wb') as f:
                pickle.dump({"distributions": self.model.task_distributions}, f)

    def dream(self):
        print("STARTING DREAMING")
        optimizer = torch.optim.Adam(self.model.model.dreamer.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90, last_epoch=-1)
        self.model.model.to(self.device)
        self.model.model.dreamer.train()
        ds = DreamingDataset(self.model.task_distributions, 30000)
        loader = DataLoader(ds, batch_size=64)
        for epoch in range(50):
            losses, hits = [], []
            for input, target in loader:
                input, target = input.to(self.device), target.to(self.device)
                bsz = input.shape[0]
                out = self.model.model.dreamer(input)
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(out, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.model.dreamer.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss * bsz))
                TP = torch.sum(torch.argmax(out, dim=1) == target)
                hits.append(int(TP))
            scheduler.step()
            print(f"Epoch: {epoch}")
            print(sum(losses) / len(ds), sum(hits) / len(ds))

        self.model.model.dreamer.eval()

    def train_backbone(self, t, trn_loader, val_loader):
        return
        model = self.model
        optimizer, lr_scheduler = self._get_optimizer()
        model.train()
        for epoch in range(self.nepochs):
            train_loss, valid_loss = [], []
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

                train_loss.append(float(bsz * loss))
            lr_scheduler.step_epoch()

            with torch.no_grad():
                for images, targets in val_loader:
                    bsz = images.shape[0]
                    images, targets = images.to(self.device), targets.to(self.device)
                    out = model(images)
                    loss = self.criterion(t, out, targets)

                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(train_loss)
            val_loss = sum(valid_loss) / len(valid_loss)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2} Val loss: {valid_loss:.2}")

    def create_distributions(self, t, trn_loader, val_loader):
        """ Create distributions for task t"""
        print("Creating distributions:")
        self.model.eval()
        with torch.no_grad():
            classes = self.model.taskcla[t][1]
            self.model.task_offset.append(self.model.task_offset[-1] + classes)
            transforms = Compose([t for t in val_loader.dataset.transform.transforms
                                  if "ToTensor" in t.__class__.__name__
                                  or "Normalize" in t.__class__.__name__])
            for c in range(classes):
                c = c + self.model.task_offset[t]
                train_indices = torch.tensor(trn_loader.dataset.labels) == c
                val_indices = torch.tensor(val_loader.dataset.labels) == c
                ds = np.concatenate((trn_loader.dataset.images[train_indices], val_loader.dataset.images[val_indices]), axis=0)
                ds = ClassDataset(ds, transforms)
                loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=0, shuffle=False)
                from_ = 0
                class_features = torch.full((2 * len(ds), 64), fill_value=-999999999.0, device=self.model.device)
                for images in loader:
                    bsz = images.shape[0]
                    images = images.to(self.device)
                    _, features = self.model(images, return_features=True)
                    class_features[from_: from_+bsz] = features
                    _, features = self.model(torch.flip(images, dims=(3,)), return_features=True)
                    class_features[from_+bsz: from_+2*bsz] = features
                    from_ += 2*bsz

                if self.remove_outliers:
                    median = torch.median(class_features, dim=0)[0]
                    dist = torch.cdist(class_features, median.unsqueeze(0), p=2).squeeze(1)
                    not_outliers = torch.topk(dist, int(0.99*class_features.shape[0]), largest=False, sorted=False)[1]
                    class_features = class_features[not_outliers]

                # Calculate distribution
                means = class_features.mean(dim=0)
                if self.use_multivariate:
                    covs = torch.cov(class_features.T)
                else:
                    covs = torch.diag(torch.std(class_features, dim=0))

                # covs += torch.diag(torch.full((covs.shape[0],), fill_value=10, device=covs.device))
                self.model.task_distributions.append(MultivariateNormal(means, covs))

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                targets = targets.to(self.device)
                # Forward current model
                _, features = self.model(images.to(self.device), return_features=True)
                hits_taw, hits_tag = self.calculate_metrics(features, targets, t)
                # Log
                total_loss = 0
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return nn.functional.cross_entropy(outputs, targets, label_smoothing=0.1)

    def calculate_metrics(self, features, targets, t):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets)

        # Task-Aware Multi-Head
        # for m in range(len(pred)):
        #     this_task = t
        #     pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (targets == targets).float()

        # WARNING: THIS CALCULATES ACCURACY OF SELECTORS, NOT ACC OF NEKS TASK AGNOSTIC, RESEARCH PURPOSE
        for m in range(len(pred)):
            this_task = self.model.predict_task_bayes(features[m:m+1])
            pred[m] = this_task
            targets[m] = t
        hits_tag = (pred == targets).float()
        return hits_taw, hits_tag

    def _get_optimizer(self):
        """Returns the optimizer"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = WarmUpScheduler(optimizer, 100, 0.95)
        return optimizer, scheduler

