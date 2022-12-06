import pickle
import random
import time

import numpy as np
import torch

from argparse import ArgumentParser

from torch import nn
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
from .incremental_learning import Inc_Learning_Appr


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
                 logger=None, use_multivariate=False, load_distributions=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)
        self.use_multivariate = use_multivariate
        self.load_distributions = load_distributions
        if load_distributions:
            with open(f"distributions.pickle", 'rb') as f:
                data_file = pickle.load(f)
                self.model.task_distributions = data_file["distributions"]
                self.model.tasks_learned_so_far = len(self.model.task_distributions)
                self.dream()

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--use-multivariate',
                            help='Use multivariate distribution',
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

        # Loop epochs

        for e in range(self.nepochs):
            self.train_epoch(t, trn_loader, val_loader)

        # if not self.load_distributions:
        #     with open(f"distributions.pickle", 'wb') as f:
        #         pickle.dump({"distributions": self.model.task_distributions}, f)

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

    def train_epoch(self, t, trn_loader, val_loader):
        """Runs a single epoch"""
        if self.load_distributions:
            return
        self.model.eval()
        if not self.model.means[t].detach().all():
            self.train_first_epoch(t, trn_loader, val_loader)
            return

    def train_first_epoch(self, t, trn_loader, val_loader):
        """ In the first epoch of a task t, calculate means and stds of selector outputs"""
        selectors_output = torch.full((2*len(trn_loader.dataset) + 2*len(val_loader.dataset), self.model.selector_features_dim), fill_value=-999999999.0,
                                      device=self.model.device)

        for i, (images, targets) in enumerate(trn_loader):
            bsz = images.shape[0]
            images = images.to(self.device)
            features = self.model(images)
            from_ = 2*i*trn_loader.batch_size
            selectors_output[from_: from_+bsz] = features
            features = self.model(torch.flip(images, dims=(3,)))
            selectors_output[from_+bsz: from_+2*bsz] = features

        for i, (images, targets) in enumerate(val_loader):
            bsz = images.shape[0]
            images = images.to(self.device)
            features = self.model(images)
            from_ = 2*len(trn_loader.dataset) + 2*i*val_loader.batch_size
            selectors_output[from_: from_+bsz] = features
            features = self.model(torch.flip(images, dims=(3,)))
            selectors_output[from_+bsz: from_+2*bsz] = features

        # Remove outliers
        # median = torch.median(selectors_output, dim=0)[0]
        # dist = torch.cdist(selectors_output, median.unsqueeze(0), p=2).squeeze(1)
        # not_outliers = torch.topk(dist, int(0.99*selectors_output.shape[0]), largest=False, sorted=False)[1]
        # selectors_output = selectors_output[not_outliers]

        # Calculate distribution
        self.model.means[t] = selectors_output.mean(dim=0)
        if self.use_multivariate:
            self.model.covs[t] = torch.cov(selectors_output.T)
        else:
            self.model.covs[t] = torch.diag(torch.std(selectors_output, dim=0))

        # self.model.covs[t] += torch.diag(torch.full((self.model.selector_features_dim,), fill_value=1e-4, device=self.model.device))
        self.model.task_distributions.append(MultivariateNormal(self.model.means[t], self.model.covs[t]))
        self.model.tasks_learned_so_far = t+1

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
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

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
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)


