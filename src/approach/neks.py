import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from PIL import Image
from torch.distributions import Normal, MultivariateNormal
from torch.utils.data import DataLoader
from .incremental_learning import Inc_Learning_Appr
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).unsqueeze_(0).unsqueeze_(0)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).unsqueeze_(0).unsqueeze_(0)


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, use_multivariate=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)
        self.use_multivariate = use_multivariate

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--use-multivariate',
                            help='Use multivariate distribution',
                            action='store_true',
                            default=False)
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset,
                                                 batch_size=trn_loader.batch_size,
                                                 shuffle=True,
                                                 num_workers=trn_loader.num_workers,
                                                 pin_memory=trn_loader.pin_memory)

        super().train_loop(t, trn_loader, val_loader)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.eval()
        if not self.model.means[t].detach().all():
            self.train_first_epoch(t, trn_loader)
            return

    def train_first_epoch(self, t, trn_loader):
        """ In the first epoch of a task t, calculate means and stds of selector outputs"""
        selectors_output = torch.full((len(trn_loader.dataset), self.model.selector_features_dim), fill_value=-999999999.0,
                                      device=self.model.device)
        self.model.heads[-1].train()
        for i, (images, targets) in enumerate(trn_loader):
            # Forward current model
            bsz = images.shape[0]
            # image = images[0].permute(1, 2, 0)
            # image *= IMAGENET_STD
            # image += IMAGENET_MEAN
            # image *= 255
            # image = Image.fromarray(np.array(image, dtype=np.uint8))
            # image.save("lol.png")
            features = self.model(images.to(self.device))
            # loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward

            # self.optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self._train_parameters(), self.clipgrad)
            # self.optimizer.step()

            from_ = i*trn_loader.batch_size
            selectors_output[from_: from_+bsz] = features

        # selectors_output = nn.functional.normalize(x, p=2, dim=1)

        self.model.means[t] = selectors_output.mean(dim=0)
        # if t == 0:
        #     self.model.means[t] += 10000
        if self.use_multivariate:
            self.model.covs[t] = torch.cov(selectors_output.T)
            self.model.covs[t] += torch.diag(torch.full((self.model.selector_features_dim,), fill_value=1e-1, device=self.model.device))
        else:
            self.model.covs[t] = torch.diag(torch.std(selectors_output, dim=0))
            self.model.covs[t] += torch.diag(torch.full((self.model.selector_features_dim,), fill_value=1e-3, device=self.model.device))

        self.model.task_distributions.append(MultivariateNormal(self.model.means[t], self.model.covs[t]))
        self.model.tasks_learned_so_far = t+1

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)

    def _train_parameters(self):
        """Includes the necessary weights to the optimizer"""
        return self.model.heads[-1].parameters()

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
            this_task = self.model.predict_task(features[m:m+1])
            pred[m] = this_task
            targets[m] = t
        hits_tag = (pred == targets).float()
        return hits_taw, hits_tag

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
