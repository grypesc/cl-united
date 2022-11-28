import time

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
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader, val_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        # if the lr decreases below minimum, stop the training session
                        print()
                        break
                    # reset patience and recover best model so far to continue training
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
        self.model.set_state_dict(best_model)

    def train_epoch(self, t, trn_loader, val_loader):
        """Runs a single epoch"""
        self.model.eval()
        if not self.model.means[t].detach().all():
            self.train_first_epoch(t, trn_loader, val_loader)
            return

    def train_first_epoch(self, t, trn_loader, val_loader):
        """ In the first epoch of a task t, calculate means and stds of selector outputs"""
        selectors_output = torch.full((len(trn_loader.dataset) + len(val_loader.dataset), self.model.selector_features_dim), fill_value=-999999999.0,
                                      device=self.model.device)

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

        for i, (images, targets) in enumerate(val_loader):
            bsz = images.shape[0]
            features = self.model(images.to(self.device))
            from_ = len(trn_loader.dataset) + i*val_loader.batch_size
            selectors_output[from_: from_+bsz] = features

        # Remove outliers
        median = torch.median(selectors_output, dim=0)[0]
        dist = torch.cdist(selectors_output, median.unsqueeze(0), p=2).squeeze(1)
        not_outliers = torch.topk(dist, int(0.99*selectors_output.shape[0]), largest=False, sorted=False)[1]
        selectors_output = selectors_output[not_outliers]

        # Calculate distribution
        self.model.means[t] = selectors_output.mean(dim=0)
        if self.use_multivariate:
            self.model.covs[t] = torch.cov(selectors_output.T)
        else:
            self.model.covs[t] = torch.diag(torch.std(selectors_output, dim=0))

        self.model.covs[t] += torch.diag(torch.full((self.model.selector_features_dim,), fill_value=1e-4, device=self.model.device))
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
