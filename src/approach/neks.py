import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from argparse import ArgumentParser

from torch.distributions import Normal, MultivariateNormal
from torch.utils.data import DataLoader
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, freeze_after=-1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)

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
        if not self.model.means[t].all():
            self.train_first_epoch(t, trn_loader)
            return

        self.model.heads[-1].train()
        for i, (images, targets) in enumerate(trn_loader):
            # Forward current model
            outputs, _ = self.model(images.to(self.device))

            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._train_parameters(), self.clipgrad)
            self.optimizer.step()

    def train_first_epoch(self, t, trn_loader):
        """ In the first epoch of a task t, calculate means and stds of selector outputs"""
        selectors_output = torch.full((len(trn_loader.dataset), self.model.selector_features_dim), fill_value=-999999999.0,
                                      device=self.model.device)
        self.model.heads[-1].train()
        for i, (images, targets) in enumerate(trn_loader):
            # Forward current model
            bsz = images.shape[0]
            outputs, features = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._train_parameters(), self.clipgrad)
            self.optimizer.step()

            from_ = i*trn_loader.batch_size
            selectors_output[from_: from_+bsz] = self.model.forward_selector(features)

        self.model.means[t] = selectors_output.mean(dim=0)
        self.model.covs[t] = torch.cov(selectors_output.T)
        self.model.covs[t] += torch.diag(torch.full((self.model.selector_features_dim,), fill_value=1e-6, device=self.model.device))
        self.model.tasks_learned_so_far = t+1
        self.model.task_distributions.append(MultivariateNormal(self.model.means[t], self.model.covs[t]))

        # task_id = -1
        # plt.xlim(-2, 2)
        # for task_id in range(t+1):
        #     plt.scatter(np.array(self.model.means[t]), [i for i in range(self.model.selector_features_dim)], marker="_", s=1000*np.array(self.model.stds[t]))
        # plt.savefig(f"mean_{t}_{task_id}.png")
        # plt.clf()
        # task_id = -1
        # plt.xlim(0.4, 1.1)
        # for task_id in range(t+1):
        #     plt.scatter(np.array(self.model.stds[t]), [i for i in range(self.model.selector_features_dim)], marker="_")
        # plt.savefig(f"std_{t}_{task_id}.png")
        # plt.clf()

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
                outputs, features = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets)
                hits_taw, hits_tag = self.calculate_metrics(outputs, features, targets, t)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def calculate_metrics(self, outputs, features, targets, t):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets)

        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = t
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets).float()

        # WARNING: THIS CALCULATES ACCURACY OF SELECTORS, NOT ACC OF NEKS TASK AGNOSTIC, RESEARCH PURPOSE
        for m in range(len(pred)):
            this_task = self.model.predict_task(features[m:m+1])
            pred[m] = this_task
            targets[m] = t
        hits_tag = (pred == targets).float()
        return hits_taw, hits_tag
