import copy
import random

import numpy as np
import torch

from argparse import ArgumentParser
from torch import nn

from .incremental_learning import Inc_Learning_Appr

torch.backends.cuda.matmul.allow_tf32 = False


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        planes = 32
        self.layers = nn.Sequential(nn.Conv2d(3, planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.Conv2d(planes, 2*planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.Conv2d(2*planes, 4*planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.MaxPool2d((2, 2)),
                                    nn.Conv2d(4*planes, 8*planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.Conv2d(8*planes, 4*planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.MaxPool2d((2, 2)),
                                    nn.Conv2d(4*planes, 2*planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.Conv2d(2*planes, planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.Conv2d(planes, 8, kernel_size=3, stride=1, padding=1)
                                    )# 8x8x8

    def forward(self, x):
        x = self.layers(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        planes = 32
        self.layers = nn.Sequential(nn.Conv2d(8, planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.Conv2d(planes, 2*planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.Conv2d(2*planes, 4*planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.Conv2d(4*planes, 8*planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.Conv2d(8*planes, 4*planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.Conv2d(4*planes, 2*planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.Conv2d(2*planes, planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                    nn.Conv2d(planes, 3, kernel_size=3, stride=1, padding=1)
                                    )

    def forward(self, x):
        x = self.layers(x)
        return x


class VisualCortex(nn.Module):
    """https://github.com/Puayny/Autoencoder-image-similarity/blob/master/Autoencoder%2C%20cifar-100%20dataset.ipynb"""
    def __init__(self, z_size, planes=(3, 64, 128, 256, 512)):
        super().__init__()
        self.z_size = z_size
        self.planes = planes

        self.linear1 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[1])
        self.encoder = EncoderBlock()
        self.decoder = DecoderBlock()

    def forward(self, x, decode=True):
        x = self.encoder(x)
        z = x.reshape(x.shape[0], -1)
        if not decode:
            return z, None
        x = z.reshape(x.shape[0], 8, 8, 8)
        x = self.decoder(x)
        return z, x

    def visualize(self, out, target):
        from PIL import Image
        out, target = out[0].cpu(), target[0].cpu()
        out = out.permute(1, 2, 0)
        target = target.permute(1, 2, 0)
        mean = torch.tensor([0.5071, 0.4866, 0.4409]).unsqueeze(0).unsqueeze(0)
        std = torch.tensor([0.2009, 0.1984, 0.2023]).unsqueeze(0).unsqueeze(0)
        out = torch.clip(255 * (out * std + mean), min=0, max=255)
        target = torch.clip(255 * (target * std + mean), min=0, max=255)
        out = Image.fromarray(np.array(out, dtype=np.uint8))
        target = Image.fromarray(np.array(target, dtype=np.uint8))
        out.save("a_out.png")
        target.save("a_gt.png")
        # target.save("a_gt.png")


class MLP(nn.Module):
    def __init__(self, z_size, hidden_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(z_size, 2 * hidden_size)
        self.linear2 = nn.Linear(2 * hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        return self.out(x)


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, tau=3.0):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)
        self.task_offset = [0]
        self.model = None
        self.tau = tau

        self.cortex = VisualCortex(512)
        self.cortex.to(device)
        self.classifier = MLP(512, 1024, 10)
        self.classifier.to(device)
        self.criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--tau',
                            help='gumbel softmax temperature',
                            type=float,
                            default=3.0)

        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        if t == 0:
            print(f"Training visual cortex")
            self.train_cortex(t, trn_loader, val_loader)
            # state_dict = torch.load("cortex_best.pth")
            # self.cortex.load_state_dict(state_dict, strict=True)
        self.train_classifier(t, trn_loader, val_loader)

    def train_cortex(self, t, trn_loader, val_loader):
        model = self.cortex
        print(f'Cortex has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        model.to(self.device)
        optimizer, lr_scheduler = self._get_optimizer(model, self.wd, milestones=[50, 80, 150, 190])
        for epoch in range(self.nepochs):
            train_loss, valid_loss = [], []
            model.train()
            for images, _ in trn_loader:
                bsz = images.shape[0]
                images = images.to(self.device)
                optimizer.zero_grad()
                z, reconstructed = model(images)
                loss = nn.functional.mse_loss(reconstructed, images)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for images, _ in val_loader:
                    bsz = images.shape[0]
                    images = images.to(self.device)
                    z, reconstructed = model(images)
                    loss = nn.functional.mse_loss(reconstructed, images)
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f}")
        self.cortex = model
        torch.save(self.cortex.state_dict(), f"cortex.pth")

    def train_classifier(self, t, trn_loader, val_loader):
        model = self.classifier
        print(f'Classifier has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

        model.to(self.device)
        optimizer, lr_scheduler = self._get_optimizer(model, self.wd, milestones=[40, 80])
        for epoch in range(self.nepochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for images, targets in trn_loader:
                targets -= self.task_offset[t]
                bsz = images.shape[0]
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    embeddings, _ = self.cortex(images, decode=True)
                    # self.cortex.visualize(_, images)
                out = model(embeddings)
                loss = self.criterion(out, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    targets -= self.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device), targets.to(self.device)
                    with torch.no_grad():
                        embeddings, _ = self.cortex(images, decode=False)
                    out = model(embeddings)
                    loss = self.criterion(out, targets)

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
        torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model.pth")

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

    @torch.no_grad()
    def calculate_metrics(self, features, targets, t):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        taw_pred, tag_pred = self.predict_class_bayes(t, features)
        hits_taw = (taw_pred == targets).float()
        hits_tag = (tag_pred == targets).float()
        return hits_taw, hits_tag

    def _get_optimizer(self, model, wd, milestones=[60, 120, 160]):
        """Returns the optimizer"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler
