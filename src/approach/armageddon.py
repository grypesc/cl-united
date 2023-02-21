import copy
import random
from itertools import compress

import numpy as np
import torch

from argparse import ArgumentParser
from torch import nn

from .incremental_learning import Inc_Learning_Appr
from src.networks.resnet32 import resnet32
from .mvgb import ClassDirectoryDataset, ClassMemoryDataset

torch.backends.cuda.matmul.allow_tf32 = False


class MembeddingDataset(torch.utils.data.Dataset):
    def __init__(self, membeddings_per_class: int):
        self.labels = torch.zeros((0,), dtype=torch.int64)
        self.data = torch.zeros((0, 512), dtype=torch.float)
        self.membeddings_per_class = membeddings_per_class

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def add(self, label, new_data):
        new_labels = label.expand(new_data.shape[0])
        self.labels = torch.cat((self.labels, new_labels), dim=0)
        self.data = torch.cat((self.data, new_data), dim=0)


class SlowLearner(nn.Module):

    class EncoderBlock(nn.Module):
        def __init__(self):
            super().__init__()
            planes = 32
            self.layers = nn.Sequential(nn.Conv2d(3, planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.Conv2d(planes, 2 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.Conv2d(2 * planes, 4 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.MaxPool2d((2, 2)),
                                        nn.Conv2d(4 * planes, 8 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.Conv2d(8 * planes, 4 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.MaxPool2d((2, 2)),
                                        nn.Conv2d(4 * planes, 2 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.Conv2d(2 * planes, planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.Conv2d(planes, 8, kernel_size=3, stride=1, padding=1)
                                        )

        def forward(self, x):
            x = self.layers(x)
            return x

    class DecoderBlock(nn.Module):
        def __init__(self):
            super().__init__()
            planes = 32
            self.layers1 = nn.Sequential(nn.Conv2d(8, planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.Conv2d(planes, 2 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2),
                                         nn.Conv2d(2 * planes, 4 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.Conv2d(4 * planes, 8 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2)
                                         )

            self.layers2 = nn.Sequential(nn.Conv2d(8 * planes, 4 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.Conv2d(4 * planes, 2 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.Conv2d(2 * planes, planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.Conv2d(planes, 3, kernel_size=3, stride=1, padding=1)
                                         )

        def forward(self, z):
            x = z.reshape(z.shape[0], 8, 8, 8)
            feature_maps = self.layers1(x)
            return feature_maps, self.layers2(feature_maps)

    def __init__(self, z_size, planes=(3, 64, 128, 256, 512)):
        super().__init__()
        self.z_size = z_size
        self.planes = planes

        self.linear1 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[1])
        self.encoder = SlowLearner.EncoderBlock()
        self.decoder = SlowLearner.DecoderBlock()

    def forward(self, x, decode=True):
        x = self.encoder(x)
        z = x.reshape(x.shape[0], -1)
        if not decode:
            return z
        feature_maps, x = self.decoder(x)
        return z, feature_maps, x

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
        self.linear1 = nn.Linear(z_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        return self.out(x)


class Appr(Inc_Learning_Appr):
    """https://www.youtube.com/watch?v=wfa9xH3cJ8E"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, membeddings=100):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)
        self.task_offset = [0]
        self.model = None
        self.membeddings_per_class = membeddings
        self.membeddings = MembeddingDataset(self.membeddings_per_class)

        self.slow_learner = SlowLearner(512)
        self.slow_learner.to(device)
        self.fast_learner = resnet32()
        self.criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--membeddings',
                            help='number of memory embeddings per class',
                            type=int,
                            default=100)

        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        old_slow_learner = copy.deepcopy(self.slow_learner)
        print(f"Training slow learner on task {t}")
        self.train_slow_learner(old_slow_learner, trn_loader, val_loader)
        # state_dict = torch.load("slow_learner_10.pth")
        # self.slow_learner.load_state_dict(state_dict, strict=True)
        self.store_membeddings(t, trn_loader, val_loader.dataset.transform, old_slow_learner)
        print(f"Training classifier")
        self.train_fast_learner(t, trn_loader, val_loader)

    def train_slow_learner(self, old_model, trn_loader, val_loader):
        old_model.eval()
        old_model.to(self.device)
        model = self.slow_learner
        model.to(self.device)
        print(f'Slow learner has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        epochs = self.nepochs
        milestones = [50, 100, 150]
        if len(self.membeddings) > 0:
            mem_loader = torch.utils.data.DataLoader(self.membeddings, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True)
            epochs = self.nepochs // 2
            milestones = [40, 60, 80]
        optimizer, lr_scheduler = self._get_optimizer(model, self.wd, milestones=milestones)
        for epoch in range(epochs):
            train_loss, valid_loss = [], []
            model.train()
            for images, _ in trn_loader:

                bsz = images.shape[0]
                images = images.to(self.device)
                optimizer.zero_grad()
                _, _, reconstructed = model(images)
                loss = nn.functional.mse_loss(reconstructed, images)

                if len(self.membeddings) > 0:
                    mem_loader_iter = iter(mem_loader)
                    membeddings = next(mem_loader_iter)[0].to(self.device)
                    with torch.no_grad():
                        mem_target = old_model.decoder(membeddings)[1]
                    _, _, mem_reconstructed = model(mem_target)
                    mem_loss = nn.functional.mse_loss(mem_reconstructed, mem_target)
                    loss = loss + mem_loss

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
                    z, _, reconstructed = model(images)
                    loss = nn.functional.mse_loss(reconstructed, images)

                    if len(self.membeddings) > 0:
                        mem_loader_iter = iter(mem_loader)
                        membeddings = next(mem_loader_iter)[0].to(self.device)
                        mem_target = old_model.decoder(membeddings)[1]
                        _, _, mem_reconstructed = model(mem_target)
                        mem_loss = nn.functional.mse_loss(mem_reconstructed, mem_target)
                        loss = loss + mem_loss

                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f}")
        self.slow_learner = model
        torch.save(self.slow_learner.state_dict(), f"slow_learner.pth")

    def train_fast_learner(self, t, trn_loader, val_loader):
        model = self.fast_learner
        model.fc = nn.Linear(64, self.task_offset[t+1])
        model.to(self.device)
        print(f'Classifier has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

        self.slow_learner.eval()
        mem_loader = torch.utils.data.DataLoader(self.membeddings, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True)
        optimizer, lr_scheduler = self._get_optimizer(model, self.wd, milestones=[50, 100, 150])
        for epoch in range(self.nepochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for membeddings, targets in mem_loader:
                bsz = membeddings.shape[0]
                membeddings, targets = membeddings.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    reconstructed = self.slow_learner.decoder(membeddings)[1]
                out = model(reconstructed)
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
                    bsz = images.shape[0]
                    images, targets = images.to(self.device), targets.to(self.device)
                    _, _, reconstructed = self.slow_learner(images, decode=True)
                    out = model(reconstructed)
                    loss = self.criterion(out, targets)
                    val_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(mem_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            train_acc = train_hits / len(mem_loader.dataset)
            val_acc = val_hits / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                  f"Train acc: {100 * train_acc:.2f} Val acc: {100 * val_acc:.2f}")
        self.fast_learner = model

    @torch.no_grad()
    def store_membeddings(self, t, trn_loader, transforms, old_slow_learner):
        old_slow_learner.eval()
        self.slow_learner.eval()

        # Update old membeddings
        if len(self.membeddings) > 0:
            mem_loader = torch.utils.data.DataLoader(self.membeddings, batch_size=trn_loader.batch_size, num_workers=0, shuffle=False)
            index = 0
            for old_membeddings, _ in mem_loader:
                bsz = old_membeddings.shape[0]
                old_membeddings = old_membeddings.to(self.device)
                _, reconstructed = old_slow_learner.decoder(old_membeddings)
                new_membeddings = self.slow_learner(reconstructed, decode=False)
                self.membeddings.data[index:index+bsz] = new_membeddings.cpu()
                index += bsz

        # Add new membeddings to memory
        labels = np.array(trn_loader.dataset.labels)
        classes_ = set(trn_loader.dataset.labels)
        self.task_offset += [len(classes_) + self.task_offset[t]]
        for i in classes_:
            class_indices = labels == i
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, class_indices))
                train_images = train_images[:self.membeddings_per_class]
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = trn_loader.dataset.images[class_indices][:self.membeddings_per_class]
                ds = ClassMemoryDataset(ds, transforms)
            loader = torch.utils.data.DataLoader(ds, batch_size=self.membeddings_per_class, num_workers=0, shuffle=True)
            for images in loader:
                images = images.to(self.device)
                membeddings = self.slow_learner(images, decode=False)
                membeddings = membeddings.cpu()
                self.membeddings.add(torch.tensor(i), membeddings)

    @torch.no_grad()
    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
        self.slow_learner.eval()
        self.fast_learner.eval()
        for images, targets in val_loader:
            targets = targets.to(self.device)
            # Forward current model
            _, _, reconstructed = self.slow_learner(images.to(self.device), decode=True)
            logits = self.fast_learner(reconstructed)
            preds = torch.argmax(logits, dim=1)
            hits_tag = preds == targets
            preds = torch.argmax(logits[:, self.task_offset[t]:self.task_offset[t+1]], dim=1) + self.task_offset[t]
            hits_taw = preds == targets
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler
