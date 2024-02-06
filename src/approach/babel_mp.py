import copy
import os
import random
import torch
import numpy as np
import multiprocessing as mp

from argparse import ArgumentParser
from itertools import compress

from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchmetrics import Accuracy

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .models.resnet32 import resnet8, resnet14, resnet20, resnet32
from .incremental_learning import Inc_Learning_Appr
from .criterions.ce import CE

class BabelMemoryDataset(torch.utils.data.Dataset):
    """ Dataset consisting of samples of only one class """
    def __init__(self, images, targets, transforms):
        self.images = images
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        image = self.transforms(image)
        return image, self.targets[index]

class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=1,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, N=5, strategy="constant", num_processes=1, use_negative=False, K=3, S=64, distiller="linear", alpha=0.5, smoothing=0., sval_fraction=0.95, activation_function="relu", nnet="resnet32"):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)

        self.N = N
        self.K = K
        self.S = S
        self.num_processes = num_processes
        if self.N % num_processes != 0:
            raise RuntimeError("N must be divisible by num processes")
        self.alpha = alpha
        self.smoothing = smoothing
        self.patience = patience
        self.use_negative_class = use_negative
        self.alpha_strategy = strategy
        self.old_model = None
        self.model = None
        mp.set_start_method('spawn')
        model_dict = {"resnet8": resnet8(num_features=S, activation_function=activation_function),
                      "resnet14": resnet14(num_features=S, activation_function=activation_function),
                      "resnet20": resnet20(num_features=S, activation_function=activation_function),
                      "resnet32": resnet32(num_features=S, activation_function=activation_function)}
        self.models = torch.nn.ModuleList()
        for _ in range(self.N):
            model = model_dict[nnet]
            model.fc = nn.Identity()
            model.to(device, non_blocking=True)
            self.models.append(model)
        self.train_data_loaders, self.val_data_loaders = [], []
        self.prototypes = torch.empty((self.N, 0, self.S), device=self.device)
        self.task_offset = [0]
        self.classes_in_tasks = []
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
                            default=3)
        parser.add_argument('--K',
                            help='number of learners sampled for task',
                            type=int,
                            default=3)
        parser.add_argument('--S',
                            help='latent space size',
                            type=int,
                            default=64)
        parser.add_argument('--alpha',
                            help='relative weight of kd loss',
                            type=float,
                            default=0.5)
        parser.add_argument('--sval-fraction',
                            help='Fraction of eigenvalues sum that is explained',
                            type=float,
                            default=0.95)
        parser.add_argument('--activation-function',
                            help='Activation functions in resnet',
                            type=str,
                            choices=["identity", "relu", "lrelu"],
                            default="relu")
        parser.add_argument('--use-negative',
                            help='Adapt prototypes',
                            action='store_true',
                            default=False)
        parser.add_argument('--distiller',
                            help='Distiller',
                            type=str,
                            choices=["linear", "mlp"],
                            default="linear")
        parser.add_argument('--strategy',
                            help='alpha strategy',
                            type=str,
                            choices=["constant", "random", "linspace"],
                            default="constant")
        parser.add_argument('--num-processes',
                            help='Number of processes',
                            type=int,
                            default=1)
        parser.add_argument('--smoothing',
                            help='label smoothing',
                            type=float,
                            default=0.0)
        parser.add_argument('--nnet',
                            type=str,
                            choices=["resnet8", "resnet14", "resnet20", "resnet32"],
                            default="resnet32")
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        num_classes_in_t = len(np.unique(trn_loader.dataset.labels))
        self.task_offset.append(num_classes_in_t + self.task_offset[-1])
        self.classes_in_tasks.append(num_classes_in_t)
        self.train_data_loaders.extend([trn_loader])
        self.val_data_loaders.extend([val_loader])

        # if t<9:
        #     return
        #
        # self.models = torch.load('../model_9.pth')
        # self.prototypes = torch.load('../proto_9.pt')
        # return

        alphas = np.full((self.N,), self.alpha)
        if self.alpha_strategy == "linspace":
            alphas = np.linspace(0.1, 1, self.N)
        elif self.alpha_strategy == "random":
            alphas = np.rand(self.N)
        results = []
        # train_child(copy.deepcopy(self.models[0]), copy.deepcopy(self.prototypes[0]), self.K, trn_loader, val_loader, t, num_classes_in_t, self.wd, self.nepochs, self.task_offset, self.alpha, self.use_negative_class, self.device)
        for i in range(self.N // self.num_processes):
            print(f"Spawned {self.num_processes} processes")
            offset = self.num_processes * i
            with mp.Pool(self.num_processes) as pool:
                multiple_results = [pool.apply_async(train_child, args=(copy.deepcopy(self.models[offset + j]), copy.deepcopy(self.prototypes[offset + j]), self.K, trn_loader, val_loader, t, num_classes_in_t, self.wd, self.nepochs, self.task_offset, alphas[offset + j], self.use_negative_class, self.device))
                                    for j in range(self.num_processes)]
                results.extend([res.get() for res in multiple_results])
            print(f"Joined {self.num_processes} processes")
        new_protos = []
        for i, (model, protos) in enumerate(results):
            self.models[i] = model
            new_protos.append(protos)
        self.prototypes = torch.stack(new_protos)
        torch.save(self.models, self.logger.exp_path + f"/models/model_{t}.pth")
        torch.save(self.prototypes, self.logger.exp_path + f"/models/proto_{t}.pt")

    @torch.no_grad()
    def eval(self, t, val_loader):
        """ Perform nearest centroids classification """
        for m in self.models:
            m.eval()
        tag_acc = Accuracy("multiclass", num_classes=self.prototypes.shape[1])
        taw_acc = Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t]
        for images, target in val_loader:
            images = images.to(self.device, non_blocking=True)
            features = [m(images) for m in self.models]
            dist = []
            for i in range(self.N):
                d = torch.cdist(features[i], self.prototypes[i, :, :])
                dist.append(d)
            dist = torch.stack(dist, dim=2)
            dist = torch.mean(dist, dim=2)
            tag_preds = torch.argmin(dist, dim=1)
            taw_preds = torch.argmin(dist[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset

            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)

        return 0, float(taw_acc.compute()), float(tag_acc.compute())


def train_child(model, prototypes, K, trn_loader, val_loader, t, num_classes_in_t, wd, nepochs, task_offset, alpha, use_negative_class, device):
    old_model = copy.deepcopy(model)
    old_model.eval()
    if t == 0:
        model = train_initial(model, t, trn_loader, val_loader, num_classes_in_t, wd, nepochs, task_offset, device)
    elif use_negative_class:
        model = train_incremental_neg(model, old_model, K, trn_loader, val_loader, nepochs, alpha, device)
    else:
        model = train_incremental(model, old_model, K, trn_loader, val_loader, nepochs, alpha, device)

    if t > 0:
        print(f"{os.getpid()}: ### Adapting prototypes ###")
        prototypes = adapt_prototypes(model, prototypes, old_model, t, trn_loader, val_loader, nepochs, device)
    prototypes = create_prototypes(model, prototypes, t, trn_loader, val_loader, num_classes_in_t, task_offset, device)
    return copy.deepcopy(model), copy.deepcopy(prototypes)


def train_initial(model, t, trn_loader, val_loader, num_classes_in_t, wd, nepochs, task_offset, device):
    print(f'{os.getpid()}: The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    print(f'{os.getpid()}: The expert has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} shared parameters\n')
    distiller = nn.Linear(64, 64)

    distiller.to(device, non_blocking=True)
    criterion = CE(num_classes_in_t, 64, device)
    parameters = list(model.parameters()) + list(criterion.parameters()) + list(distiller.parameters())
    optimizer, lr_scheduler = get_optimizer(parameters, wd, 0.1, nepochs)

    for epoch in range(nepochs):
        train_loss, train_kd_loss, valid_loss, valid_kd_loss = [], [], [], []
        train_hits, val_hits = 0, 0
        model.train()
        criterion.train()
        distiller.train()
        for images, targets in trn_loader:
            targets -= task_offset[t]
            bsz = images.shape[0]
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            features = model(images)
            loss, logits = criterion(features, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            optimizer.step()
            if logits is not None:
                train_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
            train_loss.append(float(bsz * loss))
        lr_scheduler.step()

        model.eval()
        criterion.eval()
        distiller.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                targets -= task_offset[t]
                bsz = images.shape[0]
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                features = model(images)
                loss, logits = criterion(features, targets)
                if logits is not None:
                    val_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                valid_loss.append(float(bsz * loss))

        train_loss = sum(train_loss) / len(trn_loader.dataset)
        train_kd_loss = sum(train_kd_loss) / len(trn_loader.dataset)
        valid_loss = sum(valid_loss) / len(val_loader.dataset)
        valid_kd_loss = sum(valid_kd_loss) / len(val_loader.dataset)

        train_acc = train_hits / len(trn_loader.dataset)
        val_acc = val_hits / len(val_loader.dataset)

        print(f"{os.getpid()}: Epoch: {epoch} Train: {train_loss:.2f} KD: {train_kd_loss:.3f} Acc: {100 * train_acc:.2f} "
              f"Val: {valid_loss:.2f} KD: {valid_kd_loss:.3f} Acc: {100 * val_acc:.2f}")
    return model

def prepare_expert_loader(loader, classes, use_negative_class):
    if use_negative_class:
        images = copy.deepcopy(loader.dataset.images)
        targets = copy.deepcopy(np.array(loader.dataset.labels))
        labels = np.unique(targets)
        for c in labels:
            if c not in classes:
                targets[targets == c] = -1
        for i, c in enumerate(classes):
            targets[targets == c] = i
        targets[targets == -1] = i+1
    else:
        is_in = np.isin(loader.dataset.labels, classes)
        images = copy.deepcopy(loader.dataset.images[is_in])
        targets = copy.deepcopy(np.array(loader.dataset.labels)[is_in])
        for i, c in enumerate(classes):
            targets[targets == c] = i

    ds = BabelMemoryDataset(images, targets, transforms=loader.dataset.transform)
    return torch.utils.data.DataLoader(ds, batch_size=loader.batch_size, num_workers=0, shuffle=True)


def train_incremental(model, old_model, K, trn_loader, val_loader, nepochs, alpha, device):
    print(f'{os.getpid()}: The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    print(f'{os.getpid()}: The expert has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} shared parameters\n')
    distiller = nn.Linear(64, 64)

    # Freeze batch norms
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    # Prepare expert loaders
    all_classes = np.unique(trn_loader.dataset.labels)
    classes = random.sample(list(all_classes), K)
    print(classes)

    expert_train_loader = prepare_expert_loader(trn_loader, classes, False)
    expert_val_loader = prepare_expert_loader(val_loader, classes, False)

    distiller.to(device, non_blocking=True)
    criterion = CE(len(classes), 64, device)
    parameters = list(model.parameters()) + list(criterion.parameters()) + list(distiller.parameters())
    optimizer, lr_scheduler = get_optimizer(parameters, 0, 0.1, nepochs)

    for epoch in range(nepochs):
        train_loss, train_kd_loss, valid_loss, valid_kd_loss = [], [], [], []
        train_hits, val_hits = 0, 0
        model.train()
        criterion.train()
        distiller.train()
        train_iterator = iter(trn_loader)
        for expert_images, expert_targets in expert_train_loader:
            expert_images, expert_targets = expert_images.to(device, non_blocking=True), expert_targets.to(device, non_blocking=True)
            images, _ = next(train_iterator)
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()
            expert_features = model(expert_images)
            features = model(images)
            if epoch < 10:
                features = features.detach()
                expert_features = expert_features.detach()
            loss, logits = criterion(expert_features, expert_targets)
            with torch.no_grad():
                old_features = old_model(images)
            total_loss, kd_loss = distill_knowledge(loss, features, distiller, old_features, alpha)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            optimizer.step()
            train_hits += float(torch.sum((torch.argmax(logits, dim=1) == expert_targets)))
            train_loss.append(float(expert_images.shape[0] * loss))
            train_kd_loss.append(float(images.shape[0] * kd_loss))
        lr_scheduler.step()

        model.eval()
        criterion.eval()
        distiller.eval()
        with torch.no_grad():
            val_iterator = iter(val_loader)
            for expert_images, expert_targets in expert_val_loader:
                expert_images, expert_targets = expert_images.to(device, non_blocking=True), expert_targets.to(device, non_blocking=True)
                images, _ = next(val_iterator)
                images = images.to(device, non_blocking=True)
                expert_features = model(expert_images)
                features = model(images)
                loss, logits = criterion(expert_features, expert_targets)
                old_features = old_model(images)

                _, kd_loss = distill_knowledge(loss, features, distiller, old_features, alpha)
                val_hits += float(torch.sum((torch.argmax(logits, dim=1) == expert_targets)))
                valid_loss.append(float(expert_images.shape[0] * loss))
                valid_kd_loss.append(float(images.shape[0] * kd_loss))

        train_loss = sum(train_loss) / len(trn_loader.dataset)
        train_kd_loss = sum(train_kd_loss) / len(trn_loader.dataset)
        valid_loss = sum(valid_loss) / len(val_loader.dataset)
        valid_kd_loss = sum(valid_kd_loss) / len(val_loader.dataset)

        train_acc = train_hits / len(expert_train_loader.dataset)
        val_acc = val_hits / len(expert_val_loader.dataset)

        print(f"{os.getpid()}: Epoch: {epoch} Train: {train_loss:.2f} KD: {train_kd_loss:.3f} Acc: {100 * train_acc:.2f} "
              f"Val: {valid_loss:.2f} KD: {valid_kd_loss:.3f} Acc: {100 * val_acc:.2f}")
    return model


def train_incremental_neg(model, old_model, K, trn_loader, val_loader, nepochs, alpha, device):
    print(f'{os.getpid()}: The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    print(f'{os.getpid()}: The expert has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} shared parameters\n')
    distiller = nn.Linear(64, 64)

    # Freeze batch norms
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    # Prepare expert loaders
    all_classes = np.unique(trn_loader.dataset.labels)
    classes = random.sample(list(all_classes), K)
    print(classes)
    expert_train_loader = prepare_expert_loader(trn_loader, classes, True)
    expert_val_loader = prepare_expert_loader(val_loader, classes, True)

    distiller.to(device, non_blocking=True)
    criterion = CE(len(classes)+1, 64, device)
    parameters = list(model.parameters()) + list(criterion.parameters()) + list(distiller.parameters())
    optimizer, lr_scheduler = get_optimizer(parameters, 0, 0.1, nepochs)

    for epoch in range(nepochs):
        train_loss, train_kd_loss, valid_loss, valid_kd_loss = [], [], [], []
        train_hits, val_hits = 0, 0
        model.train()
        criterion.train()
        distiller.train()
        for expert_images, expert_targets in expert_train_loader:
            expert_images, expert_targets = expert_images.to(device, non_blocking=True), expert_targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            expert_features = model(expert_images)
            if epoch < 10:
                expert_features = expert_features.detach()
            loss, logits = criterion(expert_features, expert_targets)
            with torch.no_grad():
                old_features = old_model(expert_images)
            total_loss, kd_loss = distill_knowledge(loss, expert_features, distiller, old_features, alpha)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            optimizer.step()
            train_hits += float(torch.sum((torch.argmax(logits, dim=1) == expert_targets)))
            train_loss.append(float(expert_images.shape[0] * loss))
            train_kd_loss.append(float(expert_images.shape[0] * kd_loss))
        lr_scheduler.step()

        model.eval()
        criterion.eval()
        distiller.eval()
        with torch.no_grad():
            for expert_images, expert_targets in expert_val_loader:
                expert_images, expert_targets = expert_images.to(device, non_blocking=True), expert_targets.to(device, non_blocking=True)
                expert_features = model(expert_images)
                loss, logits = criterion(expert_features, expert_targets)
                old_features = old_model(expert_images)

                _, kd_loss = distill_knowledge(loss, expert_features, distiller, old_features, alpha)
                val_hits += float(torch.sum((torch.argmax(logits, dim=1) == expert_targets)))
                valid_loss.append(float(expert_images.shape[0] * loss))
                valid_kd_loss.append(float(expert_images.shape[0] * kd_loss))

        train_loss = sum(train_loss) / len(expert_train_loader.dataset)
        train_kd_loss = sum(train_kd_loss) / len(expert_train_loader.dataset)
        valid_loss = sum(valid_loss) / len(expert_val_loader.dataset)
        valid_kd_loss = sum(valid_kd_loss) / len(expert_val_loader.dataset)

        train_acc = train_hits / len(expert_train_loader.dataset)
        val_acc = val_hits / len(expert_val_loader.dataset)

        print(f"{os.getpid()}: Epoch: {epoch} Train: {train_loss:.2f} KD: {train_kd_loss:.3f} Acc: {100 * train_acc:.2f} "
              f"Val: {valid_loss:.2f} KD: {valid_kd_loss:.3f} Acc: {100 * val_acc:.2f}")
    return model

@torch.no_grad()
def create_prototypes(model, prototypes, t, trn_loader, val_loader, num_classes_in_t, task_offset, device):
    """ Create distributions for task t"""
    print(f"{os.getpid()}: Creating prototypes")
    model.eval()
    transforms = val_loader.dataset.transform
    new_protos = torch.zeros((num_classes_in_t, 64), device=device)
    for c in range(num_classes_in_t):
        train_indices = torch.tensor(trn_loader.dataset.labels) == c + task_offset[t]
        if isinstance(trn_loader.dataset.images, list):
            train_images = list(compress(trn_loader.dataset.images, train_indices))
            ds = ClassDirectoryDataset(train_images, transforms)
        else:
            ds = trn_loader.dataset.images[train_indices]
            ds = ClassMemoryDataset(ds, transforms)
        loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=0, shuffle=False)
        from_ = 0
        class_features = torch.full((2 * len(ds), 64), fill_value=-999999999.0, device=device)
        for images in loader:
            bsz = images.shape[0]
            images = images.to(device, non_blocking=True)
            features = model(images)
            class_features[from_: from_+bsz] = features
            features = model(torch.flip(images, dims=(3,)))
            class_features[from_+bsz: from_+2*bsz] = features
            from_ += 2*bsz

        # Calculate centroid
        centroid = class_features.mean(dim=0)
        new_protos[c] = centroid

    prototypes = torch.cat((prototypes, new_protos), dim=0)
    return prototypes

def adapt_prototypes(model, prototypes, old_model, t, trn_loader, val_loader, nepochs, device):
    model.eval()
    adapter = nn.Linear(64, 64)

    adapter.to(device, non_blocking=True)
    optimizer, lr_scheduler = get_adapter_optimizer(adapter.parameters(), nepochs)
    for epoch in range(nepochs):
        adapter.train()
        train_loss, valid_loss = [], []
        for images, _ in trn_loader:
            bsz = images.shape[0]
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.no_grad():
                target = model(images)
                old_features = old_model(images)
            adapted_features = adapter(old_features)
            loss = torch.nn.functional.mse_loss(adapted_features, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1)
            optimizer.step()
            train_loss.append(float(bsz * loss))
        lr_scheduler.step()

        adapter.eval()
        with torch.no_grad():
            for images, _ in val_loader:
                bsz = images.shape[0]
                images = images.to(device, non_blocking=True)
                target = model(images)
                old_features = old_model(images)
                adapted_features = adapter(old_features)
                loss = torch.nn.functional.mse_loss(adapted_features, target)
                valid_loss.append(float(bsz * loss))

        train_loss = sum(train_loss) / len(trn_loader.dataset)
        valid_loss = sum(valid_loss) / len(val_loader.dataset)
        print(f"{os.getpid()}: Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} ")

    # Calculate new prototypes
    with torch.no_grad():
        adapter.eval()
        prototypes = adapter(prototypes)
    return prototypes

def distill_knowledge(loss, features, distiller, old_features, alpha=0.5):
    kd_loss = nn.functional.mse_loss(distiller(features), old_features)
    total_loss = (1 - alpha) * loss + alpha * kd_loss
    return total_loss, kd_loss

def get_optimizer(parameters, wd, lr, epochs):
    """Returns the optimizer"""
    milestones = (int(epochs*0.4), int(epochs*0.8))
    optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=wd, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
    return optimizer, scheduler

def get_adapter_optimizer(parameters, epochs):
    """Returns the optimizer"""
    milestones = (int(epochs*0.6), int(epochs*0.8))
    optimizer = torch.optim.SGD(parameters, lr=0.01, weight_decay=1e-5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
    return optimizer, scheduler
