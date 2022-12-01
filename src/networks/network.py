import torch
from torch import nn
from torch.distributions.normal import Normal
from copy import deepcopy

# from torchvision.models import resnet18
from src.networks.resnet32_linear_turbo import resnet32


class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        x = self.model(x)
        assert (len(self.heads) > 0), "Cannot access any head"
        y = []
        for head in self.heads:
            y.append(head(x))
        if return_features:
            return y, x
        else:
            return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass


class Extractor(LLL_Net):

    def __init__(self, backbone, taskcla, device):
        super().__init__(backbone, remove_existing_head=True)
        self.model = resnet32(num_classes=50)
        state_dict = torch.load("networks/best.pth")  # The model is trained on 50 tasks in repo: backbone-factory
        self.model.load_state_dict(state_dict)
        self.model.fc = nn.Identity()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.taskcla = taskcla
        self.selector_features_dim = 64
        self.subset_size = 10
        self.device = device
        self.head = SelectorHead(self.selector_features_dim, self.subset_size)
        tasks_total = len(taskcla)
        self.means = torch.zeros((tasks_total, self.selector_features_dim), device=device)
        self.covs = torch.zeros((tasks_total, self.selector_features_dim, self.selector_features_dim), device=device)
        self.task_distributions = []
        self.model.tasks_learned_so_far = None

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets. Head is an expert here.
        """
        pass

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
            return self.head(features)

    def predict_task(self, features):
        if self.tasks_learned_so_far == 1:
            return 0
        with torch.no_grad():
            log_probs = [self.task_distributions[t].log_prob(features) for t in range(self.tasks_learned_so_far)]
            log_probs = torch.stack(log_probs, dim=0)
            task_id = torch.argmax(log_probs)
        return task_id


class SelectorHead(nn.Module):
    def __init__(self, out_dim, subset_size):
        super().__init__()
        # self.linear = nn.Linear(512, out_dim)
        # self.linear.weight.data.uniform_(-1.0, to=1.0)
        # vals = torch.rand_like(self.linear.weight)
        # _, sorted_indices = torch.sort(vals)
        # mask = vals < 0
        # mask.scatter_(1, sorted_indices[:, :subset_size], ~mask)
        # self.linear.weight = torch.nn.Parameter(self.linear.weight * mask)
        # self.linear.requires_grad = False

    def forward(self, x):
        # x = nn.functional.normalize(x, p=2, dim=1)
        return x
