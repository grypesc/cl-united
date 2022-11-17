import torch
from torch import nn
from torch.distributions.normal import Normal
from copy import deepcopy


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


class NExpertsKSelectors(LLL_Net):
    def __init__(self, backbone, taskcla):
        super().__init__(backbone, remove_existing_head=True)
        self.taskcla = taskcla
        self.freeze_backbone()
        self.selector_features_dim = 512
        self.selectors_num = 1
        self.subset_size = 512
        self.selector_heads = nn.ModuleList([SelectorHead(self.selector_features_dim, self.subset_size) for _ in range(self.selectors_num)])
        tasks_total = len(taskcla)
        self.means = torch.zeros(tasks_total, self.selectors_num, self.selector_features_dim)
        self.covs = torch.zeros(tasks_total, self.selectors_num, self.selector_features_dim, self.selector_features_dim)
        self.task_distributions = []
        self.model.tasks_learned_so_far = None

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets. Head is an expert here.
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        with torch.no_grad():
            x = self.model(x)
        return [head(x) for head in self.heads], x

    def forward_selectors(self, features):
        with torch.no_grad():
            return [head(features) for head in self.selector_heads]

    def predict_task(self, features):
        if self.tasks_learned_so_far == 1:
            return 0
        with torch.no_grad():
            features = torch.stack([head(features) for head in self.selector_heads], dim=1)
            features = features.cpu()
            log_pdfs = [torch.cat([self.task_distributions[t][s].log_prob(features[:, s]) for s in range(self.selectors_num)], dim=0)
                        for t in range(self.tasks_learned_so_far)]
            log_pdfs = torch.stack(log_pdfs, dim=0)
            _, task_votes = torch.max(log_pdfs, dim=0)
            task_id, _ = torch.mode(task_votes)
        return task_id


class SelectorHead(nn.Module):
    def __init__(self, out_dim, subset_size):
        super().__init__()
        self.linear = nn.Linear(512, out_dim, bias=False)
        self.linear.weight.data.uniform_(-1.0, to=1.0)
        # vals = torch.rand_like(self.linear.weight)
        # _, sorted_indices = torch.sort(vals)
        # mask = vals < 0
        # mask.scatter_(1, sorted_indices[:, :subset_size], ~mask)
        # self.linear.weight = torch.nn.Parameter(self.linear.weight * mask)
        self.linear.requires_grad = False

    def forward(self, x):
        return self.linear(x)
