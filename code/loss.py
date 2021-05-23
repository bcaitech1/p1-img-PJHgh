import torch
import torch.nn as nn
import torch.nn.functional as F


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class ModifiedLabelSmoothingLoss(nn.Module):
    def __init__(self, device, classes=7, smoothing=0.01, dim=-1):
        super(ModifiedLabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.new_age_label = self._mk_true_dist()
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = self._label_select(target)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
    def _label_select(self, target):
        return torch.index_select(self.new_age_label.to(device), dim=0, index=target.long()).requires_grad_(requires_grad=False)
    
    def _mk_true_dist(self):
        '''
            label | age
                0 : ~20
                1 : 21~29
                2 : 30~49
                3 : 50~53
                4 : 54~57
                5 : 58~59
                6 : 60~
        '''
        x = np.linspace(-1., 1., 2*self.cls-1)
        gaussian_distribution = (1 / np.sqrt(2*np.pi*self.smoothing))*np.exp(-0.5*pow(x, 2)/self.smoothing)
        
        new_age_label = []
        for m in range(0, self.cls):
            age_gaussian_label = self._softmax(gaussian_distribution[int((self.cls-1)-m):int((2*self.cls-1)-m)])
            new_age_label.append(age_gaussian_label)
        return torch.tensor(new_age_label, requires_grad=False)
        
    def _softmax(self, a) :
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

def f1_loss(y_true, y_pred, Task):
    if Task == 'mask':
        num_classes=3
    elif Task == 'gender':
        num_classes=2
    elif Task == 'age':
#         num_classes=7
        num_classes=3
    else:
        num_classes=18
        
    y_true = F.one_hot(y_true.long(), num_classes)
    y_pred = F.softmax(y_pred, dim=1)
    
    tp = (y_true * y_pred).sum(dim=0)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0)
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    fn = (y_true * (1 - y_pred)).sum(dim=0)
    
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1.sum()/num_classes, recall.sum()/num_classes, precision.sum()/num_classes