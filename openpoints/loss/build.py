import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from openpoints.utils import registry
from copy import deepcopy
from pointops.functions import pointops
from torch.autograd import Variable
from pointops.functions import pointops

LOSS = registry.Registry('loss')
LOSS.register_module(name='CrossEntropy', module=CrossEntropyLoss)
LOSS.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss)
LOSS.register_module(name='BCEWithLogitsLoss', module=BCEWithLogitsLoss)

@LOSS.register_module()
class SmoothCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2, 
                 ignore_index=None, 
                 num_classes=None, 
                 weight=None, 
                 return_valid=False
                 ):
        super(SmoothCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.return_valid = return_valid
        # Reduce label values in the range of logit shape
        if ignore_index is not None:
            reducing_list = torch.range(0, num_classes).long().cuda(non_blocking=True)
            inserted_value = torch.zeros((1, )).long().cuda(non_blocking=True)
            self.reducing_list = torch.cat([
                reducing_list[:ignore_index], inserted_value,
                reducing_list[ignore_index:]
            ], 0)
        if weight is not None:
            self.weight = torch.from_numpy(weight).float().cuda(
                non_blocking=True).squeeze()
        else:
            self.weight = None
            
    def forward(self, pred, gt):
        if len(pred.shape)>2:
            pred = pred.transpose(1, 2).reshape(-1, pred.shape[1])
        gt = gt.contiguous().view(-1)
        
        if self.ignore_index is not None: 
            valid_idx = gt != self.ignore_index
            pred = pred[valid_idx, :]
            gt = gt[valid_idx]        
            gt = torch.gather(self.reducing_list, 0, gt)
            
        if self.label_smoothing > 0:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * self.label_smoothing / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            if self.weight is not None:
                loss = -(one_hot * log_prb * self.weight).sum(dim=1).mean()
            else:
                loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gt, weight=self.weight)
        
        if self.return_valid:
            return loss, pred, gt
        else:
            return loss


@LOSS.register_module()
class MaskedCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2):
        super(MaskedCrossEntropy, self).__init__()
        self.creterion = CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, logit, target, mask):
        logit = logit.transpose(1, 2).reshape(-1, logit.shape[1])
        target = target.flatten()
        mask = mask.flatten()
        idx = mask == 1
        loss = self.creterion(logit[idx], target[idx])
        return loss

@LOSS.register_module()
class BCELogits(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = BCEWithLogitsLoss(**kwargs)
        
    def forward(self, logits, targets):
        if len(logits.shape)>2:
            logits = logits.transpose(1, 2).reshape(-1, logits.shape[1])
        targets = targets.contiguous().view(-1)
        num_clsses = logits.shape[-1]
        targets_onehot = F.one_hot(targets, num_classes=num_clsses).to(device=logits.device,dtype=logits.dtype)
        return self.criterion(logits, targets_onehot)

@LOSS.register_module()
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, logit, target):
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)  # N,C,H,W => N,C,H*W
            logit = logit.transpose(1, 2)  # N,C,H*W => N,H*W,C
            logit = logit.contiguous().view(-1, logit.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(logit)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != logit.data.type():
                self.alpha = self.alpha.type_as(logit.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()




@LOSS.register_module()
class Poly1CrossEntropyLoss(torch.nn.Module):
    def __init__(self,
                 num_classes: int =50,
                 epsilon: float = 1.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        if len(logits.shape)>2:
            logits = logits.transpose(1, 2).reshape(-1, logits.shape[1])
        labels = labels.contiguous().view(-1)

        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


@LOSS.register_module()
class Poly1FocalLoss(torch.nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False, 
                 **kwargs
                 ):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon. the main one to finetune. larger values -> better performace in imagenet
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        num_classes = logits.shape[1]
        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1
    
@LOSS.register_module()
class Poly1FocalLoss_U(torch.nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False, 
                 **kwargs
                 ):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon. the main one to finetune. larger values -> better performace in imagenet
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss_U, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels, logits_pred, thresh=0.95, mask=None):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        num_classes = logits.shape[1]
        p = torch.sigmoid(logits)

        label_raw = deepcopy(labels) # [6, 16000]

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                #print(labels.shape)  torch.Size([6, 16000])
                labels = F.one_hot(labels.unsqueeze(1), num_classes).transpose(1, -1).squeeze_(-1)
                #print(labels.shape)  torch.Size([6, 17, 16000])

        labels = labels.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if mask is not None:
            thresh_mask = mask.unsqueeze(1).repeat(1, poly1.shape[1], 1) 
        else:
            thresh_mask = logits_pred.ge(torch.tensor(thresh)).bool().unsqueeze(1).repeat(1, poly1.shape[1], 1) 

        # bg_mask = torch.ones_like(label_raw).cuda()
        # bg_mask[label_raw==0] = 0
        # bg_mask = bg_mask.unsqueeze(1).repeat(1, poly1.shape[1], 1) 
        # thresh_mask = thresh_mask * bg_mask
        poly1 = torch.sum(poly1 * thresh_mask) / (thresh_mask.sum() + 0.001)

        #print(thresh_mask.shape, pt.shape, labels.shape)
        #torch.Size([6, 17, 16000]) torch.Size([6, 17, 16000]) torch.Size([6, 17, 16000])

        # if self.reduction == "mean":
        #     poly1 = poly1.mean()
        # elif self.reduction == "sum":
        #     poly1 = poly1.sum()

        return poly1
    
@LOSS.register_module()
class Poly1FocalLoss_U_Cur(torch.nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False, 
                 **kwargs
                 ):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon. the main one to finetune. larger values -> better performace in imagenet
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss_U_Cur, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels, logits_pred, thresh=0.95, cur=None):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        num_classes = logits.shape[1]
        p = torch.sigmoid(logits)

        label_raw = deepcopy(labels) # [6, 16000]

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                #print(labels.shape)  torch.Size([6, 16000])
                labels = F.one_hot(labels.unsqueeze(1), num_classes).transpose(1, -1).squeeze_(-1)
                #print(labels.shape)  torch.Size([6, 17, 16000])

        labels = labels.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        thresh_mask = cur.ge(torch.tensor(thresh)).bool().unsqueeze(1).repeat(1, poly1.shape[1], 1) 
        # bg_mask = torch.ones_like(label_raw).cuda()
        # bg_mask[label_raw==0] = 0
        # bg_mask = bg_mask.unsqueeze(1).repeat(1, poly1.shape[1], 1) 
        # thresh_mask = thresh_mask * bg_mask
        poly1 = torch.sum(poly1 * thresh_mask) / (thresh_mask.sum() + 0.001)

        #print(thresh_mask.shape, pt.shape, labels.shape)
        #torch.Size([6, 17, 16000]) torch.Size([6, 17, 16000]) torch.Size([6, 17, 16000])

        return poly1
    

@LOSS.register_module()
class Poly1FocalLoss_U_top2(torch.nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False, 
                 **kwargs
                 ):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon. the main one to finetune. larger values -> better performace in imagenet
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss_U_top2, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels, logits_pred, pred_u, pos, thresh=0.95, mask=None):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        num_classes = logits.shape[1]
        p = torch.sigmoid(logits)

        label_raw = deepcopy(labels) # [6, 16000]

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                #print(labels.shape)  torch.Size([6, 16000])
                labels = F.one_hot(labels.unsqueeze(1), num_classes).transpose(1, -1).squeeze_(-1)
                #print(labels.shape)  torch.Size([6, 17, 16000])

        labels = labels.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        thresh_mask = logits_pred.ge(torch.tensor(thresh)).bool()


        _topk, _top2_label = torch.topk(pred_u.detach(), 2, dim=1)
        _sum = _topk[:,0,:] + _topk[:,1,:] #shape: batch_size, N
        top2_mask = _sum.ge(torch.tensor(0.9)).bool()
        top2_mask = (top2_mask & (~thresh_mask))
        label1 = _top2_label[:,0,:].contiguous().view(-1)
        label2 = _top2_label[:,1,:].contiguous().view(-1)

        top_dist_index, top_dist = pointops.knn(pos, pos, 2) # B, N, n+1
        top_dist = top_dist[:, :, 1] # B, N
        top_dist_index = top_dist_index[:, :, 1].long() # B, N
        B, N = top_dist_index.shape
        factor = torch.arange((B)).unsqueeze(-1).repeat(1, N).cuda()
        top_dist_index = (top_dist_index + factor * N).long().contiguous().view(-1)

        label1_topk = torch.index_select(label1, dim=0, index=top_dist_index)
        label2_topk = torch.index_select(label2, dim=0, index=top_dist_index)
        topk_mask = ((label1==label2_topk) & (label2==label1_topk))
        topk_mask = topk_mask.contiguous().view(B, N)
        topk_mask = (top2_mask & topk_mask)

        thresh_mask = (thresh_mask | topk_mask)

        new_mask = thresh_mask.unsqueeze(1).repeat(1, poly1.shape[1], 1) 


        # bg_mask = torch.ones_like(label_raw).cuda()
        # bg_mask[label_raw==0] = 0
        # bg_mask = bg_mask.unsqueeze(1).repeat(1, poly1.shape[1], 1) 
        # thresh_mask = thresh_mask * bg_mask
        poly1 = torch.sum(poly1 * new_mask) / (new_mask.sum() + 0.001)

        #print(thresh_mask.shape, pt.shape, labels.shape)
        #torch.Size([6, 17, 16000]) torch.Size([6, 17, 16000]) torch.Size([6, 17, 16000])

        # if self.reduction == "mean":
        #     poly1 = poly1.mean()
        # elif self.reduction == "sum":
        #     poly1 = poly1.sum()

        return poly1, thresh_mask, topk_mask

@LOSS.register_module()
class Poly1FocalLoss_U_T(torch.nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False, 
                 **kwargs
                 ):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon. the main one to finetune. larger values -> better performace in imagenet
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss_U_T, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels, logits_pred, T, pred_u_t, thresh=0.95, mask=None):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        num_classes = logits.shape[1]
        p = torch.sigmoid(logits)

        label_raw = deepcopy(labels) # [6, 16000]
        batch_size, num_point = label_raw.shape

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                #print(labels.shape)  torch.Size([6, 16000])
                labels = F.one_hot(labels.unsqueeze(1), num_classes).transpose(1, -1).squeeze_(-1)
                #print(labels.shape)  torch.Size([6, 17, 16000])

        labels = labels.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)



        # pred_before_t = logits.transpose(1, 2).contiguous().view(-1)  # BNC
        # pred_u_t = logits.transpose(1, 0).contiguous().view(-1, batch_size * num_point)
        # pred_u_t = torch.matmul(T.t(), pred_u_t)
        # pred_u_t = pred_u_t.view(-1, batch_size, num_point).transpose(1, 0).contiguous()
        # pred_after_t = pred_u_t.transpose(1, 2).contiguous().view(-1)  # BNC

        # label_raw = label_raw.contiguous().view(-1) #BN
        # label_raw_index = torch.arange(batch_size*num_point).long().cuda() * num_classes
        # label_raw_index = label_raw_index + label_raw
        # logits_before_t = pred_before_t[label_raw_index]
        # logits_before_t = logits_before_t.view(batch_size, num_point)
        # logits_after_t = pred_after_t[label_raw_index]
        # logits_after_t = logits_after_t.view(batch_size, num_point)

        logits_before_t = logits_pred
        pred_u_t = pred_u_t.transpose(1, 2).contiguous().view(-1)  # BNC
        label_raw = label_raw.contiguous().view(-1) #BN
        label_raw_index = torch.arange(batch_size*num_point).long().cuda() * num_classes
        label_raw_index = label_raw_index + label_raw
        logits_after_t = pred_u_t[label_raw_index]
        logits_after_t = logits_after_t.view(batch_size, num_point)

        beta = logits_before_t / logits_after_t
        # beta = Variable(beta.unsqueeze(1), requires_grad=True)
        beta = beta.unsqueeze(1)
        poly1 = poly1 * beta


        if mask is not None:
            thresh_mask = mask.unsqueeze(1).repeat(1, poly1.shape[1], 1) 
        else:
            thresh_mask = logits_pred.ge(torch.tensor(thresh)).bool().unsqueeze(1).repeat(1, poly1.shape[1], 1) 

        # bg_mask = torch.ones_like(label_raw).cuda()
        # bg_mask[label_raw==0] = 0
        # bg_mask = bg_mask.unsqueeze(1).repeat(1, poly1.shape[1], 1) 
        # thresh_mask = thresh_mask * bg_mask
        poly1 = torch.sum(poly1 * thresh_mask) / (thresh_mask.sum() + 0.001)

        #print(thresh_mask.shape, pt.shape, labels.shape)
        #torch.Size([6, 17, 16000]) torch.Size([6, 17, 16000]) torch.Size([6, 17, 16000])

        # if self.reduction == "mean":
        #     poly1 = poly1.mean()
        # elif self.reduction == "sum":
        #     poly1 = poly1.sum()

        return poly1
    
@LOSS.register_module()
class Poly1FocalLoss_U_T_v1(torch.nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False, 
                 **kwargs
                 ):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon. the main one to finetune. larger values -> better performace in imagenet
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss_U_T_v1, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels, logits_pred, T, pred_u, delta_T, thresh=0.95, mask=None):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        num_classes = logits.shape[1]
        p = torch.sigmoid(logits)

        label_raw = deepcopy(labels) # [6, 16000]
        batch_size, num_point = label_raw.shape

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                #print(labels.shape)  torch.Size([6, 16000])
                labels = F.one_hot(labels.unsqueeze(1), num_classes).transpose(1, -1).squeeze_(-1)
                #print(labels.shape)  torch.Size([6, 17, 16000])

        labels = labels.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)


        ema_t_corr = T + delta_T
        pred_u_t = pred_u.transpose(1, 0).contiguous().view(-1, batch_size * num_point)
        pred_u_t = torch.matmul(ema_t_corr.t(), pred_u_t)
        pred_u_t = pred_u_t.view(-1, batch_size, num_point).transpose(1, 0).contiguous()

        logits_before_t = logits_pred
        pred_u_t = pred_u_t.transpose(1, 2).contiguous().view(-1)  # BNC
        label_raw = label_raw.contiguous().view(-1) #BN
        label_raw_index = torch.arange(batch_size*num_point).long().cuda() * num_classes
        label_raw_index = label_raw_index + label_raw
        logits_after_t = pred_u_t[label_raw_index]
        logits_after_t = logits_after_t.view(batch_size, num_point)


        # beta = pred_u_t.view(( batch_size, -1,num_point))

        beta = logits_before_t / logits_after_t
        beta = beta.unsqueeze(1)
        poly1 = poly1 * beta


        if mask is not None:
            thresh_mask = mask.unsqueeze(1).repeat(1, poly1.shape[1], 1) 
        else:
            thresh_mask = logits_pred.ge(torch.tensor(thresh)).bool().unsqueeze(1).repeat(1, poly1.shape[1], 1) 

        poly1 = torch.sum(poly1 * thresh_mask) / (thresh_mask.sum() + 0.001)

        return poly1, delta_T


@LOSS.register_module()
class Poly1FocalLoss_U_corr(torch.nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False, 
                 **kwargs
                 ):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon. the main one to finetune. larger values -> better performace in imagenet
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss_U_corr, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels, logits_pred, thresh=0.95, mask=None):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        num_classes = logits.shape[1]
        p = torch.sigmoid(logits)

        label_raw = deepcopy(labels) # [6, 16000]

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                #print(labels.shape)  torch.Size([6, 16000])
                labels = F.one_hot(labels.unsqueeze(1), num_classes).transpose(1, -1).squeeze_(-1)
                #print(labels.shape)  torch.Size([6, 17, 16000])

        labels = labels.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if mask is not None:
            thresh_mask = mask.unsqueeze(1).repeat(1, poly1.shape[1], 1) 
        else:
            thresh_mask = logits_pred.ge(torch.tensor(thresh)).bool().unsqueeze(1).repeat(1, poly1.shape[1], 1) 

        # bg_mask = torch.ones_like(label_raw).cuda()
        # bg_mask[label_raw==0] = 0
        # bg_mask = bg_mask.unsqueeze(1).repeat(1, poly1.shape[1], 1) 
        # thresh_mask = thresh_mask * bg_mask
        poly1 = torch.sum(poly1 * thresh_mask) / (thresh_mask.sum() + 0.001)

        #print(thresh_mask.shape, pt.shape, labels.shape)
        #torch.Size([6, 17, 16000]) torch.Size([6, 17, 16000]) torch.Size([6, 17, 16000])

        # if self.reduction == "mean":
        #     poly1 = poly1.mean()
        # elif self.reduction == "sum":
        #     poly1 = poly1.sum()

        return poly1


@LOSS.register_module()
class MultiShapeCrossEntropy(torch.nn.Module):
    def __init__(self, criterion_args, **kwargs):
        super(MultiShapeCrossEntropy, self).__init__()
        self.criterion = build_criterion_from_cfg(criterion_args)

    def forward(self, logits_all_shapes, points_labels, shape_labels):
        batch_size = shape_labels.shape[0]
        losses = 0
        for i in range(batch_size):
            sl = shape_labels[i]
            logits = torch.unsqueeze(logits_all_shapes[sl][i], 0)
            pl = torch.unsqueeze(points_labels[i], 0)
            loss = self.criterion(logits, pl)
            losses += loss
        return losses / batch_size
    
@LOSS.register_module()
class Weight_CELoss(torch.nn.Module): 
    def __init__(self, **kwargs):
        super(Weight_CELoss, self).__init__()


    def forward(self, ret, gt, class_weights):
        # loss = F.cross_entropy(ret, gt.long(), weight=class_weights.mean(dim=0), reduction='none')

        temperature = 1.
        ret = F.log_softmax(ret/temperature, dim=1)
        loss = F.nll_loss(ret, gt.long(), weight=class_weights.mean(dim=0), reduction='none')

        return loss.mean()

@LOSS.register_module()
class Weight_CELoss_U(torch.nn.Module): 
    def __init__(self, **kwargs):
        super(Weight_CELoss_U, self).__init__()

    def forward(self, ret, gt, class_weights, logits, thresh=0.95):
        thresh_mask = logits.ge(torch.tensor(thresh)).bool()
        gt[~thresh_mask] = 255
        gt[gt==0] = 255
        loss = F.cross_entropy(ret, gt.long(), weight=class_weights.mean(dim=0), ignore_index=255, reduction='none')

        return loss.mean()

@LOSS.register_module()
class MSE_Loss_U(torch.nn.Module): 
    def __init__(self, **kwargs):
        super(MSE_Loss_U, self).__init__()

    def forward(self, ret, gt, thresh=0.95):
        ret_softmax = F.softmax(ret, dim=1)
        thresh_mask = ret_softmax.ge(torch.tensor(thresh)).bool()

        loss = F.mse_loss(ret_softmax, gt, reduction='none')

        loss = torch.sum(loss * thresh_mask.unsqueeze(1)) / (torch.sum(thresh_mask) + 0.001)

        return loss

def build_criterion_from_cfg(cfg, **kwargs):
    """
    Build a criterion (loss function), defined by cfg.NAME.
    Args:
        cfg (eDICT): 
    Returns:
        criterion: a constructed loss function specified by cfg.NAME
    """
    return LOSS.build(cfg, **kwargs)