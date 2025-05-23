import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import time 
from torch_scatter import scatter_mean, scatter_max
import math


class nativeContrastLoss_class(nn.Module):
    def __init__(self, sample_nums=1000):
        super(nativeContrastLoss_class, self).__init__()
        
        self.sample_nums = sample_nums
        self.temperature = 0.1
        self.base_temperature = 1 #1 #2
        self.ignore_label = 255
        self.num_classes = 17
        self.dim = 64

        self.pixel_update_freq = 30 # number  V , of pixels
        self.pixel_size = self.pixel_update_freq * 5 # self.pixel_update_freq * 5

        self.mu = 0.99
        self.cluster_center = torch.randn((self.num_classes, self.dim),requires_grad=False).cuda()
        self.cluster_center = nn.functional.normalize(self.cluster_center, p=2, dim=1)
        self.new_cluster_center = self.cluster_center.clone()

        self.point_queue = torch.randn((self.num_classes, self.pixel_size, self.dim),requires_grad=False).cuda()
        self.point_queue = nn.functional.normalize(self.point_queue, p=2, dim=2)
        self.point_queue_ptr = torch.zeros(self.num_classes, dtype=torch.long,requires_grad=False).cuda()


    def _update_operations(self):
        self.cluster_center = self.cluster_center * self.mu + self.new_cluster_center * (1 - self.mu)
        self.cluster_center = nn.functional.normalize(self.cluster_center, p=2, dim=1).detach_()

    def _assigning_subclass_labels(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]   # 16 64

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero(as_tuple=False).shape[0]]

            classes.append(this_classes)
            total_classes += len(this_classes)

        n_view = 100

        X_ = []
        y_ = []
        
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_x = X[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero(as_tuple=False)
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero(as_tuple=False)

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)
                else:
                    indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)

                xc = this_x[indices]
                yc = this_y_hat[indices]

                X_.append(xc)
                y_.append(yc)

    
        if len(X_) != 0:
            X_ = torch.cat(X_,dim=0).float()
            y_ = torch.cat(y_,dim=0).float()
        else:
            X_ = y_ = None

        return X_, y_
    
    def _queue_operations(self, feats, labels):

        this_feat = feats.contiguous().view(self.dim, -1)
        this_label = labels.contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        # this_label_ids = [x for x in this_label_ids if (x > 0) and (x != self.ignore_label)]
        this_label_ids = [x for x in this_label_ids]

        for lb in this_label_ids:
            idxs = (this_label == lb).nonzero(as_tuple=False)

            # pixel enqueue and dequeue
            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            updata_cnt = min(num_pixel, self.pixel_update_freq)
            # feat = this_feat[:, perm[:updata_cnt]]
            feat = this_feat[:, idxs[perm[:updata_cnt]].squeeze(-1)]
            # print(idxs.shape,this_feat.shape,perm[:updata_cnt])
            # torch.Size([27, 1]) torch.Size([64, 6566]) tensor([ 7,  8, 26, 11,  6, 12,  2,  4, 23, 22])
            # torch.Size([19, 1]) torch.Size([64, 6566]) tensor([13, 12, 14,  8,  4, 18,  5, 15,  9,  6])
            feat = torch.transpose(feat, 0, 1)
            ptr = int(self.point_queue_ptr[lb])

            if ptr + updata_cnt > self.pixel_size:
                self.point_queue[lb, -updata_cnt:, :] = nn.functional.normalize(feat, p=2, dim=1).detach_()
                self.point_queue_ptr[lb] = 0
            else:
                self.point_queue[lb, ptr:ptr + updata_cnt, :] = nn.functional.normalize(feat, p=2, dim=1).detach_()
                self.point_queue_ptr[lb] = (self.point_queue_ptr[lb] + updata_cnt) % self.pixel_size

    def _sample_negative(self):
        class_num, cache_size, feat_size = self.point_queue.shape
        reduce_num = 0
        X_ = torch.zeros(((class_num - reduce_num) * cache_size , feat_size)).float().cuda()
        y_ = torch.zeros(((class_num - reduce_num) * cache_size , 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            this_q = self.point_queue[ii, :cache_size, :]
            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_
    
    def _assigning_subclass_labels_cur(self, X, y_hat, y, cur):
        batch_size, feat_dim = X.shape[0], X.shape[-1]   # 16 64

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero(as_tuple=False).shape[0]]

            classes.append(this_classes)
            total_classes += len(this_classes)

        n_view = 100

        X_ = []
        y_ = []
        
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_x = X[ii]
            this_classes = classes[ii]
            this_cur = cur[ii]

            for cls_id in this_classes:
                # hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero(as_tuple=False)
                # easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero(as_tuple=False)

                class_indices = (this_y_hat == cls_id).nonzero(as_tuple=False)
                class_cur = this_cur[class_indices]
                c_threshold = class_cur.quantile(q=0.95)
                hard_indices = class_indices[class_cur.ge(torch.tensor(c_threshold)).bool().squeeze(1)]
                easy_indices = class_indices[class_cur.le(torch.tensor(c_threshold)).bool().squeeze(1)]

                # hard_cur = this_cur[hard_indices.squeeze(1)]
                # easy_cur = this_cur[easy_indices.squeeze(1)]
                # _, hard_cur_indices = torch.sort(hard_cur, dim=0, descending=True)
                # _, easy_cur_indices = torch.sort(easy_cur, dim=0, descending=True)
                # hard_indices = hard_indices[hard_cur_indices]
                # easy_indices = easy_indices[easy_cur_indices]

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    hard_indices = hard_indices[:num_hard_keep]
                    easy_indices = easy_indices[:num_easy_keep]
                    indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    hard_indices = hard_indices[:num_hard_keep]
                    easy_indices = easy_indices[:num_easy_keep]
                    indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    hard_indices = hard_indices[:num_hard_keep]
                    easy_indices = easy_indices[:num_easy_keep]
                    indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)
                else:
                    indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)

                xc = this_x[indices]
                yc = this_y_hat[indices]

                X_.append(xc)
                y_.append(yc)
    
        if len(X_) != 0:
            X_ = torch.cat(X_,dim=0).float()
            y_ = torch.cat(y_,dim=0).float()
        else:
            X_ = y_ = None

        return X_, y_

    def _ppc_contrastive(self, X_anchor, y_anchor):
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = X_anchor  # N x dim
        anchor_num = X_anchor.shape[0]

        contrast_feature = X_anchor  # N x dim
        contrast_label = y_anchor  # N x 1

        mask = torch.eq(y_anchor, contrast_label.T).float().cuda()  # N x N   True if the same sub-class

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num).view(-1, 1).cuda(),
                                                     0)        # N x N  对角线是0
        # mask = mask * logits_mask
        # neg_mask = 1 - mask

        neg_mask = 1 - mask
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits)
        o_i = torch.where(mask.sum(1)!=0)[0]   #不是每个样本都有正样本

        mean_log_prob_pos = (mask * log_prob).sum(1)[o_i] / mask.sum(1)[o_i]

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
    
    def _ppc_contrastive_v1(self, X_anchor, y_anchor):
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = X_anchor  # N x dim
        anchor_num = X_anchor.shape[0]

        contrast_feature = X_anchor  # N x dim
        contrast_label = y_anchor  # N x 1

        mask = torch.eq(y_anchor, contrast_label.T).float().cuda()  # N x N   True if the same sub-class

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num).view(-1, 1).cuda(),
                                                     0)        # N x N  对角线是0

        neg_mask_raw = 1 - mask
        mask = mask * logits_mask

        X_contrast, y_contrast = self._sample_negative()  # 2550 x 64    2550 x 1    17*30*5=2550   num_class * pixel_size
        y_contrast = y_contrast.contiguous().view(-1, 1)
        neg_mask = torch.eq(y_anchor, y_contrast.T).float().cuda()  # N * 2550
        neg_mask = 1 - neg_mask
        anchor_dot_contrast_neg = torch.div(torch.matmul(anchor_feature, X_contrast.T),
                                        self.temperature)
        logits_max_neg, _ = torch.max(anchor_dot_contrast_neg, dim=1, keepdim=True)
        logits_neg = anchor_dot_contrast_neg - logits_max_neg.detach() # it is to avoid the numerical overflow

        neg_logits_raw = torch.exp(logits) * neg_mask_raw
        neg_logits_raw = neg_logits_raw.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        neg_logits = torch.exp(logits_neg) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits + neg_logits_raw)
        o_i = torch.where(mask.sum(1)!=0)[0]   #不是每个样本都有正样本

        mean_log_prob_pos = (mask * log_prob).sum(1)[o_i] / mask.sum(1)[o_i]

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
    
    def _pcc_contrastive(self, X_anchor, y_anchor):
        y_anchor = y_anchor.contiguous().view(-1, 1)   # N x 1
        y_contrast = torch.arange(self.num_classes).contiguous().view(-1, 1).cuda()  # class x 1

        anchor_feature = X_anchor # N x dim
        anchor_label = y_anchor # N x 1

        contrast_feature = self.cluster_center # class x dim
        contrast_label = y_contrast   # class x 1

        mask = torch.eq(anchor_label, contrast_label.T).float().cuda() # N x class

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow

        neg_mask = 1 - mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feat_s, pred_s, label_s, cur=None):
        bs, f_dim, p_num = feat_s.shape

        feat_s = feat_s.permute(0, 2, 1)  # 8, 24000, 64
        feat_s = feat_s.contiguous().view(feat_s.shape[0], -1, feat_s.shape[-1]) # 8, 24000, 64
        feat_s = nn.functional.normalize(feat_s, p=2, dim=2) # 8, 24000, 64

        # feats_, labels_ = self._assigning_subclass_labels(feat_s, label_s, pred_s)
        feats_, labels_ = self._assigning_subclass_labels_cur(feat_s, label_s, pred_s, cur)

        if feats_ is None:
            return torch.tensor((0.)).cuda()

        ppc_loss = self._ppc_contrastive_v1(feats_, labels_)

        pcc_loss = self._pcc_contrastive(feats_, labels_)

        loss =  (pcc_loss * 10 + ppc_loss)




        self._queue_operations(feats_, labels_.long())

        totol_cls = torch.unique(labels_)
        for cls_id in range(self.num_classes):
            if cls_id not in totol_cls:
                self.new_cluster_center[cls_id] = self.cluster_center[cls_id].clone().detach()
            else:
                cur_feats = feats_[labels_==cls_id]
                self.new_cluster_center[cls_id] = cur_feats.mean(0).detach()
        self._update_operations()

        return loss, pcc_loss, ppc_loss


class nativeContrastLoss_subclass(nn.Module):
    def __init__(self, sample_nums=1000):
        super(nativeContrastLoss_subclass, self).__init__()
        
        self.sample_nums = sample_nums
        self.temperature = 0.1
        self.base_temperature = 1 #1 #2
        self.ignore_label = 255
        self.num_classes = 17
        self.dim = 64
        self.K = 6
        self.K_split = [0.95, 0.85, 0.75, 0.65, 0.55]

        self.pixel_update_freq = 30 # number  V , of pixels
        self.pixel_size = self.pixel_update_freq * 5 # self.pixel_update_freq * 5

        self.mu = 0.99
        self.cluster_center = torch.randn((self.num_classes, self.K, self.dim),requires_grad=False).cuda()
        self.cluster_center = nn.functional.normalize(self.cluster_center, p=2, dim=2)
        self.new_cluster_center = self.cluster_center.clone()

        self.point_queue = torch.randn((self.num_classes*self.K, self.pixel_size, self.dim),requires_grad=False).cuda()
        self.point_queue = nn.functional.normalize(self.point_queue, p=2, dim=2)
        self.point_queue_ptr = torch.zeros(self.num_classes*self.K, dtype=torch.long,requires_grad=False).cuda()


    def _update_operations(self):
        self.cluster_center = self.cluster_center * self.mu + self.new_cluster_center * (1 - self.mu)
        self.cluster_center = nn.functional.normalize(self.cluster_center, p=2, dim=2).detach_()
    
    def _queue_operations(self, feats, labels):

        this_feat = feats.contiguous().view(self.dim, -1)
        this_label = labels.contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        # this_label_ids = [x for x in this_label_ids if (x > 0) and (x != self.ignore_label)]
        this_label_ids = [x for x in this_label_ids]

        for lb in this_label_ids:
            idxs = (this_label == lb).nonzero(as_tuple=False)

            # pixel enqueue and dequeue
            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            updata_cnt = min(num_pixel, self.pixel_update_freq)
            # feat = this_feat[:, perm[:updata_cnt]]
            feat = this_feat[:, idxs[perm[:updata_cnt]].squeeze(-1)]
            # print(idxs.shape,this_feat.shape,perm[:updata_cnt])
            # torch.Size([27, 1]) torch.Size([64, 6566]) tensor([ 7,  8, 26, 11,  6, 12,  2,  4, 23, 22])
            # torch.Size([19, 1]) torch.Size([64, 6566]) tensor([13, 12, 14,  8,  4, 18,  5, 15,  9,  6])
            feat = torch.transpose(feat, 0, 1)
            ptr = int(self.point_queue_ptr[lb])

            if ptr + updata_cnt > self.pixel_size:
                self.point_queue[lb, -updata_cnt:, :] = nn.functional.normalize(feat, p=2, dim=1).detach_()
                self.point_queue_ptr[lb] = 0
            else:
                self.point_queue[lb, ptr:ptr + updata_cnt, :] = nn.functional.normalize(feat, p=2, dim=1).detach_()
                self.point_queue_ptr[lb] = (self.point_queue_ptr[lb] + updata_cnt) % self.pixel_size

    def _sample_negative(self):
        class_num, cache_size, feat_size = self.point_queue.shape
        reduce_num = 0
        X_ = torch.zeros(((class_num - reduce_num) * cache_size , feat_size)).float().cuda()
        y_ = torch.zeros(((class_num - reduce_num) * cache_size , 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            this_q = self.point_queue[ii, :cache_size, :]
            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_
    
    def _assigning_subclass_labels_cur(self, X, y_hat, y, cur):
        batch_size, feat_dim = X.shape[0], X.shape[-1]   # 16 64

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero(as_tuple=False).shape[0]]

            classes.append(this_classes)
            total_classes += len(this_classes)

        n_view = 100 // self.K

        X_ = []
        y_ = []
        
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_x = X[ii]
            this_classes = classes[ii]
            this_cur = cur[ii]

            for cls_id in this_classes:
                # hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero(as_tuple=False)
                # easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero(as_tuple=False)

                # class_indices = (this_y_hat == cls_id).nonzero(as_tuple=False)
                # class_cur = this_cur[class_indices]
                # c_threshold = class_cur.quantile(q=0.95)
                # hard_indices = class_indices[class_cur.ge(torch.tensor(c_threshold)).bool().squeeze(1)]
                # easy_indices = class_indices[class_cur.le(torch.tensor(c_threshold)).bool().squeeze(1)]

                # hard_cur = this_cur[hard_indices.squeeze(1)]
                # easy_cur = this_cur[easy_indices.squeeze(1)]
                # _, hard_cur_indices = torch.sort(hard_cur, dim=0, descending=True)
                # _, easy_cur_indices = torch.sort(easy_cur, dim=0, descending=True)
                # hard_indices = hard_indices[hard_cur_indices]
                # easy_indices = easy_indices[easy_cur_indices]

                # num_hard = hard_indices.shape[0]
                # num_easy = easy_indices.shape[0]

                # if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                #     num_hard_keep = n_view // 2
                #     num_easy_keep = n_view - num_hard_keep
                #     perm = torch.randperm(num_hard)
                #     hard_indices = hard_indices[perm[:num_hard_keep]]
                #     perm = torch.randperm(num_easy)
                #     easy_indices = easy_indices[perm[:num_easy_keep]]
                #     hard_indices = hard_indices[:num_hard_keep]
                #     easy_indices = easy_indices[:num_easy_keep]
                #     indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)
                # elif num_hard >= n_view / 2:
                #     num_easy_keep = num_easy
                #     num_hard_keep = n_view - num_easy_keep
                #     perm = torch.randperm(num_hard)
                #     hard_indices = hard_indices[perm[:num_hard_keep]]
                #     perm = torch.randperm(num_easy)
                #     easy_indices = easy_indices[perm[:num_easy_keep]]
                #     hard_indices = hard_indices[:num_hard_keep]
                #     easy_indices = easy_indices[:num_easy_keep]
                #     indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)
                # elif num_easy >= n_view / 2:
                #     num_hard_keep = num_hard
                #     num_easy_keep = n_view - num_hard_keep
                #     perm = torch.randperm(num_hard)
                #     hard_indices = hard_indices[perm[:num_hard_keep]]
                #     perm = torch.randperm(num_easy)
                #     easy_indices = easy_indices[perm[:num_easy_keep]]
                #     hard_indices = hard_indices[:num_hard_keep]
                #     easy_indices = easy_indices[:num_easy_keep]
                #     indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)
                # else:
                #     indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)


                class_indices = (this_y_hat == cls_id).nonzero(as_tuple=False)
                class_cur = this_cur[class_indices]
                c_ths = [1.]
                for kk in range(self.K-1):
                    c_threshold = class_cur.quantile(q=self.K_split[kk])
                    c_ths.append(c_threshold)
                c_ths.append(0.)

                for kk in range(self.K):
                    cur_indices = class_indices[(class_cur.le(torch.tensor(c_ths[kk])).bool().squeeze(1)) & (class_cur.ge(torch.tensor(c_ths[kk+1])).bool().squeeze(1))]
                    num_indices = cur_indices.shape[0]
                    num_sample = min(num_indices, n_view)
                    perm = torch.randperm(num_indices)
                    indices = cur_indices[perm[:num_sample]].squeeze(1)

                    xc = this_x[indices]
                    yc = this_y_hat[indices]

                    yc = yc * self.K
                    yc = yc +  kk

                    X_.append(xc)
                    y_.append(yc)
    
        if len(X_) != 0:
            X_ = torch.cat(X_,dim=0).float()
            y_ = torch.cat(y_,dim=0).float()
        else:
            X_ = y_ = None

        return X_, y_
    
    def _ppc_contrastive_v1(self, X_anchor, y_anchor):
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = X_anchor  # N x dim
        anchor_num = X_anchor.shape[0]

        contrast_feature = X_anchor  # N x dim
        contrast_label = y_anchor  # N x 1

        mask = torch.eq(y_anchor, contrast_label.T).float().cuda()  # N x N   True if the same sub-class

        # class_id = contrast_label//self.K
        # mask_class = torch.eq(class_id, class_id.T).bool().cuda()
        # N = mask.shape[0]
        # y_anchor1 = y_anchor.repeat(1, N)
        # y_anchor2 = y_anchor.squeeze(-1).unsqueeze(0).repeat(N, 1)
        # minu = y_anchor1 - y_anchor2
        # minu = (minu<=0) & (minu>=-5)
        # mask = (mask_class & minu).float()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num).view(-1, 1).cuda(),
                                                     0)        # N x N  对角线是0

        neg_mask_raw = 1 - mask
        mask = mask * logits_mask

        X_contrast, y_contrast = self._sample_negative()  # 2550 x 64    2550 x 1    17*30*5=2550   num_class * pixel_size
        y_contrast = y_contrast.contiguous().view(-1, 1)
        neg_mask = torch.eq(y_anchor, y_contrast.T).float().cuda()  # N * 2550

        # class_id1 = contrast_label//self.K
        # class_id2 = y_contrast//self.K
        # mask_class = torch.eq(class_id1, class_id2.T).bool().cuda()
        # N1 = y_anchor.shape[0]
        # N2 = y_contrast.shape[0]
        # y_anchor1 = y_anchor.repeat(1, N2)
        # y_anchor2 = y_contrast.squeeze(-1).unsqueeze(0).repeat(N1, 1)
        # minu = y_anchor1 - y_anchor2
        # minu = (minu<=0) & (minu>=-5)
        # neg_mask = (mask_class & minu).float()

        neg_mask = 1 - neg_mask
        anchor_dot_contrast_neg = torch.div(torch.matmul(anchor_feature, X_contrast.T),
                                        self.temperature)
        logits_max_neg, _ = torch.max(anchor_dot_contrast_neg, dim=1, keepdim=True)
        logits_neg = anchor_dot_contrast_neg - logits_max_neg.detach() # it is to avoid the numerical overflow

        neg_logits_raw = torch.exp(logits) * neg_mask_raw
        neg_logits_raw = neg_logits_raw.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        neg_logits = torch.exp(logits_neg) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits + neg_logits_raw)
        o_i = torch.where(mask.sum(1)!=0)[0]   #不是每个样本都有正样本

        mean_log_prob_pos = (mask * log_prob).sum(1)[o_i] / mask.sum(1)[o_i]

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
    
    def _pcc_contrastive(self, X_anchor, y_anchor):
        y_anchor = y_anchor.contiguous().view(-1, 1)   # N x 1
        y_contrast = torch.arange(self.num_classes*self.K).contiguous().view(-1, 1).cuda()  # class x 1

        anchor_feature = X_anchor # N x dim
        anchor_label = y_anchor # N x 1

        contrast_feature = self.cluster_center # class x dim
        contrast_feature = contrast_feature.view(self.num_classes*self.K, -1)
        contrast_label = y_contrast   # class x 1

        mask = torch.eq(anchor_label, contrast_label.T).float().cuda() # N x class

        # class_id1 = y_anchor//self.K
        # class_id2 = y_contrast//self.K
        # mask_class = torch.eq(class_id1, class_id2.T).bool().cuda()
        # N1 = y_anchor.shape[0]
        # N2 = y_contrast.shape[0]
        # y_anchor1 = y_anchor.repeat(1, N2)
        # y_anchor2 = y_contrast.squeeze(-1).unsqueeze(0).repeat(N1, 1)
        # minu = y_anchor1 - y_anchor2
        # minu = (minu<=0) & (minu>=-5)
        # mask = (mask_class & minu).float()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow

        neg_mask = 1 - mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feat_s, pred_s, label_s, cur=None):
        bs, f_dim, p_num = feat_s.shape

        feat_s = feat_s.permute(0, 2, 1)  # 8, 24000, 64
        feat_s = feat_s.contiguous().view(feat_s.shape[0], -1, feat_s.shape[-1]) # 8, 24000, 64
        feat_s = nn.functional.normalize(feat_s, p=2, dim=2) # 8, 24000, 64

        # feats_, labels_ = self._assigning_subclass_labels(feat_s, label_s, pred_s)
        feats_, labels_ = self._assigning_subclass_labels_cur(feat_s, label_s, pred_s, cur)

        if feats_ is None:
            return torch.tensor((0.)).cuda()

        ppc_loss = self._ppc_contrastive_v1(feats_, labels_)

        pcc_loss = self._pcc_contrastive(feats_, labels_)

        loss =  (pcc_loss * 10 + ppc_loss)




        self._queue_operations(feats_, labels_.long())

        cls_ids = labels_//self.K
        subcls_ids = (labels_%self.K).long()
        totol_cls = torch.unique(cls_ids)
        for cls_id in range(self.num_classes):
            if cls_id not in totol_cls:
                self.new_cluster_center[cls_id] = self.cluster_center[cls_id].detach()
            else:
                cur_feats = feats_[cls_ids==cls_id]
                cur_subcls = subcls_ids[cls_ids==cls_id]
                self.new_cluster_center[cls_id] = scatter_mean(cur_feats, cur_subcls, dim=0, dim_size=self.K).detach()
                del cur_feats, cur_subcls
        del cls_ids, subcls_ids, totol_cls
        self._update_operations()

        return loss, pcc_loss, ppc_loss
    
    def pseudo_label_from_prototype(self, feats):
        bs, dim, p_num = feats.shape

        feats = feats.permute(0, 2, 1)  # 8, 24000, 64
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1]) # 8, 24000, 64
        feats = nn.functional.normalize(feats, p=2, dim=2) # 8, 24000, 64
        feats = feats.view(bs * p_num, dim)  # 8*24000, 64
        dist = torch.matmul(feats, self.cluster_center.view(self.num_classes*self.K, self.dim).T)
        # 8*24000 x C*k
        dist = F.softmax(dist, dim=1)
        # dist = dist.view(bs, p_num, self.num_classes, self.K)
        # dist = dist.sum(-1)
        pseudo_logits, pseudo_label = torch.max(dist, dim=-1)
        pseudo_label = pseudo_label // self.K
        pseudo_label = pseudo_label.view(bs, p_num)
        pseudo_logits = pseudo_logits.view(bs, p_num)
        del dist

        return pseudo_label, pseudo_logits
    


class nativeContrastLoss_subclass_t(nn.Module):
    def __init__(self, sample_nums=1000):
        super(nativeContrastLoss_subclass_t, self).__init__()
        
        self.sample_nums = sample_nums
        self.temperature = 0.1
        self.base_temperature = 1 #1 #2
        self.ignore_label = 255
        self.num_classes = 17
        self.dim = 64
        self.K = 6
        self.K_split = [0.95, 0.85, 0.75, 0.65, 0.55]
        # self.K = 5
        # self.K_split = [0.95, 0.8, 0.65, 0.5]
        # self.K = 4
        # self.K_split = [0.9, 0.7, 0.5]
        # self.K = 3
        # self.K_split = [0.8, 0.5]
        # self.K = 2
        # self.K_split = [0.95]
        # self.K = 1
        # self.K_split = []

        self.pixel_update_freq = 30 # number  V , of pixels
        self.pixel_size = self.pixel_update_freq * 5 # self.pixel_update_freq * 5

        self.mu = 0.99
        self.cluster_center = torch.randn((self.num_classes, self.K, self.dim),requires_grad=False).cuda()
        self.cluster_center = nn.functional.normalize(self.cluster_center, p=2, dim=2)
        self.new_cluster_center = self.cluster_center.clone()

        self.point_queue = torch.randn((self.num_classes*self.K, self.pixel_size, self.dim),requires_grad=False).cuda()
        self.point_queue = nn.functional.normalize(self.point_queue, p=2, dim=2)
        self.point_queue_ptr = torch.zeros(self.num_classes*self.K, dtype=torch.long,requires_grad=False).cuda()


    def _update_operations(self):
        self.cluster_center = self.cluster_center * self.mu + self.new_cluster_center * (1 - self.mu)
        self.cluster_center = nn.functional.normalize(self.cluster_center, p=2, dim=2).detach_()
    
    def _queue_operations(self, feats, labels):

        this_feat = feats.contiguous().view(self.dim, -1)
        this_label = labels.contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        # this_label_ids = [x for x in this_label_ids if (x > 0) and (x != self.ignore_label)]
        this_label_ids = [x for x in this_label_ids]

        for lb in this_label_ids:
            idxs = (this_label == lb).nonzero(as_tuple=False)

            # pixel enqueue and dequeue
            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            updata_cnt = min(num_pixel, self.pixel_update_freq)
            # feat = this_feat[:, perm[:updata_cnt]]
            feat = this_feat[:, idxs[perm[:updata_cnt]].squeeze(-1)]
            # print(idxs.shape,this_feat.shape,perm[:updata_cnt])
            # torch.Size([27, 1]) torch.Size([64, 6566]) tensor([ 7,  8, 26, 11,  6, 12,  2,  4, 23, 22])
            # torch.Size([19, 1]) torch.Size([64, 6566]) tensor([13, 12, 14,  8,  4, 18,  5, 15,  9,  6])
            feat = torch.transpose(feat, 0, 1)
            ptr = int(self.point_queue_ptr[lb])

            if ptr + updata_cnt > self.pixel_size:
                self.point_queue[lb, -updata_cnt:, :] = nn.functional.normalize(feat, p=2, dim=1).detach_()
                self.point_queue_ptr[lb] = 0
            else:
                self.point_queue[lb, ptr:ptr + updata_cnt, :] = nn.functional.normalize(feat, p=2, dim=1).detach_()
                self.point_queue_ptr[lb] = (self.point_queue_ptr[lb] + updata_cnt) % self.pixel_size

    def _sample_negative(self):
        class_num, cache_size, feat_size = self.point_queue.shape
        reduce_num = 0
        X_ = torch.zeros(((class_num - reduce_num) * cache_size , feat_size)).float().cuda()
        y_ = torch.zeros(((class_num - reduce_num) * cache_size , 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            this_q = self.point_queue[ii, :cache_size, :]
            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_
    
    def _assigning_subclass_labels_cur(self, X, y_hat, y, X_t, cur):
        batch_size, feat_dim = X.shape[0], X.shape[-1]   # 16 64

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero(as_tuple=False).shape[0]]

            classes.append(this_classes)
            total_classes += len(this_classes)

        n_view = 100 // self.K

        X_ = []
        y_ = []
        X_t_ = []
        
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_x = X[ii]
            this_classes = classes[ii]
            this_cur = cur[ii]
            this_x_t = X_t[ii]

            for cls_id in this_classes:
                class_indices = (this_y_hat == cls_id).nonzero(as_tuple=False)
                class_cur = this_cur[class_indices]
                c_ths = [1.]
                for kk in range(self.K-1):
                    c_threshold = class_cur.quantile(q=self.K_split[kk])
                    c_ths.append(c_threshold)
                c_ths.append(0.)

                for kk in range(self.K):
                    cur_indices = class_indices[(class_cur.le(torch.tensor(c_ths[kk])).bool().squeeze(1)) & (class_cur.ge(torch.tensor(c_ths[kk+1])).bool().squeeze(1))]
                    num_indices = cur_indices.shape[0]
                    num_sample = min(num_indices, n_view)
                    perm = torch.randperm(num_indices)
                    indices = cur_indices[perm[:num_sample]].squeeze(1)

                    xc = this_x[indices]
                    yc = this_y_hat[indices]
                    xc_t = this_x_t[indices]

                    yc = yc * self.K
                    yc = yc +  kk

                    X_.append(xc)
                    y_.append(yc)
                    X_t_.append(xc_t)
    
        if len(X_) != 0:
            X_ = torch.cat(X_,dim=0).float()
            y_ = torch.cat(y_,dim=0).float()
            X_t_ = torch.cat(X_t_,dim=0).float()
        else:
            X_ = y_ = X_t_ = None

        return X_, y_, X_t_
    
    def _ppc_contrastive_v1(self, X_anchor, y_anchor, t_anchor):
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = X_anchor  # N x dim
        anchor_num = X_anchor.shape[0]

        contrast_feature = t_anchor  # N x dim
        contrast_label = y_anchor  # N x 1

        mask = torch.eq(y_anchor, contrast_label.T).float().cuda()  # N x N   True if the same sub-class

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num).view(-1, 1).cuda(),
                                                     0)        # N x N  对角线是0

        neg_mask_raw = 1 - mask
        mask = mask * logits_mask

        X_contrast, y_contrast = self._sample_negative()  # 2550 x 64    2550 x 1    17*30*5=2550   num_class * pixel_size
        y_contrast = y_contrast.contiguous().view(-1, 1)
        neg_mask = torch.eq(y_anchor, y_contrast.T).float().cuda()  # N * 2550

        neg_mask = 1 - neg_mask
        anchor_dot_contrast_neg = torch.div(torch.matmul(anchor_feature, X_contrast.T),
                                        self.temperature)
        logits_max_neg, _ = torch.max(anchor_dot_contrast_neg, dim=1, keepdim=True)
        logits_neg = anchor_dot_contrast_neg - logits_max_neg.detach() # it is to avoid the numerical overflow

        neg_logits_raw = torch.exp(logits) * neg_mask_raw
        neg_logits_raw = neg_logits_raw.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        neg_logits = torch.exp(logits_neg) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits + neg_logits_raw)
        o_i = torch.where(mask.sum(1)!=0)[0]   #不是每个样本都有正样本

        mean_log_prob_pos = (mask * log_prob).sum(1)[o_i] / mask.sum(1)[o_i]

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
    
    def _pcc_contrastive(self, X_anchor, y_anchor):
        y_anchor = y_anchor.contiguous().view(-1, 1)   # N x 1
        y_contrast = torch.arange(self.num_classes*self.K).contiguous().view(-1, 1).cuda()  # class x 1

        anchor_feature = X_anchor # N x dim
        anchor_label = y_anchor # N x 1

        contrast_feature = self.cluster_center # class x dim
        contrast_feature = contrast_feature.view(self.num_classes*self.K, -1)
        contrast_label = y_contrast   # class x 1

        mask = torch.eq(anchor_label, contrast_label.T).float().cuda() # N x class

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow

        neg_mask = 1 - mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
    
    def _assigning_subclass_labels_cur_top2(self, X, y_hat, y_hat2, cur):
        batch_size, feat_dim = X.shape[0], X.shape[-1]   # 16 64

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero(as_tuple=False).shape[0]]

            classes.append(this_classes)
            total_classes += len(this_classes)

        n_view = 100 // self.K

        X_ = []
        y_ = []
        y_2 = []
        
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_x = X[ii]
            this_classes = classes[ii]
            this_cur = cur[ii]
            this_y_hat2 = y_hat2[ii]

            for cls_id in this_classes:
                class_indices = (this_y_hat == cls_id).nonzero(as_tuple=False)
                class_cur = this_cur[class_indices]
                c_ths = [1.]
                for kk in range(self.K-1):
                    c_threshold = class_cur.quantile(q=self.K_split[kk])
                    c_ths.append(c_threshold)
                c_ths.append(0.)

                for kk in range(self.K):
                    cur_indices = class_indices[(class_cur.le(torch.tensor(c_ths[kk])).bool().squeeze(1)) & (class_cur.ge(torch.tensor(c_ths[kk+1])).bool().squeeze(1))]
                    num_indices = cur_indices.shape[0]
                    num_sample = min(num_indices, n_view)
                    perm = torch.randperm(num_indices)
                    indices = cur_indices[perm[:num_sample]].squeeze(1)

                    xc = this_x[indices]
                    yc = this_y_hat[indices]
                    yc2 = this_y_hat2[indices]

                    yc = yc * self.K
                    yc = yc +  kk

                    yc2 = yc2 * self.K
                    yc2 = yc2 +  kk

                    X_.append(xc)
                    y_.append(yc)
                    y_2.append(yc2)
    
        if len(X_) != 0:
            X_ = torch.cat(X_,dim=0).float()
            y_ = torch.cat(y_,dim=0).float()
            y_2 = torch.cat(y_2,dim=0).float()
        else:
            X_ = y_ = y_2 = None

        return X_, y_, y_2
    
    def _pcc_contrastive_top2(self, X_anchor, y_anchor, y_anchor2):
        y_anchor = y_anchor.contiguous().view(-1, 1)   # N x 1
        y_contrast = torch.arange(self.num_classes*self.K).contiguous().view(-1, 1).cuda()  # class x 1

        anchor_feature = X_anchor # N x dim
        anchor_label = y_anchor # N x 1

        contrast_feature = self.cluster_center # class x dim
        contrast_feature = contrast_feature.view(self.num_classes*self.K, -1)
        contrast_label = y_contrast   # class x 1

        y_anchor2 = y_anchor2.contiguous().view(-1, 1)   # N x 1
        anchor_label2 = y_anchor2 # N x 1

        mask1 = torch.eq(anchor_label, contrast_label.T).bool().cuda() # N x class
        mask2 = torch.eq(anchor_label2, contrast_label.T).bool().cuda() # N x class
        mask = (mask1 | mask2).float()
        # mask = mask1.float()
        # neg_mask12 = (mask1 | mask2).float()

        

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow

        # neg_mask = 1 - neg_mask12 
        neg_mask = 1 - mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feat_s, pred_s, label_s, feat_t, top2_mask, _top2_label, cur=None):
        bs, f_dim, p_num = feat_s.shape

        feat_s = feat_s.permute(0, 2, 1)  # 8, 24000, 64
        feat_s = feat_s.contiguous().view(feat_s.shape[0], -1, feat_s.shape[-1]) # 8, 24000, 64
        feat_s = nn.functional.normalize(feat_s, p=2, dim=2) # 8, 24000, 64

        feat_t = feat_t.permute(0, 2, 1)  # 8, 24000, 64
        feat_t = feat_t.contiguous().view(feat_t.shape[0], -1, feat_t.shape[-1]) # 8, 24000, 64
        feat_t = nn.functional.normalize(feat_t, p=2, dim=2) # 8, 24000, 64

        # feats_, labels_ = self._assigning_subclass_labels(feat_s, label_s, pred_s)
        feats_, labels_, featt_ = self._assigning_subclass_labels_cur(feat_s, label_s, pred_s, feat_t, cur)

        if feats_ is None:
            return torch.tensor((0.)).cuda()

        ppc_loss = self._ppc_contrastive_v1(feats_, labels_, featt_)

        pcc_loss = self._pcc_contrastive(feats_, labels_)

        loss =  (pcc_loss * 2 + ppc_loss)

        label_2 = _top2_label[:,0,:]
        label_2[~top2_mask] = 255
        label_3 = _top2_label[:,1,:]
        label_3[~top2_mask] = 255
        feats_top2, labels_top2, labels_top3 = self._assigning_subclass_labels_cur_top2(feat_s[bs//2:], label_2, label_3, cur[bs//2:])
        pcc_top2_loss = self._pcc_contrastive_top2(feats_top2, labels_top2, labels_top3)
        loss += pcc_top2_loss * 2

        # pcc_top2_loss = torch.tensor((0.))


        self._queue_operations(featt_, labels_.long())

        cls_ids = labels_//self.K
        subcls_ids = (labels_%self.K).long()
        totol_cls = torch.unique(cls_ids)
        for cls_id in range(self.num_classes):
            if cls_id not in totol_cls:
                self.new_cluster_center[cls_id] = self.cluster_center[cls_id].detach()
            else:
                cur_feats = featt_[cls_ids==cls_id]
                cur_subcls = subcls_ids[cls_ids==cls_id]
                self.new_cluster_center[cls_id] = scatter_mean(cur_feats, cur_subcls, dim=0, dim_size=self.K).detach()
                del cur_feats, cur_subcls
        del cls_ids, subcls_ids, totol_cls
        self._update_operations()

        return loss, pcc_loss, ppc_loss, pcc_top2_loss
    
    def pseudo_label_from_prototype(self, feats):
        bs, dim, p_num = feats.shape

        feats = feats.permute(0, 2, 1)  # 8, 24000, 64
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1]) # 8, 24000, 64
        feats = nn.functional.normalize(feats, p=2, dim=2) # 8, 24000, 64
        feats = feats.view(bs * p_num, dim)  # 8*24000, 64
        dist = torch.matmul(feats, self.cluster_center.view(self.num_classes*self.K, self.dim).T)
        # 8*24000 x C*k
        dist = F.softmax(dist, dim=1)
        dist = dist.view(bs, p_num, self.num_classes, self.K)
        dist = dist.sum(-1)
        pseudo_logits, pseudo_label = torch.max(dist, dim=-1)
        # pseudo_label = pseudo_label // self.K
        # pseudo_label = pseudo_label.view(bs, p_num)
        # pseudo_logits = pseudo_logits.view(bs, p_num)
        del dist

        return pseudo_label, pseudo_logits
    

class nativeContrastLoss_t(nn.Module):
    def __init__(self, sample_nums=1024):
        super(nativeContrastLoss_t, self).__init__()
        
        self.sample_nums = sample_nums
        self.temperature = 0.1
        self.base_temperature = 1 #1 #2
        self.ignore_label = 255
        self.num_classes = 17
        self.dim = 128

        self.pixel_update_freq = sample_nums # number  V , of pixels
        self.pixel_size = self.pixel_update_freq * 4 # self.pixel_update_freq * 4

        self.point_queue = torch.randn((self.pixel_size, self.dim),requires_grad=False).cuda()
        self.point_queue = nn.functional.normalize(self.point_queue, p=2, dim=1)
        self.point_queue_ptr = torch.zeros(1, dtype=torch.long,requires_grad=False).cuda()

    
    def _queue_operations(self, feats):
        # pixel enqueue and dequeue
        num_pixel = feats.shape[0]
        perm = torch.randperm(num_pixel)
        updata_cnt = min(num_pixel, self.pixel_update_freq)
        feat = feats[perm[:updata_cnt], :]
        ptr = int(self.point_queue_ptr)

        if ptr + updata_cnt > self.pixel_size:
            self.point_queue[-updata_cnt:, :] = nn.functional.normalize(feat, p=2, dim=1).detach_()
            self.point_queue_ptr = 0
        else:
            self.point_queue[ptr:ptr + updata_cnt, :] = nn.functional.normalize(feat, p=2, dim=1).detach_()
            self.point_queue_ptr = (self.point_queue_ptr + updata_cnt) % self.pixel_size

    def _sample_negative(self):
        cache_size, feat_size = self.point_queue.shape
        X_ = torch.zeros((cache_size , feat_size)).float().cuda()
        sample_ptr = 0

        this_q = self.point_queue[:cache_size, :]
        X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
        sample_ptr += cache_size

        return X_
    
    def _assigning_class_labels(self, X, score, X_t, th):
        batch_size, feat_dim = X.shape[0], X.shape[-1]   # 16 64

        X_ = []
        X_t_ = []
        
        for ii in range(batch_size):
            this_x = X[ii]
            this_x_t = X_t[ii]
            this_s = score[ii]

            _indices = (this_s >= th).nonzero(as_tuple=False)
            # _indices = (this_s < th).nonzero(as_tuple=False)

            num_indices = _indices.shape[0]
            num_sample = min(num_indices, self.sample_nums)
            perm = torch.randperm(num_indices)
            indices = _indices[perm[:num_sample]].squeeze(1)

            xc = this_x[indices]
            xc_t = this_x_t[indices]

            X_.append(xc)
            X_t_.append(xc_t)
    
        if len(X_) != 0:
            X_ = torch.cat(X_,dim=0).float()
            X_t_ = torch.cat(X_t_,dim=0).float()
        else:
            X_ = X_t_ = None

        return X_, X_t_
    
    def _ppc_contrastive(self, X_anchor, t_anchor):
        y_anchor = torch.arange(X_anchor.shape[0]).cuda()
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = X_anchor  # N x dim
        anchor_num = X_anchor.shape[0]

        contrast_feature = t_anchor  # N x dim
        contrast_label = y_anchor  # N x 1

        mask = torch.eq(y_anchor, contrast_label.T).float().cuda()  # N x N   True if the same sub-class

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow


        neg_mask_raw = 1 - mask

        neg_logits_raw = torch.exp(logits) * neg_mask_raw
        neg_logits_raw = neg_logits_raw.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits_raw)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
    
    def _ppc_contrastive_bank(self, X_anchor, t_anchor):
        y_anchor = torch.arange(X_anchor.shape[0]).cuda()
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = X_anchor  # N x dim
        anchor_num = X_anchor.shape[0]

        contrast_feature = t_anchor  # N x dim
        contrast_label = y_anchor  # N x 1

        mask = torch.eq(y_anchor, contrast_label.T).float().cuda()  # N x N   True if the same sub-class

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow

        X_contrast = self._sample_negative()  # 2550 x 64    2550 x 1    17*30*5=2550   num_class * pixel_size
        anchor_dot_contrast_neg = torch.div(torch.matmul(anchor_feature, X_contrast.T),
                                        self.temperature)
        logits_max_neg, _ = torch.max(anchor_dot_contrast_neg, dim=1, keepdim=True)
        logits_neg = anchor_dot_contrast_neg - logits_max_neg.detach() # it is to avoid the numerical overflow

        neg_logits = torch.exp(logits_neg)
        neg_logits = neg_logits.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
    
    def _ppc_contrastive_andbank(self, X_anchor, t_anchor):
        y_anchor = torch.arange(X_anchor.shape[0]).cuda()
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = X_anchor  # N x dim
        anchor_num = X_anchor.shape[0]

        contrast_feature = t_anchor  # N x dim
        contrast_label = y_anchor  # N x 1

        mask = torch.eq(y_anchor, contrast_label.T).float().cuda()  # N x N   True if the same sub-class

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow

        neg_mask_raw = 1 - mask

        X_contrast = self._sample_negative()  # 2550 x 64    2550 x 1    17*30*5=2550   num_class * pixel_size
        anchor_dot_contrast_neg = torch.div(torch.matmul(anchor_feature, X_contrast.T),
                                        self.temperature)
        logits_max_neg, _ = torch.max(anchor_dot_contrast_neg, dim=1, keepdim=True)
        logits_neg = anchor_dot_contrast_neg - logits_max_neg.detach() # it is to avoid the numerical overflow

        neg_logits_raw = torch.exp(logits) * neg_mask_raw
        neg_logits_raw = neg_logits_raw.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        neg_logits = torch.exp(logits_neg)
        neg_logits = neg_logits.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits + neg_logits_raw)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
    
    

    def forward(self, feat_s, score, feat_t, cur=None):
        bs, p_num, f_dim = feat_s.shape

        feat_s = feat_s.contiguous().view(feat_s.shape[0], -1, feat_s.shape[-1]) # 8, 24000, 64
        feat_s = nn.functional.normalize(feat_s, p=2, dim=2) # 8, 24000, 64

        feat_t = feat_t.contiguous().view(feat_t.shape[0], -1, feat_t.shape[-1]) # 8, 24000, 64
        feat_t = nn.functional.normalize(feat_t, p=2, dim=2) # 8, 24000, 64

        th = torch.tensor(0.9).cuda()
        feats_, featt_ = self._assigning_class_labels(feat_s, score, feat_t, th=th)

        if feats_ is None:
            return torch.tensor((0.)).cuda()
        

        ppc_loss = self._ppc_contrastive_andbank(feats_, featt_)

        pcc_loss = torch.tensor(0)
        pcc_top2_loss = torch.tensor(0)

        loss =  ppc_loss


        self._queue_operations(featt_)


        return loss, pcc_loss, ppc_loss, pcc_top2_loss
