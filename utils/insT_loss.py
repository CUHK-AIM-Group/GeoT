import torch
import torch.nn.functional as F

from pointops.functions import pointops
from openpoints.models.layers import torch_grouping_operation, knn_point



class feature_space_loss(torch.nn.Module):
    def __init__(self, k=7, sigma=1.0, num_classes=17):
        super(feature_space_loss, self).__init__()
        self.k = k
        self.sigma = sigma
        self.num_classes = num_classes

    def forward(self, logits, labels, ins_T):
        logits = logits.permute(0, 2, 1).contiguous() # B N C
        # top_k_index, top_k_dist = pointops.knn(logits, logits, self.k+1) # B, N, k+1
        top_k_dist, top_k_index = knn_point(self.k+1, logits, logits)
        top_k_index = top_k_index[:, :, 1:]
        B, N, _ = top_k_index.shape
        factor = torch.arange((B)).unsqueeze(-1).repeat(1, N).cuda()
        logits = logits.view(B*N, -1).contiguous() # BN C
        labels = labels.view(-1) # BN
        neigh_logits = logits.clone().unsqueeze(1) # BN 1 C
        neigh_labels = labels.clone().unsqueeze(1) # BN 1
        neigh_ins_T = ins_T.clone().unsqueeze(1) # BN 1 C C
        for i in range(self.k):
            cur_index = top_k_index[:,:,i].long()
            cur_index = (cur_index + factor * N).long().contiguous().view(-1)
            top_k_logits = torch.index_select(logits, dim=0, index=cur_index)
            top_k_labels = torch.index_select(labels, dim=0, index=cur_index)
            top_k_ins_T = torch.index_select(ins_T, dim=0, index=cur_index)
            neigh_logits = torch.cat((neigh_logits, top_k_logits.unsqueeze(1)), dim=1)
            neigh_labels = torch.cat((neigh_labels, top_k_labels.unsqueeze(1)), dim=1)
            neigh_ins_T = torch.cat((neigh_ins_T, top_k_ins_T.unsqueeze(1)), dim=1)

        neigh_logits = neigh_logits[:,1:,:] # BN k C
        neigh_labels = neigh_labels[:,1:] # BN k
        neigh_ins_T = neigh_ins_T[:,1:,:,:] # BN k C C

        distance_map =  -torch.ones(B*N, self.k).cuda()
        # distance_map =  torch.zeros(B*N, self.k).cuda()
        virtual_labels = labels.unsqueeze(1).repeat(1, self.k).cuda()
        distance_map[virtual_labels==neigh_labels] = 1
        virtual_logits = logits.unsqueeze(1).repeat(1, self.k, 1).cuda()
        eij_dis = torch.exp((-torch.sum((virtual_logits - neigh_logits) ** 2, dim = 2)) / (2 * (self.sigma ** 2))) # BN k
        # print(torch.sum(distance_map,1))
        distance_map = distance_map * eij_dis
        # print(torch.sum(distance_map,1))

        virtual_ins_T = ins_T.unsqueeze(1).repeat(1, self.k, 1, 1).cuda()
        neigh_ins_T = neigh_ins_T.view(B*N, self.k, -1)
        virtual_ins_T = virtual_ins_T.view(B*N, self.k, -1)
        T_dist = torch.sum((virtual_ins_T - neigh_ins_T) ** 2, dim = 2)
        manifold_loss = torch.mean(distance_map.detach() * T_dist)

        return manifold_loss
    

class threeD_space_loss(torch.nn.Module):
    def __init__(self, k=7, sigma=1.0, num_classes=17):
        super(threeD_space_loss, self).__init__()
        self.k = k
        self.sigma = sigma
        self.num_classes = num_classes

    def forward(self, positions, labels, ins_T):
        top_k_dist, top_k_index = knn_point(self.k+1, positions, positions)
        top_k_index = top_k_index[:, :, 1:]
        B, N, _ = top_k_index.shape
        factor = torch.arange((B)).unsqueeze(-1).repeat(1, N).cuda()
        positions = positions.view(B*N, -1).contiguous() # BN C
        labels = labels.view(-1) # BN
        neigh_positions = positions.clone().unsqueeze(1) # BN 1 C
        neigh_labels = labels.clone().unsqueeze(1) # BN 1
        neigh_ins_T = ins_T.clone().unsqueeze(1) # BN 1 C C
        for i in range(self.k):
            cur_index = top_k_index[:,:,i].long()
            cur_index = (cur_index + factor * N).long().contiguous().view(-1)
            top_k_positions = torch.index_select(positions, dim=0, index=cur_index)
            top_k_labels = torch.index_select(labels, dim=0, index=cur_index)
            top_k_ins_T = torch.index_select(ins_T, dim=0, index=cur_index)
            neigh_positions = torch.cat((neigh_positions, top_k_positions.unsqueeze(1)), dim=1)
            neigh_labels = torch.cat((neigh_labels, top_k_labels.unsqueeze(1)), dim=1)
            neigh_ins_T = torch.cat((neigh_ins_T, top_k_ins_T.unsqueeze(1)), dim=1)

        neigh_positions = neigh_positions[:,1:,:] # BN k C
        neigh_labels = neigh_labels[:,1:] # BN k
        neigh_ins_T = neigh_ins_T[:,1:,:,:] # BN k C C

        # distance_map =  -torch.ones(B*N, self.k).cuda()
        distance_map =  torch.zeros(B*N, self.k).cuda()
        virtual_labels = labels.unsqueeze(1).repeat(1, self.k).cuda()
        distance_map[virtual_labels==neigh_labels] = 1
        virtual_positions = positions.unsqueeze(1).repeat(1, self.k, 1).cuda()
        eij_dis = torch.exp((-torch.sum((virtual_positions - neigh_positions) ** 2, dim = 2)) / (2 * (self.sigma ** 2))) # BN k
        # print(torch.sum(distance_map,1))
        distance_map = distance_map * eij_dis
        # print(torch.sum(distance_map,1))

        virtual_ins_T = ins_T.unsqueeze(1).repeat(1, self.k, 1, 1).cuda()
        neigh_ins_T = neigh_ins_T.view(B*N, self.k, -1)
        virtual_ins_T = virtual_ins_T.view(B*N, self.k, -1)
        T_dist = torch.sum((virtual_ins_T - neigh_ins_T) ** 2, dim = 2)
        # manifold_loss = torch.mean(distance_map.detach() * T_dist)
        manifold_loss = torch.sum(distance_map.detach() * T_dist, dim=1)/(torch.sum(distance_map.detach(), dim=1) + 0.001)
        manifold_loss = manifold_loss.mean()

        return manifold_loss


class Idenyity_loss(torch.nn.Module):
    def __init__(self):
        super(Idenyity_loss, self).__init__()

    def forward(self, insT, Identity):
        # num = insT.size(0)
        # Identity = Identity.repeat(num, 1, 1)
        # insT = insT.view(num, -1)
        # Identity = Identity.view(num, -1)
        # loss = F.mse_loss(insT, Identity, reduction='mean')

        num = insT.size(0)
        Identity = Identity.repeat(num, 1, 1)
        insT = insT.view(num, -1)
        Identity = Identity.view(num, -1)
        diff = (insT - Identity).pow(2)
        loss = torch.sum(diff * Identity, dim=1) / torch.sum(Identity, dim=1)
        loss = loss.mean()

        return loss
