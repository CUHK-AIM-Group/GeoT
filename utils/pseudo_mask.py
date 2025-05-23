import torch
from pointops.functions import pointops
import numpy as np

def get_neigbor_tensors(X: torch.tensor, n: int, pos):
    """
    Args:
        X: tensor of shape (B, C, N)
        n: number of neighbors to get
        pos: tensor, B, N, 3
    Returns:
        list of tensors of shape (B, C, N)
    """
    
    neigbors = []
    neigbors_dist = []
    B, C, N = X.shape

    top_dist_index, top_dist = pointops.knn(pos, pos, n+1) # B, N, n+1
    top_dist = top_dist[:, :, 1:] # B, N, n
    top_dist_index = top_dist_index[:, :, 1:].long() # B, N, n

    factor = torch.arange((B)).cuda().unsqueeze(-1).unsqueeze(-1) # B, 1, 1
    top_dist_index = top_dist_index + factor * N # B, N, n
    X = X.transpose(0,1).contiguous().view(C, -1) # C, B*N

    # print(top_dist[-1,1024,:])

    for ii in range(n):
        top_dist_index_ii = top_dist_index[:,:,ii].view(-1) # B*N
        topk_X = torch.index_select(X, dim=1, index=top_dist_index_ii) #  C, B*N
        topk_X = topk_X.view(C, B, N).transpose(0,1).contiguous() # B, C, N
        neigbors.append(topk_X)

    return neigbors, top_dist


def pseudo_label_refine(pred_t, th, pos, neigborhood_size=4, n_neigbors=1):
    B, C, N = pred_t.shape

    with torch.no_grad():
        neighbors, neigh_dist = get_neigbor_tensors(pred_t, n=neigborhood_size, pos=pos)
        neighbors = torch.stack(neighbors) # neigborhood_size, B, C, N

        k_neighbors, neigbor_idx = torch.topk(neighbors, k = n_neigbors, axis = 0) # n_neigbors, B, C, N
        for neighbor in k_neighbors:
            beta = torch.exp(torch.tensor(-1/2)).cuda() #for more neigbors use neigbor_idx
            pred_t = pred_t + beta*neighbor - (pred_t*neighbor)*beta

        logits_u_aug, label_u_aug = torch.max(pred_t.detach(), dim=1)
        thresh_mask = logits_u_aug.ge(torch.tensor(th)).bool()

    return thresh_mask

def pseudo_label_refine_margin(pred_t, th, pos, neigborhood_size=4, n_neigbors=1):
    E_joint = [0.9698153347167245, 0.9595924029774019, 0.9596092881209647, 
    0.9617471101196512, 0.9662687092798028, 0.9684095068416779, 
    0.9766432433032493, 0.9754884408811396, 0.9629032258064516, 
    0.9596091749248413, 0.9584221215955251, 0.9619788870996601, 
    0.9666700999073025, 0.968204136476084, 0.9760611218051148, 
    0.9746949382049295, 0.966996699669967]


    B, C, N = pred_t.shape

    with torch.no_grad():
        E = torch.tensor(E_joint).view((1, C, 1)).cuda()

        neighbors, neigh_dist = get_neigbor_tensors(pred_t, n=neigborhood_size, pos=pos)
        neighbors = torch.stack(neighbors) # neigborhood_size, B, C, N

        # pred_raw = pred_t.clone()
        # neigh_dist = neigh_dist[:,:,0].unsqueeze(1).repeat(1, C, 1)
        # neigh_mask = (neigh_dist <= 0.002)

        k_neighbors, neigbor_idx = torch.topk(neighbors, k = n_neigbors, axis = 0) # n_neigbors, B, C, N
        # k_neighbors = k_neighbors * neigh_mask.unsqueeze(0) + pred_raw.unsqueeze(0) * (~neigh_mask.unsqueeze(0))
        # k_neighbors = k_neighbors * neigh_mask.unsqueeze(0) + E.unsqueeze(0).repeat(1, B, 1, N) * (~neigh_mask.unsqueeze(0))
        for neighbor in k_neighbors:
            beta = torch.exp(torch.tensor(-1/2)).cuda() #for more neigbors use neigbor_idx
            # pred_t = pred_t + beta*neighbor - (torch.max(pred_t*neighbor, pred_t*E)*beta)
            pred_t = pred_t + beta*neighbor - (pred_t*neighbor*beta)

        # pred_t = pred_t * neigh_mask + pred_raw * (~neigh_mask)

        _topk,_=torch.topk(pred_t.detach(), 2, dim=1)
        _margin = _topk[:,0,:] - _topk[:,1,:] #shape: batch_size, N
        thresh_mask = _margin.ge(torch.tensor(th)).bool()

    return thresh_mask, _margin

def pseudo_label_refine_margin_v1(pred_t, th, drop_percent, pos, neigborhood_size=4, n_neigbors=1):
    E_joint = [0.9698153347167245, 0.9595924029774019, 0.9596092881209647, 
    0.9617471101196512, 0.9662687092798028, 0.9684095068416779, 
    0.9766432433032493, 0.9754884408811396, 0.9629032258064516, 
    0.9596091749248413, 0.9584221215955251, 0.9619788870996601, 
    0.9666700999073025, 0.968204136476084, 0.9760611218051148, 
    0.9746949382049295, 0.966996699669967]

    unrelated_matrix_list = [
            [3,4,5,6,7,8,10,11,12,13,14,15,16],
            [4,5,6,7,8,9,10,11,12,13,14,15,16],
            [1,5,6,7,8,9,10,11,12,13,14,15,16],
            [1,2,6,7,8,9,10,11,12,13,14,15,16],
            [1,2,3,7,8,9,10,11,12,13,14,15,16],
            [1,2,3,4,8,9,10,11,12,13,14,15,16],
            [1,2,3,4,5,9,10,11,12,13,14,15,16],
            [1,2,3,4,5,6,9,10,11,12,13,14,15,16],
            [2,3,4,5,6,7,8,11,12,13,14,15,16],
            [1,2,3,4,5,6,7,8,12,13,14,15,16],
            [1,2,3,4,5,6,7,8,9,13,14,15,16],
            [1,2,3,4,5,6,7,8,9,10,14,15,16],
            [1,2,3,4,5,6,7,8,9,10,11,15,16],
            [1,2,3,4,5,6,7,8,9,10,11,12,16],
            [1,2,3,4,5,6,7,8,9,10,11,12,13],
            [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        ]
    
    unrelated_matrix = []
    for ele in unrelated_matrix_list:
        unrelated_matrix.append(torch.from_numpy(np.array(ele)).cuda())


    B, C, N = pred_t.shape

    with torch.no_grad():
        E = torch.tensor(E_joint).view((1, C, 1)).cuda()

        neighbors, neigh_dist = get_neigbor_tensors(pred_t, n=neigborhood_size, pos=pos)
        neighbors = torch.stack(neighbors) # neigborhood_size, B, C, N


        k_neighbors, neigbor_idx = torch.topk(neighbors, k = n_neigbors, axis = 0) # n_neigbors, B, C, N
        for neighbor in k_neighbors:

            # neg_neighbor = 1 - neighbor
            # upper_bound = torch.zeros_like(neighbor).cuda()
            # for jj in range(len(unrelated_matrix)):
            #     cls_id = jj + 1

            #     upper_bound_cls = neighbor[:, unrelated_matrix[jj], :]
            #     upper_bound_cls_value = 1 - torch.sum(upper_bound_cls, dim=1, keepdim=False)
            #     upper_bound[:,cls_id,:] = upper_bound_cls_value * E[:,cls_id,:]

            # # upper_bound[:,0,:] = E[:,0,:]
            # upper_bound[:,0,:] = pred_t[:,0,:] * E[:,0,:]

            '''
            for jj in range(len(E_joint)):
                pA = pred_t[:, jj, :]
                pB = neighbor[:, jj, :]
                pBA = E[:, jj, :]
                pAB = pBA * pA /pB
            '''

            upper_bound = E * pred_t / neighbor



            pred_t = pred_t + neighbor - (pred_t*upper_bound)
            # pred_t = pred_t + neighbor - (torch.max(pred_t*upper_bound, pred_t*E))



        _topk,_=torch.topk(pred_t.detach(), 2, dim=1)
        _margin = _topk[:,0,:] - _topk[:,1,:] #shape: batch_size, N
        # th = np.percentile(_margin.detach().cpu().numpy().flatten(), 100-drop_percent)
        thresh_mask = _margin.ge(torch.tensor(th)).bool()

    return thresh_mask, _margin, th



class neigh_acc_count:
    def __init__(self, num_class=17):
        super(neigh_acc_count, self).__init__()

        self.num_class = num_class
        self.acc_array = np.zeros((num_class, 2))

    def update(self, pred, pos, neigborhood_size=4, n_neigbors=1):
        B, N = pred.shape
        with torch.no_grad():
            top_dist_index, top_dist = pointops.knn(pos, pos, 2) # B, N, n+1
            top_dist = top_dist[0, :, 1] #  N
            top_dist_index = top_dist_index[0, :, 1].long() #  N

            pred = pred[0]
            pred_neigh = pred[top_dist_index]
            acc = (pred==pred_neigh)

            for kk in range(self.num_class):
                mask = (pred==kk)
                self.acc_array[kk,0] += torch.sum(mask)
                cur_acc = acc * mask
                self.acc_array[kk,1] += torch.sum(cur_acc)


    '''
    [0.7074586198519727, 0.49887000028427664, 0.49550164019617393, 
    0.49767020663538325, 0.4992296928665319, 0.5031747360365424, 
    0.5213004898447716, 0.5057423136738844,   0, 
    0.4981427229783715, 0.49556925905963667, 0.495424858548379, 
    0.4995915255160394,  0.5049541075195256, 0.5197452326584849, 
    0.507299604477498,  0]

    [0.9698153347167245, 0.9595924029774019, 0.9596092881209647, 
    0.9617471101196512, 0.9662687092798028, 0.9684095068416779, 
    0.9766432433032493, 0.9754884408811396, 0.9629032258064516, 
    0.9596091749248413, 0.9584221215955251, 0.9619788870996601, 
    0.9666700999073025, 0.968204136476084, 0.9760611218051148, 
    0.9746949382049295, 0.966996699669967]
    '''
           
            
