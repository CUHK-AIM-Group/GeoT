import torch
import torch.nn as nn

from timm.models.layers import DropPath
from ..build import MODELS
from ..layers import SubsampleGroup

from pointops.functions import pointops
from pointnet2.pointnet2_modules import PointnetFPModule
import pointnet2.pointnet2_utils as pt_utils
from knn_cuda import KNN
import numpy as np

import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


@MODELS.register_module()
class PointTransformerGenEncoder(nn.Module):
    def __init__(self,
                 num_groups=256,
                 group_size=32,
                 subsample='fps',
                 group='ballquery',
                 radius=0.1,
                 encoder_dims=256,
                 trans_dim=384,
                 drop_path_rate=0.1,
                 depth=12,
                 num_heads=6,
                 **kwargs):
        super().__init__()
        # grouper
        self.group_divider = SubsampleGroup(num_groups, group_size, subsample, group, radius)
        # define the encoder
        self.encoder = Encoder(encoder_channel=encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(encoder_dims, trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = TransformerEncoder(
            embed_dim=trans_dim,
            depth=depth,
            drop_path_rate=dpr,
            num_heads=num_heads
        )

        self.norm = nn.LayerNorm(trans_dim)

    def forward_cls_feat(self, pts, x=None):
        if isinstance(pts, dict):
            pts, x = pts['pos'], pts.get('x', None)
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood.permute(0, 2, 3, 1))  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        return x[:, 1:, :], center


@MODELS.register_module()
class PointTransformerEncoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_groups=256,
                 group_size=32,
                 subsample='fps',
                 group='ballquery',
                 radius=0.1,
                 encoder_dims=256,
                 trans_dim=384,
                 drop_path_rate=0.1,
                 depth=12,
                 num_heads=6
                 ):
        super().__init__()
        # grouper
        self.group_divider = SubsampleGroup(num_groups, group_size, subsample, group, radius)
        # define the encoder
        self.encoder = Encoder(encoder_channel=encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(encoder_dims, trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = TransformerEncoder(
            embed_dim=trans_dim,
            depth=depth,
            drop_path_rate=dpr,
            num_heads=num_heads
        )

        self.norm = nn.LayerNorm(trans_dim)

    def forward_cls_feat(self, pts, f0=None):
        if isinstance(pts, dict):
            pts, x = pts['pos'], pts.get('x', None)
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood.permute(0, 2, 3, 1))  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0]], dim=-1)
        return concat_f

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pt_utils.furthest_point_sample(data, number) 
    fps_data = pt_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, idx

class DGCNN_Propagation(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        # print('using group version 2')
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(1024, 384, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 384),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pt_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pt_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """ coor, f : B 3 G ; B C G
            coor_q, f_q : B 3 N; B 3 N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q

class TransformerEncoder_h(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., finetune=False, extract_layers=None):
        super().__init__()

        self.finetune = finetune
        self.extract_layers = extract_layers
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        inter_feats = []
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if self.extract_layers is not None:
                if i+1 in self.extract_layers:
                    inter_feats.append(x)

        if self.extract_layers is not None:
            return inter_feats
        else:
            return x

@MODELS.register_module()
class PointTransformer_genencoder(nn.Module):
    def __init__(self, trans_dim, depth, drop_path_rate, nclasses, num_heads, group_size,
                 num_group, downsample_targets, extract_layers, encoder_dims, **kwargs):
        super().__init__()
        
        self.trans_dim = trans_dim
        self.depth = depth 
        self.drop_path_rate = drop_path_rate 
        self.nclasses = nclasses 
        self.num_heads = num_heads 

        self.group_size = group_size
        self.num_group = num_group

        self.downsample_targets = downsample_targets

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.extract_layers = extract_layers

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder_h(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
            finetune=True,
            extract_layers = self.extract_layers
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, p, f0=None):

        return self.forward_cls_feat(p, f0)

    def forward_cls_feat(self, p, f0=None):
        if isinstance(p, dict):
            pts, x = p['pos'], p.get('x', None)

        B, N, _ = pts.shape

        # divide the point clo  ud in the same form. This is important
        neighborhood, center, idx = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # add pos embedding
        pos = self.pos_embed(center)
        # transformer
        inter_feats = self.blocks(group_input_tokens, pos)
        inter_feats = [self.norm(x).transpose(-1, -2).contiguous() for x in inter_feats]
        

        return inter_feats[-1].transpose(1, 2), center

@MODELS.register_module()
class PointTransformer_seg(nn.Module):
    def __init__(self, trans_dim, depth, drop_path_rate, nclasses, num_heads, group_size,
                 num_group, downsample_targets, extract_layers, encoder_dims, **kwargs):
        super().__init__()
        
        self.trans_dim = trans_dim
        self.depth = depth 
        self.drop_path_rate = drop_path_rate 
        self.nclasses = nclasses 
        self.num_heads = num_heads 

        self.group_size = group_size
        self.num_group = num_group

        self.downsample_targets = downsample_targets


        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.extract_layers = extract_layers

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder_h(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
            finetune=True,
            extract_layers = self.extract_layers
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.propogation_2 = PointnetFPModule([self.trans_dim+3, self.trans_dim*4, self.trans_dim])
        self.propogation_1 = PointnetFPModule([self.trans_dim+3, self.trans_dim*4, self.trans_dim])
        self.propogation_0 = PointnetFPModule([self.trans_dim+3+2, self.trans_dim*4, self.trans_dim])

        self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

        self.seg_head = nn.Sequential(
            nn.Conv1d(self.trans_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.nclasses, 1),
        )

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, x=None, cls_label=None):
        B, N, _ = pts.shape

        # divide the point clo  ud in the same form. This is important
        neighborhood, center, idx = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # add pos embedding
        pos = self.pos_embed(center)
        # transformer
        inter_feats = self.blocks(group_input_tokens, pos)
        inter_feats = [self.norm(x).transpose(-1, -2).contiguous() for x in inter_feats]
        # one hot vector for describing upper and lower teeth
        cls_label_one_hot = F.one_hot(cls_label, 2).transpose(1, 2).float().repeat(1, 1, N)

        center_original = pts
        center_trans = center.transpose(-1, -2).contiguous()
        f_l0 = torch.cat([cls_label_one_hot, center_original.transpose(-1, -2).contiguous()], 1)
        
        # downsample the orginial point cloud
        assert len(inter_feats) == len(self.downsample_targets), \
            "the length of the cardinality and the features should be the same"
        
        center_pts = []
        for i in range(len(inter_feats)):
            center_pts.append(pointops.fps(pts, self.downsample_targets[i]))
        center_pts_trans = [pt.transpose(-1, -2).contiguous() for pt in center_pts]
        
        f_l3 = inter_feats[2]
        f_l2 = self.propogation_2(center_pts[1], center, center_pts_trans[1], inter_feats[1])
        f_l1 = self.propogation_1(center_pts[0], center, center_pts_trans[0], inter_feats[0])

        f_l2 = self.dgcnn_pro_2(center_trans, f_l3, center_pts_trans[1], f_l2)
        f_l1 = self.dgcnn_pro_1(center_pts_trans[1], f_l2, center_pts_trans[0], f_l1)
        
        f_l0 = self.propogation_0(center_original, center_pts[0], f_l0, f_l1)

        logit = self.seg_head(f_l0)

        # logit = F.log_softmax(logit, dim=1)

        return logit, 1, 2, 3 

@MODELS.register_module()
class PointTransformer_seg_cluster(nn.Module):
    def __init__(self, trans_dim, depth, drop_path_rate, nclasses, num_heads, group_size,
                 num_group, downsample_targets, extract_layers, encoder_dims, **kwargs):
        super().__init__()
        
        self.trans_dim = trans_dim
        self.depth = depth 
        self.drop_path_rate = drop_path_rate 
        self.nclasses = nclasses 
        self.num_heads = num_heads 

        self.group_size = group_size
        self.num_group = num_group

        self.downsample_targets = downsample_targets


        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.extract_layers = extract_layers

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder_h(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
            finetune=True,
            extract_layers = self.extract_layers
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.propogation_2 = PointnetFPModule([self.trans_dim+3, self.trans_dim*4, self.trans_dim])
        self.propogation_1 = PointnetFPModule([self.trans_dim+3, self.trans_dim*4, self.trans_dim])
        self.propogation_0 = PointnetFPModule([self.trans_dim+3+2, self.trans_dim*4, self.trans_dim])

        self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

        self.seg_head = nn.Sequential(
            nn.Conv1d(self.trans_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.nclasses, 1),
        )

        self.proj_head = nn.Sequential(
            nn.Conv1d(self.trans_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
        )

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, x=None, cls_label=None):
        B, N, _ = pts.shape

        # divide the point clo  ud in the same form. This is important
        neighborhood, center, idx = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # add pos embedding
        pos = self.pos_embed(center)
        # transformer
        inter_feats = self.blocks(group_input_tokens, pos)
        inter_feats = [self.norm(x).transpose(-1, -2).contiguous() for x in inter_feats]
        # one hot vector for describing upper and lower teeth
        cls_label_one_hot = F.one_hot(cls_label, 2).transpose(1, 2).float().repeat(1, 1, N)

        center_original = pts
        center_trans = center.transpose(-1, -2).contiguous()
        f_l0 = torch.cat([cls_label_one_hot, center_original.transpose(-1, -2).contiguous()], 1)
        
        # downsample the orginial point cloud
        assert len(inter_feats) == len(self.downsample_targets), \
            "the length of the cardinality and the features should be the same"
        
        center_pts = []
        for i in range(len(inter_feats)):
            center_pts.append(pointops.fps(pts, self.downsample_targets[i]))
        center_pts_trans = [pt.transpose(-1, -2).contiguous() for pt in center_pts]
        
        f_l3 = inter_feats[2]
        f_l2 = self.propogation_2(center_pts[1], center, center_pts_trans[1], inter_feats[1])
        f_l1 = self.propogation_1(center_pts[0], center, center_pts_trans[0], inter_feats[0])

        f_l2 = self.dgcnn_pro_2(center_trans, f_l3, center_pts_trans[1], f_l2)
        f_l1 = self.dgcnn_pro_1(center_pts_trans[1], f_l2, center_pts_trans[0], f_l1)
        
        f_l0 = self.propogation_0(center_original, center_pts[0], f_l0, f_l1)

        logit = self.seg_head(f_l0)

        # logit = F.log_softmax(logit, dim=1)

        feat = self.proj_head(f_l0)

        return logit, feat, 2, 3 
    

@MODELS.register_module()
class PointTransformer_seg_classifier(nn.Module):
    def __init__(self, trans_dim, depth, drop_path_rate, nclasses, num_heads, group_size,
                 num_group, downsample_targets, extract_layers, encoder_dims, **kwargs):
        super().__init__()
        
        self.trans_dim = trans_dim
        self.depth = depth 
        self.drop_path_rate = drop_path_rate 
        self.nclasses = nclasses 
        self.num_heads = num_heads 

        self.group_size = group_size
        self.num_group = num_group

        self.downsample_targets = downsample_targets


        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.extract_layers = extract_layers

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder_h(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
            finetune=True,
            extract_layers = self.extract_layers
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.propogation_2 = PointnetFPModule([self.trans_dim+3, self.trans_dim*4, self.trans_dim])
        self.propogation_1 = PointnetFPModule([self.trans_dim+3, self.trans_dim*4, self.trans_dim])
        self.propogation_0 = PointnetFPModule([self.trans_dim+3+2, self.trans_dim*4, self.trans_dim])

        self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

        self.seg_head = nn.Sequential(
            nn.Conv1d(self.trans_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.nclasses, 1),
        )

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, x=None, cls_label=None):
        B, N, _ = pts.shape

        # divide the point clo  ud in the same form. This is important
        neighborhood, center, idx = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # add pos embedding
        pos = self.pos_embed(center)
        # transformer
        inter_feats = self.blocks(group_input_tokens, pos)
        inter_feats = [self.norm(x).transpose(-1, -2).contiguous() for x in inter_feats]
        # one hot vector for describing upper and lower teeth
        cls_label_one_hot = F.one_hot(cls_label, 2).transpose(1, 2).float().repeat(1, 1, N)

        center_original = pts
        center_trans = center.transpose(-1, -2).contiguous()
        f_l0 = torch.cat([cls_label_one_hot, center_original.transpose(-1, -2).contiguous()], 1)
        
        # downsample the orginial point cloud
        assert len(inter_feats) == len(self.downsample_targets), \
            "the length of the cardinality and the features should be the same"
        
        center_pts = []
        for i in range(len(inter_feats)):
            center_pts.append(pointops.fps(pts, self.downsample_targets[i]))
        center_pts_trans = [pt.transpose(-1, -2).contiguous() for pt in center_pts]
        
        f_l3 = inter_feats[2]
        f_l2 = self.propogation_2(center_pts[1], center, center_pts_trans[1], inter_feats[1])
        f_l1 = self.propogation_1(center_pts[0], center, center_pts_trans[0], inter_feats[0])

        f_l2 = self.dgcnn_pro_2(center_trans, f_l3, center_pts_trans[1], f_l2)
        f_l1 = self.dgcnn_pro_1(center_pts_trans[1], f_l2, center_pts_trans[0], f_l1)
        
        f_l0 = self.propogation_0(center_original, center_pts[0], f_l0, f_l1)

        logit = self.seg_head(f_l0)

        logit_softmax = F.log_softmax(logit, dim=1)

        proto = self.seg_head[3].weight.detach()  # [17, 128, 1] C D
        # proto = self.seg_head[3].weight  # [17, 128, 1] C D
        proto = F.normalize(proto, p=2, dim=1).squeeze(-1)
        logit_softmax = logit_softmax.permute(0, 2, 1) # B N C
        feat = torch.matmul(logit_softmax, proto)  # B N D




        return logit, feat, 2, 3 
    

@MODELS.register_module()
class PointTransformer_seg_T(nn.Module):
    def __init__(self, trans_dim, depth, drop_path_rate, nclasses, num_heads, group_size,
                 num_group, downsample_targets, extract_layers, encoder_dims, **kwargs):
        super().__init__()
        
        self.trans_dim = trans_dim
        self.depth = depth 
        self.drop_path_rate = drop_path_rate 
        self.nclasses = nclasses 
        self.num_heads = num_heads 

        self.group_size = group_size
        self.num_group = num_group

        self.downsample_targets = downsample_targets


        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.extract_layers = extract_layers

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder_h(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
            finetune=True,
            extract_layers = self.extract_layers
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.propogation_2 = PointnetFPModule([self.trans_dim+3, self.trans_dim*4, self.trans_dim])
        self.propogation_1 = PointnetFPModule([self.trans_dim+3, self.trans_dim*4, self.trans_dim])
        self.propogation_0 = PointnetFPModule([self.trans_dim+3+2, self.trans_dim*4, self.trans_dim])

        self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

        self.seg_head = nn.Sequential(
            nn.Conv1d(self.trans_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.nclasses, 1),
        )

        self.apply(self._init_weights)

        self.T_revision = nn.Linear(self.nclasses, self.nclasses, False)
        nn.init.constant_(self.T_revision.weight, 0.0)

        self.T_linear = nn.Linear(self.nclasses, self.nclasses, False)
        nn.init.constant_(self.T_linear.weight, 0.0)

        # self.T_linear1 = nn.Linear(self.nclasses, 64, False)
        # self.relu1 = nn.ReLU()
        # nn.init.constant_(self.T_linear.weight, 0.0)
        # self.T_linear2 = nn.Linear(64, 64, False)
        # self.relu2 = nn.ReLU()
        # nn.init.constant_(self.T_linear.weight, 0.0)

        self.sigma = nn.Parameter(torch.Tensor(self.nclasses), requires_grad=True)
        nn.init.constant_(self.sigma, 0.4)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, x=None, cls_label=None, T=None):
        B, N, _ = pts.shape

        # divide the point clo  ud in the same form. This is important
        neighborhood, center, idx = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # add pos embedding
        pos = self.pos_embed(center)
        # transformer
        inter_feats = self.blocks(group_input_tokens, pos)
        inter_feats = [self.norm(x).transpose(-1, -2).contiguous() for x in inter_feats]
        # one hot vector for describing upper and lower teeth
        cls_label_one_hot = F.one_hot(cls_label, 2).transpose(1, 2).float().repeat(1, 1, N)

        center_original = pts
        center_trans = center.transpose(-1, -2).contiguous()
        f_l0 = torch.cat([cls_label_one_hot, center_original.transpose(-1, -2).contiguous()], 1)
        
        # downsample the orginial point cloud
        assert len(inter_feats) == len(self.downsample_targets), \
            "the length of the cardinality and the features should be the same"
        
        center_pts = []
        for i in range(len(inter_feats)):
            center_pts.append(pointops.fps(pts, self.downsample_targets[i]))
        center_pts_trans = [pt.transpose(-1, -2).contiguous() for pt in center_pts]
        
        f_l3 = inter_feats[2]
        f_l2 = self.propogation_2(center_pts[1], center, center_pts_trans[1], inter_feats[1])
        f_l1 = self.propogation_1(center_pts[0], center, center_pts_trans[0], inter_feats[0])

        f_l2 = self.dgcnn_pro_2(center_trans, f_l3, center_pts_trans[1], f_l2)
        f_l1 = self.dgcnn_pro_1(center_pts_trans[1], f_l2, center_pts_trans[0], f_l1)
        
        f_l0 = self.propogation_0(center_original, center_pts[0], f_l0, f_l1)

        logit = self.seg_head(f_l0)

        logit_softmax = F.log_softmax(logit, dim=1)

        if T is not None:
            correction = self.T_linear(T)
            # correction = correction + self.T_linear.weight

            # correction = self.T_linear1(T)
            # correction = self.relu1(correction)
            # correction = self.T_linear2(correction)
            # correction = self.relu2(correction)
            # correction = self.T_linear(correction)
        else:
            correction = None


        return logit, correction, self.sigma , f_l0
    

@MODELS.register_module()
class sig_t(nn.Module):
    # tensor T to parameter
    def __init__(self, nclasses):
        super(sig_t, self).__init__()
        self.nclasses = nclasses

        self.fc = nn.Linear(nclasses, nclasses * nclasses, bias=False)

        self.zeros = torch.zeros([nclasses, nclasses])
        self.w = torch.Tensor([])
        for i in range(nclasses):
            temp = self.zeros.clone()
            temp = temp+0.1/self.nclasses
            self.w = torch.cat([self.w, temp.detach()], 0)

        self.fc.weight.data = self.w


    def forward(self, x):
        out = x.permute(0, 2, 1).contiguous().view(-1, self.nclasses)  # BN C
        out = self.fc(out) # BN CC
        out = out.view(-1, self.nclasses, self.nclasses) # BN C C
        out = torch.clamp(out, min=1e-5,max=1-1e-5)
        out = F.normalize(out, p=1, dim=2).cuda()   # BN C C
        
        return out
    
@MODELS.register_module()
class sig_t_mean(nn.Module):
    # tensor T to parameter
    def __init__(self, nclasses):
        super(sig_t_mean, self).__init__()
        self.nclasses = nclasses

        self.fc = nn.ModuleList()
        for kk in range(self.nclasses):
            self.fc.append(nn.Linear(nclasses * 2, nclasses, bias=False))

        # self.fc1 = nn.ModuleList()
        # self.fc2 = nn.ModuleList()
        # self.fc3 = nn.ModuleList()
        # for kk in range(self.nclasses):
        #     self.fc1.append(nn.Linear(nclasses * 2, 256, bias=False))
        #     self.fc2.append(nn.Linear(256, 256, bias=False))
        #     self.fc3.append(nn.Linear(256, nclasses, bias=False))
        # self.relu = nn.ReLU()


    def forward(self, x, cm):
        out = x.permute(0, 2, 1).contiguous().view(-1, self.nclasses)  # BN C
        ins_T = torch.empty((out.shape[0], self.nclasses, self.nclasses)).cuda()
        for kk in range(self.nclasses):
            new_in = torch.cat((out, cm[kk].unsqueeze(0).repeat(out.shape[0], 1)), dim=1)
            new_in = self.fc[kk](new_in) # BN C
            ins_T[:, kk, :] = new_in

        ins_T = torch.clamp(ins_T, min=1e-5,max=1-1e-5)
        ins_T = F.normalize(ins_T, p=1, dim=2).cuda()   # BN C C
        
        return ins_T
    
@MODELS.register_module()
class PointTransformer_seg_2classifier(nn.Module):
    def __init__(self, trans_dim, depth, drop_path_rate, nclasses, num_heads, group_size,
                 num_group, downsample_targets, extract_layers, encoder_dims, **kwargs):
        super().__init__()
        
        self.trans_dim = trans_dim
        self.depth = depth 
        self.drop_path_rate = drop_path_rate 
        self.nclasses = nclasses 
        self.num_heads = num_heads 

        self.group_size = group_size
        self.num_group = num_group

        self.downsample_targets = downsample_targets


        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.extract_layers = extract_layers

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder_h(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
            finetune=True,
            extract_layers = self.extract_layers
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.propogation_2 = PointnetFPModule([self.trans_dim+3, self.trans_dim*4, self.trans_dim])
        self.propogation_1 = PointnetFPModule([self.trans_dim+3, self.trans_dim*4, self.trans_dim])
        self.propogation_0 = PointnetFPModule([self.trans_dim+3+2, self.trans_dim*4, self.trans_dim])

        self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

        self.seg_head = nn.Sequential(
            nn.Conv1d(self.trans_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.nclasses, 1),
        )

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, x=None, cls_label=None):
        B, N, _ = pts.shape

        # divide the point clo  ud in the same form. This is important
        neighborhood, center, idx = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # add pos embedding
        pos = self.pos_embed(center)
        # transformer
        inter_feats = self.blocks(group_input_tokens, pos)
        inter_feats = [self.norm(x).transpose(-1, -2).contiguous() for x in inter_feats]
        # one hot vector for describing upper and lower teeth
        cls_label_one_hot = F.one_hot(cls_label, 2).transpose(1, 2).float().repeat(1, 1, N)

        center_original = pts
        center_trans = center.transpose(-1, -2).contiguous()
        f_l0 = torch.cat([cls_label_one_hot, center_original.transpose(-1, -2).contiguous()], 1)
        
        # downsample the orginial point cloud
        assert len(inter_feats) == len(self.downsample_targets), \
            "the length of the cardinality and the features should be the same"
        
        center_pts = []
        for i in range(len(inter_feats)):
            center_pts.append(pointops.fps(pts, self.downsample_targets[i]))
        center_pts_trans = [pt.transpose(-1, -2).contiguous() for pt in center_pts]
        
        f_l3 = inter_feats[2]
        f_l2 = self.propogation_2(center_pts[1], center, center_pts_trans[1], inter_feats[1])
        f_l1 = self.propogation_1(center_pts[0], center, center_pts_trans[0], inter_feats[0])

        f_l2 = self.dgcnn_pro_2(center_trans, f_l3, center_pts_trans[1], f_l2)
        f_l1 = self.dgcnn_pro_1(center_pts_trans[1], f_l2, center_pts_trans[0], f_l1)
        
        f_l0 = self.propogation_0(center_original, center_pts[0], f_l0, f_l1)

        logit = self.seg_head(f_l0)

        # logit = F.log_softmax(logit, dim=1)

        return logit, 1, 2, 3 


@MODELS.register_module()
class Gragh_Matching(nn.Module):
    def __init__(self, in_channels, nclasses, sample_nums):
        super(Gragh_Matching, self).__init__()

        self.in_channels = in_channels
        self.nclasses = nclasses
        self.sample_nums = sample_nums


    def forward(self, feat_s, feat_t, label_t):
        pass

    def node_sampling(self, feat_s, feat_t, label_t):
        # auusme that label shape is (B,N)
        bs, c_num, p_num = feat_s.shape
        feat_s = feat_s.permute(0,2,1).contiguous().view(bs * p_num, c_num) 
        feat_t = feat_t.permute(0,2,1).contiguous().view(bs * p_num, c_num) 
        mask_t = []
        for i in range(self.nclasses):
            cur_mask = (label_t==i) 
            cur_mask = cur_mask.view(-1)
            cur_feat_s = feat_s[cur_mask]
            cur_feat_s = cur_feat_s.view(bs, -1, c_num)
            cur_feat_t = feat_t[cur_mask]
            cur_feat_t = cur_feat_t.view(bs, -1, c_num)
            index = torch.pe
            mask_t.append(label_t==i)
