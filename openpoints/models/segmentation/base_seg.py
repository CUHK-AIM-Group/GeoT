"""
Author: PointNeXt
"""
import os
import copy
from typing import List
import torch
import torch.nn as nn
import logging
from ...utils import get_missing_parameters_message, get_unexpected_parameters_message
from ..build import MODELS, build_model_from_cfg
from ..layers import create_linearblock, create_convblock1d


@MODELS.register_module()
class BaseSeg(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 decoder_args=None,
                 cls_args=None,
                 **kwargs):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)
        if decoder_args is not None:
            decoder_args_merged_with_encoder = copy.deepcopy(encoder_args)
            decoder_args_merged_with_encoder.update(decoder_args)
            decoder_args_merged_with_encoder.encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder,
                                                                                                         'channel_list') else None
            self.decoder = build_model_from_cfg(decoder_args_merged_with_encoder)
        else:
            self.decoder = None

        if cls_args is not None:
            if hasattr(self.decoder, 'out_channels'):
                in_channels = self.decoder.out_channels
            elif hasattr(self.encoder, 'out_channels'):
                in_channels = self.encoder.out_channels
            else:
                in_channels = cls_args.get('in_channels', None)
            cls_args.in_channels = in_channels
            self.head = build_model_from_cfg(cls_args)
        else:
            self.head = None

    def forward(self, data):
        p, f = self.encoder.forward_seg_feat(data)
        if self.decoder is not None:
            f = self.decoder(p, f).squeeze(-1)
        if self.head is not None:
            f = self.head(f)
        return f


@MODELS.register_module()
class BasePartSeg(BaseSeg):
    def __init__(self, encoder_args=None, decoder_args=None, cls_args=None, **kwargs):
        super().__init__(encoder_args, decoder_args, cls_args, **kwargs)

    def forward(self, p0, f0=None, cls0=None):
        if hasattr(p0, 'keys'):
            p0, f0, cls0 = p0['pos'], p0['x'], p0['cls']
        else:
            if f0 is None:
                f0 = p0.transpose(1, 2).contiguous()
        p, f = self.encoder.forward_seg_feat(p0, f0)
        if self.decoder is not None:
            f = self.decoder(p, f, cls0).squeeze(-1)
        elif isinstance(f, list):
            f = f[-1]
        if self.head is not None:
            f = self.head(f)
        return f
    

@MODELS.register_module()
class WholePartSeg(nn.Module):
    def __init__(self, segmentor_args=None, gm_args = None, **kwargs):
        super().__init__()
        self.segmentor = build_model_from_cfg(segmentor_args)
        # self.load_pretrain(segmentor_args.pretrained_path)

        # self.grapgmatch = build_model_from_cfg(gm_args)

    def load_pretrain(self, pretrained_path):
        # if not os.path.exists(pretrained_path):
        #     raise NotImplementedError('no checkpoint file from path %s...' % pretrained_path)
        # # load state dict
        # state_dict = torch.load(pretrained_path, map_location='cpu')['model']
        # state_dict_segmentor = {}
        # for key in state_dict.keys():
        #     if key == 'embedding.net.0.weight':
        #         continue
        #     state_dict_segmentor['segmentor.'+key] = state_dict[key]
        # incompatible = self.load_state_dict(state_dict_segmentor, strict=False)
        # print(f'load pretrained weights from {pretrained_path} successfully!')

        if not os.path.exists(pretrained_path):
            raise NotImplementedError('no checkpoint file from path %s...' % pretrained_path)
        # load state dict
        state_dict = torch.load(pretrained_path, map_location='cpu')['model']
        state_dict_segmentor = {}
        for key in state_dict.keys():
            # if key == 'encoder.embedding.net.0.weight':
            #     continue
            preflex = key.split('.')[0]
            part_key = key[len(preflex)+1:]
            state_dict_segmentor['segmentor.'+part_key] = state_dict[key]
        incompatible = self.load_state_dict(state_dict_segmentor, strict=False)
        if incompatible.missing_keys:
            logging.info('missing_keys')
            logging.info(
                get_missing_parameters_message(incompatible.missing_keys),
            )
        else:
            logging.info('No missing_keys')
        if incompatible.unexpected_keys:
            logging.info('unexpected_keys')
            logging.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )
        else:
            logging.info('No unexpected_keys')
        print(f'load pretrained weights from {pretrained_path} successfully!')

    def forward(self, p0, f0=None, cls0=None, u0=None, if_teacher=False, fixmatch=False):
        if if_teacher:
            p0, f0, cls0 = p0['pos_w'].detach(), p0['x_w'].detach(), p0['cls_w'].detach()
        elif hasattr(p0, 'keys'):
            if u0 is not None:
                if fixmatch:
                    l_p0, l_f0, l_cls0 = p0['pos'], p0['x'], p0['cls']
                    u_p0, u_f0, u_cls0 = u0['pos_s'], u0['x_s'], u0['cls_s']
                    u_p1, u_f1, u_cls1 = u0['pos_w'], u0['x_w'], u0['cls_w']
                    p0 = torch.cat((l_p0,u_p0,u_p1),0)
                    f0 = torch.cat((l_f0,u_f0,u_f1),0)
                    cls0 = torch.cat((l_cls0,u_cls0,u_cls1),0)
                else:
                    l_p0, l_f0, l_cls0 = p0['pos'], p0['x'], p0['cls']
                    u_p0, u_f0, u_cls0 = u0['pos_s'], u0['x_s'], u0['cls_s']
                    p0 = torch.cat((l_p0,u_p0),0)
                    f0 = torch.cat((l_f0,u_f0),0)
                    cls0 = torch.cat((l_cls0,u_cls0),0)
            else:
                p0, f0, cls0 = p0['pos'], p0['x'], p0['cls']
        else:
            if f0 is None:
                f0 = p0.transpose(1, 2).contiguous()
        
        if u0 is None:
            T = None
        else:
            if 'T' in u0.keys():
                T = u0['T']
            else:
                T = None

        f, p, s,_ = self.segmentor(p0, f0, cls0, T)


        return f, p, s
    

@MODELS.register_module()
class WholePartSeg_ntm(nn.Module):
    def __init__(self, segmentor_args=None, gm_args = None, **kwargs):
        super().__init__()
        self.segmentor = build_model_from_cfg(segmentor_args)
        # self.load_pretrain(segmentor_args.pretrained_path)


    def load_pretrain(self, pretrained_path):
        # if not os.path.exists(pretrained_path):
        #     raise NotImplementedError('no checkpoint file from path %s...' % pretrained_path)
        # # load state dict
        # state_dict = torch.load(pretrained_path, map_location='cpu')['model']
        # state_dict_segmentor = {}
        # for key in state_dict.keys():
        #     if key == 'embedding.net.0.weight':
        #         continue
        #     state_dict_segmentor['segmentor.'+key] = state_dict[key]
        # incompatible = self.load_state_dict(state_dict_segmentor, strict=False)
        # print(f'load pretrained weights from {pretrained_path} successfully!')

        if not os.path.exists(pretrained_path):
            raise NotImplementedError('no checkpoint file from path %s...' % pretrained_path)
        # load state dict
        state_dict = torch.load(pretrained_path, map_location='cpu')['model']
        state_dict_segmentor = {}
        for key in state_dict.keys():
            # if key == 'encoder.embedding.net.0.weight':
            #     continue
            preflex = key.split('.')[0]
            part_key = key[len(preflex)+1:]
            state_dict_segmentor['segmentor.'+part_key] = state_dict[key]
        incompatible = self.load_state_dict(state_dict_segmentor, strict=False)
        if incompatible.missing_keys:
            logging.info('missing_keys')
            logging.info(
                get_missing_parameters_message(incompatible.missing_keys),
            )
        else:
            logging.info('No missing_keys')
        if incompatible.unexpected_keys:
            logging.info('unexpected_keys')
            logging.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )
        else:
            logging.info('No unexpected_keys')
        print(f'load pretrained weights from {pretrained_path} successfully!')

    def forward(self, p0, f0=None, cls0=None, u0=None, if_teacher=False, fixmatch=False):
        if if_teacher:
            p0, f0, cls0 = p0['pos_w'].detach(), p0['x_w'].detach(), p0['cls_w'].detach()
        elif hasattr(p0, 'keys'):
            if u0 is not None:
                if fixmatch:
                    l_p0, l_f0, l_cls0 = p0['pos'], p0['x'], p0['cls']
                    u_p0, u_f0, u_cls0 = u0['pos_s'], u0['x_s'], u0['cls_s']
                    u_p1, u_f1, u_cls1 = u0['pos_w'], u0['x_w'], u0['cls_w']
                    p0 = torch.cat((l_p0,u_p0,u_p1),0)
                    f0 = torch.cat((l_f0,u_f0,u_f1),0)
                    cls0 = torch.cat((l_cls0,u_cls0,u_cls1),0)
                else:
                    l_p0, l_f0, l_cls0 = p0['pos'], p0['x'], p0['cls']
                    u_p0, u_f0, u_cls0 = u0['pos_s'], u0['x_s'], u0['cls_s']
                    p0 = torch.cat((l_p0,u_p0),0)
                    f0 = torch.cat((l_f0,u_f0),0)
                    cls0 = torch.cat((l_cls0,u_cls0),0)
            else:
                p0, f0, cls0 = p0['pos'], p0['x'], p0['cls']
        else:
            if f0 is None:
                f0 = p0.transpose(1, 2).contiguous()
        

        f, p, s,_ = self.segmentor(p0, f0, cls0)


        return f, p, s
    

@MODELS.register_module()
class Ins_T(nn.Module):
    def __init__(self, T_args=None, **kwargs):
        super().__init__()
        self.T_predictor = build_model_from_cfg(T_args)

    def forward(self, clean):
        ins_t = self.T_predictor(clean)

        return ins_t
    
@MODELS.register_module()
class Ins_T_mean(nn.Module):
    def __init__(self, T_args=None, **kwargs):
        super().__init__()
        self.T_predictor = build_model_from_cfg(T_args)

    def forward(self, clean, cm):
        ins_t = self.T_predictor(clean, cm)

        return ins_t


@MODELS.register_module()
class VariableSeg(BaseSeg):
    def __init__(self,
                 encoder_args=None,
                 decoder_args=None,
                 cls_args=None,
                 **kwargs):
        super().__init__(encoder_args, decoder_args, cls_args)
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

    def forward(self, data):
        p, f, b = self.encoder.forward_seg_feat(data)
        f = self.decoder(p, f, b).squeeze(-1)
        return self.head(f)


@MODELS.register_module()
class SegHead(nn.Module):
    def __init__(self,
                 num_classes, in_channels,
                 mlps=None,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 dropout=0.5,
                 global_feat=None, 
                 **kwargs
                 ):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
            global_feat: global features to concat. [max,avg]. Set to None if do not concat any.
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        if global_feat is not None:
            self.global_feat = global_feat.split(',')
            multiplier = len(self.global_feat) + 1
        else:
            self.global_feat = None
            multiplier = 1
        in_channels *= multiplier
        
        if mlps is None:
            mlps = [in_channels, in_channels] + [num_classes]
        else:
            if not isinstance(mlps, List):
                mlps = [mlps]
            mlps = [in_channels] + mlps + [num_classes]
        heads = []
        for i in range(len(mlps) - 2):
            heads.append(create_convblock1d(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))

        heads.append(create_convblock1d(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points):
        if self.global_feat is not None: 
            global_feats = [] 
            for feat_type in self.global_feat:
                if 'max' in feat_type:
                    global_feats.append(torch.max(end_points, dim=-1, keepdim=True)[0])
                elif feat_type in ['avg', 'mean']:
                    global_feats.append(torch.mean(end_points, dim=-1, keepdim=True))
            global_feats = torch.cat(global_feats, dim=1).expand(-1, -1, end_points.shape[-1])
            end_points = torch.cat((end_points, global_feats), dim=1)
        logits = self.head(end_points)
        return logits


@MODELS.register_module()
class VariableSegHead(nn.Module):
    def __init__(self,
                 num_classes, in_channels,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 dropout=0.5,
                 **kwargs
                 ):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        mlps = [in_channels, in_channels] + [num_classes]

        heads = []
        print(mlps, norm_args, act_args)
        for i in range(len(mlps) - 2):
            heads.append(create_linearblock(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))

        heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points):
        logits = self.head(end_points)
        return logits

@MODELS.register_module()
class MultiSegHead(nn.Module):
    def __init__(self,
                 num_classes, in_channels,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 dropout=0,
                 shape_classes=16,
                 num_parts=[4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3],
                 **kwargs
                 ):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        mlps = [in_channels, in_channels] + [shape_classes]
        self.multi_shape_heads = []

        self.num_parts=num_parts
        print(mlps, norm_args, act_args)
        self.shape_classes = shape_classes
        self.multi_shape_heads = nn.ModuleList()
        for i in range(shape_classes):
            head=[]
            for j in range(len(mlps) - 2):

                head.append(create_convblock1d(mlps[j], mlps[j + 1],
                                                norm_args=norm_args,
                                                act_args=act_args))
                if dropout:
                    head.append(nn.Dropout(dropout))
                head.append(nn.Conv1d(mlps[-2], num_parts[i], kernel_size=1, bias=True))
            self.multi_shape_heads.append(nn.Sequential(*head))

        # heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))

    def forward(self, end_points):
        logits_all_shapes = []
        for i in range(self.shape_classes):# per 16 shapes
            logits_all_shapes.append(self.multi_shape_heads[i](end_points))
        # logits = self.head(end_points)
        return logits_all_shapes


# # TODO: add distill for segmentation
# @MODELS.register_module()
# class DistillBaseSeg(BaseSeg):
#     def __init__(self,
#                  encoder_args=None,
#                  decoder_args=None,
#                  cls_args=None,
#                  distill_args=None, 
#                  criterion_args=None, 
#                  **kwargs):
#         super().__init__()
#         self.encoder = build_model_from_cfg(encoder_args)
#         if decoder_args is not None:
#             decoder_args_merged_with_encoder = copy.deepcopy(encoder_args)
#             decoder_args_merged_with_encoder.update(decoder_args)
#             decoder_args_merged_with_encoder.encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder,
#                                                                                                          'channel_list') else None
#             self.decoder = build_model_from_cfg(decoder_args_merged_with_encoder)
#         else:
#             self.decoder = None

#         if cls_args is not None:
#             if hasattr(self.decoder, 'out_channels'):
#                 in_channels = self.decoder.out_channels
#             elif hasattr(self.encoder, 'out_channels'):
#                 in_channels = self.encoder.out_channels
#             else:
#                 in_channels = cls_args.get('in_channels', None)
#             cls_args.in_channels = in_channels
#             self.head = build_model_from_cfg(cls_args)
#         else:
#             self.head = None

#     def forward(self, data):
#         p, f = self.encoder.forward_seg_feat(data)
#         if self.decoder is not None:
#             f = self.decoder(p, f).squeeze(-1)
#         if self.head is not None:
#             f = self.head(f)
#         return f