"""
Distributed training script for scene segmentation with S3DIS dataset
"""
import argparse
import yaml
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import logging
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
import torch.nn.functional as F
import warnings
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import defaultdict, Counter

torch.backends.cudnn.benchmark = False
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

from openpoints.models import build_model_from_cfg
from openpoints.models.layers import torch_grouping_operation, knn_point
from openpoints.loss import build_criterion_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_class_weights, get_features_by_keys, build_semi_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.models.layers import furthest_point_sample
from pointnet2_ops import pointnet2_utils as pt_utils
from utils.cluster_contrastloss import nativeContrastLoss_t
from utils.insT_loss import feature_space_loss, Idenyity_loss, threeD_space_loss


'''
python examples/segmentation/train.py --cfg cfgs/tooth_semi/transformer_finetune_fixmatch_ntm.yaml
'''

LABEL_PROJ = [0, 8, 7, 6, 5, 4, 3, 2, 1, 9, 10, 11, 12, 13, 14, 15, 16]

def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def part_seg_refinement(pred, pos, cls, cls2parts, n=10):
    pred_np = pred.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):  # sample_idx
        parts = cls2parts[cls[shape_idx]]
        counter_part = Counter(pred_np[shape_idx])
        if len(counter_part) > 1:
            for i in counter_part:
                if counter_part[i] < n or i not in parts:
                    less_idx = np.where(pred_np[shape_idx] == i)[0]
                    less_pos = pos[shape_idx][less_idx]
                    knn_idx = knn_point(n + 1, torch.unsqueeze(less_pos, axis=0),
                                        torch.unsqueeze(pos[shape_idx], axis=0))[1]
                    neighbor = torch_grouping_operation(pred[shape_idx:shape_idx + 1].unsqueeze(1), knn_idx)[0][0]
                    counts = batched_bincount(neighbor, 1, cls2parts[-1][-1] + 1)
                    counts[:, i] = 0
                    pred[shape_idx][less_idx] = counts.max(dim=1)[1]
    return pred


def get_ins_mious(pred, target, cls, cls2parts,
                  multihead=False,
                  ):
    """Get the Shape IoU
    shape IoU: the mean part iou for each shape
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
    Returns:
        _type_: _description_
    """
    ins_mious = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        parts = cls2parts[cls[shape_idx]]
        if multihead:
            parts = np.arange(len(parts))

        for part in parts:
            pred_part = pred[shape_idx] == part
            target_part = target[shape_idx] == part
            I = torch.logical_and(pred_part, target_part).sum()
            U = torch.logical_or(pred_part, target_part).sum()
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)
            part_ious.append(iou)
        ins_mious.append(torch.mean(torch.stack(part_ious)))
    return ins_mious


def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset_l.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir)
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.batch_size_val,
                                           cfg.dataset_l,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed,
                                           pretrain=False,
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}")
    # cfg.cls2parts = val_loader.dataset.cls2parts
    validate_fn = eval(cfg.get('val_fn', 'validate'))

    if cfg.model.get('decoder_args', False):
        cfg.model.decoder_args.cls2partembed = val_loader.dataset.cls2partembed
    if cfg.model.get('in_channels', None) is None:
        if cfg.model.get('encoder_args', None) is None:
            cfg.model.in_channels = 6
        else:
            cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).cuda()
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # transforms
    if 'vote' in cfg.datatransforms:
        voting_transform = build_transforms_from_cfg('vote', cfg.datatransforms)
    else:
        voting_transform = None

    model_module = model.module if hasattr(model, 'module') else model
    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            test_ins_miou, test_cls_miou, test_cls_mious = validate_fn(model, val_loader, cfg,
                                                                            num_votes=cfg.num_votes,
                                                                            data_transform=voting_transform
                                                                            )

            logging.info(f'\nresume val instance mIoU is {test_ins_miou}, val class mIoU is {test_cls_miou} \n ')
        else:
            if cfg.mode in ['val', 'test']:
                load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                test_ins_miou, test_cls_miou, test_cls_mious = validate_fn(model, val_loader, cfg,
                                                                            num_votes=cfg.num_votes,
                                                                            data_transform=voting_transform
                                                                            )
                return test_ins_miou
            elif cfg.mode == 'finetune':
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, pretrained_path=cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                logging.info(f'Load encoder only, finetuning from {cfg.pretrained_path}')
                load_checkpoint(model_module.encoder, pretrained_path=cfg.pretrained_path)
    else:
        logging.info('Training from scratch')

    # ---------------teacher-------------------
    model_t = build_model_from_cfg(cfg.model_t).cuda()
    if cfg.sync_bn:
        model_t = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_t)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model_t = nn.parallel.DistributedDataParallel(
            model_t.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    load_checkpoint(model_t, pretrained_path='')
    load_checkpoint(model, pretrained_path='')
   
    for p in model_t.parameters():
        p.requires_grad = False

    T_predictor = build_model_from_cfg(cfg.t_predictor).cuda()
    T_optimizer = build_optimizer_from_cfg(T_predictor, lr=cfg.lr, **cfg.optimizer)
    T_scheduler = build_scheduler_from_cfg(cfg, T_optimizer)

    
    test_loader = build_dataloader_from_cfg(cfg.batch_size_test,
                                           cfg.dataset_l,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='test',
                                           distributed=cfg.distributed,
                                           pretrain=False,
                                           )
    logging.info(f"length of test dataset: {len(test_loader.dataset)}")

    train_loader_l = build_dataloader_from_cfg(cfg.batch_size_l,
                                             cfg.dataset_l,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             pretrain=False,
                                             )
    logging.info(f"length of training dataset labeled: {len(train_loader_l.dataset)}")

    train_loader_u = build_semi_dataloader_from_cfg(cfg.batch_size_u,
                                             cfg.dataset_u,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             pretrain=False,
                                             )
    logging.info(f"length of training dataset unlabeled: {len(train_loader_u.dataset)}")

    if cfg.get('cls_weighed_loss', False):
        if hasattr(train_loader_l.dataset, 'num_per_class'):
            cfg.criterion_args.weight = None
            cfg.criterion_args.weight = get_class_weights(train_loader_l.dataset.num_per_class, normalize=True)
        else:
            logging.info('`num_per_class` attribute is not founded in dataset')
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    criterion_u = build_criterion_from_cfg(cfg.criterion_u_args).cuda()
    naive_nce_class = nativeContrastLoss_t()
    feat_S_loss = feature_space_loss(cfg.feat_k, cfg.feat_sigma, cfg.num_classes)
    identity_loss = Idenyity_loss()
    threed_loss = threeD_space_loss(cfg.threed_k, cfg.threed_sigma, cfg.num_classes)
    # ===> start training
    best_miou, best_dsc, best_acc = 0., 0., 0.
    whole_miou, whole_mdsc, whole_macc = 0., 0., 0.
    ema_t = torch.zeros((cfg.num_classes, cfg.num_classes)).cuda().scatter_(1, torch.arange(cfg.num_classes).view(-1, 1).cuda(), 1) 
    cm = cm_std = None
    cm = cal_mean_feature(model, train_loader_l, cfg)
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader_l.sampler.set_epoch(epoch)
        # some dataset sets the dataset length as a fixed steps.
        if hasattr(train_loader_l.dataset, 'epoch'):
            train_loader_l.dataset.epoch = epoch - 1
        cfg.epoch = epoch
        train_loss, train_loss_l, train_loss_u, th_percentage, \
             mean_pseudo_label_acc, mean_pseudo_label_acc_classwise, mean_th_meter_u_classwise, mean_th_meter_u_classwise_recall, \
                teacher_acc, student_acc, over_th_wobg, over_acc_wobg, ema_t, \
                     ema_t_corr, manifold_loss_feat, insT_identity_loss, insT_threed_loss = \
            train_one_epoch(model, train_loader_l, criterion, optimizer, scheduler, epoch, cfg, train_loader_u, criterion_u, 
                            naive_nce_class, ema_t, T_predictor, T_optimizer, T_scheduler, feat_S_loss, model_t, identity_loss,
                            threed_loss, cm)
        is_best = False
        if epoch % cfg.val_freq == 0 or epoch == cfg.epochs:
                    
            is_best = True
            best_epoch = epoch
            with np.printoptions(precision=2, suppress=True):
                logging.info(
                    f'Find a better ckpt @E{epoch}, val_miou {best_miou:.5f} val_dsc {best_dsc:.5f}, '
                    f'val_acc: {best_acc:.5f}')

        lr = optimizer.param_groups[0]['lr']

        with np.printoptions(precision=6, suppress=True):
            logging.info(f'Epoch {epoch} LR {lr:.6f} '
                        f'train_loss {train_loss:.5f}, train_loss_l {train_loss_l:.5f}, train_loss_u {train_loss_u:.5f}, th_percentage {th_percentage:.3f}')

        if writer is not None:
            writer.add_scalar('val_miou', whole_miou, epoch)
            writer.add_scalar('val_dsc', whole_mdsc, epoch)
            writer.add_scalar('val_acc', whole_macc, epoch)
            writer.add_scalar('best_val_miou',best_miou, epoch)
            writer.add_scalar('best_val_dsc',best_dsc, epoch)
            writer.add_scalar('best_val_acc',best_acc, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_loss_l', train_loss_l, epoch)
            writer.add_scalar('train_loss_u', train_loss_u, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('th_percentage', th_percentage, epoch)
            writer.add_scalar('train_over_th_acc', mean_pseudo_label_acc, epoch)
            writer.add_scalar('teacher_acc', teacher_acc, epoch)
            writer.add_scalar('student_acc', student_acc, epoch)
            writer.add_scalar('over_th_wobg', over_th_wobg, epoch)
            writer.add_scalar('over_acc_wobg', over_acc_wobg, epoch)
            writer.add_scalar('manifold_loss_feat', manifold_loss_feat, epoch)
            writer.add_scalar('insT_identity_loss', insT_identity_loss, epoch)
            writer.add_scalar('insT_threed_loss', insT_threed_loss, epoch)
            for ji in range(cfg.num_classes):
                writer.add_scalar('train_over_th_acc_class_'+str(ji), mean_pseudo_label_acc_classwise[ji], epoch)
                writer.add_scalar('train_over_th_num_class_'+str(ji), mean_th_meter_u_classwise[ji], epoch)
                writer.add_scalar('train_over_th_recall_class_'+str(ji), mean_th_meter_u_classwise_recall[ji], epoch)
            

        if cfg.sched_on_epoch:
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)

            
        if epoch % cfg.test_freq == 0 or epoch == cfg.epochs:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'miou': best_miou,
                                             'dsc': best_dsc,
                                             'acc': best_acc},
                            is_best=is_best
                            )
            
            logging.info(f"------------------ Start testing ------------------")
            load_checkpoint(model, pretrained_path=os.path.join(
            cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
            set_random_seed(cfg.seed)
            test_macc, test_miou, test_mdsc  = validate_fn(model, test_loader, cfg)
            with np.printoptions(precision=2, suppress=True):
                logging.info(f'---Testing---\nBest Epoch {best_epoch},'
                            f'Testing mIoU {test_miou:.5f}, '
                            f'Testing DSC {test_mdsc:.5f}, '
                            f'Testing ACC {test_macc:.5f}')

            if writer is not None:
                writer.add_scalar('test_miou', test_miou, epoch)
                writer.add_scalar('test_dsc', test_mdsc, epoch)
                writer.add_scalar('test_acc', test_macc, epoch)

            if cfg.get('num_votes', 0) > 0:
                load_checkpoint(model, pretrained_path=os.path.join(
                    cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
                set_random_seed(cfg.seed)
                vote_macc, vote_miou, vote_mdsc  = validate_fn(model, test_loader, cfg, num_votes=cfg.get('num_votes', 0),
                                            data_transform=voting_transform)
                with np.printoptions(precision=2, suppress=True):
                    logging.info(f'---Voting---\nBest Epoch {best_epoch},'
                                f'Voting mIoU {vote_miou:.5f}, '
                                f'Voting DSC {vote_mdsc:.5f}, '
                                f'Voting ACC {vote_macc:.5f}')

                if writer is not None:
                    writer.add_scalar('test_miou_voting', vote_miou, epoch)
                    writer.add_scalar('test_dsc_voting', vote_mdsc, epoch)
                    writer.add_scalar('test_acc_voting', vote_macc, epoch)


    ema_t = ema_t / torch.sum(ema_t, 1)
    ema_t_np = np.array(ema_t.cpu())
    ema_t_corr = ema_t_corr / torch.sum(ema_t_corr, 1)
    ema_t_corr_np = ema_t_corr.cpu().detach().numpy()
    with np.printoptions(precision=6, suppress=True):
        logging.info(ema_t_np)
        logging.info(ema_t_corr_np)

  
    with np.printoptions(precision=2, suppress=True):
        logging.info(
                     f'Best Epoch {best_epoch},'
                     f'mIoU {best_miou:.5f}, '
                     f'DSC {best_dsc:.5f}, '
                     f'ACC {best_acc:.5f}, '
                     f'Testing mIoU {test_miou:.5f}, '
                     f'Testing DSC {test_mdsc:.5f}, '
                     f'Testing ACC {test_macc:.5f}, '
                     )
        

    torch.cuda.synchronize()
    if writer is not None:
        writer.close()



def train_one_epoch(model, train_loader_l, criterion, optimizer, scheduler, epoch, cfg,
                    train_loader_u, criterion_u, naive_nce_class, ema_t, T_predictor, 
                    T_optimizer, T_scheduler, feat_S_loss, model_t, identity_loss,
                    threed_loss, cm):

    loss_meter = AverageMeter()
    loss_meter_l = AverageMeter()
    loss_meter_u = AverageMeter()
    th_meter_u = AverageMeter()
    th_meter_u_classwise = [AverageMeter() for _ in range(cfg.num_classes)]
    th_meter_u_classwise_recall = [AverageMeter() for _ in range(cfg.num_classes)]
    pseudo_label_acc = AverageMeter()
    pseudo_label_acc_classwise = [AverageMeter() for _ in range(cfg.num_classes)]
    model_t_acc = AverageMeter()
    model_s_acc = AverageMeter()
    th_meter_wobg = AverageMeter()
    th_acc_meter_wobg = AverageMeter()
    loss_meter_feat = AverageMeter()
    loss_meter_identity = AverageMeter()
    loss_meter_3d = AverageMeter()
    Identity_t = torch.zeros((cfg.num_classes, cfg.num_classes)).cuda().scatter_(1, torch.arange(cfg.num_classes).view(-1, 1).cuda(), 1) 
    model.train()  # set model to training mode
    T_predictor.train()

    pbar = tqdm(enumerate(train_loader_l), total=train_loader_l.__len__())
    num_iter = 0
    train_loader_u_iter = iter(train_loader_u)
    for idx, data in pbar:
        i_iter = epoch * len(train_loader_l) + idx # total iters till now
        if epoch <= cfg.supervised_epochs:
            num_iter += 1
            batch_size_l, num_point, _ = data['pos'].size()
            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)
            target = data['y']
            data['x'] = data['x'].transpose(1,2).contiguous()

            logits, _, _ = model(data)
            # logits = F.log_softmax(logits, dim=1)
            if cfg.criterion_args.NAME == 'Weight_CELoss':
                sup_loss = criterion(logits, target, data['class_weights'])
            elif cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
                sup_loss = criterion(logits, target)
            else:
                sup_loss = criterion(logits, target, data['cls'])

            unsup_loss = torch.tensor([0.]).cuda()


        else:
            # 1. generate pseudo labels
            p_threshold = cfg.threshold
            data_u = next(train_loader_u_iter)
            num_iter += 1
            batch_size_u, num_point, _ = data_u['pos_w'].size()
            for key in data_u.keys():
                data_u[key] = data_u[key].cuda(non_blocking=True)
            data_u['x_w'] = data_u['x_w'].transpose(1,2).contiguous()

            if epoch <= cfg.switch_ep:
                with torch.no_grad():
                    model_t.eval()
                    pred_u, _,_ = model_t(data_u, if_teacher=True)
                    # obtain pseudos
                    pred_u = F.softmax(pred_u, dim=1)
                    logits_u_aug, label_u_aug = torch.max(pred_u, dim=1)

            
            model.train()
            T_predictor.train()

            # 2. forward concate labeled + unlabeld into student networks
            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)
            target = data['y']
            data['x'] = data['x'].transpose(1,2).contiguous()
            data_u['x_s'] = data_u['x_s'].transpose(1,2).contiguous()

            data_u['T'] = ema_t

            pred_all, delta_T, sigma = model(data, u0=data_u, fixmatch=True)
            pred_l= pred_all[:cfg.batch_size_l]
            pred_u_strong = pred_all[cfg.batch_size_l:(cfg.batch_size_l+cfg.batch_size_u)]

            if not epoch <= cfg.switch_ep:
                pred_u = pred_all[(cfg.batch_size_l+cfg.batch_size_u):]
                pred_u = F.softmax(pred_u, dim=1)
                logits_u_aug, label_u_aug = torch.max(pred_u.detach(), dim=1)

    
            mask = None

            '''
            start estimating noisy transition matrix
            '''
            c = cfg.num_classes
            class_T = torch.empty((c, c)).cuda()
            eta_corr = pred_u.clone().detach()
            prior_T = torch.zeros((c, c)).cuda()
            for cc in range(c):
                if cfg.filter_outlier:
                    eta_thresh = eta_corr[:, cc, :].quantile(q=0.97)
                    robust_eta = eta_corr[:, cc, :]
                    robust_eta[robust_eta >= eta_thresh] = 0.0
                    robust_eta = robust_eta.contiguous().view(batch_size_u * num_point)
                    idx_best = torch.argmax(robust_eta)
                    idx_best0 = idx_best // num_point
                    idx_best1 = idx_best % num_point
                else:
                    robust_eta = eta_corr[:, cc, :]
                    robust_eta = robust_eta.contiguous().view(batch_size_u * num_point)
                    idx_best = torch.argmax(robust_eta)
                    idx_best0 = idx_best // num_point
                    idx_best1 = idx_best % num_point


                class_T[cc] = eta_corr[idx_best0, :, idx_best1]

                # generate gaussian distribution
                if cc==0: continue
                for cckk in range(c):
                    cc_proj = LABEL_PROJ[cc]
                    cckk_proj = LABEL_PROJ[cckk]
                    prior_T[cc, cckk] = gaussian(cckk_proj, cc_proj, sigma[cc])

            # V1
            prior_T[:, 0] = 0
            prior_T[0, 0] = 1
            prior_T = prior_T / torch.sum(prior_T, 1)
            new_T = cfg.geo_lambma * class_T + (1 - cfg.geo_lambma) * prior_T
            new_T[0] = class_T[0]
            new_T = new_T / torch.sum(new_T, 1)

            ema_t_grad = ema_t * cfg.ema_t_decay + new_T * (1 - cfg.ema_t_decay)
            ema_t_grad = ema_t_grad / torch.sum(ema_t_grad, 1)
            ema_t_corr = ema_t_grad #+ delta_T

            pred_u_strong_softmax = F.softmax(pred_u_strong, dim=1)
            insT = T_predictor(pred_u_strong_softmax.detach(), cm) 
            newT = cfg.lambma * ema_t_corr + (1 - cfg.lambma) * insT
            newT = F.normalize(newT, p=1, dim=2)
            pred_u_strong_corr = torch.bmm(pred_u_strong.permute(0,2,1).contiguous().view(-1, cfg.num_classes).unsqueeze(1), newT).squeeze(1)
            pred_u_strong_corr = pred_u_strong_corr.view(batch_size_u, num_point, cfg.num_classes).permute(0, 2, 1).contiguous()



            ema_t = ema_t * cfg.ema_t_decay + class_T * (1 - cfg.ema_t_decay)
            ema_t = ema_t / torch.sum(ema_t, 1)


            if cfg.use_feat_loss:
                manifold_loss_feat = feat_S_loss(pred_u_strong_softmax, label_u_aug, insT) * cfg.feat_loss_weight
            else:
                manifold_loss_feat = torch.tensor([0.])

            if cfg.use_identity_loss:
                insT_identity_loss = identity_loss(insT, Identity_t) * cfg.identity_loss_weight
            else:
                insT_identity_loss = torch.tensor([0.])

            if cfg.use_3d_loss:
                manifold_loss_3d = threed_loss(data_u['raw_pos'], label_u_aug, insT) * cfg.threed_loss_weight
            else:
                manifold_loss_3d = torch.tensor([0.])

            # 3. supervised loss
            if cfg.criterion_args.NAME == 'Weight_CELoss':
                sup_loss = criterion(pred_l, target, data['class_weights'])
            elif cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
                sup_loss = criterion(pred_l, target)
            else:
                sup_loss = criterion(pred_l, target, data['cls'])

            # 4. unsupervised loss
            if cfg.criterion_u_args.NAME == 'Weight_CELoss_U':
                # using labeled data weight here
                unsup_loss = criterion_u(pred_u_strong, label_u_aug.clone().detach(), data['class_weights'],
                        logits_u_aug.clone().detach(), thresh=p_threshold)
            elif cfg.criterion_u_args.NAME == 'Poly1FocalLoss_U':
                unsup_loss = criterion_u(pred_u_strong, label_u_aug.detach(),
                        logits_u_aug.detach(), thresh=p_threshold, mask=mask)
            elif cfg.criterion_u_args.NAME == 'Poly1FocalLoss_U_corr':
                unsup_loss = criterion_u(pred_u_strong_corr, label_u_aug.detach(),
                        logits_u_aug.detach(), thresh=p_threshold, mask=mask)
            elif cfg.criterion_u_args.NAME == 'Poly1FocalLoss_U_T':
                unsup_loss = criterion_u(pred_u_strong, label_u_aug.detach(),
                        logits_u_aug.detach(), ema_t, pred_u_strong_corr, thresh=p_threshold, mask=mask)
          

            thresh_mask = logits_u_aug.ge(torch.tensor(p_threshold)).bool()

            scale_factor = (cfg.batch_size_u * num_point) / thresh_mask.sum()
            unsup_loss *= (cfg.unsupervised_loss_weight*scale_factor)
            # unsup_loss *= cfg.unsupervised_loss_weight


            
            thresh_mask = logits_u_aug.ge(torch.tensor(p_threshold)).bool()
            over_th = torch.sum(thresh_mask) / (thresh_mask.shape[0] * thresh_mask.shape[1]) * 100

            teacher_acc = torch.sum(label_u_aug==data_u['y'].squeeze(-1)) / (label_u_aug.shape[0]*label_u_aug.shape[1])
            pred_u_strong = F.softmax(pred_u_strong, dim=1)
            logits_u_strong, label_u_strong = torch.max(pred_u_strong, dim=1)
            student_acc = torch.sum(label_u_strong==data_u['y'].squeeze(-1)) / (label_u_strong.shape[0]*label_u_strong.shape[1])


            pseudo_label = label_u_aug.clone().detach()
            target_u = data_u['y'].squeeze(-1)
            denominator = torch.sum(thresh_mask)
            pseudo_label_overall_acc = 0 if denominator==0 else torch.sum((pseudo_label==target_u)*thresh_mask)/denominator * 100
            pseudo_label_acc_classwise_list = []
            over_th_classwise =[]
            over_th_recall_classwise = []
            for ii in range(cfg.num_classes):
                cur_pred = (pseudo_label==ii).float()
                cur_gt = (target_u==ii).float()
                denominator = (torch.sum(cur_pred * thresh_mask.float()))
                cur_acc = 0 if denominator==0 else (torch.sum((cur_pred * cur_gt) *thresh_mask.float()) / denominator).item() * 100
                pseudo_label_acc_classwise_list.append(cur_acc)

                denominator = torch.sum(cur_pred)
                over_th_num = 0 if denominator==0 else (torch.sum(cur_pred * thresh_mask.float()) / denominator).item() * 100
                over_th_classwise.append(over_th_num)

                denominator = torch.sum(cur_gt)
                over_th_recall_num = 0 if denominator==0 else (torch.sum((cur_pred * cur_gt) *thresh_mask.float()) / denominator).item()  * 100
                over_th_recall_classwise.append(over_th_recall_num)

            
            cur_pred = (pseudo_label>0).float()
            cur_gt = (target_u>0).float()
            over_th_wobg = torch.sum(thresh_mask*cur_pred) / (torch.sum(cur_pred)) * 100
            total_acc = torch.sum(((pseudo_label==target_u)*cur_pred)*(thresh_mask))
            denominator = (torch.sum(cur_pred * thresh_mask.float()))
            over_acc_wobg = 0 if denominator==0 else ((total_acc) / denominator) * 100

        loss = sup_loss + unsup_loss

        if cfg.use_feat_loss:
            loss += manifold_loss_feat

        if cfg.use_identity_loss:
            loss += insT_identity_loss

        if cfg.use_3d_loss:
            loss += manifold_loss_3d

        loss.backward()

        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            T_optimizer.step()
            T_optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)
                T_scheduler.step(epoch)
        

        loss_meter.update(loss.item(), n=(cfg.batch_size_l+cfg.batch_size_u))
        loss_meter_l.update(sup_loss.item(), n=cfg.batch_size_l)
        loss_meter_u.update(unsup_loss.item(), n=cfg.batch_size_u)
        if epoch > cfg.supervised_epochs:
            th_meter_u.update(over_th, n=cfg.batch_size_u)

            loss_meter_feat.update(manifold_loss_feat.item(), n=cfg.batch_size_u)
            loss_meter_identity.update(insT_identity_loss.item(), n=cfg.batch_size_u)
            loss_meter_3d.update(manifold_loss_3d.item(), n=cfg.batch_size_u)

            model_t_acc.update(teacher_acc, n=cfg.batch_size_u)
            model_s_acc.update(student_acc, n=cfg.batch_size_u)

            th_meter_wobg.update(over_th_wobg, n=cfg.batch_size_u)
            th_acc_meter_wobg.update(over_acc_wobg, n=cfg.batch_size_u)

            pseudo_label_acc.update(pseudo_label_overall_acc, n=cfg.batch_size_u)
            for jj in range(cfg.num_classes):
                pseudo_label_acc_classwise[jj].update(pseudo_label_acc_classwise_list[jj], n=cfg.batch_size_u)
                th_meter_u_classwise[jj].update(over_th_classwise[jj], n=cfg.batch_size_u)
                th_meter_u_classwise_recall[jj].update(over_th_recall_classwise[jj], n=cfg.batch_size_u)

        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.avg:.5f} "
                                 f"Loss_sup {loss_meter_l.avg:.5f} "
                                 f"Loss_unsup {loss_meter_u.avg:.5f} "
                                 )
    train_loss = loss_meter.avg
    train_loss_l = loss_meter_l.avg
    train_loss_u = loss_meter_u.avg
    th_percentage = th_meter_u.avg

    mean_pseudo_label_acc = pseudo_label_acc.avg
    mean_pseudo_label_acc_classwise = [pseudo_label_acc_classwise[ij].avg for ij in range(cfg.num_classes)]
    mean_th_meter_u_classwise = [th_meter_u_classwise[ij].avg for ij in range(cfg.num_classes)]
    mean_th_meter_u_classwise_recall = [th_meter_u_classwise_recall[ij].avg for ij in range(cfg.num_classes)]

    return train_loss, train_loss_l, train_loss_u, th_percentage, \
            mean_pseudo_label_acc, mean_pseudo_label_acc_classwise, mean_th_meter_u_classwise, mean_th_meter_u_classwise_recall,\
            model_t_acc.avg, model_s_acc.avg, th_meter_wobg.avg, \
            th_acc_meter_wobg.avg, ema_t, ema_t_corr, loss_meter_feat.avg, loss_meter_identity.avg, loss_meter_3d.avg


@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=0, data_transform=None):
    model.eval()  # set model to eval mode
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    mandible_metric = {
        'miou': [],
        'dsc': [],
        'acc': [],
    }
    maxillary_metric  = {
        'miou': [],
        'dsc': [],
        'acc': [],
    }

    for idx, data in pbar:
        for key in data.keys():
            if isinstance(data[key], torch.Tensor): 
                data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        cls = data['cls']
        # data['x'] = get_features_by_keys(data, cfg.feature_keys)
        data['x'] = data['x'].transpose(1,2).contiguous()
        batch_size, num_point, _ = data['pos'].size()
      

        logits, _, _ = model(data)

        preds_whole = get_pred_whole(logits, data['pos'], data['points'], data['center'], data['scale'])
        acc_list, miou_list, mdsc_list = get_seg_metrics(preds_whole, data['labels'])

        for ii in range(len(acc_list)):
            if cls[ii] == 0:
                mandible_metric['miou'].append(miou_list[ii])
                mandible_metric['dsc'].append(mdsc_list[ii])
                mandible_metric['acc'].append(acc_list[ii])
            else:
                maxillary_metric['miou'].append(miou_list[ii])
                maxillary_metric['dsc'].append(mdsc_list[ii])
                maxillary_metric['acc'].append(acc_list[ii])
    
    mandible_macc, mandible_miou, mandible_mdsc = np.array(mandible_metric['acc']).mean(),\
        np.array(mandible_metric['miou']).mean(), np.array(mandible_metric['dsc']).mean()
    maxillary_macc, maxillary_miou, maxillary_mdsc = np.array(maxillary_metric['acc']).mean(),\
        np.array(maxillary_metric['miou']).mean(), np.array(maxillary_metric['dsc']).mean()
    whole_macc, whole_miou, whole_mdsc = (np.array(mandible_metric['acc']).sum() + np.array(maxillary_metric['acc']).sum()) / (len(mandible_metric['acc']) + len(maxillary_metric['acc'])), \
        (np.array(mandible_metric['miou']).sum() + np.array(maxillary_metric['miou']).sum()) / (len(mandible_metric['miou']) + len(maxillary_metric['miou'])), \
        (np.array(mandible_metric['dsc']).sum() + np.array(maxillary_metric['dsc']).sum()) / (len(mandible_metric['dsc']) + len(maxillary_metric['dsc'])),

    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Test Epoch [{cfg.epoch}/{cfg.epochs}],'
                    f'Mandible mIoU {mandible_miou:.5f}, '
                    f'Mandible DSC {mandible_mdsc:.5f}, '
                    f'Mandible ACC {mandible_macc:.5f}')
        logging.info(f'Test Epoch [{cfg.epoch}/{cfg.epochs}],'
                    f'Maxillary mIoU {maxillary_miou:.5f}, '
                    f'Maxillary DSC {maxillary_mdsc:.5f}, '
                    f'Maxillary ACC {maxillary_macc:.5f}')
        logging.info(f'Test Epoch [{cfg.epoch}/{cfg.epochs}],'
                    f'mIoU {whole_miou:.5f}, '
                    f'DSC {whole_mdsc:.5f}, '
                    f'ACC {whole_macc:.5f}')  

    return whole_macc, whole_miou, whole_mdsc

def get_pred_whole(logits, points, points_whole, center, scale):
    logits = F.softmax(logits, dim=1)  # B C N
    preds_whole = []
    for index in range(logits.shape[0]):
        logit = logits[index].unsqueeze(0).contiguous()
        point = points[index].unsqueeze(0).contiguous()
        s = scale[index].cuda().unsqueeze(0).contiguous()
        c = center[index].cuda().unsqueeze(0).contiguous()
        point_whole = points_whole[index].cuda().unsqueeze(0).contiguous()
        point = point * s + c

        dist, idx = pt_utils.three_nn(point_whole, point)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        logit_whole = pt_utils.three_interpolate(logit, idx, weight)
        pred_whole = logit_whole.argmax(dim=1)
        preds_whole.append(pred_whole)
    
    return preds_whole

def get_seg_metrics(preds_whole, labels_whole):
    acc_list, miou_list, mdsc_list = [], [], []
    for index in range(len(preds_whole)):
        pred_whole = preds_whole[index].detach()
        label_whole = labels_whole[index].cuda().unsqueeze(0)

        pred_whole = pred_whole.squeeze().cpu()
        label_whole = label_whole.squeeze().cpu()

        unq_labels = torch.unique(label_whole).cpu().numpy()
        acc, iou, dsc = [], [], []

        for jcls in unq_labels:
            if jcls == 0:
                continue

            jcls_and = torch.logical_and(pred_whole==jcls, label_whole==jcls).sum()
            jcls_or = torch.logical_or(pred_whole==jcls, label_whole==jcls).sum()

            iou.append((jcls_and / jcls_or).float())
            dsc.append((2*iou[-1] / (1 + iou[-1])))

        acc = (pred_whole == label_whole).sum() / (label_whole.view(-1).shape[0])
        miou = np.array(iou).mean()
        mdsc = np.array(dsc).mean()

        acc_list.append(acc)
        miou_list.append(miou)
        mdsc_list.append(mdsc)

    return acc_list, miou_list, mdsc_list


def gaussian(x, μ, s):
   return (1 / (s * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-((x - μ) ** 2) / (2 * s** 2))

def gaussian_batch(x, μ, s):
   return (1 / (s * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-((x - μ) ** 2) / (2 * s** 2))

def cal_confusion_matrix(model, train_loader_l, cfg):
    pbar = tqdm(enumerate(train_loader_l), total=train_loader_l.__len__())
    c = cfg.num_classes
    cm = np.zeros((c, c))
    for idx, data in pbar:
        batch_size_l, num_point, _ = data['pos'].size()
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        data['x'] = data['x'].transpose(1,2).contiguous()

        with torch.no_grad():
            model.eval()
            logits, _, _ = model(data)
            pred_logits, pred_label = torch.max(logits, dim=1)

            target = target.view(-1)
            pred_label = pred_label.view(-1)
            target = np.array(target.cpu())
            pred_label = np.array(pred_label.cpu())
            cm += confusion_matrix(target, pred_label, labels=np.arange(c))

    cm = cm / (np.sum(cm, 1) + 0.001)


    return cm

def cal_mean_feature(model, train_loader_l, cfg):
    pbar = tqdm(enumerate(train_loader_l), total=train_loader_l.__len__())
    c = cfg.num_classes
    cm = torch.zeros((c, c)).cuda()
    c_num = torch.zeros((c,)).cuda()
    for idx, data in pbar:
        batch_size_l, num_point, _ = data['pos'].size()
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        data['x'] = data['x'].transpose(1,2).contiguous()

        with torch.no_grad():
            model.eval()
            logits, _, _ = model(data)
            logits = torch.softmax(logits, dim=1)

            logits = logits.permute(0, 2, 1).contiguous().view(batch_size_l*num_point, c)
            target = target.view(-1)

            for kk in range(c):
                cur_num = torch.sum(target==kk)
                if cur_num == 0:
                    continue
                cur_feats = logits[target]
                mean_feats = cur_feats.mean(0)
                cm[kk] = (cm[kk] * c_num[kk] + mean_feats * cur_num) / (c_num[kk] + cur_num)
                c_num[kk] += cur_num

    return cm.to(torch.float32)



       


if __name__ == "__main__":
    parser = argparse.ArgumentParser('ShapeNetPart Part segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)
    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # logger
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name 
        f'train',
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)

    cfg.root_dir =  './log'

    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']

    if cfg.mode in ['resume', 'test', 'val']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg)
