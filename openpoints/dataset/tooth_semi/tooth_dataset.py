import os
import math
import torch
import random
import numpy as np
import open3d as o3d
from PIL import Image
import torch.utils.data as data
from ..build import DATASETS
from ..data_util import rotate_theta_phi
import logging
import json
from ..io import IO
from copy import deepcopy

FILTER_ID_UPPER = \
            [23, 33, 49, 55, 133, 141, 191, 221, 269, 281, 303, 321, 328, 340, 520, 545, 560, 611, 671, 698, 743, 786, 858, 961, 1031, 1039,
            1054, 1068, 1116, 1121, 1125, 1149, 1150, 1162, 1206, 1207, 1271, 1339, 1360, 1361, 1362, 1379, 1390, 1392, 1410, 1411,
            1414, 1439, 1449, 1452, 1459, 1488, 1501, 1535, 1544, 1565, 1590, 1600, 1603, 1605, 1636, 1655, 1657, 1664, 1678, 1685,
            1689, 1690, 1692, 1696, 1710, 1742, 1747, 1749, 1756, 1770, 1773, 1774, 1779, 1786, 1794, 1795, 1820, 1833, 1882, 1897,
            1913, 1914, 1947, 1950, 1956, 1958, 1962, 1963, 1969, 1986, 2010, 2024, 2036, 2049, 2052, 2054, 2066, 2068, 2076, 2079,
            2088, 2093, 2098, 2108, 2114, 2121, 2155, 2165, 2177, 2186, 2188, 2192, 2215, 2216, 2225, 2236, 2245, 2250, 2252, 2274,
            2276, 2277, 2306, 2308, 2312, 2321, 2358, 2359, 2363, 2365, 2367, 2372, 2373, 2377, 2378, 2383, 2387, 2388, 2394, 2415,
            2420, 2423, 2428, 2446, 2454, 2458, 2462, 2470, 2481, 2484, 2491, 2503, 2504, 2511, 2513, 2521, 2527, 2559, 2574, 2581,
            2587, 2592, 2593, 2606, 2607, 2609, 2615, 2617, 2621, 2625, 2630, 2647, 2650, 2651, 2661, 2667, 2669, 2678, 2687, 2703,
            2706, 2712, 2713, 2715, 2724, 2726, 2731, 2733, 2749, 2751, 2756, 2760, 2769, 2774, 2796, 2801, 2810, 2980, 2981, 2982, 
            2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999,]
        
FILTER_ID_LOWER = \
            [19, 23, 33, 49, 55, 58, 133, 141, 191, 227, 234, 281, 303, 310, 311, 321, 328, 340, 380, 448, 451, 452, 476, 520, 545, 586, 611,
            671, 698, 743, 778, 786, 858, 889, 957, 961, 1031, 1039, 1043, 1054, 1068, 1078, 1121, 1129, 1149, 1207, 1254, 1271, 1339,
            1360, 1361, 1362, 1370, 1373, 1377, 1379, 1390, 1392, 1414, 1426, 1439, 1449, 1496, 1519, 1535, 1544, 1560, 1565, 1578,
            1584, 1585, 1590, 1600, 1636, 1638, 1655, 1657, 1659, 1663, 1673, 1685, 1689, 1690, 1693, 1696, 1699, 1711, 1713, 1719,
            1734, 1742, 1749, 1758, 1770, 1773, 1774, 1779, 1786, 1793, 1833, 1844, 1853, 1861, 1870, 1882, 1884, 1886, 1890, 1897,
            1911, 1939, 1943, 1950, 1955, 1956, 1958, 1962, 1963, 1969, 1973, 1999, 2010, 2049, 2052, 2054, 2059, 2079, 2084, 2088,
            2093, 2098, 2108, 2114, 2139, 2155, 2165, 2167, 2177, 2179, 2183, 2192, 2215, 2216, 2225, 2226, 2236, 2252, 2260, 2276,
            2290, 2306, 2308, 2312, 2321, 2326, 2334, 2355, 2359, 2363, 2365, 2367, 2373, 2375, 2378, 2382, 2383, 2387, 2389, 2415,
            2418, 2424, 2428, 2433, 2439, 2446, 2448, 2454, 2458, 2459, 2460, 2483, 2495, 2496, 2499, 2512, 2521, 2526, 2527, 2535,
            2536, 2538, 2549, 2560, 2562, 2564, 2576, 2577, 2601, 2603, 2604, 2608, 2613, 2617, 2624, 2627, 2633, 2638, 2641, 2657,
            2659, 2663, 2664, 2665, 2680, 2692, 2709, 2721, 2733, 2735, 2737, 2744, 2749, 2755, 2756, 2761, 2762, 2769, 2777, 2779,
            2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999,]

    
@DATASETS.register_module()
class TeethSegSemiLDataset(data.Dataset):
    def __init__(self, 
                 data_root='data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                 num_points=16000,
                 split='train',
                 class_choice=None,
                 use_normal=True,
                 shape_classes=16,
                 presample=False,
                 sampler='fps', 
                 transform=None,
                 multihead=False,
                 **kwargs
                 ):
        self.data_root = data_root
        with open(os.path.join(data_root,"data.json"), "r") as file:
            data = json.load(file)
        self.pc_path = data['scans']
        self.gt_path = data['gt']
        # self.pc_path = os.path.join(data_root, 'scans')
        # self.gt_path = os.path.join(data_root, 'gts')
        self.split = split
        
        if split=='train':
            self.data_list_file = os.path.join(self.data_root, f'semi_l_{self.split}_0.2.txt')
        else:
            # self.data_list_file = os.path.join(self.data_root, f'full_{self.split}_finetune.txt')
            self.data_list_file = os.path.join(self.data_root, f'testing.txt')

        self.sample_points_num = num_points

        print(f'[DATASET] sample out {self.sample_points_num} points')
        print(f'[DATASET] Open file {self.data_list_file}')

        self.transform = transform
        if self.transform is not None:
            print(f'[DATASET] Using data augementation')

        self.label2id = {0:0, \
                        11:1, 12:2, 13:3, 14:4, 15:5, 16:6, 17:7, 18:8, \
                        21:9, 22:10, 23:11, 24:12, 25:13, 26:14, 27:15, 28:16, \
                        31:1, 32:2, 33:3, 34:4, 35:5, 36:6, 37:7, 38:8, \
                        41:9, 42:10, 43:11, 44:12, 45:13, 46:14, 47:15, 48:16}
    
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        self.file_list = []

        for line in lines:
            line = line.strip()
            mesh_id = line.split('_')[0]
            location = line.split('_')[1].split('.')[0]
            location = 0 if location == 'lower' else 1
            self.file_list.append({
                'location': location,
                'mesh_id': mesh_id,
                'file_path': line
            })

        # self.file_list = self.file_list[:2]

        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc, centroid, m
  
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        idx_pc_path = self.pc_path[sample['file_path']]
        idx_gt_path = self.gt_path[sample['file_path']]
        # load points and labels
        # points = IO.get(os.path.join(self.pc_path, sample['mesh_id'], sample['file_path'])).astype(np.float32)
        points = IO.get(idx_pc_path).astype(np.float32)
        cls = sample['location']
        # labels = IO.get(os.path.join(self.gt_path, sample['mesh_id'], sample['file_path'].replace('obj', 'json')))['labels']
        labels = IO.get(idx_gt_path)['labels']
        labels = np.array([self.label2id[label] for label in labels]).astype(np.int32)

        # points = points[:, [2, 0, 1]]
        
        # normalization
        points_norm, center, scale = self.pc_norm(points)

        # random sample
        replace = False if len(points_norm) >= self.sample_points_num else True
        selected_idxs = np.random.choice(len(points_norm), self.sample_points_num, replace=replace)
        # selected_idxs = np.random.choice(len(points_norm), self.sample_points_num, replace=True)
        sampled_points = points_norm[selected_idxs]
        sampled_labels = labels[selected_idxs]

        sampled_points = torch.from_numpy(sampled_points).float()
        sampled_labels = torch.from_numpy(sampled_labels).long()

        # get the class weight
        class_weights = torch.zeros((17)).float()
        tmp, _ = torch.histogram(sampled_labels.float(), bins=17, range=(0., 17.))
        class_weights += tmp
        class_weights = class_weights / torch.sum(class_weights)
        class_weights = torch.where(torch.isinf(class_weights), torch.full_like(class_weights, 0), class_weights)

        # class_weights = torch.zeros((17)).float()
        # tmp, _ = torch.histogram(sampled_labels.float(), bins=16, range=(1., 17.))
        # class_weights[1:] += tmp
        # class_weights = class_weights / torch.sum(class_weights)
        # class_weights = torch.where(torch.isinf(class_weights), torch.full_like(class_weights, 0), class_weights)

        # cur = IO.get(os.path.join(self.cur_path, sample['mesh_id'], sample['file_path'][:-4]+'.npy')).astype(np.float32)
        # cur = np.abs(cur)
        # cur = (cur - cur.min()) / (cur.max() - cur.min())
        # curvatures = torch.from_numpy(cur).float()
        # curvatures = curvatures[selected_idxs]

        if self.split in ['val', 'test']:
            points = torch.from_numpy(points).float()
            labels = torch.from_numpy(labels).long()
            center, scale = torch.tensor(center).float(), torch.tensor(scale).float()
        
            data = {'pos': sampled_points,
                    'cls': np.array([cls]).astype(np.int64),
                    'y': sampled_labels}
            # self.gravity_dim = 2
            # data['x'] = torch.cat((data['pos'],
            #                    sampled_points[:, self.gravity_dim:self.gravity_dim+1] - sampled_points[:, self.gravity_dim:self.gravity_dim+1].min()), dim=1)

            data['x'] = data['pos']

            if self.transform is not None:
                data = self.transform(data) 

            data['points'] = points
            data['labels'] = labels
            data['center'] = center
            data['scale'] = scale
            data['class_weights'] = class_weights
            # data['curvatures'] = curvatures

            data['patient'] = sample['mesh_id']

            return data

        else:
            data = {'pos': sampled_points,
                    'cls': np.array([cls]).astype(np.int64),
                    'y': sampled_labels}
            # self.gravity_dim = 2
            # data['x'] = torch.cat((data['pos'],
            #                    sampled_points[:, self.gravity_dim:self.gravity_dim+1] - sampled_points[:, self.gravity_dim:self.gravity_dim+1].min()), dim=1)
       
            data['x'] = data['pos']

            data['class_weights'] = class_weights
            # data['curvatures'] = curvatures

            if self.transform is not None:
                data = self.transform(data)

            return data

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class TeethSegSemiUDataset(data.Dataset):
    def __init__(self, 
                 data_root='data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                 num_points=16000,
                 split='train',
                 class_choice=None,
                 use_normal=True,
                 shape_classes=16,
                 presample=False,
                 sampler='fps', 
                 transform_w=None,
                 transform_s=None,
                 multihead=False,
                 **kwargs
                 ):
        self.data_root = data_root
        with open(os.path.join(data_root,"data.json"), "r") as file:
            data = json.load(file)
        self.pc_path = data['scans']
        self.gt_path = data['gt']
        # self.pc_path = os.path.join(data_root, 'scans')
        # self.gt_path = os.path.join(data_root, 'gts')
        self.split = split
        # self.cur_path = os.path.join(data_root, 'curs')

        
        if split=='train':
            self.data_list_file = os.path.join(self.data_root, f'semi_u_{self.split}_0.2.txt')
        else:
            # self.data_list_file = os.path.join(self.data_root, f'full_{self.split}_finetune.txt')
            self.data_list_file = os.path.join(self.data_root, f'testing.txt')


        self.sample_points_num = num_points

        print(f'[DATASET] sample out {self.sample_points_num} points')
        print(f'[DATASET] Open file {self.data_list_file}')

        self.transform_w=transform_w
        self.transform_s=transform_s
        if self.transform_w is not None:
            print(f'[DATASET] Using data augementation')

        self.label2id = {0:0, \
                        11:1, 12:2, 13:3, 14:4, 15:5, 16:6, 17:7, 18:8, \
                        21:9, 22:10, 23:11, 24:12, 25:13, 26:14, 27:15, 28:16, \
                        31:1, 32:2, 33:3, 34:4, 35:5, 36:6, 37:7, 38:8, \
                        41:9, 42:10, 43:11, 44:12, 45:13, 46:14, 47:15, 48:16}
        
        self.unrelated_matrix = [
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
    
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        self.file_list = []

        for line in lines:
            line = line.strip()
            mesh_id = line.split('_')[0]
            location = line.split('_')[1].split('.')[0]
            location = 0 if location == 'lower' else 1
            self.file_list.append({
                'location': location,
                'mesh_id': mesh_id,
                'file_path': line
            })

        # self.file_list = self.file_list[:2]

        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc, centroid, m
  
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        idx_pc_path = self.pc_path[sample['file_path']]
        idx_gt_path = self.gt_path[sample['file_path']]
        # load points and labels
        # points = IO.get(os.path.join(self.pc_path, sample['mesh_id'], sample['file_path'])).astype(np.float32)
        points = IO.get(idx_pc_path).astype(np.float32)
        cls = sample['location']
        # labels = IO.get(os.path.join(self.gt_path, sample['mesh_id'], sample['file_path'].replace('obj', 'json')))['labels']
        labels = IO.get(idx_gt_path)['labels']
        labels = np.array([self.label2id[label] for label in labels]).astype(np.int32)

        # cur = IO.get(os.path.join(self.cur_path, sample['mesh_id'], sample['file_path'][:-4]+'.npy')).astype(np.float32)

        # points = points[:, [2, 0, 1]]
        
        # normalization
        points_norm, center, scale = self.pc_norm(points)


        # cur = np.abs(cur)
        # cur = (cur - cur.min()) / (cur.max() - cur.min())
        # cur_ratio = 0.3
        # cur_num = int(self.sample_points_num * cur_ratio)
        # # cur_topk = torch.from_numpy(cur)
        # # topk_value, topk_index = torch.topk(cur_topk, cur_num)
        # # topk_index = np.array(topk_index)
        # topk_index = np.random.choice(len(points_norm), cur_num, replace=False, p=((1 - cur)/(1 - cur).sum()))


        # random sample
        # selected_idxs = np.random.choice(len(points_norm), self.sample_points_num, replace=True)
        replace = False if len(points_norm) >= self.sample_points_num else True
        selected_idxs = np.random.choice(len(points_norm), self.sample_points_num, replace=replace)
        # selected_idxs = np.concatenate((topk_index, selected_idxs), axis=0)
        sampled_points = points_norm[selected_idxs]
        sampled_labels = labels[selected_idxs]

        sampled_points = torch.from_numpy(sampled_points).float()
        sampled_labels = torch.from_numpy(sampled_labels).long()

        # get the class weight
        class_weights = torch.zeros((17)).float()
        tmp, _ = torch.histogram(sampled_labels.float(), bins=17, range=(0., 17.))
        class_weights += tmp
        class_weights = class_weights / torch.sum(class_weights)
        class_weights = torch.where(torch.isinf(class_weights), torch.full_like(class_weights, 0), class_weights)

        # cur = np.abs(cur)
        # cur = (cur - cur.min()) / (cur.max() - cur.min())
        # curvatures = torch.from_numpy(cur).float()
        # curvatures = curvatures[selected_idxs]

        if self.split in ['val', 'test']:
            points = torch.from_numpy(points).float()
            labels = torch.from_numpy(labels).long()
            center, scale = torch.tensor(center).float(), torch.tensor(scale).float()
        
            data = {'pos': sampled_points,
                    'cls': np.array([cls]).astype(np.int64),
                    'y': sampled_labels}
            # self.gravity_dim = 2
            # data['x'] = torch.cat((data['pos'],
            #                    sampled_points[:, self.gravity_dim:self.gravity_dim+1] - sampled_points[:, self.gravity_dim:self.gravity_dim+1].min()), dim=1)

            data['x'] = data['pos']

            # if self.transform is not None:
            #     data = self.transform(data) 

            data['points'] = points
            data['labels'] = labels
            data['center'] = center
            data['scale'] = scale
            data['class_weights'] = class_weights
            # data['curvatures'] = curvatures


            return data

        else:
            data = {'pos': sampled_points,
                    'cls': np.array([cls]).astype(np.int64),
                    'y': sampled_labels}
            # self.gravity_dim = 2
            # data['x'] = torch.cat((data['pos'],
            #                    sampled_points[:, self.gravity_dim:self.gravity_dim+1] - sampled_points[:, self.gravity_dim:self.gravity_dim+1].min()), dim=1)
       
            data['x'] = data['pos']
            data['class_weights'] = class_weights
            # data['curvatures'] = curvatures

            data_0 = deepcopy(data)
            data_1 = deepcopy(data)
            data_w = self.transform_w(data_0)
            data_s = self.transform_s(data_1)
            w_keys = list(data_w.keys())
            s_keys = list(data_s.keys())
            for key in w_keys:
                value = data_w.pop(key)
                data.update({key+"_w": value})
            for key in s_keys:
                value = data_s.pop(key)
                data.update({key+"_s": value})

            data['raw_pos'] = sampled_points

            return data

    def __len__(self):
        return len(self.file_list)