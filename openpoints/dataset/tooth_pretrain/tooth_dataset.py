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
import cv2

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
class tooth_6000(data.Dataset):
    def __init__(self,
                 data_dir,
                 n_views,
                 num_points=16000,
                 split='train',
                 gravity_dim=2,
                 transform=None,
                 random_view=False
                 ):
        self.data_root = data_dir
        self.data_json = json.load(open(os.path.join(self.data_root, split + "_pca_0.5.json")))

        self.nviews = n_views
        self.num_points = num_points
        self.total_views = 12
        self.gravity_dim = gravity_dim
        self.transform = transform
        self.random_view = random_view

        self.rotation_matrixs_lower = self.get_rotation_matrix_tooth(-1/2 + 1/6)
        self.rotation_matrixs_upper = self.get_rotation_matrix_tooth(1/2 - 1/6)

        self.file_list = self.data_json['pc_data']
        self.rgb_dir = self.data_json['rgb_data']

        logging.info(f'[DATASET] {len(self.file_list)} instances were loaded')
        logging.info(f'[DATASET] {self.num_points} points were sampled')


    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def get_rotation_matrix(self):
        phi = -1/2 + 1/6
        theta = np.linspace(0, 2, self.total_views+1)
        v_theta, v_phi = np.meshgrid(theta[:self.total_views], phi)
        angles = np.stack([v_theta, v_phi], axis=-1).reshape(-1, 2)
        angles = torch.from_numpy(angles) * math.pi
        rotation_matrixs = rotate_theta_phi(angles)
        return rotation_matrixs
    
    def get_random_rotation_matrix(self):
        phi = np.random.rand() - 0.5
        theta = np.random.rand() * 2
        angles = np.array([[phi, theta]])
        angles = torch.from_numpy(angles) * math.pi
        rotation_matrixs = rotate_theta_phi(angles)
        return rotation_matrixs
    
    def get_rotation_matrix_tooth(self, phi):
        theta = np.linspace(0, 2, self.total_views+1)
        v_theta, v_phi = np.meshgrid(theta[:self.total_views], phi)
        angles = np.stack([v_theta, v_phi], axis=-1).reshape(-1, 2)
        angles = torch.from_numpy(angles) * math.pi
        rotation_matrixs = rotate_theta_phi(angles)
        return rotation_matrixs

    def __getitem__(self, idx):
        sample_path = self.file_list[idx]

        points = IO.get(sample_path).astype(np.float32)
        # points = points[:, [2, 0, 1]]
        points_norm = self.pc_norm(points).astype(np.float32)

        selected_idxs = np.random.choice(len(points_norm), self.num_points, replace=True)
        points_norm = points_norm[selected_idxs]

        data = {
            'pos': points_norm
        }

        if self.transform is not None:
            data = self.transform(data)

        name = sample_path.split("/")[-1]
        self.rotation_matrixs = self.rotation_matrixs_lower if 'lower' in name else self.rotation_matrixs_upper

        random_view = np.random.choice(self.total_views, self.nviews, replace=False)
        view_matrix = self.rotation_matrixs[random_view]
        image_list = []
        name = sample_path.split("/")[-1][:-4]
        for v in random_view:
            image_path = os.path.join(self.rgb_dir[idx], name + "_" + str(v) + ".png")

            image = Image.open(image_path).convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(2, 0, 1)
            # image = 1. - image
            image_list.append(torch.from_numpy(image))
        
        data['x'] = torch.cat((data['pos'],
                               torch.from_numpy(points_norm[:, self.gravity_dim:self.gravity_dim+1] - points_norm[:, self.gravity_dim:self.gravity_dim+1].min())), dim=1)
        
        # data['x'] = torch.cat((data['pos'],
        #                        torch.from_numpy(points_norm[:, 0:1] - points_norm[:, 0:1].min()),
        #                        torch.from_numpy(points_norm[:, 1:2] - points_norm[:, 1:2].min()),
        #                        torch.from_numpy(points_norm[:, 2:3] - points_norm[:, 2:3].min())
        #                        ), dim=1)
        

        # data['x'] = data['pos']
        
        if self.random_view:
            assert self.nviews == 1
            data['views'] = self.get_random_rotation_matrix()
        else:
            data['views'] = view_matrix
        data['imgs'] = torch.stack(image_list, dim=0)



        # data['x'] = torch.from_numpy(data['pos']).float()
        # data['pos'] = torch.from_numpy(data['pos']).float()

        return data

    def __len__(self):
        return len(self.file_list)
    

@DATASETS.register_module()
class tooth_6000_pca(data.Dataset):
    def __init__(self,
                 data_dir,
                 n_views,
                 num_points=16000,
                 split='train',
                 gravity_dim=2,
                 transform=None,
                 random_view=False
                 ):
        self.data_root = data_dir
        self.data_json = json.load(open(os.path.join(self.data_root, split + "_pca_cur_0.5.json")))

        self.nviews = n_views
        self.num_points = num_points
        self.total_views = 9
        self.gravity_dim = gravity_dim
        self.transform = transform
        self.random_view = random_view


        theta = [[0/6,    1/6,    2/6,   10/6,   11/6,   0/6,    0/6,    0/6,     0/6,  ]]
        phi = [[90/180, 90/180, 90/180, 90/180, 90/180, 30/180, 60/180, 120/180, 150/180]]
        v_theta = np.array(theta)
        v_phi = np.array(phi)
        self.rotation_matrixs = self.get_rotation_matrix_tooth(v_theta, v_phi)

        self.file_list = self.data_json['pc_data']
        self.rgb_dir = self.data_json['rgb_data']
        self.cur_list = self.data_json['cur_data']
        self.depth_list = self.data_json['depth_data']

        self.filter = True
        if self.filter:
            new_file_list = []
            new_rgb_dir = []
            new_cur_list = []
            for index, file_name in enumerate(self.file_list):
                case_name = file_name.split("/")[-2]
                tooth_name = file_name.split("/")[-1]
                filter_id = FILTER_ID_UPPER if 'upper' in tooth_name else FILTER_ID_LOWER
                case_id = int(case_name[4:])
                if case_id in filter_id:
                    continue
                new_file_list.append(self.file_list[index])
                new_rgb_dir.append(self.rgb_dir[index])
                new_cur_list.append(self.cur_list[index])

            self.file_list = new_file_list
            self.rgb_dir = new_rgb_dir
            self.cur_list = new_cur_list


        logging.info(f'[DATASET] {len(self.file_list)} instances were loaded')
        logging.info(f'[DATASET] {self.num_points} points were sampled')


    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def get_rotation_matrix(self):
        phi = -1/2 + 1/6
        theta = np.linspace(0, 2, self.total_views+1)
        v_theta, v_phi = np.meshgrid(theta[:self.total_views], phi)
        angles = np.stack([v_theta, v_phi], axis=-1).reshape(-1, 2)
        angles = torch.from_numpy(angles) * math.pi
        rotation_matrixs = rotate_theta_phi(angles)
        return rotation_matrixs
    
    def get_random_rotation_matrix(self):
        phi = np.random.rand() - 0.5
        theta = np.random.rand() * 2
        angles = np.array([[phi, theta]])
        angles = torch.from_numpy(angles) * math.pi
        rotation_matrixs = rotate_theta_phi(angles)
        return rotation_matrixs
    
    def get_rotation_matrix_tooth(self, v_theta, v_phi):
        angles = np.stack([v_theta, v_phi], axis=-1).reshape(-1, 2)
        angles = torch.from_numpy(angles) * math.pi
        rotation_matrixs = rotate_theta_phi(angles)
        return rotation_matrixs

    def __getitem__(self, idx):
        sample_path = self.file_list[idx]

        points = IO.get(sample_path).astype(np.float32)
        # points = points[:, [2, 0, 1]]
        points_norm = self.pc_norm(points).astype(np.float32)

        selected_idxs = np.random.choice(len(points_norm), self.num_points, replace=True)
        points_norm = points_norm[selected_idxs]

        data = {
            'pos': points_norm
        }

        if self.transform is not None:
            data = self.transform(data)


        random_view = np.random.choice(self.total_views, self.nviews, replace=False)
        view_matrix = self.rotation_matrixs[random_view]
        image_list = []
        # image_list1 = []
        # image_list2 = []
        # image_list3 = []
        name = sample_path.split("/")[-1][:-4]
        for v in random_view:
            image_path = os.path.join(self.rgb_dir[idx], name + "_" + str(v) + ".png")

            image = Image.open(image_path).convert("RGB")

            # img1 = image.resize((image.width // 8, image.height // 8), resample=Image.BILINEAR)
            # img1 = np.array(img1).astype(np.float32) / 255.0
            # img1 = img1.transpose(2, 0, 1)
            # image_list1.append(torch.from_numpy(img1))
            # img2 = image.resize((image.width // 4, image.height // 4), resample=Image.BILINEAR)
            # img2 = np.array(img2).astype(np.float32) / 255.0
            # img2 = img2.transpose(2, 0, 1)
            # image_list2.append(torch.from_numpy(img2))
            # img3 = image.resize((image.width // 2, image.height // 2), resample=Image.BILINEAR)
            # img3 = np.array(img3).astype(np.float32) / 255.0
            # img3 = img3.transpose(2, 0, 1)
            # image_list3.append(torch.from_numpy(img3))

            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(2, 0, 1)
            # image = 1. - image
            image_list.append(torch.from_numpy(image))
        
        # data['x'] = torch.cat((data['pos'],
        #                        torch.from_numpy(points_norm[:, self.gravity_dim:self.gravity_dim+1] - points_norm[:, self.gravity_dim:self.gravity_dim+1].min())), dim=1)
        
        # data['x'] = torch.cat((data['pos'],
        #                        torch.from_numpy(points_norm[:, 0:1] - points_norm[:, 0:1].min()),
        #                        torch.from_numpy(points_norm[:, 1:2] - points_norm[:, 1:2].min()),
        #                        torch.from_numpy(points_norm[:, 2:3] - points_norm[:, 2:3].min())
        #                        ), dim=1)
        

        data['x'] = data['pos']
        
        if self.random_view:
            assert self.nviews == 1
            data['views'] = self.get_random_rotation_matrix()
        else:
            data['views'] = view_matrix
        data['imgs'] = torch.stack(image_list, dim=0)
        # data['imgs_1'] = torch.stack(image_list1, dim=0)
        # data['imgs_2'] = torch.stack(image_list2, dim=0)
        # data['imgs_3'] = torch.stack(image_list3, dim=0)
        # data['imgs_4'] = torch.stack(image_list, dim=0)

        gg_list = []
        for v in random_view:
            depth_path = os.path.join(self.depth_list[idx], name + "_" + str(v) + ".npy")
            image_path = os.path.join(self.rgb_dir[idx], name + "_" + str(v) + ".png")
            depth = np.load(depth_path)
            gradient_x = np.gradient(depth)[1]
            gradient_y = np.gradient(depth)[0]
            gd = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
            img = cv2.imread(image_path, 0)
            sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
            sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
            gm = cv2.sqrt(sobelx ** 2 + sobely ** 2)

            gm = np.array(gm)
            gm = gm - gm.min()
            gm = gm / (gm.max() - gm.min())

            gg = gm
            gg = gg + 0.1
            gg[gg>1.] = 1.

            # gd = gd - gd.min()
            # gd = gd / (gd.max() - gd.min())

            # gg = gm - gd
            # gg = gg - gg.min()
            # gg = gg / (gg.max() - gg.min())

            gg_list.append(torch.from_numpy(gg).float())

        data['weight'] = torch.stack(gg_list, dim=0)

        return data

    def __len__(self):
        return len(self.file_list)
    


@DATASETS.register_module()
class TeethSegFinetuneDataset(data.Dataset):
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
        self.pc_path = os.path.join(data_root, 'scans')
        self.gt_path = os.path.join(data_root, 'gts')
        self.split = split
        
        if split=='train':
            self.data_list_file = os.path.join(self.data_root, f'full_{self.split}_finetune_0.1.txt')
        else:
            self.data_list_file = os.path.join(self.data_root, f'full_{self.split}_finetune.txt')
        
        self.sample_points_num = num_points

        print(f'[DATASET] sample out {self.sample_points_num} points')
        print(f'[DATASET] Open file {self.data_list_file}')

        # self.transform = transform
        # if self.transform is not None:
        #     print(f'[DATASET] Using data augementation')

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
        # load points and labels
        points = IO.get(os.path.join(self.pc_path, sample['mesh_id'], sample['file_path'])).astype(np.float32)
        cls = sample['location']
        labels = IO.get(os.path.join(self.gt_path, sample['mesh_id'], sample['file_path'].replace('obj', 'json')))['labels']
        labels = np.array([self.label2id[label] for label in labels]).astype(np.int32)

        # points = points[:, [2, 0, 1]]
        
        # normalization
        points_norm, center, scale = self.pc_norm(points)

        # random sample
        selected_idxs = np.random.choice(len(points_norm), self.sample_points_num, replace=True)
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

            # if self.transform is not None:
            #     data = self.transform(data)

            return data

    def __len__(self):
        return len(self.file_list)
    


@DATASETS.register_module()
class TeethClsDataset(data.Dataset):
    classes = [
        "lower",
        "upper",
    ]
    num_classes = 2

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
        self.pc_path = os.path.join(data_root, 'scans')
        self.gt_path = os.path.join(data_root, 'gts')
        self.split = split
        
        if split=='train':
            self.data_list_file = os.path.join(self.data_root, f'full_{self.split}_finetune.txt')
        else:
            self.data_list_file = os.path.join(self.data_root, f'full_{self.split}_finetune.txt')
        
        self.sample_points_num = num_points

        print(f'[DATASET] sample out {self.sample_points_num} points')
        print(f'[DATASET] Open file {self.data_list_file}')

        # self.transform = transform
        # if self.transform is not None:
        #     print(f'[DATASET] Using data augementation')

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

    @property
    def num_classes(self):
        return 2
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc
  
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        # load points and labels
        points = IO.get(os.path.join(self.pc_path, sample['mesh_id'], sample['file_path'])).astype(np.float32)
        cls = sample['location']
        labels = IO.get(os.path.join(self.gt_path, sample['mesh_id'], sample['file_path'].replace('obj', 'json')))['labels']
        labels = np.array([self.label2id[label] for label in labels]).astype(np.int32)

        points = points[:, [2, 0, 1]]
        # normalization
        points_norm= self.pc_norm(points)

        # random sample
        selected_idxs = np.random.choice(len(points_norm), self.sample_points_num, replace=True)
        sampled_points = points_norm[selected_idxs]
        sampled_labels = labels[selected_idxs]

        sampled_points = torch.from_numpy(sampled_points).float()
        sampled_labels = torch.from_numpy(sampled_labels).long()
        
        data = {'pos': sampled_points,
                'y': np.array([cls]).astype(int),
                }
        self.gravity_dim = 2
        data['x'] = torch.cat((data['pos'],
                            sampled_points[:, self.gravity_dim:self.gravity_dim+1] - sampled_points[:, self.gravity_dim:self.gravity_dim+1].min()), dim=1)
    
        # if self.transform is not None:
        #     data = self.transform(data)

        return data

    def __len__(self):
        return len(self.file_list)
