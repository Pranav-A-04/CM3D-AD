import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm.auto import tqdm
import open3d as o3d
import pathlib
import glob
import numpy as np

from utils.util import normalize, random_rorate, random_patch, random_translate

all_shapenetad_cates = ['ashtray0', 'headset0']
all_reald3ad_cates = ['shell', 'starfish', 'airplane']

class ShapeNetAD(Dataset):
    
    def __init__(self, path, cates, split, scale_mode=None, num_points=2048, num_aug=4, transforms=list()):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        if 'all' in cates:
            self.cates = all_shapenetad_cates
        else:
            self.cates = cates
        self.split = split
        self.scale_mode = scale_mode
        self.num_points = num_points
        self.num_aug = num_aug
        self.transforms = transforms

        self.pointclouds = []
        self.stats = None

        #self.get_statistics()
        self.load()

    def get_statistics(self):

        stats_dir = os.path.join(self.path, '../shapenet-ad' + '_stats/')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cates) == len(all_shapenetad_cates):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cates) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        pointclouds = []
        for cate in self.cates:
            for split in ('train', 'test'):
                local_path = os.path.join(self.path, cate, split)
                for f in os.listdir(local_path):
                    local_file = os.path.join(local_path, f)
                    pcd = o3d.io.read_point_cloud(local_file)
                    pointcloud = np.array(pcd.points, np.float32)
                    choice = np.random.choice(len(pointcloud), self.num_points, False)
                    pointcloud = torch.from_numpy(pointcloud[choice])
                    if self.scale_mode is not None:
                        # For global_unit, we have a chicken-and-egg problem.
                        # Usually global_unit relies on PRE-COMPUTED raw stats. 
                        # But for shape_bbox (your case), scaling is per-instance.
                        if self.scale_mode == 'shape_bbox':
                            pc_max, _ = pointcloud.max(dim=0, keepdim=True)
                            pc_min, _ = pointcloud.min(dim=0, keepdim=True)
                            shift = ((pc_min + pc_max) / 2).view(1, 3)
                            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
                            pointcloud = (pointcloud - shift) / scale
                    pointclouds.append(pointcloud)

        all_points = torch.stack(pointclouds, dim=0) # (B, N, 3)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def scale(self, pc):
        if self.scale_mode == 'global_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = self.stats['std'].reshape(1, 1)
        elif self.scale_mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif self.scale_mode == 'shape_half':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.5)
        elif self.scale_mode == 'shape_34':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.75)
        elif self.scale_mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        else:
            shift = torch.zeros([1, 3])
            scale = torch.ones([1, 1])

        pc = (pc - shift) / scale
        
        return pc, shift, scale

    def append(self, pc, pc_raw, cate, pc_id, mask, label, shift, scale):
        # pc, shift, scale = self.scale(pc)
        # pc_raw, _, _ = self.scale(pc_raw)
        self.pointclouds.append({
            'pointcloud': pc,  # augmented version
            'pointcloud_raw': pc_raw,  # original before any patching or masking
            'cate': cate,
            'id': pc_id,
            'shift': shift,
            'scale': scale,
            'mask': mask,
            'label': label,
        })
    def save_point_cloud(self, pc_tensor, scale_info, output_path):
        pc_np = pc_tensor.cpu().numpy() * scale_info['scale'] + scale_info['center']
        reconstructed_pcd = o3d.geometry.PointCloud()
        reconstructed_pcd.points = o3d.utility.Vector3dVector(pc_np)
        o3d.io.write_point_cloud(output_path, reconstructed_pcd)
    def load(self):

        for cate in self.cates:
            if self.split == 'train':
                local_path = os.path.join(self.path, cate, 'train')
                tpls = []
                for f in os.listdir(local_path):
                    local_file = os.path.join(local_path, f)
                    pcd = o3d.io.read_point_cloud(local_file)
                    pointcloud = np.array(pcd.points, dtype=np.float32)
                    tpls.append(pointcloud)
                for pc_id in tqdm(range(self.num_aug), 'Augment'):
                    if self.num_aug == len(tpls):
                        pointcloud = tpls[pc_id]
                    else:
                        pointcloud = random.choice(tpls)
                    pc_tensor = torch.from_numpy(pointcloud)
                    pc_norm, shift, scale = self.scale(pc_tensor)
                    # convert back to numpy for augmentation
                    pointcloud = pc_norm.numpy()
                    # downsample to num_points
                    choice = np.random.choice(len(pointcloud), self.num_points, False)
                    pointcloud = pointcloud[choice]
                    # p = p.r
                    pointcloud = random_rorate(pointcloud)
                    if random.random() < 0.8:

                        #pc = torch.from_numpy(pointcloud)
                        #mask = torch.zeros(self.num_points)
                        patch_num = len(pointcloud) // 32
                        patch_scale = 0.2
                        pointcloud_aug, mask_aug = random_patch(pointcloud, int(patch_num), patch_scale)
                        #choice = np.random.choice(len(pointcloud_aug), self.num_points, False)
                        # tensorification
                        pc = torch.from_numpy(pointcloud)
                        pc_aug = torch.from_numpy(pointcloud_aug)
                        # self.save_point_cloud(pc_aug, {'center': shift.cpu().numpy(), 'scale': scale.cpu().numpy()}, f'aug_{cate}_{pc_id}.pcd')
                        # return
                        mask = torch.from_numpy(mask_aug)
                        label = 1
                        self.append(pc_aug, pc, cate, pc_id, mask, label, shift, scale)
                    else:
                        pc = torch.from_numpy(pointcloud)
                        mask = torch.zeros(self.num_points)
                        label = 0
                        self.append(pc, pc, cate, pc_id, mask, label, shift, scale)

            elif self.split == 'test':
                local_path = os.path.join(self.path, cate, 'test')
                for pc_id, f in enumerate(os.listdir(local_path)):
                    if 'positive' in f:
                        local_file = os.path.join(local_path, f)
                        pcd = o3d.io.read_point_cloud(local_file)
                        pointcloud = np.array(pcd.points, dtype=np.float32)
                        pointcloud_tensor = torch.from_numpy(pointcloud)
                        pc_norm, shift, scale = self.scale(pointcloud_tensor)
                        pointcloud = pc_norm.numpy()
                        choice = np.random.choice(len(pointcloud), self.num_points, False)
                        pc = torch.from_numpy(pointcloud[choice])
                        mask = torch.zeros(pc.shape[0])
                        label = 0
                    else:
                        local_file = os.path.join(local_path.replace('test', 'GT'), f.replace('pcd', 'txt'))
                        pointcloud_mask = np.genfromtxt(local_file, dtype=np.float32, delimiter=",")
                        pointcloud_tensor = torch.from_numpy(pointcloud_mask[:, :3])
                        mask = pointcloud_mask[:, 3]
                        pc_norm, shift, scale = self.scale(pointcloud_tensor)
                        pointcloud = pc_norm.numpy()
                        choice = np.random.choice(len(pointcloud), self.num_points, False)
                        pc = torch.from_numpy(pointcloud[choice])
                        mask = torch.from_numpy(mask[choice])
                        label = 1
                    self.append(pc, pc, cate, pc_id, mask, label, shift, scale)

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        for transform in self.transforms:
            data = transform(data)
        return data

class Real3DAD(Dataset):
    def __init__(
        self, 
        path, 
        cates, 
        split, 
        scale_mode=None, 
        num_points=4096, 
        num_aug=4, 
        transforms=None
    ):
        
        super().__init__()
            
        assert isinstance(cates, list), '`cates` must be a list of category names.'
        assert split in ('train', 'test'), '`split` must be "train" or "test"'
        assert scale_mode is None or scale_mode in (
            'global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34'
        )
        
        self.path = path
        self.cates = cates
        self.split = split
        self.scale_mode = scale_mode
        self.num_points = num_points
        self.num_aug = num_aug
        self.transforms = transforms if transforms is not None else []
        
        self.pointclouds = []
        self.stats = None
        
        # Optionally compute statistics for global scaling
        #if scale_mode == 'global_unit':
        # self.get_statistics()
        self.load()
    
    def get_statistics(self):

        stats_dir = os.path.join(self.path, '../real3d-ad' + '_stats/')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cates) == len(all_reald3ad_cates):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cates) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        pointclouds = []
        for cate in self.cates:
            for split in ('train', 'test'):
                local_path = os.path.join(self.path, cate, split)
                for f in os.listdir(local_path):
                    local_file = os.path.join(local_path, f)
                    pcd = o3d.io.read_point_cloud(local_file)
                    pointcloud = np.array(pcd.points, np.float32)
                    choice = np.random.choice(len(pointcloud), self.num_points, False)
                    pointcloud = torch.from_numpy(pointcloud[choice])
                    if self.scale_mode is not None:
                        # For global_unit, we have a chicken-and-egg problem.
                        # Usually global_unit relies on PRE-COMPUTED raw stats. 
                        # But for shape_bbox (your case), scaling is per-instance.
                        if self.scale_mode == 'shape_bbox':
                            pc_max, _ = pointcloud.max(dim=0, keepdim=True)
                            pc_min, _ = pointcloud.min(dim=0, keepdim=True)
                            shift = ((pc_min + pc_max) / 2).view(1, 3)
                            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
                            pointcloud = (pointcloud - shift) / scale
                    pointclouds.append(pointcloud)

        all_points = torch.stack(pointclouds, dim=0) # (B, N, 3)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def scale(self, pc):
        if self.scale_mode == 'global_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = self.stats['std'].reshape(1, 1)
        elif self.scale_mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif self.scale_mode == 'shape_half':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.5)
        elif self.scale_mode == 'shape_34':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.75)
        elif self.scale_mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        else:
            shift = torch.zeros([1, 3])
            scale = torch.ones([1, 1])

        pc = (pc - shift) / scale
        
        return pc, shift, scale

    def append(self, pc, pc_raw, cate, pc_id, mask, label, shift, scale):
        # pc, shift, scale = self.scale(pc)
        # pc_raw, _, _ = self.scale(pc_raw)
        self.pointclouds.append({
            'pointcloud': pc,  # augmented version
            'pointcloud_raw': pc_raw,  # original before any patching or masking
            'cate': cate,
            'id': pc_id,
            'shift': shift,
            'scale': scale,
            'mask': mask,
            'label': label,
        })
    
    def load(self):
        for cate in self.cates:
            if self.split == 'train':
                local_path = os.path.join(self.path, cate, 'train')
                tpls = []
                for f in os.listdir(local_path):
                    local_file = os.path.join(local_path, f)
                    pcd = o3d.io.read_point_cloud(local_file)
                    pointcloud = np.array(pcd.points, dtype=np.float32)
                    tpls.append(pointcloud)
                for pc_id in tqdm(range(self.num_aug), 'Augment'):
                    if self.num_aug == len(tpls):
                        pointcloud = tpls[pc_id]
                    else:
                        pointcloud = random.choice(tpls)
                    pc_tensor = torch.from_numpy(pointcloud)
                    pc_norm, shift, scale = self.scale(pc_tensor)
                    # convert back to numpy for augmentation
                    pointcloud = pc_norm.numpy()
                    # downsample to num_points
                    choice = np.random.choice(len(pointcloud), self.num_points, False)
                    pointcloud = pointcloud[choice]
                    # p = p.r
                    pointcloud = random_rorate(pointcloud)
                    if random.random() < 0.8:

                        #pc = torch.from_numpy(pointcloud)
                        #mask = torch.zeros(self.num_points)
                        patch_num = len(pointcloud) // 32
                        patch_scale = 0.2
                        pointcloud_aug, mask_aug = random_patch(pointcloud, int(patch_num), patch_scale)
                        #choice = np.random.choice(len(pointcloud_aug), self.num_points, False)
                        # tensorification
                        pc = torch.from_numpy(pointcloud)
                        pc_aug = torch.from_numpy(pointcloud_aug)
                        # self.save_point_cloud(pc_aug, {'center': shift.cpu().numpy(), 'scale': scale.cpu().numpy()}, f'aug_{cate}_{pc_id}.pcd')
                        # return
                        mask = torch.from_numpy(mask_aug)
                        label = 1
                        self.append(pc_aug, pc, cate, pc_id, mask, label, shift, scale)
                    else:
                        pc = torch.from_numpy(pointcloud)
                        mask = torch.zeros(self.num_points)
                        label = 0
                        self.append(pc, pc, cate, pc_id, mask, label, shift, scale)

            elif self.split == 'test':
                local_path = os.path.join(self.path, cate, 'test')
                for pc_id, f in enumerate(os.listdir(local_path)):
                    if 'good' in f:
                        local_file = os.path.join(local_path, f)
                        pcd = o3d.io.read_point_cloud(local_file)
                        pointcloud = np.array(pcd.points, dtype=np.float32)
                        pointcloud_tensor = torch.from_numpy(pointcloud)
                        pc_norm, shift, scale = self.scale(pointcloud_tensor)
                        pointcloud = pc_norm.numpy()
                        choice = np.random.choice(len(pointcloud), self.num_points, False)
                        pc = torch.from_numpy(pointcloud[choice])
                        mask = torch.zeros(pc.shape[0])
                        label = 0
                    else:
                        local_file = os.path.join(local_path.replace('test', 'gt'), f.replace('pcd', 'txt'))
                        pointcloud_mask = np.genfromtxt(local_file, dtype=np.float32, delimiter="")
                        pointcloud_tensor = torch.from_numpy(pointcloud_mask[:, :3])
                        mask = pointcloud_mask[:, 3]
                        pc_norm, shift, scale = self.scale(pointcloud_tensor)
                        pointcloud = pc_norm.numpy()
                        choice = np.random.choice(len(pointcloud), self.num_points, False)
                        pc = torch.from_numpy(pointcloud[choice])
                        mask = torch.from_numpy(mask[choice])
                        label = 1
                    self.append(pc, pc, cate, pc_id, mask, label, shift, scale)

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        for transform in self.transforms:
            data = transform(data)
        return data
