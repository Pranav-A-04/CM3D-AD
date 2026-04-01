import argparse
import os
import sys
import torch
import numpy as np
import open3d as o3d
import time

from utils.misc import *
from models.autoencoder import ConsistencyAutoEncoder, AutoEncoder


def load_point_cloud(file_path, num_points=2048):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.array(pcd.points, dtype=np.float32)

    if len(points) > num_points:
        choice = np.random.choice(len(points), num_points, replace=False)
        points = points[choice]
    elif len(points) < num_points:
        choice = np.random.choice(len(points), num_points, replace=True)
        points = points[choice]

    # shape_bbox scaling as implemented in utils/dataset.py:
    pc_max = points.max(axis=0, keepdims=True)
    pc_min = points.min(axis=0, keepdims=True)
    center = ((pc_min + pc_max) / 2).reshape(-1)
    scale = (pc_max - pc_min).max() / 2
    points = (points - center) / scale

    points_tensor = torch.from_numpy(points).unsqueeze(0)  # [1, N, 3]

    scale_info = {
        'center': center,
        'scale': scale
    }

    return points_tensor, scale_info



def save_reconstructed_point_cloud(pc_tensor, scale_info, output_path):
    pc_np = pc_tensor[0].cpu().numpy() * scale_info['scale'] + scale_info['center']
    reconstructed_pcd = o3d.geometry.PointCloud()
    reconstructed_pcd.points = o3d.utility.Vector3dVector(pc_np)
    o3d.io.write_point_cloud(output_path, reconstructed_pcd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--model_type', type=str, required=True, choices=['consistency', 'diffusion'],
                        help='Model type to use for inference')
    parser.add_argument('--pointcloud', type=str, required=True, help='Path to input point cloud (.pcd)')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points to sample from input')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--flexibility', type=float, default=0.0)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_args = ckpt['args']
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Instantiate model based on explicit argument
    if args.model_type == 'consistency':
        model = ConsistencyAutoEncoder(ckpt_args).to(args.device)
        output_dir = 'output_consistency'
    elif args.model_type == 'diffusion':
        model = AutoEncoder(ckpt_args).to(args.device)
        output_dir = 'output_diffusion'
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    points, scale_info = load_point_cloud(args.pointcloud, args.num_points)
    points = points.to(args.device)

    start_time = time.time()

    with torch.no_grad():
        code = model.encode(points)
        trajectory = model.decode(points, code, points.size(1), flexibility=args.flexibility, ret_traj=False)

        # if hasattr(ckpt_args, 'rel') and ckpt_args.rel:
        # trajectory = trajectory.cpu() + points.cpu()

    inference_time = time.time() - start_time

    os.makedirs(output_dir, exist_ok=True)
    input_filename = os.path.basename(args.pointcloud).split('.')[0]
    output_path = os.path.join(output_dir, f'{input_filename}_reconstructed.pcd')

    save_reconstructed_point_cloud(trajectory, scale_info, output_path)
    print("For Object: ",input_filename)
    print(f"Inference time: {inference_time:.3f} seconds")
    print(f"Reconstructed point cloud saved to {output_path}")


if __name__ == "__main__":
    main()
