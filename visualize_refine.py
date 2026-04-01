import argparse
import os
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

def refine_point_cloud(anom_path, recon_path, output_path, num_points):
    # Check if files exist
    if not os.path.exists(anom_path):
        raise FileNotFoundError(f"Anomalous PCD not found: {anom_path}")
    if not os.path.exists(recon_path):
        raise FileNotFoundError(f"Reconstructed PCD not found: {recon_path}")

    print(f"Loading Anomalous PCD: {anom_path}")
    print(f"Loading Reconstructed PCD: {recon_path}")

    # 1. Load point clouds
    pcd_anom = o3d.io.read_point_cloud(anom_path) 
    pcd_recon = o3d.io.read_point_cloud(recon_path) 

    pts_anom = np.asarray(pcd_anom.points)
    pts_recon = np.asarray(pcd_recon.points)

    # Safety check: ensure we don't try to sample more points than exist
    actual_num_points = min(num_points, len(pts_anom))
    print(f"Sampling {actual_num_points} points from the anomalous cloud...")
    
    choice = np.random.choice(len(pts_anom), actual_num_points, replace=False)
    pts_anom = pts_anom[choice]

    # 2. Fit KDTree on the RECONSTRUCTED points
    print("Fitting KDTree on reconstructed points...")
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pts_recon)

    # 3. Query with ANOMALOUS points to find where they should move
    print("Calculating displacements...")
    distances, indices = nn.kneighbors(pts_anom)
    distances = distances.flatten()
    indices = indices.flatten()

    # Target coordinates on the reconstructed surface
    matched_recon_pts = pts_recon[indices] 

    # Calculate per-point displacement vectors: Anomaly -> Reconstruction
    disp = matched_recon_pts - pts_anom 

    # 4. The Thresholding Magic
    threshold = np.mean(distances) + np.std(distances)
    print(f"Calculated distance threshold: {threshold:.4f}")

    # Create a boolean mask of anomalous points
    anomaly_mask = distances > threshold
    print(f"Identified {np.sum(anomaly_mask)} anomalous points to move.")

    # 5. Create the hybrid point cloud
    pts_new = np.copy(pts_anom)

    # Apply the displacement ONLY to the points in the anomaly mask
    pts_new[anomaly_mask] = pts_anom[anomaly_mask] + disp[anomaly_mask]

    # 6. Save and Visualize
    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(pts_new)

    # Paint the points we moved RED so you can visually verify the mask worked
    colors = np.zeros((len(pts_new), 3)) # Initialize black
    colors[~anomaly_mask] = [0.5, 0.5, 0.5] # Normal points = Grey
    colors[anomaly_mask] = [1.0, 0.0, 0.0]  # Moved anomaly points = Red
    pcd_final.colors = o3d.utility.Vector3dVector(colors)

    print(f"Saving refined point cloud to: {output_path}")
    o3d.io.write_point_cloud(output_path, pcd_final)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine an anomalous point cloud using a reconstructed reference.")
    
    # Required arguments
    parser.add_argument("-a", "--anom", required=True, type=str, help="Path to the input anomalous point cloud (.pcd)")
    parser.add_argument("-r", "--recon", required=True, type=str, help="Path to the reconstructed/reference point cloud (.pcd)")
    
    # Optional arguments
    parser.add_argument("-o", "--output", default="refined_output.pcd", type=str, help="Path to save the output point cloud (default: refined_output.pcd)")
    parser.add_argument("-n", "--num_points", default=2048, type=int, help="Number of points to sample (default: 2048)")

    args = parser.parse_args()

    refine_point_cloud(
        anom_path=args.anom,
        recon_path=args.recon,
        output_path=args.output,
        num_points=args.num_points
    )