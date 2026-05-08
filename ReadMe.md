# Two Steps Are All You Need: Efficient 3D Point Cloud Anomaly Detection with Consistency Models

[![arXiv](https://img.shields.io/badge/arXiv-2605.05372-b31b1b.svg)](https://arxiv.org/abs/2605.05372)
[![Paper License](https://img.shields.io/badge/Paper%20License-CC%20BY%204.0-lightgreen.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🚀 Architecture Overview

![Architecture Diagram](/result_images/arch_diag.png)

- **Input**: 3D point cloud with surface anomalies synthesized using Patch-Gen
- **Encoder**: PointNet/PointNet++ backbone to encode into latent space  
- **CM**: Consistency Model trained using self-supervised consistency training  
- **Output**: Reconstructed (anomaly-free) point cloud  
- **Detection**: Chamfer/EMD-based anomaly scoring via input–output deviation


## 📖 Abstract

Diffusion models are rapidly redefining 3D anomaly detection in point cloud data. As 3D sensing becomes integral to modern manufacturing, reliable anomaly detection is essential for high-throughput quality assurance and process control. Yet practical deployment on resource-constrained, latency-critical systems remains limited. Existing methods are often computationally prohibitive or unreliable in complex, unmasked regions, and diffusion pipelines are inherently bottlenecked by iterative denoising. In this work, we address this bottleneck by reformulating reconstructionbased anomaly detection through consistency learning, enabling direct prediction of anomaly-free geometry in one or two network evaluations. We further introduce a novel hybrid loss formulation that explicitly enforces reconstruction toward clean data. This design substantially reduces inference cost, achieving up to 80× faster runtime than the current state-of-the-art method, without GPU acceleration, while preserving strong detection performance. It outperforms R3D-AD on Anomaly-ShapeNet with 76.20% I-AUROC and remains competitive on Real3DAD with 72.80% I-AUROC, enabling efficient, low-latency anomaly detection on resource-constrained platforms, including drones, smart industrial cameras, and other edge devices.

## ⚙️ Training

Create a virtual environment:
```
python -m venv myenv
```

Install the required packages:
```
pip install -r requirements.txt
```

Install the following CUDA Ops packages:
```
pip install "git+https://github.com/unlimblue/KNN_CUDA.git#egg=knn_cuda&subdirectory=."
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

## 🚀 Run Training
```
python train_test.py PATH_TO_CONFIG
```

## 🔍 Inference
```
python inference.py --checkpoint path/to/ckpt --model_type {consistency/diffusion} --pointcloud path/to/pcd --num_points <INPUT_PCD_NUM_POINTS>
```

## Acknowledgement
Thanks to the following open-sourced repos:

1. R3D-AD
2. OpenAI Consistency Models

## 🖊️ Citations
If you find this work useful in your research, please consider citing:
```bibtex
@misc{a2026stepsneedefficient3d,
      title={Two Steps Are All You Need: Efficient 3D Point Cloud Anomaly Detection with Consistency Models}, 
      author={Pranav A and Shashank B and Pranav Siddappa and Dominik Seuss and Minal Moharir and Subramanya KN},
      year={2026},
      eprint={2605.05372},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.05372}, 
}
```