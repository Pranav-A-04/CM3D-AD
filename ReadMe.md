# Two Steps Are All You Need: Efficient 3D Point Cloud Anomaly Detection with Consistency Models

---

## Abstract

Diffusion models are rapidly redefining 3D anomaly detection in point cloud data. As 3D sensing becomes integral to modern manufacturing, reliable anomaly detection is essential for high-throughput quality assurance and process control. Yet practical deployment on resource-constrained, latency-critical systems remains limited. Existing methods are often computationally prohibitive or unreliable in complex, unmasked regions, and diffusion pipelines are inherently bottlenecked by iterative denoising. In this work, we address this bottleneck by reformulating reconstructionbased anomaly detection through consistency learning, enabling direct prediction of anomaly-free geometry in one or two network evaluations. We further introduce a novel hybrid loss formulation that explicitly enforces reconstruction toward clean data. This design substantially reduces inference cost, achieving up to 80× faster runtime than the current state-of-the-art method, without GPU acceleration, while preserving strong detection performance. It outperforms R3D-AD on Anomaly-ShapeNet with 76.20% I-AUROC and remains competitive on Real3DAD with 72.80% I-AUROC, enabling efficient, low-latency anomaly detection on resource-constrained platforms, including drones, smart industrial cameras, and other edge devices.

## Architecture Overview

![Architecture Diagram](/result_images/arch_diag.png)

- **Input**: 3D point cloud with surface anomalies synthesized using Patch-Gen
- **Encoder**: PointNet/PointNet++ backbone to encode into latent space  
- **CM**: Consistency Model trained using self-supervised consistency training  
- **Output**: Reconstructed (anomaly-free) point cloud  
- **Detection**: Chamfer/EMD-based anomaly scoring via input–output deviation

## Citations
@article{song2023consistency,<br/>
  title={Consistency Models},<br/>
  author={Song, Yang and Meng, Chenlin and Ermon, Stefano},<br/>
  journal={arXiv preprint arXiv:2303.01469},<br/>
  year={2023}<br/>
}

@inproceedings{cao2023r3dad,
  title={R3D-AD: Reconstructing 3D Shapes for Unsupervised Anomaly Detection in Point Clouds},<br/>
  author={Cao, Xiyang and Zhang, Ziyang and Liu, Lanqing and Yan, Xiaokang and Yang, Kailun and Zhao<br/> Hengshuang and Geiger, Andreas and Shi, Jianping},<br/>
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},<br/>
  year={2023}<br/>
}
