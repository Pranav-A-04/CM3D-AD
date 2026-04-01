inference-consistency: ashtray0-consistency headset0-consistency jar0-consistency 

inference-diffusion: ashtray0-diffusion headset0-diffusion jar0-diffusion

ashtray0-consistency:
	python inference.py --checkpoint "/content/Fast-CMAD/ckpts/consistency/ashtray0.pt" --model_type consistency --pointcloud "/content/Fast-CMAD/sample_pcds/ashtray0.pcd" --num_points 43586

headset0-consistency:
	python inference.py --checkpoint "/content/Fast-CMAD/ckpts/consistency/headset0.pt" --model_type consistency --pointcloud "/content/Fast-CMAD/sample_pcds/headset0.pcd" --num_points 51552

jar0-consistency:
	python inference.py --checkpoint "/content/Fast-CMAD/ckpts/consistency/jar0.pt" --model_type consistency --pointcloud "/content/Fast-CMAD/sample_pcds/jar0.pcd" --num_points 32270

ashtray0-diffusion:
	python inference.py --checkpoint "/content/Fast-CMAD/ckpts/diffusion/ashtray0.pt" --model_type diffusion --pointcloud "/content/Fast-CMAD/sample_pcds/ashtray0.pcd" --num_points 43586

headset0-diffusion:
	python inference.py --checkpoint "/content/Fast-CMAD/ckpts/diffusion/headset0.pt" --model_type diffusion --pointcloud "/content/Fast-CMAD/sample_pcds/headset0.pcd" --num_points 51552

jar0-diffusion:
	python inference.py --checkpoint "/content/Fast-CMAD/ckpts/diffusion/jar0.pt" --model_type diffusion --pointcloud "/content/Fast-CMAD/sample_pcds/jar0.pcd" --num_points 32270

download-dependencies:
	pip install --no-cache-dir -r requirements.txt