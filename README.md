# Facial-Detection-Final
for running YOLO face detection, there's some setup of software required.
steps:

1. download nvidia cuda toolkit for your GPU: https://developer.nvidia.com/cuda-downloads
2. install pytorch with gpu support from cmd with the command in cmd: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
3. To ensure that PyTorch was installed correctly, we can verify the installation by running sample PyTorch code.

	From the command line, type:

	python

	to check if your GPU driver and CUDA is enabled and accessible by PyTorch, run the following commands to return whether or not the CUDA driver is enabled:

	import torch
	torch.cuda.is_available()
4. pip install ultralytics
5. Keep yolov8n-face.pt in the same directory as the multi-face-detection-YOLO.py script and run the script.
