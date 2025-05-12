import torch

if torch.cuda.is_available():
		device = torch.device("cuda")
		print(f"CUDA is available: {torch.cuda.device_count()} device(s), using {torch.cuda.get_device_name(0)}")
else:
		device = torch.device("cpu")
		print("CUDA is not available, using CPU")