import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    print('CUDA is available! You can use GPU acceleration.')
else:
    print('CUDA is not available. GPU acceleration is not possible.')
