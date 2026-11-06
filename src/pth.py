import torch

# 加载保存的pth文件
checkpoint = torch.load("/home/wang/下载/NetMamba/src/output/finetune/CICIoT2022/checkpoint-best.pth")

if 'args' in checkpoint:
    print(checkpoint['args'])

# print(checkpoint['model'])

# 打印检查保存内容
print(checkpoint.keys())
