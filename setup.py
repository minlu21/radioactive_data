import torch
import torch.nn as nn

from torchvision import models

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for name, param in resnet18.named_parameters():
    if 'fc' not in name:
        param.requires_grad=False
resnet18.fc = nn.Linear(in_features=512, out_features=1000, bias=True)

torch.save({
    "model": resnet18.state_dict(),
    "params": {
        "num_classes": 1000,
        "architecture": "resnet18",
    }
  }, "imagenet10k_resnet18.pth")

n_classes, dim = 1000, 512
carriers = torch.randn(n_classes, dim)
carriers /= torch.norm(carriers, dim=1, keepdim=True)
print(f"Vector of carriers is of shape: {carriers.shape}")
torch.save(carriers, "carriers.pth")