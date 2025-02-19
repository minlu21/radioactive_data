import random
import os
import sys

import wandb
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms

from data.src.loaders import tinyimagenet as imgnet
from data.src.loaders import cifar10 as cifar
from src.dataset import NORMALIZE_CIFAR


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def train_loop(dataloader, model, loss_fn, optimizer, device="cpu"):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    train_accs = dict()
    for batch, (image, target) in enumerate(dataloader):
        optimizer.zero_grad()
        image_d, target_d = image.to(device), target.to(device)
        prediction = model(image_d)
        loss = loss_fn(prediction, target_d)
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            train_loss += loss.item()
        wandb.log({"Training Loss": train_loss / (batch + 1)})
    model.eval()
    with torch.no_grad():
        for image, target in dataloader:
            image_d, target_d = image.to(device), target.to(device)
            prediction = model(image_d)
            for k in [1, 5]:
                try:
                    train_accs[k] += check_top_k_accuracy(prediction, target_d, k)
                except:
                    train_accs[k] = check_top_k_accuracy(prediction, target_d, k) 
        for k in [1, 5]:
            train_accs[k] /= size
        wandb.log({"Top 1 Train Accuracy": 100 * train_accs[1], "Top 5 Train Accuracy": 100 * train_accs[5]})
            


def validate_loop(dataloader, model, loss_fn, device="cpu"):
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0
    accuracies = dict()
    with torch.no_grad():
        for image, target in dataloader:
            image_d, target_d = image.to(device), target.to(device)
            prediction = model(image_d)
            test_loss += loss_fn(prediction, target_d).item()
            # for k in [1, 5, 10, 50]:\
            for k in [1, 5]:
                try:
                    accuracies[k] += check_top_k_accuracy(prediction, target_d, k)
                except:
                    accuracies[k] = check_top_k_accuracy(prediction, target_d, k) 
    test_loss /= len(dataloader)
    # for k in [1, 5, 10, 50]:
    for k in [1, 5]:
        accuracies[k] /= size
    # wandb.log({"Top 1 Accuracy": 100 * accuracies[1], "Top 5 Accuracy": 100 * accuracies[5], "Top 10 Accuracy": 100 * accuracies[10], "Top 50 Accuracies": 100 * accuracies[50], "Average Validation Loss": test_loss})
    wandb.log({"Validation Loss": test_loss, "Top 1 Validation Accuracy": 100 * accuracies[1], "Top 5 Validation Accuracy": 100 * accuracies[5]}) 


def check_top_k_accuracy(prediction, target, k=1):
    _, top_k_preds = torch.topk(prediction, k)
    target_idx = torch.argmax(target)
    compare_result = torch.eq(top_k_preds, target_idx.expand(top_k_preds.size()))
    return torch.sum(compare_result).int().item()
            

image_mean = torch.Tensor(NORMALIZE_CIFAR.mean).view(-1, 1, 1)
image_std = torch.Tensor(NORMALIZE_CIFAR.std).view(-1, 1, 1)


# configs = {
#     "learning rate": 0.8,
#     "architecture": "resnet18",
#     "dataset": "Tiny Imagenet",
#     "num classes": 200,
#     "epochs": 100
# }

configs = {
    "learning rate": 0.8,
    "architecture": "resnet18",
    "dataset": "CIFAR 10",
    "num classes": 10,
    "momentum": 0.9,
    "weight decay": 1e-4,
    "batch size": 128,
    "epochs": 240,
    "seed": 42
}

wandb.init(
    project="vanilla-cifar10-small-resnet18",
    config={
        "learning_rate": configs["learning rate"],
        "architecture": configs["architecture"],
        "dataset": configs["dataset"],
        "num classes": configs["num classes"],
        "epochs": configs["epochs"]
    }
)

torch.manual_seed(configs["seed"])

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Optimizer: SGD w/ momentum of 0.9 and weight decay of 10^-4
# Epochs: 90
# Learning Rate Scheduler: 0.8 -> divide by 10 every 30 epochs
# Top 1 Accuracy: 69%, Top 5 Accuracy: 89%
# model = models.resnet18()
model = ResNet18()
# model.fc = nn.Linear(in_features=512, out_features=configs["num classes"], bias=True)
model.to(device)
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=configs["learning rate"], momentum=configs["momentum"], weight_decay=configs["weight decay"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# imagenet_dataset = imgnet.TinyImagenet(train_dataset="tinyimagenet_train.csv", val_dataset="tinyimagenet_val.csv", labels="mapping.csv", transform=[transforms.ToTensor(), transforms.Normalize(image_mean, image_std)])
# train_data_loader, val_data_loader, test_data_loader = imagenet_dataset.get_dataloaders()

cifar_dataset = cifar.Cifar10("./scratch/cifar-10", transform=[transforms.ToTensor(),
                                                               transforms.RandomHorizontalFlip(p=0.5), 
                                                               transforms.Normalize(image_mean, image_std)])
train_data_loader, val_data_loader, test_data_loader = cifar_dataset.get_dataloaders(splits=(0.8, 0.19, 0.01), 
                                                                                     batch_sizes=(configs["batch size"], configs["batch size"], configs["batch size"])
                                                                                     )

for e in tqdm(range(configs["epochs"]), file=sys.stdout):
    print(f"Epoch {e}\n----------------------------------", flush=True)
    train_loop(train_data_loader, model, loss_fn, optimizer, device)
    validate_loop(val_data_loader, model, loss_fn, device)
    wandb.log({"Learning Rate": scheduler.get_last_lr()[0]})
    scheduler.step()


print("Done!")